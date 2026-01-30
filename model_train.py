import torch
import torch.utils.data as data_utils
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

import time
import copy
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
from typing import Dict, Tuple, List

#from rollout import generate_rollout
from patient_data import DataMatrix, PatientRecord


class ModelTrainer:
    '''
    MoA Model Trainer with all improved features as default:
    - Switch Transformer / CV² load-balancing loss
    - Router warmup strategy
    - Expert usage tracking and logging
    - Clean validation metrics (no LB loss in val_loss)
    - Proper gradient accumulation with pooled LB loss
    '''

    def __init__(
            self,
            model,
            dataloaders,
            epochs,
            optimizer,
            sched_builder,
            class_count,
            device,
            save_path,
            early_stop=20,
            grad_accum=20,
            lb_coef: float = 0.0,
            lb_loss_type: str = 'switch',  # 'cv2' or 'switch'
            router_warmup_epochs: int = 0,
            force_uniform_epochs: int = 0,
            num_experts: int = 1,
            # Gumbel routing config (fixed: 1.0→0.5 with 0.95 decay)
            use_gumbel: bool = False):
        self.model = model
        self.dataloaders = dataloaders
        self.epochs = epochs
        self.optimizer = optimizer
        self.scheduler = sched_builder.scheduler
        self.scheduler_name = sched_builder.name
        self.class_count = class_count
        self.early_stop = early_stop
        self.device = device
        self.save_path = save_path
        self.grad_accum = grad_accum
        self.data_obj = DataMatrix(class_count)
        self.metric_mode = (
            "max" if self.scheduler_name == "ReduceLROnPlateau" and self.scheduler.mode == "max" else "min"
        )
        # coefficient for load-balancing penalty on router gates
        self.lb_coef = float(lb_coef) if lb_coef is not None else 0.0
        self.lb_loss_type = lb_loss_type
        self.router_warmup_epochs = router_warmup_epochs
        self.force_uniform_epochs = force_uniform_epochs
        self.num_experts = num_experts

        # Gumbel routing (fixed defaults: 1.0→0.5 with 0.95 decay)
        self.use_gumbel = bool(use_gumbel)
        self.gumbel_tau_start = 1.0
        self.gumbel_tau_min = 0.5
        self.gumbel_decay = 0.95

        # Expert usage tracking
        self.expert_usage_history = []

    def _compute_lb_loss_switch(self, gates, k_active):
        """
        Switch Transformer style load-balancing loss.
        Encourages uniform distribution across experts.

        Args:
            gates: [B, E] soft router probabilities
            k_active: number of experts selected in top-k
        Returns:
            lb_loss: scalar load-balancing loss
        """
        E = gates.size(1)
        # Soft routing probabilities (average gate weights per expert)
        P = gates.mean(dim=0)  # [E]

        # Hard routing load (fraction of samples routed to each expert)
        # Get top-k experts for each sample
        topk_idx = torch.topk(gates, k=k_active, dim=1).indices  # [B, k]
        # Count how many times each expert was selected
        load_counts = torch.bincount(topk_idx.view(-1), minlength=E).float()
        f = load_counts / (load_counts.sum() + 1e-9)  # [E]

        # Switch loss: num_experts * sum(f_i * P_i)
        lb_loss = E * (f * P).sum()
        return lb_loss

    def _compute_lb_loss_cv2(self, gates, k_active):
        """
        Coefficient of variation squared (CV²) based load-balancing loss.
        Penalizes variance in expert usage.

        Args:
            gates: [B, E] soft router probabilities
            k_active: number of experts selected in top-k
        Returns:
            lb_loss: scalar load-balancing loss
        """
        E = gates.size(1)

        # Importance (soft routing)
        importance = gates.sum(0) / (gates.sum() + 1e-9)  # [E]
        imp_cv2 = importance.var(unbiased=False) / (importance.mean()**2 + 1e-9)

        # Load (hard routing)
        topk_idx = torch.topk(gates, k=k_active, dim=1).indices
        load_counts = torch.bincount(topk_idx.view(-1), minlength=E).to(gates.dtype).to(gates.device)
        load = load_counts / (load_counts.sum() + 1e-9)
        load_cv2 = load.var(unbiased=False) / (load.mean()**2 + 1e-9)

        lb_loss = 0.5 * (imp_cv2 + load_cv2)
        return lb_loss

    def _log_expert_usage(self, gates, epoch, split='train'):
        """Log expert usage statistics."""
        if gates is None:
            return

        with torch.no_grad():
            E = gates.size(1)
            # Soft routing probabilities
            soft_usage = gates.mean(dim=0).cpu().numpy()  # [E]

            # Hard routing (based on argmax)
            hard_assignments = gates.argmax(dim=1)  # [B]
            hard_counts = torch.bincount(hard_assignments, minlength=E).cpu().numpy()
            hard_usage = hard_counts / (hard_counts.sum() + 1e-9)

            # Entropy (measure of routing diversity)
            avg_probs = gates.mean(dim=0)  # [E]
            entropy = -(avg_probs * torch.log(avg_probs + 1e-9)).sum().item()
            max_entropy = np.log(E)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

            stats = {
                'epoch': epoch,
                'split': split,
                'soft_usage': soft_usage.tolist(),
                'hard_usage': hard_usage.tolist(),
                'entropy': entropy,
                'normalized_entropy': normalized_entropy,
            }

            # Print summary
            print(f"  Expert Usage ({split}): soft={soft_usage}, hard={hard_usage}, entropy={normalized_entropy:.3f}")

            return stats

    def launch_training(self):
        '''initializes training process.'''
        #print("Initialized")
        best_metric = -np.inf if self.metric_mode == "max" else np.inf
        best_state = copy.deepcopy(self.model.state_dict())
        _no_improve_epochs = 0
        warmup_complete = False

        for ep in range(self.epochs):
            # Compute current Gumbel temperature with decay starting AFTER warmup
            # Phase 1/2: use tau=1.0 (internal warmup drives temp to 1)
            # Phase 3: tau decays 1.0→0.5 with rate 0.95 per epoch
            if self.use_gumbel:
                if ep < self.router_warmup_epochs:
                    current_tau = 1.0
                else:
                    eff_ep = ep - self.router_warmup_epochs
                    current_tau = max(self.gumbel_tau_min, (self.gumbel_decay ** eff_ep))
            else:
                current_tau = 1.0
            # perform train/val iteration
            val_metrics = self._run_epoch(ep, backprop_every=self.grad_accum, train_temp=current_tau, use_gumbel=self.use_gumbel)
            torch.cuda.empty_cache()
            target_metric = val_metrics["weighted_f1"] if self.metric_mode == "max" else val_metrics["val_loss"]
            is_better = (target_metric > best_metric) if self.metric_mode == "max" else (target_metric < best_metric)

            # Check if warmup just completed
            if not warmup_complete and ep >= self.router_warmup_epochs:
                warmup_complete = True
                # Reset early stopping counter at end of warmup
                _no_improve_epochs = 0
                # Reset best metric to start fresh after warmup
                best_metric = target_metric
                best_state = copy.deepcopy(self.model.state_dict())
                torch.save(best_state, self.save_path / "cAItomorph_best_weights.pth")
                print(f"✅ Warmup complete! Resetting early stopping counter.")
                print(f"🔖  New baseline metric after warmup: {best_metric:.4f}")
                continue  # Skip early stopping check this epoch

            # Early stopping only applies AFTER warmup
            if warmup_complete:
                if is_better:
                    best_metric = target_metric
                    best_state = copy.deepcopy(self.model.state_dict())
                    _no_improve_epochs = 0
                    torch.save(best_state, self.save_path / "cAItomorph_best_weights.pth")
                    print(f"🔖  Saved new best model (metric={best_metric:.4f})")
                else:
                    _no_improve_epochs += 1
                    print(f"⏳ No improvement for {_no_improve_epochs}/{self.early_stop} epochs")
                    if _no_improve_epochs >= self.early_stop:
                        print(f"⏹️  Early-stopping triggered after {ep+1} total epochs ({ep+1-self.router_warmup_epochs} post-warmup)")
                        break
            else:
                # During warmup: track best but don't count for early stopping
                if is_better:
                    best_metric = target_metric
                    best_state = copy.deepcopy(self.model.state_dict())
                    torch.save(best_state, self.save_path / "cAItomorph_best_weights.pth")
                    print(f"🔖  Warmup: saved best so far (metric={best_metric:.4f})")

            if self.scheduler_name == "ReduceLROnPlateau":
                self.scheduler.step(target_metric)
            elif self.scheduler is not None:
                self.scheduler.step()

        # load best performing model, and launch on test set
        self.model.load_state_dict(best_state)
        #Save logits for val

        _ = self._infer_split('train')
        _ = self._infer_split('val')
        test_metrics = self._infer_split('test')
        print("Inference completed for splits: train, val, test")

        self.data_obj.save_hdf5(self.save_path / "patient_data.h5")

        # Save expert usage history to CSV
        if len(self.expert_usage_history) > 0:
            df_usage = pd.DataFrame(self.expert_usage_history)
            df_usage.to_csv(self.save_path / "expert_usage_history.csv", index=False)
            print(f"Saved expert usage history to {self.save_path}/expert_usage_history.csv")

        return self.model, test_metrics['conf_matrix']
    

    def _run_epoch(self, epoch, backprop_every=20, train_temp: float = 1.0, use_gumbel: bool = False):
        train_loss = 0
        all_predictions, all_labels = [], []
        time_pre_epoch = time.time()
        self.optimizer.zero_grad()
        backprop_counter = 0

        # Buffers for gates and CE losses (to pool over accum steps)
        gate_buf, ce_buf = [], []

        # === ADAPTIVE 3-PHASE WARMUP ===
        # Phase 1: Forced uniform routing (epochs 0 to force_uniform_epochs-1)
        # Phase 2: Gradual specialization (epochs force_uniform_epochs to warmup_epochs-1)
        # Phase 3: Normal top-k routing (epochs warmup_epochs+)

        # Access k_active from the underlying MixtureOfAggregators model
        base_model = self.model.module if hasattr(self.model, 'module') else self.model
        target_k = getattr(base_model.model, 'k_active', 2)

        if epoch < self.force_uniform_epochs:
            # Phase 1: Forced Uniform
            current_k = self.num_experts
            phase = 1
            print(f"  🔄 Phase 1/3 (Forced Uniform): epoch {epoch}/{self.force_uniform_epochs-1}")
            print(f"     All {self.num_experts} experts receive equal training (k={current_k})")

        elif epoch < self.router_warmup_epochs:
            # Phase 2: Gradual Specialization
            alpha = (epoch - self.force_uniform_epochs) / (self.router_warmup_epochs - self.force_uniform_epochs)
            current_k = int(self.num_experts - (self.num_experts - target_k) * alpha)
            current_k = max(target_k, current_k)
            phase = 2
            print(f"  🔄 Phase 2/3 (Gradual Specialization): epoch {epoch}/{self.router_warmup_epochs-1}")
            print(f"     Transition progress: {alpha:.2f}, k={current_k} → {target_k}, temp: 10.0 → 1.0")

        else:
            # Phase 3: Normal Top-k Routing
            current_k = target_k
            phase = 3
            if epoch == self.router_warmup_epochs:
                print(f"  ✅ Phase 3/3 (Specialized Top-k): Warmup complete!")
                print(f"     Using k={current_k} with specialized expert routing")

        self.model.train()
        for i, (bag, label, img_paths, patient_id) in enumerate(tqdm(self.dataloaders['train'])):
            label = label.to(self.device)
            bag   = bag.to(self.device)

            # Forward pass with adaptive 3-phase warmup
            try:
                latent, prediction, gates = self.model(
                    bag,
                    return_latent=True,
                    return_gates=True,
                    temp=train_temp,
                    use_gumbel=use_gumbel,
                    k=current_k,
                    epoch=epoch,  
                    force_uniform_epochs=self.force_uniform_epochs,
                    warmup_epochs=self.router_warmup_epochs,
                )
            except TypeError:
                # Fallback if model doesn't support new parameters
                latent, prediction, gates = self.model(
                    bag,
                    return_latent=True,
                    return_gates=True,
                    temp=train_temp,
                    use_gumbel=use_gumbel,
                )

            if prediction.dim() == 3:
                prediction = prediction.squeeze(0)
            ce_loss = nn.CrossEntropyLoss()(prediction, label)
            ce_buf.append(ce_loss)

            if self.lb_coef > 0.0 and gates is not None:
                g2d = gates if gates.dim() == 2 else gates.view(gates.size(0), -1)
                gate_buf.append(g2d)

            backprop_counter += 1

            # At accumulation boundary: compute pooled LB, backprop, step
            if (backprop_counter % backprop_every == 0) or (i == len(self.dataloaders['train']) - 1):
                ce_mean = torch.stack(ce_buf).mean()
                total_loss = ce_mean

                if self.lb_coef > 0.0 and len(gate_buf) > 0:
                    G = torch.cat(gate_buf, dim=0)
                    k_for_lb = current_k

                    # Compute LB loss based on specified type
                    if self.lb_loss_type == 'switch':
                        lb_loss = self._compute_lb_loss_switch(G, k_for_lb)
                    elif self.lb_loss_type == 'cv2':
                        lb_loss = self._compute_lb_loss_cv2(G, k_for_lb)
                    else:
                        raise ValueError(f"Unknown lb_loss_type: {self.lb_loss_type}")

                    total_loss = total_loss + self.lb_coef * lb_loss

                total_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                ce_buf.clear()
                gate_buf.clear()
                train_loss += total_loss.item()

            # Track metrics
            label_prediction = torch.argmax(prediction, dim=1).item()
            all_predictions.append(label_prediction)
            all_labels.append(label.item())

        train_loss /= len(self.dataloaders['train'])
        accuracy = accuracy_score(all_labels, all_predictions)
        balanced_acc = balanced_accuracy_score(all_labels, all_predictions)
        w_f1 = f1_score(all_labels, all_predictions, average='weighted')

        print('- ep: {}/{}, loss: {:.3f}, acc: {:.3f}, balanced acc: {:.3f}, weighted_f1: {:.3f}, {}s, {}'.format(
            epoch + 1, self.epochs, train_loss,
            accuracy, balanced_acc, w_f1, int(time.time() - time_pre_epoch), 'train'))

        # Log expert usage for training
        if len(gate_buf) > 0 or self.lb_coef > 0.0:
            # Collect all gates from the epoch for logging
            self.model.eval()
            all_train_gates = []
            with torch.no_grad():
                for (bag, label, _, _) in self.dataloaders['train']:
                    bag = bag.to(self.device)
                    try:
                        _, _, gates = self.model(bag, return_latent=True, return_gates=True)
                        if gates is not None:
                            all_train_gates.append(gates)
                    except:
                        pass
                    if len(all_train_gates) >= 50:  # Sample first 50 for efficiency
                        break
            if len(all_train_gates) > 0:
                train_gates = torch.cat(all_train_gates, dim=0)
                stats = self._log_expert_usage(train_gates, epoch, 'train')
                if stats:
                    self.expert_usage_history.append(stats)

        # ---------------- Validation ----------------
        val_ce_sum = 0.0
        total_samples = 0
        all_predictions, all_labels = [], []
        val_gate_buf: List[torch.Tensor] = []

        self.model.eval()
        time_pre_epoch = time.time()
        with torch.no_grad():
            for (bag, label, img_paths, patient_id) in tqdm(self.dataloaders['val']):
                label = label.to(self.device)
                bag   = bag.to(self.device)

                # Validation mimics test conditions: no routing parameters, use model defaults
                latent, prediction, gates = self.model(
                    bag,
                    return_latent=True,
                    return_gates=True,
                )

                if prediction.dim() == 3:
                    prediction = prediction.squeeze(0)

                ce = nn.CrossEntropyLoss(reduction="sum")(prediction, label)
                val_ce_sum += ce.item()
                total_samples += label.size(0)

                if gates is not None:
                    g2d = gates if gates.dim() == 2 else gates.view(gates.size(0), -1)
                    val_gate_buf.append(g2d)

                label_prediction  = torch.argmax(prediction, dim=1).item()
                all_predictions.append(label_prediction)
                all_labels.append(label.item())

        val_ce_mean = val_ce_sum / max(1, total_samples)

      
        val_loss = val_ce_mean

        # Log LB term separately for monitoring (not used in early stopping)
        if self.lb_coef > 0.0 and len(val_gate_buf) > 0:
            Gv = torch.cat(val_gate_buf, dim=0)
            # Access k_active from the underlying MixtureOfAggregators model
            base_model = self.model.module if hasattr(self.model, 'module') else self.model
            k_val = getattr(base_model.model, 'k_active', 2)

            if self.lb_loss_type == 'switch':
                val_lb_term = self._compute_lb_loss_switch(Gv, k_val).item()
            elif self.lb_loss_type == 'cv2':
                val_lb_term = self._compute_lb_loss_cv2(Gv, k_val).item()
            else:
                val_lb_term = 0.0

            print(f"  Validation LB loss ({self.lb_loss_type}): {val_lb_term:.4f} (not included in val_loss)")

        accuracy    = accuracy_score(all_labels, all_predictions)
        balanced_ac = balanced_accuracy_score(all_labels, all_predictions)
        wf1         = f1_score(all_labels, all_predictions, average='weighted')

        print('- ep: {}/{}, loss: {:.3f}, acc: {:.3f}, balanced acc: {:.3f}, weighted_f1: {:.3f}, {}s, {}'.format(
            epoch + 1, self.epochs, val_loss,
            accuracy, balanced_ac, wf1, int(time.time() - time_pre_epoch), 'val'))

        # Log expert usage for validation
        if len(val_gate_buf) > 0:
            val_gates = torch.cat(val_gate_buf, dim=0)
            stats = self._log_expert_usage(val_gates, epoch, 'val')
            if stats:
                self.expert_usage_history.append(stats)

        return { 'train_loss': train_loss, 'val_loss': val_loss, 'accuracy': accuracy, 'weighted_f1': wf1 }


    def _infer_split(
        self,
        split: str,
        *,
        save_logits: bool = True,
     ) -> Dict[str, float]:
        self.model.eval()
        dataloader = self.dataloaders[split]

        preds, label_preds, labels, patients = [], [], [], []
        all_gates = []  # <--- store gates here
        running_loss = 0.0

        with torch.no_grad():
            for bag, label, img_paths, patient_id in tqdm(dataloader, desc=f"Inference {split}"):
                patient_id = patient_id[0]
                bag   = bag.to(self.device)
                label = label.to(self.device)

                if getattr(self.model, "save_gates", False):
                    latent, logits, gates = self.model(bag, return_latent=True, return_gates=True)
                    if gates is not None:
                        # detach and move to CPU for saving
                        all_gates.append(gates.detach().cpu().numpy())
                else:
                    latent, logits = self.model(bag, return_latent=True)
                if logits.dim() == 3:
                    logits = logits.squeeze(0)  # Ensure logits is 2D
                loss = nn.CrossEntropyLoss()(logits, label)
                running_loss += loss.item() * bag.size(0)

                #cls_attention_scores = generate_rollout(self.model, bag, start_layer=0)
                cls_attention_scores = torch.Tensor([0,0,0])  # Placeholder, if you want to compute attention scores, uncomment the line above
                cls_attention_scores = cls_attention_scores.squeeze(0)

                label_prediction = torch.argmax(logits, dim=1).item()
                label_groundtruth = label.item()

                self.data_obj.add_patient(
                    patient_id,
                    PatientRecord.from_tensors(
                        true_label=label_groundtruth,
                        pred_label=label_prediction,
                        latent=latent,
                        attention=cls_attention_scores,
                        image_paths=img_paths,
                        prediction_vector=logits,
                        loss=loss,
                    ),
                )

                label_preds.append(label_prediction)
                labels.append(label_groundtruth)
                preds.append(logits)
                patients.append(patient_id)

        preds = torch.cat(preds, dim=0).detach().cpu().numpy()

        epoch_loss = running_loss / len(dataloader.dataset)
        acc = accuracy_score(labels, label_preds)
        bal_acc = balanced_accuracy_score(labels, label_preds)
        f1_weighted = f1_score(labels, label_preds, average='weighted')
        conf_matrix = confusion_matrix(labels, label_preds, labels=list(range(self.class_count)))

        print(f"{split}: loss={epoch_loss:.4f} acc={acc:.3f} balAcc={bal_acc:.3f} f1w={f1_weighted:.3f}")

        if save_logits:
            pd.DataFrame({
                "patient": patients,
                "label": labels,
                "prediction": [list(p) for p in preds],
            }).to_csv(self.save_path / f"metadata_results_{split}.csv", index=False)

        if getattr(self.model, "save_gates", False) and len(all_gates) > 0:
            np.save(self.save_path / f"gates_{split}.npy", np.array(all_gates))
            print(f"Saved gates for {split} to {self.save_path}/gates_{split}.npy")

        return {
            "loss": epoch_loss,
            "acc": acc,
            "balanced_acc": bal_acc,
            "f1": f1_weighted,
            "conf_matrix": conf_matrix,
        }
