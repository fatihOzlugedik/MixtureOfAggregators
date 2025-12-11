import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer_expert import TransformerExpert
from .abmil_expert import ABMILExpert
from .transmil_expert import TransMILExpert
#from .dsmil_expert import DSMILExpert


class MixtureOfAggregators(nn.Module):
    """

    Architecture (always):
    - Separate experts (each initialized with unique seed for diversity)
    - No local heads per expert (experts only produce latent representations)
    - Router weights combine latents
    - Single shared classification head

    router_style:
      - "dense" : soft mixture of ALL experts
      - "topk"  : sparse mixture of only k experts
    """
    def __init__(self,
                 num_classes,
                 expert_arch,
                 input_dim=2048,
                 dim=512,
                 depth=2,
                 heads=8,
                 mlp_dim=512,
                 dim_head=64,
                 dropout=0.1,
                 emb_dropout=0.1,
                 pool="cls",
                 pos_enc=None,
                 num_experts=4,
                 router_style="dense",    # "dense" or "topk"
                 k_active=2,
                 router_type="linear",    # "linear", "mlp", "transformer", or "vit"
                 diversity_init=False,    # Enable diverse expert initialization
                 base_seed=None,          # Base seed for diversity
                 ):
        super().__init__()
        self.num_experts = num_experts
        self.router_style = router_style
        self.k_active = k_active
        self.router_type = router_type

        # === V2: Simplified expert setup ===
        # Always separate mode (no shared projection)
        # Always no local head (experts produce latents only)
        base_kwargs = {
            "num_classes": num_classes,
            "input_dim": input_dim,
            "mode": "separate",           # V2: Always separate (hard-coded)
            "shared_proj": None,          # V2: No shared projection
            "use_local_head": False,      # V2: No local heads (hard-coded)
        }

        if expert_arch == 'Transformer':
            ExpertClass = TransformerExpert
            expert_kwargs = {
                **base_kwargs,
                "dim": dim,
                "depth": depth,
                "heads": heads,
                "mlp_dim": mlp_dim,
                "dim_head": dim_head,
                "dropout": dropout,
                "emb_dropout": emb_dropout,
                "pool": pool,
                "pos_enc": pos_enc,
            }
        elif expert_arch == 'ABMIL':
            ExpertClass = ABMILExpert
            expert_kwargs = {
                **base_kwargs,
                "size_arg": "big",
                "dropout": dropout,
            }
        elif expert_arch == 'TransMIL':
            ExpertClass = TransMILExpert
            # TransMIL uses in_dim instead of input_dim, and requires embed_dim
            expert_kwargs = {
                "num_classes": num_classes,
                "in_dim": input_dim,
                "embed_dim": dim,
                "mode": "separate",
                "shared_proj": None,
                "use_local_head": False,
                "dropout": dropout,
                # Uses defaults: num_fc_layers=1, num_attention_layers=2, num_heads=4
            }
    
        else:
            raise ValueError(f"Unknown expert_arch '{expert_arch}'. Supported: 'Transformer', 'ABMIL', 'TransMIL', 'DSMIL'")

        # === Create experts with diversity initialization ===
        self.experts = nn.ModuleList()
        for i in range(num_experts):
            if diversity_init and base_seed is not None:
                # Set unique seed for each expert to encourage diversity
                expert_seed = base_seed + i * 1000
                torch.manual_seed(expert_seed)
                torch.cuda.manual_seed(expert_seed)
                torch.cuda.manual_seed_all(expert_seed)
            expert = ExpertClass(**expert_kwargs)
            self.experts.append(expert)

        # Restore original seed after expert initialization
        if diversity_init and base_seed is not None:
            torch.manual_seed(base_seed)
            torch.cuda.manual_seed(base_seed)
            torch.cuda.manual_seed_all(base_seed)

        # === router ===
        proj_dim = heads * dim_head
        self.router_proj = nn.Sequential(nn.Linear(input_dim, proj_dim, bias=True), nn.ReLU())

        if self.router_type == "mlp":
            self.router_fc = nn.Sequential(
                nn.Linear(proj_dim, 256), nn.ReLU(),
                nn.Linear(256, num_experts)
            )
        elif self.router_type == "linear":
            self.router_fc = nn.Linear(proj_dim, num_experts)

        elif self.router_type == "ABMIL":
            router_kwargs = {
            "num_classes": num_experts,  ## V2: Output num_experts logits,it will be used for routing
            "size_arg": "big",
            "dropout": dropout,
            "input_dim": input_dim,
            "mode": "separate",           # V2: Always separate (hard-coded)
            "shared_proj": None,          # V2: No shared projection
            "use_local_head": True,      # V2: Use local head for router
            }

            self.router_fc = ABMILExpert(**router_kwargs)
        elif self.router_type == "transformer":
            router_kwargs = {
                "num_classes": num_experts,
                "input_dim": input_dim,
                "dim": dim,
                "depth": depth,
                "heads": heads,
                "mlp_dim": mlp_dim,
                "dim_head": dim_head,
                "dropout": dropout,
                "emb_dropout": emb_dropout,
                "pool": pool,
                "pos_enc": pos_enc,
                "mode": "separate",
                "shared_proj": None,
                "use_local_head": True,
            }
            self.router_fc = TransformerExpert(**router_kwargs)
        else:
            raise ValueError(f"Unknown router_type '{self.router_type}'. Supported: 'linear', 'mlp', 'transformer'")

   
        if expert_arch == 'Transformer':
            # Match original Transformer head structure: dim -> 64 -> num_classes
            # Note: Original uses mlp_dim, but latent has dimension dim (works because dim==mlp_dim==512 by default)
            self.head = nn.Sequential(
                nn.Linear(dim, 64), nn.ReLU(),
                nn.Linear(64, num_classes)
            )
        elif expert_arch == 'ABMIL':
            self.head = nn.Sequential(nn.Linear(64, num_classes))
        elif expert_arch == 'TransMIL':
            # TransMIL uses simple linear classifier: embed_dim -> num_classes
            self.head = nn.Linear(dim, num_classes)
  
        else:
            raise ValueError(f"Unknown expert_arch '{expert_arch}'. Supported: 'Transformer', 'ABMIL', 'TransMIL', 'DSMIL'")

    def forward(self, x, temp=1.0, k=None, use_gumbel=False, epoch=None,
                force_uniform_epochs=None, warmup_epochs=None):
        """
        Forward pass with adaptive 3-phase warmup:

        Phase 1 (epochs 0 to force_uniform_epochs-1): Forced uniform routing
        Phase 2 (epochs force_uniform_epochs to warmup_epochs-1): Gradual specialization
        Phase 3 (epochs warmup_epochs+): Normal top-k routing

        Args:
            x: input tensor [B, N, D]
            temp: temperature for softmax (used in Phase 3)
            k: number of experts to use in top-k
            use_gumbel: whether to use Gumbel noise (used in Phase 3)
            epoch: current training epoch (None during inference)
            force_uniform_epochs: number of epochs for forced uniform routing
            warmup_epochs: total warmup epochs
        """
        B, N, _ = x.shape
        k = k or self.k_active

        # Compute router logits
        if self.router_type == "transformer" or self.router_type == "ABMIL":
            #They already include projection internally
            router_in = x
            _, router_logits = self.router_fc(router_in)  # [B, E]
        else:
            router_in = self.router_proj(x).mean(dim=1)  # [B, proj_dim]
            router_logits = self.router_fc(router_in)    # [B, E]

        # === ADAPTIVE 3-PHASE WARMUP ===
        if epoch is not None and warmup_epochs is not None and force_uniform_epochs is not None:
            # Phase 1: Forced Uniform (epochs 0 to force_uniform_epochs-1)
            if epoch < force_uniform_epochs:
                # Complete uniformity - all experts get equal weight
                g_soft = torch.ones_like(router_logits) / self.num_experts

                # Still compute learned gates for gradient flow
                gates_learned = F.softmax(router_logits / 10.0, dim=-1)

                # Straight-through: forward uses uniform, backward uses learned
                g_soft = g_soft.detach() + (gates_learned - gates_learned.detach())

                # Use all experts during uniform phase
                k_effective = self.num_experts

            # Phase 2: Gradual Transition (epochs force_uniform_epochs to warmup_epochs-1)
            elif epoch < warmup_epochs:
                # Interpolation parameter: 0 → 1
                alpha = (epoch - force_uniform_epochs) / (warmup_epochs - force_uniform_epochs)

                # Temperature annealing: 10.0 → 1.0
                temp_annealed = 10.0 * (0.1 ** alpha)

                # Interpolate between uniform and learned
                gates_uniform = torch.ones_like(router_logits) / self.num_experts
                gates_learned = F.softmax(router_logits / temp_annealed, dim=-1)

                g_soft = (1.0 - alpha) * gates_uniform + alpha * gates_learned

                # Gradually reduce k from num_experts to target k
                k_effective = int(self.num_experts - (self.num_experts - k) * alpha)
                k_effective = max(k, k_effective)  # Ensure k_effective >= k

            # Phase 3: Normal Routing (epochs warmup_epochs+)
            else:
                # Optional Gumbel-softmax noise during training
                if use_gumbel and self.training:
                    eps = 1e-9
                    U = torch.rand_like(router_logits).clamp_(eps, 1.0 - eps)
                    gumbel = -torch.log(-torch.log(U))
                    logits_with_gumbel = router_logits + gumbel
                else:
                    logits_with_gumbel = router_logits

                g_soft = F.softmax(logits_with_gumbel / temp, dim=-1)
                k_effective = k
        else:
            # No warmup specified (e.g., during inference), use normal routing
            if use_gumbel and self.training:
                eps = 1e-9
                U = torch.rand_like(router_logits).clamp_(eps, 1.0 - eps)
                gumbel = -torch.log(-torch.log(U))
                logits_with_gumbel = router_logits + gumbel
            else:
                logits_with_gumbel = router_logits

            g_soft = F.softmax(logits_with_gumbel / temp, dim=-1)
            k_effective = k

        if self.router_style == "dense":
            # V2: All experts produce latents only (no local heads)
            latents = []
            for expert in self.experts:
                latent, _ = expert(x)  # Ignore logits (always None)
                latents.append(latent)

            latents = torch.stack(latents, dim=1)  # [B, E, D]

            # Weight latents by router gates
            gates = g_soft.unsqueeze(-1)  # [B, E, 1]
            latent = (latents * gates).sum(dim=1)  # [B, D]

            # V2: Always use shared classification head
            logits = self.head(latent)  # [B, C]
            return latent, logits, g_soft

        elif self.router_style == "topk":
            # V2: Sparse top-k routing with shared head
            k_effective = min(k_effective, self.num_experts)
            topk = g_soft.topk(k_effective, dim=-1)
            idx = topk.indices  # [B, k_effective]
            weights = topk.values  # [B, k_effective]
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

            # V2: All experts produce latents only (no local heads)
            z_list = []
            for b in range(B):
                z_b = 0
                for j in range(k_effective):
                    e_idx = idx[b, j].item()
                    w = weights[b, j]
                    latent_bj, _ = self.experts[e_idx](x[b].unsqueeze(0))  # Ignore logits
                    if latent_bj.dim() == 1:
                        latent_bj = latent_bj.unsqueeze(0)
                    z_b = z_b + w * latent_bj
                z_list.append(z_b)

            latent = torch.stack(z_list, dim=0).squeeze(1)  # [B, D]

            # V2: Always use shared classification head
            logits = self.head(latent)  # [B, C]
            return latent, logits, g_soft
