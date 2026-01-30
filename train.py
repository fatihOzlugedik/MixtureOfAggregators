from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.multiprocessing
import torch
import os
import time
import argparse as ap
import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

from model_train import ModelTrainer
from classifier import MIL_aggregtor
from dataset import MILDataset
from plot_confusion import plot_confusion_matrix
from scheduler import BuildScheduler

torch.multiprocessing.set_sharing_strategy('file_system')

import pandas as pd


def main():
    parser = ap.ArgumentParser(description='MoA Training Pipeline - All improvements enabled by default')

    # Training parameters
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--grad_accum', type=int, default=16, help='gradient accumulation steps')
    parser.add_argument('--scheduler', default='ReduceLROnPlateau', help='scheduler')
    parser.add_argument('--ep', type=int, default=150, help='number of epochs to train')
    parser.add_argument('--es', type=int, default=15, help='early stopping if no decrease in loss for x epochs')
    parser.add_argument('--wd', type=float, default=0.01, help='weight decay')
    parser.add_argument('--metric', choices=['loss', 'f1'], type=str, default='loss', help='primary metric')
    parser.add_argument('--seed', type=int, default=38, help='random seed')

    # Model architecture
    parser.add_argument('--arch', default="Transformer", help='model architecture')

    # Continue from checkpoint
    parser.add_argument('--checkpoint', default=None, help='checkpoint')

    # Data path (embeddings root)
    parser.add_argument('--data_path',help='embeddings root path')

    # Results
    parser.add_argument('--result_folder', required=False, help='(unused, kept for compatibility)')
    parser.add_argument('--saving_name', required=True, help='base directory to save results into')

    # Cross-validation / CSV inputs
    parser.add_argument('--csv_root',
                        default="data_cross_val_3_classes_Bracs",
                        help='root directory that contains data_fold_*/train.csv|val.csv|test.csv')
    parser.add_argument('--label_map_csv',
                        default=None,
                        help='path to label mapping CSV; defaults to <csv_root>/3class_label_mapping.csv')
    parser.add_argument('--extension', default='h5', choices=['h5', 'pt'], help='feature file extension')

    # Mixture-of-Experts / Aggregator setup
    # Simplified architecture - always separate experts with shared head
    parser.add_argument('--router_style', choices=['topk', 'dense'],
                        default='topk', help="Routing strategy")
    parser.add_argument('--topk', type=int, default=1,
                        help="Top-k experts to use when router_style=topk")
    parser.add_argument('--save_gates', action='store_true',
                        help="Save router gate activations during training/validation")
    parser.add_argument('--num_expert', type=int, default=1,
                        help="Number of experts to use")
    parser.add_argument('--router_type', choices=['linear', 'mlp'],
                        default='linear', help="Router type")

    parser.add_argument('--expert_arch', choices=['Transformer', 'ABMIL', 'TransMIL', 'DSMIL'], default='Transformer')

    # Load-balancing loss
    parser.add_argument('--use_lb_loss', action='store_true',
                        help='Enable load-balancing loss on router gates')
    parser.add_argument('--lb_coef', type=float, default=0.0,
                        help='Coefficient for load-balancing loss')
    parser.add_argument('--lb_loss_type', choices=['cv2', 'switch'], default='switch',
                        help='Load-balancing loss type (default: switch)')

    # Adaptive 3-phase warmup (default: automatically scaled with num_experts)
    parser.add_argument('--router_warmup_epochs', type=int, default=None,
                       )
    parser.add_argument('--force_uniform_epochs', type=int, default=None,
                       )



    # Gumbel-softmax routing (training only)
    parser.add_argument('--use_gumbel', action='store_true',
                        help='Use Gumbel-softmax noise in router during training (decays from 1.0→0.5 with rate 0.95)')

    parser.add_argument('--prepared_fold', action='store_true', help='Use precomputed 5-fold split from CSVs')

    args = parser.parse_args()

    # Set random seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # store results in target folder
    checkpoint = args.checkpoint
    start = time.time()

    data_path = args.data_path
    backbone_name = data_path.rstrip('/').split('/')[-1]
    save_root = Path(args.saving_name)

    if args.use_lb_loss == False:
        args.lb_coef = 0.0


    force_uniform_epochs = args.force_uniform_epochs
 
    router_warmup_epochs = args.router_warmup_epochs



    RESULT_FOLDER_ROOT = f"Results_5fold_testfixed_{backbone_name}_{args.arch}_{args.router_style}_topk{args.topk}_router_arch_{args.router_type}_seed{seed}_lb{args.lb_coef}_{args.lb_loss_type}_warmup{router_warmup_epochs}_gumbel{args.use_gumbel}_st{args.use_st}_expert_{args.expert_arch}"

    RESULT_FOLDER_ROOT = Path(RESULT_FOLDER_ROOT)
    print('='*80)
    print('MoA Training Pipeline - Adaptive 3-Phase Warmup')
    print('='*80)
    print('Results will be saved under: ', RESULT_FOLDER_ROOT)
    print()
    print('Adaptive Warmup Schedule (scaled with num_experts):')
    print(f'  📊 Configuration: {args.num_expert} experts, target k={args.topk}')
    print(f'  ⚙️  Phase 1 (Forced Uniform): Epochs 0-{force_uniform_epochs-1}')
    print(f'      → All {args.num_expert} experts receive equal training')
    print(f'  ⚙️  Phase 2 (Gradual Specialization): Epochs {force_uniform_epochs}-{router_warmup_epochs-1}')
    print(f'      → Smooth transition with temperature annealing (10.0 → 1.0)')
    print(f'      → k reduces from {args.num_expert} → {args.topk}')
    print(f'  ⚙️  Phase 3 (Specialized Top-k): Epoch {router_warmup_epochs}+')
    print(f'      → Expert specialization with k={args.topk}')
    print(f'  📈 Total warmup duration: {router_warmup_epochs} epochs ({100*router_warmup_epochs/args.ep:.1f}% of training)')
    print()

    # ──────────────────────────────────────────────────────────────────────────
    # FAST-SKIP: if fold_4 already finished for this combination, skip training
    # ──────────────────────────────────────────────────────────────────────────
    result_root_dir = save_root / f'{args.num_expert}_experts' / RESULT_FOLDER_ROOT
    fold4_dir = result_root_dir / 'fold_4'
    fold4_png = fold4_dir / 'confusion_matrix.png'
    fold4_npy = fold4_dir / 'test_conf_matrix.npy'
    if fold4_png.exists() or fold4_npy.exists():
        print(f"[SKIP] Detected completed results for this configuration at: {fold4_dir}")
        print("       Found:", ("confusion_matrix.png" if fold4_png.exists() else "test_conf_matrix.npy"))
        return

    # 2: Dataset
    print("\nInitialize datasets...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ngpu = torch.cuda.device_count()
    print("Found device: ", ngpu, "x ", device)

    datasets = {}

    csv_root = args.csv_root
    label_map_path = args.label_map_csv
    label_to_diagnose = pd.read_csv(label_map_path)

    class_count = len(label_to_diagnose)

    print('Reading files from: ', os.path.join(csv_root, 'data_fold_1'))
    t_files = pd.read_csv(os.path.join(csv_root, 'data_fold_1', "train.csv"))
    v_files = pd.read_csv(os.path.join(csv_root, 'data_fold_1', "val.csv"))
    train_val_files = pd.concat([t_files, v_files], ignore_index=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    for fold, (train_index, val_index) in enumerate(skf.split(train_val_files, train_val_files['labels'])):
        fold_dir = result_root_dir / f'fold_{fold}'
        fold_png = fold_dir / 'confusion_matrix.png'
        if fold_png.exists() :
            print(f"[SKIP] Detected completed results for this configuration at: {fold_dir}")
            print("       Found:", ("confusion_matrix.png" if fold_png.exists() else "test_conf_matrix.npy"))
            continue
        
        if args.prepared_fold:
            # Use pre-computed fold splits from CSV files
            train_files = pd.read_csv(os.path.join(csv_root, f'data_fold_{fold}', "train.csv"))
            val_files = pd.read_csv(os.path.join(csv_root, f'data_fold_{fold}', "val.csv"))
            test_files = pd.read_csv(os.path.join(csv_root, f'data_fold_{fold}', "test.csv"))
        else:
            # Use StratifiedKFold split
            train_files = train_val_files.iloc[train_index].reset_index(drop=True)
            val_files = train_val_files.iloc[val_index].reset_index(drop=True)
            test_files = pd.read_csv(os.path.join(csv_root, 'data_fold_1', "test.csv"))

        RESULT_FOLDER = result_root_dir / f'fold_{fold}'
        RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

        datasets['train'] = MILDataset(data_path, train_files, ext=args.extension)
        datasets['val'] = MILDataset(data_path, val_files, ext=args.extension)
        datasets['test'] = MILDataset(data_path, test_files, ext=args.extension)

        embedding_dim = datasets['train'].get_dimension()
        print(f"Embedding dimension: {embedding_dim}")

        # Initialize dataloaders
        print("Initialize dataloaders...")
        dataloaders = {}
        num_workers = 0
        dataloaders['train'] = DataLoader(datasets['train'], batch_size=1, shuffle=True, num_workers=num_workers, pin_memory=True)
        dataloaders['val'] = DataLoader(datasets['val'], batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
        dataloaders['test'] = DataLoader(datasets['test'], batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
        print("Dataloaders are ready..")

        # Simplified architecture - separate experts with shared head
        model = MIL_aggregtor(
            class_count=class_count,
            arch=args.arch,
            embedding_dim=embedding_dim,
            expert_arch=args.expert_arch,
            router_style=args.router_style,
            topk=args.topk,
            save_gates=args.save_gates,
            num_expert=args.num_expert,
            router_type=args.router_type,
            base_seed=seed,  # diversity_init always enabled
            use_st=args.use_st  # Optional Straight-Through estimator
        )

        if checkpoint is not None:
            pre = torch.load(checkpoint)
            vit_state_dict = {k.replace("module.", ""): v for k, v in pre.items()}
            model.load_state_dict(vit_state_dict, strict=True)
            print(f"Using weights from {checkpoint}")

        if ngpu > 1:
            model = torch.nn.DataParallel(model)
        model = model.to(device)
        print(model.eval())
        print("Setup complete.\n")

        optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=args.wd)
        sched_builder = BuildScheduler(optimizer, args)
        print("Using scheduler:", sched_builder.name)

        # Determine effective LB coefficient based on toggle
        effective_lb_coef = float(args.lb_coef) if (bool(args.use_lb_loss) or float(args.lb_coef) > 0.0) else 0.0

        # Always use ModelTrainer with all improvements
        print("="*80)
        print("ModelTrainer Configuration:")
        print(f"  ✅ Expert diversity initialization: ENABLED")
        print(f"  ✅ Adaptive 3-phase warmup: ENABLED")
        print(f"  ✅ Load-balancing loss ({args.lb_loss_type}): {'ENABLED' if effective_lb_coef > 0 else 'DISABLED'}")
        print(f"  ✅ Straight-Through estimator: {'ENABLED' if args.use_st else 'DISABLED'}")
        print(f"  ✅ Gumbel exploration: {'ENABLED' if args.use_gumbel else 'DISABLED'}")
        print(f"  ✅ Expert usage tracking: ENABLED")
        print(f"  ✅ Clean validation metrics: ENABLED")
        print("="*80)

        train_obj = ModelTrainer(
            model=model,
            dataloaders=dataloaders,
            epochs=int(args.ep),
            optimizer=optimizer,
            sched_builder=sched_builder,
            class_count=class_count,
            device=device,
            early_stop=int(args.es),
            save_path=RESULT_FOLDER,
            grad_accum=int(args.grad_accum),
            lb_coef=effective_lb_coef,
            lb_loss_type=args.lb_loss_type,
            router_warmup_epochs=router_warmup_epochs,
            force_uniform_epochs=force_uniform_epochs,
            num_experts=args.num_expert,
            use_gumbel=bool(args.use_gumbel)
        )

        print("Starting training...")

        model, conf_matrix = train_obj.launch_training()

        np.save(RESULT_FOLDER / 'test_conf_matrix.npy', conf_matrix)
        plot_confusion_matrix(conf_matrix, RESULT_FOLDER, label_to_diagnose)

        end = time.time()
        runtime = end - start
        time_str = f"{int(runtime // 3600)}h{int((runtime % 3600) // 60)}min{int(runtime % 60)}s"

        # other parameters
        print("\n------------------------Final Report--------------------------")
        print('Runtime:', time_str)
        print('Fold:', fold)
        print('Architecture:', args.arch)
        print('Num Experts:', args.num_expert, f'(k={args.topk})')
        print('Warmup Schedule:', f'{force_uniform_epochs} uniform + {router_warmup_epochs-force_uniform_epochs} gradual = {router_warmup_epochs} total')
        print('Data path:', args.data_path)
        print('Result folder:', RESULT_FOLDER)
        print('Scheduler:', args.scheduler)
        print('Learning rate:', args.lr, f'(wd={args.wd})')
        print('Early stopping:', args.es)
        print('Gradient accumulation:', args.grad_accum)
        print('Seed:', seed)
        print('Max epochs:', args.ep)
        print('MoA - Adaptive 3-Phase Warmup')


if __name__ == "__main__":
    main()
