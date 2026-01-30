# Mixture of Aggregators (MoA)

A PyTorch implementation for whole slide image classification using multiple MIL aggregation experts with learned routing.

## Project Structure

```
MoA_paper_code/
├── train.py                 # Main training script
├── classifier.py            # MIL_aggregtor wrapper
├── model_train.py           # Training loop and metrics
├── dataset.py               # MILDataset for .h5/.pt files
├── scheduler.py             # LR scheduler
├── plot_confusion.py        # Confusion matrix plotting
├── patient_data.py          # Patient data structures
│
└── models/
    ├── mixture_of_aggregators.py    # MoA architecture
    ├── abmil_expert.py              # ABMIL expert
    ├── transmil_expert.py           # TransMIL expert
    ├── transformer_expert.py        # Transformer expert
    └── ...
```

## Requirements

```bash
pip install torch pandas numpy scikit-learn h5py tqdm
```

## Data Layout

```
embeddings/
├── patient_001.h5    # Shape: [N_patches, D_features]
├── patient_002.h5
└── ...

csv_root/
├── label_mapping.csv
├── data_fold_0/
│   ├── train.csv     # Columns: patient_name, labels
│   ├── val.csv
│   └── test.csv
└── ...
```

## Training

### MoA with ABMIL Experts

```bash
python train.py \
  --data_path /path/to/embeddings \
  --saving_name results/moa_abmil \
  --csv_root /path/to/csv_root \
  --label_map_csv /path/to/csv_root/label_mapping.csv \
  --num_expert 4 \
  --expert_arch ABMIL \
  --router_style topk \
  --topk 2 \
  --use_lb_loss \
  --lb_coef 0.01 \
  --use_gumbel
```

### MoA with TransMIL Experts

```bash
python train.py \
  --data_path /path/to/embeddings \
  --saving_name results/moa_transmil \
  --csv_root /path/to/csv_root \
  --label_map_csv /path/to/csv_root/label_mapping.csv \
  --num_expert 4 \
  --expert_arch TransMIL \
  --router_style topk \
  --topk 2 \
  --use_lb_loss \
  --lb_coef 0.01 \
  --use_gumbel
```

## Key Arguments

| Argument | Value | Description |
|----------|-------|-------------|
| `--num_expert` | 4 | Number of experts |
| `--expert_arch` | ABMIL / TransMIL | Expert architecture |
| `--router_style` | topk | Sparse routing |
| `--topk` | 2 | Top-2 expert selection |
| `--use_lb_loss` | flag | Enable load balancing |
| `--lb_coef` | 0.01 | Load balancing coefficient |
| `--use_gumbel` | flag | Gumbel-softmax exploration |

## Output

Results saved to: `<saving_name>/<num_expert>_experts/Results_5fold_.../fold_*/`
- `best_model.pt` - Model checkpoint
- `confusion_matrix.png` - Test results
