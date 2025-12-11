# MoA v2 Training


## Requirements
- Python 3.9+, PyTorch, pandas, numpy, scikit-learn

## Data layout
- `--data_path`: root with precomputed slide embeddings (`.h5` or `.pt`).
- `--csv_root`: folder containing `data_fold_*` with `train.csv`, `val.csv`, `test.csv` (use `--prepared_fold` to rely on these splits).
- `--label_map_csv`: CSV mapping label ids to diagnoses (defaults to `<csv_root>/3class_label_mapping.csv`).

## Run
```bash
python train.py \
  --data_path /path/to/embeddings \
  --saving_name runs/moa_v2 \
  --csv_root data_cross_val_3_classes_Bracs \
  --label_map_csv data_cross_val_3_classes_Bracs/3class_label_mapping.csv \
  --arch Transformer --expert_arch Transformer \
  --router_style topk --router_type linear --num_expert 2 --topk 1 \
  --lr 5e-5 --wd 0.01 --ep 150 --es 15 --grad_accum 16
```
Results are saved under `runs/moa_v2/<num_expert>_experts/Results_v2_.../fold_*`. Already-finished folds (with `confusion_matrix.png` or `test_conf_matrix.npy`) are skipped.

## Notes
- Uses CUDA if available; set `--checkpoint` to resume weights.
- Load-balancing loss stays off unless `--use_lb_loss` is passed (tune `--lb_coef` and `--lb_loss_type`).
- Optional exploration: `--use_gumbel` adds Gumbel noise; router warmup is controlled by `--router_warmup_epochs` and `--force_uniform_epochs`.
