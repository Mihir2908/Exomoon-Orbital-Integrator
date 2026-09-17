import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
from torch.utils.data import DataLoader

from exomoon.ml.dataset import make_splits, MoonDataset, TARGET_DIST_COLS
from exomoon.ml.model import MoonRNN

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "models"
print(f"MODEL_DIR={MODEL_DIR}")

train_df, val_df, sys_scaler, state_scaler = make_splits("ml_dataset.parquet", val_frac=0.20, seed=42)
val_ds = MoonDataset(val_df, sys_scaler, state_scaler)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

model = MoonRNN.load(MODEL_DIR, rnn_type="gru")
model.eval()

abs_err_sum = np.zeros(len(TARGET_DIST_COLS))
count = 0

with torch.no_grad():
    for state_seq, sys_p, targets, dist_mask in val_loader:
        dist_pred, flag_logits = model(state_seq, sys_p)
        target_dist = targets[..., :len(TARGET_DIST_COLS)]
        abs_err = (dist_pred - target_dist).abs() * dist_mask   # only count stable (masked) rows
        abs_err_sum += abs_err.sum(dim=(0, 1)).numpy()
        count += dist_mask.sum().item()

print("Per-column MAE (only over stable/masked rows):")
for name, total in zip(TARGET_DIST_COLS, abs_err_sum):
    print(f"  {name:25s}: {total / count:.5f}")
