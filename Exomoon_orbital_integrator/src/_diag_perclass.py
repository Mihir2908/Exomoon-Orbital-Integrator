"""
Per-class precision/recall for stable and habitable on the actual validation set,
using the just-trained model -- raw flag_accuracy is class-imbalance-blind and
doesn't tell us whether habitable specifically improved.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from torch.utils.data import DataLoader

from exomoon.ml.dataset import make_splits, MoonDataset, TARGET_FLAG_COLS
from exomoon.ml.model import MoonRNN

train_df, val_df, sys_scaler, state_scaler = make_splits("ml_dataset.parquet", val_frac=0.20, seed=42)
val_ds = MoonDataset(val_df, sys_scaler, state_scaler)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

model = MoonRNN.load("models", rnn_type="gru")
model.eval()

all_flag_pred = []
all_flag_true = []

with torch.no_grad():
    for state_seq, sys_p, targets, dist_mask in val_loader:
        dist_pred, flag_logits = model(state_seq, sys_p)
        flag_pred = (flag_logits > 0).float()
        flag_true = targets[..., -2:]
        all_flag_pred.append(flag_pred.reshape(-1, 2).numpy())
        all_flag_true.append(flag_true.reshape(-1, 2).numpy())

pred = np.concatenate(all_flag_pred, axis=0)
true = np.concatenate(all_flag_true, axis=0)

for i, name in enumerate(TARGET_FLAG_COLS):
    p, t = pred[:, i], true[:, i]
    tp = ((p == 1) & (t == 1)).sum()
    fp = ((p == 1) & (t == 0)).sum()
    fn = ((p == 0) & (t == 1)).sum()
    tn = ((p == 0) & (t == 0)).sum()
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    print(f"=== {name} ===")
    print(f"  true positive rate in val set: {t.mean():.4f}")
    print(f"  predicted positive rate:       {p.mean():.4f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  precision={precision:.4f}  recall={recall:.4f}")
    print()
