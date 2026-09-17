"""
Diagnostic: is the low-mp_earth/rhill_AU recall gap caused by (a) genuinely harder/
noisier ground-truth dynamics in that regime (the task itself is harder, not a model
failure), or (b) the model's continuous regression being specifically worse there
(a real representational/learning failure)? Checks, per validation sim:
  - habitable label volatility: how many times habitable flips 0<->1 across the
    1000-step sequence (a flippier ground truth is a genuinely harder target,
    independent of model quality)
  - teacher-forced continuous regression error (MSE) on moon_star_dist_norm specifically
    (not just the thresholded binary recall) -- elevated MSE there would mean the
    model's predictions are off in real terms, not just mis-classified near a boundary

Read-only, no training, no model files modified.
"""
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from exomoon.ml.dataset import TARGET_DIST_COLS, TARGET_FLAG_COLS, MoonDataset, make_splits, load_normalizer
from exomoon.ml.model import MoonRNN

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "models_logfix"
LOG_TRANSFORM = sys.argv[2].lower() != "false" if len(sys.argv) > 2 else ("logfix" in MODEL_DIR)
HABITABLE_IDX = len(TARGET_DIST_COLS) + TARGET_FLAG_COLS.index("habitable")
MSD_IDX = TARGET_DIST_COLS.index("moon_star_dist_norm")   # index within dist-pred output

print(f"[physmech] Loading validation split (log_transform={LOG_TRANSFORM}) ...", flush=True)
train_df, val_df, _, _ = make_splits("ml_dataset.parquet", log_transform=LOG_TRANSFORM)
model = MoonRNN.load(MODEL_DIR, rnn_type="gru"); model.eval()
sys_scaler, state_scaler = load_normalizer(os.path.join(MODEL_DIR, "normalizer.pkl"))

ds = MoonDataset(val_df, sys_scaler, state_scaler)
loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
sim_first = val_df.groupby("sim_id").first()

# habitable label volatility computed directly from val_df (ground truth, not model)
hab_by_sim = val_df.sort_values(["sim_id", "t_frac"]).groupby("sim_id")["habitable"]
flip_counts = hab_by_sim.apply(lambda s: int(np.sum(np.abs(np.diff(s.values)) > 0)))

rows = []
with torch.no_grad():
    for i, (state_seq, sys_p, targets, _mask) in enumerate(loader):
        sim_id = ds.sim_ids[i]
        hab_target = targets[..., HABITABLE_IDX:HABITABLE_IDX+1]
        msd_target = targets[..., MSD_IDX:MSD_IDX+1]
        dist_pred, flag_logits = model(state_seq, sys_p)
        msd_pred = dist_pred[..., MSD_IDX:MSD_IDX+1]
        mse = float(torch.mean((msd_pred - msd_target) ** 2))
        hab_logit = flag_logits[..., 1:2]
        pred = (hab_logit > 0).float()
        tp = ((pred==1)&(hab_target==1)).sum().item()
        fn = ((pred==0)&(hab_target==1)).sum().item()
        recall = tp / max(tp+fn, 1) if (tp+fn) > 0 else np.nan
        mp = float(sim_first.loc[sim_id, "mp_earth"])
        rhill = float(sim_first.loc[sim_id, "rhill_AU"])
        flips = int(flip_counts.loc[sim_id])
        rows.append((sim_id, mp, rhill, mse, recall, tp+fn, flips))

import pandas as pd
res = pd.DataFrame(rows, columns=["sim_id","mp_earth","rhill_AU","msd_mse","tf_recall","n_pos","flips"])
print(f"Total val sims: {len(res)}")
print()
for col in ["mp_earth", "rhill_AU"]:
    res[f"{col}_pct"] = res[col].rank(pct=True) * 100
    low  = res[res[f"{col}_pct"] < 10]
    mid  = res[(res[f"{col}_pct"]>=45)&(res[f"{col}_pct"]<55)]
    high = res[res[f"{col}_pct"] >= 90]
    print(f"--- grouped by {col} percentile ---")
    for label, grp in [("bottom10%", low), ("mid45-55%", mid), ("top10%", high)]:
        r = grp[grp["n_pos"]>0]["tf_recall"].mean()
        print(f"  {label} (n={len(grp)}): mean habitable label flips/sim={grp['flips'].mean():6.2f}   "
              f"mean moon_star_dist_norm MSE={grp['msd_mse'].mean():.5f}   mean tf_recall={r:.4f}")
    print()
