"""
Diagnostic: per-simulation TEACHER-FORCED habitable performance on the validation
split, correlated against that sim's ep and rhill_AU, to properly test (with n=595,
not n=2) whether either variable predicts how well the model represents habitable
dynamics -- as opposed to the autoregressive-rollout-specific divergence already
characterized elsewhere. Teacher-forced isolates representational capacity from
exposure bias: every input here is the true ground-truth value, never the model's
own prior prediction.

Read-only: no training, no model files modified.
"""
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from exomoon.ml.dataset import TARGET_DIST_COLS, TARGET_FLAG_COLS, MoonDataset, make_splits
from exomoon.ml.model import MoonRNN
from exomoon.ml.dataset import load_normalizer

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "models_temphead"
# models_temphead / models_temphead_ss_prefix were trained before the log-transform
# fix existed -- their normalizer.pkl was fit on raw values. models_logfix (and
# anything warm-started from it) expects the log-transform. Pass log_transform=False
# explicitly for pre-fix checkpoints or this silently corrupts every prediction.
LOG_TRANSFORM = sys.argv[2].lower() != "false" if len(sys.argv) > 2 else ("logfix" in MODEL_DIR)
HABITABLE_IDX = len(TARGET_DIST_COLS) + TARGET_FLAG_COLS.index("habitable")

print(f"[persim] Loading validation split (log_transform={LOG_TRANSFORM}) ...", flush=True)
train_df, val_df, _, _ = make_splits("ml_dataset.parquet", log_transform=LOG_TRANSFORM)
model = MoonRNN.load(MODEL_DIR, rnn_type="gru"); model.eval()
sys_scaler, state_scaler = load_normalizer(os.path.join(MODEL_DIR, "normalizer.pkl"))

ds = MoonDataset(val_df, sys_scaler, state_scaler)
loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

# per-sim ep / rhill_AU (raw, unscaled) for correlation
sim_first = val_df.groupby("sim_id").first()

rows = []
with torch.no_grad():
    for i, (state_seq, sys_p, targets, _mask) in enumerate(loader):
        sim_id = ds.sim_ids[i]
        hab_target = targets[..., HABITABLE_IDX:HABITABLE_IDX+1]
        _, flag_logits = model(state_seq, sys_p)
        hab_logit = flag_logits[..., 1:2]
        pred = (hab_logit > 0).float()
        correct = (pred == hab_target).float().mean().item()
        tp = ((pred==1)&(hab_target==1)).sum().item()
        fn = ((pred==0)&(hab_target==1)).sum().item()
        recall = tp / max(tp+fn, 1) if (tp+fn) > 0 else np.nan
        ep = float(sim_first.loc[sim_id, "ep"])
        rhill = float(sim_first.loc[sim_id, "rhill_AU"])
        ts = float(sim_first.loc[sim_id, "Ts"])
        ap = float(sim_first.loc[sim_id, "ap_AU"])
        mp = float(sim_first.loc[sim_id, "mp_earth"])
        mm = float(sim_first.loc[sim_id, "mm_earth"])
        tsim = float(sim_first.loc[sim_id, "t_sim"])
        rows.append((sim_id, ep, rhill, ts, ap, mp, mm, tsim, correct, recall, tp+fn))

import pandas as pd
res = pd.DataFrame(rows, columns=["sim_id","ep","rhill_AU","Ts","ap_AU","mp_earth","mm_earth","t_sim","tf_accuracy","tf_recall","n_pos"])
print(f"Total val sims: {len(res)}")
print()
print("--- Correlation: per-sim teacher-forced habitable accuracy vs ep / rhill_AU / Ts / ap_AU ---")
for col in ["ep", "rhill_AU", "Ts", "ap_AU"]:
    r = res[["tf_accuracy", col]].corr().iloc[0,1]
    print(f"  corr(tf_accuracy, {col}) = {r:+.4f}")
print()
print("--- ep groups: teacher-forced accuracy ---")
low_ep = res[res["ep"] < 0.02]
high_ep = res[res["ep"] >= 0.02]
print(f"  ep<0.02  (n={len(low_ep)}):  mean accuracy={low_ep['tf_accuracy'].mean():.4f}  mean recall(n_pos>0)={low_ep[low_ep['n_pos']>0]['tf_recall'].mean():.4f}")
print(f"  ep>=0.02 (n={len(high_ep)}): mean accuracy={high_ep['tf_accuracy'].mean():.4f}  mean recall(n_pos>0)={high_ep[high_ep['n_pos']>0]['tf_recall'].mean():.4f}")
print()
print("--- dose-response check: bottom/mid/top decile recall for each log-sampled column ---")
print("    (if resolution-compression were the dominant driver, the column with the")
print("     WORST raw compression -- mp_earth, 290x -- should show the WORST bottom-decile")
print("     gap pre-fix, and the BIGGEST improvement post-fix. rhill_AU was 21x, mm_earth 5.5x, t_sim 14x.)")
print()
for col in ["mp_earth", "rhill_AU", "t_sim", "mm_earth"]:
    res[f"{col}_pct"] = res[col].rank(pct=True) * 100
    low  = res[res[f"{col}_pct"] < 10]
    mid  = res[(res[f"{col}_pct"]>=45)&(res[f"{col}_pct"]<55)]
    high = res[res[f"{col}_pct"] >= 90]
    low_r  = low[low["n_pos"]>0]["tf_recall"].mean()
    mid_r  = mid[mid["n_pos"]>0]["tf_recall"].mean()
    high_r = high[high["n_pos"]>0]["tf_recall"].mean()
    print(f"  {col:10s}: bottom10% recall={low_r:.4f} (n={len(low)})   mid recall={mid_r:.4f} (n={len(mid)})   top10% recall={high_r:.4f} (n={len(high)})")
