"""
eval_aux_mlp.py — TESTING ONLY, no production changes.

Trains a small auxiliary MLP on simulation-level aggregate labels from
ml_dataset.parquet, then evaluates three scenarios on the ground-truth gates:
  1. models_stageC_seed123 GRU alone
  2. Auxiliary MLP alone
  3. models_stageC_seed123 GRU OR MLP

Two training modes (--mode):
  reg  — regression (MSE on frac_stable, frac_habitable); threshold at --thresh
  cls  — binary classification (BCE on strict stable/habitable labels)

Output directories (never overwrite each other):
  eval_aux_mlp_output/regression/   ← regression model
  eval_aux_mlp_output/binary/       ← binary classification model

No existing model directories (models_temphead*, models_stageC*) are touched.

models_temphead recall (reference baseline):
  K-452b-v2 pro   stable=0.964  hab=0.396
  K-452b-v2 retro stable=0.939  hab=0.591
  K-1229b pro     stable=0.836  hab=0.218
  K-1229b retro   stable=0.946  hab=0.602
"""

import argparse
import os
import sys
import json
import pickle
import time

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

from exomoon.ml.dataset import SYS_COLS, LOG_SYS_COLS
from exomoon.ml.inference import predict_stability_map
from exomoon.params import SystemParams
from exomoon.initial_conditions import initial_state
from exomoon.habitable_zone import hz_bounds_au
from exomoon.constants import rsun

# ── paths ───────────────────────────────────────────────────────────────────────
PARQUET = os.path.join(SRC, "ml_dataset.parquet")
GRU_DIR = os.path.join(SRC, "models_stageC_seed123")
OUT_REG = os.path.join(SRC, "eval_aux_mlp_output", "regression")
OUT_BIN = os.path.join(SRC, "eval_aux_mlp_output", "binary")

GATES = {
    "K-452b-v2 pro":    os.path.join(SRC, "ground_truth_grids", "Kepler_452_b_v2_prograde"),
    "K-452b-v2 retro":  os.path.join(SRC, "ground_truth_grids", "Kepler_452_b_v2_retrograde"),
    "K-1229b pro":      os.path.join(SRC, "ground_truth_grids", "Kepler_1229_b_prograde"),
    "K-1229b retro":    os.path.join(SRC, "ground_truth_grids", "Kepler_1229_b_retrograde"),
    "TRAPPIST-1e pro":  os.path.join(SRC, "ground_truth_grids", "TRAPPIST_1_e_prograde"),
    "TRAPPIST-1e retro":os.path.join(SRC, "ground_truth_grids", "TRAPPIST_1_e_retrograde"),
}
# Keep only gates whose NPZ file actually exists
GATES = {k: v for k, v in GATES.items() if os.path.exists(v + ".npz")}

TEMPHEAD_REF = {
    "K-452b-v2 pro":   (0.964, 0.396),
    "K-452b-v2 retro": (0.939, 0.591),
    "K-1229b pro":     (0.836, 0.218),
    "K-1229b retro":   (0.946, 0.602),
}

SEED    = 42
EPOCHS  = 100
BATCH   = 256
LR      = 1e-3
HIDDEN  = 64
# Fraction threshold for binary-label construction:
# GT uses np.all() after warmup_steps=10 over 1000 resampled steps → 990/1000 = 0.990
BINARY_LABEL_THRESH = 0.99


# ── models ──────────────────────────────────────────────────────────────────────
class AuxMLP(nn.Module):
    """Regression model: outputs (frac_stable_pred, frac_hab_pred) via Sigmoid."""
    def __init__(self, input_dim=14, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, 2),         nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class AuxMLPBinary(nn.Module):
    """Binary classifier: outputs raw logits for (stable, habitable). No Sigmoid —
    BCEWithLogitsLoss is applied during training; sigmoid at inference."""
    def __init__(self, input_dim=14, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


# ── shared data helpers ─────────────────────────────────────────────────────────
def build_per_sim_df(parquet_path):
    """Collapse per-timestep parquet to one row per simulation.

    Adds frac_stable, frac_habitable (mean over all timesteps) and binary
    labels (fraction >= BINARY_LABEL_THRESH, matching the strict GT criterion
    which checks post-warmup steps ≈ 990/1000 = 0.990).
    """
    df      = pd.read_parquet(parquet_path)
    sys_df  = df.groupby("sim_id")[SYS_COLS].first().reset_index()
    frac_df = df.groupby("sim_id")[["stable", "habitable"]].mean().reset_index()
    frac_df.columns = ["sim_id", "frac_stable", "frac_habitable"]
    out = sys_df.merge(frac_df, on="sim_id")
    out["binary_stable"]   = (out["frac_stable"]   >= BINARY_LABEL_THRESH).astype(np.float32)
    out["binary_habitable"] = (out["frac_habitable"] >= BINARY_LABEL_THRESH).astype(np.float32)
    return out


def apply_log_transform(arr: np.ndarray) -> np.ndarray:
    arr = arr.copy()
    log_idx = [SYS_COLS.index(c) for c in LOG_SYS_COLS if c in SYS_COLS]
    arr[:, log_idx] = np.log(np.clip(arr[:, log_idx], 1e-10, None))
    return arr


def _make_splits(per_sim_df):
    sim_ids = per_sim_df["sim_id"].values
    np.random.seed(SEED)
    unique = np.unique(sim_ids)
    np.random.shuffle(unique)
    n_val   = int(len(unique) * 0.2)
    val_ids = set(unique[:n_val])
    trn_ids = set(unique[n_val:])
    return per_sim_df[per_sim_df["sim_id"].isin(trn_ids)], \
           per_sim_df[per_sim_df["sim_id"].isin(val_ids)]


# ── regression training ─────────────────────────────────────────────────────────
def train_mlp_regression(per_sim_df, out_dir):
    """MSE on (frac_stable, frac_habitable). Saves to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    train_df, val_df = _make_splits(per_sim_df)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(
        apply_log_transform(train_df[SYS_COLS].values.astype(np.float32))
    ).astype(np.float32)
    X_val   = scaler.transform(
        apply_log_transform(val_df[SYS_COLS].values.astype(np.float32))
    ).astype(np.float32)
    y_train = train_df[["frac_stable", "frac_habitable"]].values.astype(np.float32)
    y_val   = val_df  [["frac_stable", "frac_habitable"]].values.astype(np.float32)

    torch.manual_seed(SEED)
    model   = AuxMLP(input_dim=len(SYS_COLS), hidden=HIDDEN)
    opt     = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    loader  = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=BATCH, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    best_val, best_state, t0 = float("inf"), None, time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(torch.from_numpy(X_val)), torch.from_numpy(y_val)).item()
        if vl < best_val:
            best_val, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 20 == 0:
            print(f"  epoch {epoch:>3}/{EPOCHS}  val_loss={vl:.5f}  best={best_val:.5f}  ({time.time()-t0:.0f}s)")

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(out_dir, "aux_mlp.pt"))
    with open(os.path.join(out_dir, "aux_mlp_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Saved → {out_dir}/  (aux_mlp.pt, aux_mlp_scaler.pkl)\n")
    return model, scaler


# ── binary classification training ─────────────────────────────────────────────
def train_mlp_binary(per_sim_df, out_dir):
    """BCE on (binary_stable, binary_habitable) strict labels. Saves to out_dir.

    Binary labels: frac >= BINARY_LABEL_THRESH (0.99), matching the GT all-or-nothing
    criterion applied over post-warmup timesteps (~990/1000).
    Pos weights compensate for class imbalance (habitable class is rare).
    """
    os.makedirs(out_dir, exist_ok=True)
    train_df, val_df = _make_splits(per_sim_df)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(
        apply_log_transform(train_df[SYS_COLS].values.astype(np.float32))
    ).astype(np.float32)
    X_val   = scaler.transform(
        apply_log_transform(val_df[SYS_COLS].values.astype(np.float32))
    ).astype(np.float32)
    y_train = train_df[["binary_stable", "binary_habitable"]].values.astype(np.float32)
    y_val   = val_df  [["binary_stable", "binary_habitable"]].values.astype(np.float32)

    # Pos weights: (n_neg / n_pos) per output — handles habitable class imbalance
    n_pos = y_train.sum(axis=0).clip(1)
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32)
    print(f"  Binary label rates (train):  stable={y_train[:,0].mean():.3f}  habitable={y_train[:,1].mean():.3f}")
    print(f"  Pos weights:                 stable={pos_weight[0]:.2f}  habitable={pos_weight[1]:.2f}")

    torch.manual_seed(SEED)
    model   = AuxMLPBinary(input_dim=len(SYS_COLS), hidden=HIDDEN)
    opt     = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loader  = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=BATCH, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    best_val, best_state, t0 = float("inf"), None, time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(torch.from_numpy(X_val)), torch.from_numpy(y_val)).item()
        if vl < best_val:
            best_val, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 20 == 0:
            print(f"  epoch {epoch:>3}/{EPOCHS}  val_loss={vl:.5f}  best={best_val:.5f}  ({time.time()-t0:.0f}s)")

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(out_dir, "aux_mlp_binary.pt"))
    with open(os.path.join(out_dir, "aux_mlp_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    # Save a small config so callers know which model class to reconstruct
    with open(os.path.join(out_dir, "model_config.json"), "w") as f:
        json.dump({"mode": "binary", "hidden": HIDDEN, "input_dim": len(SYS_COLS),
                   "label_thresh": BINARY_LABEL_THRESH}, f, indent=2)
    print(f"  Saved to {out_dir}/  (aux_mlp_binary.pt, aux_mlp_scaler.pkl, model_config.json)\n")
    return model, scaler


# ── gate evaluation helpers ─────────────────────────────────────────────────────
def _rhill_hz_for_gate(meta):
    sp = meta["system_params"]
    p  = SystemParams(
        Ts=float(sp["Ts"]), rs_solar=float(sp["rs_solar"]),
        ms_solar=float(sp["ms_solar"]), mp_earth=float(sp["mp_earth"]),
        ap_AU=float(sp["ap_AU"]), ep=float(sp["ep"]),
        mm_earth=0.5, am_hill=0.4,
        em=float(meta["em"]), moon_retrograde=bool(meta["moon_retrograde"]),
    )
    st   = initial_state(p)
    rs_m = float(sp["rs_solar"]) * rsun
    a_in, a_out = hz_bounds_au(float(sp["Ts"]), rs_m)
    return float(st["rhill_AU"]), float(a_in), float(a_out)


def run_gru(meta):
    sp  = meta["system_params"]
    res = predict_stability_map(
        system_params=sp, t_sim=meta["t_sim"],
        moon_retrograde=meta["moon_retrograde"], em=meta["em"],
        mm_resolution=meta["mm_resolution"], am_resolution=meta["am_resolution"],
        model_dir=GRU_DIR, n_steps=1000,
    )
    if not res.get("ok"):
        return None
    return {
        "ml_stable": np.array(res["map_stable"]),
        "ml_hab":    np.array(res["map_habitable"]),
        "mm_grid":   np.array(res["mm_grid"]),
        "am_grid":   np.array(res["am_grid"]),
    }


def run_mlp(model, scaler, meta, mm_grid, am_grid, rhill, a_in, a_out, *, binary, thresh=0.5):
    """Evaluate MLP over the full (mm, am) grid for one gate.

    binary=True  → apply sigmoid to raw logits before thresholding
    binary=False → model already outputs probabilities via Sigmoid layer
    """
    sp    = meta["system_params"]
    retro = float(int(meta["moon_retrograde"]))
    em    = float(meta["em"])
    t_sim = float(meta["t_sim"])
    mm_r, am_r = len(mm_grid), len(am_grid)

    vecs = []
    for mm in mm_grid:
        for am in am_grid:
            vecs.append([
                float(sp["ms_solar"]), float(sp["rs_solar"]), float(sp["Ts"]),
                float(sp["mp_earth"]), float(sp["ap_AU"]),    float(sp["ep"]),
                float(mm), float(am), em, retro,
                t_sim, rhill, a_in, a_out,
            ])
    X_sc = scaler.transform(apply_log_transform(
        np.array(vecs, dtype=np.float32)
    )).astype(np.float32)

    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(X_sc))
        if binary:
            probs = torch.sigmoid(out).numpy()
        else:
            probs = out.numpy()

    return (probs[:, 0] >= thresh).reshape(mm_r, am_r), \
           (probs[:, 1] >= thresh).reshape(mm_r, am_r)


def recall_precision(gt_s, gt_h, ml_s, ml_h):
    gt_both = gt_s & gt_h
    n_gt    = int(gt_both.sum())
    if n_gt == 0:
        return None, None, None, None
    ml_both = ml_s & ml_h
    tp      = int((gt_both & ml_both).sum())
    rec     = tp / n_gt
    pre     = tp / max(int(ml_both.sum()), 1)
    f1      = 2 * rec * pre / (rec + pre) if (rec + pre) > 0 else 0.0
    return rec, pre, f1, n_gt


# ── main ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",   choices=["reg", "cls"], default="cls",
                        help="reg=regression/MSE (threshold at --thresh); cls=binary/BCE")
    parser.add_argument("--thresh", type=float, default=0.5,
                        help="Decision threshold for regression mode (default 0.5)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Load existing saved model, skip training")
    args = parser.parse_args()

    out_dir  = OUT_BIN if args.mode == "cls" else OUT_REG
    is_binary = args.mode == "cls"
    label    = "BINARY CLASSIFICATION (BCE)" if is_binary else f"REGRESSION (MSE, thresh={args.thresh})"
    pt_name  = "aux_mlp_binary.pt" if is_binary else "aux_mlp.pt"

    print()
    print("=" * 100)
    print(f"AUXILIARY MLP EVALUATION — {label}")
    print(f"Mode         : {args.mode}")
    print(f"GRU partner  : models_stageC_seed123")
    print(f"Output dir   : {out_dir}")
    print("=" * 100)

    # ── 1. aggregate dataset ──────────────────────────────────────────────────
    print("\n[1] Aggregating ml_dataset.parquet per simulation ...")
    per_sim = build_per_sim_df(PARQUET)
    print(f"    {len(per_sim)} simulations")
    print(f"    frac_stable mean      : {per_sim['frac_stable'].mean():.3f}")
    print(f"    frac_habitable mean   : {per_sim['frac_habitable'].mean():.3f}")
    if is_binary:
        print(f"    binary_stable rate    : {per_sim['binary_stable'].mean():.3f}  (frac >= {BINARY_LABEL_THRESH})")
        print(f"    binary_habitable rate : {per_sim['binary_habitable'].mean():.3f}  (frac >= {BINARY_LABEL_THRESH})")

    # ── 2. train or load MLP ──────────────────────────────────────────────────
    if args.skip_train:
        print(f"\n[2] Loading existing MLP from {out_dir}/ ...")
        with open(os.path.join(out_dir, "aux_mlp_scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        if is_binary:
            model = AuxMLPBinary(input_dim=len(SYS_COLS), hidden=HIDDEN)
        else:
            model = AuxMLP(input_dim=len(SYS_COLS), hidden=HIDDEN)
        model.load_state_dict(torch.load(os.path.join(out_dir, pt_name), weights_only=True))
        print("  Loaded.\n")
    else:
        print(f"\n[2] Training auxiliary MLP ({EPOCHS} epochs, hidden={HIDDEN}, seed={SEED}) ...")
        if is_binary:
            model, scaler = train_mlp_binary(per_sim, out_dir)
        else:
            model, scaler = train_mlp_regression(per_sim, out_dir)

    # ── 3. evaluate on gates ──────────────────────────────────────────────────
    print(f"[3] Evaluating on {len(GATES)} ground-truth gates ...\n")

    W  = 8
    print(f"{'Gate':<20}  {'Metric':<10}  {'seed123 GRU':>{W*3+4}}  {'Aux MLP':>{W*3+4}}  {'GRU OR MLP':>{W*3+4}}  {'temphead':>10}")
    print(f"{'':20}  {'':10}  {'rec':>{W}} {'prec':>{W}} {'F1':>{W}}  {'rec':>{W}} {'prec':>{W}} {'F1':>{W}}  {'rec':>{W}} {'prec':>{W}} {'F1':>{W}}  {'hab rec':>10}")
    print("-" * 120)

    all_gru, all_mlp, all_or = [], [], []

    for gate_name, gate_prefix in GATES.items():
        npz  = np.load(f"{gate_prefix}.npz", allow_pickle=True)
        with open(f"{gate_prefix}.meta.json") as f:
            meta = json.load(f)

        gt_s = npz["map_stable"].astype(bool)
        gt_h = npz["map_habitable"].astype(bool)

        gru = run_gru(meta)
        if gru is None:
            print(f"{gate_name:<20}  GRU load error — check {GRU_DIR}")
            continue

        rhill, a_in, a_out = _rhill_hz_for_gate(meta)
        mlp_s, mlp_h = run_mlp(
            model, scaler, meta,
            gru["mm_grid"], gru["am_grid"], rhill, a_in, a_out,
            binary=is_binary, thresh=args.thresh,
        )

        or_s = gru["ml_stable"] | mlp_s
        or_h = gru["ml_hab"]    | mlp_h

        gru_r, gru_p, gru_f, n_gt = recall_precision(gt_s, gt_h, gru["ml_stable"], gru["ml_hab"])
        mlp_r, mlp_p, mlp_f, _    = recall_precision(gt_s, gt_h, mlp_s, mlp_h)
        or_r,  or_p,  or_f,  _    = recall_precision(gt_s, gt_h, or_s,  or_h)

        th_h = TEMPHEAD_REF.get(gate_name, (None, None))[1]
        th_str = f"{th_h:.3f}" if th_h is not None else "   —"

        def _fmt(v): return f"{v:.3f}" if v is not None else "  —  "

        tag = f"(n_gt={n_gt})" if n_gt else ""
        print(f"{gate_name:<20}  {'hab':10}  "
              f"{_fmt(gru_r):>{W}} {_fmt(gru_p):>{W}} {_fmt(gru_f):>{W}}  "
              f"{_fmt(mlp_r):>{W}} {_fmt(mlp_p):>{W}} {_fmt(mlp_f):>{W}}  "
              f"{_fmt(or_r):>{W}}  {_fmt(or_p):>{W}} {_fmt(or_f):>{W}}  "
              f"{th_str:>10}  {tag}")

        if gru_r is not None: all_gru.append(gru_r)
        if mlp_r is not None: all_mlp.append(mlp_r)
        if or_r  is not None: all_or.append(or_r)

    print("-" * 120)
    if all_mlp:
        print(f"{'Average hab recall':<32}  "
              f"{'':>{W}} {'':>{W}} {'':>{W}}  "
              f"{np.mean(all_gru):.3f}{'':>{W}} {'':>{W}}  "
              f"{np.mean(all_mlp):.3f}{'':>{W}} {'':>{W}}  "
              f"{np.mean(all_or):.3f}  [temphead ref: 0.452]")
    print(f"\nDone. Model saved in {out_dir}/  — no production models modified.")


if __name__ == "__main__":
    main()
