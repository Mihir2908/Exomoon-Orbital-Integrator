"""
train_force_mlp.py — Train three direct force regression MLPs for 2D trajectory preview.

This is the second trajectory-preview option alongside HNN v12 (models_hnn_dist_v12/).
HNN v12 is the primary completed model; these MLPs provide an alternative path that
avoids HNN's autograd chain-rule scaling problem for t_pm.

Three independent MLPs trained jointly with a single Adam optimizer:
  MLP_sp : (d_sp_n, sys_enc) -> t_sp   [star-planet force]
  MLP_sm : (d_sm_n, sys_enc) -> t_sm   [star-moon force]
  MLP_pm : (d_pm_n, sys_enc) -> t_pm   [planet-moon force]

Uses the same dataset and training filter as HNN v12:
  - Exclude simulations whose first timestep is not both stable+habitable
  - Exclude rows after the first timestep where stable=0 AND habitable=0
  (filter is applied inside hnn_dataset.py's make_hnn_splits)

Loss: log-MSE on each pair independently, summed:
  L = log_mse(pred_sp, t_sp) + log_mse(pred_sm, t_sm) + log_mse(pred_pm, t_pm)
  log_mse(p, t) = mean((log(p) - log(t))^2)

Usage:
  py train_force_mlp.py --data ml_dataset.parquet --out models_force_mlp/ --epochs 50

Output (all in --out directory):
  force_mlp_sp.pt          best MLP_sp weights (lowest combined val_loss)
  force_mlp_sm.pt          best MLP_sm weights
  force_mlp_pm.pt          best MLP_pm weights
  force_mlp_config.json    architecture + training metadata
  hnn_sys_scaler.pkl       StandardScaler for sys_enc (same format as HNN v12)
  force_mlp_history.json   per-epoch loss curves

ISOLATION: does NOT touch models/, models_temphead/, eval_aux_mlp_output/,
           models_hnn_dist_v12/, or any GRU/classification infrastructure.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from exomoon.ml.hnn_dataset     import make_hnn_splits, save_sys_scaler
from exomoon.ml.force_mlp_model import ForceMLP, ForceMLPEnsemble


# ── Loss ──────────────────────────────────────────────────────────────────────

def log_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Scale-invariant log-domain MSE: mean((log(pred) - log(target))^2).
    Same loss used in HNN v12 training."""
    lp = torch.log(pred.clamp(min=1e-9))
    lt = torch.log(target.clamp(min=1e-9))
    return ((lp - lt) ** 2).mean()


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    data_path:  str,
    out_dir:    str,
    epochs:     int   = 50,
    batch_size: int   = 4096,
    lr:         float = 1e-3,
    hidden:     int   = 128,
    layers:     int   = 3,
    val_frac:   float = 0.20,
    seed:       int   = 42,
    verbose:    bool  = True,
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    t0 = time.time()

    if verbose:
        print(f"[force_mlp] Loading {data_path} …", flush=True)
    train_ds, val_ds, sys_scaler = make_hnn_splits(data_path, val_frac=val_frac, seed=seed)
    if verbose:
        print(f"[force_mlp] Train: {len(train_ds):,} rows  |  Val: {len(val_ds):,} rows", flush=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )

    # Three independent MLPs — one per body pair
    mlp_sp = ForceMLP(sys_dim=5, hidden=hidden, layers=layers)
    mlp_sm = ForceMLP(sys_dim=5, hidden=hidden, layers=layers)
    mlp_pm = ForceMLP(sys_dim=5, hidden=hidden, layers=layers)
    ensemble = ForceMLPEnsemble(mlp_sp, mlp_sm, mlp_pm)

    all_params = list(mlp_sp.parameters()) + list(mlp_sm.parameters()) + list(mlp_pm.parameters())
    opt = torch.optim.Adam(all_params, lr=lr)

    # Val tensors preloaded for fast per-epoch validation
    val_d  = val_ds.d_n       # (N_val, 3)
    val_sy = val_ds.sys_enc   # (N_val, 5)
    val_t  = val_ds.t_grad    # (N_val, 3)

    history: dict = {
        "train_loss": [], "val_loss": [],
        "val_sp": [], "val_sm": [], "val_pm": [],
    }
    best_val    = float("inf")
    best_states = None

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        ensemble.train()
        tr_loss = 0.0
        n_rows  = 0
        for d_n, sys_enc, t_grad in train_loader:
            pred_sp = mlp_sp(d_n[:, 0:1], sys_enc)
            pred_sm = mlp_sm(d_n[:, 1:2], sys_enc)
            pred_pm = mlp_pm(d_n[:, 2:3], sys_enc)

            loss = (log_mse(pred_sp, t_grad[:, 0:1])
                  + log_mse(pred_sm, t_grad[:, 1:2])
                  + log_mse(pred_pm, t_grad[:, 2:3]))

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=5.0)
            opt.step()

            tr_loss += loss.item() * len(d_n)
            n_rows  += len(d_n)
        tr_loss /= max(n_rows, 1)

        # ── Validate ──────────────────────────────────────────────────────────
        ensemble.eval()
        with torch.no_grad():
            p_sp = mlp_sp(val_d[:, 0:1], val_sy)
            p_sm = mlp_sm(val_d[:, 1:2], val_sy)
            p_pm = mlp_pm(val_d[:, 2:3], val_sy)
            v_sp    = log_mse(p_sp, val_t[:, 0:1]).item()
            v_sm    = log_mse(p_sm, val_t[:, 1:2]).item()
            v_pm    = log_mse(p_pm, val_t[:, 2:3]).item()
            v_total = v_sp + v_sm + v_pm

        history["train_loss"].append(round(tr_loss, 6))
        history["val_loss"].append(round(v_total, 6))
        history["val_sp"].append(round(v_sp, 6))
        history["val_sm"].append(round(v_sm, 6))
        history["val_pm"].append(round(v_pm, 6))

        if v_total < best_val:
            best_val    = v_total
            best_states = [
                {k: v.clone() for k, v in m.state_dict().items()}
                for m in [mlp_sp, mlp_sm, mlp_pm]
            ]

        if verbose and (epoch % 5 == 0 or epoch == epochs):
            print(
                f"  epoch {epoch:>3}/{epochs}  "
                f"train={tr_loss:.5f}  val={v_total:.5f}  "
                f"[sp={v_sp:.4f}  sm={v_sm:.4f}  pm={v_pm:.4f}]  "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )

    # Load best weights
    for mlp, state in zip([mlp_sp, mlp_sm, mlp_pm], best_states):
        mlp.load_state_dict(state)

    best_epoch = int(np.argmin(history["val_loss"])) + 1
    if verbose:
        print(f"\n[force_mlp] Best val_loss={best_val:.5f}  (epoch {best_epoch}/{epochs})", flush=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    cfg = {
        "sys_dim":       5,
        "hidden":        hidden,
        "layers":        layers,
        "epochs":        epochs,
        "lr":            lr,
        "batch_size":    batch_size,
        "best_val_loss": best_val,
        "best_epoch":    best_epoch,
        "data_path":     os.path.abspath(data_path),
    }
    ensemble.save(out_dir, cfg)
    save_sys_scaler(sys_scaler, out_dir)    # writes hnn_sys_scaler.pkl (same format as HNN v12)

    history["best_val_loss"] = best_val
    history["best_epoch"]    = best_epoch
    history["hyperparams"]   = cfg
    with open(os.path.join(out_dir, "force_mlp_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    if verbose:
        print(f"[force_mlp] Saved to {out_dir}/", flush=True)

    return history


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Train three direct force regression MLPs for 2D trajectory preview."
    )
    ap.add_argument("--data",       required=True, help="Path to .parquet training dataset")
    ap.add_argument("--out",        required=True, help="Output directory (NOT models/ or models_hnn_dist_v12/)")
    ap.add_argument("--epochs",     type=int,   default=50)
    ap.add_argument("--batch_size", type=int,   default=4096)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--hidden",     type=int,   default=128)
    ap.add_argument("--layers",     type=int,   default=3)
    ap.add_argument("--seed",       type=int,   default=42)
    ap.add_argument("--quiet",      action="store_true")
    args = ap.parse_args()

    # Guard against accidentally clobbering protected dirs
    forbidden = {"models", "models_temphead", "eval_aux_mlp_output", "models_hnn_dist_v12"}
    out_base  = os.path.basename(os.path.normpath(args.out))
    if out_base in forbidden:
        raise SystemExit(f"ERROR: --out '{args.out}' targets a protected directory. "
                         f"Use a separate directory such as models_force_mlp/")

    train(
        data_path  = args.data,
        out_dir    = args.out,
        epochs     = args.epochs,
        batch_size = args.batch_size,
        lr         = args.lr,
        hidden     = args.hidden,
        layers     = args.layers,
        seed       = args.seed,
        verbose    = not args.quiet,
    )
