"""
hnn_rollout_train_hill.py  —  Hill-frame HNN rollout training (Path 3).

Trains HNN_Hill with a MIXED loss:
  L_total = L_hinge(grad_0, t_hill_0)  +  lambda_r * L_rollout

  L_hinge  : existing hinge loss at the starting step (force supervision, GT position).
  L_rollout: mean normalised position MSE over k integration steps using HNN forces.

             L_rollout = (1/k) * sum_t { ||q_mm_hnn_t - q_mm_gt_t||^2 / rhill^2 }

Why this fixes covariate shift
  Standard training (hnn_train_hill.py) sees only GT positions -> each batch is IID
  snapshots shuffled across all sims.  The model never sees its own integrated positions
  during training, so small per-step force errors compound over hundreds of leapfrog
  steps at inference.  L_rollout forces the model to produce trajectories close to GT,
  not just point-wise correct forces.

Rollout integration (replicates hnn_inference_hill.py EXACTLY, create_graph=True):
  - KDK leapfrog in INERTIAL coordinates (star + planet + moon)
  - Hill frame rebuilt per step from current r_sp (not fixed at t=0)
  - HNN force in Hill frame: a_cons_h = -grad_V * V_ref / (mm_msun * rhill)
  - Centrifugal stripped: a_hill = a_cons_h - Omega^2 * q_h_plane
  - Rotated to inertial: a_rel_in = a_hill @ [x_hat, y_hat, z_hat]^T
  - Moon inertial accel: a_m = a_rel_in + a_p (planet accel from star)
  - Star accel: from moon + planet
  - Timestep: dt_resamp = t_sim / n_resampled_steps (matches parquet, NOT physics dt)

Dataset:
  Groups parquet by sim_id.  For each sim of n_rows rows, creates (n_rows - k) windows
  of length k+1.  Applies same filter as hnn_dataset_hill.py (trim at first step where
  both stable=0 AND habitable=0).  Returns initial state + next k GT moon positions.

Warmstart: --warmstart models_hnn_hill_hinge4 (required; do not train from scratch)
Output:    --out models_hnn_hill_hinge8_rollout (default)

Artifacts saved to --out:
  hnn_hill_model.pt, hnn_hill_config.json, hnn_hill_sys_scaler.pkl
  hnn_hill_train_status.json, hnn_hill_training_history.json

Usage
-----
  cd Exomoon_orbital_integrator/src
  py hnn_rollout_train_hill.py ^
      --data ml_dataset.parquet ^
      --warmstart models_hnn_hill_hinge4 ^
      --out models_hnn_hill_hinge8_rollout ^
      --epochs 30 --k 10 --batch 32 --lr 1e-5 --lambda_r 0.01
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from exomoon.constants            import FOUR_PI2
from exomoon.ml.hnn_model_hill    import HNN_Hill
from exomoon.ml.hnn_dataset_hill  import (
    _compute_hill_targets,
    save_hill_sys_scaler,
)

_MERTH_OVER_MSUN = 5.976e24 / 1.989e30


# ── CLI ────────────────────────────────────────────────────────────────────────

ap = argparse.ArgumentParser(description="Hill-frame HNN rollout training (Path 3)")
ap.add_argument("--data",       default="ml_dataset.parquet")
ap.add_argument("--out",        default="models_hnn_hill_rollout_v3",
                help="Output directory. Default: models_hnn_hill_rollout_v3 (rollout_v3 = physics-dt mode).")
ap.add_argument("--warmstart",  default="models_hnn_hill_hinge4",
                help="Directory to load initial weights from (required for stability)")
ap.add_argument("--epochs",     type=int,   default=30)
ap.add_argument("--batch",      type=int,   default=32,
                help="Batch size (smaller than snapshot training due to rollout cost)")
ap.add_argument("--lr",         type=float, default=1e-5)
ap.add_argument("--k",          type=int,   default=1,
                help="Number of parquet-step lookaheads per window. Default 1 for rollout_v3 (physics-dt mode).")
ap.add_argument("--n_fine",     type=int,   default=5,
                help="[Legacy / --no_physics_dt mode only] Sub-steps per parquet interval at dt_resamp/n_fine. "
                     "Ignored when --physics_dt is active (default).")
ap.add_argument("--no_physics_dt", action="store_true", default=False,
                help="Revert to rollout_v2 behaviour: n_fine sub-steps at dt_resamp/n_fine with loss clamp. "
                     "Default OFF — physics-dt mode (rollout_v3) is the default.")
ap.add_argument("--lambda_r",   type=float, default=0.1,
                help="Weight for rollout MSE loss (auxiliary). Meaningful after clamp.")
ap.add_argument("--max_win",    type=int,   default=10,
                help="Max windows sampled per sim (default 10). "
                     "Limits dataset to ~N_sims*max_win windows — avoids the "
                     "1.9M-window explosion from full sliding-window coverage.")
ap.add_argument("--val_frac",   type=float, default=0.20)
ap.add_argument("--seed",       type=int,   default=42)
ap.add_argument("--patience",      type=int,   default=7,
                help="LR scheduler patience (ReduceLROnPlateau).")
ap.add_argument("--patience_stop", type=int,   default=8,
                help="Early stopping patience: halt if val_loss does not improve "
                     "for this many consecutive epochs.")
ap.add_argument("--hidden",     type=int,   default=256)
ap.add_argument("--layers",     type=int,   default=3)
args = ap.parse_args()

# Safety: refuse protected dirs
_PROTECTED = {
    "models", "models_temphead", "models_hnn_log", "models_hnn_tidal",
    "models_hnn_dist_v12", "eval_aux_mlp_output", "models_force_mlp",
    "models_hnn_greydanus", "models_hnn_greydanus_optionC",
    "models_hnn_greydanus_maskedpm",
    "models_hnn_hill", "models_hnn_hill_sign", "models_hnn_hill_hinge",
    "models_hnn_hill_hinge2", "models_hnn_hill_hinge3", "models_hnn_hill_hinge4",
    "models_hnn_hill_hinge5_fresh", "models_hnn_hill_hinge6", "models_hnn_hill_hinge7",
    "models_hnn_hill_hinge8_rollout_v2", "models_hnn_hill_rollout_v3",
    "models_temphead_ss", "models_temphead_ss_prefix", "models_logfix",
}
_out_base = os.path.basename(os.path.normpath(args.out))
if _out_base in _PROTECTED:
    raise RuntimeError(f"Refusing to write to protected directory '{args.out}'.")

OUT_DIR     = os.path.join(SRC, args.out)
DATA_PATH   = args.data if os.path.isabs(args.data) else os.path.join(SRC, args.data)
STATUS_PATH  = os.path.join(OUT_DIR, "hnn_hill_train_status.json")
HISTORY_PATH = os.path.join(OUT_DIR, "hnn_hill_training_history.json")
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
F32 = torch.float32
F64 = torch.float64

_USE_PHYSICS_DT = not args.no_physics_dt
_DT_PHYS_CONST  = 1.0 / 20000.0                        # [yr] — matches inference
_N_SUB_EST      = int(np.ceil(0.01 / _DT_PHYS_CONST))  # ~200 for t_sim=10yr, n=1000

print(f"Hill-frame HNN rollout training (Path 3)")
print(f"  Data      : {DATA_PATH}")
print(f"  Out       : {OUT_DIR}")
print(f"  Warmstart : {args.warmstart}")
print(f"  Device    : {device}")
if _USE_PHYSICS_DT:
    print(f"  Mode      : PHYSICS-DT (rollout_v3) — dt_phys={_DT_PHYS_CONST:.2e} yr  "
          f"k={args.k}  n_sub~{_N_SUB_EST}  NO CLAMP")
else:
    print(f"  Mode      : LEGACY (rollout_v2) — n_fine={args.n_fine}  k={args.k}  "
          f"lambda_r={args.lambda_r}  CLAMPED")
print(f"  max_win={args.max_win}  batch={args.batch}  lr={args.lr}  patience_stop={args.patience_stop}")
print()


# ── Dataset: consecutive k+1 step windows ────────────────────────────────────

_POS_COLS = {
    "star":   ["star_x",   "star_y",   "star_z"],
    "planet": ["planet_x", "planet_y", "planet_z"],
    "moon":   ["moon_x",   "moon_y",   "moon_z"],
}
_VEL_COLS = {
    "star":   ["star_vx",   "star_vy",   "star_vz"],
    "planet": ["planet_vx", "planet_vy", "planet_vz"],
    "moon":   ["moon_vx",   "moon_vy",   "moon_vz"],
}


class HnnHillRolloutDataset(Dataset):
    """
    Returns windows of k+1 consecutive rows from the same sim.

    Each item:
      q_ms_seq  : (k+1, 3) float64 — GT star positions
      q_mp_seq  : (k+1, 3) float64 — GT planet positions
      q_mm_seq  : (k+1, 3) float64 — GT moon positions
      v_ms_0    : (3,)     float64 — initial star velocity
      v_mp_0    : (3,)     float64 — initial planet velocity
      v_mm_0    : (3,)     float64 — initial moon velocity
      q_h_norm_0: (3,)     float32 — normalised Hill position at step 0 (for hinge)
      t_hill_0  : (3,)     float32 — hinge target at step 0
      sys_static : (5,)    float32 — [ms_solar, mp_earth, mm_earth, ap_AU, rhill_AU]
      Omega_0   : scalar   float64 — instantaneous angular velocity at step 0
      dt_resamp : scalar   float64 — timestep between resampled parquet rows [yr]
      V_ref     : scalar   float64 — energy scale [Msun AU2/yr2]
    """

    def __init__(self, df: pd.DataFrame, k: int, max_win: int = 10,
                 seed: int = 42) -> None:
        self.k = k
        self.windows: List[Tuple] = []
        rng = np.random.default_rng(seed)

        for _sim_id, grp in df.groupby("sim_id", sort=False):
            grp = grp.reset_index(drop=True)
            n = len(grp)
            if n < k + 1:
                continue

            # Hill targets already computed (by _load_parquet_hill → _compute_hill_targets)
            q_h_norm_all = grp[["q_h_norm_x", "q_h_norm_y", "q_h_norm_z"]].values.astype(np.float32)
            t_hill_all   = grp[["t_hill_x", "t_hill_y", "t_hill_z"]].values.astype(np.float32)
            Omega_all    = grp["Omega_inst"].values.astype(np.float64)

            # Positions and velocities (inertial, F64)
            q_ms_all = grp[_POS_COLS["star"]].values.astype(np.float64)
            q_mp_all = grp[_POS_COLS["planet"]].values.astype(np.float64)
            q_mm_all = grp[_POS_COLS["moon"]].values.astype(np.float64)
            v_ms_all = grp[_VEL_COLS["star"]].values.astype(np.float64)
            v_mp_all = grp[_VEL_COLS["planet"]].values.astype(np.float64)
            v_mm_all = grp[_VEL_COLS["moon"]].values.astype(np.float64)

            # System params (constant per sim — just take row 0)
            row0 = grp.iloc[0]
            ms  = float(row0["ms_solar"])
            mp  = float(row0["mp_earth"])
            mm  = float(row0["mm_earth"])
            ap  = float(row0["ap_AU"])
            rh  = float(row0["rhill_AU"])
            sys_static = np.array([ms, mp, mm, ap, rh], dtype=np.float32)

            mp_msun = mp * _MERTH_OVER_MSUN
            V_ref   = float(FOUR_PI2 * ms * mp_msun / ap)

            # dt_resamp: step in t_years between consecutive resampled rows
            t_col = grp["t_years"].values.astype(np.float64)
            dt_resamp = float((t_col[-1] - t_col[0]) / (n - 1)) if n > 1 else 1e-3

            # Subsample starting indices: pick up to max_win evenly-spaced starts.
            # Consecutive sliding windows are nearly identical — max_win diverse
            # starting positions give comparable gradient signal with 100× fewer
            # forward passes per epoch.
            n_valid = n - k               # number of valid start indices
            if n_valid <= max_win:
                starts = list(range(n_valid))
            else:
                starts = sorted(rng.choice(n_valid, size=max_win, replace=False).tolist())

            for i in starts:
                self.windows.append((
                    q_ms_all[i:i+k+1],    # (k+1, 3)
                    q_mp_all[i:i+k+1],
                    q_mm_all[i:i+k+1],
                    v_ms_all[i],           # (3,)  — step 0 only
                    v_mp_all[i],
                    v_mm_all[i],
                    q_h_norm_all[i],       # (3,)
                    t_hill_all[i],         # (3,)
                    sys_static,            # (5,)
                    Omega_all[i],          # scalar
                    dt_resamp,             # scalar
                    V_ref,                 # scalar
                ))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx):
        (q_ms_seq, q_mp_seq, q_mm_seq,
         v_ms_0, v_mp_0, v_mm_0,
         q_h_norm_0, t_hill_0,
         sys_static, Omega_0, dt_resamp, V_ref) = self.windows[idx]
        return (
            torch.from_numpy(q_ms_seq),      # (k+1, 3) F64
            torch.from_numpy(q_mp_seq),
            torch.from_numpy(q_mm_seq),
            torch.from_numpy(v_ms_0.copy()),  # (3,) F64
            torch.from_numpy(v_mp_0.copy()),
            torch.from_numpy(v_mm_0.copy()),
            torch.from_numpy(q_h_norm_0.copy()),  # (3,) F32
            torch.from_numpy(t_hill_0.copy()),    # (3,) F32
            torch.from_numpy(sys_static.copy()),  # (5,) F32
            torch.tensor(Omega_0, dtype=F64),     # scalar
            torch.tensor(dt_resamp, dtype=F64),   # scalar
            torch.tensor(V_ref, dtype=F64),       # scalar
        )


def _collate(batch):
    """Stack all elements; handle mixed F32/F64."""
    return tuple(torch.stack([b[i] for b in batch]) for i in range(len(batch[0])))


# ── Loss functions ─────────────────────────────────────────────────────────────

_EPS_MAG  = 1e-6
_EPS_SIGN = 1e-3
_LAMBDA_SIGN = 50.0


def _hill_force_loss(grad_pred: Tensor, t_target: Tensor) -> Tensor:
    L_mag  = torch.zeros(1, device=grad_pred.device)
    L_sign = torch.zeros(1, device=grad_pred.device)
    for j in range(3):
        p_j = grad_pred[:, j].abs() + _EPS_MAG
        t_j = t_target[:, j].abs() + _EPS_MAG
        L_mag = L_mag + (torch.log(p_j) - torch.log(t_j)).pow(2).mean()
        norm_agree = (grad_pred[:, j] * t_target[:, j].sign()
                      / (t_target[:, j].abs() + _EPS_SIGN))
        L_sign = L_sign + F.relu(-norm_agree).mean()
    return L_mag / 3.0 + _LAMBDA_SIGN * L_sign / 3.0


# ── Differentiable Hill-frame force (single step, batch of B) ─────────────────

def _hill_forces_diff(
    q_ms: Tensor, q_mp: Tensor, q_mm: Tensor,
    v_ms: Tensor, v_mp: Tensor,
    ms_b: Tensor, mp_msun_b: Tensor, mm_msun_b: Tensor,
    ap_b: Tensor, rhill_b: Tensor, V_ref_b: Tensor,
    model: HNN_Hill,
    scaler_mean: Tensor, scaler_std: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Differentiable analogue of _hill_forces() in hnn_inference_hill.py.

    All tensors are (B, 3) F64 except scalar params (B,) F64.
    Uses create_graph=True so gradients flow back through all rollout steps.
    Matches inference code exactly: same Hill frame, same centrifugal strip,
    same inertial rotation.

    Returns (a_s, a_p, a_m) each (B, 3) F64.
    """
    B = q_ms.shape[0]
    dev = q_ms.device

    r_sp   = q_mp - q_ms                                         # (B, 3)
    v_sp   = v_mp - v_ms

    r_sp_sq = (r_sp * r_sp).sum(1).clamp(min=1e-20)             # (B,)
    r_sp_n  = r_sp_sq.sqrt().unsqueeze(1)                        # (B, 1)
    x_hat   = r_sp / r_sp_n                                      # (B, 3) radial unit

    L_sp  = torch.linalg.cross(r_sp, v_sp)                      # (B, 3)
    L_n2  = (L_sp * L_sp).sum(1).clamp(min=1e-20)
    L_n   = L_n2.sqrt().unsqueeze(1)                             # (B, 1)
    z_hat = L_sp / L_n                                           # (B, 3) normal
    Omega = L_n.squeeze(1) / r_sp_sq                            # (B,) [rad/yr]
    y_hat = torch.linalg.cross(z_hat, x_hat)                    # (B, 3) tangential

    # Moon position in Hill frame
    r_mp_vec = q_mm - q_mp                                       # (B, 3)
    q_h_x = (r_mp_vec * x_hat).sum(1)                           # (B,) [AU]
    q_h_y = (r_mp_vec * y_hat).sum(1)
    q_h_z = (r_mp_vec * z_hat).sum(1)
    q_h_AU = torch.stack([q_h_x, q_h_y, q_h_z], dim=1)         # (B, 3)
    q_h_n  = q_h_AU / rhill_b.unsqueeze(1)                      # (B, 3) normalised

    # sys_enc: log-standardise [ms_solar, mp_earth, mm_earth, ap_AU, rhill_AU, Omega]
    # ms_b, ap_b are in solar/AU units; mp/mm are in M_earth; rhill in AU
    # We need raw [ms_solar, mp_earth, mm_earth, ap_AU, rhill_AU] + Omega
    # Note: mp_msun_b = mp_earth * MERTH_OVER_MSUN so recover mp_earth = mp_msun_b / MERTH
    _MERTH = torch.tensor(_MERTH_OVER_MSUN, dtype=F64, device=dev)
    mp_earth_b = mp_msun_b / _MERTH                              # (B,) M_earth
    mm_earth_b = mm_msun_b / _MERTH                              # (B,) M_earth

    sys_raw = torch.stack([ms_b, mp_earth_b, mm_earth_b, ap_b, rhill_b, Omega], dim=1).to(F32)  # (B, 6)
    sys_log = sys_raw.clamp(min=1e-20).log()
    sys_enc = (sys_log - scaler_mean) / scaler_std               # (B, 6) F32

    # HNN forward + autograd (F32 for model)
    q_h_f32 = q_h_n.to(F32).detach().requires_grad_(True)       # (B, 3)
    with torch.enable_grad():
        V_theta = model(q_h_f32, sys_enc)                        # (B,)
        grad = torch.autograd.grad(
            V_theta.sum(), q_h_f32, create_graph=True
        )[0]                                                      # (B, 3)

    # Conservative Hill accel: a_cons_h = -grad * V_ref / (mm_msun * rhill)
    a_cons_h = (-grad.to(F64) * V_ref_b.unsqueeze(1)
                / (mm_msun_b.unsqueeze(1) * rhill_b.unsqueeze(1)))  # (B, 3) [AU/yr²]

    # Strip centrifugal: cf = Omega^2 * (q_h_x, q_h_y, 0)
    omega2    = (Omega ** 2).unsqueeze(1)                        # (B, 1)
    q_h_plane = torch.cat([q_h_AU[:, :2],
                            torch.zeros(B, 1, dtype=F64, device=dev)], dim=1)
    cf        = omega2 * q_h_plane                               # (B, 3)
    a_hill    = a_cons_h - cf                                    # (B, 3) Hill frame

    # Rotate to inertial
    a_rel_in = (a_hill[:, 0:1] * x_hat +
                a_hill[:, 1:2] * y_hat +
                a_hill[:, 2:3] * z_hat)                          # (B, 3)

    # Planet accel from star
    r_sp_cu = r_sp_sq.unsqueeze(1) ** 1.5                       # (B, 1)
    a_p     = -FOUR_PI2 * ms_b.unsqueeze(1) * r_sp / r_sp_cu   # (B, 3)

    # Moon inertial accel
    a_m = a_rel_in + a_p                                         # (B, 3)

    # Star accel: from moon + from planet
    r_sm    = q_mm - q_ms                                        # (B, 3)
    d_sm_cu = ((r_sm * r_sm).sum(1, keepdim=True)).clamp(min=1e-20) ** 1.5
    a_s     = (FOUR_PI2 * mm_msun_b.unsqueeze(1) / d_sm_cu * r_sm
               + FOUR_PI2 * mp_msun_b.unsqueeze(1) * r_sp / r_sp_cu)  # (B, 3)

    return a_s, a_p, a_m


def _kdk_step(
    q_ms, q_mp, q_mm, v_ms, v_mp, v_mm,
    ms_b, mp_msun_b, mm_msun_b, ap_b, rhill_b, V_ref_b, dt_b,
    model, scaler_mean, scaler_std,
):
    """One KDK leapfrog step (differentiable). Returns updated (q, v) tuples."""
    half_dt = (dt_b / 2.0).unsqueeze(1)   # (B, 1)
    full_dt = dt_b.unsqueeze(1)

    # Initial forces at current positions
    a_s, a_p, a_m = _hill_forces_diff(
        q_ms, q_mp, q_mm, v_ms, v_mp,
        ms_b, mp_msun_b, mm_msun_b, ap_b, rhill_b, V_ref_b,
        model, scaler_mean, scaler_std,
    )

    # Half-kick velocities
    v_ms = v_ms + a_s * half_dt
    v_mp = v_mp + a_p * half_dt
    v_mm = v_mm + a_m * half_dt

    # Full drift positions
    q_ms = q_ms + v_ms * full_dt
    q_mp = q_mp + v_mp * full_dt
    q_mm = q_mm + v_mm * full_dt

    # Forces at new positions
    a_s, a_p, a_m = _hill_forces_diff(
        q_ms, q_mp, q_mm, v_ms, v_mp,
        ms_b, mp_msun_b, mm_msun_b, ap_b, rhill_b, V_ref_b,
        model, scaler_mean, scaler_std,
    )

    # Half-kick velocities again
    v_ms = v_ms + a_s * half_dt
    v_mp = v_mp + a_p * half_dt
    v_mm = v_mm + a_m * half_dt

    return q_ms, q_mp, q_mm, v_ms, v_mp, v_mm


# ── Fast data loading (bypasses slow groupby.apply in _load_parquet_hill) ─────
#
# _load_parquet_hill uses groupby.apply(_trim) which is O(n_groups) Python
# calls and hangs for minutes on a cold OS page cache.  We replicate its
# logic here with a vectorized cumsum trim that is fast regardless of cache.

def _fast_load(path: str, val_frac: float, seed: int):
    """
    Fast replacement for _load_parquet_hill.

    The standard function uses groupby.apply(_trim) which makes one Python
    function call per sim group (~3000 calls on a cold page cache = minutes).
    This version uses pandas groupby.cumsum() — a vectorised Cython path that
    completes in seconds regardless of cache state.

    Produces identical output to _load_parquet_hill.
    """
    df = pd.read_parquet(path)

    # Step 1: keep sims whose FIRST row is stable=1 AND habitable=1
    first = df.groupby("sim_id")[["stable", "habitable"]].first()
    valid_sims = set(first.index[(first["stable"] == 1) & (first["habitable"] == 1)])
    df = df[df["sim_id"].isin(valid_sims)].copy()

    # Step 2: vectorised per-sim trim at first step where stable=0 AND habitable=0.
    # cumsum_bad[i] = 0  ⟺  no bad row has appeared yet in this sim → keep.
    # cumsum_bad[i] ≥ 1  ⟺  first-or-later bad row → drop.
    df = df.sort_values(["sim_id", "t_years"]).reset_index(drop=True)
    bad = ((df["stable"] == 0) & (df["habitable"] == 0)).astype(np.int8)
    cumsum_bad = bad.groupby(df["sim_id"]).cumsum()   # pandas Cython path — O(N)
    df = df[cumsum_bad == 0].reset_index(drop=True)

    # Step 3: Hill-frame targets (vectorised, same as _load_parquet_hill)
    q_h_norm, t_hill, Omega_inst, _valid = _compute_hill_targets(df)
    df["q_h_norm_x"] = q_h_norm[:, 0].astype(np.float32)
    df["q_h_norm_y"] = q_h_norm[:, 1].astype(np.float32)
    df["q_h_norm_z"] = q_h_norm[:, 2].astype(np.float32)
    df["t_hill_x"]   = t_hill[:, 0].astype(np.float32)
    df["t_hill_y"]   = t_hill[:, 1].astype(np.float32)
    df["t_hill_z"]   = t_hill[:, 2].astype(np.float32)
    df["Omega_inst"] = Omega_inst

    # Step 4: split by sim_id (no row-level leakage)
    all_sims = np.array(sorted(df["sim_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(all_sims)
    n_val = max(1, int(len(all_sims) * val_frac))
    val_sims   = set(all_sims[:n_val])
    train_sims = set(all_sims[n_val:])
    train_df = df[df["sim_id"].isin(train_sims)].reset_index(drop=True)
    val_df   = df[df["sim_id"].isin(val_sims)].reset_index(drop=True)
    return train_df, val_df


# ── Data loading ───────────────────────────────────────────────────────────────

print("Loading and preprocessing data …")
t0 = time.perf_counter()

train_df, val_df = _fast_load(DATA_PATH, args.val_frac, args.seed)

# Fit scaler on training rows
from sklearn.preprocessing import StandardScaler
sys_raw_train = np.column_stack([
    train_df["ms_solar"].values,
    train_df["mp_earth"].values,
    train_df["mm_earth"].values,
    train_df["ap_AU"].values,
    train_df["rhill_AU"].values,
    train_df["Omega_inst"].values,
]).astype(np.float64)
sys_scaler = StandardScaler()
sys_scaler.fit(np.log(np.maximum(sys_raw_train, 1e-20)))

save_hill_sys_scaler(sys_scaler, OUT_DIR)

train_ds = HnnHillRolloutDataset(train_df, k=args.k, max_win=args.max_win, seed=args.seed)
val_ds   = HnnHillRolloutDataset(val_df,   k=args.k, max_win=args.max_win, seed=args.seed + 1)

print(f"  Train: {len(train_ds):,} windows  Val: {len(val_ds):,} windows  "
      f"({time.perf_counter() - t0:.1f}s)")

train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          collate_fn=_collate, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                          collate_fn=_collate, num_workers=0)

# Precompute scaler as device tensors
scaler_mean = torch.tensor(sys_scaler.mean_,  dtype=F32, device=device)  # (6,)
scaler_std  = torch.tensor(sys_scaler.scale_, dtype=F32, device=device)  # (6,)


# ── Model ─────────────────────────────────────────────────────────────────────

model = HNN_Hill(hidden=args.hidden, layers=args.layers).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel: HNN_Hill  params={n_params:,}")

if args.warmstart:
    ws_dir   = args.warmstart if os.path.isabs(args.warmstart) else os.path.join(SRC, args.warmstart)
    ws_ckpt  = os.path.join(ws_dir, "hnn_hill_model.pt")
    ws_state = torch.load(ws_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ws_state)
    print(f"  Warm-start weights loaded from {ws_ckpt}")
else:
    print("  WARNING: no --warmstart specified; training from random init is unstable.")
print()

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=args.patience
)


# ── Training loop ──────────────────────────────────────────────────────────────

best_val_loss = float("inf")
no_improve_count = 0
history = {
    "train_loss": [], "val_loss": [],
    "train_hinge": [], "train_rollout": [],
    "epochs": args.epochs,
    "hyperparams": vars(args),
}
t_start = time.perf_counter()

for epoch in range(1, args.epochs + 1):
    model.train()
    tr_losses, tr_hinge, tr_rollout, tr_clamped = [], [], [], []

    for batch in train_loader:
        (q_ms_seq, q_mp_seq, q_mm_seq,
         v_ms_0, v_mp_0, v_mm_0,
         q_h_norm_0, t_hill_0,
         sys_static, Omega_0, dt_resamp, V_ref) = [x.to(device) for x in batch]

        # sys_static: (B, 5) -> ms, mp_earth, mm_earth, ap_AU, rhill_AU
        ms_b     = sys_static[:, 0].to(F64)   # (B,) solar masses
        mp_b     = sys_static[:, 1].to(F64)   # (B,) M_earth
        mm_b     = sys_static[:, 2].to(F64)   # (B,) M_earth
        ap_b     = sys_static[:, 3].to(F64)   # (B,) AU
        rhill_b  = sys_static[:, 4].to(F64)   # (B,) AU
        mp_msun_b = mp_b * _MERTH_OVER_MSUN   # (B,) M_sun
        mm_msun_b = mm_b * _MERTH_OVER_MSUN   # (B,) M_sun
        V_ref_b  = V_ref.to(F64)              # (B,)
        dt_b     = dt_resamp.to(F64)          # (B,)

        # ── Hinge loss at step 0 (force supervision on GT position) ─────────
        q_leaf = q_h_norm_0.detach().requires_grad_(True)   # (B, 3) F32

        # Build sys_enc at step 0 using stored Omega_0
        sys_raw_0 = torch.stack([
            sys_static[:, 0],                    # ms_solar (F32)
            sys_static[:, 1],                    # mp_earth
            sys_static[:, 2],                    # mm_earth
            sys_static[:, 3],                    # ap_AU
            sys_static[:, 4],                    # rhill_AU
            Omega_0.to(F32),                     # Omega_inst
        ], dim=1)                                # (B, 6) F32
        sys_log_0 = sys_raw_0.clamp(min=1e-20).log()
        sys_enc_0 = (sys_log_0 - scaler_mean) / scaler_std   # (B, 6)

        with torch.enable_grad():
            V0   = model(q_leaf, sys_enc_0)
            grad0 = torch.autograd.grad(
                V0.sum(), q_leaf, create_graph=True
            )[0]                                # (B, 3)

        L_hinge = _hill_force_loss(grad0, t_hill_0.to(F32))

        # ── Rollout loss ──────────────────────────────────────────────────────
        # Physics-dt mode (rollout_v3, default):
        #   dt_sub = 1/20000 yr (physics dt, matches inference).
        #   n_sub  = ceil(dt_resamp / dt_sub) ≈ 200 sub-steps per parquet interval.
        #   No loss clamp — physics dt is leapfrog-stable for ALL cells including
        #   K-1229b inner moons (ωΔt ≈ 0.43 << 2), so no divergence occurs and
        #   every cell receives full gradient signal.
        #
        # Legacy mode (--no_physics_dt, rollout_v2 behaviour):
        #   dt_sub = dt_resamp / n_fine.  Inner moons diverge → clamped to zero grad.
        if _USE_PHYSICS_DT:
            dt_fine_b  = torch.full_like(dt_b, _DT_PHYS_CONST)          # (B,) physics dt
            n_sub_steps = max(int(np.ceil(float(dt_b[0].item()) / _DT_PHYS_CONST)), 1)
        else:
            dt_fine_b   = dt_b / args.n_fine                             # (B,) legacy dt
            n_sub_steps = args.n_fine

        q_ms = q_ms_seq[:, 0].to(F64)           # (B, 3)
        q_mp = q_mp_seq[:, 0].to(F64)
        q_mm = q_mm_seq[:, 0].to(F64)
        v_ms = v_ms_0.to(F64)
        v_mp = v_mp_0.to(F64)
        v_mm = v_mm_0.to(F64)

        L_rollout = torch.zeros(1, device=device, dtype=F64)
        n_clamped_total = 0
        for t in range(1, args.k + 1):
            # n_sub_steps KDK sub-steps to advance from parquet row t-1 → t
            for _ in range(n_sub_steps):
                q_ms, q_mp, q_mm, v_ms, v_mp, v_mm = _kdk_step(
                    q_ms, q_mp, q_mm, v_ms, v_mp, v_mm,
                    ms_b, mp_msun_b, mm_msun_b, ap_b, rhill_b, V_ref_b, dt_fine_b,
                    model, scaler_mean, scaler_std,
                )
            # GT moon position at parquet row t
            q_mm_gt = q_mm_seq[:, t].to(F64)   # (B, 3)
            diff = (q_mm - q_mm_gt) / rhill_b.unsqueeze(1)   # normalised by rhill
            per_elem = (diff * diff).sum(1)                   # (B,) squared L2 error
            if not _USE_PHYSICS_DT:
                # Legacy clamp: zero gradient on cells that violate leapfrog stability
                n_clamped_total += int((per_elem.detach() > 4.0).sum().item())
                per_elem = per_elem.clamp(max=4.0)
            L_rollout = L_rollout + per_elem.mean()

        L_rollout = L_rollout / args.k
        L_total   = L_hinge + args.lambda_r * L_rollout.to(F32)

        optimizer.zero_grad()
        L_total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        tr_losses.append(L_total.item())
        tr_hinge.append(L_hinge.item())
        tr_rollout.append(L_rollout.item())
        tr_clamped.append(n_clamped_total / (args.batch * args.k))

    # ── Validation ────────────────────────────────────────────────────────────
    model.eval()
    val_losses = []

    for batch in val_loader:
        (q_ms_seq, q_mp_seq, q_mm_seq,
         v_ms_0, v_mp_0, v_mm_0,
         q_h_norm_0, t_hill_0,
         sys_static, Omega_0, dt_resamp, V_ref) = [x.to(device) for x in batch]

        sys_raw_0 = torch.stack([
            sys_static[:, 0], sys_static[:, 1], sys_static[:, 2],
            sys_static[:, 3], sys_static[:, 4], Omega_0.to(F32),
        ], dim=1)
        sys_log_0 = sys_raw_0.clamp(min=1e-20).log()
        sys_enc_0 = (sys_log_0 - scaler_mean) / scaler_std

        q_leaf = q_h_norm_0.detach().requires_grad_(True)
        with torch.enable_grad():
            V0    = model(q_leaf, sys_enc_0)
            grad0 = torch.autograd.grad(V0.sum(), q_leaf, create_graph=False)[0]

        L_hinge = _hill_force_loss(grad0, t_hill_0.to(F32))
        val_losses.append(L_hinge.item())   # val only tracks hinge (no rollout BPTT)

    train_loss  = float(np.mean(tr_losses))
    val_loss    = float(np.mean(val_losses))
    h_loss      = float(np.mean(tr_hinge))
    r_loss      = float(np.mean(tr_rollout))
    clamp_frac  = float(np.mean(tr_clamped))
    lr_now      = optimizer.param_groups[0]["lr"]
    elapsed     = time.perf_counter() - t_start

    print(f"Epoch {epoch:3d}/{args.epochs}  "
          f"train={train_loss:.5f}  val={val_loss:.5f}  "
          f"hinge={h_loss:.5f}  rollout={r_loss:.5f}  "
          f"clamp={clamp_frac:.2f}  lr={lr_now:.2e}  {elapsed:.0f}s")

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_hinge"].append(h_loss)
    history["train_rollout"].append(r_loss)
    history.setdefault("clamp_frac", []).append(clamp_frac)

    # Save best checkpoint by val hinge loss; track early stopping
    improved = val_loss < best_val_loss
    if improved:
        best_val_loss = val_loss
        model.save(OUT_DIR)
        print(f"  [saved] best model  val={val_loss:.5f}")
        no_improve_count = 0
    else:
        no_improve_count += 1

    scheduler.step(val_loss)

    if no_improve_count >= args.patience_stop:
        print(f"  Early stop: val_loss did not improve for {args.patience_stop} epochs.")
        break

    # Live status
    status = {
        "status":        "running",
        "epoch":         epoch,
        "total_epochs":  args.epochs,
        "train_loss":    train_loss,
        "val_loss":      val_loss,
        "hinge_loss":    h_loss,
        "rollout_loss":  r_loss,
        "elapsed_s":     elapsed,
    }
    with open(STATUS_PATH, "w") as f:
        json.dump(status, f, indent=2)

# ── Final ─────────────────────────────────────────────────────────────────────

status["status"] = "complete"
with open(STATUS_PATH, "w") as f:
    json.dump(status, f, indent=2)

with open(HISTORY_PATH, "w") as f:
    json.dump(history, f, indent=2)

print(f"\nDone. Best val_loss={best_val_loss:.5f}")
print(f"Artifacts: {OUT_DIR}")
