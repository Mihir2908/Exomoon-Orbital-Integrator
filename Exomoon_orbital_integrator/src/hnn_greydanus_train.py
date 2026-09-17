"""
hnn_greydanus_train.py — Option E (v3): Greydanus-style joint H_θ(d_n, p_n, sys_enc).

Three simultaneous supervision signals:
  1. Force supervision:    mixed loss on ∂H_θ/∂d_n  (see below)
  2. Momentum supervision: MSRE( ∂H_θ/∂p_n_i, v_n_i )    — Hamilton: ∂H/∂p = v
  3. Value supervision:    MSE(  H_θ,  H_Newton_n )        — Option C (lambda_h > 0)

Option C (value supervision) pins the absolute value of H_θ to the Newtonian
Hamiltonian, fixing the wrong-integration-constant problem that causes the moon
to appear unbound under H_θ even when forces and velocities are correct:

  H_Newton_n = T_Newton_n + V_Newton_n  (both in H_ref = G·ms·mp/ap units)

  T_Newton_n = 0.5 × (p_n · v_n).sum()     — derived from p_n=(m_i/mp)·v_n
  V_Newton_n = -(1/d_sp_n + (mm/mp)/d_sm_n + (mm/ms)·(ap/rhill)/d_pm_n)

  T_Newton_n computation uses p_n·v_n dot product directly — no mass-ratio
  tensors needed, because (m_i/mp)·v_n_i × v_n_i = (m_i/mp)·v_n_i² and
  sum over all 9 components gives T/H_ref exactly.

  V_Newton_n requires raw sys_params (ms, mp, mm, ap, rhill) — stored in
  sys_raw column added to GreydanusDataset in this version.

Force loss — mixed per-component loss with masked planet-moon term:
  t_sp, t_sm: MSRE with eps=0.01, computed on ALL rows.
  t_pm:       log-MSE on bound rows only (stable=1, d_pm_n ≤ 1.0).

  Why log-MSE for t_pm instead of MSRE:
    MSRE = ((pred−target)/target)² — asymmetric in multiplicative error:
      10× over-prediction → 81,  10× under-prediction → 0.81.
    Under-prediction of the planet-moon binding force is penalised 100× less
    than over-prediction.  The model drifts toward systematically under-
    predicting t_pm → moon appears unbound → HNN=ESCAPED at inference.
    Log-MSE = (log pred − log target)² — fully symmetric:
      10× over or under → (log 10)² ≈ 5.3 each.
    clamp(min=1e-8) rather than abs() ensures wrong-sign predictions are
    severely penalised (log(1e-8)−log(targ) ≈ −11 for targ≈0.001 → loss≈132).

  Escaped-habitable rows (stable=0 but habitable=1) still contribute to
  t_sp and t_sm supervision — only t_pm weight is zeroed for those rows.
  The stable column is read directly from the Parquet — no extra computation.

INPUT: p_n_i = (m_i/mp)·v_n_i  — dimensionless canonical momentum (not velocity).
  Star:    p_n_star   = (ms/mp) · v_n_star   — O(1) even though v_star is tiny
  Planet:  p_n_planet = 1.0    · v_n_planet  — same as v_n_planet
  Moon:    p_n_moon   = (mm/mp)· v_n_moon    — small

TARGET for momentum supervision: v_n_i = v_i/v_circ — the dimensionless velocity.
  Hamilton's equation ∂H/∂p = dq/dt = v — target is velocity, input is momentum.

Data loading and training filter: identical to hnn_train.py.
  - Exclude sims where t=0 is not both stable AND habitable.
  - Trim each trajectory at the first row where stable=0 AND habitable=0.
Extra columns: _VEL_COLS (star/planet/moon vx/vy/vz in AU/yr).

Usage (log-MSE t_pm, no value supervision):
    py hnn_greydanus_train.py \\
        --data ml_dataset.parquet \\
        --out  models_hnn_greydanus_logpm/ \\
        --epochs 50 --batch 512 --lr 1e-3 --hidden 128 --layers 2

Usage (with Option C value supervision):
    py hnn_greydanus_train.py \\
        --data ml_dataset.parquet \\
        --out  models_hnn_greydanus_logpm_optionC/ \\
        --epochs 50 --batch 512 --lr 1e-3 --hidden 128 --layers 2 \\
        --lambda_vel 1.0 --lambda_h 0.1

Output (never touches models/, models_temphead/, models_hnn_dist_v12/, eval_aux_mlp_output/):
    hnn_model.pt              — best weights by val loss
    hnn_config.json           — architecture config for HNNGreydanus.load()
    hnn_sys_scaler.pkl        — StandardScaler for sys_params
    hnn_train_status.json     — live progress updated each epoch
    hnn_training_history.json — final loss curves

ISOLATION: no imports from dataset.py, model.py, train.py, inference.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from exomoon.ml.hnn_greydanus_model import HNNGreydanus
from exomoon.ml.hnn_dataset         import save_sys_scaler, _MERTH_OVER_MSUN

_FOUR_PI2 = 4.0 * np.pi ** 2

_SYS_COLS = ["ms_solar", "mp_earth", "mm_earth", "ap_AU", "rhill_AU"]
_Q_COLS   = [
    "star_x",   "star_y",   "star_z",
    "planet_x", "planet_y", "planet_z",
    "moon_x",   "moon_y",   "moon_z",
]
_VEL_COLS = [
    "star_vx",   "star_vy",   "star_vz",
    "planet_vx", "planet_vy", "planet_vz",
    "moon_vx",   "moon_vy",   "moon_vz",
]
T_MAX    = 1000.0   # clip ceiling for t_pm near Roche limit (same as hnn_dataset.py)
_EPS     = 0.01     # MSRE eps for t_sp and t_sm (appropriate for O(1) targets)
_LOG_EPS = 1e-8     # log-MSE clamp for t_pm — keeps log in valid domain; t_pm_min≈2.5e-5 >> 1e-8


# ── Dataset ───────────────────────────────────────────────────────────────────

class GreydanusDataset(Dataset):
    """Per-timestep dataset with distances, momenta, velocities, sys params, force targets."""

    def __init__(
        self,
        d_n:     np.ndarray,   # (N, 3)  physics-normalised distances
        p_n:     np.ndarray,   # (N, 9)  (m_i/mp)*v_n — normalised momenta (model input)
        v_n:     np.ndarray,   # (N, 9)  v_circ-normalised velocities (momentum supervision target)
        sys_enc: np.ndarray,   # (N, 5)  StandardScaler sys params (model input)
        t_grad:  np.ndarray,   # (N, 3)  dimensionless force targets
        sys_raw: np.ndarray,   # (N, 5)  raw [ms_solar, mp_earth, mm_earth, ap_AU, rhill_AU]
        stable:  np.ndarray,   # (N,)    1 if d_pm_n ≤ 1.0 (moon within Hill sphere)
    ) -> None:
        self.d_n     = torch.from_numpy(d_n.astype(np.float32))
        self.p_n     = torch.from_numpy(p_n.astype(np.float32))
        self.v_n     = torch.from_numpy(v_n.astype(np.float32))
        self.sys_enc = torch.from_numpy(sys_enc.astype(np.float32))
        self.t_grad  = torch.from_numpy(t_grad.astype(np.float32))
        self.sys_raw = torch.from_numpy(sys_raw.astype(np.float32))
        self.stable  = torch.from_numpy(stable.astype(np.bool_))

    def __len__(self) -> int:
        return len(self.d_n)

    def __getitem__(self, idx):
        return (
            self.d_n[idx],
            self.p_n[idx],
            self.v_n[idx],
            self.sys_enc[idx],
            self.t_grad[idx],
            self.sys_raw[idx],
            self.stable[idx],
        )


def _load_and_split(
    path: str,
    val_frac: float = 0.20,
    seed: int = 42,
) -> tuple:
    """
    Load Parquet, apply stable+habitable filter, compute all training arrays.

    Returns (train_ds, val_ds, sys_scaler).
    """
    load_cols = (["sim_id"] + _Q_COLS + _VEL_COLS + _SYS_COLS
                 + ["stable", "habitable"])
    print(f"  Loading {path} ...")
    df = pd.read_parquet(path, columns=load_cols)
    n_raw = len(df)

    # 1. Exclude sims that do not start stable+habitable
    sim_first  = df.groupby("sim_id")[["stable", "habitable"]].first()
    valid_sims = sim_first[
        (sim_first["stable"] == 1) & (sim_first["habitable"] == 1)
    ].index
    n_excl = df["sim_id"].nunique() - len(valid_sims)
    df = df[df["sim_id"].isin(valid_sims)].copy()

    # 2. Trim each sim at first row where stable=0 AND habitable=0 simultaneously.
    #    Rows where stable=0 but habitable=1 (escaped-habitable) are kept — they
    #    still contribute to t_sp and t_sm supervision. The stable column is
    #    retained here so the training loop can mask t_pm loss to bound rows only.
    df["_bad"] = ((df["stable"] == 0) & (df["habitable"] == 0)).astype(np.int8)
    df["_cum"] = df.groupby("sim_id")["_bad"].cumsum()
    df = df[df["_cum"] == 0].drop(columns=["_bad", "_cum", "habitable"])

    n_kept = len(df)
    print(f"  {n_raw:,} total rows  |  {n_excl} sims excluded (non-stable+habitable start)")
    print(f"  After filter: {n_kept:,} rows kept ({100*n_kept/n_raw:.1f}%)")

    # ── Stable mask (d_pm_n ≤ 1) ──────────────────────────────────────────────
    stable = df["stable"].to_numpy(dtype=np.int8)
    frac_stable = stable.mean()
    print(f"  Bound (stable=1) rows: {frac_stable*100:.1f}%  "
          f"({stable.sum():,} / {n_kept:,})  — t_pm loss applied here only")

    # ── Extract arrays ─────────────────────────────────────────────────────────
    ms_solar = df["ms_solar"].to_numpy(dtype=np.float64)
    mp_earth = df["mp_earth"].to_numpy(dtype=np.float64)
    mm_earth = df["mm_earth"].to_numpy(dtype=np.float64)
    ap_AU    = df["ap_AU"].to_numpy(dtype=np.float64)
    rhill_AU = df["rhill_AU"].to_numpy(dtype=np.float64)

    ms = ms_solar
    mp = mp_earth * _MERTH_OVER_MSUN   # solar masses
    mm = mm_earth * _MERTH_OVER_MSUN   # solar masses

    # ── Pair distances (same as hnn_dataset.py) ─────────────────────────────────
    q = df[_Q_COLS].to_numpy(dtype=np.float64)   # (N, 9) AU
    r_s = q[:, 0:3]; r_p = q[:, 3:6]; r_m = q[:, 6:9]

    d_sp = np.linalg.norm(r_p - r_s, axis=1)
    d_sm = np.linalg.norm(r_m - r_s, axis=1)
    d_pm = np.linalg.norm(r_m - r_p, axis=1)

    d_sp_n = d_sp / ap_AU
    d_sm_n = d_sm / ap_AU
    d_pm_n = d_pm / rhill_AU

    print(f"  d_sp_n: {d_sp_n.min():.4f}–{d_sp_n.max():.4f}  "
          f"d_sm_n: {d_sm_n.min():.4f}–{d_sm_n.max():.4f}  "
          f"d_pm_n: {d_pm_n.min():.4f}–{d_pm_n.max():.4f}")

    # ── Dimensionless force targets (same as hnn_dataset.py) ──────────────────
    t_sp = 1.0 / d_sp_n ** 2
    t_sm = (mm / mp) / d_sm_n ** 2
    t_pm = (mm / ms) * (ap_AU / rhill_AU) / d_pm_n ** 2
    t_grads = np.clip(
        np.stack([t_sp, t_sm, t_pm], axis=1), 0.0, T_MAX,
    )   # (N, 3)

    # ── Velocity normalisation: v_n = v / v_circ ──────────────────────────────
    v_circ = np.sqrt(_FOUR_PI2 * ms_solar / ap_AU)   # (N,) AU/yr, planet orbital speed

    v_raw = df[_VEL_COLS].to_numpy(dtype=np.float64)   # (N, 9) AU/yr
    v_n   = v_raw / v_circ[:, None]                     # (N, 9) dimensionless

    print(f"  v_circ: {v_circ.min():.3f}–{v_circ.max():.3f} AU/yr")

    # ── Mass ratios: (m_i/mp) per body per row ────────────────────────────────
    mr_s  = (ms / mp)[:, None] * np.ones((1, 3))   # (N, 3)
    mr_p  = np.ones((len(df), 3))                   # (N, 3) — planet is reference
    mr_m  = (mm / mp)[:, None] * np.ones((1, 3))   # (N, 3)
    mass_ratio = np.concatenate([mr_s, mr_p, mr_m], axis=1)   # (N, 9)

    # p_n = (m_i/mp) * v_n — canonical normalised momentum (model input)
    # v_n stays as-is — it is the velocity supervision target (∂H/∂p_n = v_n)
    p_n = mass_ratio * v_n   # (N, 9)

    print(f"  v_n range: {v_n.min():.3f}–{v_n.max():.3f}  "
          f"p_n range: {p_n.min():.3f}–{p_n.max():.3f}")

    d_n     = np.stack([d_sp_n, d_sm_n, d_pm_n], axis=1)   # (N, 3)
    sys_raw = df[_SYS_COLS].to_numpy(dtype=np.float64)       # (N, 5)
    sim_ids = df["sim_id"].to_numpy()

    # ── Train/val split by sim_id ─────────────────────────────────────────────
    unique_sims = np.unique(sim_ids)
    rng         = np.random.default_rng(seed)
    rng.shuffle(unique_sims)

    n_val      = max(1, int(len(unique_sims) * val_frac))
    val_set    = set(unique_sims[:n_val].tolist())
    train_mask = np.array([s not in val_set for s in sim_ids])
    val_mask   = ~train_mask

    print(f"  Train rows: {train_mask.sum():,}  |  Val rows: {val_mask.sum():,}  (split by sim_id)")

    sys_scaler = StandardScaler().fit(sys_raw[train_mask])

    def _make(mask: np.ndarray) -> GreydanusDataset:
        return GreydanusDataset(
            d_n     = d_n[mask],
            p_n     = p_n[mask],
            v_n     = v_n[mask],
            sys_enc = sys_scaler.transform(sys_raw[mask]),
            t_grad  = t_grads[mask],
            sys_raw = sys_raw[mask],
            stable  = stable[mask],
        )

    return _make(train_mask), _make(val_mask), sys_scaler


# ── Losses & Newtonian Hamiltonian ────────────────────────────────────────────

def msre_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """Mean squared relative error: ((pred - target) / (|target| + eps))²."""
    return ((pred - target) / (target.abs() + eps)).pow(2).mean()


def _force_loss(
    dH_dd:  torch.Tensor,   # (batch, 3)  predicted ∂H/∂d_n
    t_grad: torch.Tensor,   # (batch, 3)  force targets [t_sp, t_sm, t_pm]
    stable: torch.Tensor,   # (batch,)    bool — True where d_pm_n ≤ 1
) -> torch.Tensor:
    """Mixed per-component force loss with log-MSE masked planet-moon term.

    t_sp, t_sm: MSRE with eps=0.01 on ALL rows.
    t_pm:       log-MSE on bound rows only (stable=True).

    Why log-MSE for t_pm: MSRE penalises 10× over-prediction 100× more than
    10× under-prediction, causing the model to systematically under-predict the
    planet-moon binding force → moon escapes at inference.  Log-MSE is
    symmetric in log-scale (10× over = 10× under = (log10)² ≈ 5.3).

    clamp(min=_LOG_EPS) on the raw prediction (not abs) means a wrong-sign
    prediction (dH_dd[:,2] < 0) is mapped to ~1e-8 while target ≈ 0.001,
    yielding loss ≈ (log(1e-8)−log(0.001))² ≈ 132 — a large, corrective signal.
    """
    dev  = dH_dd.device
    pm_w = stable.float().to(dev)   # (batch,)

    # t_sp and t_sm — MSRE on all rows
    L_sp = ((dH_dd[:, 0] - t_grad[:, 0]) / (t_grad[:, 0].abs() + _EPS)).pow(2).mean()
    L_sm = ((dH_dd[:, 1] - t_grad[:, 1]) / (t_grad[:, 1].abs() + _EPS)).pow(2).mean()

    # t_pm — log-MSE on bound rows only
    pred_pm = dH_dd[:, 2].clamp(min=_LOG_EPS)               # wrong sign → large penalty
    targ_pm = t_grad[:, 2].clamp(min=_LOG_EPS)              # always > 0 for bound rows
    sq_pm   = (torch.log(pred_pm) - torch.log(targ_pm)).pow(2)
    L_pm    = (sq_pm * pm_w).sum() / pm_w.sum().clamp(min=1.0)

    return (L_sp + L_sm + L_pm) / 3.0


def _compute_h_newton_n(
    d_n:     torch.Tensor,   # (batch, 3)  [d_sp_n, d_sm_n, d_pm_n]
    p_n:     torch.Tensor,   # (batch, 9)  canonical momenta (m_i/mp)*v_n
    v_n:     torch.Tensor,   # (batch, 9)  velocities v_n = v/v_circ (supervision targets)
    sys_raw: torch.Tensor,   # (batch, 5)  raw [ms_solar, mp_earth, mm_earth, ap_AU, rhill_AU]
) -> torch.Tensor:
    """Compute H_Newton_n = T_Newton_n + V_Newton_n per sample (no autograd targets).

    H is in units of H_ref = G·ms·mp/ap = 4π²·ms·mp/ap.

    T_Newton_n = 0.5 × Σ_i (m_i/mp)·v_n_i² = 0.5 × (p_n · v_n).sum(-1)
        because p_n_i = (m_i/mp)·v_n_i → p_n_i·v_n_i = (m_i/mp)·v_n_i²  ✓

    V_Newton_n = -(1/d_sp_n + (mm/mp)/d_sm_n + (mm/ms)·(ap/rhill)/d_pm_n)
        same formula as _compute_v_newton_n() in hnn_dataset.py.
    """
    ms_solar = sys_raw[:, 0]
    mp_earth = sys_raw[:, 1]
    mm_earth = sys_raw[:, 2]
    ap_AU    = sys_raw[:, 3]
    rhill_AU = sys_raw[:, 4]

    mm_over_mp = mm_earth / mp_earth
    mm_over_ms = mm_earth * _MERTH_OVER_MSUN / ms_solar
    ap_over_rh = ap_AU / rhill_AU

    d_sp_n = d_n[:, 0]; d_sm_n = d_n[:, 1]; d_pm_n = d_n[:, 2]

    V_n = -(1.0 / d_sp_n + mm_over_mp / d_sm_n + mm_over_ms * ap_over_rh / d_pm_n)

    # T_Newton_n = 0.5 × (p_n · v_n) summed over all 9 velocity components
    T_n = 0.5 * (p_n.detach() * v_n).sum(dim=-1)

    return (T_n + V_n).detach()   # target — no grad propagated back through H_Newton


def log_mse_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = _LOG_EPS) -> torch.Tensor:
    """Log-space MSE for strictly positive quantities.

    Symmetric: 10× over or under → (log 10)² ≈ 5.3 each.
    Uses clamp(min=eps) not abs() — wrong-sign pred gives log(eps)−log(targ),
    a large penalty proportional to the magnitude mismatch plus sign violation.
    Only valid when target > 0.
    """
    return (torch.log(pred.clamp(min=eps)) - torch.log(target.clamp(min=eps))).pow(2).mean()


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    data_path:   str,
    out_dir:     str,
    epochs:      int   = 30,
    batch_size:  int   = 512,
    lr:          float = 1e-3,
    weight_decay: float = 0.0,
    dropout:     float = 0.0,
    hidden:      int   = 128,
    layers:      int   = 2,
    lambda_vel:  float = 1.0,
    lambda_h:    float = 0.0,   # Option C: weight on H_θ value supervision (0 = disabled)
    val_frac:    float = 0.20,
    seed:        int   = 42,
    device:      str   = "cpu",
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    use_value = lambda_h > 0.0
    print("HNN Greydanus training (Option E -- joint H_theta with force + velocity supervision)")
    if use_value:
        print(f"  *** Option C enabled: H_Newton value supervision (lambda_h={lambda_h}) ***")
    print(f"  data       : {data_path}")
    print(f"  out        : {out_dir}")
    print(f"  arch       : {hidden}×{layers}  dropout={dropout}")
    print(f"  optim      : epochs={epochs}  batch={batch_size}  lr={lr}  "
          f"weight_decay={weight_decay}")
    print(f"  lambda_vel : {lambda_vel}  (weight on velocity supervision vs force supervision)")
    print(f"  t_pm loss  : log-MSE on bound rows only (clamp min={_LOG_EPS:.0e})")
    if use_value:
        print(f"  lambda_h   : {lambda_h}  (weight on H_Newton value supervision)")

    # ── Dataset ────────────────────────────────────────────────────────────────
    print("\nLoading dataset...")
    train_ds, val_ds, sys_scaler = _load_and_split(
        data_path, val_frac=val_frac, seed=seed,
    )
    save_sys_scaler(sys_scaler, out_dir)
    print(f"  hnn_sys_scaler.pkl -> {out_dir}/")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device != "cpu"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    model   = HNNGreydanus(hidden=hidden, layers=layers, dropout=dropout).to(device)
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    in_dim  = model.d_dim + model.v_dim + model.sys_dim
    print(f"\nModel: {n_param:,} trainable params  (input dim = {in_dim})")

    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", patience=5, factor=0.5,
    )

    history = {
        "train_loss":  [],
        "val_loss":    [],
        "train_force": [],
        "train_vel":   [],
        "train_h":     [],
        "val_force":   [],
        "val_vel":     [],
        "val_h":       [],
        "epochs":      0,
        "hyperparams": {
            "hidden": hidden, "layers": layers, "lr": lr,
            "batch_size": batch_size, "lambda_vel": lambda_vel,
            "lambda_h": lambda_h, "tpm_loss": "log_mse",
        },
    }
    best_val = float("inf")
    t_start  = time.time()
    status   = {}

    h_hdr = f"  {'L_h':>8}" if use_value else ""
    print(f"\n{'Epoch':>6}  {'train':>10}  {'val':>10}  "
          f"{'t_force':>9}  {'t_vel':>9}"
          + h_hdr
          + f"  {'lr':>8}  {'t (s)':>7}")
    sep_len = 68 + (11 if use_value else 0)
    print("-" * sep_len)

    for epoch in range(1, epochs + 1):

        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        tr_total = tr_force = tr_vel = tr_h = 0.0
        n_tr     = 0

        for d_n, p_n, v_n, sys_enc, t_grad, sys_raw_b, stable_b in train_loader:
            d_n       = d_n.to(device)
            p_n       = p_n.to(device)
            v_n       = v_n.to(device)
            sys_enc   = sys_enc.to(device)
            t_grad    = t_grad.to(device)
            sys_raw_b = sys_raw_b.to(device)
            stable_b  = stable_b.to(device)

            opt.zero_grad()

            d_n_leaf = d_n.detach().requires_grad_(True)
            p_n_leaf = p_n.detach().requires_grad_(True)

            H = model(d_n_leaf, p_n_leaf, sys_enc)   # (batch,)

            # Force supervision: masked per-component MSRE
            dH_dd = torch.autograd.grad(
                H.sum(), d_n_leaf, create_graph=True, retain_graph=True,
            )[0]   # (batch, 3)
            L_force = _force_loss(dH_dd, t_grad, stable_b)

            # Momentum supervision: MSRE on ∂H/∂p_n vs v_n (Hamilton: ∂H/∂p = v)
            # retain_graph=True always: H's intermediate activations must survive
            # until L.backward() — the second autograd.grad call would otherwise
            # free them, breaking the dH_dd second-order graph that L.backward() needs.
            dH_dp = torch.autograd.grad(
                H.sum(), p_n_leaf, create_graph=True, retain_graph=True,
            )[0]   # (batch, 9)
            L_vel = msre_loss(dH_dp, v_n)

            L = L_force + lambda_vel * L_vel

            # Option C: value supervision — pin H_θ to H_Newton_n
            if use_value:
                H_newton = _compute_h_newton_n(d_n_leaf, p_n_leaf, v_n, sys_raw_b)
                L_h = (H - H_newton).pow(2).mean()
                L = L + lambda_h * L_h
                tr_h += L_h.item()

            L.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

            tr_total += L.item()
            tr_force += L_force.item()
            tr_vel   += L_vel.item()
            n_tr     += 1

        tr_total /= max(n_tr, 1)
        tr_force /= max(n_tr, 1)
        tr_vel   /= max(n_tr, 1)
        tr_h     /= max(n_tr, 1)

        # ── Validation ─────────────────────────────────────────────────────────
        model.eval()
        va_total = va_force = va_vel = va_h = 0.0
        n_va     = 0

        with torch.enable_grad():   # autograd.grad needs grad even in eval
            for d_n, p_n, v_n, sys_enc, t_grad, sys_raw_b, stable_b in val_loader:
                d_n       = d_n.to(device)
                p_n       = p_n.to(device)
                v_n       = v_n.to(device)
                sys_enc   = sys_enc.to(device)
                t_grad    = t_grad.to(device)
                sys_raw_b = sys_raw_b.to(device)
                stable_b  = stable_b.to(device)

                d_n_leaf = d_n.detach().requires_grad_(True)
                p_n_leaf = p_n.detach().requires_grad_(True)

                H = model(d_n_leaf, p_n_leaf, sys_enc)

                dH_dd = torch.autograd.grad(
                    H.sum(), d_n_leaf, create_graph=False, retain_graph=True,
                )[0]
                L_force = _force_loss(dH_dd, t_grad, stable_b)

                dH_dp = torch.autograd.grad(
                    H.sum(), p_n_leaf, create_graph=False,
                    retain_graph=use_value,
                )[0]
                L_vel = msre_loss(dH_dp, v_n)

                va_total += (L_force + lambda_vel * L_vel).item()
                va_force += L_force.item()
                va_vel   += L_vel.item()

                if use_value:
                    H_newton = _compute_h_newton_n(d_n_leaf, p_n_leaf, v_n, sys_raw_b)
                    L_h = (H - H_newton).pow(2).mean()
                    va_h += L_h.item()

                n_va += 1

        va_total /= max(n_va, 1)
        va_force /= max(n_va, 1)
        va_vel   /= max(n_va, 1)
        va_h     /= max(n_va, 1)

        sched.step(va_total)
        elapsed = time.time() - t_start
        cur_lr  = opt.param_groups[0]["lr"]

        h_str = f"  {tr_h:>8.5f}" if use_value else ""
        print(f"{epoch:>6d}  {tr_total:>10.5f}  {va_total:>10.5f}  "
              f"{tr_force:>9.5f}  {tr_vel:>9.5f}"
              + h_str
              + f"  {cur_lr:>8.1e}  {elapsed:>7.1f}")

        history["train_loss"].append(tr_total)
        history["val_loss"].append(va_total)
        history["train_force"].append(tr_force)
        history["train_vel"].append(tr_vel)
        history["train_h"].append(tr_h)
        history["val_force"].append(va_force)
        history["val_vel"].append(va_vel)
        history["val_h"].append(va_h)

        status = {
            "status":       "running",
            "epoch":        epoch,
            "total_epochs": epochs,
            "train_loss":   tr_total,
            "val_loss":     va_total,
            "train_force":  tr_force,
            "train_vel":    tr_vel,
            "train_h":      tr_h,
            "val_force":    va_force,
            "val_vel":      va_vel,
            "val_h":        va_h,
            "elapsed_s":    elapsed,
        }
        with open(os.path.join(out_dir, "hnn_train_status.json"), "w") as f:
            json.dump(status, f)

        if va_total < best_val:
            best_val = va_total
            model.save(out_dir)
            print(f"         * saved best model (val={best_val:.5f})")

    # ── Finalise ───────────────────────────────────────────────────────────────
    history["epochs"] = epochs
    with open(os.path.join(out_dir, "hnn_training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    status["status"] = "complete"
    with open(os.path.join(out_dir, "hnn_train_status.json"), "w") as f:
        json.dump(status, f)

    print(f"\nTraining complete.  Best val_loss = {best_val:.5f}")
    print(f"Model and config saved to: {out_dir}/")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train HNN Greydanus (Option E — joint H_θ with force + velocity supervision)"
    )
    ap.add_argument("--data",         required=True, help="Path to ml_dataset.parquet")
    ap.add_argument("--out",          default="models_hnn_greydanus_logpm/",
                    help="Output dir (never use models/, models_temphead/, models_hnn_dist_v12/)")
    ap.add_argument("--epochs",       type=int,   default=30)
    ap.add_argument("--batch",        type=int,   default=512)
    ap.add_argument("--lr",           type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--dropout",      type=float, default=0.0)
    ap.add_argument("--hidden",       type=int,   default=128)
    ap.add_argument("--layers",       type=int,   default=2)
    ap.add_argument("--lambda_vel",   type=float, default=1.0,
                    help="Weight on velocity supervision loss (default 1.0 = equal to force loss)")
    ap.add_argument("--lambda_h",     type=float, default=0.0,
                    help="Weight on H_Newton value supervision (Option C). 0=disabled (default)")
    ap.add_argument("--device",       default="cpu")
    args = ap.parse_args()

    train(
        data_path    = args.data,
        out_dir      = args.out,
        epochs       = args.epochs,
        batch_size   = args.batch,
        lr           = args.lr,
        weight_decay = args.weight_decay,
        dropout      = args.dropout,
        hidden       = args.hidden,
        layers       = args.layers,
        lambda_vel   = args.lambda_vel,
        lambda_h     = args.lambda_h,
        device       = args.device,
    )


if __name__ == "__main__":
    main()
