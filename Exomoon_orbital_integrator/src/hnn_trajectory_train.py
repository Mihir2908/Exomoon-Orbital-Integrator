"""
hnn_trajectory_train.py — Option D: trajectory-level training for HNN v12.

Replaces static force supervision with a trajectory distance loss:

    L = (1/T) Σ_t [ ((d_pm_HNN(t) - d_pm_true(t)) / rhill)²
                   + ((d_sm_HNN(t) - d_sm_true(t)) / ap)²    ]

rolled out via differentiable KDK leapfrog for T consecutive resampled steps.

Why this solves the static-supervision failure:
  The 80% escaped-habitable training rows that collapsed t_pm in static training
  contribute negligible position drift (small force → small displacement), so the
  trajectory loss naturally de-weights them regardless of proportion in the dataset.
  The stable-orbit rows — where force errors cause large orbit drift — dominate
  the gradient. No distribution filter needed.

Warmstart: loads weights from --warmstart_dir (default models_hnn_dist_v12/).
Output:    writes to --out (default models_hnn_option_d/).

Usage:
    py hnn_trajectory_train.py \\
        --data ml_dataset.parquet \\
        --warmstart_dir models_hnn_dist_v12 \\
        --out models_hnn_option_d \\
        --epochs 30 --lr 5e-5 --T 10 --batch 32 --steps_per_epoch 300

ISOLATION: never writes to models/, models_temphead/, models_hnn_dist_v12/,
           models_hnn_dist_factorized_v*/, eval_aux_mlp_output/.
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
from torch.utils.data import Dataset, DataLoader

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from exomoon.ml.hnn_model   import HNN
from exomoon.ml.hnn_dataset import load_sys_scaler, save_sys_scaler, _MERTH_OVER_MSUN
from exomoon.constants      import FOUR_PI2

_POS_COLS = [
    "star_x",   "star_y",   "star_z",
    "planet_x", "planet_y", "planet_z",
    "moon_x",   "moon_y",   "moon_z",
]
_VEL_COLS = [
    "star_vx",  "star_vy",  "star_vz",
    "planet_vx","planet_vy","planet_vz",
    "moon_vx",  "moon_vy",  "moon_vz",
]
_SYS_COLS = ["ms_solar", "mp_earth", "mm_earth", "ap_AU", "rhill_AU"]
_LOAD_COLS = (["sim_id", "t_years", "stable", "habitable"]
              + _POS_COLS + _VEL_COLS + _SYS_COLS)


# ── Dataset ───────────────────────────────────────────────────────────────────

class TrajSubseqDataset(Dataset):
    """
    Provides (T+1)-step position/velocity subsequences from training simulations.

    Items are (q_seq, v_seq, sys_raw, V_ref, dt, rhill_AU, ap_AU, inv_masses)
    where q_seq and v_seq are (T+1, 9) consecutive resampled timesteps.

    dt here is the RESAMPLED step size (t_sim / n_steps), not the physics dt.
    The KDK rollout uses this dt; it is coarser than the physics dt but the
    trajectory loss still drives V_θ toward the correct potential shape.
    """

    def __init__(self, sims: list[dict], T: int = 10) -> None:
        self.T    = T
        self.sims = sims
        self.index: list[tuple[int, int]] = []   # (sim_idx, start_row)

        for i, sim in enumerate(sims):
            n = len(sim["q"])
            for start in range(n - T):
                self.index.append((i, start))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        sim_idx, start = self.index[idx]
        sim = self.sims[sim_idx]
        T   = self.T
        sl  = slice(start, start + T + 1)

        return (
            torch.from_numpy(sim["q"][sl].astype(np.float32)),        # (T+1, 9)
            torch.from_numpy(sim["v"][sl].astype(np.float32)),        # (T+1, 9)
            torch.from_numpy(sim["sys_raw"].astype(np.float32)),      # (5,)
            torch.tensor(sim["V_ref"],    dtype=torch.float32),       # scalar
            torch.tensor(sim["dt"],       dtype=torch.float32),       # scalar
            torch.tensor(sim["rhill_AU"], dtype=torch.float32),       # scalar
            torch.tensor(sim["ap_AU"],    dtype=torch.float32),       # scalar
            torch.from_numpy(sim["inv_masses"].astype(np.float32)),   # (9,)
        )


def _load_sims(parquet_path: str, sys_scaler, val_frac: float, seed: int):
    """
    Load Parquet, apply stable+habitable trajectory filter (same as hnn_dataset.py),
    group by sim_id, and compute per-sim derived quantities.

    Returns (train_sims, val_sims) as lists of dicts.
    """
    print(f"  Loading {parquet_path} ...")
    df = pd.read_parquet(parquet_path, columns=_LOAD_COLS)
    n_raw = len(df)

    # 1. Keep sims that start stable+habitable
    first = df.groupby("sim_id")[["stable", "habitable"]].first()
    valid = first[(first["stable"] == 1) & (first["habitable"] == 1)].index
    df = df[df["sim_id"].isin(valid)].copy()

    # 2. Per-sim: trim at first stable=0 AND habitable=0 step
    df["_bad"] = ((df["stable"] == 0) & (df["habitable"] == 0)).astype(np.int8)
    df["_cum"] = df.groupby("sim_id")["_bad"].cumsum()
    df = df[df["_cum"] == 0].drop(columns=["_bad", "_cum", "stable", "habitable"])

    print(f"  {n_raw:,} raw rows -> {len(df):,} kept "
          f"({len(valid)} valid sims of {n_raw//1000} total)")

    # Train/val split by sim_id
    all_sids = np.array(valid)
    rng = np.random.default_rng(seed)
    rng.shuffle(all_sids)
    n_val   = max(1, int(len(all_sids) * val_frac))
    val_set = set(all_sids[:n_val].tolist())

    sims_train, sims_val = [], []
    for sid, grp in df.groupby("sim_id"):
        grp = grp.sort_values("t_years")

        q    = grp[_POS_COLS].to_numpy(dtype=np.float64)   # (n, 9)  AU
        v    = grp[_VEL_COLS].to_numpy(dtype=np.float64)   # (n, 9)  AU/yr
        n    = len(q)
        if n < 3:
            continue

        sys_row  = grp[_SYS_COLS].iloc[0]
        ms       = float(sys_row["ms_solar"])
        mp_earth = float(sys_row["mp_earth"])
        mm_earth = float(sys_row["mm_earth"])
        ap       = float(sys_row["ap_AU"])
        rhill    = float(sys_row["rhill_AU"])

        mp_msun  = mp_earth * _MERTH_OVER_MSUN
        mm_msun  = mm_earth * _MERTH_OVER_MSUN

        # V_ref = G·ms·mp/ap  (same convention as eval_fair_trajectory.py)
        V_ref    = FOUR_PI2 * ms * mp_msun / ap

        # Per-body inverse masses (9-vector, one per coordinate)
        inv_m = np.repeat(
            [1.0 / ms, 1.0 / mp_msun, 1.0 / mm_msun], 3
        ).astype(np.float64)   # [1/ms, 1/ms, 1/ms, 1/mp, 1/mp, 1/mp, 1/mm, 1/mm, 1/mm]

        t_arr = grp["t_years"].to_numpy()
        dt    = float((t_arr[-1] - t_arr[0]) / max(len(t_arr) - 1, 1))

        sys_raw = np.array([[ms, mp_earth, mm_earth, ap, rhill]], dtype=np.float64)
        sys_enc = sys_scaler.transform(sys_raw).astype(np.float32)   # (1, 5)

        rec = {
            "q":         q,
            "v":         v,
            "sys_raw":   sys_raw[0],
            "sys_enc":   sys_enc[0],
            "V_ref":     V_ref,
            "dt":        dt,
            "rhill_AU":  rhill,
            "ap_AU":     ap,
            "inv_masses": inv_m,
        }

        if sid in val_set:
            sims_val.append(rec)
        else:
            sims_train.append(rec)

    print(f"  Train sims: {len(sims_train)}  |  Val sims: {len(sims_val)}")
    return sims_train, sims_val


# ── Differentiable KDK leapfrog ───────────────────────────────────────────────

def _accel_batch(
    q:       torch.Tensor,   # (B, 9)  — requires_grad must be True
    sys_enc: torch.Tensor,   # (B, 5)
    model:   HNN,
    ap_b:    torch.Tensor,   # (B,)
    rhill_b: torch.Tensor,   # (B,)
    V_ref_b: torch.Tensor,   # (B,)
    inv_m_b: torch.Tensor,   # (B, 9)
) -> torch.Tensor:
    """
    Differentiable force step: positions → accelerations (AU/yr²).

    V_ref = G·ms·mp/ap converts the dimensionless HNN potential gradient to
    physical forces (M☉·AU/yr²), then divides by per-body mass (AU/yr²).
    """
    B = q.shape[0]

    d_sp = (q[:, 3:6] - q[:, 0:3]).norm(dim=1, keepdim=True)   # (B,1) AU
    d_sm = (q[:, 6:9] - q[:, 0:3]).norm(dim=1, keepdim=True)
    d_pm = (q[:, 6:9] - q[:, 3:6]).norm(dim=1, keepdim=True)

    ap_col    = ap_b.view(B, 1)
    rhill_col = rhill_b.view(B, 1)
    d_n = torch.cat([d_sp / ap_col, d_sm / ap_col, d_pm / rhill_col], dim=1)  # (B,3)

    V   = model(d_n, sys_enc)                                                  # (B,)
    dV  = torch.autograd.grad(V.sum(), q, create_graph=True)[0]               # (B,9)

    # F = -∂V_normed/∂q × V_ref   [M☉·AU/yr²]
    # a = F / mass   [AU/yr²]
    return -dV * V_ref_b.view(B, 1) * inv_m_b   # (B,9)


def kdk_rollout(
    q0_batch: torch.Tensor,   # (B, 9) initial positions, float32
    v0_batch: torch.Tensor,   # (B, 9) initial velocities, float32
    T:        int,
    model:    HNN,
    sys_enc:  torch.Tensor,   # (B, 5)
    ap_b:     torch.Tensor,   # (B,)
    rhill_b:  torch.Tensor,   # (B,)
    V_ref_b:  torch.Tensor,   # (B,)
    inv_m_b:  torch.Tensor,   # (B, 9)
    dt_b:     torch.Tensor,   # (B,)
) -> list[torch.Tensor]:
    """
    Differentiable T-step KDK leapfrog rollout.

    Gradient flows fully back through all T steps to model parameters via
    create_graph=True in _accel_batch.  Memory is O(T × model_size × B).

    Returns: list of T+1 position tensors, each (B, 9).
    """
    half_dt = (dt_b / 2.0).view(-1, 1)   # (B, 1)
    dt_col  = dt_b.view(-1, 1)            # (B, 1)

    # Initial positions as leaf tensors so autograd.grad(V, q) can operate on them
    q = q0_batch.float().requires_grad_(True)   # (B, 9) leaf
    v = v0_batch.float()                        # (B, 9) no grad (initial data)

    q_list = [q]

    # Pre-compute initial acceleration
    a = _accel_batch(q, sys_enc, model, ap_b, rhill_b, V_ref_b, inv_m_b)

    for _ in range(T):
        v_half = v + a * half_dt          # (B,9) — in graph through a → model
        q_new  = q + v_half * dt_col      # (B,9) — in graph through v_half → a
        a_new  = _accel_batch(q_new, sys_enc, model, ap_b, rhill_b, V_ref_b, inv_m_b)
        v_new  = v_half + a_new * half_dt

        q_list.append(q_new)
        q, v, a = q_new, v_new, a_new

    return q_list   # T+1 tensors, each (B, 9)


def traj_loss(
    q_pred_list: list[torch.Tensor],   # T+1 tensors (B,9), index 0 = t=0
    q_true:      torch.Tensor,          # (B, T+1, 9) ground truth positions
    rhill_b:     torch.Tensor,          # (B,)
    ap_b:        torch.Tensor,          # (B,)
) -> torch.Tensor:
    """
    Normalised trajectory distance loss over T predicted steps.

    d_pm and d_sm errors are normalised by rhill and ap respectively so that
    both terms have comparable scale across systems.  The t=0 step is skipped
    (initial conditions are exact).
    """
    rhill_col = rhill_b.view(-1, 1)   # (B,1)
    ap_col    = ap_b.view(-1, 1)      # (B,1)
    loss = torch.tensor(0.0, requires_grad=True)

    for t in range(1, len(q_pred_list)):
        q_p = q_pred_list[t]                         # (B,9) predicted
        q_t = q_true[:, t, :].to(q_p.device)        # (B,9) true

        d_pm_p = (q_p[:, 6:9] - q_p[:, 3:6]).norm(dim=1, keepdim=True)  # (B,1)
        d_sm_p = (q_p[:, 6:9] - q_p[:, 0:3]).norm(dim=1, keepdim=True)

        d_pm_t = (q_t[:, 6:9] - q_t[:, 3:6]).norm(dim=1, keepdim=True)
        d_sm_t = (q_t[:, 6:9] - q_t[:, 0:3]).norm(dim=1, keepdim=True)

        err_pm = ((d_pm_p - d_pm_t) / rhill_col) ** 2
        err_sm = ((d_sm_p - d_sm_t) / ap_col)    ** 2

        loss = loss + (err_pm + err_sm).mean()

    return loss / (len(q_pred_list) - 1)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    data_path:       str,
    warmstart_dir:   str,
    out_dir:         str,
    epochs:          int   = 30,
    lr:              float = 5e-5,
    T:               int   = 10,
    batch_size:      int   = 32,
    steps_per_epoch: int   = 300,
    val_steps:       int   = 50,
    val_frac:        float = 0.20,
    seed:            int   = 42,
    device:          str   = "cpu",
    grad_clip:       float = 1.0,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    print("HNN trajectory training (Option D)")
    print(f"  data         : {data_path}")
    print(f"  warmstart    : {warmstart_dir}")
    print(f"  out          : {out_dir}")
    print(f"  T (rollout)  : {T} resampled steps")
    print(f"  batch        : {batch_size}  steps/epoch: {steps_per_epoch}")
    print(f"  lr           : {lr}  grad_clip: {grad_clip}")
    print(f"  epochs       : {epochs}")

    # ── Load model (warmstart) ─────────────────────────────────────────────
    print(f"\nLoading HNN from {warmstart_dir} ...")
    model     = HNN.load(warmstart_dir).to(device)
    sys_scaler = load_sys_scaler(warmstart_dir)

    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {n_param:,} trainable parameters")
    print(f"  Warmstarted — original weights in {warmstart_dir}/ are UNCHANGED")

    # Copy sys_scaler to out_dir so models_hnn_option_d/ is self-contained for inference
    save_sys_scaler(sys_scaler, out_dir)
    print(f"  hnn_sys_scaler.pkl copied to {out_dir}/")

    # ── Dataset ────────────────────────────────────────────────────────────
    print("\nLoading dataset ...")
    sims_train, sims_val = _load_sims(data_path, sys_scaler, val_frac, seed)

    train_ds = TrajSubseqDataset(sims_train, T=T)
    val_ds   = TrajSubseqDataset(sims_val,   T=T)
    print(f"  Train items: {len(train_ds):,}  |  Val items: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=True)

    # ── Optimiser ─────────────────────────────────────────────────────────
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", patience=5, factor=0.5
    )

    # ── Training ──────────────────────────────────────────────────────────
    history   = {"train_loss": [], "val_loss": [], "epochs": 0,
                 "hyperparams": {"T": T, "lr": lr, "batch_size": batch_size,
                                 "warmstart_dir": warmstart_dir}}
    best_val  = float("inf")
    t_start   = time.time()

    print(f"\n{'Epoch':>6}  {'train':>10}  {'val':>10}  {'lr':>8}  {'t (s)':>7}")
    print("-" * 50)

    for epoch in range(1, epochs + 1):

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_iter  = iter(train_loader)
        train_total = 0.0
        n_train     = 0

        for step in range(steps_per_epoch):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            q_seq, v_seq, sys_raw, V_ref, dt, rhill, ap, inv_m = [
                b.to(device) for b in batch
            ]
            # q_seq: (B, T+1, 9)
            # sys_raw: (B, 5)

            sys_enc = torch.from_numpy(
                sys_scaler.transform(sys_raw.cpu().numpy()).astype(np.float32)
            ).to(device)   # (B, 5)

            opt.zero_grad()

            q_pred = kdk_rollout(
                q0_batch=q_seq[:, 0, :],
                v0_batch=v_seq[:, 0, :],
                T=T,
                model=model,
                sys_enc=sys_enc,
                ap_b=ap,
                rhill_b=rhill,
                V_ref_b=V_ref,
                inv_m_b=inv_m,
                dt_b=dt,
            )

            loss = traj_loss(q_pred, q_seq, rhill, ap)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            train_total += loss.item()
            n_train     += 1

        train_loss = train_total / max(n_train, 1)

        # ── Validation ──────────────────────────────────────────────────────
        model.eval()
        val_iter  = iter(val_loader)
        val_total = 0.0
        n_val     = 0

        with torch.enable_grad():   # needed for create_graph=True inside rollout
            for step in range(val_steps):
                try:
                    batch = next(val_iter)
                except StopIteration:
                    break

                q_seq, v_seq, sys_raw, V_ref, dt, rhill, ap, inv_m = [
                    b.to(device) for b in batch
                ]
                sys_enc = torch.from_numpy(
                    sys_scaler.transform(sys_raw.cpu().numpy()).astype(np.float32)
                ).to(device)

                q_pred = kdk_rollout(
                    q0_batch=q_seq[:, 0, :],
                    v0_batch=v_seq[:, 0, :],
                    T=T,
                    model=model,
                    sys_enc=sys_enc,
                    ap_b=ap,
                    rhill_b=rhill,
                    V_ref_b=V_ref,
                    inv_m_b=inv_m,
                    dt_b=dt,
                )

                loss_v = traj_loss(q_pred, q_seq, rhill, ap)
                val_total += loss_v.item()
                n_val     += 1

        val_loss = val_total / max(n_val, 1)
        sched.step(val_loss)

        elapsed = time.time() - t_start
        cur_lr  = opt.param_groups[0]["lr"]
        print(f"{epoch:>6d}  {train_loss:>10.5f}  {val_loss:>10.5f}  "
              f"{cur_lr:>8.1e}  {elapsed:>7.1f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        status = {"status": "running", "epoch": epoch, "total_epochs": epochs,
                  "train_loss": train_loss, "val_loss": val_loss, "elapsed_s": elapsed}
        with open(os.path.join(out_dir, "hnn_traj_train_status.json"), "w") as f:
            json.dump(status, f)

        if val_loss < best_val:
            best_val = val_loss
            model.save(out_dir)
            print(f"         * saved best (val={best_val:.5f})")

    # ── Finalise ───────────────────────────────────────────────────────────
    history["epochs"] = epochs
    with open(os.path.join(out_dir, "hnn_traj_training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    status["status"] = "complete"
    with open(os.path.join(out_dir, "hnn_traj_train_status.json"), "w") as f:
        json.dump(status, f)

    print(f"\nDone. Best val_loss = {best_val:.5f}")
    print(f"Weights saved to: {out_dir}/  (original {warmstart_dir}/ unchanged)")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="HNN trajectory-level training (Option D)")
    ap.add_argument("--data",            required=True,   help="ml_dataset.parquet path")
    ap.add_argument("--warmstart_dir",   default="models_hnn_dist_v12")
    ap.add_argument("--out",             default="models_hnn_option_d")
    ap.add_argument("--epochs",          type=int,   default=30)
    ap.add_argument("--lr",              type=float, default=5e-5)
    ap.add_argument("--T",               type=int,   default=10,
                    help="Rollout steps per training window (default 10)")
    ap.add_argument("--batch",           type=int,   default=32)
    ap.add_argument("--steps_per_epoch", type=int,   default=300,
                    help="Gradient steps per epoch (random subsample)")
    ap.add_argument("--val_steps",       type=int,   default=50)
    ap.add_argument("--grad_clip",       type=float, default=1.0)
    ap.add_argument("--device",          default="cpu")
    args = ap.parse_args()

    train(
        data_path       = args.data,
        warmstart_dir   = args.warmstart_dir,
        out_dir         = args.out,
        epochs          = args.epochs,
        lr              = args.lr,
        T               = args.T,
        batch_size      = args.batch,
        steps_per_epoch = args.steps_per_epoch,
        val_steps       = args.val_steps,
        grad_clip       = args.grad_clip,
        device          = args.device,
    )


if __name__ == "__main__":
    main()
