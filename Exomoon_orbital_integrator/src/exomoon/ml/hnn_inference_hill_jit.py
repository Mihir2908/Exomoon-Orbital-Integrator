"""
exomoon/ml/hnn_inference_hill_jit.py — JIT-compiled KDK inference for HNN_Hill hinge4.

Two-step speedup over hnn_inference_hill.py:
  Step 1: Replace torch.autograd.grad() with analytical chain-rule Jacobian through
          the 3-layer Tanh MLP (makes the force function torch.jit.script-able).
  Step 2: torch.jit.script the ENTIRE KDK inner loop — converts 200,000 Python
          loop iterations into a single compiled C++ call.

Expected speedup: 10–30× over autograd baseline.

hnn_inference_hill.py is NOT touched.

Architecture assumed: HNN_Hill hinge4 — 3 hidden layers, hidden=256, Tanh activations.
  net[0]: Linear(9, 256)   net[2]: Linear(256,256)   net[4]: Linear(256,256)
  net[6]: Linear(256, 1)
JIT functions _analytical_grad_3layer and _hill_forces_jit hardcode this 3-layer
structure (no Python loop in backward = maximum JIT performance).
"""

from __future__ import annotations

import os
import sys
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from exomoon.constants            import FOUR_PI2, au, merth, msun
from exomoon.constants            import rsun as _rsun
from exomoon.habitable_zone       import hz_bounds_au
from exomoon.ml.batch_leapfrog    import _build_initial_states
from exomoon.ml.hnn_dataset_hill  import load_hill_sys_scaler, _MERTH_OVER_MSUN
from exomoon.ml.hnn_model_hill    import HNN_Hill, is_hill_dir

_PLANET_DENSITY_CGS  = 5.5
_PLANET_DENSITY_SI   = 5500.0
_MOON_DENSITY_CGS    = 3.0
_WARMUP_STEPS        = 5
_FOUR_PI2_F          = float(FOUR_PI2)   # 4π² ≈ 39.4784 — passed as float to JIT


# ── JIT-scriptable helper functions ──────────────────────────────────────────

@torch.jit.script
def _cross3_jit(a: Tensor, b: Tensor) -> Tensor:
    return torch.stack([
        a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
        a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
        a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0],
    ], dim=1)


@torch.jit.script
def _analytical_grad_3layer(
    q_h_f32: Tensor,   # (N, 3) F32 — normalised Hill position, NO requires_grad
    sys_enc:  Tensor,  # (N, 6) F32 — log-scaled system parameters
    W1:  Tensor, b1: Tensor,   # Linear(9→256)
    W2:  Tensor, b2: Tensor,   # Linear(256→256)
    W3:  Tensor, b3: Tensor,   # Linear(256→256)
    w4:  Tensor,               # (256,) — output weight row, squeezed
    W1_q: Tensor,              # (256, 3) — first 3 cols of W1 (wrt q_h_norm only)
) -> Tensor:   # (N, 3) F32 — dV/d(q_h_norm)
    """
    Analytical Jacobian of V wrt q_h_norm through a 3-hidden-layer Tanh MLP.
    Equivalent to autograd.grad(V.sum(), q_h_f32)[0] but without a backward graph.
    Hardcoded for 3 layers — no Python for-loop in backward.
    """
    x  = torch.cat([q_h_f32, sys_enc], dim=1)   # (N, 9)
    h1 = torch.tanh(x  @ W1.T + b1)              # (N, 256)
    h2 = torch.tanh(h1 @ W2.T + b2)              # (N, 256)
    h3 = torch.tanh(h2 @ W3.T + b3)              # (N, 256)
    # Tanh derivative: d_tanh(h) = 1 - h²
    d3 = 1.0 - h3 * h3
    d2 = 1.0 - h2 * h2
    d1 = 1.0 - h1 * h1
    # Chain rule backward (delta = dV/d(pre-activation))
    delta = w4 * d3              # (N, 256): dV/dh3 × d3
    delta = (delta @ W3) * d2   # (N, 256): backprop through W3 to h2
    delta = (delta @ W2) * d1   # (N, 256): backprop through W2 to h1
    return delta @ W1_q          # (N, 3):  backprop through W1 to q_h_norm


@torch.jit.script
def _hill_forces_jit(
    q_ms: Tensor, q_mp: Tensor, q_mm: Tensor,
    v_ms: Tensor, v_mp: Tensor,
    # Precomputed model weights (F32, on device)
    W1: Tensor, b1: Tensor,
    W2: Tensor, b2: Tensor,
    W3: Tensor, b3: Tensor,
    w4: Tensor,
    W1_q: Tensor,
    # System tensors
    sys_raw_static: Tensor,   # (N, 5) F32: [ms, mp, mm, ap, rhill]
    mm_msun:        Tensor,   # (N, 1) F64: moon mass in solar units
    scaler_mean:    Tensor,   # (6,) F32
    scaler_std:     Tensor,   # (6,) F32
    # Physics scalars
    rhill_AU:   float,
    V_ref_val:  float,
    mp_msun:    float,
    ms_solar:   float,
    FOUR_PI2:   float,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute (a_star, a_planet, a_moon) in inertial frame — fully JIT-scriptable."""

    # ── Hill frame ────────────────────────────────────────────────────────────
    r_sp  = q_mp - q_ms
    v_sp  = v_mp - v_ms
    r_sp_sq1 = (r_sp * r_sp).sum(1).clamp(min=1e-20)
    r_sp_n   = r_sp_sq1.sqrt().unsqueeze(1)
    x_hat    = r_sp / r_sp_n

    L_sp  = _cross3_jit(r_sp, v_sp)
    L_n2  = (L_sp * L_sp).sum(1).clamp(min=1e-20)
    L_n   = L_n2.sqrt().unsqueeze(1)
    z_hat = L_sp / L_n
    Omega = L_n.squeeze(1) / r_sp_sq1
    y_hat = _cross3_jit(z_hat, x_hat)

    r_mp  = q_mm - q_mp
    q_h_AU = torch.stack([
        (r_mp * x_hat).sum(1),
        (r_mp * y_hat).sum(1),
        (r_mp * z_hat).sum(1),
    ], dim=1)
    q_h_n = q_h_AU / rhill_AU   # (N, 3) F64 normalised

    # ── System encoding ───────────────────────────────────────────────────────
    Omega_f32 = Omega.float().unsqueeze(1)                       # (N, 1) F32
    sys_raw   = torch.cat([sys_raw_static, Omega_f32], dim=1)   # (N, 6) F32
    sys_enc   = (sys_raw.clamp(min=1e-20).log() - scaler_mean) / scaler_std

    # ── Analytical Jacobian (no autograd, no requires_grad) ───────────────────
    q_h_f32 = q_h_n.float()   # (N, 3) F32 — cast only, no grad tracking
    grad_f32 = _analytical_grad_3layer(
        q_h_f32, sys_enc, W1, b1, W2, b2, W3, b3, w4, W1_q
    )
    a_cons_h = -grad_f32.double() * V_ref_val / (mm_msun * rhill_AU)  # (N, 3) F64

    # ── Centrifugal correction (Hill frame x-y plane) ─────────────────────────
    omega2    = (Omega * Omega).unsqueeze(1)
    zeros_col = torch.zeros_like(q_h_AU[:, :1])
    q_h_plane = torch.cat([q_h_AU[:, :2], zeros_col], dim=1)
    a_hill    = a_cons_h - omega2 * q_h_plane

    # ── Rotate back to inertial frame ─────────────────────────────────────────
    a_rel_in = (a_hill[:, 0:1] * x_hat
              + a_hill[:, 1:2] * y_hat
              + a_hill[:, 2:3] * z_hat)

    # ── Newtonian terms ───────────────────────────────────────────────────────
    r_sp_cu = r_sp_sq1.unsqueeze(1) ** 1.5
    a_p     = -FOUR_PI2 * ms_solar * r_sp / r_sp_cu
    a_m     = a_rel_in + a_p

    r_sm    = q_mm - q_ms
    d_sm_cu = ((r_sm * r_sm).sum(1, keepdim=True)) ** 1.5
    a_s     = (FOUR_PI2 * mm_msun / d_sm_cu * r_sm
             + FOUR_PI2 * mp_msun * r_sp / r_sp_cu)

    return a_s, a_p, a_m


@torch.jit.script
def kdk_loop_jit(
    # Initial state
    q_ms: Tensor, q_mp: Tensor, q_mm: Tensor,
    v_ms: Tensor, v_mp: Tensor, v_mm: Tensor,
    stopped:   Tensor,   # (N, 1) bool
    elapsed_t: Tensor,   # (N, 1) F64
    # Model weights
    W1: Tensor, b1: Tensor,
    W2: Tensor, b2: Tensor,
    W3: Tensor, b3: Tensor,
    w4: Tensor,
    W1_q: Tensor,
    # System tensors
    sys_raw_static: Tensor,
    mm_msun:        Tensor,
    scaler_mean:    Tensor,
    scaler_std:     Tensor,
    # Physics scalars
    rhill_AU:   float,
    V_ref_val:  float,
    mp_msun:    float,
    ms_solar:   float,
    FOUR_PI2:   float,
    dt:         float,
    half_dt:    float,
    escape_factor: float,
    a_inner:    float,
    a_outer:    float,
    t_sim:      float,
    n_phys:     int,
    stride:     int,
    n_out:      int,
    # Pre-allocated output buffers (written in-place)
    mpd_out: Tensor,    # (N, n_out) F32
    msd_out: Tensor,    # (N, n_out) F32
    traj_mp: Tensor,    # (N, n_out, 3) F32
    traj_ms: Tensor,    # (N, n_out, 3) F32
    traj_mm: Tensor,    # (N, n_out, 3) F32
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
           Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int]:
    """
    Entire KDK leapfrog loop — compiled to C++ by torch.jit.script.
    Eliminates 200,000 Python→C++ crossings per simulation.
    """
    rhill_t: float = escape_factor * rhill_AU

    # Initial forces
    a_s, a_p, a_m = _hill_forces_jit(
        q_ms, q_mp, q_mm, v_ms, v_mp,
        W1, b1, W2, b2, W3, b3, w4, W1_q,
        sys_raw_static, mm_msun, scaler_mean, scaler_std,
        rhill_AU, V_ref_val, mp_msun, ms_solar, FOUR_PI2,
    )

    out_idx: int = 0

    for step in range(n_phys):
        mask = (~stopped).to(torch.float64)

        # KDK: half-kick velocity
        v_ms = v_ms + a_s * (half_dt * mask)
        v_mp = v_mp + a_p * (half_dt * mask)
        v_mm = v_mm + a_m * (half_dt * mask)
        # Full-kick position
        q_ms = q_ms + v_ms * (dt * mask)
        q_mp = q_mp + v_mp * (dt * mask)
        q_mm = q_mm + v_mm * (dt * mask)
        # Re-compute forces
        a_s, a_p, a_m = _hill_forces_jit(
            q_ms, q_mp, q_mm, v_ms, v_mp,
            W1, b1, W2, b2, W3, b3, w4, W1_q,
            sys_raw_static, mm_msun, scaler_mean, scaler_std,
            rhill_AU, V_ref_val, mp_msun, ms_solar, FOUR_PI2,
        )
        # Half-kick velocity
        v_ms = v_ms + a_s * (half_dt * mask)
        v_mp = v_mp + a_p * (half_dt * mask)
        v_mm = v_mm + a_m * (half_dt * mask)

        # Distances
        diff_mp = q_mm - q_mp
        diff_ms = q_mm - q_ms
        mpd_col = (diff_mp * diff_mp).sum(1, keepdim=True).sqrt()
        msd_col = (diff_ms * diff_ms).sum(1, keepdim=True).sqrt()

        # Store at stride
        if step % stride == 0 and out_idx < n_out:
            mpd_out[:, out_idx] = mpd_col.squeeze(1).float()
            msd_out[:, out_idx] = msd_col.squeeze(1).float()
            traj_mp[:, out_idx, :] = q_mp.float()
            traj_ms[:, out_idx, :] = q_ms.float()
            traj_mm[:, out_idx, :] = q_mm.float()
            out_idx = out_idx + 1

        # Stopping criterion
        elapsed_t = elapsed_t + mask * dt
        unstable      = mpd_col > rhill_t
        uninhabitable = (msd_col < a_inner) | (msd_col > a_outer)
        stopped = stopped | (unstable & uninhabitable) | (elapsed_t >= t_sim)

        if stopped.all():
            break

    return (q_ms, q_mp, q_mm, v_ms, v_mp, v_mm, stopped, elapsed_t,
            mpd_out, msd_out, traj_mp, traj_ms, traj_mm, out_idx)


# ── Public entry point ────────────────────────────────────────────────────────

def batch_hnn_hill_trajectories_jit(
    system_params:   dict,
    t_sim:           float,
    moon_retrograde: bool           = False,
    em:              float          = 0.0,
    mm_resolution:   int            = 50,
    am_resolution:   int            = 50,
    n_steps:         int            = 1000,
    escape_factor:   float          = 1.0,
    model_dir:       str            = "models_hnn_hill_hinge4",
    device:          str            = "cpu",
    n_orbits:        "int | None"   = None,
    eligible_mask:   "np.ndarray | None" = None,
) -> dict:
    """
    JIT-compiled KDK inference. Drop-in replacement for batch_hnn_hill_trajectories().
    The entire 200,000-step inner loop runs as compiled C++ (torch.jit.script).
    """
    t0 = time.perf_counter()

    ms_solar = float(system_params["ms_solar"])
    rs_solar = float(system_params.get("rs_solar", 1.0))
    Ts       = float(system_params.get("Ts", 5772.0))
    mp_earth = float(system_params["mp_earth"])
    ap_AU    = float(system_params["ap_AU"])
    ep       = float(system_params.get("ep", 0.0))

    _model_dir = model_dir if os.path.isabs(model_dir) else os.path.join(_SRC, model_dir)

    if not is_hill_dir(_model_dir):
        return {"ok": False, "error": "no_model",
                "message": f"No HNN_Hill checkpoint at {_model_dir}"}

    dev = torch.device(device)
    F64 = torch.float64
    F32 = torch.float32

    hill_model  = HNN_Hill.load(_model_dir)
    hill_model.eval()
    hill_model  = hill_model.to(dev)
    hill_scaler = load_hill_sys_scaler(_model_dir)

    scaler_mean = torch.tensor(hill_scaler.mean_,  dtype=F32, device=dev)
    scaler_std  = torch.tensor(hill_scaler.scale_, dtype=F32, device=dev)

    # ── Extract weights once (F32, on device) ─────────────────────────────────
    linears = [m for m in hill_model.net if isinstance(m, nn.Linear)]
    assert len(linears) == 4, (
        f"Expected 4 Linear layers for hinge4 (layers=3), got {len(linears)}. "
        "hnn_inference_hill_jit.py is hardcoded for the 3-hidden-layer architecture."
    )
    W1   = linears[0].weight.detach()        # (256, 9)
    b1   = linears[0].bias.detach()          # (256,)
    W2   = linears[1].weight.detach()        # (256, 256)
    b2   = linears[1].bias.detach()          # (256,)
    W3   = linears[2].weight.detach()        # (256, 256)
    b3   = linears[2].bias.detach()          # (256,)
    w4   = linears[3].weight.detach().squeeze(0)  # (256,)
    W1_q = linears[0].weight.detach()[:, :3]      # (256, 3)

    # ── Grid ──────────────────────────────────────────────────────────────────
    mp_kg    = mp_earth * merth
    rp_m     = (0.75 * mp_kg / (np.pi * _PLANET_DENSITY_SI)) ** (1.0 / 3.0)
    a_roche  = 2.456 * rp_m * (_PLANET_DENSITY_CGS / _MOON_DENSITY_CGS) ** (1.0 / 3.0) / au

    mp_msun  = mp_earth * _MERTH_OVER_MSUN
    ms_gp    = ms_solar * FOUR_PI2
    mp_gp    = mp_msun  * FOUR_PI2
    rhill_AU = ap_AU * (1.0 - ep) * (mp_gp / (3.0 * ms_gp)) ** (1.0 / 3.0)

    am_min  = max(a_roche / rhill_AU, 1e-3)
    mm_max  = min(mp_earth, 3.0)

    mm_grid = np.exp(np.linspace(np.log(0.107), np.log(mm_max), mm_resolution))
    am_grid = np.linspace(am_min, 1.0, am_resolution)

    a_inner_au, a_outer_au = hz_bounds_au(Ts, rs_solar * _rsun)

    # ── Initial states ─────────────────────────────────────────────────────────
    (pos_mp0, pos_ms0, pos_mm0, vel_mp0, vel_ms0, vel_mm0,
     mu_ms, mu_mp, _mu_mm_arr, _rhill_check) = _build_initial_states(
        ms_solar=ms_solar, mp_earth=mp_earth,
        ap_AU=ap_AU, ep=ep, em=em,
        moon_retrograde=moon_retrograde,
        mm_grid=mm_grid, am_grid=am_grid,
    )

    N = mm_resolution * am_resolution
    mm_per_cell = np.repeat(mm_grid, am_resolution)
    mm_msun_arr = mm_per_cell * _MERTH_OVER_MSUN

    # ── Physics dt ─────────────────────────────────────────────────────────────
    am_ref_AU  = float(am_grid[am_resolution // 2]) * rhill_AU
    mm_mid     = float(mm_grid[mm_resolution // 2])
    mu_mm_ref  = mm_mid * _MERTH_OVER_MSUN * FOUR_PI2
    T_moon_ref = 2.0 * np.pi * np.sqrt(am_ref_AU ** 3 / (mu_mp + mu_mm_ref))
    if n_orbits is not None:
        t_sim = float(n_orbits) * T_moon_ref
    dt_phys    = min(T_moon_ref / 100.0, 1.0 / 20000.0)
    n_phys     = max(int(np.ceil(t_sim / dt_phys)), n_steps)
    stride     = max(1, n_phys // n_steps)
    n_out      = n_phys // stride

    mode_tag = f"n_orbits={n_orbits}" if n_orbits is not None else f"t_sim={t_sim:.2f}yr"
    print(f"  Hill HNN (JIT/analytical-Jacobian) batch: "
          f"physics dt={dt_phys:.2e} yr  T_moon_ref={T_moon_ref:.4f} yr  "
          f"n_phys={n_phys:,}  N={N}  ({mode_tag})")

    V_ref_val = FOUR_PI2 * ms_solar * mp_msun / ap_AU

    # ── System static tensor ───────────────────────────────────────────────────
    _sys_raw_static = np.column_stack([
        np.full(N, ms_solar),
        np.full(N, mp_earth),
        mm_per_cell,
        np.full(N, ap_AU),
        np.full(N, rhill_AU),
    ])
    sys_raw_static_t = torch.tensor(_sys_raw_static, dtype=F32, device=dev)
    mm_msun_t        = torch.tensor(mm_msun_arr,     dtype=F64, device=dev).unsqueeze(1)

    # ── State tensors ──────────────────────────────────────────────────────────
    q_ms = torch.tensor(pos_ms0, dtype=F64, device=dev)
    q_mp = torch.tensor(pos_mp0, dtype=F64, device=dev)
    q_mm = torch.tensor(pos_mm0, dtype=F64, device=dev)
    v_ms = torch.tensor(vel_ms0, dtype=F64, device=dev)
    v_mp = torch.tensor(vel_mp0, dtype=F64, device=dev)
    v_mm = torch.tensor(vel_mm0, dtype=F64, device=dev)

    # ── Initial velocity calibration (analytical Jacobian, before loop) ────────
    def _cross3(a, b):
        return torch.stack([
            a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
            a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
            a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0],
        ], dim=1)

    with torch.no_grad():
        r_sp0    = q_mp - q_ms
        v_sp0    = v_mp - v_ms
        r_sq0    = (r_sp0 * r_sp0).sum(1).clamp(min=1e-20)
        r_n0     = r_sq0.sqrt().unsqueeze(1)
        x_h0     = r_sp0 / r_n0
        L0       = _cross3(r_sp0, v_sp0)
        L_n0     = (L0 * L0).sum(1).clamp(min=1e-20).sqrt().unsqueeze(1)
        z_h0     = L0 / L_n0
        Omega0   = L_n0.squeeze(1) / r_sq0
        y_h0     = _cross3(z_h0, x_h0)
        r_mp0    = q_mm - q_mp
        q_h_AU0  = torch.stack([(r_mp0 * x_h0).sum(1),
                                 (r_mp0 * y_h0).sum(1),
                                 (r_mp0 * z_h0).sum(1)], dim=1)
        q_h_n0   = q_h_AU0 / rhill_AU
        Om0_f32  = Omega0.float().unsqueeze(1)
        sys_r0   = torch.cat([sys_raw_static_t, Om0_f32], dim=1)
        sys_e0   = (sys_r0.clamp(min=1e-20).log() - scaler_mean) / scaler_std
        grad0    = _analytical_grad_3layer(
            q_h_n0.float(), sys_e0, W1, b1, W2, b2, W3, b3, w4, W1_q
        )
        a_ch0    = -grad0.double() * V_ref_val / (mm_msun_t * rhill_AU)
        ach_mag0 = a_ch0.norm(dim=1)
        rmp0_mag = r_mp0.norm(dim=1)
        v_circ0  = torch.where(
            (ach_mag0 > 1e-30) & (rmp0_mag > 1e-30),
            (rmp0_mag * ach_mag0).sqrt(),
            torch.zeros(N, dtype=F64, device=dev),
        )
        v_rel0     = v_mm - v_mp
        v_rel0_mag = v_rel0.norm(dim=1).clamp(min=1e-30)
        v_rel0_dir = v_rel0 / v_rel0_mag.unsqueeze(1)
        ok_cal     = (v_circ0 > 1e-30) & (v_rel0_mag > 1e-30)
        v_mm_cal   = v_mp + v_circ0.unsqueeze(1) * v_rel0_dir
        v_mm       = torch.where(ok_cal.unsqueeze(1), v_mm_cal, v_mm)

    # ── Stopping mask (ineligible cells pre-stopped) ────────────────────────────
    stopped   = torch.zeros(N, 1, dtype=torch.bool,  device=dev)
    elapsed_t = torch.zeros(N, 1, dtype=F64,          device=dev)
    if eligible_mask is not None:
        inelig  = torch.tensor(
            ~np.asarray(eligible_mask, dtype=bool).reshape(N, 1),
            dtype=torch.bool, device=dev,
        )
        stopped = stopped | inelig

    # ── Pre-allocate output buffers on device (written inside JIT) ─────────────
    mpd_out = torch.zeros(N, n_out,    dtype=F32, device=dev)
    msd_out = torch.zeros(N, n_out,    dtype=F32, device=dev)
    traj_mp = torch.zeros(N, n_out, 3, dtype=F32, device=dev)
    traj_ms = torch.zeros(N, n_out, 3, dtype=F32, device=dev)
    traj_mm = torch.zeros(N, n_out, 3, dtype=F32, device=dev)

    # ── JIT-compiled KDK loop ──────────────────────────────────────────────────
    print(f"  Launching JIT loop ({n_phys:,} steps × {N} cells)...", flush=True)
    t_loop = time.perf_counter()

    (q_ms, q_mp, q_mm, v_ms, v_mp, v_mm, stopped, elapsed_t,
     mpd_out, msd_out, traj_mp, traj_ms, traj_mm, out_idx) = kdk_loop_jit(
        q_ms, q_mp, q_mm, v_ms, v_mp, v_mm, stopped, elapsed_t,
        W1, b1, W2, b2, W3, b3, w4, W1_q,
        sys_raw_static_t, mm_msun_t, scaler_mean, scaler_std,
        float(rhill_AU), float(V_ref_val), float(mp_msun), float(ms_solar),
        _FOUR_PI2_F,
        float(dt_phys), float(dt_phys / 2.0),
        float(escape_factor), float(a_inner_au), float(a_outer_au), float(t_sim),
        int(n_phys), int(stride), int(n_out),
        mpd_out, msd_out, traj_mp, traj_ms, traj_mm,
    )

    t_loop_end = time.perf_counter()
    print(f"  JIT loop done in {t_loop_end - t_loop:.1f}s", flush=True)

    # ── To numpy ───────────────────────────────────────────────────────────────
    mpd_np      = mpd_out[:, :out_idx].cpu().numpy()
    msd_np      = msd_out[:, :out_idx].cpu().numpy()
    traj_mp_np  = traj_mp[:, :out_idx, :].cpu().numpy()
    traj_ms_np  = traj_ms[:, :out_idx, :].cpu().numpy()
    traj_mm_np  = traj_mm[:, :out_idx, :].cpu().numpy()

    # ── Classification ──────────────────────────────────────────────────────────
    w         = min(_WARMUP_STEPS, out_idx - 1)
    mpd_post  = mpd_np[:, w:]
    msd_post  = msd_np[:, w:]

    map_stable    = (mpd_post.max(axis=1) <= escape_factor * rhill_AU)
    map_habitable = ((msd_post.min(axis=1) >= a_inner_au) &
                     (msd_post.max(axis=1) <= a_outer_au))
    map_both      = map_stable & map_habitable

    is_unstable_out      = mpd_np > escape_factor * rhill_AU
    is_uninhabitable_out = (msd_np < a_inner_au) | (msd_np > a_outer_au)
    both_bad_out         = is_unstable_out & is_uninhabitable_out
    has_stop             = both_bad_out.any(axis=1)
    stop_step            = np.where(
        has_stop, both_bad_out.argmax(axis=1), -1
    ).astype(np.int32)

    initially_habitable = (
        (msd_np[:, 0] >= a_inner_au) & (msd_np[:, 0] <= a_outer_au)
    )

    map_stable    = map_stable.reshape(mm_resolution, am_resolution)
    map_habitable = map_habitable.reshape(mm_resolution, am_resolution)
    map_both      = map_both.reshape(mm_resolution, am_resolution)

    if eligible_mask is not None:
        em_2d = np.asarray(eligible_mask, dtype=bool).reshape(mm_resolution, am_resolution)
        map_stable    &= em_2d
        map_habitable &= em_2d
        map_both      &= em_2d

    valid_mm_range  = None
    valid_am_per_mm = []
    valid_rows = np.where(map_both.any(axis=1))[0]
    if len(valid_rows):
        valid_mm_range = [float(mm_grid[valid_rows[0]]),
                         float(mm_grid[valid_rows[-1]])]
    for i_mm in range(mm_resolution):
        cols = np.where(map_both[i_mm])[0]
        valid_am_per_mm.append(
            [float(am_grid[cols[0]]), float(am_grid[cols[-1]])] if len(cols) else None
        )

    elapsed = time.perf_counter() - t0

    return {
        "ok":               True,
        "traj_planet":      traj_mp_np,
        "traj_star":        traj_ms_np,
        "traj_moon":        traj_mm_np,
        "t_grid":           np.linspace(0.0, t_sim, out_idx),
        "moon_planet_dist": mpd_np,
        "map_stable":       map_stable.tolist(),
        "map_habitable":    map_habitable.tolist(),
        "map_both":         map_both.tolist(),
        "stop_step":        stop_step,
        "initially_habitable": initially_habitable,
        "mm_grid":          mm_grid.tolist(),
        "am_grid":          am_grid.tolist(),
        "valid_mm_range":   valid_mm_range,
        "valid_am_per_mm":  valid_am_per_mm,
        "rhill_AU":         float(rhill_AU),
        "a_inner_au":       float(a_inner_au),
        "a_outer_au":       float(a_outer_au),
        "n_phys":           n_phys,
        "dt_phys":          dt_phys,
        "n_out":            out_idx,
        "elapsed_s":        elapsed,
    }
