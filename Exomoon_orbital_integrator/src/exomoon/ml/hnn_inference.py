"""
exomoon/ml/hnn_inference.py — Batched trajectory integration using the distance-based HNN.

Provides batch_hnn_trajectories(), a drop-in sibling to batch_leapfrog_trajectories()
that uses the trained HNN potential V_θ(d_n, sys_enc) → scalar instead of explicit
Newtonian forces.  All 2,500 grid cells are integrated simultaneously in one
batched autograd pass per integration step.

Force recovery via chain rule
------------------------------
The HNN is trained with d_n = [d_sp/ap, d_sm/ap, d_pm/rhill] as inputs.
At inference, absolute positions q_abs are the leaf tensors:

    dV/dq_abs = (dV/dd_n) · (dd_n/dq_abs)    [chain rule, autograd handles this]

Multiply by V_ref = G·ms·mp/ap (M_sun·AU²/yr²) to recover forces in M_sun·AU/yr²:
    F = -(dV_θ/dq_abs) × V_ref

Then divide by body mass to get accelerations.

V_ref derivation
----------------
V_θ is dimensionless (dimensionless inputs d_n → dimensionless potential).
Its gradient ∂V_θ/∂q_abs is in units of 1/AU (differentiating w.r.t. AU positions).
Multiplying by V_ref = G·ms·mp/ap [M_sun·AU²/yr²] gives M_sun·AU/yr² = force units.
The moon interaction terms are captured inside V_θ itself (via d_sm, d_pm inputs
and mm in sys_enc); V_ref uses only the dominant star-planet energy scale as
the reference constant.

Speed model (CPU, N=2500)
--------------------------
  Physics leapfrog  :  200 000 steps × 18 small tensor ops  → ~460 s
  HNN dt_factor=10  :   20 000 steps × 1 batched fwd+autograd → ~212 s  (2.2×)
  HNN dt_factor=50  :    4 000 steps                          → ~43 s  (10.8×) ← default

ISOLATION: no imports from dataset.py, model.py, train.py, inference.py.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from exomoon.constants import FOUR_PI2, au, merth, msun
from exomoon.constants import rsun as _rsun
from exomoon.habitable_zone import hz_bounds_au

from exomoon.ml.batch_leapfrog import _build_initial_states
from exomoon.ml.hnn_dataset     import load_sys_scaler, _MERTH_OVER_MSUN
from exomoon.ml.hnn_model       import HNN

_PLANET_DENSITY_CGS = 5.5
_PLANET_DENSITY_SI  = 5500.0
_MOON_DENSITY_CGS   = 3.0
_WARMUP_STEPS       = 5


def _load_model(model_dir: str, device: str) -> tuple:
    """Load HNN model + sys_scaler. Returns (model, sys_scaler) or raises."""
    if not os.path.isfile(os.path.join(model_dir, "hnn_model.pt")):
        raise FileNotFoundError(
            f"HNN model not found at {model_dir}/hnn_model.pt — "
            "train with: py -m exomoon.ml.hnn_train --out models_hnn_dist/"
        )
    model = HNN.load(model_dir, map_location=device)
    model.eval()
    sys_scaler = load_sys_scaler(model_dir)
    return model, sys_scaler


def batch_hnn_trajectories(
    system_params:   dict,
    t_sim:           float,
    moon_retrograde: bool  = False,
    em:              float = 0.0,
    mm_resolution:   int   = 50,
    am_resolution:   int   = 50,
    n_steps:          int        = 1000,
    escape_factor:    float      = 1.0,
    model_dir:        str        = "models_hnn_dist/",
    dt_factor:        int        = 50,
    track_energy_idx: int | None = None,
    device:           str        = "cpu",
) -> dict:
    """
    Run HNN-driven leapfrog for all (mm_earth × am_hill) cells simultaneously.

    Parameters
    ----------
    system_params  : dict  — keys: ms_solar, rs_solar, Ts, mp_earth, ap_AU, ep
    t_sim          : float — total simulation duration (years)
    n_steps        : int   — number of output frames stored
    escape_factor  : float — moon escaped when mpd > escape_factor × rhill_AU
    model_dir      : str   — directory with hnn_model.pt + hnn_sys_scaler.pkl
    dt_factor      : int   — integration coarseness relative to physics leapfrog.
                             dt_factor=50 → ~10× speedup vs full physics leapfrog.

    track_energy_idx : int or None — if set, also computes H_θ (learned energy)
                       and H_true (Newtonian energy) at each output step for that
                       cell index.  Added to return dict as "energy_htrue" and
                       "energy_htheta" numpy arrays.  Diagnostic only — not used
                       in classification or stability maps.

    Returns
    -------
    Same dict structure as batch_leapfrog_trajectories() — drop-in compatible.
    Returns {"ok": False, "error": "no_model", ...} if model not found.
    """
    t0 = time.perf_counter()

    ms_solar = float(system_params["ms_solar"])
    rs_solar = float(system_params.get("rs_solar", 1.0))
    Ts       = float(system_params.get("Ts", 5772.0))
    mp_earth = float(system_params["mp_earth"])
    ap_AU    = float(system_params["ap_AU"])
    ep       = float(system_params.get("ep", 0.0))

    try:
        model, sys_scaler = _load_model(model_dir, device)
    except FileNotFoundError as e:
        return {"ok": False, "error": "no_model", "message": str(e)}

    # ── Grid construction ──────────────────────────────────────────────────────
    mp_kg   = mp_earth * merth
    rp_m    = (0.75 * mp_kg / (np.pi * _PLANET_DENSITY_SI)) ** (1.0 / 3.0)
    a_roche = 2.456 * rp_m * (_PLANET_DENSITY_CGS / _MOON_DENSITY_CGS) ** (1.0 / 3.0) / au

    ms_gp_tmp = ms_solar * FOUR_PI2
    mp_gp_tmp = mp_earth * _MERTH_OVER_MSUN * FOUR_PI2
    rhill_AU  = ap_AU * (1.0 - ep) * (mp_gp_tmp / (3.0 * ms_gp_tmp)) ** (1.0 / 3.0)

    am_min  = max(a_roche / rhill_AU, 1e-3)
    mm_max  = min(mp_earth, 3.0)

    mm_grid = np.exp(np.linspace(np.log(0.107), np.log(mm_max), mm_resolution))
    am_grid = np.linspace(am_min, 1.0, am_resolution)

    a_inner_au, a_outer_au = hz_bounds_au(Ts, rs_solar * _rsun)

    # ── Initial states ─────────────────────────────────────────────────────────
    (pos_mp0, pos_ms0, pos_mm0, vel_mp0, vel_ms0, vel_mm0,
     mu_ms, mu_mp, _mu_mm_arr, rhill) = _build_initial_states(
        ms_solar=ms_solar, mp_earth=mp_earth,
        ap_AU=ap_AU, ep=ep, em=em,
        moon_retrograde=moon_retrograde,
        mm_grid=mm_grid, am_grid=am_grid,
    )

    N   = mm_resolution * am_resolution
    dev = torch.device(device)
    F64 = torch.float64
    F32 = torch.float32

    # ── Per-cell moon mass array ───────────────────────────────────────────────
    mm_per_cell = np.repeat(mm_grid, am_resolution)   # (N,)

    # ── Physics constants as tensors ───────────────────────────────────────────
    # V_ref = G·ms·mp/ap  [M_sun·AU²/yr²]
    # This is the dominant energy scale.  The HNN captures moon contributions
    # inside V_θ itself; V_ref is only the overall unit reference.
    _ms_msun  = ms_solar
    _mp_msun  = mp_earth * _MERTH_OVER_MSUN
    V_ref_val = FOUR_PI2 * _ms_msun * _mp_msun / ap_AU   # M_sun·AU²/yr²
    V_ref_t   = torch.tensor(V_ref_val, dtype=F64, device=dev)   # scalar

    # ap and rhill as F32 scalars for distance normalisation inside autograd graph
    ap_t_f32    = torch.tensor(float(ap_AU),    dtype=F32, device=dev)
    rhill_t_f32 = torch.tensor(float(rhill_AU), dtype=F32, device=dev)

    # Per-body mass tensors for F→a conversion
    ms_t = torch.tensor(ms_solar,                        dtype=F64, device=dev)
    mp_t = torch.tensor(mp_earth * _MERTH_OVER_MSUN,    dtype=F64, device=dev)
    mm_t = torch.tensor(mm_per_cell * _MERTH_OVER_MSUN, dtype=F64, device=dev).unsqueeze(1)  # (N,1)

    # ── Adaptive timestep ──────────────────────────────────────────────────────
    am_ref_AU  = float(am_grid[am_resolution // 2]) * rhill_AU
    mm_mid     = float(mm_grid[mm_resolution // 2])
    mu_mm_ref  = mm_mid * _MERTH_OVER_MSUN * FOUR_PI2
    T_moon_ref = 2.0 * np.pi * np.sqrt(am_ref_AU ** 3 / (mu_mp + mu_mm_ref))
    dt_base    = min(T_moon_ref / 100.0, 1.0 / 20000.0)
    dt_phys    = dt_base * dt_factor
    n_phys     = int(np.ceil(t_sim / dt_phys))
    n_phys     = max(n_phys, n_steps)
    dt         = t_sim / n_phys
    half_dt    = dt * 0.5
    stride     = max(1, n_phys // n_steps)
    n_out      = n_phys // stride

    print(f"  HNN leapfrog: dt_factor={dt_factor}  dt={dt:.5f} yr  "
          f"n_phys={n_phys}  n_out={n_out}")

    # ── Per-cell sys_enc (constant throughout integration) ─────────────────────
    sys_raw = np.column_stack([
        np.full(N, ms_solar),
        np.full(N, mp_earth),
        mm_per_cell,
        np.full(N, ap_AU),
        np.full(N, rhill_AU),
    ])   # (N, 5): matches hnn_dataset._SYS_COLS
    sys_enc = torch.tensor(
        sys_scaler.transform(sys_raw).astype(np.float32),
        dtype=F32, device=dev,
    )   # (N, 5)

    # ── Initial state tensors (F64 for integration precision) ─────────────────
    q_ms = torch.tensor(pos_ms0, dtype=F64, device=dev)   # (N, 3)
    q_mp = torch.tensor(pos_mp0, dtype=F64, device=dev)
    q_mm = torch.tensor(pos_mm0, dtype=F64, device=dev)
    v_ms = torch.tensor(vel_ms0, dtype=F64, device=dev)
    v_mp = torch.tensor(vel_mp0, dtype=F64, device=dev)
    v_mm = torch.tensor(vel_mm0, dtype=F64, device=dev)

    # ── Force closure ──────────────────────────────────────────────────────────
    # q_abs_f32 is the leaf; autograd chain rule propagates ∂V/∂d_n through
    # ∂d_n/∂q_abs to give ∂V/∂q_abs.  Multiply by V_ref to get forces.
    def _compute_forces(q_ms_f64, q_mp_f64, q_mm_f64):
        """Returns (a_s, a_p, a_m) in AU/yr² as F64 tensors of shape (N, 3)."""
        with torch.enable_grad():
            # Build F32 leaf from current F64 positions
            # Body ordering: star [0:3], planet [3:6], moon [6:9]
            q_abs_f32 = torch.cat([q_ms_f64, q_mp_f64, q_mm_f64], dim=1).float()   # (N, 9)
            q_leaf    = q_abs_f32.detach().requires_grad_(True)                      # F32 leaf

            # Pair distances from q_leaf (inside graph)
            d_sp = (q_leaf[:, 3:6] - q_leaf[:, 0:3]).norm(dim=1, keepdim=True)   # (N,1) F32
            d_sm = (q_leaf[:, 6:9] - q_leaf[:, 0:3]).norm(dim=1, keepdim=True)   # (N,1) F32
            d_pm = (q_leaf[:, 6:9] - q_leaf[:, 3:6]).norm(dim=1, keepdim=True)   # (N,1) F32

            # Physics-based normalisation (same as training)
            d_n = torch.cat(
                [d_sp / ap_t_f32, d_sm / ap_t_f32, d_pm / rhill_t_f32], dim=1
            )   # (N, 3) F32

            V = model(d_n, sys_enc)   # (N,) F32

            # ∂V/∂q_abs via chain rule through distance computation
            # No create_graph needed: we don't backprop through this grad
            dV_dq = torch.autograd.grad(V.sum(), q_leaf, create_graph=False)[0]   # (N,9) F32

        # F = -dV_dq × V_ref  [M_sun·AU/yr²]  cast to F64 for numerical precision
        F_raw = -dV_dq.detach().to(F64) * V_ref_t   # (N, 9)

        a_s = F_raw[:, 0:3] / ms_t    # (N, 3)
        a_p = F_raw[:, 3:6] / mp_t    # (N, 3)
        a_m = F_raw[:, 6:9] / mm_t    # (N, 3)  mm_t is (N,1) → broadcasts correctly
        return a_s, a_p, a_m

    # ── Pre-allocate output buffers ────────────────────────────────────────────
    traj_mp_t = torch.empty(N, n_out, 3, dtype=F32, device=dev)
    traj_ms_t = torch.empty(N, n_out, 3, dtype=F32, device=dev)
    traj_mm_t = torch.empty(N, n_out, 3, dtype=F32, device=dev)
    mpd_t     = torch.empty(N, n_out, dtype=F32, device=dev)
    msd_t     = torch.empty(N, n_out, dtype=F32, device=dev)

    # ── Energy tracking setup (diagnostic, single cell) ───────────────────────
    _energy_htrue  = [] if track_energy_idx is not None else None
    _energy_htheta = [] if track_energy_idx is not None else None
    _mm_track = (float(mm_per_cell[track_energy_idx]) * _MERTH_OVER_MSUN
                 if track_energy_idx is not None else None)

    # ── Velocity-Verlet (KDK leapfrog) ────────────────────────────────────────
    # a = -∂V_θ/∂q_abs × V_ref / mass
    # v_{n+½} = v_n + a_n × dt/2
    # q_{n+1} = q_n + v_{n+½} × dt
    # a_{n+1} = _compute_forces(q_{n+1})
    # v_{n+1} = v_{n+½} + a_{n+1} × dt/2

    # torch.no_grad() governs position/velocity arithmetic; _compute_forces
    # internally uses torch.enable_grad() for its own autograd.grad call.
    with torch.no_grad():
        a_s, a_p, a_m = _compute_forces(q_ms, q_mp, q_mm)

        out_idx = 0
        for step in range(n_phys):

            # Half-kick
            v_ms = v_ms + a_s * half_dt
            v_mp = v_mp + a_p * half_dt
            v_mm = v_mm + a_m * half_dt

            # Full drift
            q_ms = q_ms + v_ms * dt
            q_mp = q_mp + v_mp * dt
            q_mm = q_mm + v_mm * dt

            # Force at new positions
            a_s, a_p, a_m = _compute_forces(q_ms, q_mp, q_mm)

            # Half-kick
            v_ms = v_ms + a_s * half_dt
            v_mp = v_mp + a_p * half_dt
            v_mm = v_mm + a_m * half_dt

            if step % stride == 0 and out_idx < n_out:
                traj_mp_t[:, out_idx] = q_mp.float()
                traj_ms_t[:, out_idx] = q_ms.float()
                traj_mm_t[:, out_idx] = q_mm.float()
                diff_mp = (q_mm - q_mp).float()
                diff_ms = (q_mm - q_ms).float()
                mpd_t[:, out_idx] = (diff_mp * diff_mp).sum(dim=1).sqrt()
                msd_t[:, out_idx] = (diff_ms * diff_ms).sum(dim=1).sqrt()

                # ── Energy tracking for one cell (diagnostic) ──────────────
                if track_energy_idx is not None:
                    _i = track_energy_idx
                    q_s_e = q_ms[_i];  q_p_e = q_mp[_i];  q_m_e = q_mm[_i]
                    v_s_e = v_ms[_i];  v_p_e = v_mp[_i];  v_m_e = v_mm[_i]

                    d_sp_e = float((q_p_e - q_s_e).norm())
                    d_sm_e = float((q_m_e - q_s_e).norm())
                    d_pm_e = float((q_m_e - q_p_e).norm())

                    # Kinetic energy T [M_sun·AU²/yr²]
                    T_e = 0.5 * (
                        float(_ms_msun * v_s_e.dot(v_s_e)) +
                        float(_mp_msun * v_p_e.dot(v_p_e)) +
                        float(_mm_track * v_m_e.dot(v_m_e))
                    )

                    # True Newtonian potential V_true = -G·Σ(mᵢmⱼ/rᵢⱼ)
                    H_true_e = T_e - FOUR_PI2 * (
                        _ms_msun * _mp_msun / d_sp_e +
                        _ms_msun * _mm_track  / d_sm_e +
                        _mp_msun * _mm_track  / d_pm_e
                    )

                    # Learned potential V_θ (model fwd pass, no grad needed)
                    d_n_e = torch.tensor(
                        [[d_sp_e / ap_AU, d_sm_e / ap_AU, d_pm_e / rhill_AU]],
                        dtype=F32, device=dev,
                    )
                    V_theta_e = float(model(d_n_e, sys_enc[_i:_i + 1]))
                    H_theta_e = T_e + V_theta_e * V_ref_val

                    _energy_htrue.append(H_true_e)
                    _energy_htheta.append(H_theta_e)

                out_idx += 1

    # ── Numpy conversion ───────────────────────────────────────────────────────
    traj_mp = traj_mp_t.numpy()
    traj_ms = traj_ms_t.numpy()
    traj_mm = traj_mm_t.numpy()
    mpd     = mpd_t.numpy()
    msd     = msd_t.numpy()

    # ── Stability and habitability maps (classification — full trajectory) ──────
    w        = min(_WARMUP_STEPS, n_out - 1)
    mpd_post = mpd[:, w:]
    msd_post = msd[:, w:]

    map_stable    = (mpd_post.max(axis=1) <= escape_factor * rhill_AU)
    map_habitable = ((msd_post.min(axis=1) >= a_inner_au) &
                     (msd_post.max(axis=1) <= a_outer_au))
    map_both      = map_stable & map_habitable

    # ── Trajectory preview stopping criteria ────────────────────────────────────
    # A cell's preview stops at the first output step where both stable=0 (moon
    # outside Hill sphere) AND habitable=0 (moon outside HZ) simultaneously.
    # This captures both orderings: escape-then-uninhabitable and vice versa.
    # stop_step[i] = -1 means neither condition combination is ever reached.
    is_unstable_out      = mpd > escape_factor * rhill_AU
    is_uninhabitable_out = (msd < a_inner_au) | (msd > a_outer_au)
    both_bad_out         = is_unstable_out & is_uninhabitable_out          # (N, n_out)
    has_stop             = both_bad_out.any(axis=1)                        # (N,)
    stop_step            = np.where(
        has_stop, both_bad_out.argmax(axis=1), -1
    ).astype(np.int32)                                                      # (N,)

    # Initial habitability: check moon_star_dist at t=0 output step
    initially_habitable = (
        (msd[:, 0] >= a_inner_au) & (msd[:, 0] <= a_outer_au)
    )   # (N,) bool — only cells with True get a trajectory preview

    map_stable    = map_stable.reshape(mm_resolution, am_resolution)
    map_habitable = map_habitable.reshape(mm_resolution, am_resolution)
    map_both      = map_both.reshape(mm_resolution, am_resolution)

    valid_mm_range  = None
    valid_am_per_mm = []
    valid_rows      = np.where(map_both.any(axis=1))[0]
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
        "traj_planet":      traj_mp,
        "traj_star":        traj_ms,
        "traj_moon":        traj_mm,
        "t_grid":           np.linspace(0.0, t_sim, n_out),
        "moon_planet_dist": mpd,
        "moon_star_dist":   msd,
        "map_stable":       map_stable.tolist(),
        "map_habitable":    map_habitable.tolist(),
        "map_both":         map_both.tolist(),
        "mm_grid":          mm_grid.tolist(),
        "am_grid":          am_grid.tolist(),
        "valid_mm_range":   valid_mm_range,
        "valid_am_per_mm":  valid_am_per_mm,
        "rhill_AU":         float(rhill_AU),
        "a_inner_au":       float(a_inner_au),
        "a_outer_au":       float(a_outer_au),
        "n_phys":           n_phys,
        "dt_phys":          dt,
        "dt_factor":        dt_factor,
        "n_out":            n_out,
        "elapsed_s":        elapsed,
        "energy_htrue":       np.array(_energy_htrue)  if _energy_htrue  is not None else None,
        "energy_htheta":      np.array(_energy_htheta) if _energy_htheta is not None else None,
        "stop_step":          stop_step,
        "initially_habitable": initially_habitable,
    }
