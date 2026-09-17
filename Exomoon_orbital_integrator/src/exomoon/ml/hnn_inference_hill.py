"""
exomoon/ml/hnn_inference_hill.py — Batched trajectory integration using the Hill-frame HNN.

Provides batch_hnn_hill_trajectories(), a direct analogue of batch_hnn_trajectories()
from hnn_inference.py, but using HNN_Hill forces instead of the distance-based HNN.

All (mm_resolution × am_resolution) grid cells are integrated simultaneously.
At each leapfrog step:
  - Per-cell Hill frames: each cell's own star-planet vector builds its own
    (x_hat, y_hat, z_hat, Omega). All (N, 3) operations are element-wise.
  - Moon Hill-frame positions q_h_norm projected per cell.
  - HNN_Hill forward pass + autograd runs once over the full (N, 3) batch.
  - Centrifugal stripping and rotation back to inertial are fully vectorised.

Timestep: physics dt = min(T_moon_ref/100, 1/20_000), matching the GT integrator.
T_moon_ref is computed from the median (mm, am) cell; dt is uniform for all cells.
HNN force F = -dV/dq_h_norm is position-based — no dt dependence — so running at
physics dt integrates the same learned force field with higher accuracy than parquet dt.
n_phys = max(ceil(t_sim / dt_phys), n_steps); n_steps is a floor, not the actual step count.

sys_enc column layout (must match hnn_dataset_hill.py):
  [ms_solar, mp_earth, mm_earth, ap_AU, rhill_AU, Omega_yr] → log → StandardScaler

ISOLATION: does not touch models/, models_temphead/, eval_aux_mlp_output/,
           models_force_mlp/, models_hnn_dist_v12/.
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

from exomoon.constants            import FOUR_PI2, au, merth, msun
from exomoon.constants            import rsun as _rsun
from exomoon.habitable_zone       import hz_bounds_au
from exomoon.ml.batch_leapfrog    import _build_initial_states
from exomoon.ml.hnn_dataset_hill  import load_hill_sys_scaler, _MERTH_OVER_MSUN
from exomoon.ml.hnn_model_hill    import HNN_Hill, is_hill_dir

_PLANET_DENSITY_CGS = 5.5
_PLANET_DENSITY_SI  = 5500.0
_MOON_DENSITY_CGS   = 3.0
_WARMUP_STEPS       = 5
_STEPS_PER_YEAR     = 20_000.0


def batch_hnn_hill_trajectories(
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
    mm_grid_override: "np.ndarray | None" = None,
    am_grid_override: "np.ndarray | None" = None,
) -> dict:
    """
    Run Hill-frame HNN leapfrog for all (mm_earth x am_hill) cells simultaneously.

    Timestep: physics dt = min(T_moon_ref/100, 1/20_000). ωdt < 2 for all cells.

    n_orbits (optional): when given, overrides t_sim with n_orbits × T_moon_ref.
      This is the system-agnostic mode: always simulates N complete moon orbital
      periods of the reference (median am) cell regardless of calendar years.
      Physics dt is unchanged — ωdt ≈ 0.063 for reference cell, ≤ 2 for all cells.
      n_phys = N_orbits × 100 (100 steps per orbit at physics dt).
      Speedup vs 10yr: 200 000 / (N_orbits × 100).  N_orbits=100 → 20× speedup.

    Returns same dict structure as batch_leapfrog_trajectories() — drop-in compatible,
    including traj_planet, traj_star, traj_moon (N, n_out, 3) position arrays.
    Returns {"ok": False, "error": "no_model"} if checkpoint not found.
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

    # Precompute scaler as device tensors so _hill_forces stays on-device.
    # StandardScaler.transform(x) = (x - mean_) / scale_
    _scaler_mean = torch.tensor(hill_scaler.mean_,  dtype=F32, device=dev)  # (6,)
    _scaler_std  = torch.tensor(hill_scaler.scale_, dtype=F32, device=dev)  # (6,)

    # ── Grid ──────────────────────────────────────────────────────────────────
    mp_kg   = mp_earth * merth
    rp_m    = (0.75 * mp_kg / (np.pi * _PLANET_DENSITY_SI)) ** (1.0 / 3.0)
    a_roche = 2.456 * rp_m * (_PLANET_DENSITY_CGS / _MOON_DENSITY_CGS) ** (1.0 / 3.0) / au

    mp_msun  = mp_earth * _MERTH_OVER_MSUN         # planet mass in solar masses
    ms_gp    = ms_solar * FOUR_PI2                  # star grav param [AU3/yr2]
    mp_gp    = mp_msun  * FOUR_PI2                  # planet grav param
    rhill_AU = ap_AU * (1.0 - ep) * (mp_gp / (3.0 * ms_gp)) ** (1.0 / 3.0)

    am_min  = max(a_roche / rhill_AU, 1e-3)
    mm_max  = min(mp_earth, 3.0)

    mm_grid = np.exp(np.linspace(np.log(0.107), np.log(mm_max), mm_resolution))
    am_grid = np.linspace(am_min, 1.0, am_resolution)

    # Per-cell overrides — used by /trajectory/cell_preview for single-cell trajectories
    if mm_grid_override is not None:
        mm_grid = np.asarray(mm_grid_override, dtype=np.float64)
        mm_resolution = len(mm_grid)
    if am_grid_override is not None:
        am_grid = np.asarray(am_grid_override, dtype=np.float64)
        am_resolution = len(am_grid)

    a_inner_au, a_outer_au = hz_bounds_au(Ts, rs_solar * _rsun)

    # ── Initial states (N = mm_res x am_res cells) ───────────────────────────
    (pos_mp0, pos_ms0, pos_mm0, vel_mp0, vel_ms0, vel_mm0,
     mu_ms, mu_mp, _mu_mm_arr, _rhill_check) = _build_initial_states(
        ms_solar=ms_solar, mp_earth=mp_earth,
        ap_AU=ap_AU, ep=ep, em=em,
        moon_retrograde=moon_retrograde,
        mm_grid=mm_grid, am_grid=am_grid,
    )

    N   = mm_resolution * am_resolution

    # mm_earth per cell: cell k = i*am_res + j -> mm_grid[i]
    mm_per_cell  = np.repeat(mm_grid, am_resolution)         # (N,) M_earth
    mm_msun_arr  = mm_per_cell * _MERTH_OVER_MSUN            # (N,) M_sun

    # ── Physics dt: min(T_moon_ref/100, 1/20_000) — matches GT integrator ────────
    # Uniform scalar dt for all cells; derived from the median (am, mm) cell.
    # HNN force F = -dV/dq is position-based — no mathematical dt dependence.
    # Physics dt ensures wdt << 2 (leapfrog stability) for every cell including
    # inner moons that would violate wdt < 2 at the coarser parquet dt.
    am_ref_AU  = float(am_grid[am_resolution // 2]) * rhill_AU
    mm_mid     = float(mm_grid[mm_resolution // 2])
    mu_mm_ref  = mm_mid * _MERTH_OVER_MSUN * FOUR_PI2
    T_moon_ref = 2.0 * np.pi * np.sqrt(am_ref_AU ** 3 / (mu_mp + mu_mm_ref))
    if n_orbits is not None:
        t_sim = float(n_orbits) * T_moon_ref
    dt_phys    = min(T_moon_ref / 100.0, 1.0 / 20000.0)       # scalar [yr]
    n_phys     = max(int(np.ceil(t_sim / dt_phys)), n_steps)
    stride     = max(1, n_phys // n_steps)
    n_out      = n_phys // stride

    dt_vec      = torch.full((N, 1), dt_phys, dtype=F64, device=dev)    # (N,1)
    half_dt_vec = dt_vec * 0.5
    mode_tag = f"n_orbits={n_orbits}" if n_orbits is not None else f"t_sim={t_sim:.2f}yr"
    print(f"  Hill HNN batch: physics dt={dt_phys:.2e} yr  T_moon_ref={T_moon_ref:.4f} yr  "
          f"n_phys={n_phys:,}  N={N}  ({mode_tag})")

    # V_ref: dominant energy scale [M_sun * AU2 / yr2]
    V_ref_val = FOUR_PI2 * ms_solar * mp_msun / ap_AU

    # Static part of sys_enc [ms_solar, mp_earth, mm_earth, ap_AU, rhill_AU]
    # Omega_yr appended per-step inside _hill_forces.
    _sys_raw_static = np.column_stack([
        np.full(N, ms_solar),
        np.full(N, mp_earth),
        mm_per_cell,
        np.full(N, ap_AU),
        np.full(N, rhill_AU),
    ])   # (N, 5) constant throughout integration

    # Device tensors precomputed once — _hill_forces accesses these via closure.
    _sys_raw_static_t = torch.tensor(_sys_raw_static, dtype=F32, device=dev)  # (N, 5)
    _mm_msun_t = torch.tensor(mm_msun_arr, dtype=F64, device=dev).unsqueeze(1)  # (N, 1)

    # ── State tensors ─────────────────────────────────────────────────────────
    q_ms = torch.tensor(pos_ms0, dtype=F64, device=dev)  # (N, 3)
    q_mp = torch.tensor(pos_mp0, dtype=F64, device=dev)
    q_mm = torch.tensor(pos_mm0, dtype=F64, device=dev)
    v_ms = torch.tensor(vel_ms0, dtype=F64, device=dev)
    v_mp = torch.tensor(vel_mp0, dtype=F64, device=dev)
    v_mm = torch.tensor(vel_mm0, dtype=F64, device=dev)

    # ── Batched Hill-frame force computation (fully on-device, no CPU bouncing) ─
    def _cross3(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Batch cross product for (N, 3) tensors, same dtype/device as inputs."""
        return torch.stack([
            a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
            a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
            a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0],
        ], dim=1)

    def _hill_forces(q_ms_t, q_mp_t, q_mm_t, v_ms_t, v_mp_t):
        """
        Returns (a_s, a_p, a_m) as F64 tensors of shape (N, 3).
        All operations stay on dev — no CPU↔device transfers.
        """
        # Star→planet vectors (F64)
        r_sp = q_mp_t - q_ms_t                               # (N, 3) [AU]
        v_sp = v_mp_t - v_ms_t                               # (N, 3) [AU/yr]

        # |r_sp|² and |r_sp|; clamp squared to avoid division by zero
        r_sp_sq1  = (r_sp * r_sp).sum(1).clamp(min=1e-20)    # (N,) [AU²]
        r_sp_n    = r_sp_sq1.sqrt().unsqueeze(1)              # (N, 1) [AU]
        x_hat     = r_sp / r_sp_n                             # (N, 3) radial unit

        # Angular momentum L = r_sp × v_sp
        L_sp  = _cross3(r_sp, v_sp)                           # (N, 3)
        L_n2  = (L_sp * L_sp).sum(1).clamp(min=1e-20)        # (N,)
        L_n   = L_n2.sqrt().unsqueeze(1)                      # (N, 1)
        z_hat = L_sp / L_n                                    # (N, 3) orbital normal
        Omega = L_n.squeeze(1) / r_sp_sq1                     # (N,) [rad/yr]
        y_hat = _cross3(z_hat, x_hat)                         # (N, 3) tangential

        # Moon position in per-cell Hill frame (F64)
        r_mp  = q_mm_t - q_mp_t                               # (N, 3) [AU]
        q_h_x = (r_mp * x_hat).sum(1)                         # (N,)
        q_h_y = (r_mp * y_hat).sum(1)
        q_h_z = (r_mp * z_hat).sum(1)
        q_h_AU = torch.stack([q_h_x, q_h_y, q_h_z], dim=1)   # (N, 3) [AU]
        q_h_n  = q_h_AU / rhill_AU                            # (N, 3) dimensionless

        # sys_enc: concat static cols (F32) + Omega (→ F32), log-standardise
        sys_raw = torch.cat([_sys_raw_static_t,
                             Omega.to(F32).unsqueeze(1)], dim=1)   # (N, 6)
        sys_log = sys_raw.clamp(min=1e-20).log()                    # (N, 6)
        sys_enc = (sys_log - _scaler_mean) / _scaler_std            # (N, 6)

        # HNN_Hill forward + autograd (F32 for model, result converted to F64)
        q_h_f32  = q_h_n.to(F32).detach().requires_grad_(True)     # (N, 3)
        with torch.enable_grad():
            V_theta = hill_model(q_h_f32, sys_enc)                  # (N,)
            grad    = torch.autograd.grad(
                V_theta.sum(), q_h_f32, create_graph=False
            )[0]                                                      # (N, 3)

        # Conservative Hill accel: a_cons_h = −∇V × V_ref / (mm × rhill)
        a_cons_h = (-grad.to(F64) * V_ref_val
                    / (_mm_msun_t * rhill_AU))                       # (N, 3) [AU/yr²]

        # Strip centrifugal: cf = Ω² × (q_h_x, q_h_y, 0)
        omega2    = (Omega ** 2).unsqueeze(1)                        # (N, 1)
        q_h_plane = torch.cat([q_h_AU[:, :2],
                               torch.zeros(N, 1, dtype=F64, device=dev)], dim=1)
        cf        = omega2 * q_h_plane                               # (N, 3)
        a_hill    = a_cons_h - cf                                    # (N, 3) in Hill frame

        # Rotate to inertial
        a_rel_in = (a_hill[:, 0:1] * x_hat +
                    a_hill[:, 1:2] * y_hat +
                    a_hill[:, 2:3] * z_hat)                          # (N, 3)

        # Planet accel from star
        r_sp_cu  = r_sp_sq1.unsqueeze(1) ** 1.5                     # (N, 1) = |r_sp|³
        a_p      = -FOUR_PI2 * ms_solar * r_sp / r_sp_cu            # (N, 3)

        # Moon inertial accel
        a_m = a_rel_in + a_p                                         # (N, 3)

        # Star accel: from moon + from planet
        r_sm    = q_mm_t - q_ms_t                                    # (N, 3)
        d_sm_cu = ((r_sm * r_sm).sum(1, keepdim=True)) ** 1.5       # (N, 1) = |r_sm|³
        a_s     = (FOUR_PI2 * _mm_msun_t / d_sm_cu * r_sm
                   + FOUR_PI2 * mp_msun * r_sp / r_sp_cu)           # (N, 3)

        return a_s, a_p, a_m

    # ── Initial velocity calibration (fully on-device) ────────────────────────
    # Compute per-cell HNN circular speed at t=0 and rescale v_mm to match,
    # keeping the Newtonian velocity direction.  Costs one batched forward pass.
    r_sp0     = q_mp - q_ms                                              # (N, 3) F64
    v_sp0     = v_mp - v_ms                                              # (N, 3) F64
    r_sp_sq0  = (r_sp0 * r_sp0).sum(1).clamp(min=1e-20)                 # (N,)
    r_sp_n0   = r_sp_sq0.sqrt().unsqueeze(1)                             # (N, 1)
    x_h0      = r_sp0 / r_sp_n0                                          # (N, 3)
    L0        = _cross3(r_sp0, v_sp0)                                    # (N, 3)
    L_n0      = (L0 * L0).sum(1).clamp(min=1e-20).sqrt().unsqueeze(1)   # (N, 1)
    z_h0      = L0 / L_n0                                                # (N, 3)
    Omega0    = L_n0.squeeze(1) / r_sp_sq0                               # (N,)
    y_h0      = _cross3(z_h0, x_h0)                                      # (N, 3)
    r_mp0     = q_mm - q_mp                                              # (N, 3)
    q_h_AU0   = torch.stack([(r_mp0 * x_h0).sum(1),
                              (r_mp0 * y_h0).sum(1),
                              (r_mp0 * z_h0).sum(1)], dim=1)             # (N, 3)
    q_h_n0    = q_h_AU0 / rhill_AU                                       # (N, 3)
    sys_raw0  = torch.cat([_sys_raw_static_t,
                           Omega0.to(F32).unsqueeze(1)], dim=1)          # (N, 6)
    sys_enc0  = (sys_raw0.clamp(min=1e-20).log() - _scaler_mean) / _scaler_std
    q_h_lf0   = q_h_n0.to(F32).detach().requires_grad_(True)            # (N, 3)
    with torch.enable_grad():
        V0    = hill_model(q_h_lf0, sys_enc0)
        grad0 = torch.autograd.grad(V0.sum(), q_h_lf0, create_graph=False)[0]
    a_ch0     = (-grad0.to(F64) * V_ref_val
                 / (_mm_msun_t * rhill_AU))                               # (N, 3) Hill
    ach_mag0  = a_ch0.norm(dim=1)                                         # (N,)
    r_pm0_mag = r_mp0.norm(dim=1)                                         # (N,)
    v_circ0   = torch.where(
        (ach_mag0 > 1e-30) & (r_pm0_mag > 1e-30),
        (r_pm0_mag * ach_mag0).sqrt(),
        torch.zeros(N, dtype=F64, device=dev),
    )                                                                      # (N,)
    v_rel0     = v_mm - v_mp                                              # (N, 3)
    v_rel0_mag = v_rel0.norm(dim=1).clamp(min=1e-30)                      # (N,)
    v_rel0_dir = v_rel0 / v_rel0_mag.unsqueeze(1)                         # (N, 3)
    ok_cal     = (v_circ0 > 1e-30) & (v_rel0_mag > 1e-30)                # (N,) bool
    v_mm_cal   = v_mp + v_circ0.unsqueeze(1) * v_rel0_dir                 # (N, 3)
    v_mm       = torch.where(ok_cal.unsqueeze(1), v_mm_cal, v_mm)         # (N, 3)

    # ── Output buffers ─────────────────────────────────────────────────────────
    mpd_out     = np.empty((N, n_out),    dtype=np.float32)
    msd_out     = np.empty((N, n_out),    dtype=np.float32)
    traj_mp_out = np.empty((N, n_out, 3), dtype=np.float32)   # planet positions
    traj_ms_out = np.empty((N, n_out, 3), dtype=np.float32)   # star positions
    traj_mm_out = np.empty((N, n_out, 3), dtype=np.float32)   # moon positions

    # ── KDK leapfrog ──────────────────────────────────────────────────────────
    # Per-cell stop mask: cell i stops updating when it is simultaneously
    # unstable (mpd > rhill) AND uninhabitable (msd outside HZ) — matching
    # the training-data trim criterion in hnn_dataset_hill._trim().
    # Escaped-but-habitable cells CONTINUE integrating (hard constraint).
    a_inner_t = torch.tensor(a_inner_au, dtype=F64, device=dev)
    a_outer_t = torch.tensor(a_outer_au, dtype=F64, device=dev)
    rhill_t   = torch.tensor(rhill_AU,   dtype=F64, device=dev)
    stopped = torch.zeros(N, 1, dtype=torch.bool, device=dev)
    # Ineligible cells (MLP-rejected) start as stopped — never integrated
    if eligible_mask is not None:
        inelig = torch.tensor(
            ~np.asarray(eligible_mask, dtype=bool).reshape(N, 1),
            dtype=torch.bool, device=dev,
        )
        stopped = stopped | inelig

    _LOG_EVERY = 50_000   # print progress every this many physics steps
    _wall_t0  = time.perf_counter()

    with torch.no_grad():
        elapsed_t = torch.zeros(N, 1, dtype=F64, device=dev)        # per-cell elapsed time
        a_s, a_p, a_m = _hill_forces(q_ms, q_mp, q_mm, v_ms, v_mp)

        out_idx = 0
        for step in range(n_phys):
            # Half kick (zero for stopped/completed cells)
            mask = (~stopped).to(F64)                          # (N, 1) 0.0 or 1.0
            v_ms = v_ms + a_s * (half_dt_vec * mask)
            v_mp = v_mp + a_p * (half_dt_vec * mask)
            v_mm = v_mm + a_m * (half_dt_vec * mask)
            # Full drift
            q_ms = q_ms + v_ms * (dt_vec * mask)
            q_mp = q_mp + v_mp * (dt_vec * mask)
            q_mm = q_mm + v_mm * (dt_vec * mask)
            # Forces at new positions
            a_s, a_p, a_m = _hill_forces(q_ms, q_mp, q_mm, v_ms, v_mp)
            # Half kick
            v_ms = v_ms + a_s * (half_dt_vec * mask)
            v_mp = v_mp + a_p * (half_dt_vec * mask)
            v_mm = v_mm + a_m * (half_dt_vec * mask)

            # Record distances
            diff_mp = (q_mm - q_mp)
            diff_ms = (q_mm - q_ms)
            mpd_col = (diff_mp * diff_mp).sum(1, keepdim=True).sqrt()   # (N, 1)
            msd_col = (diff_ms * diff_ms).sum(1, keepdim=True).sqrt()   # (N, 1)

            if step > 0 and step % _LOG_EVERY == 0:
                _wall_el = time.perf_counter() - _wall_t0
                _pct     = 100.0 * step / n_phys
                _eta_s   = _wall_el / step * (n_phys - step)
                _n_act   = int((~stopped).sum().item())
                print(f"    step {step:,}/{n_phys:,} ({_pct:.0f}%)  "
                      f"active={_n_act}/{N}  "
                      f"elapsed={_wall_el:.0f}s  ETA={_eta_s:.0f}s", flush=True)

            if step % stride == 0 and out_idx < n_out:
                mpd_out[:, out_idx]        = mpd_col.squeeze(1).float().cpu().numpy()
                msd_out[:, out_idx]        = msd_col.squeeze(1).float().cpu().numpy()
                traj_mp_out[:, out_idx, :] = q_mp.float().cpu().numpy()
                traj_ms_out[:, out_idx, :] = q_ms.float().cpu().numpy()
                traj_mm_out[:, out_idx, :] = q_mm.float().cpu().numpy()
                out_idx += 1

            # Advance per-cell elapsed time (only active cells moved this step)
            elapsed_t = elapsed_t + dt_vec * mask
            # Stop when BOTH unstable AND uninhabitable, OR when cell has
            # reached t_sim. Escaped-but-habitable cells are NOT stopped.
            unstable      = mpd_col > (escape_factor * rhill_t)
            uninhabitable = (msd_col < a_inner_t) | (msd_col > a_outer_t)
            stopped = stopped | (unstable & uninhabitable) | (elapsed_t >= t_sim)

            if stopped.all():
                break

    # Trim output to actually filled frames (early-break may leave trailing zeros)
    mpd_out = mpd_out[:, :out_idx]
    msd_out = msd_out[:, :out_idx]

    # ── Classification ─────────────────────────────────────────────────────────
    w         = min(_WARMUP_STEPS, out_idx - 1)
    mpd_post  = mpd_out[:, w:]
    msd_post  = msd_out[:, w:]

    map_stable    = (mpd_post.max(axis=1) <= escape_factor * rhill_AU)
    map_habitable = ((msd_post.min(axis=1) >= a_inner_au) &
                     (msd_post.max(axis=1) <= a_outer_au))
    map_both = map_stable & map_habitable

    # ── stop_step and initially_habitable (matching hnn_inference.py return schema)
    is_unstable_out      = mpd_out > escape_factor * rhill_AU
    is_uninhabitable_out = (msd_out < a_inner_au) | (msd_out > a_outer_au)
    both_bad_out         = is_unstable_out & is_uninhabitable_out          # (N, n_out_actual)
    has_stop             = both_bad_out.any(axis=1)
    stop_step            = np.where(
        has_stop, both_bad_out.argmax(axis=1), -1
    ).astype(np.int32)

    initially_habitable = (
        (msd_out[:, 0] >= a_inner_au) & (msd_out[:, 0] <= a_outer_au)
    )

    map_stable    = map_stable.reshape(mm_resolution, am_resolution)
    map_habitable = map_habitable.reshape(mm_resolution, am_resolution)
    map_both      = map_both.reshape(mm_resolution, am_resolution)

    # Force ineligible cells to False (same logic as batch_leapfrog)
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
        "traj_planet":      traj_mp_out[:, :out_idx, :],  # (N, n_out, 3) planet positions
        "traj_star":        traj_ms_out[:, :out_idx, :],  # (N, n_out, 3) star positions
        "traj_moon":        traj_mm_out[:, :out_idx, :],  # (N, n_out, 3) moon positions
        "t_grid":           np.linspace(0.0, t_sim, out_idx),  # (n_out,) years
        "moon_planet_dist": mpd_out,           # (N, n_frames_actual)
        "map_stable":       map_stable.tolist(),
        "map_habitable":    map_habitable.tolist(),
        "map_both":         map_both.tolist(),
        "stop_step":        stop_step,          # (N,) int32, -1 = never both-bad
        "initially_habitable": initially_habitable,  # (N,) bool
        "mm_grid":          mm_grid.tolist(),
        "am_grid":          am_grid.tolist(),
        "valid_mm_range":   valid_mm_range,
        "valid_am_per_mm":  valid_am_per_mm,
        "rhill_AU":         float(rhill_AU),
        "a_inner_au":       float(a_inner_au),
        "a_outer_au":       float(a_outer_au),
        "n_phys":           n_phys,
        "dt_phys":          dt_phys,
        "n_out":            n_out,
        "elapsed_s":        elapsed,
    }
