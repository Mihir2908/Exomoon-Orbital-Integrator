"""
exomoon/ml/hnn_inference_hill_fast.py — Analytical-Jacobian variant of hnn_inference_hill.py.

Identical physics and output to batch_hnn_hill_trajectories() in hnn_inference_hill.py.
The only change: the two torch.autograd.grad() calls are replaced with an explicit
chain-rule backward pass through the Tanh MLP, removing autograd overhead from the
inner loop.

Original file (hnn_inference_hill.py) is NOT touched.

Analytical Jacobian for a depth-L Tanh MLP with input x = [q_h_norm (3), sys_enc (6)]:
  Forward (per hidden layer i):  h[i] = tanh(W[i] @ h[i-1] + b[i])
  Backward (chain rule):
    delta = W_out.squeeze(0) * (1 - h[L-1]^2)
    for i = L-1 down to 1:
        delta = (delta @ W[i]) * (1 - h[i-1]^2)
    dV/d(q_h_norm) = delta @ W[0][:, :3]   (first 3 columns of input weight matrix)

This replaces autograd.grad() with 3 matmuls + 3 elementwise ops per step.
No requires_grad_, no enable_grad context, no backward graph — stays inside no_grad.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

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


def _extract_linears(model: nn.Module) -> list[nn.Linear]:
    """Return all nn.Linear modules in model.net in order."""
    return [m for m in model.net if isinstance(m, nn.Linear)]


def _analytical_forward_and_grad(
    q_h_f32:  torch.Tensor,   # (N, 3) F32  — q_h_norm (no requires_grad needed)
    sys_enc:  torch.Tensor,   # (N, 6) F32
    linears:  list[nn.Linear],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Manual forward pass + analytical Jacobian dV/d(q_h_norm).

    Returns (V, grad) where:
      V    : (N,)    F32  — same as hill_model(q_h_f32, sys_enc)
      grad : (N, 3) F32  — same as autograd.grad(V.sum(), q_h_f32)[0]

    Works inside torch.no_grad() — no backward graph allocated.
    """
    x = torch.cat([q_h_f32, sys_enc], dim=-1)   # (N, 9)

    # Forward: pass through each hidden layer, store activations for backward
    n_hidden = len(linears) - 1   # output layer is the last one
    hiddens: list[torch.Tensor] = []
    h = x
    for i in range(n_hidden):
        h = torch.tanh(h @ linears[i].weight.T + linears[i].bias)
        hiddens.append(h)

    # Output
    V = (hiddens[-1] @ linears[-1].weight.T + linears[-1].bias).squeeze(-1)  # (N,)

    # Analytical backward: chain rule through Tanh layers
    # delta accumulates dV/d(pre-tanh activation) layer by layer going backward
    delta = linears[-1].weight.squeeze(0) * (1.0 - hiddens[-1] ** 2)   # (N, hidden)
    for i in range(n_hidden - 1, 0, -1):
        delta = (delta @ linears[i].weight) * (1.0 - hiddens[i - 1] ** 2)

    # Gradient wrt q_h_norm (first 3 input dimensions)
    grad = delta @ linears[0].weight[:, :3]   # (N, 3)

    return V, grad


def batch_hnn_hill_trajectories_fast(
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
    Analytical-Jacobian variant of batch_hnn_hill_trajectories().

    Drop-in replacement with identical physics and output format.
    Only difference: autograd.grad() replaced with explicit chain-rule backward,
    removing overhead from 200,000 inner-loop iterations.
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

    _scaler_mean = torch.tensor(hill_scaler.mean_,  dtype=F32, device=dev)
    _scaler_std  = torch.tensor(hill_scaler.scale_, dtype=F32, device=dev)

    # Precompute Linear layers once — used by _analytical_forward_and_grad every step
    _linears = _extract_linears(hill_model)

    # ── Grid ──────────────────────────────────────────────────────────────────
    mp_kg   = mp_earth * merth
    rp_m    = (0.75 * mp_kg / (np.pi * _PLANET_DENSITY_SI)) ** (1.0 / 3.0)
    a_roche = 2.456 * rp_m * (_PLANET_DENSITY_CGS / _MOON_DENSITY_CGS) ** (1.0 / 3.0) / au

    mp_msun  = mp_earth * _MERTH_OVER_MSUN
    ms_gp    = ms_solar * FOUR_PI2
    mp_gp    = mp_msun  * FOUR_PI2
    rhill_AU = ap_AU * (1.0 - ep) * (mp_gp / (3.0 * ms_gp)) ** (1.0 / 3.0)

    am_min  = max(a_roche / rhill_AU, 1e-3)
    mm_max  = min(mp_earth, 3.0)

    mm_grid = np.exp(np.linspace(np.log(0.107), np.log(mm_max), mm_resolution))
    am_grid = np.linspace(am_min, 1.0, am_resolution)

    a_inner_au, a_outer_au = hz_bounds_au(Ts, rs_solar * _rsun)

    # ── Initial states ────────────────────────────────────────────────────────
    (pos_mp0, pos_ms0, pos_mm0, vel_mp0, vel_ms0, vel_mm0,
     mu_ms, mu_mp, _mu_mm_arr, _rhill_check) = _build_initial_states(
        ms_solar=ms_solar, mp_earth=mp_earth,
        ap_AU=ap_AU, ep=ep, em=em,
        moon_retrograde=moon_retrograde,
        mm_grid=mm_grid, am_grid=am_grid,
    )

    N   = mm_resolution * am_resolution

    mm_per_cell  = np.repeat(mm_grid, am_resolution)
    mm_msun_arr  = mm_per_cell * _MERTH_OVER_MSUN

    # ── Physics dt ────────────────────────────────────────────────────────────
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

    dt_vec      = torch.full((N, 1), dt_phys, dtype=F64, device=dev)
    half_dt_vec = dt_vec * 0.5
    mode_tag = f"n_orbits={n_orbits}" if n_orbits is not None else f"t_sim={t_sim:.2f}yr"
    print(f"  Hill HNN (fast/analytical-Jacobian) batch: "
          f"physics dt={dt_phys:.2e} yr  T_moon_ref={T_moon_ref:.4f} yr  "
          f"n_phys={n_phys:,}  N={N}  ({mode_tag})")

    V_ref_val = FOUR_PI2 * ms_solar * mp_msun / ap_AU

    _sys_raw_static = np.column_stack([
        np.full(N, ms_solar),
        np.full(N, mp_earth),
        mm_per_cell,
        np.full(N, ap_AU),
        np.full(N, rhill_AU),
    ])
    _sys_raw_static_t = torch.tensor(_sys_raw_static, dtype=F32, device=dev)
    _mm_msun_t = torch.tensor(mm_msun_arr, dtype=F64, device=dev).unsqueeze(1)

    # ── State tensors ─────────────────────────────────────────────────────────
    q_ms = torch.tensor(pos_ms0, dtype=F64, device=dev)
    q_mp = torch.tensor(pos_mp0, dtype=F64, device=dev)
    q_mm = torch.tensor(pos_mm0, dtype=F64, device=dev)
    v_ms = torch.tensor(vel_ms0, dtype=F64, device=dev)
    v_mp = torch.tensor(vel_mp0, dtype=F64, device=dev)
    v_mm = torch.tensor(vel_mm0, dtype=F64, device=dev)

    def _cross3(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.stack([
            a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
            a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
            a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0],
        ], dim=1)

    def _hill_forces_fast(q_ms_t, q_mp_t, q_mm_t, v_ms_t, v_mp_t):
        """
        Identical to _hill_forces() in hnn_inference_hill.py but uses
        _analytical_forward_and_grad() instead of autograd.grad().
        No requires_grad_, no enable_grad context — stays in no_grad.
        """
        r_sp = q_mp_t - q_ms_t
        v_sp = v_mp_t - v_ms_t

        r_sp_sq1  = (r_sp * r_sp).sum(1).clamp(min=1e-20)
        r_sp_n    = r_sp_sq1.sqrt().unsqueeze(1)
        x_hat     = r_sp / r_sp_n

        L_sp  = _cross3(r_sp, v_sp)
        L_n2  = (L_sp * L_sp).sum(1).clamp(min=1e-20)
        L_n   = L_n2.sqrt().unsqueeze(1)
        z_hat = L_sp / L_n
        Omega = L_n.squeeze(1) / r_sp_sq1
        y_hat = _cross3(z_hat, x_hat)

        r_mp  = q_mm_t - q_mp_t
        q_h_x = (r_mp * x_hat).sum(1)
        q_h_y = (r_mp * y_hat).sum(1)
        q_h_z = (r_mp * z_hat).sum(1)
        q_h_AU = torch.stack([q_h_x, q_h_y, q_h_z], dim=1)
        q_h_n  = q_h_AU / rhill_AU

        sys_raw = torch.cat([_sys_raw_static_t,
                             Omega.to(F32).unsqueeze(1)], dim=1)
        sys_log = sys_raw.clamp(min=1e-20).log()
        sys_enc = (sys_log - _scaler_mean) / _scaler_std

        # Analytical Jacobian — no autograd, no requires_grad, no enable_grad
        q_h_f32 = q_h_n.to(F32)   # (N, 3) F32, no requires_grad
        _V, grad = _analytical_forward_and_grad(q_h_f32, sys_enc, _linears)

        a_cons_h = (-grad.to(F64) * V_ref_val
                    / (_mm_msun_t * rhill_AU))

        omega2    = (Omega ** 2).unsqueeze(1)
        q_h_plane = torch.cat([q_h_AU[:, :2],
                               torch.zeros(N, 1, dtype=F64, device=dev)], dim=1)
        cf        = omega2 * q_h_plane
        a_hill    = a_cons_h - cf

        a_rel_in = (a_hill[:, 0:1] * x_hat +
                    a_hill[:, 1:2] * y_hat +
                    a_hill[:, 2:3] * z_hat)

        r_sp_cu  = r_sp_sq1.unsqueeze(1) ** 1.5
        a_p      = -FOUR_PI2 * ms_solar * r_sp / r_sp_cu
        a_m = a_rel_in + a_p

        r_sm    = q_mm_t - q_ms_t
        d_sm_cu = ((r_sm * r_sm).sum(1, keepdim=True)) ** 1.5
        a_s     = (FOUR_PI2 * _mm_msun_t / d_sm_cu * r_sm
                   + FOUR_PI2 * mp_msun * r_sp / r_sp_cu)

        return a_s, a_p, a_m

    # ── Initial velocity calibration (analytical Jacobian) ────────────────────
    r_sp0     = q_mp - q_ms
    v_sp0     = v_mp - v_ms
    r_sp_sq0  = (r_sp0 * r_sp0).sum(1).clamp(min=1e-20)
    r_sp_n0   = r_sp_sq0.sqrt().unsqueeze(1)
    x_h0      = r_sp0 / r_sp_n0
    L0        = _cross3(r_sp0, v_sp0)
    L_n0      = (L0 * L0).sum(1).clamp(min=1e-20).sqrt().unsqueeze(1)
    z_h0      = L0 / L_n0
    Omega0    = L_n0.squeeze(1) / r_sp_sq0
    y_h0      = _cross3(z_h0, x_h0)
    r_mp0     = q_mm - q_mp
    q_h_AU0   = torch.stack([(r_mp0 * x_h0).sum(1),
                              (r_mp0 * y_h0).sum(1),
                              (r_mp0 * z_h0).sum(1)], dim=1)
    q_h_n0    = q_h_AU0 / rhill_AU
    sys_raw0  = torch.cat([_sys_raw_static_t,
                           Omega0.to(F32).unsqueeze(1)], dim=1)
    sys_enc0  = (sys_raw0.clamp(min=1e-20).log() - _scaler_mean) / _scaler_std
    q_h_f32_0 = q_h_n0.to(F32)   # no requires_grad
    _V0, grad0 = _analytical_forward_and_grad(q_h_f32_0, sys_enc0, _linears)
    a_ch0     = (-grad0.to(F64) * V_ref_val
                 / (_mm_msun_t * rhill_AU))
    ach_mag0  = a_ch0.norm(dim=1)
    r_pm0_mag = r_mp0.norm(dim=1)
    v_circ0   = torch.where(
        (ach_mag0 > 1e-30) & (r_pm0_mag > 1e-30),
        (r_pm0_mag * ach_mag0).sqrt(),
        torch.zeros(N, dtype=F64, device=dev),
    )
    v_rel0     = v_mm - v_mp
    v_rel0_mag = v_rel0.norm(dim=1).clamp(min=1e-30)
    v_rel0_dir = v_rel0 / v_rel0_mag.unsqueeze(1)
    ok_cal     = (v_circ0 > 1e-30) & (v_rel0_mag > 1e-30)
    v_mm_cal   = v_mp + v_circ0.unsqueeze(1) * v_rel0_dir
    v_mm       = torch.where(ok_cal.unsqueeze(1), v_mm_cal, v_mm)

    # ── Output buffers ─────────────────────────────────────────────────────────
    mpd_out     = np.empty((N, n_out),    dtype=np.float32)
    msd_out     = np.empty((N, n_out),    dtype=np.float32)
    traj_mp_out = np.empty((N, n_out, 3), dtype=np.float32)
    traj_ms_out = np.empty((N, n_out, 3), dtype=np.float32)
    traj_mm_out = np.empty((N, n_out, 3), dtype=np.float32)

    # ── KDK leapfrog ──────────────────────────────────────────────────────────
    a_inner_t = torch.tensor(a_inner_au, dtype=F64, device=dev)
    a_outer_t = torch.tensor(a_outer_au, dtype=F64, device=dev)
    rhill_t   = torch.tensor(rhill_AU,   dtype=F64, device=dev)
    stopped = torch.zeros(N, 1, dtype=torch.bool, device=dev)
    if eligible_mask is not None:
        inelig = torch.tensor(
            ~np.asarray(eligible_mask, dtype=bool).reshape(N, 1),
            dtype=torch.bool, device=dev,
        )
        stopped = stopped | inelig

    _LOG_EVERY = 50_000
    _wall_t0  = time.perf_counter()

    with torch.no_grad():
        elapsed_t = torch.zeros(N, 1, dtype=F64, device=dev)
        a_s, a_p, a_m = _hill_forces_fast(q_ms, q_mp, q_mm, v_ms, v_mp)

        out_idx = 0
        for step in range(n_phys):
            mask = (~stopped).to(F64)
            v_ms = v_ms + a_s * (half_dt_vec * mask)
            v_mp = v_mp + a_p * (half_dt_vec * mask)
            v_mm = v_mm + a_m * (half_dt_vec * mask)
            q_ms = q_ms + v_ms * (dt_vec * mask)
            q_mp = q_mp + v_mp * (dt_vec * mask)
            q_mm = q_mm + v_mm * (dt_vec * mask)
            a_s, a_p, a_m = _hill_forces_fast(q_ms, q_mp, q_mm, v_ms, v_mp)
            v_ms = v_ms + a_s * (half_dt_vec * mask)
            v_mp = v_mp + a_p * (half_dt_vec * mask)
            v_mm = v_mm + a_m * (half_dt_vec * mask)

            diff_mp = (q_mm - q_mp)
            diff_ms = (q_mm - q_ms)
            mpd_col = (diff_mp * diff_mp).sum(1, keepdim=True).sqrt()
            msd_col = (diff_ms * diff_ms).sum(1, keepdim=True).sqrt()

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

            elapsed_t = elapsed_t + dt_vec * mask
            unstable      = mpd_col > (escape_factor * rhill_t)
            uninhabitable = (msd_col < a_inner_t) | (msd_col > a_outer_t)
            stopped = stopped | (unstable & uninhabitable) | (elapsed_t >= t_sim)

            if stopped.all():
                break

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

    is_unstable_out      = mpd_out > escape_factor * rhill_AU
    is_uninhabitable_out = (msd_out < a_inner_au) | (msd_out > a_outer_au)
    both_bad_out         = is_unstable_out & is_uninhabitable_out
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
        "traj_planet":      traj_mp_out[:, :out_idx, :],
        "traj_star":        traj_ms_out[:, :out_idx, :],
        "traj_moon":        traj_mm_out[:, :out_idx, :],
        "t_grid":           np.linspace(0.0, t_sim, out_idx),
        "moon_planet_dist": mpd_out,
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
        "n_out":            n_out,
        "elapsed_s":        elapsed,
    }
