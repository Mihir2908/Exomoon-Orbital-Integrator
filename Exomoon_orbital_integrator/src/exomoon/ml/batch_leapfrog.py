"""
exomoon/ml/batch_leapfrog.py — Batched PyTorch 3-body leapfrog for trajectory preview.

Reimplements the same symplectic leapfrog from integrator.py as batched
torch.Tensor operations, processing all N = (mm_res × am_res) grid cells
simultaneously via BLAS matrix multiply rather than @njit sequential loops.

Speed source: each leapfrog step is a handful of (N,3) tensor operations that
map to a single BLAS SGEMM call — O(N) work executed in one vectorised kernel
rather than N sequential Python/Numba function calls.

Physics: identical to integrator._leapfrog_integrate().  The same 3-body
velocity-Verlet (leapfrog) scheme conserves the same modified Hamiltonian H̃.
No ML approximation is involved — this is exact gravitational mechanics.

CRITICAL ISOLATION — this file has ZERO interaction with MLP infrastructure:
  • No imports from exomoon.ml.dataset / .model / .inference
  • No read/write of eval_aux_mlp_output/, models/, models_temphead/, any .pt
  • MLP training columns, weights, and training pipeline are completely untouched
"""

from __future__ import annotations

import time
import numpy as np
import torch

from exomoon.constants import au, msun, merth, rsun as _rsun, FOUR_PI2
from exomoon.habitable_zone import hz_bounds_au

_PLANET_DENSITY_CGS = 5.5     # g/cm³ — matches run_ml_dataset.py default
_PLANET_DENSITY_SI  = 5500.0  # kg/m³
_MOON_DENSITY_CGS   = 3.0     # g/cm³ — rocky moon Roche limit assumption
_WARMUP_STEPS       = 5       # steps excluded from ever_escaped accumulation


# ── Batched gravitational acceleration ─────────────────────────────────────────

def _accel_batch(
    pos_a: torch.Tensor,   # (N, 3)  position of body being accelerated
    pos_b: torch.Tensor,   # (N, 3)  position of attracting body
    mu_b,                   # scalar float OR (N, 1) tensor — GM of attracting body
) -> torch.Tensor:          # (N, 3)
    """
    Vectorised gravitational acceleration: a_A = -μ_B * r̂ / |r|²
    where r = pos_a − pos_b.  Broadcasting handles both scalar and per-cell μ.
    Equivalent to integrator._accel() extended across N bodies simultaneously.
    """
    r  = pos_a - pos_b                          # (N, 3)
    r2 = (r * r).sum(dim=1, keepdim=True)       # (N, 1)  |r|²
    r3 = r2.sqrt() * r2                         # (N, 1)  |r|³
    return -mu_b * r / r3                       # (N, 3)  acceleration


# ── Vectorised initial state generation ────────────────────────────────────────

def _build_initial_states(
    ms_solar: float,
    mp_earth: float,
    ap_AU: float,
    ep: float,
    em: float,
    moon_retrograde: bool,
    mm_grid: np.ndarray,   # (mm_res,) M_earth values
    am_grid: np.ndarray,   # (am_res,) Hill-radii fractions
) -> tuple:
    """
    Vectorised equivalent of initial_conditions.initial_state(), covering all
    N = mm_res × am_res (mm_earth, am_hill) cells without a Python loop.

    Ordering: cell index k = i * am_res + j  maps to mm_grid[i], am_grid[j].

    Returns
    -------
    pos_mp, pos_ms, pos_mm  (N, 3) float64 — initial positions (AU)
    vel_mp, vel_ms, vel_mm  (N, 3) float64 — initial velocities (AU/yr)
    mu_ms                   float           — star grav param (FOUR_PI2 units)
    mu_mp                   float           — planet grav param
    mu_mm_arr               (N,) float64    — per-cell moon grav param
    rhill                   float           — Hill radius (AU), same for all cells
    """
    ms_gp = ms_solar * FOUR_PI2
    mp_gp = mp_earth * (merth / msun) * FOUR_PI2
    mm_gp = mm_grid  * (merth / msun) * FOUR_PI2   # (mm_res,)

    # Hill radius — depends only on ms/mp/ap/ep which are fixed across the grid
    rhill = ap_AU * (1.0 - ep) * (mp_gp / (3.0 * ms_gp)) ** (1.0 / 3.0)

    mm_res, am_res = len(mm_grid), len(am_grid)

    # Broadcast to (mm_res, am_res)
    mm_v  = mm_gp[:, None]                       # (mm_res, 1)
    am_AU = am_grid[None, :] * rhill             # (1,   am_res)  moon semi-major axis

    mp_mm   = mp_gp + mm_v                       # (mm_res, 1)  planet+moon mass
    M_total = ms_gp + mp_mm                      # (mm_res, 1)  total system mass

    # Planet-moon barycenter position and velocity in system frame
    xpm  = ap_AU * (1.0 - ep) * ms_gp / M_total          # (mm_res, 1)
    vypm = ms_gp / np.sqrt((1.0 - ep) * M_total * ap_AU) # (mm_res, 1)

    # Star (periapsis start, same pattern as initial_conditions.py)
    xs  = -ap_AU * (1.0 - ep) * mp_mm / M_total           # (mm_res, 1)
    vys = -mp_mm / np.sqrt((1.0 - ep) * M_total * ap_AU)  # (mm_res, 1)

    # Planet — displacement from PM barycenter, then shift to system bary
    xp  = xpm + (-am_AU * (1.0 - em) * mm_v / mp_mm)      # (mm_res, am_res)
    vyp = -mm_v / np.sqrt((1.0 - em) * mp_mm * am_AU)     # (mm_res, am_res)

    # Moon — displacement from PM barycenter, then shift to system bary
    xm  = xpm + (am_AU * (1.0 - em) * mp_gp / mp_mm)      # (mm_res, am_res)
    vym = mp_gp / np.sqrt((1.0 - em) * mp_mm * am_AU)     # (mm_res, am_res)

    # Retrograde: flip tangential velocity sign relative to PM barycenter
    dir_sign = -1.0 if moon_retrograde else 1.0
    vyp = dir_sign * vyp + vypm   # add PM barycenter velocity
    vym = dir_sign * vym + vypm

    # Broadcast (mm_res, 1) terms to (mm_res, am_res) then flatten
    xs_flat  = np.broadcast_to(xs,   (mm_res, am_res)).copy().ravel()
    vys_flat = np.broadcast_to(vys,  (mm_res, am_res)).copy().ravel()

    N = mm_res * am_res
    z = np.zeros(N, dtype=np.float64)

    pos_mp = np.stack([xp.ravel(),  z, z], axis=-1)
    pos_ms = np.stack([xs_flat,     z, z], axis=-1)
    pos_mm = np.stack([xm.ravel(),  z, z], axis=-1)
    vel_mp = np.stack([z, vyp.ravel(), z], axis=-1)
    vel_ms = np.stack([z, vys_flat,    z], axis=-1)
    vel_mm = np.stack([z, vym.ravel(), z], axis=-1)

    # mu_mm varies along mm axis only (am_hill does not affect moon mass)
    mu_mm_arr = np.repeat(mm_gp, am_res)   # (N,)

    return (pos_mp, pos_ms, pos_mm, vel_mp, vel_ms, vel_mm,
            ms_gp, mp_gp, mu_mm_arr, rhill)


# ── Main entry point ────────────────────────────────────────────────────────────

def batch_leapfrog_trajectories(
    system_params:  dict,
    t_sim:          float,
    moon_retrograde: bool           = False,
    em:             float           = 0.0,
    mm_resolution:  int             = 50,
    am_resolution:  int             = 50,
    n_steps:        int             = 1000,
    escape_factor:  float           = 1.0,
    device:         str             = "cpu",
    n_orbits:       "int | None"    = None,
    eligible_mask:  "np.ndarray | None" = None,
    mm_grid_override: "np.ndarray | None" = None,
    am_grid_override: "np.ndarray | None" = None,
) -> dict:
    """
    Run the 3-body leapfrog integrator in batched PyTorch mode over a
    (mm_earth × am_hill) grid, producing trajectories and stability maps.

    Parameters
    ----------
    system_params : dict — keys: ms_solar, rs_solar, Ts, mp_earth, ap_AU, ep
    t_sim         : float — total simulation duration (years)
    n_steps       : int   — number of OUTPUT frames stored (spatial preview resolution)
    escape_factor : float — moon escapes when moon-planet dist > escape_factor × rhill

    Timestep selection
    ------------------
    Identical formula to run_simulation_for_years: dt = min(T_moon/100, 1/20000),
    evaluated at the median am_hill as a single shared dt for all batch cells.
    The Numba integrator uses per-cell T_moon; here one dt is shared across all
    2500 cells simultaneously.  n_phys = ceil(t_sim / dt) — ranges from ~4 000
    for wide-orbit systems to 200 000+ for compact systems (e.g. TRAPPIST-1).
    Outputs are stride-subsampled to n_steps frames, mirroring _resample_traj().

    Returns
    -------
    dict:
      ok              : bool
      traj_planet     : ndarray (N, n_out, 3) float32 — planet positions (AU)
      traj_star       : ndarray (N, n_out, 3) float32 — star positions (AU)
      traj_moon       : ndarray (N, n_out, 3) float32 — moon positions (AU)
      t_grid          : ndarray (n_out,) float64       — time axis (years)
      moon_planet_dist: ndarray (N, n_out) float32     — ||moon-planet|| AU
      moon_star_dist  : ndarray (N, n_out) float32     — ||moon-star|| AU
      map_stable      : list[list[bool]] shape (mm_res, am_res)
      map_habitable   : list[list[bool]] shape (mm_res, am_res)
      map_both        : list[list[bool]] shape (mm_res, am_res)
      mm_grid         : list[float]                       — M_earth values
      am_grid         : list[float]                       — Hill-radii fractions
      valid_mm_range  : [min, max] or None
      valid_am_per_mm : list of [min, max] or None per mm row
      rhill_AU        : float
      a_inner_au      : float
      a_outer_au      : float
      n_phys          : int   — actual number of physics integration steps used
      dt_phys         : float — physics timestep (years)
      elapsed_s       : float
    """
    t0 = time.perf_counter()

    # ── System parameters ──────────────────────────────────────────────────
    ms_solar = float(system_params["ms_solar"])
    rs_solar = float(system_params.get("rs_solar", 1.0))
    Ts       = float(system_params.get("Ts", 5772.0))
    mp_earth = float(system_params["mp_earth"])
    ap_AU    = float(system_params["ap_AU"])
    ep       = float(system_params.get("ep", 0.0))

    # ── Grid construction — identical to inference.py ──────────────────────
    mp_kg   = mp_earth * merth
    rp_m    = (0.75 * mp_kg / (np.pi * _PLANET_DENSITY_SI)) ** (1.0 / 3.0)
    a_roche = 2.456 * rp_m * (_PLANET_DENSITY_CGS / _MOON_DENSITY_CGS) ** (1.0 / 3.0) / au

    ms_gp_tmp = ms_solar * FOUR_PI2
    mp_gp_tmp = mp_earth * (merth / msun) * FOUR_PI2
    rhill_AU  = ap_AU * (1.0 - ep) * (mp_gp_tmp / (3.0 * ms_gp_tmp)) ** (1.0 / 3.0)

    am_min   = max(a_roche / rhill_AU, 1e-3)
    mm_max   = min(mp_earth, 3.0)

    mm_grid  = np.exp(np.linspace(np.log(0.107), np.log(mm_max), mm_resolution))
    am_grid  = np.linspace(am_min, 1.0, am_resolution)

    # Per-cell overrides — used by /trajectory/cell_preview for single-cell trajectories
    if mm_grid_override is not None:
        mm_grid = np.asarray(mm_grid_override, dtype=np.float64)
        mm_resolution = len(mm_grid)
    if am_grid_override is not None:
        am_grid = np.asarray(am_grid_override, dtype=np.float64)
        am_resolution = len(am_grid)

    a_inner_au, a_outer_au = hz_bounds_au(Ts, rs_solar * _rsun)

    # ── Vectorised initial states ──────────────────────────────────────────
    (pos_mp0, pos_ms0, pos_mm0, vel_mp0, vel_ms0, vel_mm0,
     mu_ms, mu_mp, mu_mm_arr, rhill) = _build_initial_states(
        ms_solar=ms_solar, mp_earth=mp_earth,
        ap_AU=ap_AU, ep=ep, em=em,
        moon_retrograde=moon_retrograde,
        mm_grid=mm_grid, am_grid=am_grid,
    )

    N = mm_resolution * am_resolution

    # ── Move to torch float64 (matches integrator.py precision) ───────────
    dev   = torch.device(device)
    dtype = torch.float64

    p_mp = torch.tensor(pos_mp0, dtype=dtype, device=dev)
    p_ms = torch.tensor(pos_ms0, dtype=dtype, device=dev)
    p_mm = torch.tensor(pos_mm0, dtype=dtype, device=dev)
    v_mp = torch.tensor(vel_mp0, dtype=dtype, device=dev)
    v_ms = torch.tensor(vel_ms0, dtype=dtype, device=dev)
    v_mm = torch.tensor(vel_mm0, dtype=dtype, device=dev)

    # Per-cell moon mu: (N, 1) for broadcasting in _accel_batch
    mu_mm_t = torch.tensor(mu_mm_arr[:, None], dtype=dtype, device=dev)

    # ── Shared physics timestep: dt = min(T_moon_ref/100, 1/20000) ─────────
    # Uses the median (am_hill, mm) cell's moon period — identical formula to
    # run_simulation_for_years() / integrator.py.  One shared dt for all N
    # cells.  n_orbits overrides t_sim duration only, not the timestep.
    # Option D (per-cell K=16) was removed 2026-09-12 — it produced
    # GT stable=0.000/0.005 on K-1229b vs expected 0.484. See
    # batch_leapfrog_option_d_backup.py for the removed code.
    am_ref_AU  = float(am_grid[am_resolution // 2]) * rhill_AU
    mm_mid     = float(mm_grid[mm_resolution // 2])
    mu_mm_ref  = mm_mid * (merth / msun) * FOUR_PI2
    T_moon_ref = 2.0 * np.pi * np.sqrt(am_ref_AU**3 / (mu_mp + mu_mm_ref))
    if n_orbits is not None:
        t_sim = float(n_orbits) * T_moon_ref
    dt_fixed   = min(T_moon_ref / 100.0, 1.0 / 20_000.0)
    n_phys     = max(int(np.ceil(t_sim / dt_fixed)), n_steps)
    stride     = max(1, n_phys // n_steps)
    n_out      = n_phys // stride

    dt_t      = torch.full((N, 1), dt_fixed, dtype=dtype, device=dev)
    half_dt_t = dt_t * 0.5
    print(f"  GT batch leapfrog: fixed dt={dt_fixed:.2e} yr  n_phys={n_phys:,}  N={N}")

    # Stopping-criterion scalars (same escape logic as hnn_inference_hill.py)
    rhill_tf   = torch.tensor(rhill_AU,   dtype=dtype, device=dev)
    a_inner_tf = torch.tensor(a_inner_au, dtype=dtype, device=dev)
    a_outer_tf = torch.tensor(a_outer_au, dtype=dtype, device=dev)

    # ── Pre-allocate trajectory storage (float32 — half memory vs float64) ─
    traj_mp_t = torch.empty(N, n_out, 3, dtype=torch.float32, device=dev)
    traj_ms_t = torch.empty(N, n_out, 3, dtype=torch.float32, device=dev)
    traj_mm_t = torch.empty(N, n_out, 3, dtype=torch.float32, device=dev)

    # Inline distance accumulation avoids a separate O(N×n_out×3) post-pass
    mpd_t = torch.empty(N, n_out, dtype=torch.float32, device=dev)
    msd_t = torch.empty(N, n_out, dtype=torch.float32, device=dev)

    # ── Integration loop (symplectic velocity-Verlet / leapfrog) ──────────
    with torch.no_grad():
        elapsed_t = torch.zeros(N, 1, dtype=dtype, device=dev)
        done_t    = torch.zeros(N, 1, dtype=torch.bool, device=dev)
        # Ineligible cells (MLP-rejected) start as done — never integrated
        if eligible_mask is not None:
            inelig = torch.tensor(
                ~np.asarray(eligible_mask, dtype=bool).reshape(N, 1),
                dtype=torch.bool, device=dev,
            )
            done_t = done_t | inelig

        out_idx = 0
        for i in range(n_phys):
            active = (~done_t).to(dtype)   # (N,1): 0.0 for done/ineligible cells

            # Planet (KDK leapfrog; frozen for done/ineligible cells via active mask)
            p2_mp = p_mp + v_mp * (half_dt_t * active)
            a_mp  = (_accel_batch(p2_mp, p_ms, mu_ms)
                   + _accel_batch(p2_mp, p_mm, mu_mm_t))
            v_mp  = v_mp + a_mp * (dt_t * active)
            p_mp  = p2_mp + v_mp * (half_dt_t * active)

            # Star
            p2_ms = p_ms + v_ms * (half_dt_t * active)
            a_ms  = (_accel_batch(p2_ms, p_mp, mu_mp)
                   + _accel_batch(p2_ms, p_mm, mu_mm_t))
            v_ms  = v_ms + a_ms * (dt_t * active)
            p_ms  = p2_ms + v_ms * (half_dt_t * active)

            # Moon
            p2_mm = p_mm + v_mm * (half_dt_t * active)
            a_mm  = (_accel_batch(p2_mm, p_mp, mu_mp)
                   + _accel_batch(p2_mm, p_ms, mu_ms))
            v_mm  = v_mm + a_mm * (dt_t * active)
            p_mm  = p2_mm + v_mm * (half_dt_t * active)

            # Compute distances at every step for stopping criterion + output
            diff_mp = p_mm - p_mp                                             # (N,3) float64
            diff_ms = p_mm - p_ms
            mpd_s = (diff_mp * diff_mp).sum(dim=1, keepdim=True).sqrt()      # (N,1) float64
            msd_s = (diff_ms * diff_ms).sum(dim=1, keepdim=True).sqrt()

            if i % stride == 0 and out_idx < n_out:
                traj_mp_t[:, out_idx] = p_mp.float()
                traj_ms_t[:, out_idx] = p_ms.float()
                traj_mm_t[:, out_idx] = p_mm.float()
                mpd_t[:, out_idx] = mpd_s.squeeze(1).float()
                msd_t[:, out_idx] = msd_s.squeeze(1).float()
                out_idx += 1

            # Stop when BOTH unstable AND uninhabitable, OR elapsed time reached
            elapsed_t += dt_t * active
            unstable      = mpd_s > (escape_factor * rhill_tf)
            uninhabitable = (msd_s < a_inner_tf) | (msd_s > a_outer_tf)
            done_t = done_t | (unstable & uninhabitable) | (elapsed_t >= t_sim)
            if done_t.all():
                break

    # ── Convert to numpy then trim to filled frames ────────────────────────
    # .cpu() required when device="cuda" — numpy() only works on CPU tensors.
    # Trim AFTER numpy(): torch slices are non-contiguous and .numpy() requires
    # a contiguous tensor; numpy slices are always safe.
    n_out   = out_idx
    traj_mp = traj_mp_t.cpu().numpy()[:, :n_out]   # (N, n_out, 3) float32
    traj_ms = traj_ms_t.cpu().numpy()[:, :n_out]
    traj_mm = traj_mm_t.cpu().numpy()[:, :n_out]
    mpd     = mpd_t.cpu().numpy()[:, :n_out]        # (N, n_out) float32
    msd     = msd_t.cpu().numpy()[:, :n_out]

    # ── Stability and habitability maps ────────────────────────────────────
    # Exclude warm-up window (same rationale as inference.py warm-up exclusion)
    w = min(_WARMUP_STEPS, n_out - 1)
    mpd_post = mpd[:, w:]
    msd_post = msd[:, w:]

    map_stable    = (mpd_post.max(axis=1) <= escape_factor * rhill_AU)  # (N,)
    map_habitable = ((msd_post.min(axis=1) >= a_inner_au) &
                     (msd_post.max(axis=1) <= a_outer_au))               # (N,)
    map_both      = map_stable & map_habitable

    map_stable    = map_stable.reshape(mm_resolution, am_resolution)
    map_habitable = map_habitable.reshape(mm_resolution, am_resolution)
    map_both      = map_both.reshape(mm_resolution, am_resolution)

    # Force ineligible cells to False — their frozen initial positions would
    # otherwise appear as stable+habitable (initial mpd = am_AU ≤ rhill, initial
    # msd ≈ ap_AU ∈ HZ for in-HZ systems).
    if eligible_mask is not None:
        em_2d = np.asarray(eligible_mask, dtype=bool).reshape(mm_resolution, am_resolution)
        map_stable    &= em_2d
        map_habitable &= em_2d
        map_both      &= em_2d

    # Valid ranges (mirrors inference.py)
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
        "dt_phys":          dt_fixed,
        "n_out":            n_out,
        "elapsed_s":        elapsed,
    }
