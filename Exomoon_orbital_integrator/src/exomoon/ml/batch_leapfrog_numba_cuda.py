"""
exomoon/ml/batch_leapfrog_numba_cuda.py — Numba CUDA 3-body leapfrog.

Identical physics and interface to batch_leapfrog_trajectories() in
batch_leapfrog.py, but the integration loop runs entirely on-device via a
Numba CUDA @cuda.jit kernel.  One CUDA thread per grid cell → zero Python
iterations → zero GPU→CPU sync overhead during integration.

Produces BOTH map outputs (stable/habitable per cell) AND full trajectory
arrays (traj_planet, traj_star, traj_moon) shaped (N, n_out, 3).  Positions
are stored at evenly-spaced stride intervals during integration so that the
n_out output frames span the full simulation duration.

Timing scope (t0) is identical to batch_leapfrog_trajectories():
  t0 covers IC computation for ALL N cells + kernel launch + kernel
  completion (cuda.synchronize()) + result post-processing.

Physics equivalence:
  - Same dt formula: dt = min(T_moon_ref/100, 1/20000) from median cell
  - Same n_phys: max(ceil(t_sim/dt), n_steps)
  - Same warmup: _WARMUP_STEPS steps excluded from stability checks
  - Same stopping criterion: break when instantaneous mpd > threshold
    AND instantaneous msd outside HZ (mirrors done_t logic in batch_leapfrog.py)
  - Same output maps: map_stable = never exceeded rhill; map_habitable = always in HZ
"""

from __future__ import annotations

import math
import os
import time
import numpy as np

# ── NVVM library resolution (must happen before any numba.cuda import) ─────────
# The pytorch Docker container has CUDA runtime but not the compiler toolkit.
# nvidia-cuda-nvcc-cu12==12.4.131 (pip) installs libnvvm for the matching version.
#
# Numba 0.67 find_lib() uses regex libnvvm\.so\.[0-9]+ — it matches versioned
# names (libnvvm.so.4) but NOT the bare libnvvm.so that pip installs.
# Fix: create a versioned symlink in the same directory so numba can find it.
def _resolve_nvvm() -> None:
    import glob as _glob

    # ── Step 1: find libnvvm.so from pip-installed cuda_nvcc package ─────────
    _nvvm_candidates = [
        "/opt/conda/lib/python3.11/site-packages/nvidia/cuda_nvcc/nvvm/lib64/libnvvm.so",
        "/opt/conda/lib/python3.10/site-packages/nvidia/cuda_nvcc/nvvm/lib64/libnvvm.so",
        "/usr/local/cuda/nvvm/lib64/libnvvm.so",
        "/usr/local/cuda-12.4/nvvm/lib64/libnvvm.so",
    ]
    _nvvm_candidates += _glob.glob(
        "/opt/conda/lib/python3.*/site-packages/nvidia/cuda_nvcc/nvvm/lib64/libnvvm.so"
    )
    _nvvm_path = None
    for _path in _nvvm_candidates:
        if os.path.exists(_path):
            _nvvm_path = _path
            break

    # ── Step 2: create versioned symlink if bare .so exists but .so.X does not ─
    # Numba's find_lib() requires libnvvm.so.X — create the symlink so it matches.
    if _nvvm_path:
        _versioned = _nvvm_path + ".4"
        if not os.path.exists(_versioned):
            try:
                os.symlink(_nvvm_path, _versioned)
            except OSError:
                pass  # read-only fs or already exists race — fall through

    # ── Step 3: set CUDA_HOME so numba can locate nvvm/lib64 and libdevice ────
    if "CUDA_HOME" not in os.environ and "CUDA_PATH" not in os.environ:
        _home_candidates = [
            "/opt/conda/lib/python3.11/site-packages/nvidia/cuda_nvcc",
            "/opt/conda/lib/python3.10/site-packages/nvidia/cuda_nvcc",
            "/usr/local/cuda",
            "/usr/local/cuda-12.4",
        ]
        _home_candidates += _glob.glob(
            "/opt/conda/lib/python3.*/site-packages/nvidia/cuda_nvcc"
        )
        for _home in _home_candidates:
            if os.path.exists(
                os.path.join(_home, "nvvm", "libdevice", "libdevice.10.bc")
            ):
                os.environ["CUDA_HOME"] = _home
                break

_resolve_nvvm()

from numba import cuda

from exomoon.constants import au, msun, merth, rsun as _rsun, FOUR_PI2
from exomoon.habitable_zone import hz_bounds_au
from exomoon.ml.batch_leapfrog import _build_initial_states, _WARMUP_STEPS

_PLANET_DENSITY_CGS = 5.5
_PLANET_DENSITY_SI  = 5500.0
_MOON_DENSITY_CGS   = 3.0

_THREADS_PER_BLOCK  = 32   # one warp per block; N=900 fits in ceil(900/32)=29 blocks


@cuda.jit
def _leapfrog_3body_kernel(
    pos_mp, pos_ms, pos_mm,    # (N, 3) float64 initial positions (AU)
    vel_mp, vel_ms, vel_mm,    # (N, 3) float64 initial velocities (AU/yr)
    mu_mm_arr,                  # (N,)   float64 per-cell moon grav param
    mu_ms_val, mu_mp_val,       # float64 scalars: star and planet grav params
    dt, t_sim,                  # float64 scalars: physics timestep, total duration
    rhill, escape_factor,       # float64 scalars: Hill radius, escape multiplier
    a_inner, a_outer,           # float64 scalars: HZ bounds (AU)
    n_warmup, n_steps,          # int scalars: warmup steps, total physics steps
    stride,                     # int scalar: store trajectory every `stride` steps
    n_out,                      # int scalar: number of output frames
    out_stable, out_habitable,  # (N,) uint8 output arrays (1=yes, 0=no)
    out_traj_p,                 # (N, n_out, 3) float64 planet trajectory
    out_traj_s,                 # (N, n_out, 3) float64 star trajectory
    out_traj_m,                 # (N, n_out, 3) float64 moon trajectory
):
    """
    3-body KDK leapfrog for one (mm, am) grid cell per CUDA thread.

    Each thread runs n_steps leapfrog steps independently on-device.
    Accumulates ever_escaped / ever_uninhabitable flags post-warmup and
    writes final stable/habitable booleans to output arrays.
    Also stores planet/star/moon positions every `stride` steps into the
    trajectory output arrays for cell-click previews.
    """
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx >= pos_mp.shape[0]:
        return

    # Load initial state into thread-local registers
    px_p = pos_mp[idx, 0]; py_p = pos_mp[idx, 1]; pz_p = pos_mp[idx, 2]
    px_s = pos_ms[idx, 0]; py_s = pos_ms[idx, 1]; pz_s = pos_ms[idx, 2]
    px_m = pos_mm[idx, 0]; py_m = pos_mm[idx, 1]; pz_m = pos_mm[idx, 2]

    vx_p = vel_mp[idx, 0]; vy_p = vel_mp[idx, 1]; vz_p = vel_mp[idx, 2]
    vx_s = vel_ms[idx, 0]; vy_s = vel_ms[idx, 1]; vz_s = vel_ms[idx, 2]
    vx_m = vel_mm[idx, 0]; vy_m = vel_mm[idx, 1]; vz_m = vel_mm[idx, 2]

    mu_mm   = mu_mm_arr[idx]
    half_dt = dt * 0.5
    esc_sq  = (escape_factor * rhill) * (escape_factor * rhill)

    ever_escaped       = False
    ever_uninhabitable = False

    # Store initial positions (frame 0)
    out_traj_p[idx, 0, 0] = px_p; out_traj_p[idx, 0, 1] = py_p; out_traj_p[idx, 0, 2] = pz_p
    out_traj_s[idx, 0, 0] = px_s; out_traj_s[idx, 0, 1] = py_s; out_traj_s[idx, 0, 2] = pz_s
    out_traj_m[idx, 0, 0] = px_m; out_traj_m[idx, 0, 1] = py_m; out_traj_m[idx, 0, 2] = pz_m

    frame_idx = 1  # next frame to write (frame 0 already written above)

    for step in range(n_steps):

        # ── Planet KDK leapfrog ───────────────────────────────────────────
        p2x_p = px_p + vx_p * half_dt
        p2y_p = py_p + vy_p * half_dt
        p2z_p = pz_p + vz_p * half_dt

        # accel on planet from star
        dx = p2x_p - px_s; dy = p2y_p - py_s; dz = p2z_p - pz_s
        r2 = dx*dx + dy*dy + dz*dz
        r3 = r2 * math.sqrt(r2)
        ax_p = -mu_ms_val * dx / r3
        ay_p = -mu_ms_val * dy / r3
        az_p = -mu_ms_val * dz / r3

        # accel on planet from moon
        dx = p2x_p - px_m; dy = p2y_p - py_m; dz = p2z_p - pz_m
        r2 = dx*dx + dy*dy + dz*dz
        r3 = r2 * math.sqrt(r2)
        ax_p += -mu_mm * dx / r3
        ay_p += -mu_mm * dy / r3
        az_p += -mu_mm * dz / r3

        vx_p += ax_p * dt; vy_p += ay_p * dt; vz_p += az_p * dt
        px_p = p2x_p + vx_p * half_dt
        py_p = p2y_p + vy_p * half_dt
        pz_p = p2z_p + vz_p * half_dt

        # ── Star KDK leapfrog ─────────────────────────────────────────────
        p2x_s = px_s + vx_s * half_dt
        p2y_s = py_s + vy_s * half_dt
        p2z_s = pz_s + vz_s * half_dt

        # accel on star from planet
        dx = p2x_s - px_p; dy = p2y_s - py_p; dz = p2z_s - pz_p
        r2 = dx*dx + dy*dy + dz*dz
        r3 = r2 * math.sqrt(r2)
        ax_s = -mu_mp_val * dx / r3
        ay_s = -mu_mp_val * dy / r3
        az_s = -mu_mp_val * dz / r3

        # accel on star from moon
        dx = p2x_s - px_m; dy = p2y_s - py_m; dz = p2z_s - pz_m
        r2 = dx*dx + dy*dy + dz*dz
        r3 = r2 * math.sqrt(r2)
        ax_s += -mu_mm * dx / r3
        ay_s += -mu_mm * dy / r3
        az_s += -mu_mm * dz / r3

        vx_s += ax_s * dt; vy_s += ay_s * dt; vz_s += az_s * dt
        px_s = p2x_s + vx_s * half_dt
        py_s = p2y_s + vy_s * half_dt
        pz_s = p2z_s + vz_s * half_dt

        # ── Moon KDK leapfrog ─────────────────────────────────────────────
        p2x_m = px_m + vx_m * half_dt
        p2y_m = py_m + vy_m * half_dt
        p2z_m = pz_m + vz_m * half_dt

        # accel on moon from planet
        dx = p2x_m - px_p; dy = p2y_m - py_p; dz = p2z_m - pz_p
        r2 = dx*dx + dy*dy + dz*dz
        r3 = r2 * math.sqrt(r2)
        ax_m = -mu_mp_val * dx / r3
        ay_m = -mu_mp_val * dy / r3
        az_m = -mu_mp_val * dz / r3

        # accel on moon from star
        dx = p2x_m - px_s; dy = p2y_m - py_s; dz = p2z_m - pz_s
        r2 = dx*dx + dy*dy + dz*dz
        r3 = r2 * math.sqrt(r2)
        ax_m += -mu_ms_val * dx / r3
        ay_m += -mu_ms_val * dy / r3
        az_m += -mu_ms_val * dz / r3

        vx_m += ax_m * dt; vy_m += ay_m * dt; vz_m += az_m * dt
        px_m = p2x_m + vx_m * half_dt
        py_m = p2y_m + vy_m * half_dt
        pz_m = p2z_m + vz_m * half_dt

        # ── Store trajectory frame at stride intervals ─────────────────────
        if stride > 0 and (step + 1) % stride == 0 and frame_idx < n_out:
            out_traj_p[idx, frame_idx, 0] = px_p
            out_traj_p[idx, frame_idx, 1] = py_p
            out_traj_p[idx, frame_idx, 2] = pz_p
            out_traj_s[idx, frame_idx, 0] = px_s
            out_traj_s[idx, frame_idx, 1] = py_s
            out_traj_s[idx, frame_idx, 2] = pz_s
            out_traj_m[idx, frame_idx, 0] = px_m
            out_traj_m[idx, frame_idx, 1] = py_m
            out_traj_m[idx, frame_idx, 2] = pz_m
            frame_idx += 1

        # ── Distances ─────────────────────────────────────────────────────
        dx = px_m - px_p; dy = py_m - py_p; dz = pz_m - pz_p
        mpd_sq = dx*dx + dy*dy + dz*dz

        dx = px_m - px_s; dy = py_m - py_s; dz = pz_m - pz_s
        msd = math.sqrt(dx*dx + dy*dy + dz*dz)

        # Instantaneous escape / habitability flags
        cur_escaped       = mpd_sq > esc_sq
        cur_uninhabitable = (msd < a_inner) or (msd > a_outer)

        # Accumulate post-warmup stability history
        if step >= n_warmup:
            if cur_escaped:
                ever_escaped = True
            if cur_uninhabitable:
                ever_uninhabitable = True

        # Early stop: instantaneous both-conditions
        if cur_escaped and cur_uninhabitable:
            break

    out_stable[idx]    = 0 if ever_escaped else 1
    out_habitable[idx] = 0 if ever_uninhabitable else 1


def batch_leapfrog_numba_trajectories(
    system_params:   dict,
    t_sim:           float,
    moon_retrograde: bool            = False,
    em:              float           = 0.0,
    mm_resolution:   int             = 50,
    am_resolution:   int             = 50,
    n_steps:         int             = 5000,   # output frames stored per cell
    escape_factor:   float           = 1.0,
    device:          str             = "cuda",
    n_orbits: "int | None"           = None,
    eligible_mask: "np.ndarray | None" = None,
) -> dict:
    """
    Run the 3-body leapfrog integrator via Numba CUDA over a (mm_earth × am_hill)
    grid, returning stability/habitability maps AND full trajectory arrays.

    Drop-in replacement for batch_leapfrog_trajectories() with identical physics
    parameters, grid construction, dt formula, warmup, and stopping criterion.

    Parameters
    ----------
    n_steps : int
        Number of output trajectory frames to store per cell (default 5000).
        The physics integration runs for max(ceil(t_sim/dt), n_steps) steps;
        positions are stored at stride = n_phys // n_steps intervals so exactly
        n_steps frames are written across the full simulation duration.

    Returns
    -------
    dict with same keys as batch_leapfrog_trajectories, including:
      traj_planet / traj_star / traj_moon : np.ndarray shape (N, n_steps, 3)
      t_grid                              : np.ndarray shape (n_steps,)
    """
    t0 = time.perf_counter()

    # ── System parameters ──────────────────────────────────────────────────
    ms_solar = float(system_params["ms_solar"])
    rs_solar = float(system_params.get("rs_solar", 1.0))
    Ts       = float(system_params.get("Ts", 5772.0))
    mp_earth = float(system_params["mp_earth"])
    ap_AU    = float(system_params["ap_AU"])
    ep       = float(system_params.get("ep", 0.0))

    # ── Grid construction — identical to batch_leapfrog_trajectories ───────
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

    # ── Shared physics timestep ────────────────────────────────────────────
    am_ref_AU  = float(am_grid[am_resolution // 2]) * rhill_AU
    mm_mid     = float(mm_grid[mm_resolution // 2])
    mu_mm_ref  = mm_mid * (merth / msun) * FOUR_PI2
    T_moon_ref = 2.0 * np.pi * np.sqrt(am_ref_AU**3 / (mu_mp + mu_mm_ref))
    if n_orbits is not None:
        t_sim = float(n_orbits) * T_moon_ref
    dt_fixed = min(T_moon_ref / 100.0, 1.0 / 20_000.0)
    n_phys   = max(int(np.ceil(t_sim / dt_fixed)), n_steps)

    # stride: store one frame every `stride` physics steps → exactly n_steps frames
    stride = max(n_phys // n_steps, 1)
    n_out  = n_steps  # number of output frames (frame 0 = t=0, stored before loop)

    print(f"  GT Numba CUDA leapfrog: dt={dt_fixed:.2e} yr  n_phys={n_phys:,}  "
          f"N={N}  n_out={n_out}  stride={stride}  threads_per_block={_THREADS_PER_BLOCK}")

    # ── Transfer initial states to GPU ─────────────────────────────────────
    d_pos_mp = cuda.to_device(np.ascontiguousarray(pos_mp0, dtype=np.float64))
    d_pos_ms = cuda.to_device(np.ascontiguousarray(pos_ms0, dtype=np.float64))
    d_pos_mm = cuda.to_device(np.ascontiguousarray(pos_mm0, dtype=np.float64))
    d_vel_mp = cuda.to_device(np.ascontiguousarray(vel_mp0, dtype=np.float64))
    d_vel_ms = cuda.to_device(np.ascontiguousarray(vel_ms0, dtype=np.float64))
    d_vel_mm = cuda.to_device(np.ascontiguousarray(vel_mm0, dtype=np.float64))
    d_mu_mm  = cuda.to_device(np.ascontiguousarray(mu_mm_arr, dtype=np.float64))

    d_out_stable    = cuda.device_array(N, dtype=np.uint8)
    d_out_habitable = cuda.device_array(N, dtype=np.uint8)

    # Trajectory output buffers on device: (N, n_out, 3) float64
    d_traj_p = cuda.device_array((N, n_out, 3), dtype=np.float64)
    d_traj_s = cuda.device_array((N, n_out, 3), dtype=np.float64)
    d_traj_m = cuda.device_array((N, n_out, 3), dtype=np.float64)

    # ── CUDA kernel launch ─────────────────────────────────────────────────
    blocks = (N + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK

    _leapfrog_3body_kernel[blocks, _THREADS_PER_BLOCK](
        d_pos_mp, d_pos_ms, d_pos_mm,
        d_vel_mp, d_vel_ms, d_vel_mm,
        d_mu_mm,
        float(mu_ms), float(mu_mp),
        float(dt_fixed), float(t_sim),
        float(rhill_AU), float(escape_factor),
        float(a_inner_au), float(a_outer_au),
        int(_WARMUP_STEPS), int(n_phys),
        int(stride), int(n_out),
        d_out_stable, d_out_habitable,
        d_traj_p, d_traj_s, d_traj_m,
    )

    cuda.synchronize()

    # ── Retrieve results ───────────────────────────────────────────────────
    out_stable    = d_out_stable.copy_to_host().astype(bool)
    out_habitable = d_out_habitable.copy_to_host().astype(bool)
    traj_planet   = d_traj_p.copy_to_host()   # (N, n_out, 3)
    traj_star     = d_traj_s.copy_to_host()   # (N, n_out, 3)
    traj_moon     = d_traj_m.copy_to_host()   # (N, n_out, 3)

    map_stable    = out_stable.reshape(mm_resolution, am_resolution)
    map_habitable = out_habitable.reshape(mm_resolution, am_resolution)
    map_both      = map_stable & map_habitable

    if eligible_mask is not None:
        em_2d = np.asarray(eligible_mask, dtype=bool).reshape(mm_resolution, am_resolution)
        map_stable    &= em_2d
        map_habitable &= em_2d
        map_both      &= em_2d

    # ── Valid ranges ───────────────────────────────────────────────────────
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

    t_grid = np.linspace(0.0, t_sim, n_out)
    elapsed = time.perf_counter() - t0

    return {
        "ok":              True,
        "traj_planet":     traj_planet,   # (N, n_out, 3) numpy array
        "traj_star":       traj_star,
        "traj_moon":       traj_moon,
        "t_grid":          t_grid,
        "moon_planet_dist": None,         # not computed; use traj arrays directly
        "moon_star_dist":  None,
        # Maps
        "map_stable":      map_stable.tolist(),
        "map_habitable":   map_habitable.tolist(),
        "map_both":        map_both.tolist(),
        "mm_grid":         mm_grid.tolist(),
        "am_grid":         am_grid.tolist(),
        "valid_mm_range":  valid_mm_range,
        "valid_am_per_mm": valid_am_per_mm,
        # System info
        "rhill_AU":        float(rhill_AU),
        "a_inner_au":      float(a_inner_au),
        "a_outer_au":      float(a_outer_au),
        "n_phys":          n_phys,
        "dt_phys":         dt_fixed,
        "n_out":           n_out,
        "elapsed_s":       elapsed,
    }
