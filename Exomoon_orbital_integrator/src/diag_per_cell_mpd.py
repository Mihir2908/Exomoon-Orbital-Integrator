"""
Per-cell mpd diagnostic: tracks the actual moon-planet distance for cell k=0
(mm=0.107, am_hill=0.035, innermost K-1229b cell) at every single physics step
for the first 500 steps with K=10 and K=16.

If K=10 causes escape: we'll see mpd grow from ~1.75e-4 AU toward rhill=0.00504 AU.
If mpd stays bounded: the stability criterion logic must be wrong somewhere else.
"""
import json, os, sys, time
import numpy as np
import torch

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from exomoon.constants      import FOUR_PI2, au, merth, msun
from exomoon.constants      import rsun as _rsun
from exomoon.habitable_zone import hz_bounds_au
from exomoon.ml.batch_leapfrog import _build_initial_states, _accel_batch

_PLANET_DENSITY_CGS = 5.5
_PLANET_DENSITY_SI  = 5500.0
_MOON_DENSITY_CGS   = 3.0


def run_single_cell_trace(sp, K, n_steps_trace=500):
    ms_solar = float(sp["ms_solar"])
    mp_earth = float(sp["mp_earth"])
    ap_AU    = float(sp["ap_AU"])
    ep       = float(sp.get("ep", 0.0))

    ms_gp_tmp = ms_solar * FOUR_PI2
    mp_gp_tmp = mp_earth * (merth / msun) * FOUR_PI2
    rhill_AU  = ap_AU * (1 - ep) * (mp_gp_tmp / (3 * ms_gp_tmp))**(1/3)

    mp_kg   = mp_earth * merth
    rp_m    = (0.75 * mp_kg / (np.pi * _PLANET_DENSITY_SI))**(1/3)
    a_roche = 2.456 * rp_m * (_PLANET_DENSITY_CGS / _MOON_DENSITY_CGS)**(1/3) / au
    am_min  = max(a_roche / rhill_AU, 1e-3)
    mm_max  = min(mp_earth, 3.0)

    MM_RES, AM_RES = 50, 50
    mm_grid = np.exp(np.linspace(np.log(0.107), np.log(mm_max), MM_RES))
    am_grid = np.linspace(am_min, 1.0, AM_RES)

    print(f"\n  rhill_AU={rhill_AU:.5f}  am_min={am_min:.4f}  mm_max={mm_max:.4f}")
    print(f"  mm_grid[0]={mm_grid[0]:.4f}  am_grid[0]={am_grid[0]:.4f}")
    print(f"  am_AU(inner)={am_grid[0]*rhill_AU:.4e} AU")

    (pos_mp0, pos_ms0, pos_mm0, vel_mp0, vel_ms0, vel_mm0,
     mu_ms, mu_mp, mu_mm_arr, rhill) = _build_initial_states(
        ms_solar=ms_solar, mp_earth=mp_earth, ap_AU=ap_AU, ep=ep, em=0.0,
        moon_retrograde=False, mm_grid=mm_grid, am_grid=am_grid,
    )

    N = MM_RES * AM_RES
    dtype = torch.float64
    p_mp = torch.tensor(pos_mp0, dtype=dtype)
    p_ms = torch.tensor(pos_ms0, dtype=dtype)
    p_mm = torch.tensor(pos_mm0, dtype=dtype)
    v_mp = torch.tensor(vel_mp0, dtype=dtype)
    v_ms = torch.tensor(vel_ms0, dtype=dtype)
    v_mm = torch.tensor(vel_mm0, dtype=dtype)
    mu_mm_t = torch.tensor(mu_mm_arr[:, None], dtype=dtype)

    # Per-cell T_moon and dt
    am_per_cell    = np.tile(am_grid, MM_RES)
    am_AU_per_cell = am_per_cell * rhill_AU
    T_moon_cell    = 2 * np.pi * np.sqrt(am_AU_per_cell**3 / (mu_mp + mu_mm_arr))
    dt_arr         = T_moon_cell / K
    dt_t           = torch.tensor(dt_arr[:, None], dtype=dtype)
    half_dt_t      = dt_t * 0.5

    dt_cell0 = dt_arr[0]
    T_moon0  = T_moon_cell[0]
    am_AU0   = am_AU_per_cell[0]
    print(f"  K={K}  T_moon[k=0]={T_moon0:.4e} yr  dt[k=0]={dt_cell0:.4e} yr")
    print(f"  ωdt[k=0] = 2π/K = {2*np.pi/K:.4f} rad")
    print(f"  rhill/am_AU = {rhill_AU/am_AU0:.2f}")

    # Initial mpd for k=0
    diff0 = pos_mm0[0] - pos_mp0[0]
    mpd0  = np.sqrt((diff0**2).sum())
    print(f"  Initial mpd[k=0] = {mpd0:.4e} AU  (should ≈ am_AU={am_AU0:.4e})")

    # Trace k=0 for first n_steps_trace steps
    mpd_trace = []
    rhill_tf = torch.tensor(rhill_AU, dtype=dtype)
    done_t   = torch.zeros(N, 1, dtype=torch.bool)
    elapsed_t = torch.zeros(N, 1, dtype=dtype)

    with torch.no_grad():
        for i in range(n_steps_trace):
            active = (~done_t).to(dtype)

            p2_mp = p_mp + v_mp * (half_dt_t * active)
            a_mp  = (_accel_batch(p2_mp, p_ms, mu_ms)
                   + _accel_batch(p2_mp, p_mm, mu_mm_t))
            v_mp  = v_mp + a_mp * (dt_t * active)
            p_mp  = p2_mp + v_mp * (half_dt_t * active)

            p2_ms = p_ms + v_ms * (half_dt_t * active)
            a_ms  = (_accel_batch(p2_ms, p_mp, mu_mp)
                   + _accel_batch(p2_ms, p_mm, mu_mm_t))
            v_ms  = v_ms + a_ms * (dt_t * active)
            p_ms  = p2_ms + v_ms * (half_dt_t * active)

            p2_mm = p_mm + v_mm * (half_dt_t * active)
            a_mm  = (_accel_batch(p2_mm, p_mp, mu_mp)
                   + _accel_batch(p2_mm, p_ms, mu_ms))
            v_mm  = v_mm + a_mm * (dt_t * active)
            p_mm  = p2_mm + v_mm * (half_dt_t * active)

            diff_mp = p_mm - p_mp
            mpd_s   = (diff_mp * diff_mp).sum(dim=1, keepdim=True).sqrt()

            # Record k=0 only
            mpd_trace.append(float(mpd_s[0, 0]))

            elapsed_t += dt_t * active
            diff_ms = p_mm - p_ms
            msd_s   = (diff_ms * diff_ms).sum(dim=1, keepdim=True).sqrt()
            a_inner_au, a_outer_au = hz_bounds_au(
                float(sp.get("Ts", 5772.0)), float(sp.get("rs_solar",1.0)) * _rsun)
            a_inner_tf = torch.tensor(a_inner_au, dtype=dtype)
            a_outer_tf = torch.tensor(a_outer_au, dtype=dtype)
            unstable      = mpd_s > (1.0 * rhill_tf)
            uninhabitable = (msd_s < a_inner_tf) | (msd_s > a_outer_tf)
            done_t = done_t | (unstable & uninhabitable) | (elapsed_t >= 2.0)

    mpd_arr = np.array(mpd_trace)
    print(f"\n  First {n_steps_trace} steps — mpd[k=0]:")
    print(f"  step  0 : {mpd_arr[0]:.6e} AU  ({mpd_arr[0]/rhill_AU:.6f} × rhill)")
    for s in [1,2,3,4,5,9,10,14,15,19,20,24,25,29,49,99,199,499]:
        if s < len(mpd_arr):
            print(f"  step {s:3d}: {mpd_arr[s]:.6e} AU  ({mpd_arr[s]/rhill_AU:.6f} × rhill)")
    print(f"  max in first {n_steps_trace} steps: {mpd_arr.max():.6e} AU  "
          f"({mpd_arr.max()/rhill_AU:.6f} × rhill)")
    print(f"  min: {mpd_arr.min():.6e} AU  ({mpd_arr.min()/rhill_AU:.6f} × rhill)")
    print(f"  rhill = {rhill_AU:.6e} AU")
    print(f"  am_AU = {am_AU0:.6e} AU")


meta = json.load(open(os.path.join(SRC, "ground_truth_grids",
                                   "Kepler_1229_b_prograde.meta.json")))
sp   = {k: meta["system_params"][k]
        for k in ("ms_solar","rs_solar","Ts","mp_earth","ap_AU")}
sp["ep"] = meta["system_params"].get("ep", 0.0)

print(f"{'='*60}")
print(f"  K-1229b  K=10  — per-cell mpd trace for innermost cell")
print(f"{'='*60}")
run_single_cell_trace(sp, K=10, n_steps_trace=500)

print(f"\n{'='*60}")
print(f"  K-1229b  K=16  — per-cell mpd trace for innermost cell")
print(f"{'='*60}")
run_single_cell_trace(sp, K=16, n_steps_trace=500)
