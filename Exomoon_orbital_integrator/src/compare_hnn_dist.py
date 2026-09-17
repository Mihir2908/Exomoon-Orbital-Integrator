"""
compare_hnn_dist.py — Same trajectory comparison test as compare_hnn_trajectory.py
but pointed at models_hnn_dist (new distance-based architecture).
"""

import os, sys, json
import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from exomoon.ml.hnn_inference  import batch_hnn_trajectories
from exomoon.ml.hnn_dataset    import _MERTH_OVER_MSUN
from exomoon.constants         import FOUR_PI2
from exomoon.params            import SystemParams
from exomoon.simulation        import run_simulation_for_years

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--model_dir", default="models_hnn_dist")
_ap.add_argument("--mm_idx",    type=int, default=25)
_ap.add_argument("--am_idx",    type=int, default=25)
_ap.add_argument("--dt_factor", type=int, default=50)
_args = _ap.parse_args()

GT_META   = os.path.join(SRC, "ground_truth_grids", "Kepler_452_b_v2_prograde.meta.json")
HNN_DIR   = os.path.join(SRC, _args.model_dir)
MM_RES    = AM_RES = 50
MM_IDX    = _args.mm_idx
AM_IDX    = _args.am_idx
DT_FACTOR = _args.dt_factor

with open(GT_META) as f:
    gt_meta = json.load(f)

sp = gt_meta["system_params"]
system_params = {k: float(sp[k]) for k in ("ms_solar", "rs_solar", "Ts", "mp_earth", "ap_AU")}
system_params["ep"] = float(sp.get("ep", 0.0))
T_SIM = float(gt_meta["t_sim"])

print("=" * 72)
print("HNN (distance-based) vs Ground-Truth Leapfrog — K-452b-v2 prograde")
print(f"Cell: mm_idx={MM_IDX}, am_idx={AM_IDX}  |  t_sim={T_SIM} yr  |  model: models_hnn_dist")
print("=" * 72)

cell_idx = MM_IDX * AM_RES + AM_IDX

print("\n[1/2] Running HNN inference (2500 cells, dt_factor=50)...")
hnn_res = batch_hnn_trajectories(
    system_params=system_params, t_sim=T_SIM,
    mm_resolution=MM_RES, am_resolution=AM_RES,
    n_steps=1000, model_dir=HNN_DIR, dt_factor=DT_FACTOR,
    track_energy_idx=cell_idx,
)
if not hnn_res["ok"]:
    print(f"HNN failed: {hnn_res}")
    sys.exit(1)

mm_grid       = np.array(hnn_res["mm_grid"])
am_grid       = np.array(hnn_res["am_grid"])
rhill_AU      = float(hnn_res["rhill_AU"])
mm_earth_cell = float(mm_grid[MM_IDX])
am_hill_cell  = float(am_grid[AM_IDX])
am_AU_cell    = am_hill_cell * rhill_AU

hnn_mpd  = hnn_res["moon_planet_dist"][cell_idx]   # (n_out,)
energy_htheta = hnn_res["energy_htheta"]            # (n_out,) H_θ = T + V_θ×V_ref
energy_htrue_hnn = hnn_res["energy_htrue"]          # (n_out,) H_true along HNN traj
t_hnn    = hnn_res["t_grid"]
print(f"  mm_earth={mm_earth_cell:.5f}  am_hill={am_hill_cell:.4f}  am_AU={am_AU_cell:.5f}  rhill={rhill_AU:.5f}")
print(f"  HNN elapsed = {hnn_res['elapsed_s']:.1f}s  |  n_phys={hnn_res['n_phys']}")

print("\n[2/2] Running ground-truth Numba leapfrog (single cell)...")
p = SystemParams(
    ms_solar=system_params["ms_solar"], rs_solar=system_params["rs_solar"],
    Ts=system_params["Ts"], mp_earth=system_params["mp_earth"],
    dp_cgs=5.5, ap_AU=system_params["ap_AU"], ep=system_params["ep"],
    mm_earth=mm_earth_cell, am_hill=am_hill_cell,
    em=0.0, moon_retrograde=False,
)
sim         = run_simulation_for_years(p, T_SIM)
traj        = sim["traj"]
mp_pos      = traj["xyzarr_mp"]
mm_pos      = traj["xyzarr_mm"]
gt_mpd_full = np.linalg.norm(mm_pos - mp_pos, axis=1)
N_gt        = len(gt_mpd_full)
idx_gt      = np.round(np.linspace(0, N_gt - 1, len(hnn_mpd))).astype(int)
gt_mpd      = gt_mpd_full[idx_gt]
print(f"  GT steps: {N_gt:,}  ->  resampled to {len(gt_mpd)} points")

# GT energy H_true along GT trajectory
_ms = system_params["ms_solar"]
_mp = system_params["mp_earth"] * _MERTH_OVER_MSUN
_mm = mm_earth_cell * _MERTH_OVER_MSUN
pos_s = traj["xyzarr_ms"];  pos_p = traj["xyzarr_mp"];  pos_m = traj["xyzarr_mm"]
vel_s = traj["velarr_ms"];   vel_p = traj["velarr_mp"];  vel_m = traj["velarr_mm"]
d_sp_gt = np.linalg.norm(pos_p - pos_s, axis=1)
d_sm_gt = np.linalg.norm(pos_m - pos_s, axis=1)
d_pm_gt = np.linalg.norm(pos_m - pos_p, axis=1)
T_gt    = 0.5 * (_ms * np.sum(vel_s**2, axis=1) +
                 _mp * np.sum(vel_p**2, axis=1) +
                 _mm * np.sum(vel_m**2, axis=1))
V_gt    = -FOUR_PI2 * (_ms * _mp / d_sp_gt + _ms * _mm / d_sm_gt + _mp * _mm / d_pm_gt)
H_gt_full = T_gt + V_gt                         # (N_gt,)  [M_sun·AU²/yr²]
H_gt      = H_gt_full[idx_gt]                   # resampled to match HNN grid

print()
print(f"{'t (yr)':>8}  {'HNN mpd':>12}  {'GT mpd':>12}  {'HNN/rhill':>10}  {'GT/rhill':>10}  {'|err| AU':>10}  {'err%':>7}")
print("-" * 78)
compare_idx = np.round(np.linspace(0, len(hnn_mpd) - 1, 20)).astype(int)
for i in compare_idx:
    t       = t_hnn[i]
    h       = float(hnn_mpd[i])
    g       = float(gt_mpd[i])
    err_au  = abs(h - g)
    err_pct = err_au / (g + 1e-12) * 100
    print(f"{t:>8.2f}  {h:>12.6f}  {g:>12.6f}  {h/rhill_AU:>10.4f}  {g/rhill_AU:>10.4f}  {err_au:>10.6f}  {err_pct:>6.1f}%")

abs_errs_all = np.abs(hnn_mpd - gt_mpd)
print()
print("-" * 78)
print(f"  Mean |error|: {abs_errs_all.mean():.6f} AU  ({abs_errs_all.mean()/rhill_AU*100:.2f}% of rhill)")
print(f"  Max  |error|: {abs_errs_all.max():.6f} AU  ({abs_errs_all.max()/rhill_AU*100:.2f}% of rhill)")
print(f"  RMSE        : {np.sqrt((abs_errs_all**2).mean()):.6f} AU")
print()
hnn_stable = hnn_mpd.max() <= rhill_AU
gt_stable  = gt_mpd_full.max() <= rhill_AU
print(f"  Stability - HNN: {'STABLE' if hnn_stable else 'ESCAPED'}  "
      f"|  GT: {'STABLE' if gt_stable else 'ESCAPED'}  "
      f"|  Agreement: {'OK' if hnn_stable == gt_stable else 'MISMATCH'}")
print(f"  HNN max mpd/rhill: {hnn_mpd.max()/rhill_AU:.4f}")
print(f"  GT  max mpd/rhill: {gt_mpd_full.max()/rhill_AU:.4f}")
print("-" * 78)

# ── Energy tracking report ──────────────────────────────────────────────────
print()
print("Energy conservation (fractional drift from t=0):")
print(f"  H_θ = T + V_θ×V_ref  (HNN conserved quantity)")
print(f"  H_true = T + V_newton (true Newtonian energy)")
print()
print(f"{'t (yr)':>8}  {'H_θ drift%':>12}  {'H_true(HNN)%':>14}  {'H_true(GT)%':>13}")
print("-" * 55)
H0_theta    = energy_htheta[0]
H0_true_hnn = energy_htrue_hnn[0]
H0_true_gt  = H_gt[0]
for i in compare_idx:
    t   = t_hnn[i]
    dth = (energy_htheta[i]    - H0_theta)    / abs(H0_theta)    * 100
    dhn = (energy_htrue_hnn[i] - H0_true_hnn) / abs(H0_true_hnn) * 100
    dgt = (H_gt[i]             - H0_true_gt)  / abs(H0_true_gt)  * 100
    print(f"{t:>8.2f}  {dth:>+12.4f}%  {dhn:>+13.4f}%  {dgt:>+12.4f}%")
print("-" * 55)
print(f"  Max |H_θ drift|       : {np.max(np.abs((energy_htheta - H0_theta)/abs(H0_theta)*100)):.4f}%")
print(f"  Max |H_true(HNN) drift|: {np.max(np.abs((energy_htrue_hnn - H0_true_hnn)/abs(H0_true_hnn)*100)):.4f}%")
print(f"  Max |H_true(GT) drift| : {np.max(np.abs((H_gt - H0_true_gt)/abs(H0_true_gt)*100)):.4f}%")
