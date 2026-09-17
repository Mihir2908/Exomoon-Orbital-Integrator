"""
compare_hnn_trajectory.py

Compares HNN-driven leapfrog trajectory against the ground-truth Numba leapfrog
for a single representative cell from the K-452b-v2 prograde 50×50 grid.

Picks cell (mm_idx=25, am_idx=25) — near the centre of the grid.
Prints moon_planet_dist (AU) at 20 evenly-spaced time points for both integrators,
then summarises max deviation and stability classification agreement.
"""

import os, sys, json
import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from exomoon.ml.hnn_inference import batch_hnn_trajectories
from exomoon.params            import SystemParams
from exomoon.simulation        import run_simulation_for_years

GT_META  = os.path.join(SRC, "ground_truth_grids", "Kepler_452_b_v2_prograde.meta.json")
HNN_DIR  = os.path.join(SRC, "models_hnn_fref")
MM_RES   = AM_RES = 50
MM_IDX   = 25
AM_IDX   = 25
DT_FACTOR = 50   # same default used in benchmarks

with open(GT_META) as f:
    gt_meta = json.load(f)

sp = gt_meta["system_params"]
system_params = {
    "ms_solar": float(sp["ms_solar"]),
    "rs_solar": float(sp["rs_solar"]),
    "Ts":       float(sp["Ts"]),
    "mp_earth": float(sp["mp_earth"]),
    "ap_AU":    float(sp["ap_AU"]),
    "ep":       float(sp.get("ep", 0.0)),
}
T_SIM = float(gt_meta["t_sim"])

print("=" * 72)
print("HNN vs Ground-Truth Leapfrog — K-452b-v2 prograde")
print(f"Cell: mm_idx={MM_IDX}, am_idx={AM_IDX}  |  t_sim={T_SIM} yr")
print("=" * 72)

# ── HNN inference (all 2500 cells, extract centre cell) ──────────────────────
print("\n[1/2] Running HNN inference (2500 cells, dt_factor=50)...")
hnn_res = batch_hnn_trajectories(
    system_params=system_params,
    t_sim=T_SIM,
    mm_resolution=MM_RES,
    am_resolution=AM_RES,
    n_steps=1000,
    model_dir=HNN_DIR,
    dt_factor=DT_FACTOR,
)

if not hnn_res["ok"]:
    print(f"HNN failed: {hnn_res}")
    sys.exit(1)

mm_grid      = np.array(hnn_res["mm_grid"])
am_grid      = np.array(hnn_res["am_grid"])
rhill_AU     = float(hnn_res["rhill_AU"])
mm_earth_cell = float(mm_grid[MM_IDX])
am_hill_cell  = float(am_grid[AM_IDX])
am_AU_cell    = am_hill_cell * rhill_AU

cell_idx = MM_IDX * AM_RES + AM_IDX
hnn_mpd  = hnn_res["moon_planet_dist"][cell_idx]   # (1000,) AU
t_hnn    = hnn_res["t_grid"]                        # (1000,) yr
print(f"  mm_earth = {mm_earth_cell:.5f} M_earth  |  am_hill = {am_hill_cell:.4f}  |  am_AU = {am_AU_cell:.5f} AU")
print(f"  rhill_AU = {rhill_AU:.5f} AU  |  HNN elapsed = {hnn_res['elapsed_s']:.1f}s")

# ── Ground-truth leapfrog (single cell) ──────────────────────────────────────
print("\n[2/2] Running ground-truth Numba leapfrog (single cell)...")
p = SystemParams(
    ms_solar       = system_params["ms_solar"],
    rs_solar       = system_params["rs_solar"],
    Ts             = system_params["Ts"],
    mp_earth       = system_params["mp_earth"],
    dp_cgs         = 5.5,
    ap_AU          = system_params["ap_AU"],
    ep             = system_params["ep"],
    mm_earth       = mm_earth_cell,
    am_hill        = am_hill_cell,
    em             = 0.0,
    moon_retrograde= False,
)
sim  = run_simulation_for_years(p, T_SIM)
traj = sim["traj"]

mp_pos      = traj["xyzarr_mp"]                        # (N, 3)
mm_pos      = traj["xyzarr_mm"]
gt_mpd_full = np.linalg.norm(mm_pos - mp_pos, axis=1)  # (N,)
N_gt        = len(gt_mpd_full)

# Resample GT to the same 1000-point grid as HNN output
idx_gt = np.round(np.linspace(0, N_gt - 1, len(hnn_mpd))).astype(int)
gt_mpd = gt_mpd_full[idx_gt]
print(f"  GT steps: {N_gt:,}  →  resampled to {len(gt_mpd)} points")

# ── Point-by-point comparison ─────────────────────────────────────────────────
print()
print(f"{'t (yr)':>8}  {'HNN mpd':>12}  {'GT mpd':>12}  "
      f"{'HNN/rhill':>10}  {'GT/rhill':>10}  {'|err| AU':>10}  {'err%':>7}")
print("-" * 78)

n_pts = 20
compare_idx = np.round(np.linspace(0, len(hnn_mpd) - 1, n_pts)).astype(int)
abs_errs = []
for i in compare_idx:
    t   = t_hnn[i]
    h   = float(hnn_mpd[i])
    g   = float(gt_mpd[i])
    err_au  = abs(h - g)
    err_pct = err_au / (g + 1e-12) * 100
    abs_errs.append(err_au)
    print(f"{t:>8.2f}  {h:>12.6f}  {g:>12.6f}  "
          f"{h/rhill_AU:>10.4f}  {g/rhill_AU:>10.4f}  {err_au:>10.6f}  {err_pct:>6.1f}%")

# ── Summary ───────────────────────────────────────────────────────────────────
abs_errs_all = np.abs(hnn_mpd - gt_mpd)
print()
print("─" * 78)
print(f"  Mean |error|  : {abs_errs_all.mean():.6f} AU  "
      f"({abs_errs_all.mean()/rhill_AU*100:.2f}% of rhill)")
print(f"  Max  |error|  : {abs_errs_all.max():.6f} AU  "
      f"({abs_errs_all.max()/rhill_AU*100:.2f}% of rhill)")
print(f"  RMSE          : {np.sqrt((abs_errs_all**2).mean()):.6f} AU")
print()
hnn_stable = hnn_mpd.max() <= rhill_AU
gt_stable  = gt_mpd_full.max() <= rhill_AU
print(f"  Stability — HNN: {'STABLE' if hnn_stable else 'ESCAPED'}  "
      f"|  GT: {'STABLE' if gt_stable else 'ESCAPED'}  "
      f"|  Agreement: {'✓' if hnn_stable == gt_stable else '✗'}")
print(f"  HNN max mpd/rhill: {hnn_mpd.max()/rhill_AU:.4f}")
print(f"  GT  max mpd/rhill: {gt_mpd_full.max()/rhill_AU:.4f}")
print("─" * 78)
