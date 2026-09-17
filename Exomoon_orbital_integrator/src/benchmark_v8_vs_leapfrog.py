"""
benchmark_v8_vs_leapfrog.py — Head-to-head timing: HNN v8 vs batch leapfrog.

Tests both options on the same system (K-452b-v2 prograde, 50x50 grid).
Reports wall time, n_phys, and early-stopping stats from the post-hoc
stop_step output (neither option currently exits the integration loop early —
times shown are full upper-bound costs).

ISOLATION: no MLP infrastructure touched.
"""

import os, sys, json, time
import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from exomoon.ml.batch_leapfrog import batch_leapfrog_trajectories
from exomoon.ml.hnn_inference  import batch_hnn_trajectories

# ── System: K-452b-v2 prograde ────────────────────────────────────────────────
GT_META = os.path.join(SRC, "ground_truth_grids", "Kepler_452_b_v2_prograde.meta.json")
with open(GT_META) as f:
    meta = json.load(f)

sp = meta["system_params"]
SYSTEM_PARAMS = dict(
    ms_solar=sp["ms_solar"], rs_solar=sp["rs_solar"],
    Ts=sp["Ts"],             mp_earth=sp["mp_earth"],
    ap_AU=sp["ap_AU"],       ep=sp.get("ep", 0.0),
)
T_SIM = float(meta["t_sim"])
MM_RES = AM_RES = 50
N_CELLS = MM_RES * AM_RES

HNN_MODEL_DIR = os.path.join(SRC, "models_hnn_dist_v8")

print("=" * 70)
print("Head-to-head: HNN v8 vs batch leapfrog")
print(f"System : K-452b-v2 prograde")
print(f"Grid   : {MM_RES}x{AM_RES} = {N_CELLS} cells")
print(f"t_sim  : {T_SIM} yr")
print(f"Note   : neither option exits the loop early — times are upper bounds.")
print(f"         Stopping criteria (stop_step) computed post-hoc from output.")
print("=" * 70)


def report_stop_stats(result, label):
    """Print early-stopping stats from post-hoc stop_step output."""
    if "stop_step" not in result:
        return
    stop  = result["stop_step"]           # (N,) int32, -1 = never stops
    n_out = result["n_out"]
    frac_stops = (stop >= 0).mean()
    mean_stop  = stop[stop >= 0].mean() / n_out if (stop >= 0).any() else None
    init_hab   = result.get("initially_habitable")
    n_hab0     = int(init_hab.sum()) if init_hab is not None else "N/A"

    print(f"  [{label}] Post-hoc stop stats (n_out={n_out}):")
    print(f"    Cells that would stop early : {(stop>=0).sum():4d} / {N_CELLS}"
          f"  ({100*frac_stops:.1f}%)")
    if mean_stop is not None:
        print(f"    Mean stop step (fraction)   : {mean_stop:.3f}  "
              f"({mean_stop*n_out:.0f} / {n_out} output steps)")
    print(f"    Cells starting in HZ (hab@t0): {n_hab0} / {N_CELLS}")


# ── Option 1: Batch leapfrog ──────────────────────────────────────────────────
print()
print("OPTION 1: Batch leapfrog (exact physics, dt_factor=1)")
print("-" * 70)
t0 = time.perf_counter()
res_lf = batch_leapfrog_trajectories(
    system_params=SYSTEM_PARAMS,
    t_sim=T_SIM,
    moon_retrograde=False,
    em=0.0,
    mm_resolution=MM_RES,
    am_resolution=AM_RES,
    n_steps=1000,
    escape_factor=1.0,
)
elapsed_lf = time.perf_counter() - t0

print(f"  n_phys     : {res_lf['n_phys']:,}")
print(f"  dt         : {res_lf['dt_phys']:.5f} yr")
print(f"  Wall time  : {elapsed_lf:.2f} s")
print(f"  ms/step    : {elapsed_lf/res_lf['n_phys']*1000:.3f} ms")


# ── Option 2: HNN v8 — dt_factor=50 (default, ~10x coarser) ──────────────────
print()
print("OPTION 2: HNN v8 — dt_factor=50 (default, ~10x fewer steps)")
print("-" * 70)
t0 = time.perf_counter()
res_hnn50 = batch_hnn_trajectories(
    system_params=SYSTEM_PARAMS,
    t_sim=T_SIM,
    moon_retrograde=False,
    em=0.0,
    mm_resolution=MM_RES,
    am_resolution=AM_RES,
    n_steps=1000,
    escape_factor=1.0,
    model_dir=HNN_MODEL_DIR,
    dt_factor=50,
)
elapsed_hnn50 = time.perf_counter() - t0

if res_hnn50.get("ok"):
    print(f"  n_phys     : {res_hnn50['n_phys']:,}")
    print(f"  dt         : {res_hnn50['dt_phys']:.5f} yr  (dt_factor=50)")
    print(f"  Wall time  : {elapsed_hnn50:.2f} s")
    print(f"  ms/step    : {elapsed_hnn50/res_hnn50['n_phys']*1000:.3f} ms")
    report_stop_stats(res_hnn50, "HNN-50")
else:
    print(f"  ERROR: {res_hnn50.get('message')}")


# ── Option 3: HNN v8 — dt_factor=10 (higher accuracy, ~5x coarser) ───────────
print()
print("OPTION 3: HNN v8 — dt_factor=10 (higher accuracy, ~5x fewer steps)")
print("-" * 70)
t0 = time.perf_counter()
res_hnn10 = batch_hnn_trajectories(
    system_params=SYSTEM_PARAMS,
    t_sim=T_SIM,
    moon_retrograde=False,
    em=0.0,
    mm_resolution=MM_RES,
    am_resolution=AM_RES,
    n_steps=1000,
    escape_factor=1.0,
    model_dir=HNN_MODEL_DIR,
    dt_factor=10,
)
elapsed_hnn10 = time.perf_counter() - t0

if res_hnn10.get("ok"):
    print(f"  n_phys     : {res_hnn10['n_phys']:,}")
    print(f"  dt         : {res_hnn10['dt_phys']:.5f} yr  (dt_factor=10)")
    print(f"  Wall time  : {elapsed_hnn10:.2f} s")
    print(f"  ms/step    : {elapsed_hnn10/res_hnn10['n_phys']*1000:.3f} ms")
    report_stop_stats(res_hnn10, "HNN-10")
else:
    print(f"  ERROR: {res_hnn10.get('message')}")


# ── Summary table ─────────────────────────────────────────────────────────────
print()
print("=" * 70)
print(f"  {'Option':<30}  {'n_phys':>8}  {'time (s)':>9}  {'speedup':>8}")
print("=" * 70)
print(f"  {'Leapfrog (exact, dt×1)':<30}  {res_lf['n_phys']:>8,}  "
      f"{elapsed_lf:>9.2f}  {'1.00×':>8}")
if res_hnn50.get("ok"):
    su50 = elapsed_lf / elapsed_hnn50
    print(f"  {'HNN v8 (dt×50)':<30}  {res_hnn50['n_phys']:>8,}  "
          f"{elapsed_hnn50:>9.2f}  {su50:>7.2f}x")
if res_hnn10.get("ok"):
    su10 = elapsed_lf / elapsed_hnn10
    print(f"  {'HNN v8 (dt×10)':<30}  {res_hnn10['n_phys']:>8,}  "
          f"{elapsed_hnn10:>9.2f}  {su10:>7.2f}x")
print("=" * 70)
print()
print("With per-cell early stopping (not yet in loop): both options would be")
print("faster by ~(1 - mean_stop_fraction). Relative speedup stays the same.")
