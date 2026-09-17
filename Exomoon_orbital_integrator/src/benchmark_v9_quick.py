"""
benchmark_v9_quick.py — Quick stop-stats + timing check for v9 (256x3) at dt_factor=50.

Compares v8 (128x2) and v9 (256x3) at dt_factor=50 to see whether
the pred_pm d=1.0 fix in v9 reduces the 100% early-stop rate.
"""

import os, sys, json, time
import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from exomoon.ml.hnn_inference import batch_hnn_trajectories

GT_META = os.path.join(SRC, "ground_truth_grids", "Kepler_452_b_v2_prograde.meta.json")
with open(GT_META) as f:
    meta = json.load(f)

sp = meta["system_params"]
SYSTEM_PARAMS = dict(
    ms_solar=sp["ms_solar"], rs_solar=sp["rs_solar"],
    Ts=sp["Ts"],             mp_earth=sp["mp_earth"],
    ap_AU=sp["ap_AU"],       ep=sp.get("ep", 0.0),
)
T_SIM   = float(meta["t_sim"])
MM_RES  = AM_RES = 50
N_CELLS = MM_RES * AM_RES

def run_and_report(model_dir, dt_factor, label):
    t0 = time.perf_counter()
    res = batch_hnn_trajectories(
        system_params=SYSTEM_PARAMS,
        t_sim=T_SIM,
        moon_retrograde=False,
        em=0.0,
        mm_resolution=MM_RES,
        am_resolution=AM_RES,
        n_steps=1000,
        escape_factor=1.0,
        model_dir=model_dir,
        dt_factor=dt_factor,
    )
    elapsed = time.perf_counter() - t0

    if not res.get("ok"):
        print(f"  [{label}] ERROR: {res.get('message')}")
        return None

    n_phys  = res["n_phys"]
    n_out   = res["n_out"]
    stop    = res["stop_step"]
    n_early = int((stop >= 0).sum())
    mean_frac = stop[stop >= 0].mean() / n_out if n_early > 0 else 0.0

    print(f"  [{label}]  n_phys={n_phys:,}  dt={res['dt_phys']:.5f} yr"
          f"  wall={elapsed:.1f}s  ms/step={elapsed/n_phys*1000:.3f}")
    print(f"           early_stop: {n_early}/{N_CELLS} ({100*n_early/N_CELLS:.1f}%)  "
          f"mean_stop_frac={mean_frac:.3f}  ({mean_frac*n_out:.0f}/{n_out} steps)")
    return elapsed, n_phys

print("=" * 70)
print("v8 (128x2) vs v9 (256x3) — dt_factor=50  stop-stats comparison")
print(f"System: K-452b-v2 prograde  |  grid: {MM_RES}x{AM_RES}  |  t_sim={T_SIM} yr")
print("=" * 70)

# v8 reference
print()
print("v8 (128x2):"); run_and_report("models_hnn_dist_v8", 50, "v8 dt×50")
print(); print("v9 (256x3):"); run_and_report("models_hnn_dist_v9", 50, "v9 dt×50")

# Also run dt×10 for v9 to check
print(); print("v9 (256x3) dt×10:"); run_and_report("models_hnn_dist_v9", 10, "v9 dt×10")
print("=" * 70)
