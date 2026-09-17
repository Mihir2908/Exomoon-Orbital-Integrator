"""
benchmark_hnn_dt1.py — Time HNN v8 at dt_factor=1 (same timestep as leapfrog).

Standalone: re-uses the prior leapfrog + HNN-50/HNN-10 numbers from the
benchmark_v8_vs_leapfrog.py run and adds the dt_factor=1 measurement so
we can compare on a level playing field.

ISOLATION: no MLP infrastructure touched.
"""

import os, sys, json, time
import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from exomoon.ml.hnn_inference import batch_hnn_trajectories

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
T_SIM   = float(meta["t_sim"])
MM_RES  = AM_RES = 50
N_CELLS = MM_RES * AM_RES

HNN_MODEL_DIR = os.path.join(SRC, "models_hnn_dist_v8")

print("=" * 70)
print("HNN v8 at dt_factor=1  (same timestep as batch leapfrog)")
print(f"System : K-452b-v2 prograde")
print(f"Grid   : {MM_RES}x{AM_RES} = {N_CELLS} cells  |  t_sim={T_SIM} yr")
print("=" * 70)

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
    model_dir=HNN_MODEL_DIR,
    dt_factor=1,
)
elapsed = time.perf_counter() - t0

if not res.get("ok"):
    print(f"ERROR: {res.get('message')}")
    sys.exit(1)

n_phys  = res["n_phys"]
ms_step = elapsed / n_phys * 1000

print(f"  n_phys     : {n_phys:,}")
print(f"  dt         : {res['dt_phys']:.5f} yr  (dt_factor=1)")
print(f"  Wall time  : {elapsed:.2f} s")
print(f"  ms/step    : {ms_step:.3f} ms")

stop    = res["stop_step"]
n_out   = res["n_out"]
n_early = int((stop >= 0).sum())
if n_early > 0:
    mean_frac = stop[stop >= 0].mean() / n_out
    print(f"  Cells stopping early: {n_early}/{N_CELLS}  ({100*n_early/N_CELLS:.1f}%)")
    print(f"  Mean stop fraction  : {mean_frac:.3f}  ({mean_frac*n_out:.0f}/{n_out} output steps)")
else:
    print(f"  No cells stop early (all stable+habitable throughout)")

# ── Level-playing-field comparison ───────────────────────────────────────────
print()
print("=" * 70)
print("  Level-playing-field summary (same n_phys = 200,000):")
print(f"  {'Option':<32}  {'n_phys':>8}  {'time (s)':>9}  {'ms/step':>8}  {'vs leapfrog':>12}")
print("=" * 70)
LF_TIME   = 360.89    # from prior benchmark run
LF_PHYS   = 200_000
LF_MSSTEP = LF_TIME / LF_PHYS * 1000
HNN50_T   = 24.06
HNN50_N   = 4_000
HNN10_T   = 110.83
HNN10_N   = 20_000

# Extrapolate HNN at dt×1: 200,000 steps at the measured ms/step
hnn_dt1_extrap = ms_step * 200_000 / 1000
print(f"  {'Leapfrog (exact, dt×1)':<32}  {LF_PHYS:>8,}  {LF_TIME:>9.2f}  "
      f"{LF_MSSTEP:>8.3f}  {'1.00× (baseline)':>12}")
print(f"  {'HNN v8 (dt×50 — measured)':<32}  {HNN50_N:>8,}  {HNN50_T:>9.2f}  "
      f"{HNN50_T/HNN50_N*1000:>8.3f}  {'—':>12}")
print(f"  {'HNN v8 (dt×10 — measured)':<32}  {HNN10_N:>8,}  {HNN10_T:>9.2f}  "
      f"{HNN10_T/HNN10_N*1000:>8.3f}  {'—':>12}")
print(f"  {'HNN v8 (dt×1 — measured)':<32}  {n_phys:>8,}  {elapsed:>9.2f}  "
      f"{ms_step:>8.3f}  {LF_TIME/elapsed:>11.2f}×")
print(f"  {'HNN v8 (dt×1 — extrap to 200k)':<32}  {200_000:>8,}  {hnn_dt1_extrap:>9.2f}  "
      f"{ms_step:>8.3f}  {LF_TIME/hnn_dt1_extrap:>11.2f}×")
print("=" * 70)
