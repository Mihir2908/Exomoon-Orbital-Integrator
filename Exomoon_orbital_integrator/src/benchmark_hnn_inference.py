"""
benchmark_hnn_inference.py — Time the HNN leapfrog and compare to batch_leapfrog.

This benchmark creates a TEMPORARY untrained HNN (random weights) to measure the
per-step and total inference cost.  Stability maps from untrained weights will be
meaningless — this is purely a timing test.  Replace model_dir with a real trained
model to evaluate accuracy.

Usage:
    py benchmark_hnn_inference.py

CRITICAL: does not touch any MLP infrastructure.
"""

import os, sys, time, json, tempfile
import numpy as np
import torch

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from exomoon.ml.hnn_model     import HNN
from exomoon.ml.hnn_dataset   import save_sys_scaler
from exomoon.ml.hnn_inference import batch_hnn_trajectories
from exomoon.ml.batch_leapfrog import batch_leapfrog_trajectories
from sklearn.preprocessing     import StandardScaler
import numpy as np

# ── System params from K-452b-v2 meta.json ─────────────────────────────────────
GT_META_PATH = os.path.join(SRC, "ground_truth_grids",
                             "Kepler_452_b_v2_prograde.meta.json")
with open(GT_META_PATH) as f:
    gt_meta = json.load(f)

sp = gt_meta["system_params"]
SYSTEM_PARAMS = {
    "ms_solar": sp["ms_solar"],
    "rs_solar": sp["rs_solar"],
    "Ts":       sp["Ts"],
    "mp_earth": sp["mp_earth"],
    "ap_AU":    sp["ap_AU"],
    "ep":       sp.get("ep", 0.0),
}
T_SIM = float(gt_meta["t_sim"])

print(f"System: {SYSTEM_PARAMS}")
print(f"t_sim  : {T_SIM} yr")
print()

# ── Create a temporary model dir with an untrained HNN ──────────────────────────
tmpdir = tempfile.mkdtemp(prefix="hnn_bench_")
print(f"Temp model dir: {tmpdir}")

model = HNN(hidden=256, layers=3)
model.save(tmpdir)

# Create a dummy sys_scaler fitted on plausible sys_params values
dummy_sys = np.array([
    [1.0, 3.0, 0.2, 0.018, 0.99, 1.80],    # rough K-452b ranges
    [0.5, 10.0, 0.5, 0.005, 0.20, 0.40],
    [1.5, 100.0, 1.0, 0.050, 1.50, 2.80],
])
scaler = StandardScaler()
scaler.fit(dummy_sys)
save_sys_scaler(scaler, tmpdir)
print(f"Saved untrained model + dummy scaler to {tmpdir}")
print()

# ── Timing benchmark ────────────────────────────────────────────────────────────
MM_RES, AM_RES = 50, 50
N_CELLS        = MM_RES * AM_RES

print("=" * 68)
print("CHECK A: Reference — batch_leapfrog (physics integrator, no ML)")
print("=" * 68)
t_leap = time.perf_counter()
r_leap = batch_leapfrog_trajectories(
    system_params=SYSTEM_PARAMS, t_sim=T_SIM,
    mm_resolution=MM_RES, am_resolution=AM_RES, n_steps=1000,
)
t_leap = time.perf_counter() - t_leap
print(f"  n_phys={r_leap['n_phys']}  dt={r_leap['dt_phys']:.5f} yr  n_out={r_leap['n_out']}")
print(f"  Elapsed: {t_leap:.2f} s  ({t_leap/N_CELLS*1000:.2f} ms/cell)")

for dt_factor in [10, 20, 50]:
    print()
    print("=" * 68)
    print(f"CHECK B: HNN leapfrog — dt_factor={dt_factor}  (untrained weights)")
    print("=" * 68)
    t_hnn = time.perf_counter()
    r_hnn = batch_hnn_trajectories(
        system_params=SYSTEM_PARAMS, t_sim=T_SIM,
        mm_resolution=MM_RES, am_resolution=AM_RES, n_steps=1000,
        model_dir=tmpdir, dt_factor=dt_factor,
    )
    t_hnn = time.perf_counter() - t_hnn
    print(f"  n_phys={r_hnn['n_phys']}  dt={r_hnn['dt_phys']:.5f} yr  n_out={r_hnn['n_out']}")
    print(f"  Elapsed : {t_hnn:.2f} s  ({t_hnn/N_CELLS*1000:.2f} ms/cell)")
    print(f"  vs leapfrog: {t_leap/t_hnn:.1f}× {'faster' if t_hnn < t_leap else 'slower'}")
    print(f"  map_stable fraction: {np.array(r_hnn['map_stable']).mean():.3f}  "
          f"(meaningless — untrained model)")

print()
print(f"Temp files at: {tmpdir}  (safe to delete)")
print("To use trained HNN: replace tmpdir with models_hnn/")
print("Done.")
