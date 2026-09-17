"""
benchmark_hnn_arch_sweep.py — HNN architecture timing sweep.

All architectures use the SAME code path: HNN.save() → batch_hnn_trajectories()
(which calls HNN.load() internally). No monkey-patching, no separate scripts.
Asymmetric architectures use the new layer_sizes parameter added to HNN.

Both dt_factor=50 and dt_factor=10 measured directly. No extrapolation.

Cross-checks vs prior benchmark:
  [64×2]  dt_factor=10: expect ~45s
  [128×2] dt_factor=10: expect ~68s

ISOLATION: does not touch any MLP infrastructure.
"""

import os, sys, json, time, tempfile
import numpy as np
from sklearn.preprocessing import StandardScaler

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from exomoon.ml.hnn_model     import HNN
from exomoon.ml.hnn_dataset   import save_sys_scaler
from exomoon.ml.hnn_inference import batch_hnn_trajectories

GT_META_PATH = os.path.join(SRC, "ground_truth_grids",
                             "Kepler_452_b_v2_prograde.meta.json")
with open(GT_META_PATH) as f:
    gt_meta = json.load(f)

sp = gt_meta["system_params"]
SYSTEM_PARAMS = {
    "ms_solar": sp["ms_solar"], "rs_solar": sp["rs_solar"],
    "Ts": sp["Ts"], "mp_earth": sp["mp_earth"],
    "ap_AU": sp["ap_AU"], "ep": sp.get("ep", 0.0),
}
T_SIM   = float(gt_meta["t_sim"])
MM_RES  = AM_RES = 50
N_CELLS = MM_RES * AM_RES

DUMMY_SYS = np.array([
    [1.0, 3.0,   0.2, 0.018, 0.99, 1.80],
    [0.5, 10.0,  0.5, 0.005, 0.20, 0.40],
    [1.5, 100.0, 1.0, 0.050, 1.50, 2.80],
])

# Each entry: (label, hidden, layers, layer_sizes)
# Symmetric:  layer_sizes=None  → uses (hidden, layers)
# Asymmetric: layer_sizes=[...]  → overrides hidden/layers
ARCHS = [
    ("15→16→1",         16,  1, None),
    ("15→32→1",         32,  1, None),
    ("15→64→1",         64,  1, None),
    ("15→32→32→1",      32,  2, None),
    ("15→64→64→1",      64,  2, None),
    ("15→128→64→1",     128, 2, [128, 64]),   # asymmetric — uses layer_sizes
    ("15→128→128→1",    128, 2, None),
]

DT_FACTORS = [50, 10]

print(f"HNN architecture sweep — identical code path for all models")
print(f"All use HNN.save() → batch_hnn_trajectories() → HNN.load()")
print(f"dt_factor=50 → n_phys≈4,000   dt_factor=10 → n_phys≈20,000")
print(f"System: K-452b-v2  |  {N_CELLS} cells  |  t_sim={T_SIM} yr")
print(f"batch_leapfrog reference: 463.52s  (dt_factor=1, n_phys≈200,000, 2.30 ms/step)")
print()

results = {label: {} for label, _, _, _ in ARCHS}

for label, hidden, layers, layer_sizes in ARCHS:
    model  = HNN(hidden=hidden, layers=layers, layer_sizes=layer_sizes)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"  {label}  (params={n_param:,})")

    for dt_factor in DT_FACTORS:
        tmpdir = tempfile.mkdtemp(prefix="hnn_sweep_")
        model.save(tmpdir)
        scaler = StandardScaler().fit(DUMMY_SYS)
        save_sys_scaler(scaler, tmpdir)

        t0 = time.perf_counter()
        r  = batch_hnn_trajectories(
            system_params=SYSTEM_PARAMS, t_sim=T_SIM,
            mm_resolution=MM_RES, am_resolution=AM_RES, n_steps=1000,
            model_dir=tmpdir, dt_factor=dt_factor,
        )
        elapsed = time.perf_counter() - t0
        n_phys  = r["n_phys"]
        ms_step = elapsed / n_phys * 1000

        results[label][dt_factor] = (elapsed, n_phys, ms_step)
        print(f"    dt_factor={dt_factor:>2}  n_phys={n_phys:>6}  "
              f"elapsed={elapsed:>7.2f}s  {ms_step:.2f} ms/step")
    print()

W = 100
print("─" * W)
print(f"  {'Architecture':<22}  {'Params':>7}  "
      f"{'dt50 elapsed':>13}  {'ms/step':>8}  "
      f"{'dt10 elapsed':>13}  {'ms/step':>8}  {'dt10/dt50':>10}")
print("─" * W)
for label, hidden, layers, layer_sizes in ARCHS:
    n_param = sum(p.numel() for p in HNN(hidden=hidden, layers=layers,
                                          layer_sizes=layer_sizes).parameters())
    e50, n50, ms50 = results[label][50]
    e10, n10, ms10 = results[label][10]
    ratio = e10 / e50
    print(f"  {label:<22}  {n_param:>7,}  "
          f"{e50:>12.2f}s  {ms50:>8.2f}  "
          f"{e10:>12.2f}s  {ms10:>8.2f}  {ratio:>9.2f}×")
print("─" * W)
print(f"  {'batch_leapfrog':<22}  {'—':>7}  "
      f"{'—':>13}  {'—':>8}  "
      f"{'463.52s':>13}  {'2.30':>8}  "
      f"{'(ref)':>10}  ← dt_factor=1, n_phys≈200,000")
print("─" * W)
