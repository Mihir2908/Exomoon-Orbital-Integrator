"""
benchmark_hnn_128_64.py — Time [15→128→64→1] at dt_factor=10 and dt_factor=50.

Cross-checks against prior measurements:
  dt_factor=50: [64×2] gave 9.92s, [128×2] gave 14.31s  → expect ~12s for [128→64]
  dt_factor=10: [64×2] gave 49.67s, [128×2] gave 75.32s  → expect ~60s for [128→64]

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
    [1.0, 3.0, 0.2, 0.018, 0.99, 1.80],
    [0.5, 10.0, 0.5, 0.005, 0.20, 0.40],
    [1.5, 100.0, 1.0, 0.050, 1.50, 2.80],
])

# Custom asymmetric architecture: [15→128→64→1]
# HNN.__init__ takes hidden + layers, which builds equal-width layers.
# We need to override net directly for the asymmetric case.
import torch
import torch.nn as nn
import json as _json

class HNN_128_64(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(15, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )
    def forward(self, q, sys_params):
        x = torch.cat([q, sys_params], dim=-1)
        return self.net(x).squeeze(-1)
    def accel(self, q, sys_params, create_graph=False):
        import torch
        with torch.enable_grad():
            q_in = q.detach().requires_grad_(True)
            V    = self.forward(q_in, sys_params)
            dV   = torch.autograd.grad(V.sum(), q_in, create_graph=create_graph)[0]
        return -dV.detach() if not create_graph else -dV
    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(out_dir, "hnn_model.pt"))
        with open(os.path.join(out_dir, "hnn_config.json"), "w") as f:
            _json.dump({"rnn_type": "custom_128_64", "hidden": 128,
                        "layers": 2, "system_dim": 6, "q_dim": 9}, f)

n_param = sum(p.numel() for p in HNN_128_64().parameters() if p.requires_grad)
print(f"Architecture: 15→128→64→1   params={n_param:,}")
print(f"Reference    [15→64→64→1]:  params=5,249  | dt50=9.92s  dt10=49.67s")
print(f"Reference    [15→128×2→1]:  params=18,689 | dt50=14.31s dt10=75.32s")
print(f"batch_leapfrog: 463.52s @ dt_factor=1  (2.30 ms/step)")
print()

results = {}

for dt_factor in [50, 10]:
    tmpdir = tempfile.mkdtemp(prefix="hnn_12864_")
    model  = HNN_128_64()
    model.save(tmpdir)
    scaler = StandardScaler()
    scaler.fit(DUMMY_SYS)
    save_sys_scaler(scaler, tmpdir)

    # Patch hnn_config so hnn_inference loads the custom model
    # hnn_inference uses HNN.load() which reads hnn_config.json then loads weights.
    # We need to ensure it finds our custom weights. Simplest: monkey-patch HNN.load.
    import exomoon.ml.hnn_model as _hnn_mod
    _orig_load = _hnn_mod.HNN.load
    def _patched_load(model_dir, map_location="cpu"):
        m = HNN_128_64()
        m.load_state_dict(torch.load(
            os.path.join(model_dir, "hnn_model.pt"), map_location=map_location, weights_only=True))
        m.eval()
        return m
    _hnn_mod.HNN.load = staticmethod(_patched_load)

    t0 = time.perf_counter()
    r  = batch_hnn_trajectories(
        system_params=SYSTEM_PARAMS, t_sim=T_SIM,
        mm_resolution=MM_RES, am_resolution=AM_RES, n_steps=1000,
        model_dir=tmpdir, dt_factor=dt_factor,
    )
    elapsed = time.perf_counter() - t0

    _hnn_mod.HNN.load = _orig_load  # restore

    n_phys      = r["n_phys"]
    ms_per_step = elapsed / n_phys * 1000
    projected   = ms_per_step * (n_phys * dt_factor) / 1000

    results[dt_factor] = (elapsed, n_phys, ms_per_step, projected)
    print(f"  dt_factor={dt_factor:>2}  n_phys={n_phys}  elapsed={elapsed:.2f}s  "
          f"{ms_per_step:.2f} ms/step  → projected @dt_factor=1: {projected:.0f}s")

print()
print("─" * 70)
print(f"  dt50 elapsed : {results[50][0]:.2f}s   ({results[50][2]:.2f} ms/step)")
print(f"  dt10 elapsed : {results[10][0]:.2f}s   ({results[10][2]:.2f} ms/step)")
ratio = results[10][0] / results[50][0]
print(f"  dt10/dt50 ratio: {ratio:.2f}×  (expect 5× if linear)")
print(f"  Projected @dt_factor=1:  dt50→{results[50][3]:.0f}s  |  dt10→{results[10][3]:.0f}s")
print(f"  batch_leapfrog reference: 463s")
print("─" * 70)
