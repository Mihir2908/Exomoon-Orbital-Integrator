"""
benchmark_hnn_128_64_full.py — Run [15→128→64→1] at dt_factor=50 and dt_factor=10,
all physics steps, no extrapolation.

Adds to the existing sweep results for the full comparison table.
ISOLATION: does not touch any MLP infrastructure.
"""

import os, sys, json, time, tempfile
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from exomoon.ml.hnn_dataset   import save_sys_scaler, load_sys_scaler, Q_DIM, SYS_DIM
from exomoon.ml.hnn_inference import batch_hnn_trajectories
import exomoon.ml.hnn_model as _hnn_mod

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


class _HNN_128_64(nn.Module):
    """Asymmetric HNN: [Q_DIM+SYS_DIM=15 → 128 → 64 → 1] with Tanh."""

    def __init__(self):
        super().__init__()
        in_dim = Q_DIM + SYS_DIM   # 9 + 6 = 15
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.Tanh(),
            nn.Linear(128, 64),    nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, q, sys_params):
        x = torch.cat([q, sys_params], dim=-1)
        return self.net(x).squeeze(-1)

    def accel(self, q, sys_params, create_graph=False):
        with torch.enable_grad():
            q_in = q.detach().requires_grad_(True)
            V    = self.forward(q_in, sys_params)
            dV   = torch.autograd.grad(V.sum(), q_in,
                                       create_graph=create_graph)[0]
        return -dV.detach() if not create_graph else -dV

    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        torch.save(self.state_dict(),
                   os.path.join(out_dir, "hnn_model.pt"))
        with open(os.path.join(out_dir, "hnn_config.json"), "w") as f:
            json.dump({"hidden": 128, "layers": 2,
                       "q_dim": Q_DIM, "system_dim": SYS_DIM,
                       "_arch": "128_64"}, f)


def _load_128_64(model_dir, map_location="cpu"):
    m = _HNN_128_64()
    m.load_state_dict(torch.load(
        os.path.join(model_dir, "hnn_model.pt"),
        map_location=map_location, weights_only=True,
    ))
    m.eval()
    return m


n_param = sum(p.numel() for p in _HNN_128_64().parameters())
print(f"Architecture: 15→128→64→1   params={n_param:,}")
print(f"dt_factor=50 → n_phys≈4,000   dt_factor=10 → n_phys≈20,000")
print(f"System: K-452b-v2  |  {N_CELLS} cells  |  t_sim={T_SIM} yr")
print()

results = {}

for dt_factor in [50, 10]:
    tmpdir = tempfile.mkdtemp(prefix="hnn12864_")

    model = _HNN_128_64()
    model.save(tmpdir)

    scaler = StandardScaler()
    scaler.fit(DUMMY_SYS)
    save_sys_scaler(scaler, tmpdir)

    # Patch HNN.load so batch_hnn_trajectories loads our custom architecture
    _orig_load = _hnn_mod.HNN.load
    _hnn_mod.HNN.load = staticmethod(_load_128_64)

    t0      = time.perf_counter()
    r       = batch_hnn_trajectories(
        system_params=SYSTEM_PARAMS, t_sim=T_SIM,
        mm_resolution=MM_RES, am_resolution=AM_RES, n_steps=1000,
        model_dir=tmpdir, dt_factor=dt_factor,
    )
    elapsed = time.perf_counter() - t0

    _hnn_mod.HNN.load = _orig_load   # restore

    n_phys  = r["n_phys"]
    ms_step = elapsed / n_phys * 1000
    results[dt_factor] = (elapsed, n_phys, ms_step)

    print(f"  dt_factor={dt_factor:>2}  n_phys={n_phys:>6}  "
          f"elapsed={elapsed:>7.2f}s  {ms_step:.2f} ms/step")

print()
ratio = results[10][0] / results[50][0]
print(f"  dt10 / dt50 ratio: {ratio:.2f}×  (expect 5× if linear)")
print()

# ── Full combined table ─────────────────────────────────────────────────────────
prior = [
    ("15→16→1",          273,      3.68, 0.92,  17.24, 0.86),
    ("15→32→1",          545,      4.24, 1.06,  20.32, 1.02),
    ("15→32→32→1",     1_601,      5.26, 1.31,  22.61, 1.13),
    ("15→64→64→1",     5_249,      6.24, 1.56,  45.09, 2.25),
    ("15→128→64→1",   n_param, results[50][0], results[50][2],
                               results[10][0], results[10][2]),
    ("15→128→128→1",  18_689,     14.50, 3.62,  67.99, 3.40),
]

W = 96
print("─" * W)
print(f"  {'Architecture':<22}  {'Params':>7}  "
      f"{'dt50 elapsed':>13}  {'ms/step':>8}  "
      f"{'dt10 elapsed':>13}  {'ms/step':>8}")
print("─" * W)
for label, params, e50, ms50, e10, ms10 in prior:
    print(f"  {label:<22}  {params:>7,}  "
          f"{e50:>12.2f}s  {ms50:>8.2f}  "
          f"{e10:>12.2f}s  {ms10:>8.2f}")
print("─" * W)
print(f"  {'batch_leapfrog':<22}  {'—':>7}  "
      f"{'—':>13}  {'—':>8}  "
      f"{'463.52s':>13}  {'2.30':>8}  ← dt_factor=1, n_phys=200,000")
print("─" * W)
