"""
diagnose_hnn_dist.py — Check what the HNN actually predicts for gradient components.

Loads the model and evaluates predicted [t_sp, t_sm, t_pm] vs analytical truth
at the exact configuration of the compare_hnn_dist.py test cell (mm_idx=25, am_idx=25)
across a range of d_pm_n values. Reveals whether t_pm is still ~0 or learned.
"""

import os, sys, json, argparse
import numpy as np
import torch

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from exomoon.ml.hnn_model             import HNN
from exomoon.ml.hnn_factorized_model  import HNNFactorized
from exomoon.ml.hnn_dataset           import load_sys_scaler, _MERTH_OVER_MSUN

ap = argparse.ArgumentParser()
ap.add_argument("--model_dir", default="models_hnn_dist_v3")
args = ap.parse_args()

MODEL_DIR = os.path.join(SRC, args.model_dir)
GT_META   = os.path.join(SRC, "ground_truth_grids", "Kepler_452_b_v2_prograde.meta.json")

with open(GT_META) as f:
    gt = json.load(f)
sp = gt["system_params"]
ms_solar = float(sp["ms_solar"])
mp_earth = float(sp["mp_earth"])
ap_AU    = float(sp["ap_AU"])
ep       = float(sp.get("ep", 0.0))

# Test cell: mm_idx=25 in 50-point log grid from 0.107 to min(mp*0.3, 0.5)
mm_max   = min(mp_earth * 0.30, 0.5)
mm_grid  = np.exp(np.linspace(np.log(0.107), np.log(mm_max), 50))
mm_earth = float(mm_grid[25])

# Hill radius
mp_msun  = mp_earth * _MERTH_OVER_MSUN
ms_msun  = ms_solar
rhill_AU = ap_AU * (1 - ep) * (mp_msun / (3 * ms_msun)) ** (1/3)

print(f"Model: {args.model_dir}")
print(f"System: ms={ms_solar} M☉  mp={mp_earth} M⊕  ap={ap_AU} AU  rhill={rhill_AU:.5f} AU")
print(f"Test cell: mm_earth={mm_earth:.5f} M⊕")
print()

# Load model + scaler (auto-detect factorized vs monolithic)
if os.path.exists(os.path.join(MODEL_DIR, "hnn_factorized_config.json")):
    model = HNNFactorized.load(MODEL_DIR)
else:
    model = HNN.load(MODEL_DIR)
model.eval()
sys_scaler = load_sys_scaler(MODEL_DIR)

# Build sys_enc for this one cell
sys_raw = np.array([[ms_solar, mp_earth, mm_earth, ap_AU, rhill_AU]])
sys_enc = torch.tensor(sys_scaler.transform(sys_raw).astype(np.float32))  # (1, 5)

# Analytical targets at d_sp_n=1, d_sm_n=1 (typical) as d_pm_n varies
d_sp_n_val = 1.0   # star-planet at semi-major axis
d_sm_n_val = 1.0   # star-moon at same distance (approx)

mm_msun = mm_earth * _MERTH_OVER_MSUN

# Analytical:
t_sp_true = 1.0 / d_sp_n_val**2                           # (ap/d_sp)²
t_sm_true = (mm_msun / mp_msun) / d_sm_n_val**2           # (mm/mp)/d_sm_n²
# t_pm varies with d_pm_n

print(f"  Analytical t_sp (d_sp_n=1.0): {t_sp_true:.6f}")
print(f"  Analytical t_sm (d_sm_n=1.0): {t_sm_true:.6f}")
print()

header = f"{'d_pm_n':>8}  {'t_pm_true':>12}  {'pred_sp':>10}  {'pred_sm':>10}  {'pred_pm':>10}  {'err_sp%':>8}  {'err_sm%':>8}  {'err_pm%':>8}"
print(header)
print("-" * len(header))

d_pm_n_values = [0.05, 0.10, 0.20, 0.30, 0.40, 0.516, 0.60, 0.70, 0.80, 0.90, 1.00]

for d_pm_n_val in d_pm_n_values:
    t_pm_true = (mm_msun / ms_msun) * (ap_AU / rhill_AU) / d_pm_n_val**2

    d_n = torch.tensor([[d_sp_n_val, d_sm_n_val, d_pm_n_val]], dtype=torch.float32)

    d_leaf = d_n.detach().requires_grad_(True)
    V      = model(d_leaf, sys_enc)
    grad   = torch.autograd.grad(V.sum(), d_leaf)[0]  # (1, 3)

    pred_sp, pred_sm, pred_pm = grad[0].tolist()

    err_sp = (pred_sp - t_sp_true) / (t_sp_true + 1e-12) * 100
    err_sm = (pred_sm - t_sm_true) / (t_sm_true + 1e-12) * 100
    err_pm = (pred_pm - t_pm_true) / (t_pm_true + 1e-12) * 100

    print(f"{d_pm_n_val:>8.3f}  {t_pm_true:>12.6f}  {pred_sp:>10.6f}  {pred_sm:>10.6f}  {pred_pm:>10.6f}  {err_sp:>8.1f}%  {err_sm:>8.1f}%  {err_pm:>8.1f}%")

print()
print("If pred_pm ≈ 0 across all rows: model never learned t_pm (scale issue persists).")
print("If pred_pm has wrong sign or magnitude: training converged to wrong solution.")
print("If pred_pm ≈ t_pm_true: model learned correctly, failure is elsewhere.")
