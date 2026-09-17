"""
diagnose_hnn_2d.py — 2D force-error heatmap for HNN v12 over the full
                     (mm_earth x am_hill) inference grid.

For each grid cell (mm_i, am_j) the model sees:
  d_pm_n  = am_j          (initial normalised moon-planet distance)
  sys_enc = f(ms, mp, mm_i, ap, rhill)

Analytical target:
  t_pm = (mm/ms) * (ap/rhill) / am_j^2

d_sp_n and d_sm_n are held at 1.0 (planet at periapsis, star-moon ~= star-planet).

Outputs:
  - Console: min/max/mean absolute error summary per mm row
  - PNG: heatmap saved to diagnose_hnn_2d_<model>.png
"""

import os, sys, json, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from exomoon.ml.hnn_model            import HNN
from exomoon.ml.hnn_factorized_model import HNNFactorized
from exomoon.ml.hnn_dataset          import load_sys_scaler, _MERTH_OVER_MSUN

ap = argparse.ArgumentParser()
ap.add_argument("--model_dir",   default="models_hnn_dist_v12")
ap.add_argument("--mm_res",      type=int,   default=50)
ap.add_argument("--am_res",      type=int,   default=50)
ap.add_argument("--system",      default="kepler452b",
                help="kepler452b | kepler1229b | custom")
ap.add_argument("--ms_solar",    type=float, default=None)
ap.add_argument("--mp_earth",    type=float, default=None)
ap.add_argument("--ap_AU",       type=float, default=None)
ap.add_argument("--ep",          type=float, default=0.0)
args = ap.parse_args()

# ── System parameters ─────────────────────────────────────────────────────────
SYSTEMS = {
    "kepler452b":  dict(ms_solar=1.037, mp_earth=3.29,  ap_AU=1.200, ep=0.00),
    "kepler1229b": dict(ms_solar=0.540, mp_earth=2.72,  ap_AU=0.301, ep=0.00),
}

if args.system == "custom":
    assert args.ms_solar and args.mp_earth and args.ap_AU, \
        "Custom system needs --ms_solar, --mp_earth, --ap_AU"
    sys_p = dict(ms_solar=args.ms_solar, mp_earth=args.mp_earth,
                 ap_AU=args.ap_AU, ep=args.ep)
else:
    sys_p = SYSTEMS[args.system]

ms_solar = sys_p["ms_solar"]
mp_earth = sys_p["mp_earth"]
ap_AU    = sys_p["ap_AU"]
ep       = sys_p["ep"]

mp_msun  = mp_earth * _MERTH_OVER_MSUN
ms_msun  = ms_solar
rhill_AU = ap_AU * (1 - ep) * (mp_msun / (3 * ms_msun)) ** (1/3)

MODEL_DIR = os.path.join(SRC, args.model_dir)
model_tag = args.model_dir.replace("models_hnn_dist_", "")

print(f"Model  : {args.model_dir}")
print(f"System : ms={ms_solar} M☉  mp={mp_earth} M⊕  ap={ap_AU} AU"
      f"  rhill={rhill_AU:.5f} AU")
print(f"Grid   : {args.mm_res} mm × {args.am_res} am")
print()

# ── Load model ────────────────────────────────────────────────────────────────
if os.path.exists(os.path.join(MODEL_DIR, "hnn_factorized_config.json")):
    model = HNNFactorized.load(MODEL_DIR)
else:
    model = HNN.load(MODEL_DIR)
model.eval()
sys_scaler = load_sys_scaler(MODEL_DIR)

# ── Build grids ───────────────────────────────────────────────────────────────
mm_max  = min(mp_earth * 0.30, 0.5)
mm_grid = np.exp(np.linspace(np.log(0.107), np.log(mm_max), args.mm_res))

# Roche limit fraction (same formula as inference.py)
roche_frac = (2 * (mm_grid[0] * _MERTH_OVER_MSUN / ms_msun) ** (1/3))
am_min  = max(roche_frac, 0.01)
am_grid = np.linspace(am_min, 1.0, args.am_res)

d_sp_n = 1.0   # fixed: planet at semi-major axis
d_sm_n = 1.0   # fixed: star-moon ≈ star-planet

# ── Sweep ──────────────────────────────────────────────────────────────────────
err_pm_grid   = np.zeros((args.mm_res, args.am_res))   # signed %
abserr_pm_grid = np.zeros((args.mm_res, args.am_res))

print(f"{'mm_earth':>10}  {'am_min_err%':>12}  {'am_max_err%':>12}  {'am_mean|err|%':>14}  {'am_rms_err%':>12}")
print("-" * 66)

with torch.no_grad():
    pass  # we need grad for HNN; use enable_grad below

for i, mm_earth in enumerate(mm_grid):
    mm_msun = mm_earth * _MERTH_OVER_MSUN

    sys_raw = np.array([[ms_solar, mp_earth, mm_earth, ap_AU, rhill_AU]])
    sys_enc = torch.tensor(sys_scaler.transform(sys_raw).astype(np.float32))  # (1,5)

    row_errs = []
    for j, am_hill in enumerate(am_grid):
        d_pm_n = am_hill

        t_pm_true = (mm_msun / ms_msun) * (ap_AU / rhill_AU) / d_pm_n**2

        d_n  = torch.tensor([[d_sp_n, d_sm_n, d_pm_n]], dtype=torch.float32)
        d_leaf = d_n.detach().requires_grad_(True)

        with torch.enable_grad():
            V    = model(d_leaf, sys_enc)
            grad = torch.autograd.grad(V.sum(), d_leaf)[0]

        pred_pm = grad[0, 2].item()
        err_pct = (pred_pm - t_pm_true) / (t_pm_true + 1e-30) * 100.0

        err_pm_grid[i, j]    = err_pct
        abserr_pm_grid[i, j] = abs(err_pct)
        row_errs.append(err_pct)

    row_errs = np.array(row_errs)
    print(f"{mm_earth:>10.4f}  {row_errs.min():>+12.1f}  {row_errs.max():>+12.1f}"
          f"  {np.abs(row_errs).mean():>14.1f}  {np.sqrt((row_errs**2).mean()):>12.1f}")

print()
print(f"Overall: mean |err_pm|% = {abserr_pm_grid.mean():.1f}%"
      f"  max |err_pm|% = {abserr_pm_grid.max():.1f}%"
      f"  rms = {np.sqrt((err_pm_grid**2).mean()):.1f}%")

# ── Heatmap ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"HNN {model_tag} — pred_pm error over (mm_earth × am_hill) grid\n"
             f"System: {args.system}  ms={ms_solar}M☉  mp={mp_earth}M⊕  ap={ap_AU}AU",
             fontsize=11)

# Panel 1: signed error (diverging, capped at ±50%)
vmax = 50.0
cmap1 = plt.cm.RdBu_r
norm1 = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
im1 = axes[0].imshow(
    err_pm_grid, origin="lower", aspect="auto",
    extent=[am_grid[0], am_grid[-1], np.log10(mm_grid[0]), np.log10(mm_grid[-1])],
    cmap=cmap1, norm=norm1
)
axes[0].set_xlabel("am_hill (Hill radii)")
axes[0].set_ylabel("log10(mm_earth / M⊕)")
axes[0].set_title("Signed error: (pred_pm − t_pm) / t_pm  [%]  (capped ±50%)")
plt.colorbar(im1, ax=axes[0], label="error %")
axes[0].axvline(x=0.516, color="white", linestyle="--", linewidth=0.8, alpha=0.7,
                label="am=0.516 (v12 ref)")
axes[0].legend(fontsize=8)

# Panel 2: absolute error
cmap2 = plt.cm.YlOrRd
im2 = axes[1].imshow(
    abserr_pm_grid, origin="lower", aspect="auto",
    extent=[am_grid[0], am_grid[-1], np.log10(mm_grid[0]), np.log10(mm_grid[-1])],
    cmap=cmap2, vmin=0, vmax=vmax
)
axes[1].set_xlabel("am_hill (Hill radii)")
axes[1].set_ylabel("log10(mm_earth / M⊕)")
axes[1].set_title("|error| [%]  (capped at 50%)")
plt.colorbar(im2, ax=axes[1], label="|error| %")

# Contour at 15% and 30% absolute error
try:
    cs = axes[1].contour(
        am_grid,
        np.log10(mm_grid),
        abserr_pm_grid,
        levels=[15, 30],
        colors=["yellow", "red"],
        linewidths=1.0
    )
    axes[1].clabel(cs, fmt="%d%%", fontsize=8)
except Exception:
    pass

plt.tight_layout()
out_png = os.path.join(SRC, f"diagnose_hnn_2d_{model_tag}.png")
plt.savefig(out_png, dpi=120, bbox_inches="tight")
print(f"Saved: {out_png}")
