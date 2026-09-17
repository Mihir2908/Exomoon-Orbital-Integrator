"""
diagnose_hnn_2d_table.py — Detailed 2D pred_pm error table for HNN v12.

Prints signed error% at a set of representative (mm_earth, am_hill) breakpoints,
plus per-row and per-column summary statistics.

Legend for table cells:
  |err| <  15%  →  [  ok  ]   acceptable for trajectory preview
  |err| <  30%  →  [ warn ]   noticeable drift over multi-orbit preview
  |err| >= 30%  →  [ FAIL ]   force prediction unreliable
"""

import os, sys, json, argparse
import numpy as np
import torch

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from exomoon.ml.hnn_model            import HNN
from exomoon.ml.hnn_factorized_model import HNNFactorized
from exomoon.ml.hnn_dataset          import load_sys_scaler, _MERTH_OVER_MSUN

ap = argparse.ArgumentParser()
ap.add_argument("--model_dir", default="models_hnn_dist_v12")
ap.add_argument("--meta",      default=None,
                help="Path to a ground_truth_grids/*.meta.json file. "
                     "If omitted, runs all meta files found in ground_truth_grids/.")
args = ap.parse_args()

GT_DIR = os.path.join(SRC, "ground_truth_grids")

def load_meta(path):
    with open(path) as f:
        d = json.load(f)
    sp = d["system_params"]
    return dict(
        ms_solar = float(sp["ms_solar"]),
        mp_earth = float(sp["mp_earth"]),
        ap_AU    = float(sp["ap_AU"]),
        ep       = float(sp.get("ep", 0.0)),
    )

# Collect unique systems (deduplicate by rounded key; prograde==retrograde for HNN)
if args.meta:
    meta_files = [args.meta]
else:
    meta_files = sorted(
        p for p in (os.path.join(GT_DIR, f) for f in os.listdir(GT_DIR)
                    if f.endswith(".meta.json"))
        if "prograde" in p   # one per system (pro==retro for HNN sys_enc)
    )

systems_to_run = []
seen_keys = set()
for mf in meta_files:
    sp = load_meta(mf)
    key = (round(sp["ms_solar"],3), round(sp["mp_earth"],2),
           round(sp["ap_AU"],4),    round(sp["ep"],3))
    if key not in seen_keys:
        seen_keys.add(key)
        label = os.path.basename(mf).replace("_prograde.meta.json","").replace(".meta.json","")
        systems_to_run.append((label, sp))

# ── Load model once (same model for all systems) ─────────────────────────────
MODEL_DIR = os.path.join(SRC, args.model_dir)
if os.path.exists(os.path.join(MODEL_DIR, "hnn_factorized_config.json")):
    model = HNNFactorized.load(MODEL_DIR)
else:
    model = HNN.load(MODEL_DIR)
model.eval()
sys_scaler = load_sys_scaler(MODEL_DIR)

print(f"Model   : {args.model_dir}")
print(f"Running {len(systems_to_run)} unique system(s)")
print()

def tag(err_pct):
    a = abs(err_pct)
    if a < 15:   return "ok  "
    if a < 30:   return "warn"
    return           "FAIL"

def fmt(err_pct):
    s = f"{err_pct:+.1f}%"
    return f"{s:>8}"

def pct_within(arr, thresh):
    return (np.abs(arr) < thresh).mean() * 100

# ── Test points (am axis) ─────────────────────────────────────────────────────
am_vals = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50,
                    0.60, 0.70, 0.80, 0.90, 1.00])

d_sp_n = 1.0
d_sm_n = 1.0

# ── Run one system at a time ──────────────────────────────────────────────────
for sys_idx, (sys_label, sys_p) in enumerate(systems_to_run):
    ms_solar = sys_p["ms_solar"]
    mp_earth = sys_p["mp_earth"]
    ap_AU    = sys_p["ap_AU"]
    ep       = sys_p["ep"]

    mp_msun  = mp_earth * _MERTH_OVER_MSUN
    ms_msun  = ms_solar
    rhill_AU = ap_AU * (1 - ep) * (mp_msun / (3 * ms_msun)) ** (1/3)

    # ── Test points (mm axis, per system) ────────────────────────────────────
    mm_max  = min(mp_earth, 3.0)           # matches run_ml_dataset.py MM_MAX_CAP
    mm_vals = np.array([0.107, 0.150, 0.200, 0.300, 0.400, 0.500,
                        0.700, 1.000, 1.500, 2.000, 2.500, mm_max])
    mm_vals = np.unique(np.clip(mm_vals, 0.107, mm_max))

    # ── Header ────────────────────────────────────────────────────────────────
    sep = "=" * 100
    if sys_idx > 0:
        print()
        print(sep)
    print(f"\nSYSTEM  : {sys_label}")
    print(f"Params  : ms={ms_solar}M☉  mp={mp_earth}M⊕  ap={ap_AU}AU  ep={ep}"
          f"  rhill={rhill_AU:.5f}AU")
    print(f"mm_max  : {mm_max:.3f} M⊕  (= min(mp_earth, 3.0))")
    print(f"Model   : {args.model_dir}")
    print()
    print("Signed error: (pred_pm − t_pm) / t_pm × 100%")
    print("Cells flagged: |err|<15% → ok | 15-30% → warn | >30% → FAIL")
    print()

    # Column header
    print(f"{'mm_earth':>10}  |  " + "  ".join(f"{'am='+f'{v:.2f}':>9}" for v in am_vals)
          + "  |  row_mean  row_rms  row_max|e|")
    print("-" * (12 + 11 * len(am_vals) + 30))

    # ── Sweep ─────────────────────────────────────────────────────────────────
    col_errs = {v: [] for v in am_vals}
    summary  = []

    for mm_earth in mm_vals:
        mm_msun  = mm_earth * _MERTH_OVER_MSUN
        sys_raw  = np.array([[ms_solar, mp_earth, mm_earth, ap_AU, rhill_AU]])
        sys_enc  = torch.tensor(sys_scaler.transform(sys_raw).astype(np.float32))

        row_errs = []
        cells    = []

        for am_hill in am_vals:
            t_pm_true = (mm_msun / ms_msun) * (ap_AU / rhill_AU) / am_hill**2

            d_n    = torch.tensor([[d_sp_n, d_sm_n, am_hill]], dtype=torch.float32)
            d_leaf = d_n.detach().requires_grad_(True)
            with torch.enable_grad():
                V    = model(d_leaf, sys_enc)
                grad = torch.autograd.grad(V.sum(), d_leaf)[0]

            pred_pm = grad[0, 2].item()
            err_pct = (pred_pm - t_pm_true) / (t_pm_true + 1e-30) * 100.0

            row_errs.append(err_pct)
            col_errs[am_hill].append(err_pct)
            cells.append(err_pct)

        row_arr  = np.array(row_errs)
        row_mean = np.abs(row_arr).mean()
        row_rms  = np.sqrt((row_arr**2).mean())
        row_max  = np.abs(row_arr).max()

        cell_str = "  ".join(f"{e:>+8.1f}%" for e in cells)
        print(f"{mm_earth:>10.4f}  |  {cell_str}  |  {row_mean:>8.1f}%  {row_rms:>7.1f}%  {row_max:>9.1f}%")
        summary.append((mm_earth, row_mean, row_rms, row_max))

    print("-" * (12 + 11 * len(am_vals) + 30))

    # Column summaries
    col_mean_str = "  ".join(f"{np.abs(col_errs[v]).mean():>+8.1f}%" for v in am_vals)
    col_rms_str  = "  ".join(f"{np.sqrt(np.mean(np.array(col_errs[v])**2)):>+8.1f}%" for v in am_vals)
    col_max_str  = "  ".join(f"{np.abs(col_errs[v]).max():>+8.1f}%" for v in am_vals)
    print(f"{'col mean|e|':>10}  |  {col_mean_str}")
    print(f"{'col rms':>10}  |  {col_rms_str}")
    print(f"{'col max|e|':>10}  |  {col_max_str}")

    # ── Summary by region ─────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("REGION SUMMARY")
    print("=" * 70)

    regions = [
        ("Near Roche (am < 0.15)",   am_vals < 0.15),
        ("Inner (am 0.15–0.40)",     (am_vals >= 0.15) & (am_vals <= 0.40)),
        ("Mid   (am 0.40–0.70)",     (am_vals > 0.40)  & (am_vals <= 0.70)),
        ("Outer (am 0.70–1.00)",     am_vals > 0.70),
    ]
    mm_regions = [
        ("Lower (mm < 0.30 M⊕)",        mm_vals < 0.30),
        ("Mid   (mm 0.30–1.00 M⊕)", (mm_vals >= 0.30) & (mm_vals <= 1.00)),
        ("Upper (mm > 1.00 M⊕)",        mm_vals > 1.00),
    ]

    all_errs = np.zeros((len(mm_vals), len(am_vals)))
    for i, mm in enumerate(mm_vals):
        for j, am in enumerate(am_vals):
            all_errs[i, j] = col_errs[am][i]

    hdr = f"  {'Region':<30}  {'mean|e|':>8}  {'rms':>6}  {'max|e|':>7}  {'<15%':>6}  {'<30%':>6}  {'<50%':>6}"

    print("\n-- By am_hill band (across all mm) --")
    print(hdr)
    for label, mask in regions:
        sub = all_errs[:, mask]
        print(f"  {label:<30}  {np.abs(sub).mean():>7.1f}%  {np.sqrt((sub**2).mean()):>5.1f}%"
              f"  {np.abs(sub).max():>6.1f}%"
              f"  {pct_within(sub,15):>5.0f}%"
              f"  {pct_within(sub,30):>5.0f}%"
              f"  {pct_within(sub,50):>5.0f}%")

    print("\n-- By mm_earth band (across all am) --")
    print(f"  (Note: M⊕ = Earth mass. Min=0.107 M⊕ = Mars mass. Earth Moon = 0.012 M⊕, Ganymede = 0.025 M⊕)")
    print(f"  (All grid moons are 4–{mm_max/0.025:.0f}× Ganymede. 'Lower/mid/upper' refer to grid thirds, not absolute scale.)")
    print(hdr)
    for label, mask in mm_regions:
        if not mask.any():
            continue
        sub = all_errs[mask, :]
        print(f"  {label:<30}  {np.abs(sub).mean():>7.1f}%  {np.sqrt((sub**2).mean()):>5.1f}%"
              f"  {np.abs(sub).max():>6.1f}%"
              f"  {pct_within(sub,15):>5.0f}%"
              f"  {pct_within(sub,30):>5.0f}%"
              f"  {pct_within(sub,50):>5.0f}%")

    print()
    all_flat = all_errs.flatten()
    print(f"Overall grid  mean|e|%={np.abs(all_flat).mean():.1f}%  "
          f"rms={np.sqrt((all_flat**2).mean()):.1f}%  "
          f"max={np.abs(all_flat).max():.1f}%")
    print(f"  <15%: {pct_within(all_flat,15):.0f}%"
          f"  |  <30%: {pct_within(all_flat,30):.0f}%"
          f"  |  <50%: {pct_within(all_flat,50):.0f}%"
          f"  of all grid cells")
