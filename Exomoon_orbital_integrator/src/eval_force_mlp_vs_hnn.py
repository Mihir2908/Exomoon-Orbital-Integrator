"""
eval_force_mlp_vs_hnn.py — Side-by-side force prediction accuracy:
  HNN v12  (models_hnn_dist_v12/)  vs  Force regression MLP  (models_force_mlp/)

Metric: signed error% = (pred - true) / true × 100   for each of t_sp, t_sm, t_pm.
Summary: % of (mm_earth, am_hill) grid cells within 15% / 30% / 50% error.

Runs across all unique systems found in ground_truth_grids/*.meta.json
(deduplicated by system params — prograde and retrograde share the same star/planet,
so HNN sys_enc is identical for both; one pass per unique system is sufficient).

This is the same breakpoint-grid metric used in diagnose_hnn_2d_table.py for HNN v12.

ISOLATION: does not touch models/, models_temphead/, eval_aux_mlp_output/.
"""

import os, sys, json, argparse
import numpy as np
import torch

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from exomoon.ml.hnn_model            import HNN
from exomoon.ml.hnn_dataset          import load_sys_scaler, _MERTH_OVER_MSUN
from exomoon.ml.force_mlp_model      import ForceMLPEnsemble

ap = argparse.ArgumentParser()
ap.add_argument("--hnn_dir",   default="models_hnn_dist_v12",
                help="Directory for HNN v12 weights")
ap.add_argument("--fmlp_dir",  default="models_force_mlp",
                help="Directory for force regression MLP weights")
ap.add_argument("--meta",      default=None,
                help="Single *.meta.json to evaluate; omit for all systems")
args = ap.parse_args()

GT_DIR   = os.path.join(SRC, "ground_truth_grids")
HNN_DIR  = os.path.join(SRC, args.hnn_dir)
FMLP_DIR = os.path.join(SRC, args.fmlp_dir)


# ── Load models ───────────────────────────────────────────────────────────────

print(f"HNN  dir : {args.hnn_dir}")
print(f"FMLP dir : {args.fmlp_dir}")
print()

hnn_model  = HNN.load(HNN_DIR)
hnn_model.eval()
hnn_scaler = load_sys_scaler(HNN_DIR)

fmlp       = ForceMLPEnsemble.load(FMLP_DIR)
fmlp.eval()
fmlp_scaler = load_sys_scaler(FMLP_DIR)


# ── Collect systems ───────────────────────────────────────────────────────────

def _load_meta(path):
    with open(path) as f:
        d = json.load(f)
    sp = d["system_params"]
    return dict(
        ms_solar = float(sp["ms_solar"]),
        mp_earth = float(sp["mp_earth"]),
        ap_AU    = float(sp["ap_AU"]),
        ep       = float(sp.get("ep", 0.0)),
    )

if args.meta:
    meta_files = [args.meta]
else:
    meta_files = sorted(
        os.path.join(GT_DIR, f)
        for f in os.listdir(GT_DIR)
        if f.endswith(".meta.json") and "prograde" in f
    )

systems = []
seen_keys = set()
for mf in meta_files:
    sp  = _load_meta(mf)
    key = (round(sp["ms_solar"], 3), round(sp["mp_earth"], 2),
           round(sp["ap_AU"], 4),    round(sp["ep"], 3))
    if key not in seen_keys:
        seen_keys.add(key)
        label = (os.path.basename(mf)
                 .replace("_prograde.meta.json", "")
                 .replace(".meta.json", ""))
        systems.append((label, sp))

print(f"Systems  : {len(systems)}")
print()


# ── Test grid ────────────────────────────────────────────────────────────────

AM_VALS = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40,
                    0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
MM_BASE = np.array([0.107, 0.150, 0.200, 0.300, 0.400, 0.500,
                    0.700, 1.000, 1.500, 2.000, 2.500])

D_SP_N = 1.0   # fixed to isolate t_pm variation (same as diagnose_hnn_2d_table.py)
D_SM_N = 1.0


def pct_within(arr, thresh):
    return (np.abs(arr) < thresh).mean() * 100


# ── HNN force prediction at a single (d_sp_n, d_sm_n, d_pm_n) ───────────────

def hnn_pred_forces(hnn, sys_enc_t, d_sp_n, d_sm_n, d_pm_n):
    """Return (pred_sp, pred_sm, pred_pm) from HNN autograd."""
    d_n    = torch.tensor([[d_sp_n, d_sm_n, d_pm_n]], dtype=torch.float32)
    d_leaf = d_n.detach().requires_grad_(True)
    with torch.enable_grad():
        V    = hnn(d_leaf, sys_enc_t)
        grad = torch.autograd.grad(V.sum(), d_leaf)[0]
    return grad[0, 0].item(), grad[0, 1].item(), grad[0, 2].item()


# ── Main loop ─────────────────────────────────────────────────────────────────

PAIRS = ("sp", "sm", "pm")

# Accumulate per-pair errors across all systems for grand summary
grand_hnn  = {p: [] for p in PAIRS}
grand_fmlp = {p: [] for p in PAIRS}

SEP  = "=" * 110
DSEP = "-" * 110

for sys_idx, (sys_label, sys_p) in enumerate(systems):
    ms_solar = sys_p["ms_solar"]
    mp_earth = sys_p["mp_earth"]
    ap_AU    = sys_p["ap_AU"]
    ep       = sys_p["ep"]

    mp_msun  = mp_earth * _MERTH_OVER_MSUN
    rhill_AU = ap_AU * (1 - ep) * (mp_msun / (3 * ms_solar)) ** (1 / 3)
    mm_max   = min(mp_earth, 3.0)
    mm_vals  = np.unique(np.clip(MM_BASE, 0.107, mm_max))

    # System-level sys_enc for HNN and force MLP share the same 5-dim format
    # [ms_solar, mp_earth, mm_earth, ap_AU, rhill_AU] — mm_earth varies per row
    if sys_idx > 0:
        print()
    print(SEP)
    print(f"  SYSTEM  : {sys_label}")
    print(f"  Params  : ms={ms_solar}Msun  mp={mp_earth}Mearth  ap={ap_AU}AU  ep={ep}"
          f"  rhill={rhill_AU:.5f}AU  mm_max={mm_max:.3f}Mearth")
    print(SEP)

    # Per-pair error arrays for this system
    sys_hnn  = {p: [] for p in PAIRS}
    sys_fmlp = {p: [] for p in PAIRS}

    for mm_earth in mm_vals:
        mm_msun = mm_earth * _MERTH_OVER_MSUN

        # True (analytical) force targets
        t_sp_true = {am: 1.0 / D_SP_N ** 2                              for am in AM_VALS}
        t_sm_true = {am: (mm_msun / mp_msun)  / D_SM_N ** 2             for am in AM_VALS}
        t_pm_true = {am: (mm_msun / ms_solar) * (ap_AU / rhill_AU) / am**2  for am in AM_VALS}

        # HNN sys_enc
        sys_raw_hnn  = np.array([[ms_solar, mp_earth, mm_earth, ap_AU, rhill_AU]])
        sys_enc_hnn  = torch.tensor(
            hnn_scaler.transform(sys_raw_hnn).astype(np.float32)
        )

        # Force MLP sys_enc (same 5-dim, same format, different scaler)
        sys_raw_fmlp = sys_raw_hnn.copy()
        sys_enc_fmlp = torch.tensor(
            fmlp_scaler.transform(sys_raw_fmlp).astype(np.float32)
        )

        for am_hill in AM_VALS:
            # HNN predictions
            h_sp, h_sm, h_pm = hnn_pred_forces(
                hnn_model, sys_enc_hnn, D_SP_N, D_SM_N, am_hill
            )

            # Force MLP predictions
            d_n_t = torch.tensor([[D_SP_N, D_SM_N, am_hill]], dtype=torch.float32)
            with torch.no_grad():
                f_out = fmlp.forward(d_n_t, sys_enc_fmlp)
            f_sp = f_out[0, 0].item()
            f_sm = f_out[0, 1].item()
            f_pm = f_out[0, 2].item()

            # Error%
            for pair, h, f, t in [
                ("sp", h_sp, f_sp, t_sp_true[am_hill]),
                ("sm", h_sm, f_sm, t_sm_true[am_hill]),
                ("pm", h_pm, f_pm, t_pm_true[am_hill]),
            ]:
                h_err = (h - t) / (t + 1e-30) * 100
                f_err = (f - t) / (t + 1e-30) * 100
                sys_hnn[pair].append(h_err)
                sys_fmlp[pair].append(f_err)

    # ── Per-system summary ────────────────────────────────────────────────────

    print(f"\n  {'':30}  {'HNN v12':^38}  {'Force MLP':^38}")
    print(f"  {'Pair':30}  {'mean|e|':>8}  {'rms':>7}  {'<15%':>6}  {'<30%':>6}  {'<50%':>6}"
          f"  {'mean|e|':>8}  {'rms':>7}  {'<15%':>6}  {'<30%':>6}  {'<50%':>6}")
    print("  " + DSEP[:106])

    for pair in PAIRS:
        h_arr = np.array(sys_hnn[pair])
        f_arr = np.array(sys_fmlp[pair])
        grand_hnn[pair].extend(h_arr.tolist())
        grand_fmlp[pair].extend(f_arr.tolist())

        label = f"t_{pair}"
        def _fmt(arr):
            return (f"{np.abs(arr).mean():>8.1f}%  {np.sqrt((arr**2).mean()):>6.1f}%"
                    f"  {pct_within(arr,15):>5.0f}%  {pct_within(arr,30):>5.0f}%"
                    f"  {pct_within(arr,50):>5.0f}%")
        print(f"  {label:<30}  {_fmt(h_arr)}  {_fmt(f_arr)}")


# ── Grand summary across all systems ─────────────────────────────────────────

print()
print(SEP)
print("  GRAND SUMMARY — all systems combined")
print(SEP)
print(f"\n  {'':30}  {'HNN v12':^38}  {'Force MLP':^38}")
print(f"  {'Pair':30}  {'mean|e|':>8}  {'rms':>7}  {'<15%':>6}  {'<30%':>6}  {'<50%':>6}"
      f"  {'mean|e|':>8}  {'rms':>7}  {'<15%':>6}  {'<30%':>6}  {'<50%':>6}")
print("  " + DSEP[:106])

for pair in PAIRS:
    h_arr = np.array(grand_hnn[pair])
    f_arr = np.array(grand_fmlp[pair])
    label = f"t_{pair}"
    def _fmt(arr):
        return (f"{np.abs(arr).mean():>8.1f}%  {np.sqrt((arr**2).mean()):>6.1f}%"
                f"  {pct_within(arr,15):>5.0f}%  {pct_within(arr,30):>5.0f}%"
                f"  {pct_within(arr,50):>5.0f}%")
    print(f"  {label:<30}  {_fmt(h_arr)}  {_fmt(f_arr)}")

print()
