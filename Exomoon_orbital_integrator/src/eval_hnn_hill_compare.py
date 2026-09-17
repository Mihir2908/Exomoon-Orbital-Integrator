"""
eval_hnn_hill_compare.py — Gate recall comparison: hinge4 vs rollout_v2 (or any two Hill HNN dirs).

For each ground-truth grid, runs batch_hnn_hill_trajectories() for both model directories
using the SAME resolution and t_sim as the GT grid, then computes:

  stable_recall   = |gt_stable & hnn_stable| / |gt_stable|
  habitable_recall = |gt_both & hnn_both| / |gt_both|   (4-gate CLEAN-SUBSET metric)

where gt_both = gt_stable & gt_habitable, hnn_both = hnn_stable & hnn_habitable.

Prints a comparison table: dir_a vs dir_b, delta column, win/tie/lose summary.

Usage
-----
  cd Exomoon_orbital_integrator/src
  py eval_hnn_hill_compare.py
  py eval_hnn_hill_compare.py --dir_a models_hnn_hill_hinge4 --dir_b models_hnn_hill_hinge8_rollout_v2
  py eval_hnn_hill_compare.py --systems K442,K452v2,K1229,TRAPPIST   (substring filters)
  py eval_hnn_hill_compare.py --include_retro

ISOLATION: does not touch models/, models_temphead/, eval_aux_mlp_output/, or any
           directory not explicitly passed via --dir_a / --dir_b.
"""

import argparse, json, os, pickle, sys, time
import numpy as np
import torch

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from exomoon.ml.hnn_inference_hill import batch_hnn_hill_trajectories
from exomoon.ml.dataset            import SYS_COLS, LOG_SYS_COLS
from exomoon.constants             import FOUR_PI2, au, merth, msun
from exomoon.constants             import rsun as _rsun
from exomoon.habitable_zone        import hz_bounds_au

_MERTH_OVER_MSUN    = merth / msun
_PLANET_DENSITY_SI  = 5500.0
_PLANET_DENSITY_CGS = 5.5
_MOON_DENSITY_CGS   = 3.0

MLP_DIR = os.path.join(SRC, "eval_aux_mlp_output", "binary")
_mlp_model  = None
_mlp_scaler = None

def _load_mlp():
    global _mlp_model, _mlp_scaler
    if _mlp_model is not None:
        return True
    pt   = os.path.join(MLP_DIR, "aux_mlp_binary.pt")
    sc   = os.path.join(MLP_DIR, "aux_mlp_scaler.pkl")
    cfg  = os.path.join(MLP_DIR, "model_config.json")
    if not (os.path.exists(pt) and os.path.exists(sc)):
        print(f"  [eligible_mask] AuxMLPBinary not found at {MLP_DIR} — skipping mask")
        return False
    with open(cfg) as f:
        c = json.load(f)
    import torch.nn as nn
    class _AuxMLPBinary(nn.Module):
        def __init__(self, input_dim=14, hidden=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden),    nn.ReLU(),
                nn.Linear(hidden, hidden),    nn.ReLU(),
                nn.Linear(hidden, 2),
            )
        def forward(self, x): return self.net(x)
    m = _AuxMLPBinary(input_dim=c["input_dim"], hidden=c["hidden"])
    m.load_state_dict(torch.load(pt, map_location="cpu"))
    m.eval()
    _mlp_model = m
    with open(sc, "rb") as f:
        _mlp_scaler = pickle.load(f)
    return True


def _compute_eligible_mask(system_params, t_sim, moon_retrograde, mm_res, am_res, em=0.0):
    """Return (mm_res, am_res) bool array: True = MLP predicts stable+habitable."""
    if not _load_mlp():
        return None

    ms_solar = float(system_params["ms_solar"])
    rs_solar = float(system_params.get("rs_solar", 1.0))
    Ts       = float(system_params.get("Ts", 5772.0))
    mp_earth = float(system_params["mp_earth"])
    ap_AU    = float(system_params["ap_AU"])
    ep       = float(system_params.get("ep", 0.0))

    mp_kg    = mp_earth * merth
    rp_m     = (0.75 * mp_kg / (np.pi * _PLANET_DENSITY_SI)) ** (1.0 / 3.0)
    a_roche  = 2.456 * rp_m * (_PLANET_DENSITY_CGS / _MOON_DENSITY_CGS) ** (1.0 / 3.0) / au
    mp_msun  = mp_earth * _MERTH_OVER_MSUN
    rhill_AU = ap_AU * (1.0 - ep) * ((mp_msun / (3.0 * ms_solar)) ** (1.0 / 3.0))
    am_min   = max(a_roche / rhill_AU, 1e-3)
    mm_max   = min(mp_earth, 3.0)
    mm_grid  = np.exp(np.linspace(np.log(0.107), np.log(mm_max), mm_res))
    am_grid  = np.linspace(am_min, 1.0, am_res)
    a_in, a_out = hz_bounds_au(Ts, rs_solar * _rsun)
    retro    = float(int(moon_retrograde))

    vecs = []
    for mm in mm_grid:
        for am in am_grid:
            vecs.append([ms_solar, rs_solar, Ts, mp_earth, ap_AU, ep,
                         float(mm), float(am), em, retro,
                         t_sim, rhill_AU, a_in, a_out])
    X = np.array(vecs, dtype=np.float32)
    log_idx = [SYS_COLS.index(c) for c in LOG_SYS_COLS]
    X[:, log_idx] = np.log(np.clip(X[:, log_idx], 1e-10, None))
    X_sc = _mlp_scaler.transform(X).astype(np.float32)

    with torch.no_grad():
        logits = _mlp_model(torch.from_numpy(X_sc))
        probs  = torch.sigmoid(logits).numpy()
    eligible = (probs[:, 0] >= 0.5) & (probs[:, 1] >= 0.5)
    mask = eligible.reshape(mm_res, am_res)
    n_elig = int(mask.sum())
    print(f"  [eligible_mask] MLP eligible cells: {n_elig}/{mm_res * am_res} "
          f"({100.*n_elig/(mm_res*am_res):.1f}%)")
    return mask

GT_DIR = os.path.join(SRC, "ground_truth_grids")

# ── CLI ────────────────────────────────────────────────────────────────────────

ap = argparse.ArgumentParser(description="HNN Hill gate recall comparison")
ap.add_argument("--dir_a", default="models_hnn_hill_hinge4",
                help="Model A directory (baseline, default hinge4)")
ap.add_argument("--dir_b", default="models_hnn_hill_hinge8_rollout_v2",
                help="Model B directory (candidate, default rollout_v2)")
ap.add_argument("--systems", default=None,
                help="Comma-separated substrings to filter GT grid filenames "
                     "(e.g. 'K452,K1229'). Default: all prograde grids.")
ap.add_argument("--include_retro", action="store_true",
                help="Also evaluate retrograde grids (doubles runtime)")
ap.add_argument("--n_steps", type=int, default=1000,
                help="HNN leapfrog steps (default 1000 — matches parquet training resolution)")
args = ap.parse_args()

DIR_A = args.dir_a if os.path.isabs(args.dir_a) else os.path.join(SRC, args.dir_a)
DIR_B = args.dir_b if os.path.isabs(args.dir_b) else os.path.join(SRC, args.dir_b)


# ── Discover GT grids ──────────────────────────────────────────────────────────

all_npz = sorted(f for f in os.listdir(GT_DIR) if f.endswith(".npz"))
if not args.include_retro:
    all_npz = [f for f in all_npz if "prograde" in f]

if args.systems:
    filters = [s.strip() for s in args.systems.split(",")]
    all_npz = [f for f in all_npz if any(flt.lower() in f.lower() for flt in filters)]

if not all_npz:
    print("No GT grids found after filtering. Check --systems filter.")
    sys.exit(1)

print(f"Model A : {args.dir_a}")
print(f"Model B : {args.dir_b}")
print(f"n_steps : {args.n_steps}")
print(f"Grids   : {len(all_npz)}")
print()


def _recall(gt_bool, hnn_bool):
    """Recall = TP / (TP + FN). Returns nan if no positives in GT."""
    total = int(gt_bool.sum())
    if total == 0:
        return float("nan")
    return float((gt_bool & hnn_bool).sum()) / total


rows = []

for npz_fname in all_npz:
    stem      = npz_fname[:-4]                    # strip .npz
    meta_path = os.path.join(GT_DIR, stem + ".meta.json")
    npz_path  = os.path.join(GT_DIR, npz_fname)

    with open(meta_path) as f:
        meta = json.load(f)

    sp             = meta["system_params"]
    t_sim          = float(meta["t_sim"])
    moon_retro     = bool(meta.get("moon_retrograde", False))
    mm_res         = int(meta.get("mm_resolution", 30))
    am_res         = int(meta.get("am_resolution", 30))

    system_params = {
        "ms_solar": sp["ms_solar"],
        "rs_solar": sp.get("rs_solar", 1.0),
        "Ts":       sp.get("Ts", 5772.0),
        "mp_earth": sp["mp_earth"],
        "ap_AU":    sp["ap_AU"],
        "ep":       sp.get("ep", 0.0),
    }
    em_val = float(meta.get("em", 0.0))

    d = np.load(npz_path)
    gt_stable    = d["map_stable"].astype(bool)     # (mm_res, am_res)
    gt_habitable = d["map_habitable"].astype(bool)
    gt_both      = gt_stable & gt_habitable

    N = mm_res * am_res
    gt_frac_stable = float(gt_stable.mean())
    gt_frac_both   = float(gt_both.mean())

    direction = "retro" if moon_retro else "pro"
    label     = stem.replace("_", " ").replace(".meta", "")

    print(f"{'-'*70}")
    print(f"  {label}   [t_sim={t_sim}yr  {mm_res}x{am_res}  "
          f"gt_frac_stable={gt_frac_stable:.3f}  gt_frac_both={gt_frac_both:.3f}]")

    # Compute MLP eligible mask once per system (both models share same grid)
    t_mask = time.perf_counter()
    eligible_mask = _compute_eligible_mask(
        system_params, t_sim, moon_retro, mm_res, am_res, em=em_val
    )
    print(f"  [eligible_mask] computed in {time.perf_counter()-t_mask:.1f}s")

    row = {
        "label": label, "direction": direction,
        "gt_frac_stable": gt_frac_stable, "gt_frac_both": gt_frac_both,
    }

    for tag, model_dir in [("A", DIR_A), ("B", DIR_B)]:
        print(f"  [{tag}] Starting {os.path.basename(model_dir)} ...", flush=True)
        t0 = time.perf_counter()
        r  = batch_hnn_hill_trajectories(
            system_params   = system_params,
            t_sim           = t_sim,
            moon_retrograde = moon_retro,
            mm_resolution   = mm_res,
            am_resolution   = am_res,
            n_steps         = args.n_steps,
            model_dir       = model_dir,
            eligible_mask   = eligible_mask,
        )
        elapsed = time.perf_counter() - t0
        print(f"  [{tag}] Inference done: {elapsed:.0f}s", flush=True)

        if not r.get("ok", False):
            print(f"  [{tag}] ERROR: {r.get('error')} {r.get('message', '')}")
            row[f"stable_{tag}"]      = float("nan")
            row[f"hab_{tag}"]         = float("nan")
            row[f"frac_stable_{tag}"] = float("nan")
            row[f"frac_both_{tag}"]   = float("nan")
            continue

        hnn_stable    = np.array(r["map_stable"],    dtype=bool)
        hnn_habitable = np.array(r["map_habitable"], dtype=bool)
        hnn_both      = hnn_stable & hnn_habitable

        stab_rec  = _recall(gt_stable, hnn_stable)
        hab_rec   = _recall(gt_both,   hnn_both)
        frac_stab = float(hnn_stable.mean())
        frac_both = float(hnn_both.mean())

        row[f"stable_{tag}"]      = stab_rec
        row[f"hab_{tag}"]         = hab_rec
        row[f"frac_stable_{tag}"] = frac_stab
        row[f"frac_both_{tag}"]   = frac_both

        print(f"  [{tag}] stable_recall={stab_rec:.3f}  hab_recall={hab_rec:.3f}  "
              f"frac_stable={frac_stab:.3f}  frac_both={frac_both:.3f}  "
              f"({elapsed:.0f}s)")

    # Delta (B - A)
    ds = row.get("stable_B", float("nan")) - row.get("stable_A", float("nan"))
    dh = row.get("hab_B",    float("nan")) - row.get("hab_A",    float("nan"))
    row["delta_stable"] = ds
    row["delta_hab"]    = dh
    print(f"  [delta B-A]  stable={ds:+.3f}  hab={dh:+.3f}  "
          f"({'IMPROVED' if dh > 0.01 else 'REGRESSED' if dh < -0.01 else 'TIE'})")

    rows.append(row)


# ── Summary tables ─────────────────────────────────────────────────────────────

print()
W = 100
print(f"{'='*W}")
print(f"RECALL SUMMARY  (Model A={args.dir_a}   Model B={args.dir_b})")
print(f"  Metric: recall = |GT_pos & HNN_pos| / |GT_pos|  (denominator: GT-positive cells)")
print(f"{'='*W}")
hdr = f"{'System':<35}  {'Stb_A':>6} {'Stb_B':>6} {'dStb':>7}  {'Hab_A':>6} {'Hab_B':>6} {'dHab':>7}"
print(hdr)
print("-" * W)

n_imp_hab = n_reg_hab = n_tie_hab = 0
for r in rows:
    lbl    = r["label"][:34]
    sa, sb = r.get("stable_A", float("nan")), r.get("stable_B", float("nan"))
    ha, hb = r.get("hab_A",    float("nan")), r.get("hab_B",    float("nan"))
    ds, dh = r.get("delta_stable", float("nan")), r.get("delta_hab", float("nan"))
    flag = " **" if dh > 0.01 else (" !!" if dh < -0.01 else "")
    print(f"{lbl:<35}  {sa:>6.3f} {sb:>6.3f} {ds:>+7.3f}  {ha:>6.3f} {hb:>6.3f} {dh:>+7.3f}{flag}")
    if not np.isnan(dh):
        if dh > 0.01:    n_imp_hab += 1
        elif dh < -0.01: n_reg_hab += 1
        else:            n_tie_hab += 1

print("-" * W)
print(f"Habitable recall (B vs A):  improved={n_imp_hab}  tie={n_tie_hab}  regressed={n_reg_hab}")
print("** = improved >1pp   !! = regressed >1pp")
print(f"{'='*W}")

print()
print(f"{'='*W}")
print(f"FRACTION SUMMARY  (Model A={args.dir_a}   Model B={args.dir_b})")
print(f"  Metric: fraction = HNN_pos.sum() / N  (denominator: all N cells, same as GT fraction)")
print(f"  Fair GT comparison: GT_frac vs A_frac vs B_frac  (all on same denominator)")
print(f"{'='*W}")
hdr2 = (f"{'System':<35}  {'GT_stb':>7} {'A_stb':>7} {'B_stb':>7}   "
        f"{'GT_hab':>7} {'A_hab':>7} {'B_hab':>7}")
print(hdr2)
print("-" * W)

for r in rows:
    lbl  = r["label"][:34]
    gts  = r.get("gt_frac_stable",  float("nan"))
    gth  = r.get("gt_frac_both",    float("nan"))
    fas  = r.get("frac_stable_A",   float("nan"))
    fbs  = r.get("frac_stable_B",   float("nan"))
    fah  = r.get("frac_both_A",     float("nan"))
    fbh  = r.get("frac_both_B",     float("nan"))
    print(f"{lbl:<35}  {gts:>7.3f} {fas:>7.3f} {fbs:>7.3f}   "
          f"{gth:>7.3f} {fah:>7.3f} {fbh:>7.3f}")

print("-" * W)
print(f"{'='*W}")
