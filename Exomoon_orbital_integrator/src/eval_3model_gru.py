"""
eval_3model_gru.py — TESTING ONLY, no production changes.

Evaluates the 4-model GRU split (models_mdwarf/, models_ktype/, models_gtype/, models_gfa/)
against all 10 ground-truth gates. Each gate is routed to the correct sub-model
based on ms_solar from its meta.json.

Routing:
  ms_solar < 0.45  → models_mdwarf/   (TRAPPIST-1e, OGLE-390Lb)
  ms_solar < 0.80  → models_ktype/    (K-1229b, Kepler-442b)
  ms_solar < 1.20  → models_gtype/    (K-452b-v2)
  else             → models_gfa/      (F/A-type, deferred — falls back to models_gfa/ as OOD)

Habitable metric (non-OGLE gates):
  recall = (gt_stable & gt_habitable & ml_stable & ml_habitable) / (gt_stable & gt_habitable)

OGLE gates (gt_habitable=0 everywhere): reports false positive count only.

No MLP files, no production model directories, and no models_temphead are read or written.
"""

import os, sys, json
import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from exomoon.ml.inference import predict_stability_map

MODEL_MDWARF = os.path.join(SRC, "models_mdwarf")
MODEL_KTYPE  = os.path.join(SRC, "models_ktype")
MODEL_GTYPE  = os.path.join(SRC, "models_gtype")
MODEL_GFA    = os.path.join(SRC, "models_gfa")   # F/A — deferred, OOD fallback

GATES = {
    "K-452b-v2 pro":     os.path.join(SRC, "ground_truth_grids", "Kepler_452_b_v2_prograde"),
    "K-452b-v2 retro":   os.path.join(SRC, "ground_truth_grids", "Kepler_452_b_v2_retrograde"),
    "K-1229b pro":       os.path.join(SRC, "ground_truth_grids", "Kepler_1229_b_prograde"),
    "K-1229b retro":     os.path.join(SRC, "ground_truth_grids", "Kepler_1229_b_retrograde"),
    "TRAPPIST-1e pro":   os.path.join(SRC, "ground_truth_grids", "TRAPPIST_1_e_prograde"),
    "TRAPPIST-1e retro": os.path.join(SRC, "ground_truth_grids", "TRAPPIST_1_e_retrograde"),
    "Kepler-442b pro":   os.path.join(SRC, "ground_truth_grids", "Kepler_442_b_prograde"),
    "Kepler-442b retro": os.path.join(SRC, "ground_truth_grids", "Kepler_442_b_retrograde"),
    "OGLE-390Lb pro":    os.path.join(SRC, "ground_truth_grids", "OGLE_2005_BLG_390Lb_prograde"),
    "OGLE-390Lb retro":  os.path.join(SRC, "ground_truth_grids", "OGLE_2005_BLG_390Lb_retrograde"),
}
GATES = {k: v for k, v in GATES.items() if os.path.exists(v + ".npz")}

OGLE_GATES = {"OGLE-390Lb pro", "OGLE-390Lb retro"}

# temphead baseline for comparison (from CLAUDE.md, AND criterion applied)
TEMPHEAD_STABLE = {
    "K-452b-v2 pro":   0.964, "K-452b-v2 retro":   0.944,
    "K-1229b pro":     0.887, "K-1229b retro":     0.950,
}
TEMPHEAD_HAB = {
    "K-452b-v2 pro":   0.386, "K-452b-v2 retro":   0.581,
    "K-1229b pro":     0.170, "K-1229b retro":     0.596,
}


def route_model(ms_solar: float) -> tuple[str, str]:
    if ms_solar < 0.45:
        return MODEL_MDWARF, "M-dwarf"
    if ms_solar < 0.80:
        return MODEL_KTYPE,  "K-type"
    if ms_solar < 1.20:
        return MODEL_GTYPE,  "G-type"
    return MODEL_GFA, "F/A(OOD)"   # F/A model deferred — routes to gfa as fallback


def pr_f1(gt_mask, ml_mask):
    tp    = int((gt_mask & ml_mask).sum())
    n_gt  = int(gt_mask.sum())
    n_ml  = int(ml_mask.sum())
    rec   = tp / n_gt if n_gt > 0 else None
    prec  = tp / n_ml if n_ml > 0 else None
    f1    = (2 * rec * prec / (rec + prec)) if (rec and prec and (rec + prec) > 0) else None
    return rec, prec, f1, n_gt


def fmt(v, width=6):
    return f"{v:.3f}" if v is not None else f"{'--':>{width}}"


def check_models():
    missing = []
    for label, path in [("M-dwarf", MODEL_MDWARF), ("K-type", MODEL_KTYPE), ("G-type", MODEL_GTYPE)]:
        pt = os.path.join(path, "gru_model.pt")
        if not os.path.exists(pt):
            missing.append(f"  {label}: {pt}")
    if missing:
        print("WARNING — the following model weights have not been trained yet:")
        for m in missing:
            print(m)
        print("Run the three training commands in CLAUDE.md before evaluating.\n")
        return False
    return True


def main():
    all_present = check_models()
    if not all_present:
        print("Evaluation aborted — train all 3 models first.")
        return

    W = 7
    sep = "=" * 90
    div = "-" * 90

    for metric_label, use_stable in [("STABLE", True), ("HABITABLE (4-gate metric)", False)]:
        print()
        print(sep)
        print(f"  {metric_label}  —  3-model GRU split vs. temphead baseline")
        if not use_stable:
            print("  Denominator: gt_stable & gt_habitable  |  Numerator: ml_stable & ml_habitable")
        print(sep)
        print(f"  {'Gate':22} {'Model':8} {'recall':>{W}} {'prec':>{W}} {'F1':>{W}}  {'temphead':>8}  {'delta':>7}")
        print(div)

        rec_vals = []

        for gate_name, gate_prefix in GATES.items():
            npz  = np.load(f"{gate_prefix}.npz", allow_pickle=True)
            with open(f"{gate_prefix}.meta.json") as f:
                meta = json.load(f)

            gt_s = npz["map_stable"].astype(bool)
            gt_h = npz["map_habitable"].astype(bool)

            sp       = meta["system_params"]
            ms_solar = float(sp["ms_solar"])
            model_dir, model_label = route_model(ms_solar)

            res = predict_stability_map(
                system_params   = sp,
                t_sim           = meta["t_sim"],
                moon_retrograde = meta["moon_retrograde"],
                em              = meta["em"],
                mm_resolution   = meta["mm_resolution"],
                am_resolution   = meta["am_resolution"],
                model_dir       = model_dir,
                n_steps         = 1000,
            )

            if not res.get("ok"):
                print(f"  {gate_name:22} {model_label:8}  ERROR: {res.get('error', 'unknown')}")
                continue

            ml_s = np.array(res["map_stable"])
            ml_h = np.array(res["map_habitable"])

            is_ogle = gate_name in OGLE_GATES

            if use_stable:
                ref = TEMPHEAD_STABLE.get(gate_name)
                if is_ogle:
                    print(f"  {gate_name:22} {model_label:8}  (no stable reference for OGLE)")
                    continue
                rec, prec, f1, n_gt = pr_f1(gt_s, ml_s)
                delta = f"{rec - ref:+.3f}" if (rec is not None and ref is not None) else "  --"
                print(f"  {gate_name:22} {model_label:8} {fmt(rec):>{W}} {fmt(prec):>{W}} {fmt(f1):>{W}}  {fmt(ref):>8}  {delta:>7}")
                if rec is not None:
                    rec_vals.append(rec)
            else:
                ref = TEMPHEAD_HAB.get(gate_name)
                if is_ogle:
                    n_fp    = int((ml_s & ml_h).sum())
                    n_grid  = gt_s.size
                    print(f"  {gate_name:22} {model_label:8}  FP={n_fp}/{n_grid} (gt_habitable=0 everywhere)")
                    continue
                gt_mask = gt_s & gt_h
                ml_mask = ml_s & ml_h
                rec, prec, f1, n_gt = pr_f1(gt_mask, ml_mask)
                delta = f"{rec - ref:+.3f}" if (rec is not None and ref is not None) else "  --"
                print(f"  {gate_name:22} {model_label:8} {fmt(rec):>{W}} {fmt(prec):>{W}} {fmt(f1):>{W}}  {fmt(ref):>8}  {delta:>7}")
                if rec is not None:
                    rec_vals.append(rec)

        print(div)
        if rec_vals:
            avg = sum(rec_vals) / len(rec_vals)
            if use_stable:
                ref_avg = sum(TEMPHEAD_STABLE.values()) / len(TEMPHEAD_STABLE)
            else:
                ref_avg = sum(TEMPHEAD_HAB.values()) / len(TEMPHEAD_HAB)
            delta = f"{avg - ref_avg:+.3f}"
            print(f"  {'Average (non-OGLE)':22} {'':8} {fmt(avg):>{W}} {'':>{W}} {'':>{W}}  {fmt(ref_avg):>8}  {delta:>7}")

    print()
    print("Done. No production models modified.")


if __name__ == "__main__":
    main()
