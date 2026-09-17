"""
eval_comprehensive_table.py — TESTING ONLY, no production changes.

Prints a single comprehensive table: STABLE and HABITABLE recall/precision/F1
for all 4 model variants across all available ground-truth gates.

Models evaluated:
  1. GRU seed123 (models_stageC_seed123)
  2. Binary cls MLP  (eval_aux_mlp_output/binary/aux_mlp_binary.pt)
  3. Reg MLP t=0.50  (eval_aux_mlp_output/regression/aux_mlp.pt)
  4. Reg MLP t=0.90  (eval_aux_mlp_output/regression/aux_mlp.pt, thresh=0.90)

No production model directories are read or written.
"""

import os, sys, json, pickle
import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import torch
import torch.nn as nn
from exomoon.ml.dataset import SYS_COLS, LOG_SYS_COLS
from exomoon.ml.inference import predict_stability_map
from exomoon.params import SystemParams
from exomoon.initial_conditions import initial_state
from exomoon.habitable_zone import hz_bounds_au
from exomoon.constants import rsun

GRU_DIR  = os.path.join(SRC, "models_stageC_seed123")
OUT_REG  = os.path.join(SRC, "eval_aux_mlp_output", "regression")
OUT_BIN  = os.path.join(SRC, "eval_aux_mlp_output", "binary")

GATES = {
    "K-452b-v2 pro":    os.path.join(SRC, "ground_truth_grids", "Kepler_452_b_v2_prograde"),
    "K-452b-v2 retro":  os.path.join(SRC, "ground_truth_grids", "Kepler_452_b_v2_retrograde"),
    "K-1229b pro":      os.path.join(SRC, "ground_truth_grids", "Kepler_1229_b_prograde"),
    "K-1229b retro":    os.path.join(SRC, "ground_truth_grids", "Kepler_1229_b_retrograde"),
    "TRAPPIST-1e pro":  os.path.join(SRC, "ground_truth_grids", "TRAPPIST_1_e_prograde"),
    "TRAPPIST-1e retro":os.path.join(SRC, "ground_truth_grids", "TRAPPIST_1_e_retrograde"),
    "Kepler-442b pro":  os.path.join(SRC, "ground_truth_grids", "Kepler_442_b_prograde"),
    "Kepler-442b retro":os.path.join(SRC, "ground_truth_grids", "Kepler_442_b_retrograde"),
    # OGLE: planet far outside HZ — gt_habitable=0 for all candidates.
    # Habitable column reports false positive count rather than recall.
    "OGLE-390Lb pro":   os.path.join(SRC, "ground_truth_grids", "OGLE_2005_BLG_390Lb_prograde"),
    "OGLE-390Lb retro": os.path.join(SRC, "ground_truth_grids", "OGLE_2005_BLG_390Lb_retrograde"),
}
GATES = {k: v for k, v in GATES.items() if os.path.exists(v + ".npz")}

# Gates where gt_habitable=0 by construction (planet far outside HZ).
# For these, habitable recall is undefined; report false positive count instead.
OGLE_GATES = {"OGLE-390Lb pro", "OGLE-390Lb retro"}

TEMPHEAD_REF = {
    "K-452b-v2 pro":   (0.964, 0.396),
    "K-452b-v2 retro": (0.939, 0.591),
    "K-1229b pro":     (0.836, 0.218),
    "K-1229b retro":   (0.946, 0.602),
}


# ── model classes (must match training definitions) ─────────────────────────────
class AuxMLP(nn.Module):
    def __init__(self, input_dim=14, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, 2),         nn.Sigmoid(),
        )
    def forward(self, x): return self.net(x)


class AuxMLPBinary(nn.Module):
    def __init__(self, input_dim=14, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, 2),
        )
    def forward(self, x): return self.net(x)


# ── helpers ─────────────────────────────────────────────────────────────────────
def apply_log(arr):
    arr = arr.copy()
    log_idx = [SYS_COLS.index(c) for c in LOG_SYS_COLS if c in SYS_COLS]
    arr[:, log_idx] = np.log(np.clip(arr[:, log_idx], 1e-10, None))
    return arr


def pr_f1(gt_mask, ml_mask):
    tp  = int((gt_mask & ml_mask).sum())
    n_gt = int(gt_mask.sum())
    n_ml = int(ml_mask.sum())
    rec  = tp / n_gt if n_gt > 0 else None
    pre  = tp / n_ml if n_ml > 0 else None
    f    = (2*rec*pre/(rec+pre)) if (rec and pre and (rec+pre)>0) else None
    return rec, pre, f, n_gt


def rhill_hz(meta):
    sp = meta["system_params"]
    p  = SystemParams(
        Ts=float(sp["Ts"]), rs_solar=float(sp["rs_solar"]),
        ms_solar=float(sp["ms_solar"]), mp_earth=float(sp["mp_earth"]),
        ap_AU=float(sp["ap_AU"]), ep=float(sp["ep"]),
        mm_earth=0.5, am_hill=0.4,
        em=float(meta["em"]), moon_retrograde=bool(meta["moon_retrograde"]),
    )
    st = initial_state(p)
    rs_m = float(sp["rs_solar"]) * rsun
    ain, aout = hz_bounds_au(float(sp["Ts"]), rs_m)
    return float(st["rhill_AU"]), float(ain), float(aout)


def mlp_grid(model, scaler, meta, mm_grid, am_grid, rhill, ain, aout, binary, thresh=0.5):
    sp    = meta["system_params"]
    retro = float(int(meta["moon_retrograde"]))
    em    = float(meta["em"])
    tsim  = float(meta["t_sim"])
    vecs  = []
    for mm in mm_grid:
        for am in am_grid:
            vecs.append([float(sp["ms_solar"]), float(sp["rs_solar"]), float(sp["Ts"]),
                         float(sp["mp_earth"]), float(sp["ap_AU"]), float(sp["ep"]),
                         float(mm), float(am), em, retro, tsim, rhill, ain, aout])
    X = scaler.transform(apply_log(np.array(vecs, dtype=np.float32))).astype(np.float32)
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(X))
        probs = torch.sigmoid(out).numpy() if binary else out.numpy()
    mm_r, am_r = len(mm_grid), len(am_grid)
    return (probs[:,0] >= thresh).reshape(mm_r, am_r), (probs[:,1] >= thresh).reshape(mm_r, am_r)


# ── load models ──────────────────────────────────────────────────────────────────
def load_reg():
    m = AuxMLP(input_dim=len(SYS_COLS), hidden=64)
    m.load_state_dict(torch.load(os.path.join(OUT_REG, "aux_mlp.pt"), weights_only=True))
    with open(os.path.join(OUT_REG, "aux_mlp_scaler.pkl"), "rb") as f:
        sc = pickle.load(f)
    return m, sc


def load_bin():
    m = AuxMLPBinary(input_dim=len(SYS_COLS), hidden=64)
    m.load_state_dict(torch.load(os.path.join(OUT_BIN, "aux_mlp_binary.pt"), weights_only=True))
    with open(os.path.join(OUT_BIN, "aux_mlp_scaler.pkl"), "rb") as f:
        sc = pickle.load(f)
    return m, sc


# ── main ────────────────────────────────────────────────────────────────────────
def main():
    print("Loading models ...")
    reg_model, reg_sc = load_reg()
    bin_model, bin_sc = load_bin()

    MODELS = [
        ("GRU seed123",  None, None, None, None),
        ("Binary cls",   bin_model, bin_sc, True,  0.5),
        ("Reg t=0.50",   reg_model, reg_sc, False, 0.5),
        ("Reg t=0.90",   reg_model, reg_sc, False, 0.9),
    ]

    W = 6
    COL = W*3 + 4

    def row_header():
        h1 = f"  {'':20}"
        h2 = f"  {'Gate':20}"
        for name, *_ in MODELS:
            h1 += f"  {name:^{COL}}"
            h2 += f"  {'rec':>{W}} {'prec':>{W}} {'F1':>{W}}"
        h1 += f"  {'temphead':^8}"
        h2 += f"  {'rec':>8}"
        return h1, h2

    def fmt(v): return f"{v:.3f}" if v is not None else "  -- "

    sep = "=" * (22 + len(MODELS) * (COL+2) + 12)
    div = "-" * (22 + len(MODELS) * (COL+2) + 12)

    for metric_label, use_stable in [("STABLE", True), ("HABITABLE (4-gate)", False)]:
        print()
        print(sep)
        print(f"  {metric_label} RECALL / PRECISION / F1   --   all models, all gates")
        if not use_stable:
            print("  Denominator: gt_stable & gt_habitable  |  Numerator: ml_stable & ml_habitable")
        print(sep)
        h1, h2 = row_header()
        print(h1)
        print(h2)
        print(div)

        gate_avgs = {name: {"rec":[], "pre":[], "f1":[]} for name, *_ in MODELS}

        for gate_name, gate_prefix in GATES.items():
            npz  = np.load(f"{gate_prefix}.npz", allow_pickle=True)
            with open(f"{gate_prefix}.meta.json") as f:
                meta = json.load(f)

            gt_s = npz["map_stable"].astype(bool)
            gt_h = npz["map_habitable"].astype(bool)

            # GRU inference (once per gate, shared across all model rows)
            sp  = meta["system_params"]
            res = predict_stability_map(
                system_params=sp, t_sim=meta["t_sim"],
                moon_retrograde=meta["moon_retrograde"], em=meta["em"],
                mm_resolution=meta["mm_resolution"], am_resolution=meta["am_resolution"],
                model_dir=GRU_DIR, n_steps=1000,
            )
            gru_s = np.array(res["map_stable"])
            gru_h = np.array(res["map_habitable"])
            mm_grid = np.array(res["mm_grid"])
            am_grid = np.array(res["am_grid"])

            rh, ain, aout = rhill_hz(meta)

            is_ogle = gate_name in OGLE_GATES
            row = f"  {gate_name:20}"
            th_s_ref, th_h_ref = TEMPHEAD_REF.get(gate_name, (None, None))

            ml_preds = {}
            for name, model, sc, binary, thresh in MODELS:
                if model is None:
                    ml_s, ml_h = gru_s, gru_h
                else:
                    ml_s, ml_h = mlp_grid(model, sc, meta, mm_grid, am_grid, rh, ain, aout, binary, thresh)
                ml_preds[name] = (ml_s, ml_h)

            for name, model, sc, binary, thresh in MODELS:
                ml_s, ml_h = ml_preds[name]

                if use_stable:
                    gt_mask = gt_s
                    ml_mask = ml_s
                    th_ref  = th_s_ref
                    rec, pre, f, n_gt = pr_f1(gt_mask, ml_mask)
                    row += f"  {fmt(rec):>{W}} {fmt(pre):>{W}} {fmt(f):>{W}}"
                    if rec is not None: gate_avgs[name]["rec"].append(rec)
                    if pre is not None: gate_avgs[name]["pre"].append(pre)
                    if f   is not None: gate_avgs[name]["f1"].append(f)
                else:
                    th_ref = th_h_ref
                    if is_ogle:
                        # gt_habitable=0 everywhere — report false positive count only
                        n_fp = int((ml_s & ml_h).sum())
                        n_grid = gt_s.size
                        row += f"  {'FP=' + str(n_fp):>{W}} {'/' + str(n_grid):>{W}} {'':>{W}}"
                    else:
                        gt_mask = gt_s & gt_h
                        ml_mask = ml_s & ml_h
                        rec, pre, f, n_gt = pr_f1(gt_mask, ml_mask)
                        row += f"  {fmt(rec):>{W}} {fmt(pre):>{W}} {fmt(f):>{W}}"
                        if rec is not None: gate_avgs[name]["rec"].append(rec)
                        if pre is not None: gate_avgs[name]["pre"].append(pre)
                        if f   is not None: gate_avgs[name]["f1"].append(f)

            th_ref = th_s_ref if use_stable else th_h_ref
            row += f"  {fmt(th_ref):>8}"
            print(row)

        print(div)
        avg_row = f"  {'Average':20}"
        for name, *_ in MODELS:
            rs = gate_avgs[name]["rec"]
            ps = gate_avgs[name]["pre"]
            fs = gate_avgs[name]["f1"]
            avg_row += (f"  {(sum(rs)/len(rs) if rs else 0):>{W}.3f}"
                        f" {(sum(ps)/len(ps) if ps else 0):>{W}.3f}"
                        f" {(sum(fs)/len(fs) if fs else 0):>{W}.3f}")
        avg_row += f"  {'ref':>8}"
        print(avg_row)

    print()
    print("Done. No production models modified.")


if __name__ == "__main__":
    main()
