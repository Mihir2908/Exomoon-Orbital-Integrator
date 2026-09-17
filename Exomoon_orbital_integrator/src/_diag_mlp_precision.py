"""_diag_mlp_precision.py — Full precision+recall breakdown: GRU seed123, Aux MLP, OR."""
import os, sys, json, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from exomoon.ml.dataset import SYS_COLS, LOG_SYS_COLS
from exomoon.ml.inference import predict_stability_map
from exomoon.params import SystemParams
from exomoon.initial_conditions import initial_state
from exomoon.habitable_zone import hz_bounds_au
from exomoon.constants import rsun

OUT_DIR = "eval_aux_mlp_output"
GRU_DIR = "models_stageC_seed123"

GATES = {
    "K-452b-v2 pro":   "ground_truth_grids/Kepler_452_b_v2_prograde",
    "K-452b-v2 retro": "ground_truth_grids/Kepler_452_b_v2_retrograde",
    "K-1229b pro":     "ground_truth_grids/Kepler_1229_b_prograde",
    "K-1229b retro":   "ground_truth_grids/Kepler_1229_b_retrograde",
}

TEMPHEAD_REF = {
    "K-452b-v2 pro":   (0.964, 0.396),
    "K-452b-v2 retro": (0.939, 0.591),
    "K-1229b pro":     (0.836, 0.218),
    "K-1229b retro":   (0.946, 0.602),
}


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


def apply_log(arr):
    arr = arr.copy()
    log_idx = [SYS_COLS.index(c) for c in LOG_SYS_COLS if c in SYS_COLS]
    arr[:, log_idx] = np.log(np.clip(arr[:, log_idx], 1e-10, None))
    return arr


def pr(gt_mask, ml_mask):
    tp  = int((gt_mask & ml_mask).sum())
    rec = tp / max(int(gt_mask.sum()), 1)
    pre = tp / max(int(ml_mask.sum()), 1)
    return rec, pre, tp, int(gt_mask.sum()), int(ml_mask.sum())


def main():
    # Load MLP
    mlp = AuxMLP()
    mlp.load_state_dict(torch.load(os.path.join(OUT_DIR, "aux_mlp.pt"), weights_only=True))
    mlp.eval()
    with open(os.path.join(OUT_DIR, "aux_mlp_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    print()
    print("GROUND TRUTH DEFINITIONS")
    print("  gt_stable    = np.all(moon_planet_dist <= rhill_AU) at every post-warmup timestep")
    print("  gt_habitable = np.all(a_inner_au <= moon_star_dist <= a_outer_au) at every post-warmup timestep")
    print()
    print("ML PREDICTION DEFINITIONS")
    print("  GRU stable: stable_logit > 0 at every step (never fires ever_unstable flag)")
    print("  GRU hab   : AND criterion — both dist head outside HZ AND flag head uninh must fire at same step")
    print("  MLP stable: frac_stable_pred >= 0.50  (model trained on mean(stable) per simulation)")
    print("  MLP hab   : frac_hab_pred    >= 0.50  (model trained on mean(habitable) per simulation)")
    print()
    print("RECALL  = TP / GT_positive   (what fraction of true positives are detected)")
    print("PRECISION = TP / ML_positive  (what fraction of ML positives are actually correct)")
    print()

    SEP = "=" * 112
    DIV = "-" * 112
    HDR = f"  {'':30} {'seed123 GRU':>17} {'Aux MLP':>17} {'GRU OR MLP':>17}  {'temphead ref':>12}"

    all_rows = {k: {} for k in GATES}

    for gate_name, prefix in GATES.items():
        npz = np.load(f"{prefix}.npz", allow_pickle=True)
        with open(f"{prefix}.meta.json") as f:
            meta = json.load(f)

        gt_s = npz["map_stable"].astype(bool)
        gt_h = npz["map_habitable"].astype(bool)
        gt_b = gt_s & gt_h
        N    = gt_s.size

        # GRU inference
        sp  = meta["system_params"]
        res = predict_stability_map(
            system_params=sp, t_sim=meta["t_sim"],
            moon_retrograde=meta["moon_retrograde"], em=meta["em"],
            mm_resolution=meta["mm_resolution"], am_resolution=meta["am_resolution"],
            model_dir=GRU_DIR, n_steps=1000,
        )
        gru_s  = np.array(res["map_stable"])
        gru_h  = np.array(res["map_habitable"])
        gru_b  = gru_s & gru_h
        mm_grid = np.array(res["mm_grid"])
        am_grid = np.array(res["am_grid"])

        # MLP inference
        p = SystemParams(
            Ts=float(sp["Ts"]), rs_solar=float(sp["rs_solar"]),
            ms_solar=float(sp["ms_solar"]), mp_earth=float(sp["mp_earth"]),
            ap_AU=float(sp["ap_AU"]), ep=float(sp["ep"]),
            mm_earth=0.5, am_hill=0.4,
            em=float(meta["em"]), moon_retrograde=bool(meta["moon_retrograde"]),
        )
        st    = initial_state(p)
        rhill = float(st["rhill_AU"])
        rs_m  = float(sp["rs_solar"]) * rsun
        a_in, a_out = hz_bounds_au(float(sp["Ts"]), rs_m)

        vecs = []
        for mm in mm_grid:
            for am in am_grid:
                vecs.append([
                    float(sp["ms_solar"]), float(sp["rs_solar"]), float(sp["Ts"]),
                    float(sp["mp_earth"]), float(sp["ap_AU"]),    float(sp["ep"]),
                    float(mm), float(am), float(meta["em"]),
                    float(int(meta["moon_retrograde"])),
                    float(meta["t_sim"]), rhill, a_in, a_out,
                ])
        X    = np.array(vecs, dtype=np.float32)
        X_sc = scaler.transform(apply_log(X)).astype(np.float32)
        with torch.no_grad():
            preds = mlp(torch.from_numpy(X_sc)).numpy()
        mm_r, am_r = len(mm_grid), len(am_grid)
        mlp_s = (preds[:, 0] >= 0.5).reshape(mm_r, am_r)
        mlp_h = (preds[:, 1] >= 0.5).reshape(mm_r, am_r)
        mlp_b = mlp_s & mlp_h

        # OR
        or_s = gru_s | mlp_s
        or_h = gru_h | mlp_h
        or_b = or_s & or_h

        th_s, th_h = TEMPHEAD_REF.get(gate_name, (None, None))

        print(SEP)
        print(f"  {gate_name}   (grid: {N} candidates  |  gt_stable={int(gt_s.sum())}  gt_habitable={int(gt_h.sum())}  gt_both={int(gt_b.sum())})")
        print(DIV)
        print(HDR)
        print(DIV)

        # --- STABLE ---
        gr_sr, gr_sp, gr_stp, gt_sn, gr_sml = pr(gt_s, gru_s)
        mp_sr, mp_sp, mp_stp, _,     mp_sml = pr(gt_s, mlp_s)
        or_sr, or_sp, or_stp, _,     or_sml = pr(gt_s, or_s)

        print(f"  {'STABLE':30}")
        print(f"  {'  predicted stable':30} {gr_sml:>17} {mp_sml:>17} {or_sml:>17}  {'(of ' + str(N) + ')':>12}")
        th_s_str = f"rec={th_s:.3f}" if th_s else ""
        print(f"  {'  recall':30} {gr_sr:>17.3f} {mp_sr:>17.3f} {or_sr:>17.3f}  {th_s_str:>12}")
        print(f"  {'  precision':30} {gr_sp:>17.3f} {mp_sp:>17.3f} {or_sp:>17.3f}")
        print()

        # --- HABITABLE (4-gate) ---
        gr_hr, gr_hp, gr_htp, gt_bn, gr_bml = pr(gt_b, gru_b)
        mp_hr, mp_hp, mp_htp, _,     mp_bml = pr(gt_b, mlp_b)
        or_hr, or_hp, or_htp, _,     or_bml = pr(gt_b, or_b)
        # precision uses ml_both as denominator
        gr_hp = gr_htp / max(int(gru_b.sum()), 1)
        mp_hp = mp_htp / max(int(mlp_b.sum()), 1)
        or_hp = or_htp / max(int(or_b.sum()), 1)

        th_h_str = f"rec={th_h:.3f}" if th_h else ""
        print(f"  {'HABITABLE (4-gate: gt_both denom)':30}")
        print(f"  {'  predicted both':30} {int(gru_b.sum()):>17} {int(mlp_b.sum()):>17} {int(or_b.sum()):>17}  {'(gt_both=' + str(gt_bn) + ')':>12}")
        print(f"  {'  recall':30} {gr_hr:>17.3f} {mp_hr:>17.3f} {or_hr:>17.3f}  {th_h_str:>12}")
        print(f"  {'  precision':30} {gr_hp:>17.3f} {mp_hp:>17.3f} {or_hp:>17.3f}")
        print()

    print(SEP)


if __name__ == "__main__":
    main()
