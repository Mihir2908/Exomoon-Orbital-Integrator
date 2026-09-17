"""
Diagnostic: for clean false negatives (gt_stable=True & gt_habitable=True but
flag head says uninhabitable), measure what the distance head's moon_star_dist_norm
prediction is doing at the same steps where flag_logit < 0.

Reports for each system:
  - At steps where flag says uninhabitable (logit < 0), what fraction have
    dist_pred[moon_star_dist_norm] > 0 (distance head says habitable)?
  - Distribution of dist_pred values at those steps
  - Overall: if we used dist_pred < 0 | > arcsinh(1) as the criterion instead
    of flag logit < 0, how many of the clean FN cells would become TP?
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from exomoon.params import SystemParams
from exomoon.initial_conditions import initial_state
from exomoon.habitable_zone import hz_bounds_au, moon_effective_temp_K
from exomoon.constants import rsun, au, merth, msun
from exomoon.ml.model import MoonRNN
from exomoon.ml.dataset import load_normalizer, SYS_COLS, STATE_COLS

MODEL_DIR   = sys.argv[1] if len(sys.argv) > 1 else "models_temphead"
SYSTEM_NAME = sys.argv[2] if len(sys.argv) > 2 else "Kepler_452_b_v2"
WARMUP      = int(sys.argv[3]) if len(sys.argv) > 3 else 10
N_STEPS     = 1000

ARCSINH_1 = float(np.arcsinh(1.0))   # ≈ 0.881 — outer HZ boundary in moon_star_dist_norm


def run_direction(direction: str):
    base  = f"ground_truth_grids/{SYSTEM_NAME}_{direction}"
    npz   = np.load(f"{base}.npz")
    with open(f"{base}.meta.json") as f:
        meta = json.load(f)
    sys_params     = meta["system_params"]
    moon_retrograde = meta["moon_retrograde"]
    t_sim          = meta["t_sim"]
    mm_grid        = npz["mm_grid"]
    am_grid        = npz["am_grid"]
    gt_habitable   = npz["map_habitable"].flatten()
    gt_stable      = npz["map_stable"].flatten()

    model = MoonRNN.load(MODEL_DIR, rnn_type="gru")
    model.eval()
    sys_scaler, state_scaler = load_normalizer(os.path.join(MODEL_DIR, "normalizer.pkl"))

    a_in, a_out = hz_bounds_au(sys_params["Ts"], sys_params["rs_solar"] * rsun)
    hz_width = max(a_out - a_in, 1e-6)

    n_mm, n_am = len(mm_grid), len(am_grid)
    n = n_mm * n_am
    all_state = np.zeros((n, len(STATE_COLS)), dtype=np.float32)
    all_sys   = np.zeros((n, len(SYS_COLS)),   dtype=np.float32)
    valid_mask = np.ones(n, dtype=bool)

    idx = 0
    for mm in mm_grid:
        for am in am_grid:
            p = SystemParams(
                Ts=sys_params["Ts"], rs_solar=sys_params["rs_solar"],
                ms_solar=sys_params["ms_solar"], mp_earth=sys_params["mp_earth"],
                ap_AU=sys_params["ap_AU"], ep=sys_params.get("ep", 0.0),
                mm_earth=float(mm), am_hill=float(am), em=0.0,
                moon_retrograde=moon_retrograde,
            )
            try:
                st = initial_state(p)
            except Exception:
                valid_mask[idx] = False; idx += 1; continue
            rhill = st["rhill_AU"]
            pos_mp, pos_ms, pos_mm = st["pos_mp"], st["pos_ms"], st["pos_mm"]
            vel_mp, vel_mm = st["vel_mp"], st["vel_mm"]
            mpd = float(np.linalg.norm(pos_mm - pos_mp))
            psd = float(np.linalg.norm(pos_mp - pos_ms))
            msd = float(np.linalg.norm(pos_mm - pos_ms))
            msp = float(np.linalg.norm(vel_mm))
            psp = float(np.linalg.norm(vel_mp))
            Tm  = moon_effective_temp_K(sys_params["Ts"], sys_params["rs_solar"] * rsun, msd * au)
            Th  = moon_effective_temp_K(sys_params["Ts"], sys_params["rs_solar"] * rsun, a_in * au)
            Tc  = moon_effective_temp_K(sys_params["Ts"], sys_params["rs_solar"] * rsun, a_out * au)
            tw  = max(Th - Tc, 1e-6)
            all_state[idx] = [
                mpd / rhill,
                np.arcsinh((msd - a_in) / hz_width),
                np.arcsinh((Tm - Tc) / tw),
                psd,        # raw planet_star_dist (AU)
                msp, psp, 0.0,
            ]
            all_sys[idx] = [
                sys_params["ms_solar"], sys_params["rs_solar"], sys_params["Ts"],
                sys_params["mp_earth"], sys_params["ap_AU"], sys_params.get("ep", 0.0),
                mm, am, 0.0, float(moon_retrograde),
                t_sim, rhill, a_in, a_out,
            ]
            idx += 1

    sn  = state_scaler.transform(all_state).astype(np.float32)
    syn = sys_scaler.transform(all_sys).astype(np.float32)

    state_t = torch.from_numpy(sn).unsqueeze(1)
    sys_t   = torch.from_numpy(syn).unsqueeze(1)
    h       = torch.zeros(model.layers, n, model.hidden)

    # Per-candidate accumulators
    # flag-based: original approach
    ever_uninhabited_flag = np.zeros(n, dtype=bool)
    # dist-based: pure dist replacement
    ever_uninhabited_dist = np.zeros(n, dtype=bool)
    # AND: both dist AND flag must agree at the same step
    ever_uninhabited_and  = np.zeros(n, dtype=bool)

    # For clean FNs: at steps where flag fires (logit < 0), what does dist say?
    # Store step-level agreement for post-processing
    flag_neg_steps      = np.zeros(n, dtype=np.int32)  # steps where flag fires
    flag_neg_dist_agree = np.zeros(n, dtype=np.int32)  # of those, steps where dist also fires (dist < 0 | > ARCSINH_1)
    flag_neg_dist_pos   = np.zeros(n, dtype=np.int32)  # of those, steps where dist says habitable (0 <= dist <= ARCSINH_1)

    t_fracs = np.linspace(0, 1, N_STEPS)

    with torch.no_grad():
        for step in range(N_STEPS):
            rnn_in  = torch.cat([state_t, sys_t], dim=-1)
            rnn_out, h = model.rnn(rnn_in, h)
            raw     = model.head_dist(rnn_out)
            dist_pred = torch.cat([
                torch.relu(raw[..., 0:1]),
                raw[..., 1:3],
                torch.relu(raw[..., 3:6]),
            ], dim=-1)
            flags   = model.head_flags(rnn_out)
            dn      = dist_pred.squeeze(1).numpy()   # (N, N_DIST)
            fn      = flags.squeeze(1).numpy()       # (N, N_FLAGS)

            if step >= WARMUP:
                flag_uninh = fn[:, 1] < 0
                dist_uninh = (dn[:, 1] < 0) | (dn[:, 1] > ARCSINH_1)

                ever_uninhabited_flag |= flag_uninh
                ever_uninhabited_dist |= dist_uninh
                ever_uninhabited_and  |= flag_uninh & dist_uninh   # both at same step

                # Track per-step disagreement at steps where flag fires
                flag_neg_steps      += flag_uninh.astype(np.int32)
                flag_neg_dist_agree += (flag_uninh & dist_uninh).astype(np.int32)
                flag_neg_dist_pos   += (flag_uninh & ~dist_uninh).astype(np.int32)

            nxt = np.concatenate([dn, np.full((n, 1), t_fracs[step])], axis=1).astype(np.float32)
            state_t = torch.from_numpy(state_scaler.transform(nxt)).unsqueeze(1)

    # Mark invalid candidates
    ever_uninhabited_flag[~valid_mask] = True
    ever_uninhabited_dist[~valid_mask] = True
    ever_uninhabited_and[~valid_mask]  = True

    ml_habitable_flag = ~ever_uninhabited_flag
    ml_habitable_dist = ~ever_uninhabited_dist
    ml_habitable_and  = ~ever_uninhabited_and

    clean_mask = gt_stable & gt_habitable   # unambiguous positive cases
    n_clean    = clean_mask.sum()

    # Current recall (flag-based)
    clean_tp_flag = (clean_mask & ml_habitable_flag).sum()
    clean_tp_dist = (clean_mask & ml_habitable_dist).sum()
    clean_tp_and  = (clean_mask & ml_habitable_and).sum()

    # Among clean FNs under the flag criterion — what does the dist head say?
    clean_fn_flag = clean_mask & ~ml_habitable_flag   # false negatives under flag

    print(f"\n=== {SYSTEM_NAME} {direction} ===")
    print(f"  Clean subset (gt_stable=True & gt_habitable=True): n={n_clean}")
    print(f"  Recall  — flag only:     {clean_tp_flag}/{n_clean} = {clean_tp_flag/n_clean:.3f}")
    print(f"  Recall  — dist only:     {clean_tp_dist}/{n_clean} = {clean_tp_dist/n_clean:.3f}  "
          f"({'better' if clean_tp_dist > clean_tp_flag else 'worse' if clean_tp_dist < clean_tp_flag else 'equal'})")
    print(f"  Recall  — AND (both):    {clean_tp_and}/{n_clean} = {clean_tp_and/n_clean:.3f}  "
          f"({'better' if clean_tp_and > clean_tp_flag else 'worse' if clean_tp_and < clean_tp_flag else 'equal'})")

    n_fn = clean_fn_flag.sum()
    if n_fn == 0:
        print("  (no clean false negatives under flag criterion)")
        return

    fn_flag_neg   = flag_neg_steps[clean_fn_flag]       # steps flag fired for each FN
    fn_dist_agree = flag_neg_dist_agree[clean_fn_flag]  # of those, dist also fired
    fn_dist_pos   = flag_neg_dist_pos[clean_fn_flag]    # of those, dist said habitable

    n_post_warmup = N_STEPS - WARMUP
    # For clean FNs: fraction of flag-negative steps where dist DISAGREES (dist says habitable)
    frac_disagree = np.where(fn_flag_neg > 0, fn_dist_pos / fn_flag_neg, np.nan)
    frac_disagree_valid = frac_disagree[~np.isnan(frac_disagree)]

    print(f"\n  Among {n_fn} clean false negatives (flag says uninhabitable, gt says habitable):")
    print(f"  At steps where flag fires (logit < 0), what does the distance head say?")
    if len(frac_disagree_valid):
        pct = np.percentile(frac_disagree_valid, [5, 25, 50, 75, 95])
        print(f"    Fraction of flag-negative steps where dist says HABITABLE (disagrees with flag):")
        print(f"    p5={pct[0]:.3f}  p25={pct[1]:.3f}  median={pct[2]:.3f}  p75={pct[3]:.3f}  p95={pct[4]:.3f}")
        for thresh in [0.25, 0.50, 0.75, 0.90, 1.00]:
            cnt = (frac_disagree_valid >= thresh).sum()
            print(f"    Cells with dist-habitable fraction >= {thresh:.2f}: {cnt}/{n_fn} ({100*cnt/n_fn:.1f}%)")

    # How many clean FNs would flip to TP if we used dist-based criterion?
    # A cell flips from FN (flag) to TP (dist) if dist head never fires ever_uninhabited
    flip_to_tp = clean_fn_flag & ml_habitable_dist
    stay_fn    = clean_fn_flag & ~ml_habitable_dist
    print(f"\n  Under dist-based criterion, of the {n_fn} clean FNs:")
    print(f"    Would become TP (dist says habitable throughout): {flip_to_tp.sum()} ({100*flip_to_tp.sum()/n_fn:.1f}%)")
    print(f"    Would remain FN (dist also fires uninhabitable):  {stay_fn.sum()} ({100*stay_fn.sum()/n_fn:.1f}%)")

    # New FPs introduced by dist criterion (cells where flag says habitable but dist doesn't)
    new_fp = ~clean_mask & ml_habitable_dist & ~ml_habitable_flag
    lost_tn = ~clean_mask & ~ml_habitable_flag & ml_habitable_dist
    print(f"\n  Precision impact of switching to dist criterion:")
    print(f"    Cells with gt_habitable=False that flag correctly rejects but dist accepts (new FP): {new_fp.sum()}")
    total_gt_neg = (~gt_habitable & valid_mask).sum()
    flag_rejects_gt_neg  = (~gt_habitable & ~ml_habitable_flag & valid_mask).sum()
    dist_rejects_gt_neg  = (~gt_habitable & ~ml_habitable_dist & valid_mask).sum()
    print(f"    Of {total_gt_neg} gt_habitable=False cells:")
    print(f"      Flag correctly rejects: {flag_rejects_gt_neg}  ({100*flag_rejects_gt_neg/max(total_gt_neg,1):.1f}%)")
    print(f"      Dist correctly rejects: {dist_rejects_gt_neg}  ({100*dist_rejects_gt_neg/max(total_gt_neg,1):.1f}%)")


if __name__ == "__main__":
    run_direction("prograde")
    run_direction("retrograde")
