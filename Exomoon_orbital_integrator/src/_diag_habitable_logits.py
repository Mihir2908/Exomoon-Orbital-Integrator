"""
One-off diagnostic: for the Kepler-1229 b grid (pro/retro), compare ground-truth
habitable labels (from ground_truth_grids/*.npz) against the ML model's predicted
habitable_logit, specifically looking at the *minimum* post-warmup habitable_logit
reached during the rollout for each grid cell (since that's what determines
map_habitable in predict_stability_map -- a single dip below 0 anywhere marks the
whole candidate not-habitable). Buckets false negatives vs true positives/negatives
by how negative that minimum logit got, to distinguish a threshold-calibration
problem (logits barely below 0) from a deeper representation problem (logits very
negative).

Read-only: does not modify inference.py, models/, or any ground-truth files.
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
from exomoon.ml.dataset import load_normalizer, SYS_COLS

import sys as _sys
MODEL_DIR = _sys.argv[1] if len(_sys.argv) > 1 else "models"
SYSTEM_NAME = _sys.argv[2] if len(_sys.argv) > 2 else "Kepler_1229_b"
WARMUP_STEPS = int(_sys.argv[3]) if len(_sys.argv) > 3 else 30
N_STEPS = 1000


def run_direction(direction: str):
    base = f"ground_truth_grids/{SYSTEM_NAME}_{direction}"
    npz = np.load(f"{base}.npz")
    with open(f"{base}.meta.json") as f:
        meta = json.load(f)
    sys_params = meta["system_params"]
    moon_retrograde = meta["moon_retrograde"]
    t_sim = meta["t_sim"]
    mm_grid = npz["mm_grid"]; am_grid = npz["am_grid"]
    gt_habitable = npz["map_habitable"]

    model = MoonRNN.load(MODEL_DIR, rnn_type="gru"); model.eval()
    sys_scaler, state_scaler = load_normalizer(os.path.join(MODEL_DIR, "normalizer.pkl"))

    a_in, a_out = hz_bounds_au(sys_params["Ts"], sys_params["rs_solar"] * rsun)
    hz_width = max(a_out - a_in, 1e-6)

    n_mm, n_am = len(mm_grid), len(am_grid)
    n = n_mm * n_am
    all_state = np.zeros((n, 7), dtype=np.float32)
    all_sys = np.zeros((n, 14), dtype=np.float32)
    valid_mask = np.ones(n, dtype=bool)

    idx = 0
    for mm in mm_grid:
        for am in am_grid:
            p = SystemParams(Ts=sys_params["Ts"], rs_solar=sys_params["rs_solar"],
                              ms_solar=sys_params["ms_solar"], mp_earth=sys_params["mp_earth"],
                              ap_AU=sys_params["ap_AU"], ep=sys_params.get("ep", 0.0),
                              mm_earth=float(mm), am_hill=float(am), em=0.0,
                              moon_retrograde=moon_retrograde)
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
            Tm = moon_effective_temp_K(sys_params["Ts"], sys_params["rs_solar"] * rsun, msd * au)
            Th = moon_effective_temp_K(sys_params["Ts"], sys_params["rs_solar"] * rsun, a_in * au)
            Tc = moon_effective_temp_K(sys_params["Ts"], sys_params["rs_solar"] * rsun, a_out * au)
            tw = max(Th - Tc, 1e-6)
            all_state[idx] = [mpd / rhill, np.arcsinh((msd - a_in) / hz_width),
                               np.arcsinh((Tm - Tc) / tw), psd, msp, psp, 0.0]
            all_sys[idx] = [sys_params["ms_solar"], sys_params["rs_solar"], sys_params["Ts"],
                             sys_params["mp_earth"], sys_params["ap_AU"], sys_params.get("ep", 0.0),
                             mm, am, 0.0, float(moon_retrograde), t_sim, rhill, a_in, a_out]
            idx += 1

    sn = state_scaler.transform(all_state).astype(np.float32)
    syn = sys_scaler.transform(all_sys).astype(np.float32)
    state_t = torch.from_numpy(sn).unsqueeze(1)
    sys_t = torch.from_numpy(syn).unsqueeze(1)
    h = torch.zeros(model.layers, n, model.hidden)

    min_habitable_logit = np.full(n, np.inf, dtype=np.float32)
    first_neg_step = np.full(n, -1, dtype=np.int32)
    ever_neg = np.zeros(n, dtype=bool)
    neg_step_count = np.zeros(n, dtype=np.int32)
    t_fracs = np.linspace(0, 1, N_STEPS)

    with torch.no_grad():
        for step in range(N_STEPS):
            rnn_in = torch.cat([state_t, sys_t], dim=-1)
            rnn_out, h = model.rnn(rnn_in, h)
            raw = model.head_dist(rnn_out)
            dist_pred = torch.cat([torch.relu(raw[..., 0:1]), raw[..., 1:3],
                                    torch.relu(raw[..., 3:6])], dim=-1)
            flags = model.head_flags(rnn_out)
            dn = dist_pred.squeeze(1).numpy(); fn = flags.squeeze(1).numpy()
            if step >= WARMUP_STEPS:
                min_habitable_logit = np.minimum(min_habitable_logit, fn[:, 1])
                newly_neg = (fn[:, 1] < 0) & ~ever_neg
                first_neg_step[newly_neg] = step
                ever_neg |= (fn[:, 1] < 0)
                neg_step_count += (fn[:, 1] < 0).astype(np.int32)
            nxt = np.concatenate([dn, np.full((n, 1), t_fracs[step])], axis=1).astype(np.float32)
            state_t = torch.from_numpy(state_scaler.transform(nxt)).unsqueeze(1)

    ml_habitable = (min_habitable_logit > 0) & valid_mask
    gt_flat = gt_habitable.flatten()

    tp = gt_flat & ml_habitable
    fp = ~gt_flat & ml_habitable
    fn_mask = gt_flat & ~ml_habitable
    tn = ~gt_flat & ~ml_habitable

    print(f"\n=== {SYSTEM_NAME} {direction} ===")
    print(f"TP={tp.sum()} FP={fp.sum()} FN={fn_mask.sum()} TN={tn.sum()}")

    def describe(mask, label):
        vals = min_habitable_logit[mask]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            print(f"  {label}: (no cases)")
            return
        pct = np.percentile(vals, [5, 25, 50, 75, 95])
        print(f"  {label} (n={len(vals)}): min={vals.min():.3f} p5={pct[0]:.3f} "
              f"p25={pct[1]:.3f} median={pct[2]:.3f} p75={pct[3]:.3f} p95={pct[4]:.3f} max={vals.max():.3f}")
        # how many are "barely" negative (within 1.0 of the 0 threshold) vs deeply negative
        barely = np.sum((vals < 0) & (vals >= -1.0))
        deep   = np.sum(vals < -1.0)
        print(f"    of negative cases: barely-negative (>=-1.0)={barely}  deeply-negative (<-1.0)={deep}")

    describe(fn_mask, "False Negatives (gt=habitable, ml=not)")
    describe(tp, "True Positives (gt=habitable, ml=habitable)")
    describe(tn, "True Negatives (gt=not, ml=not)")
    describe(fp, "False Positives (gt=not, ml=habitable)")

    gt_stable = npz["map_stable"].flatten()
    print(f"  FN breakdown: gt_stable=True (clean, no freeze-artifact) = {(fn_mask & gt_stable).sum()}  "
          f"gt_stable=False (escaped, freeze-influenced label) = {(fn_mask & ~gt_stable).sum()}")

    clean_mask = gt_stable & gt_flat   # gt_stable=True AND gt_habitable=True: unambiguous cases
    clean_tp = (clean_mask & ml_habitable).sum()
    clean_total = clean_mask.sum()
    print(f"  CLEAN-SUBSET habitable recall (gt_stable=True & gt_habitable=True, n={clean_total}): "
          f"{clean_tp}/{clean_total} = {clean_tp/clean_total:.3f}" if clean_total else "  (no clean cases)")

    clean_fn_mask = clean_mask & ~ml_habitable
    onset = first_neg_step[clean_fn_mask]
    onset = onset[onset >= 0]
    if len(onset) > 0:
        print(f"  CLEAN-FN divergence-onset step distribution (n={len(onset)}):")
        for lo, hi in [(10, 20), (21, 100), (101, 300), (301, 600), (601, 999)]:
            cnt = ((onset >= lo) & (onset <= hi)).sum()
            print(f"    steps {lo}-{hi}: {cnt}  ({100*cnt/len(onset):.1f}%)")
        pct = np.percentile(onset, [5, 25, 50, 75, 95])
        print(f"    percentiles: p5={pct[0]:.0f} p25={pct[1]:.0f} median={pct[2]:.0f} "
              f"p75={pct[3]:.0f} p95={pct[4]:.0f}")

    n_post_warmup = N_STEPS - WARMUP_STEPS
    frac_neg = neg_step_count[clean_fn_mask] / n_post_warmup
    if len(frac_neg) > 0:
        pct = np.percentile(frac_neg, [5, 25, 50, 75, 95])
        print(f"  CLEAN-FN fraction-of-steps-negative distribution (n={len(frac_neg)}):")
        print(f"    p5={pct[0]:.3f} p25={pct[1]:.3f} median={pct[2]:.3f} p75={pct[3]:.3f} p95={pct[4]:.3f}")
        for thresh in [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]:
            cnt = (frac_neg <= thresh).sum()
            print(f"    fraction <= {thresh:.2f}: {cnt}/{len(frac_neg)} ({100*cnt/len(frac_neg):.1f}%)")

        # simulate fractional-tolerance recall: how much would clean-subset recall
        # improve if we required habitable for >=X% of post-warmup steps instead of 100%?
        frac_neg_all = neg_step_count[clean_mask] / n_post_warmup
        print(f"  Simulated CLEAN-SUBSET recall under fractional tolerance (n={clean_total}):")
        for tol in [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]:
            would_pass = (frac_neg_all <= tol) & valid_mask[clean_mask]
            print(f"    tolerance <= {tol:.2f} fraction negative: recall = "
                  f"{would_pass.sum()}/{clean_total} = {would_pass.sum()/clean_total:.3f}")


if __name__ == "__main__":
    run_direction("prograde")
    run_direction("retrograde")
