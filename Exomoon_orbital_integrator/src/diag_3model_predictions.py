"""
diag_3model_predictions.py — DIAGNOSTIC ONLY, no production changes.

Two checks:
1. Prediction coverage: what fraction of the 50×50 grid each model predicts
   as stable / habitable. A fraction near 1.0 = degenerate "predict everything" classifier.

2. G/F/A rollout trajectory for K-452b-v2: tracks mean msd_norm and mpd_norm
   across all 2500 candidates over the 1000-step autoregressive rollout, showing
   whether the dist head drifts immediately (warm-up artifact) or tracks then collapses.
"""

import os, sys, json
import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from exomoon.ml.inference import predict_stability_map, ARCSINH_1, ARCSINH_1_OUTER

MODEL_MDWARF = os.path.join(SRC, "models_mdwarf")
MODEL_KTYPE  = os.path.join(SRC, "models_ktype")
MODEL_GTYPE  = os.path.join(SRC, "models_gtype")
MODEL_GFA    = os.path.join(SRC, "models_gfa")

GATES = {
    "K-452b-v2 pro":     (os.path.join(SRC, "ground_truth_grids", "Kepler_452_b_v2_prograde"),    MODEL_GTYPE),
    "K-452b-v2 retro":   (os.path.join(SRC, "ground_truth_grids", "Kepler_452_b_v2_retrograde"),  MODEL_GTYPE),
    "K-1229b pro":       (os.path.join(SRC, "ground_truth_grids", "Kepler_1229_b_prograde"),      MODEL_KTYPE),
    "K-1229b retro":     (os.path.join(SRC, "ground_truth_grids", "Kepler_1229_b_retrograde"),    MODEL_KTYPE),
    "TRAPPIST-1e pro":   (os.path.join(SRC, "ground_truth_grids", "TRAPPIST_1_e_prograde"),       MODEL_MDWARF),
    "TRAPPIST-1e retro": (os.path.join(SRC, "ground_truth_grids", "TRAPPIST_1_e_retrograde"),     MODEL_MDWARF),
    "Kepler-442b pro":   (os.path.join(SRC, "ground_truth_grids", "Kepler_442_b_prograde"),       MODEL_KTYPE),
    "Kepler-442b retro": (os.path.join(SRC, "ground_truth_grids", "Kepler_442_b_retrograde"),     MODEL_KTYPE),
}
GATES = {k: v for k, v in GATES.items() if os.path.exists(v[0] + ".npz")}


# ── Check 1: prediction coverage ─────────────────────────────────────────────

print("=" * 80)
print("CHECK 1: Prediction coverage (fraction of 50×50 grid predicted stable / habitable)")
print("  ~1.0 = degenerate 'predict everything' classifier")
print("=" * 80)
print(f"  {'Gate':22} {'model':8} {'pred_stable':>12} {'pred_hab':>12} {'gt_stable':>12} {'gt_hab':>12}")
print("-" * 80)

for gate_name, (gate_prefix, model_dir) in GATES.items():
    npz  = np.load(f"{gate_prefix}.npz", allow_pickle=True)
    with open(f"{gate_prefix}.meta.json") as f:
        meta = json.load(f)

    gt_s = npz["map_stable"].astype(bool)
    gt_h = npz["map_habitable"].astype(bool)
    n    = gt_s.size

    res = predict_stability_map(
        system_params   = meta["system_params"],
        t_sim           = meta["t_sim"],
        moon_retrograde = meta["moon_retrograde"],
        em              = meta["em"],
        mm_resolution   = meta["mm_resolution"],
        am_resolution   = meta["am_resolution"],
        model_dir       = model_dir,
        n_steps         = 1000,
    )

    ml_s = np.array(res["map_stable"])
    ml_h = np.array(res["map_habitable"])

    model_label = {MODEL_MDWARF: "M-dwarf", MODEL_KTYPE: "K-type", MODEL_GTYPE: "G-type", MODEL_GFA: "G/F/A"}[model_dir]

    print(f"  {gate_name:22} {model_label:8}"
          f"  {ml_s.mean():>11.3f}"
          f"  {ml_h.mean():>11.3f}"
          f"  {gt_s.mean():>11.3f}"
          f"  {(gt_s & gt_h).mean():>11.3f}")

print()


# ── Check 2: G/F/A rollout trajectory for K-452b-v2 ─────────────────────────

print("=" * 80)
print("CHECK 2: G/F/A autoregressive rollout — mean msd_norm and mpd_norm over time")
print("  for K-452b-v2 prograde (all 2500 candidates averaged)")
print(f"  Thresholds: ARCSINH_1={ARCSINH_1:.3f} (inner HZ), ARCSINH_1_OUTER={ARCSINH_1_OUTER:.3f} (outer HZ+5%)")
print("=" * 80)

try:
    import torch
    from exomoon.ml.model   import MoonRNN
    from exomoon.ml.dataset import load_normalizer, SYS_COLS, STATE_COLS
    from exomoon.params     import SystemParams
    from exomoon.initial_conditions import initial_state
    from exomoon.habitable_zone import hz_bounds_au, moon_effective_temp_K
    from exomoon.constants  import rsun, merth, msun, au

    gate_prefix = os.path.join(SRC, "ground_truth_grids", "Kepler_452_b_v2_prograde")
    with open(f"{gate_prefix}.meta.json") as f:
        meta = json.load(f)

    sp       = meta["system_params"]
    ms       = float(sp["ms_solar"]); rs = float(sp["rs_solar"]); Ts = float(sp["Ts"])
    mp       = float(sp["mp_earth"]); ap = float(sp["ap_AU"]);    ep = float(sp["ep"])
    t_sim    = float(meta["t_sim"])
    retro    = bool(meta["moon_retrograde"])
    em       = float(meta["em"])
    mm_res   = int(meta["mm_resolution"]); am_res = int(meta["am_resolution"])

    model = MoonRNN.load(MODEL_GTYPE, rnn_type="gru")
    model.eval()
    sys_scaler, state_scaler = load_normalizer(os.path.join(MODEL_GTYPE, "normalizer.pkl"))

    _PDC = 5.5          # planet density g/cm³ (matches run_ml_dataset.py)
    PLANET_DENSITY_SI = 5500.0
    MOON_DENSITY_CGS  = 3.0
    M_EARTH_SOLAR     = merth / msun

    mp_kg    = mp * merth
    rp_m     = (0.75 * mp_kg / (np.pi * PLANET_DENSITY_SI)) ** (1/3)
    a_roche  = 2.456 * rp_m * (_PDC / MOON_DENSITY_CGS) ** (1/3) / au
    mp_sol   = mp * M_EARTH_SOLAR
    rhill    = ap * (1 - ep) * (mp_sol / (3 * ms)) ** (1/3)
    am_min   = max(a_roche / rhill, 1e-3)

    mm_max   = min(mp * 0.30, 0.5)
    mm_grid  = np.exp(np.linspace(np.log(0.107), np.log(max(mm_max, 0.108)), mm_res))
    am_grid  = np.linspace(am_min, 1.0, am_res)

    a_inner, a_outer = hz_bounds_au(Ts, rs * rsun)
    hz_width = max(a_outer - a_inner, 1e-6)
    T_hot    = moon_effective_temp_K(Ts, rs * rsun, a_inner * au)
    T_cold   = moon_effective_temp_K(Ts, rs * rsun, a_outer * au)
    temp_width = max(T_hot - T_cold, 1e-6)

    # Build initial states for all 2500 candidates
    states_list = []
    for mm in mm_grid:
        for am in am_grid:
            try:
                p = SystemParams(Ts=Ts, rs_solar=rs, ms_solar=ms, mp_earth=mp,
                                 ap_AU=ap, ep=ep, mm_earth=float(mm),
                                 am_hill=float(am), em=em, moon_retrograde=retro)
                st = initial_state(p)
                mpd   = float(np.linalg.norm(st["pos_mm"][:2] - st["pos_mp"][:2]))
                msd   = float(np.linalg.norm(st["pos_mm"][:2] - st["pos_ms"][:2]))
                mspd  = float(np.linalg.norm(st["vel_mm"][:2]))
                pspd  = float(np.linalg.norm(st["vel_mp"][:2]))
                psd   = float(np.linalg.norm(st["pos_mp"][:2] - st["pos_ms"][:2]))
                Tm    = moon_effective_temp_K(Ts, rs * rsun, msd * au)

                mpd_norm = mpd / max(rhill, 1e-9)
                msd_norm = float(np.arcsinh((msd - a_inner) / hz_width))
                tmp_norm = float(np.arcsinh((Tm  - T_cold)  / temp_width))

                states_list.append([mpd_norm, msd_norm, tmp_norm, psd, mspd, pspd, 0.0])
            except Exception:
                states_list.append([1.0, 0.0, 0.0, ap, 0.0, 0.0, 0.0])

    state_arr = np.array(states_list, dtype=np.float32)     # (2500, 7)
    sys_vec   = np.array([[ms, rs, Ts, mp, ap, ep, 0.107, 0.5, em,
                           float(retro), t_sim, rhill, a_inner, a_outer]] * len(states_list),
                         dtype=np.float32)

    state_norm = state_scaler.transform(state_arr)
    sys_norm   = sys_scaler.transform(sys_vec)

    state_t = torch.tensor(state_norm[:, None, :], dtype=torch.float32)  # (N, 1, 7)
    sys_t   = torch.tensor(sys_norm,               dtype=torch.float32)  # (N, 14)

    n_steps = 1000
    msd_trace = np.zeros((n_steps, len(states_list)), dtype=np.float32)
    mpd_trace = np.zeros((n_steps, len(states_list)), dtype=np.float32)

    with torch.no_grad():
        h = torch.zeros(model.layers, len(states_list), model.hidden)
        cur = state_t.squeeze(1)   # (N, 7)
        sys_exp = sys_t            # (N, 14)

        for step in range(n_steps):
            rnn_in         = torch.cat([cur, sys_exp], dim=-1).unsqueeze(1)
            rnn_out, h     = model.rnn(rnn_in, h)
            raw_dist       = model.head_dist(rnn_out.squeeze(1))
            dist_pred      = torch.cat([
                torch.relu(raw_dist[:, 0:1]),
                raw_dist[:, 1:3],
                torch.relu(raw_dist[:, 3:6]),
            ], dim=-1)

            # msd_norm is index 1 in dist_pred (moon_star_dist_norm)
            # mpd_norm is index 0 (moon_planet_dist_norm)
            msd_trace[step] = dist_pred[:, 1].numpy()
            mpd_trace[step] = dist_pred[:, 0].numpy()

            t_frac = torch.full((len(states_list), 1), (step + 1) / n_steps)
            pred_state_raw = torch.cat([dist_pred, t_frac], dim=-1)
            cur = torch.tensor(
                state_scaler.transform(pred_state_raw.numpy()), dtype=torch.float32
            )

    # Print trace at key steps
    print(f"\n  Step-by-step mean msd_norm and mpd_norm across all 2500 candidates:")
    print(f"  {'step':>6}  {'mean_msd_norm':>14}  {'mean_mpd_norm':>14}  "
          f"{'frac_msd>thresh':>16}  {'frac_mpd>1.0':>13}")
    print(f"  {'':>6}  {'(hab if <' + f'{ARCSINH_1_OUTER:.3f})':>14}  "
          f"{'(stab if <1.0)':>14}")
    print("  " + "-" * 72)

    checkpoints = list(range(0, 50)) + [100, 200, 300, 500, 750, 999]
    for step in checkpoints:
        mean_msd = float(msd_trace[step].mean())
        mean_mpd = float(mpd_trace[step].mean())
        frac_msd_over = float((msd_trace[step] > ARCSINH_1_OUTER).mean())
        frac_mpd_over = float((mpd_trace[step] > 1.0).mean())
        print(f"  {step:>6}  {mean_msd:>14.4f}  {mean_mpd:>14.4f}  "
              f"  {frac_msd_over:>14.3f}  {frac_mpd_over:>13.3f}")

except Exception as e:
    import traceback
    print(f"  Rollout diagnostic failed: {e}")
    traceback.print_exc()

print()
print("Done. No production models modified.")
