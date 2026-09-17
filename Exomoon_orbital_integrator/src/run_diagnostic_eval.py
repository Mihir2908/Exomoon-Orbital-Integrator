"""
run_diagnostic_eval.py — Comprehensive 4-gate evaluation + rollout diagnostics.

Usage:
    py -3 run_diagnostic_eval.py                         # eval all models
    py -3 run_diagnostic_eval.py --trace models_nologfix # eval + per-step traces
    py -3 run_diagnostic_eval.py --trace models          # trace current (temphead)
"""

import argparse, os, sys, json
import numpy as np

_src = os.path.dirname(__file__)
if _src not in sys.path:
    sys.path.insert(0, _src)

import torch
from exomoon.ml.model    import MoonRNN
from exomoon.ml.dataset  import load_normalizer, SYS_COLS, STATE_COLS
from exomoon.ml.inference import predict_stability_map, ARCSINH_1, ARCSINH_1_OUTER

GATES = {
    "K-452b-v2 pro":   "ground_truth_grids/Kepler_452_b_v2_prograde",
    "K-452b-v2 retro": "ground_truth_grids/Kepler_452_b_v2_retrograde",
    "K-1229b pro":     "ground_truth_grids/Kepler_1229_b_prograde",
    "K-1229b retro":   "ground_truth_grids/Kepler_1229_b_retrograde",
}

ALL_MODEL_DIRS = [
    "models",
    "models_temphead",
    "models_nologfix",
    "models_logfix",
    "models_psdfix",
    "models_psdfix_rerun",
    "models_temphead_ss_prefix",
]


def load_gate(prefix):
    npz  = np.load(f"{prefix}.npz", allow_pickle=True)
    meta = json.load(open(f"{prefix}.meta.json"))
    return npz, meta


def gate_recall(npz, meta, model_dir):
    """Run predict_stability_map; compare against GT maps in NPZ."""
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
    if not res.get("ok"):
        return {"error": res.get("message", "unknown"), "ok": False}

    ml_stable = np.array(res["map_stable"])
    ml_hab    = np.array(res["map_habitable"])

    # Ground truth from the NPZ (fields are map_stable, map_habitable, map_both)
    gt_stable = npz["map_stable"].astype(bool)
    gt_hab    = npz["map_habitable"].astype(bool)
    gt_both   = gt_stable & gt_hab
    n_gt      = int(gt_both.sum())

    if n_gt == 0:
        return {"n_gt": 0, "stable_recall": None, "hab_recall": None}

    n_ml_stab     = int((gt_both & ml_stable).sum())
    n_ml_both     = int((gt_both & ml_stable & ml_hab).sum())
    n_ml_hab_head = int((gt_both & ml_hab).sum())    # flag head alone (ignoring stable)

    return {
        "ok": True,
        "stable_recall": n_ml_stab      / n_gt,
        "hab_recall":    n_ml_both      / n_gt,
        "hab_head_only": n_ml_hab_head  / n_gt,
        "n_gt":          n_gt,
        "n_ml_stab":     n_ml_stab,
        "n_ml_both":     n_ml_both,
        "n_ml_hab_head": n_ml_hab_head,
        "ml_stable":     ml_stable,
        "ml_hab":        ml_hab,
        "gt_both":       gt_both,
        "mm_grid":       np.array(res["mm_grid"]),
        "am_grid":       np.array(res["am_grid"]),
    }


def run_rollout_trace(meta, model_dir, gt_both, mm_grid, am_grid, ml_both,
                      n_trace=5, n_steps=250):
    """
    Pick GT-habitable-but-ML-wrong candidates, run the GRU step by step,
    and record msdn / mpdn / tmpn / logits at each step.
    """
    from exomoon.params import SystemParams
    from exomoon.initial_conditions import initial_state
    from exomoon.habitable_zone import hz_bounds_au, moon_effective_temp_K
    from exomoon.constants import rsun, au

    sys_p = meta["system_params"]
    t_sim = meta["t_sim"]
    retro = meta["moon_retrograde"]
    em    = meta["em"]

    model = MoonRNN.load(model_dir, rnn_type="gru")
    model.eval()
    sys_scaler, state_scaler = load_normalizer(os.path.join(model_dir, "normalizer.pkl"))

    ts = float(sys_p["Ts"]); rs = float(sys_p["rs_solar"])
    ms = float(sys_p["ms_solar"]); mp = float(sys_p["mp_earth"])
    ap = float(sys_p["ap_AU"]);  ep = float(sys_p["ep"])

    rs_m = rs * rsun
    a_in, a_out = hz_bounds_au(ts, rs_m)
    hz_w   = max(a_out - a_in, 1e-6)
    T_hot  = moon_effective_temp_K(ts, rs_m, a_in * au)
    T_cold = moon_effective_temp_K(ts, rs_m, a_out * au)
    tw     = max(T_hot - T_cold, 1e-6)

    # Priority: GT-both but ML-wrong; fallback: any GT-both
    wrong = gt_both & ~ml_both
    pool  = wrong if wrong.sum() > 0 else gt_both
    cands = list(zip(*np.where(pool)))
    step_s = max(1, len(cands) // n_trace)
    cands  = cands[::step_s][:n_trace]

    t_fracs = np.linspace(0.0, 1.0, n_steps + 1)
    traces  = {}

    for (mi, ai) in cands:
        mm = float(mm_grid[mi]); am = float(am_grid[ai])
        p  = SystemParams(Ts=ts, rs_solar=rs, ms_solar=ms, mp_earth=mp, ap_AU=ap, ep=ep,
                          mm_earth=mm, am_hill=am, em=em, moon_retrograde=retro)
        try:
            st = initial_state(p)
        except Exception:
            continue

        rhill = float(st["rhill_AU"])
        pos_mm, pos_mp, pos_ms = st["pos_mm"], st["pos_mp"], st["pos_ms"]
        vel_mm, vel_mp = st["vel_mm"], st["vel_mp"]

        mpd0 = float(np.linalg.norm(pos_mm - pos_mp))
        msd0 = float(np.linalg.norm(pos_mm - pos_ms))
        psd0 = float(np.linalg.norm(pos_mp - pos_ms))
        ms0  = float(np.linalg.norm(vel_mm))
        ps0  = float(np.linalg.norm(vel_mp))
        Tm0  = moon_effective_temp_K(ts, rs_m, msd0 * au)

        s0 = np.array([[
            mpd0 / max(rhill, 1e-9),
            np.arcsinh((msd0 - a_in) / hz_w),
            np.arcsinh((Tm0 - T_cold) / tw),
            psd0, ms0, ps0, 0.0,
        ]], dtype=np.float32)

        sys0 = np.array([[ms, rs, ts, mp, ap, ep,
                          mm, am, em, float(int(retro)),
                          t_sim, rhill, a_in, a_out]], dtype=np.float32)

        sn = state_scaler.transform(s0).astype(np.float32)
        yn = sys_scaler.transform(sys0).astype(np.float32)
        state_t = torch.from_numpy(sn).unsqueeze(1)
        sys_exp = torch.from_numpy(yn).unsqueeze(1)

        h = torch.zeros(model.layers, 1, model.hidden)

        mpdn_tr = [float(s0[0, 0])]
        msdn_tr = [float(s0[0, 1])]
        tmpn_tr = [float(s0[0, 2])]
        psd_tr  = [psd0]
        sl_tr, hl_tr, htl_tr = [], [], []

        with torch.no_grad():
            for step in range(n_steps):
                rnn_in  = torch.cat([state_t, sys_exp], dim=-1)
                rnn_out, h = model.rnn(rnn_in, h)
                raw_d = model.head_dist(rnn_out)
                dist_p = torch.cat([
                    torch.relu(raw_d[..., 0:1]),
                    raw_d[..., 1:3],
                    torch.relu(raw_d[..., 3:6]),
                ], dim=-1)
                flag_p = model.head_flags(rnn_out)

                d = dist_p.squeeze().cpu().numpy()
                f = flag_p.squeeze().cpu().numpy()

                mpdn_tr.append(float(d[0]))
                msdn_tr.append(float(d[1]))
                tmpn_tr.append(float(d[2]))
                psd_tr.append(float(d[3]))
                sl_tr.append(float(f[0]))
                hl_tr.append(float(f[1]))
                htl_tr.append(float(f[2]))

                nxt = np.array([[d[0], d[1], d[2], d[3], d[4], d[5],
                                 t_fracs[step + 1]]], dtype=np.float32)
                state_t = torch.from_numpy(
                    state_scaler.transform(nxt).astype(np.float32)
                ).unsqueeze(1)

        traces[(int(mi), int(ai))] = {
            "mm": mm, "am": am, "rhill": rhill,
            "mpdn": mpdn_tr, "msdn": msdn_tr, "tmpn": tmpn_tr, "psd": psd_tr,
            "sl": sl_tr, "hl": hl_tr, "htl": htl_tr,
            "a_in": a_in, "a_out": a_out,
            "gt_both": bool(gt_both[int(mi), int(ai)]),
            "ml_both": bool(ml_both[int(mi), int(ai)]),
        }

    return traces


def print_trace(gate_name, traces):
    if not traces:
        print("    (no traces generated)")
        return
    for (mi, ai), tr in traces.items():
        mpdn = np.array(tr["mpdn"])
        msdn = np.array(tr["msdn"])
        tmpn = np.array(tr["tmpn"])
        sl   = np.array(tr["sl"])
        hl   = np.array(tr["hl"])
        n    = len(mpdn)

        bad_msdn = np.where((msdn < 0) | (msdn > ARCSINH_1_OUTER))[0]
        bad_mpdn = np.where(mpdn > 1.0)[0]
        bad_sl   = np.where(sl < 0)[0]    # starts at index 0 = step 1
        bad_hl   = np.where(hl < 0)[0]

        print(f"\n    mm={tr['mm']:.3f} Me  am={tr['am']:.3f} Hill  rhill={tr['rhill']:.5f} AU  "
              f"gt={tr['gt_both']}  ml={tr['ml_both']}")
        print(f"    a_in={tr['a_in']:.4f} AU  a_out={tr['a_out']:.4f} AU")
        print(f"    HZ window (arcsinh): msdn in [0.000, {ARCSINH_1_OUTER:.3f}]")
        print()
        print(f"    {'step':>5}  {'mpdn':>8}  {'msdn':>8}  {'tmpn':>8}  {'stab_l':>8}  {'hab_l':>8}")
        print(f"    {'-----'}  {'--------'}  {'--------'}  {'--------'}  {'--------'}  {'--------'}")
        show_steps = [0, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200]
        for s in show_steps:
            if s >= n:
                break
            sl_v  = float(sl[s-1]) if s > 0 and s-1 < len(sl) else float("nan")
            hl_v  = float(hl[s-1]) if s > 0 and s-1 < len(hl) else float("nan")
            flag  = ""
            if s > 0:
                if msdn[s] < 0 or msdn[s] > ARCSINH_1_OUTER:
                    flag += "  << msdn EXIT"
                if mpdn[s] > 1.0:
                    flag += "  << mpdn>1"
            print(f"    {s:>5}  {mpdn[s]:>8.4f}  {msdn[s]:>8.4f}  {tmpn[s]:>8.4f}  "
                  f"{sl_v:>8.4f}  {hl_v:>8.4f}{flag}")
        print()
        print(f"    First msdn outside [0, {ARCSINH_1_OUTER:.3f}]:  "
              + (f"step {bad_msdn[0]}  (val={msdn[bad_msdn[0]]:.4f})" if len(bad_msdn) else "NEVER"))
        print(f"    First mpdn > 1.0 (dist head):         "
              + (f"step {bad_mpdn[0]}" if len(bad_mpdn) else "NEVER"))
        print(f"    First stable_logit < 0 (flag head):   "
              + (f"step {bad_sl[0]+1}" if len(bad_sl) else "NEVER"))
        print(f"    First hab_logit < 0 (flag head):      "
              + (f"step {bad_hl[0]+1}" if len(bad_hl) else "NEVER"))
        print(f"    msdn fixed-point at step 200:  {msdn[min(200,n-1)]:.4f}")
        print(f"    msdn range steps 10-200:  [{msdn[10:201].min():.4f}, {msdn[10:201].max():.4f}]")
        print(f"    stable_logit range [10-200]:  [{sl[9:200].min():.4f}, {sl[9:200].max():.4f}]")
        print(f"    hab_logit    range [10-200]:  [{hl[9:200].min():.4f}, {hl[9:200].max():.4f}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=str, default=None)
    parser.add_argument("--model_dirs", type=str, default=None)
    args = parser.parse_args()

    model_dirs = (
        [d.strip() for d in args.model_dirs.split(",")]
        if args.model_dirs else ALL_MODEL_DIRS
    )
    model_dirs = [d for d in model_dirs if os.path.isdir(d)]

    gate_data = {}
    for gname, gpath in GATES.items():
        try:
            gate_data[gname] = load_gate(gpath)
        except Exception as e:
            print(f"Warning: gate {gname} not found: {e}")

    W = 24
    print("\n" + "="*115)
    print("4-GATE EVALUATION — ALL STATE_DIM=7 MODELS")
    print("Metric: (gt_stable & gt_hab & ml_stable & ml_hab) / (gt_stable & gt_hab)")
    print("="*115)
    print(f"\n  {'Model':<34}", end="")
    for gn in gate_data:
        print(f"{gn:^{W}}", end="")
    print()
    print("  " + "-"*110)

    gate_results_by_model = {}
    for mdir in model_dirs:
        cfg_p = os.path.join(mdir, "gru_model_config.json")
        if not os.path.exists(cfg_p):
            continue
        with open(cfg_p) as f:
            cfg = json.load(f)
        if cfg.get("state_dim", 0) != 7:
            continue
        try:
            st = json.load(open(os.path.join(mdir, "train_status.json")))
            vl = f"val={st.get('val_loss',0):.4f}"
        except Exception:
            vl = "val=?"

        gate_res = {}
        for gname, (npz, meta) in gate_data.items():
            try:
                gate_res[gname] = gate_recall(npz, meta, mdir)
            except Exception as e:
                gate_res[gname] = {"error": str(e)}
        gate_results_by_model[mdir] = gate_res

        print(f"\n  {mdir:<34}  [{vl}  state_dim={cfg['state_dim']}]")
        for row_label, key in [
            ("stable recall", "stable_recall"),
            ("hab recall (both heads)", "hab_recall"),
            ("hab flag head only", "hab_head_only"),
        ]:
            print(f"    {row_label:<26}", end="")
            for gname in gate_data:
                r = gate_res.get(gname, {})
                if not r or r.get("error"):
                    cell = f"ERR: {str(r.get('error',''))[:10]}"
                elif r.get("n_gt", 0) == 0:
                    cell = "no GT"
                elif r.get(key) is None:
                    cell = "—"
                else:
                    v = r[key]
                    n = r.get("n_ml_stab" if key=="stable_recall"
                              else ("n_ml_both" if key=="hab_recall" else "n_ml_hab_head"), "?")
                    cell = f"{v:.3f} ({n}/{r['n_gt']})"
                print(f"{cell:^{W}}", end="")
            print()

    # ── Rollout traces ─────────────────────────────────────────────────────────
    if args.trace:
        mdir = args.trace
        print(f"\n{'='*115}")
        print(f"ROLLOUT TRACES — {mdir}")
        print(f"Tracing GT-habitable-but-ML-wrong candidates, first 200 steps")
        print(f"HZ window: msdn in [0.000, {ARCSINH_1_OUTER:.3f}]  |  Stable: mpdn < 1.0")
        print("="*115)

        for gname, (npz, meta) in gate_data.items():
            print(f"\n{'-'*80}\nGate: {gname}")
            gr = gate_results_by_model.get(mdir, {}).get(gname)
            if not gr or gr.get("error") or not gr.get("ok"):
                # Try evaluating now
                try:
                    gr = gate_recall(npz, meta, mdir)
                except Exception as e:
                    print(f"  Eval error: {e}"); continue

            if not gr or not gr.get("ok"):
                print(f"  No valid result"); continue

            print(f"  stable={gr['stable_recall']:.3f}  hab={gr['hab_recall']:.3f}  "
                  f"hab_head_only={gr['hab_head_only']:.3f}  n_gt={gr['n_gt']}")

            gt_both = gr["gt_both"]
            ml_both = gr["ml_stable"] & gr["ml_hab"]
            mm_grid = gr["mm_grid"]
            am_grid = gr["am_grid"]

            try:
                traces = run_rollout_trace(
                    meta, mdir, gt_both, mm_grid, am_grid, ml_both,
                    n_trace=3, n_steps=250,
                )
                print_trace(gname, traces)
            except Exception as e:
                print(f"  Trace error: {e}")
                import traceback; traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    main()
