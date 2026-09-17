"""
benchmark_jacobian.py — Compare autograd vs analytical-Jacobian HNN inference.

Runs both batch_hnn_hill_trajectories (original, unmodified) and
batch_hnn_hill_trajectories_fast (analytical Jacobian) on the same system,
then reports:
  - elapsed_s for each
  - speedup ratio
  - whether map_stable / map_habitable / map_both agree exactly

Usage:
    py benchmark_jacobian.py [--system <name>] [--model_dir <dir>] \
                              [--mm_res 50] [--am_res 50] [--t_sim 10.0]

    --system: one of k452bv2_pro, k452bv2_retro, k1229b_pro, k1229b_retro
              (default: k452bv2_pro — relatively fast at 50×50)

Does NOT touch or modify hnn_inference_hill.py or any model directory.
Results are printed to stdout and written to benchmark_jacobian_<system>.txt.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# ── System parameter presets (same as gate eval) ──────────────────────────────
_SYSTEMS = {
    "k452bv2_pro": {
        "params": {"ms_solar": 1.037, "rs_solar": 1.11, "Ts": 5757.0,
                   "mp_earth": 8.3, "ap_AU": 1.046, "ep": 0.0},
        "retrograde": False,
        "label": "Kepler-452b-v2 prograde",
    },
    "k452bv2_retro": {
        "params": {"ms_solar": 1.037, "rs_solar": 1.11, "Ts": 5757.0,
                   "mp_earth": 8.3, "ap_AU": 1.046, "ep": 0.0},
        "retrograde": True,
        "label": "Kepler-452b-v2 retrograde",
    },
    "k1229b_pro": {
        "params": {"ms_solar": 0.54, "rs_solar": 0.51, "Ts": 3820.0,
                   "mp_earth": 2.7, "ap_AU": 0.3006, "ep": 0.0},
        "retrograde": False,
        "label": "Kepler-1229b prograde",
    },
    "k1229b_retro": {
        "params": {"ms_solar": 0.54, "rs_solar": 0.51, "Ts": 3820.0,
                   "mp_earth": 2.7, "ap_AU": 0.3006, "ep": 0.0},
        "retrograde": True,
        "label": "Kepler-1229b retrograde",
    },
}

_DEFAULT_MODEL = "models_hnn_hill_hinge4"


def _maps_agree(r_orig: dict, r_fast: dict, label: str) -> bool:
    ok = True
    for key in ("map_stable", "map_habitable", "map_both"):
        a = np.array(r_orig[key], dtype=bool)
        b = np.array(r_fast[key], dtype=bool)
        if not np.array_equal(a, b):
            diff = int(np.sum(a != b))
            print(f"  [MISMATCH] {label} — {key}: {diff} cells differ")
            ok = False
        else:
            print(f"  [OK] {label} — {key}: identical ({a.sum()} True cells)")
    return ok


def run_benchmark(system_key: str, model_dir: str, mm_res: int, am_res: int,
                  t_sim: float, out_file: str) -> None:

    from exomoon.ml.hnn_inference_hill     import batch_hnn_hill_trajectories
    from exomoon.ml.hnn_inference_hill_jit import batch_hnn_hill_trajectories_jit

    sys_cfg  = _SYSTEMS[system_key]
    params   = sys_cfg["params"]
    retro    = sys_cfg["retrograde"]
    label    = sys_cfg["label"]
    # Pass absolute path so both inference functions resolve it correctly
    # regardless of which subdirectory their own _SRC points to
    md_path  = model_dir if os.path.isabs(model_dir) else os.path.join(_SRC, model_dir)

    lines = []
    def pr(s=""):
        print(s)
        lines.append(s)

    pr(f"{'='*66}")
    pr(f"  Benchmark: {label}")
    pr(f"  System  : {json.dumps(params)}")
    pr(f"  retro   : {retro}")
    pr(f"  Grid    : {mm_res}×{am_res} = {mm_res * am_res} cells")
    pr(f"  t_sim   : {t_sim} yr")
    pr(f"  model   : {model_dir}")
    pr(f"{'='*66}")

    # ── Original (autograd) ────────────────────────────────────────────────────
    pr("\n[1/2] Original autograd inference ...")
    t_orig_start = time.perf_counter()
    r_orig = batch_hnn_hill_trajectories(
        system_params=params,
        t_sim=t_sim,
        moon_retrograde=retro,
        mm_resolution=mm_res,
        am_resolution=am_res,
        model_dir=md_path,
    )
    t_orig_end = time.perf_counter()

    if not r_orig.get("ok"):
        pr(f"  ERROR: {r_orig}")
        return

    t_orig   = r_orig["elapsed_s"]
    n_phys   = r_orig["n_phys"]
    dt_phys  = r_orig["dt_phys"]
    pr(f"  elapsed_s  : {t_orig:.1f}s  (wall: {t_orig_end - t_orig_start:.1f}s)")
    pr(f"  n_phys     : {n_phys:,}")
    pr(f"  dt_phys    : {dt_phys:.2e} yr")
    pr(f"  map_both   : {np.array(r_orig['map_both']).sum()} True cells")

    # ── JIT (analytical Jacobian + torch.jit.script loop) ─────────────────────
    pr("\n[2/2] JIT-compiled inference (analytical Jacobian + scripted C++ loop) ...")
    t_fast_start = time.perf_counter()
    r_fast = batch_hnn_hill_trajectories_jit(
        system_params=params,
        t_sim=t_sim,
        moon_retrograde=retro,
        mm_resolution=mm_res,
        am_resolution=am_res,
        model_dir=md_path,
    )
    t_fast_end = time.perf_counter()

    if not r_fast.get("ok"):
        pr(f"  ERROR: {r_fast}")
        return

    t_fast = r_fast["elapsed_s"]
    pr(f"  elapsed_s  : {t_fast:.1f}s  (wall: {t_fast_end - t_fast_start:.1f}s)")
    pr(f"  n_phys     : {r_fast['n_phys']:,}")
    pr(f"  map_both   : {np.array(r_fast['map_both']).sum()} True cells")

    # ── Comparison ─────────────────────────────────────────────────────────────
    pr("\n── Correctness check ────────────────────────────────────────────────")
    agree = _maps_agree(r_orig, r_fast, label)

    speedup = t_orig / t_fast if t_fast > 0 else float("inf")
    saved_s = t_orig - t_fast

    pr("\n── Timing summary ───────────────────────────────────────────────────")
    pr(f"  autograd elapsed  : {t_orig:.1f}s")
    pr(f"  analytical elapsed: {t_fast:.1f}s")
    pr(f"  speedup           : {speedup:.2f}×")
    pr(f"  time saved        : {saved_s:.1f}s")
    pr(f"  maps agree        : {'YES' if agree else 'NO — CHECK MISMATCH ABOVE'}")

    # ── Record ────────────────────────────────────────────────────────────────
    record = {
        "system":          system_key,
        "label":           label,
        "mm_res":          mm_res,
        "am_res":          am_res,
        "t_sim":           t_sim,
        "model_dir":       model_dir,
        "n_phys":          n_phys,
        "dt_phys":         dt_phys,
        "autograd_elapsed_s":    t_orig,
        "analytical_elapsed_s":  t_fast,
        "speedup":         speedup,
        "time_saved_s":    saved_s,
        "maps_agree":      agree,
    }
    with open(out_file, "w") as f:
        json.dump(record, f, indent=2)
    pr(f"\n  Record saved → {out_file}")
    pr("="*66)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--system",    default="k452bv2_pro",
                    choices=list(_SYSTEMS.keys()),
                    help="System preset (default: k452bv2_pro)")
    ap.add_argument("--model_dir", default=_DEFAULT_MODEL,
                    help=f"HNN model directory relative to src/ (default: {_DEFAULT_MODEL})")
    ap.add_argument("--mm_res",    type=int,   default=50)
    ap.add_argument("--am_res",    type=int,   default=50)
    ap.add_argument("--t_sim",     type=float, default=10.0,
                    help="Simulation time in years (default: 10.0)")
    args = ap.parse_args()

    out_file = os.path.join(_SRC, f"benchmark_jacobian_{args.system}.json")

    run_benchmark(
        system_key=args.system,
        model_dir=args.model_dir,
        mm_res=args.mm_res,
        am_res=args.am_res,
        t_sim=args.t_sim,
        out_file=out_file,
    )


if __name__ == "__main__":
    main()
