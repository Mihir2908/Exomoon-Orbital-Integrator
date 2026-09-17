"""
run_ground_truth_grid.py — Generate a ground-truth stable+habitable (mm_earth x am_hill)
grid via the real Numba leapfrog integrator, for direct comparison against the ML model's
predict_stability_map() output.

This is a read-only validation tool. It does NOT touch ml_dataset.parquet, the models/
directory, or any existing training/inference code — it only imports (without modifying)
the resampling/labeling helpers from run_ml_dataset.py so the ground-truth labels are
computed identically to how training labels were generated. Output is written to its own
ground_truth_grids/ directory.

Usage:
    # Fetch real system params from the NASA archive
    python run_ground_truth_grid.py --planet "Kepler-1229 b" --t_sim 10 --mm_resolution 30 --am_resolution 30
    python run_ground_truth_grid.py --planet "Kepler-1229 b" --t_sim 10 --retrograde

    # Or specify system params manually
    python run_ground_truth_grid.py --Ts 3784 --rs_solar 0.51 --ms_solar 0.54 --mp_earth 2.54 --ap_AU 0.301 --ep 0.0
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

_src = os.path.dirname(__file__)
if _src not in sys.path:
    sys.path.insert(0, _src)

from exomoon.params import SystemParams
from exomoon.simulation import run_simulation_for_years
from exomoon.eda import traj_to_frame
from exomoon.constants import merth, msun, au
from run_ml_dataset import _resample_traj, _freeze_post_escape, MOON_DENSITY_CGS  # reuse, unmodified

MARS_MASS_EARTH    = 0.107
PLANET_DENSITY_CGS = 5.5
PLANET_DENSITY_SI  = 5500.0

OUT_DIR = os.path.join(_src, "ground_truth_grids")


def build_grids(mp_earth: float, ms_solar: float, ap_AU: float, ep: float,
                 mm_resolution: int, am_resolution: int):
    """Mirrors inference.py's predict_stability_map grid construction exactly."""
    mm_min = MARS_MASS_EARTH
    mm_max = min(mp_earth, 3.0)   # matches inference.py: min(mp_earth, MM_MAX_CAP=3.0)
    if mm_max <= mm_min:
        mm_max = mm_min * 1.01
    mm_grid = np.exp(np.linspace(np.log(mm_min), np.log(mm_max), mm_resolution))

    mp_kg     = mp_earth * merth
    rp_m      = (0.75 * mp_kg / (np.pi * PLANET_DENSITY_SI)) ** (1.0 / 3.0)
    a_roche_m = 2.456 * rp_m * (PLANET_DENSITY_CGS / MOON_DENSITY_CGS) ** (1.0 / 3.0)
    a_roche_AU = a_roche_m / au
    ms_kg     = ms_solar * msun
    rhill_AU  = ap_AU * (1.0 - ep) * (mp_kg / (3.0 * ms_kg)) ** (1.0 / 3.0)
    am_min    = max(a_roche_AU / rhill_AU, 1e-3)
    am_grid   = np.linspace(am_min, 1.0, am_resolution)
    return mm_grid, am_grid


def _run_one_point(args: tuple) -> dict | None:
    """Worker: run one real simulation at a single (mm, am) grid point and return
    the per-step stable/habitable/habitable_from_temp arrays, post warmup exclusion."""
    (i, j, mm, am, sys_params, em, moon_retrograde, t_sim, n_resample, warmup_steps) = args
    try:
        p = SystemParams(
            Ts=sys_params["Ts"], rs_solar=sys_params["rs_solar"], ms_solar=sys_params["ms_solar"],
            mp_earth=sys_params["mp_earth"], ap_AU=sys_params["ap_AU"], ep=sys_params.get("ep", 0.0),
            mm_earth=float(mm), am_hill=float(am), em=em, moon_retrograde=moon_retrograde,
        )
        sim = run_simulation_for_years(p, t_sim)
        rhill = sim["state"]["rhill_AU"]
        if rhill < 1e-4:
            return {"i": i, "j": j, "stable": False, "habitable": False, "habitable_from_temp": False}

        frame = traj_to_frame(sim)
        frame_dict = {col: frame[col].to_numpy() for col in frame.columns}
        resampled = _resample_traj(frame_dict, n_steps=n_resample)
        labeled = _freeze_post_escape(resampled)

        stable_arr         = labeled["stable"][warmup_steps:]
        habitable_arr       = labeled["habitable"][warmup_steps:]
        habitable_temp_arr  = labeled["habitable_from_temp"][warmup_steps:]

        return {
            "i": i, "j": j,
            "stable":              bool(np.all(stable_arr == 1)),
            "habitable":           bool(np.all(habitable_arr == 1)),
            "habitable_from_temp": bool(np.all(habitable_temp_arr == 1)),
        }
    except Exception:
        return {"i": i, "j": j, "stable": False, "habitable": False, "habitable_from_temp": False}


def run_grid(system_params: dict, t_sim: float, moon_retrograde: bool, em: float,
             mm_resolution: int, am_resolution: int, warmup_steps: int = 10,
             n_resample: int = 1000, n_workers: int = 6, verbose: bool = True) -> dict:
    mm_grid, am_grid = build_grids(
        system_params["mp_earth"], system_params["ms_solar"],
        system_params["ap_AU"], system_params.get("ep", 0.0),
        mm_resolution, am_resolution,
    )

    map_stable              = np.zeros((mm_resolution, am_resolution), dtype=bool)
    map_habitable            = np.zeros((mm_resolution, am_resolution), dtype=bool)
    map_habitable_from_temp  = np.zeros((mm_resolution, am_resolution), dtype=bool)

    tasks = []
    for i, mm in enumerate(mm_grid):
        for j, am in enumerate(am_grid):
            tasks.append((i, j, float(mm), float(am), system_params, em, moon_retrograde,
                          t_sim, n_resample, warmup_steps))

    n_total = len(tasks)
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_run_one_point, t) for t in tasks]
        for fut in as_completed(futures):
            r = fut.result()
            map_stable[r["i"], r["j"]] = r["stable"]
            map_habitable[r["i"], r["j"]] = r["habitable"]
            map_habitable_from_temp[r["i"], r["j"]] = r["habitable_from_temp"]
            done += 1
            if verbose and done % max(1, n_total // 10) == 0:
                print(f"  {100*done/n_total:5.1f}%  ({done}/{n_total})", flush=True)
    elapsed = time.time() - t0

    map_both = map_stable & map_habitable

    return {
        "mm_grid": mm_grid, "am_grid": am_grid,
        "map_stable": map_stable, "map_habitable": map_habitable,
        "map_habitable_from_temp": map_habitable_from_temp, "map_both": map_both,
        "elapsed_s": elapsed,
        "meta": {
            "system_params": system_params, "t_sim": t_sim,
            "moon_retrograde": moon_retrograde, "em": em,
            "mm_resolution": mm_resolution, "am_resolution": am_resolution,
            "warmup_steps": warmup_steps, "n_resample": n_resample,
        },
    }


def save_grid(result: dict, out_name: str) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    npz_path = os.path.join(OUT_DIR, f"{out_name}.npz")
    json_path = os.path.join(OUT_DIR, f"{out_name}.meta.json")
    np.savez(
        npz_path,
        mm_grid=result["mm_grid"], am_grid=result["am_grid"],
        map_stable=result["map_stable"], map_habitable=result["map_habitable"],
        map_habitable_from_temp=result["map_habitable_from_temp"], map_both=result["map_both"],
    )
    with open(json_path, "w") as f:
        json.dump({**result["meta"], "elapsed_s": result["elapsed_s"]}, f, indent=2)
    return npz_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planet", type=str, default=None, help="NASA archive planet name, e.g. 'Kepler-1229 b'")
    ap.add_argument("--Ts", type=float, default=None)
    ap.add_argument("--rs_solar", type=float, default=None)
    ap.add_argument("--ms_solar", type=float, default=None)
    ap.add_argument("--mp_earth", type=float, default=None)
    ap.add_argument("--ap_AU", type=float, default=None)
    ap.add_argument("--ep", type=float, default=0.0)
    ap.add_argument("--em", type=float, default=0.0)
    ap.add_argument("--t_sim", type=float, default=10.0)
    ap.add_argument("--mm_resolution", type=int, default=30)
    ap.add_argument("--am_resolution", type=int, default=30)
    ap.add_argument("--warmup_steps", type=int, default=10)
    ap.add_argument("--n_workers", type=int, default=6)
    ap.add_argument("--retrograde", action="store_true")
    ap.add_argument("--out", type=str, default=None, help="Output base name (no extension)")
    args = ap.parse_args()

    if args.planet:
        from exomoon.exoplanet_archive import fetch_system_by_planet
        fetched = fetch_system_by_planet(args.planet)
        if fetched is None:
            print(f"Could not find '{args.planet}' in the NASA archive.")
            sys.exit(1)
        system_params = {
            "Ts": fetched["Ts"], "rs_solar": fetched["rs_solar"], "ms_solar": fetched["ms_solar"],
            "mp_earth": fetched["mp_earth"], "ap_AU": fetched["ap_AU"], "ep": fetched["ep"] or 0.0,
        }
        print(f"Fetched {fetched['pl_name']} ({fetched['hostname']}): {system_params}")
        default_name = fetched["pl_name"].replace(" ", "_").replace("-", "_")
    else:
        required = ["Ts", "rs_solar", "ms_solar", "mp_earth", "ap_AU"]
        missing = [r for r in required if getattr(args, r) is None]
        if missing:
            print(f"Missing required params (or use --planet): {missing}")
            sys.exit(1)
        system_params = {r: getattr(args, r) for r in required}
        system_params["ep"] = args.ep
        default_name = "custom_system"

    direction = "retrograde" if args.retrograde else "prograde"
    out_name = args.out or f"{default_name}_{direction}"

    print(f"Running {args.mm_resolution}x{args.am_resolution} ground-truth grid "
          f"({direction}, t_sim={args.t_sim}yr, warmup_steps={args.warmup_steps})...")
    result = run_grid(
        system_params=system_params, t_sim=args.t_sim, moon_retrograde=args.retrograde, em=args.em,
        mm_resolution=args.mm_resolution, am_resolution=args.am_resolution,
        warmup_steps=args.warmup_steps, n_workers=args.n_workers,
    )
    print(f"Done in {result['elapsed_s']:.1f}s. "
          f"map_both valid: {result['map_both'].sum()}/{result['map_both'].size}")

    path = save_grid(result, out_name)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
