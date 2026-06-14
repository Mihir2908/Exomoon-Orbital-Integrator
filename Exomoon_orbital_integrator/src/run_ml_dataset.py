"""
run_ml_dataset.py — Generate a training dataset for the ML moon stability predictor.

Uses Latin Hypercube Sampling (LHS) to draw N parameter combinations, runs each
through the Numba leapfrog integrator, resamples every trajectory to a fixed 1000-step
grid, and writes a single Parquet file suitable for training MoonRNN.

Usage:
    python run_ml_dataset.py --n_samples 3000 --n_workers 6 --out ml_dataset.parquet
    python run_ml_dataset.py --n_samples 50   --n_workers 2 --out test.parquet   # quick smoke test
"""

import argparse
import os
import sys
import uuid
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# ── allow running from the src/ directory or from the repo root ──────────────
_src = os.path.join(os.path.dirname(__file__))
if _src not in sys.path:
    sys.path.insert(0, _src)

from exomoon.params import SystemParams
from exomoon.simulation import run_simulation_for_years
from exomoon.eda import traj_to_frame

# ── parameter bounds ─────────────────────────────────────────────────────────
# All continuous params are drawn via LHS; moon_retrograde is Bernoulli(0.5)
PARAM_BOUNDS = {
    # (min, max, scale)   scale: "linear" | "log"
    "ms_solar": (0.4,    2.0,   "linear"),
    "rs_solar": (0.4,    2.0,   "linear"),
    "Ts":       (3000.0, 12000.0, "linear"),
    "mp_earth": (0.5,    300.0, "log"),
    "ap_AU":    (0.2,    3.5,   "linear"),
    "ep":       (0.0,    0.35,  "linear"),
    # mm_earth handled separately (adaptive upper bound = min(mp*0.30, 0.5))
    "am_hill":  (0.05,   0.80,  "linear"),
    "em":       (0.0,    0.25,  "linear"),
    "t_sim":    (1.0,    20.0,  "log"),
}
MARS_MASS_EARTH = 0.107   # lower bound for mm_earth (Mars mass in M_earth)
MM_MAX_FRAC     = 0.30    # moon mass <= 30% of planet mass
MM_MAX_CAP      = 0.50    # absolute cap in M_earth


def _lhs_sample(n: int, keys: list[str], rng: np.random.Generator) -> dict[str, np.ndarray]:
    """
    Draw n LHS samples for the named continuous parameters.
    Returns a dict mapping param name → array of n values (already scaled).
    """
    k = len(keys)
    # LHS: divide [0,1] into n equal-width bins, draw one uniform sample per bin
    cuts = np.linspace(0.0, 1.0, n + 1)
    samples_unit = np.empty((n, k))
    for j in range(k):
        pts = rng.uniform(cuts[:-1], cuts[1:])   # one per bin
        samples_unit[:, j] = rng.permutation(pts)

    result = {}
    for j, key in enumerate(keys):
        lo, hi, scale = PARAM_BOUNDS[key]
        u = samples_unit[:, j]
        if scale == "log":
            result[key] = np.exp(u * (np.log(hi) - np.log(lo)) + np.log(lo))
        else:
            result[key] = u * (hi - lo) + lo
    return result


def _build_param_rows(n: int, seed: int = 42) -> list[dict]:
    """
    Generate n parameter dicts via LHS.  mm_earth uses an adaptive upper bound
    derived from the already-sampled mp_earth for each row.
    """
    rng  = np.random.default_rng(seed)
    keys = list(PARAM_BOUNDS.keys())
    samp = _lhs_sample(n, keys, rng)

    # Bernoulli draw for moon_retrograde
    retrograde = rng.integers(0, 2, size=n).astype(bool)   # 50/50

    rows = []
    for i in range(n):
        mp = float(samp["mp_earth"][i])
        mm_max = min(mp * MM_MAX_FRAC, MM_MAX_CAP)
        mm_min = MARS_MASS_EARTH
        if mm_max <= mm_min:
            mm_max = mm_min * 1.01   # degenerate case: tiny planet, keep mm tiny too
        # Sample mm_earth log-uniformly in [mm_min, mm_max] using a fresh uniform draw
        u_mm = rng.uniform(0.0, 1.0)
        mm = np.exp(u_mm * (np.log(mm_max) - np.log(mm_min)) + np.log(mm_min))

        rows.append({
            "ms_solar":        float(samp["ms_solar"][i]),
            "rs_solar":        float(samp["rs_solar"][i]),
            "Ts":              float(samp["Ts"][i]),
            "mp_earth":        mp,
            "ap_AU":           float(samp["ap_AU"][i]),
            "ep":              float(samp["ep"][i]),
            "mm_earth":        float(mm),
            "am_hill":         float(samp["am_hill"][i]),
            "em":              float(samp["em"][i]),
            "moon_retrograde": bool(retrograde[i]),
            "t_sim":           float(samp["t_sim"][i]),
        })
    return rows


def _resample_traj(frame_dict: dict, n_steps: int = 1000) -> dict:
    """
    Resample a trajectory (dict of 1-D or 2-D arrays keyed by column name)
    from its native timestep grid to n_steps uniformly-spaced steps via linear interp.
    Returns a new dict with the same keys but length n_steps.
    """
    t_orig = frame_dict["t_years"]
    t_new  = np.linspace(t_orig[0], t_orig[-1], n_steps)
    out = {}
    for k, v in frame_dict.items():
        if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == len(t_orig):
            out[k] = np.interp(t_new, t_orig, v)
        else:
            out[k] = v   # scalars / non-array values passed through unchanged
    return out


def _run_one(args: tuple) -> dict | None:
    """
    Worker function: run one simulation and return a dict of resampled arrays.
    Returns None on failure (degenerate params, crashed integration, etc.).
    """
    row, sim_id, n_resample = args
    try:
        p = SystemParams(
            Ts            = row["Ts"],
            rs_solar      = row["rs_solar"],
            ms_solar      = row["ms_solar"],
            mp_earth      = row["mp_earth"],
            ap_AU         = row["ap_AU"],
            ep            = row["ep"],
            mm_earth      = row["mm_earth"],
            am_hill       = row["am_hill"],
            em            = row["em"],
            moon_retrograde = row["moon_retrograde"],
        )
        years = row["t_sim"]

        # Run simulation
        sim = run_simulation_for_years(p, years)

        # Skip degenerate Hill spheres or moon starting outside Hill sphere
        rhill = sim["state"]["rhill_AU"]
        am_AU = sim["state"]["am_AU"]
        if rhill < 1e-4 or am_AU > 0.95 * rhill:
            return None

        # Convert to columnar frame (includes the 17 new ML columns)
        frame = traj_to_frame(sim)

        # Convert DataFrame → dict of numpy arrays
        if hasattr(frame, "to_dict"):
            frame_dict = {col: frame[col].to_numpy() for col in frame.columns}
        else:
            frame_dict = {k: np.asarray(v) for k, v in frame.items()}

        # Resample to fixed grid
        resampled = _resample_traj(frame_dict, n_steps=n_resample)

        # Tag with simulation_id
        n = n_resample
        resampled["sim_id"] = np.full(n, sim_id, dtype="U36")
        # t_frac (normalised time 0→1, used as a GRU feature)
        resampled["t_frac"] = np.linspace(0.0, 1.0, n)

        return resampled

    except Exception:
        return None


def generate_dataset(
    n_samples:  int   = 3000,
    n_workers:  int   = 4,
    n_resample: int   = 1000,
    seed:       int   = 42,
    out_path:   str   = "ml_dataset.parquet",
    verbose:    bool  = True,
) -> None:
    """
    Generate the full ML training dataset and write it to a Parquet file.
    """
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError("pandas is required — pip install pandas pyarrow")

    if verbose:
        print(f"[dataset] Generating {n_samples} simulations -> {out_path}")
        print(f"[dataset] Workers: {n_workers}  |  Steps per sim: {n_resample}")

    rows = _build_param_rows(n_samples, seed=seed)
    sim_ids = [str(uuid.uuid4()) for _ in range(n_samples)]
    tasks = [(row, sid, n_resample) for row, sid in zip(rows, sim_ids)]

    completed = 0
    skipped   = 0
    all_chunks: list[dict] = []

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_one, t): i for i, t in enumerate(tasks)}
        for fut in as_completed(futures):
            result = fut.result()
            if result is None:
                skipped += 1
            else:
                all_chunks.append(result)
                completed += 1
            if verbose and (completed + skipped) % max(1, n_samples // 20) == 0:
                pct = 100 * (completed + skipped) / n_samples
                print(f"  {pct:5.1f}%  completed={completed}  skipped={skipped}", flush=True)

    if verbose:
        print(f"[dataset] Done: {completed} sims written, {skipped} skipped")

    if not all_chunks:
        raise RuntimeError("All simulations failed — check parameter bounds or integrator.")

    # Concatenate all resampled chunks into one DataFrame
    col_keys = list(all_chunks[0].keys())
    merged: dict[str, list] = {k: [] for k in col_keys}
    for chunk in all_chunks:
        for k in col_keys:
            arr = chunk.get(k)
            if arr is not None:
                merged[k].append(arr)

    df_dict = {}
    for k in col_keys:
        parts = merged[k]
        if not parts:
            continue
        try:
            df_dict[k] = np.concatenate(parts) if isinstance(parts[0], np.ndarray) else parts
        except Exception:
            df_dict[k] = parts

    df = pd.DataFrame(df_dict)

    # Write Parquet
    df.to_parquet(out_path, index=False)
    if verbose:
        mb = os.path.getsize(out_path) / 1e6
        print(f"[dataset] Wrote {len(df):,} rows x {len(df.columns)} cols -> {out_path} ({mb:.1f} MB)")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ML moon stability training dataset")
    parser.add_argument("--n_samples",  type=int,   default=3000,                help="Number of simulations to run")
    parser.add_argument("--n_workers",  type=int,   default=4,                   help="Parallel worker processes")
    parser.add_argument("--n_resample", type=int,   default=1000,                help="Timesteps per trajectory after resampling")
    parser.add_argument("--seed",       type=int,   default=42,                  help="RNG seed for LHS")
    parser.add_argument("--out",        type=str,   default="ml_dataset.parquet",help="Output Parquet path")
    parser.add_argument("--quiet",      action="store_true",                     help="Suppress progress output")
    args = parser.parse_args()

    generate_dataset(
        n_samples  = args.n_samples,
        n_workers  = args.n_workers,
        n_resample = args.n_resample,
        seed       = args.seed,
        out_path   = args.out,
        verbose    = not args.quiet,
    )
