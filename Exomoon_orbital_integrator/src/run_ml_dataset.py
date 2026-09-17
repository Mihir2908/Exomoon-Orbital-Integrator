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
from exomoon.constants import merth, msun, au, rsun
from exomoon.habitable_zone import moon_effective_temp_K, hz_bounds_au

# ── parameter bounds ─────────────────────────────────────────────────────────
# All continuous params are drawn via LHS; moon_retrograde is Bernoulli(0.5)
# mm_earth and am_hill are handled separately with adaptive per-simulation bounds
PARAM_BOUNDS = {
    # (min, max, scale)   scale: "linear" | "log"
    "ms_solar": (0.08,   2.0,     "linear"),
    "rs_solar": (0.08,   2.0,     "linear"),
    "Ts":       (2500.0, 12000.0, "linear"),
    "mp_earth": (0.5,    300.0,   "log"),
    "ap_AU":    (0.01,   3.5,     "linear"),
    "ep":       (0.0,    0.35,    "linear"),
    "em":       (0.0,    0.80,    "linear"),
    "t_sim":    (1.0,    20.0,    "log"),
}
MARS_MASS_EARTH     = 0.107    # lower bound for mm_earth (Mars mass in M_earth)
MM_MAX_CAP          = 3.0      # moon mass <= min(mp_earth, 3.0 M_earth)
MOON_DENSITY_CGS    = 3.0      # g/cm³, rocky moon (matches inference.py)
PLANET_DENSITY_CGS  = 5.5      # g/cm³, dp_cgs default
PLANET_DENSITY_SI   = 5500.0   # kg/m³ (= PLANET_DENSITY_CGS × 1000)
M_EARTH_SOLAR       = merth / msun   # M_earth in solar masses (~3e-6)


def _lhs_sample(n: int, keys: list[str], rng: np.random.Generator,
                bounds: dict | None = None) -> dict[str, np.ndarray]:
    """
    Draw n LHS samples for the named continuous parameters.
    Returns a dict mapping param name → array of n values (already scaled).
    `bounds` defaults to the module-level PARAM_BOUNDS when None.
    """
    if bounds is None:
        bounds = PARAM_BOUNDS
    k = len(keys)
    # LHS: divide [0,1] into n equal-width bins, draw one uniform sample per bin
    cuts = np.linspace(0.0, 1.0, n + 1)
    samples_unit = np.empty((n, k))
    for j in range(k):
        pts = rng.uniform(cuts[:-1], cuts[1:])   # one per bin
        samples_unit[:, j] = rng.permutation(pts)

    result = {}
    for j, key in enumerate(keys):
        lo, hi, scale = bounds[key]
        u = samples_unit[:, j]
        if scale == "log":
            result[key] = np.exp(u * (np.log(hi) - np.log(lo)) + np.log(lo))
        else:
            result[key] = u * (hi - lo) + lo
    return result


def _make_row(rng, ms_i, rs_i, Ts_i, mp_i, ap_i, ep_i, em_i, t_sim_i, retro_i):
    """
    Build one parameter dict given fixed scalar draws, computing adaptive mm/am.
    Returns None when the Roche limit >= Hill sphere (no stable moon orbit possible).
    """
    mm_max = min(mp_i, MM_MAX_CAP)
    mm_min = MARS_MASS_EARTH
    if mm_max <= mm_min:
        mm_max = mm_min * 1.01
    u_mm = rng.uniform(0.0, 1.0)
    mm_i = float(np.exp(u_mm * (np.log(mm_max) - np.log(mm_min)) + np.log(mm_min)))

    mp_kg      = mp_i * merth
    rp_m       = (0.75 * mp_kg / (np.pi * PLANET_DENSITY_SI)) ** (1.0 / 3.0)
    a_roche_m  = 2.456 * rp_m * (PLANET_DENSITY_CGS / MOON_DENSITY_CGS) ** (1.0 / 3.0)
    a_roche_AU = a_roche_m / au
    mp_solar   = mp_i * M_EARTH_SOLAR
    rhill_AU_i = ap_i * (1.0 - ep_i) * (mp_solar / (3.0 * ms_i)) ** (1.0 / 3.0)
    am_min_val = max(a_roche_AU / rhill_AU_i, 1e-3) if rhill_AU_i > 0 else 1e-3

    if am_min_val >= 1.0:
        return None   # Roche limit >= Hill sphere — no valid moon orbit exists

    am_hill_i  = float(rng.uniform(am_min_val, 1.0))

    return {
        "ms_solar":        ms_i,
        "rs_solar":        rs_i,
        "Ts":              Ts_i,
        "mp_earth":        mp_i,
        "ap_AU":           ap_i,
        "ep":              ep_i,
        "mm_earth":        mm_i,
        "am_hill":         am_hill_i,
        "em":              em_i,
        "moon_retrograde": retro_i,
        "t_sim":           t_sim_i,
    }


def _build_param_rows(n: int, seed: int = 42, hz_fraction: float = 0.5,
                      param_overrides: dict | None = None) -> list[dict]:
    """
    Generate n parameter dicts with a controlled in-HZ / out-of-HZ split.

    hz_fraction of n sims have a_inner_au < ap_AU < a_outer_au (planet in HZ).
    Remaining (1 - hz_fraction) sims use full-range ap_AU (out-of-HZ).

    In-HZ pool:
      All params except ap_AU are LHS-sampled over their full PARAM_BOUNDS ranges.
      ap_AU is then drawn per-row from Uniform(a_inner_au, a_outer_au) for that
      row's Ts/rs_solar — so full parameter coverage is preserved for all other dims
      and ap_AU covers the HZ band of each sampled star.

    Out-of-HZ pool:
      Full LHS over all params (including ap_AU over full 0.01-3.5 range), with
      in-HZ draws rejected. Oversample by 4x to absorb rejection rate.
    """
    rng = np.random.default_rng(seed)

    # Build effective bounds: start from module-level defaults, apply any overrides.
    bounds = dict(PARAM_BOUNDS)
    if param_overrides:
        for k, v in param_overrides.items():
            if k in bounds:
                lo_old, hi_old, scale = bounds[k]
                new_lo = v[0] if v[0] is not None else lo_old
                new_hi = v[1] if v[1] is not None else hi_old
                bounds[k] = (new_lo, new_hi, scale)

    n_hz    = int(round(n * hz_fraction))
    n_nonhz = n - n_hz

    keys_no_ap = [k for k in bounds if k != "ap_AU"]
    keys_all   = list(bounds.keys())
    ap_lo_global, ap_hi_global = bounds["ap_AU"][0], bounds["ap_AU"][1]

    # ── In-HZ pool ───────────────────────────────────────────────────────────
    # LHS over all params except ap_AU (2x oversample to absorb degenerate rejections);
    # ap_AU sampled per-row inside HZ band.
    hz_oversample = n_hz * 2
    samp_hz  = _lhs_sample(hz_oversample, keys_no_ap, rng, bounds=bounds)
    retro_hz = rng.integers(0, 2, size=hz_oversample).astype(bool)

    rows_hz = []
    for i in range(hz_oversample):
        if len(rows_hz) >= n_hz:
            break
        ms_i  = float(samp_hz["ms_solar"][i])
        rs_i  = float(samp_hz["rs_solar"][i])
        Ts_i  = float(samp_hz["Ts"][i])
        mp_i  = float(samp_hz["mp_earth"][i])
        ep_i  = float(samp_hz["ep"][i])

        a_in, a_out = hz_bounds_au(Ts_i, rs_i * rsun)
        ap_lo = max(a_in,  ap_lo_global)
        ap_hi = min(a_out, ap_hi_global)
        if ap_lo >= ap_hi:
            # HZ entirely outside sampled ap_AU range for this star (very rare) —
            # fall back to full-range uniform so no row is wasted.
            ap_i = float(rng.uniform(ap_lo_global, ap_hi_global))
        else:
            ap_i = float(rng.uniform(ap_lo, ap_hi))

        row = _make_row(
            rng, ms_i, rs_i, Ts_i, mp_i, ap_i, ep_i,
            float(samp_hz["em"][i]), float(samp_hz["t_sim"][i]), bool(retro_hz[i]),
        )
        if row is not None:
            rows_hz.append(row)

    if len(rows_hz) < n_hz:
        raise RuntimeError(
            f"In-HZ pool: needed {n_hz}, collected only {len(rows_hz)} "
            f"from {hz_oversample} LHS draws. Increase hz_oversample factor."
        )

    # ── Out-of-HZ pool ───────────────────────────────────────────────────────
    # hz_fraction=0.0: pure LHS over all params, no HZ conditioning or rejection
    #   (reproduces the original pre-conditioned sampling behaviour exactly).
    # hz_fraction>0.0: full LHS with 4x oversampling; reject in-HZ draws so the
    #   in-HZ fraction is supplied entirely by the in-HZ pool above.
    pure_lhs_mode = (hz_fraction == 0.0)
    oversample    = n_nonhz if pure_lhs_mode else n_nonhz * 4
    samp_all      = _lhs_sample(oversample, keys_all, rng, bounds=bounds)
    retro_all     = rng.integers(0, 2, size=oversample).astype(bool)

    rows_nonhz = []
    for i in range(oversample):
        if len(rows_nonhz) >= n_nonhz:
            break
        ms_i = float(samp_all["ms_solar"][i])
        rs_i = float(samp_all["rs_solar"][i])
        Ts_i = float(samp_all["Ts"][i])
        ap_i = float(samp_all["ap_AU"][i])
        mp_i = float(samp_all["mp_earth"][i])
        ep_i = float(samp_all["ep"][i])

        if not pure_lhs_mode:
            a_in, a_out = hz_bounds_au(Ts_i, rs_i * rsun)
            if a_in < ap_i < a_out:
                continue   # reject: planet in HZ (conditioned mode only)

        row = _make_row(
            rng, ms_i, rs_i, Ts_i, mp_i, ap_i, ep_i,
            float(samp_all["em"][i]), float(samp_all["t_sim"][i]), bool(retro_all[i]),
        )
        if row is not None:
            rows_nonhz.append(row)

    if len(rows_nonhz) < n_nonhz:
        raise RuntimeError(
            f"Out-of-HZ pool: needed {n_nonhz}, collected only {len(rows_nonhz)} "
            f"from {oversample} LHS draws. Increase oversample factor."
        )

    return rows_hz + rows_nonhz


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


def _freeze_post_escape(resampled: dict) -> dict:
    """
    Once moon_planet_dist first exceeds rhill_AU, the leapfrog integrator keeps
    running for the full t_sim with no truncation -- the ejected moon drifts to
    physically meaningless distances, which swamps the MSE regression loss.

    Freeze moon_star_dist/moon_speed at the value at the moment of escape.
    For moon_planet_dist, clamp to exactly rhill_AU (not the drift-inflated
    resampled value): this runs on the already-resampled 1000-step trajectory,
    so the "freeze_idx" step corresponds to a resampled timestep of t_sim/1000 --
    for small-rhill systems (M-dwarfs, close-in planets, rhill~0.001 AU) the moon
    can drift several×rhill in that one resampled step, making the freeze value >>
    rhill and inflating moon_planet_dist_norm to 10-1000x instead of ≈1.0. Clamping
    to rhill gives the physically correct training signal: post-escape moon_planet_dist
    sits at the Hill boundary, moon_planet_dist_norm = 1.0, stable = 0.

    Also recomputes stable/habitable from clean (post-freeze) distances rather than
    relying on _resample_traj's linear interpolation of the original 0/1 flags, which
    can leave fractional values (e.g. 0.3) at the transition step.
    """
    rhill = resampled["rhill_AU"]
    a_in  = resampled["a_inner_au"]
    a_out = resampled["a_outer_au"]

    stable_before_freeze = resampled["moon_planet_dist"] <= rhill
    unstable_idx = np.where(~stable_before_freeze)[0]

    if len(unstable_idx) > 0:
        freeze_idx = unstable_idx[0]
        # moon_planet_dist: clamp to rhill_AU × (1 + ε) for post-escape steps.
        # rhill is a broadcast array (same value for all steps in this sim); take [0].
        # Using exactly rhill would make (moon_planet_dist <= rhill) True → stable=1
        # (wrong — escaped moon should be stable=0). A tiny ε keeps moon_planet_dist_norm
        # ≈ 1.0 while correctly producing stable=0 from the ≤ rhill check below.
        rhill_val = float(rhill[0]) if hasattr(rhill, '__len__') else float(rhill)
        vals = resampled["moon_planet_dist"].copy()
        vals[freeze_idx:] = rhill_val * (1.0 + 1e-6)   # just above rhill → stable=0
        resampled["moon_planet_dist"] = vals
        for col in ("moon_star_dist", "moon_speed"):
            vals = resampled[col].copy()
            vals[freeze_idx:] = vals[freeze_idx]
            resampled[col] = vals

    resampled["stable"]    = (resampled["moon_planet_dist"] <= rhill).astype(int)
    resampled["habitable"] = (
        (resampled["moon_star_dist"] >= a_in) & (resampled["moon_star_dist"] <= a_out)
    ).astype(int)

    # System-agnostic normalized distances: moon_planet_dist_norm expresses the
    # moon's distance as a fraction of its own system's Hill radius (stable <=> <= 1.0,
    # the same criterion regardless of how large or small rhill_AU is for this system).
    # moon_star_dist_norm expresses position within the HZ band (habitable <=> in [0,1]).
    # Without this, the model has to relearn what an absolute AU value means anew for
    # every combination of stellar/orbital parameters, which both slows learning and
    # contributes to autoregressive drift during inference.
    resampled["moon_planet_dist_norm"] = resampled["moon_planet_dist"] / np.clip(rhill, 1e-9, None)
    hz_width = np.clip(a_out - a_in, 1e-6, None)
    # ap_AU is sampled independently of Ts/rs_solar, and hz_width can be tiny for cool/
    # small stars, so the raw ratio's tail is heavy and wildly variable in scale (observed
    # up to ~39 in this dataset) -- a global StandardScaler fit over that tail compresses
    # the meaningful [0,1] habitable band into a tiny sliver of standardized space.
    # arcsinh is linear near 0 (the band itself keeps full resolution: [0,1] -> [0, 0.881])
    # and compresses logarithmically for large |x| -- unlike a hard clip, it's monotonic
    # and never has zero gradient, so distinct extreme values (e.g. 5.2 vs 39) never alias
    # to the same representation and still contribute some training signal.
    resampled["moon_star_dist_norm"] = np.arcsinh((resampled["moon_star_dist"] - a_in) / hz_width)

    # Moon effective temperature: a second, independently-learned encoding of the
    # SAME habitability fact as moon_star_dist_norm. The Stefan-Boltzmann relation
    # tying Tm to moon_star_dist (and Ts/rs_solar) is only deterministic here in how
    # these training LABELS are generated -- the model's two prediction heads for
    # moon_star_dist_norm vs moon_temp_norm are separately-parameterized nn.Linear
    # layers (see model.py forward()) with no hard-coded conversion between them at
    # inference time, so their agreement/disagreement during autoregressive rollout
    # is a genuine empirical signal of how reliable a given step's prediction is,
    # not a guaranteed-by-construction tautology.
    rs_m = resampled["rs_solar"] * rsun
    Ts_K = resampled["Ts"]
    Tm_K    = moon_effective_temp_K(Ts_K, rs_m, resampled["moon_star_dist"] * au)
    T_hot   = moon_effective_temp_K(Ts_K, rs_m, a_in * au)    # inner HZ edge -> hottest allowed
    T_cold  = moon_effective_temp_K(Ts_K, rs_m, a_out * au)   # outer HZ edge -> coldest allowed
    resampled["habitable_from_temp"] = ((Tm_K <= T_hot) & (Tm_K >= T_cold)).astype(int)
    temp_width = np.clip(T_hot - T_cold, 1e-6, None)
    resampled["moon_temp_norm"] = np.arcsinh((Tm_K - T_cold) / temp_width)

    return resampled


def _run_one(args: tuple) -> dict | None:
    """
    Worker function: run one simulation and return a dict of resampled arrays.
    Returns None on failure (degenerate params, crashed integration, etc.).
    """
    row, sim_id, n_per_orbit, n_max_resample = args
    try:
        # Pre-flight: estimate integration step count before launching the Numba
        # integrator.  Very tight moon orbits produce dt = T_moon/100 << 1/20000,
        # driving n_steps into the millions and causing MemoryError in array allocation.
        # Formula: T_moon = am_AU^1.5 / (mp+mm)^0.5  (Kepler, G=4π², AU/yr/M☉ units)
        _mp_sol = row["mp_earth"] * 3e-6
        _mm_sol = row["mm_earth"] * 3e-6
        _rhill  = (row["ap_AU"] * (1.0 - row["ep"])
                   * (_mp_sol / (3.0 * row["ms_solar"])) ** (1.0 / 3.0))
        _am_AU  = row["am_hill"] * _rhill
        if _am_AU > 0:
            _T_moon = _am_AU ** 1.5 / (_mp_sol + _mm_sol) ** 0.5
            _dt     = min(_T_moon / 100.0, 1.0 / 20_000.0)
            if _dt > 0 and row["t_sim"] / _dt > 500_000:
                return None   # >500k steps → OOM risk; skip cleanly
            # Adaptive resampling: n_per_orbit samples per moon orbital period,
            # capped at n_max_resample. Ensures inner-orbit systems get dense
            # phase coverage (e.g. am_hill=0.094, T_moon=3.6e-3yr, t_sim=10yr
            # → 2778 orbits → 10000 steps, vs 1000 fixed → 0.36 samples/orbit).
            _n_orbits  = row["t_sim"] / max(_T_moon, 1e-10)
            n_resample = max(min(int(n_per_orbit * _n_orbits), n_max_resample), 100)
        else:
            n_resample = 100

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

        # Skip degenerate Hill spheres
        rhill = sim["state"]["rhill_AU"]
        if rhill < 1e-4:
            return None

        # Convert to columnar frame (includes the 17 new ML columns)
        frame = traj_to_frame(sim)

        # Convert DataFrame → dict of numpy arrays
        if hasattr(frame, "to_dict"):
            frame_dict = {col: frame[col].to_numpy() for col in frame.columns}
        else:
            frame_dict = {k: np.asarray(v) for k, v in frame.items()}

        # Skip numerically divergent simulations where the planet escapes the star
        # (planet_star_dist >> ap_AU indicates integrator blow-up, not physical dynamics).
        if "planet_star_dist" in frame_dict and "ap_AU" in frame_dict:
            max_psd = float(np.max(frame_dict["planet_star_dist"]))
            ap_val  = float(frame_dict["ap_AU"][0]) if hasattr(frame_dict["ap_AU"], '__len__') else float(row["ap_AU"])
            if max_psd > 10.0 * ap_val:
                return None   # integrator divergence — discard

        # Resample to fixed grid
        resampled = _resample_traj(frame_dict, n_steps=n_resample)

        # Freeze regression targets after escape (see _freeze_post_escape docstring)
        resampled = _freeze_post_escape(resampled)

        # Tag with simulation_id
        n = n_resample
        resampled["sim_id"] = np.full(n, sim_id, dtype="U36")
        # t_frac (normalised time 0→1, used as a GRU feature)
        resampled["t_frac"] = np.linspace(0.0, 1.0, n)

        return resampled

    except Exception:
        return None


def generate_dataset(
    n_samples:       int   = 3000,
    n_workers:       int   = 4,
    n_per_orbit:     int   = 20,
    n_max_resample:  int   = 10000,
    seed:            int   = 42,
    out_path:        str   = "ml_dataset.parquet",
    hz_fraction:     float = 0.5,
    verbose:         bool  = True,
    param_overrides: dict | None = None,
) -> None:
    """
    Generate the full ML training dataset and write it to a Parquet file.

    Resampling is adaptive per simulation: each trajectory is resampled to
    min(n_per_orbit × (t_sim / T_moon), n_max_resample) steps.  This guarantees
    at least n_per_orbit training samples per moon orbital period regardless of
    how tight the orbit is — fixing the sparse inner-orbit coverage that caused
    HNN force quality failure at small am_hill.
    """
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError("pandas is required — pip install pandas pyarrow")

    n_hz    = int(round(n_samples * hz_fraction))
    n_nonhz = n_samples - n_hz
    if verbose:
        print(f"[dataset] Generating {n_samples} simulations -> {out_path}")
        print(f"[dataset] Adaptive resampling: {n_per_orbit} steps/orbit, cap {n_max_resample}")
        print(f"[dataset] HZ split: {n_hz} in-HZ ({100*hz_fraction:.0f}%) / {n_nonhz} out-of-HZ ({100*(1-hz_fraction):.0f}%)")
        if param_overrides:
            for k, v in param_overrides.items():
                print(f"[dataset] Override: {k} -> [{v[0]}, {v[1]}]")

    rows = _build_param_rows(n_samples, seed=seed, hz_fraction=hz_fraction,
                             param_overrides=param_overrides)
    sim_ids = [str(uuid.uuid4()) for _ in range(n_samples)]
    tasks = [(row, sid, n_per_orbit, n_max_resample) for row, sid in zip(rows, sim_ids)]

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
    parser.add_argument("--n_samples",      type=int,   default=3000,   help="Number of simulations to run")
    parser.add_argument("--n_workers",      type=int,   default=4,      help="Parallel worker processes")
    parser.add_argument("--n_per_orbit",    type=int,   default=20,     help="Resampled steps per moon orbital period (adaptive)")
    parser.add_argument("--n_max_resample", type=int,   default=10000,  help="Maximum resampled steps per simulation")
    parser.add_argument("--seed",           type=int,   default=42,     help="RNG seed for LHS")
    parser.add_argument("--out",        type=str,   default="ml_dataset.parquet",help="Output Parquet path")
    parser.add_argument("--hz_fraction", type=float, default=0.5,                help="Fraction of sims with planet inside stellar HZ (default 0.5)")
    parser.add_argument("--quiet",       action="store_true",                    help="Suppress progress output")
    # Stellar sub-range overrides (for 3-model GRU split — do not affect MLP)
    parser.add_argument("--ms_min",  type=float, default=None, help="Override ms_solar lower bound")
    parser.add_argument("--ms_max",  type=float, default=None, help="Override ms_solar upper bound")
    parser.add_argument("--rs_min",  type=float, default=None, help="Override rs_solar lower bound")
    parser.add_argument("--rs_max",  type=float, default=None, help="Override rs_solar upper bound")
    parser.add_argument("--ts_min",  type=float, default=None, help="Override Ts lower bound (K)")
    parser.add_argument("--ts_max",  type=float, default=None, help="Override Ts upper bound (K)")
    parser.add_argument("--ap_min",  type=float, default=None, help="Override ap_AU lower bound")
    parser.add_argument("--ap_max",  type=float, default=None, help="Override ap_AU upper bound")
    args = parser.parse_args()

    # Build overrides dict — only include params where at least one bound is specified
    param_overrides = {}
    if args.ms_min is not None or args.ms_max is not None:
        param_overrides["ms_solar"] = (args.ms_min, args.ms_max)
    if args.rs_min is not None or args.rs_max is not None:
        param_overrides["rs_solar"] = (args.rs_min, args.rs_max)
    if args.ts_min is not None or args.ts_max is not None:
        param_overrides["Ts"] = (args.ts_min, args.ts_max)
    if args.ap_min is not None or args.ap_max is not None:
        param_overrides["ap_AU"] = (args.ap_min, args.ap_max)

    generate_dataset(
        n_samples       = args.n_samples,
        n_workers       = args.n_workers,
        n_per_orbit     = args.n_per_orbit,
        n_max_resample  = args.n_max_resample,
        seed            = args.seed,
        out_path        = args.out,
        hz_fraction     = args.hz_fraction,
        verbose         = not args.quiet,
        param_overrides = param_overrides if param_overrides else None,
    )
