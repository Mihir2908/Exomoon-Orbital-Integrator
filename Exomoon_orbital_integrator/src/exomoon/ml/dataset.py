"""
exomoon/ml/dataset.py — PyTorch Dataset for MoonRNN training.

Each sample is one simulation resampled to 1000 steps.
  - x_seq   : (1000, STATE_DIM)  — per-step state features
  - y_seq   : (1000, OUT_DIM)    — per-step targets (next-step prediction)
  - sys_params: (SYS_DIM,)       — constant system parameters for this sim

The dataset normalises all features using a StandardScaler fitted on the
training split; the scaler is persisted as normalizer.pkl alongside the model.
"""

import os
import pickle
from typing import Optional

import numpy as np

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

try:
    import torch
    from torch.utils.data import Dataset
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# ── feature columns ───────────────────────────────────────────────────────────
# System params: constant per simulation (used as conditioning at every step)
# a_inner_au/a_outer_au are deterministic Stefan-Boltzmann HZ bounds derived from
# Ts/rs_solar -- the same category of closed-form physics quantity as rhill_AU
# (already a SYS_COLS input), so giving the model both is consistent, not a new
# kind of information leak. Without a_inner_au/a_outer_au, the model had to
# reverse-engineer the HZ-location-from-stellar-params relationship purely from
# training examples, which was a likely contributor to habitable being the harder,
# more error-prone of the two flag classifications.
SYS_COLS = [
    "ms_solar", "rs_solar", "Ts", "mp_earth", "ap_AU", "ep",
    "mm_earth", "am_hill", "em", "moon_retrograde", "t_sim", "rhill_AU",
    "a_inner_au", "a_outer_au",
]
SYS_DIM = len(SYS_COLS)   # 14

# These four columns are LHS-sampled (or, for rhill_AU, multiplicatively derived from
# LHS-sampled quantities) on a LOG scale, but were previously fed raw into a linear
# StandardScaler. A linear scaler preserves whatever skew the raw values have, so
# equal-count percentile groups at the low end of a log-sampled range get crushed into
# a tiny z-score window relative to the high end (empirically: mp_earth's bottom decile
# spans ~290x less z-score resolution than its top decile, despite equal sample counts
# and equal multiplicative spread in both). Log-transforming before scaling re-aligns
# the scaling geometry with the sampling geometry, exactly as arcsinh already does for
# moon_star_dist_norm/moon_temp_norm elsewhere in this pipeline.
LOG_SYS_COLS = ["mp_earth", "mm_earth", "t_sim", "rhill_AU"]


def _log_transform_sys_cols(df: "pd.DataFrame") -> "pd.DataFrame":
    """Return a copy of df with LOG_SYS_COLS replaced by their natural log.
    All four columns are strictly positive physical quantities (masses, durations,
    Hill radii), so log() is always well-defined."""
    df = df.copy()
    for col in LOG_SYS_COLS:
        if col in df.columns:
            df[col] = np.log(df[col].values.astype(np.float64))
    return df


def _add_planet_star_dist_norm(df: "pd.DataFrame") -> "pd.DataFrame":
    """Add planet_star_dist_norm = planet_star_dist / ap_AU.

    Raw planet_star_dist (AU) means something different in every system, exactly the
    same problem that motivated normalizing moon_planet_dist (by rhill_AU) and
    moon_star_dist (by HZ band) instead of feeding them raw -- this was the one
    distance column left un-normalized. Dividing by ap_AU (the system's own orbital
    semi-major axis, already a SYS_COLS input) re-expresses it as orbital phase: ~1.0
    at zero eccentricity, oscillating in [1-ep, 1+ep] for eccentric orbits -- a
    bounded, cross-system-comparable ratio, mirroring exactly how am_hill already
    expresses the moon's orbit as a Hill-radius fraction rather than raw AU.
    planet_star_dist is also part of TARGET_DIST_COLS (fed back autoregressively),
    so an un-normalized, system-scale-dependent representation there could plausibly
    degrade its own rollout stability -- and since moon_star_dist is dominated by
    planet_star_dist, that instability would propagate into the already-weak
    habitable channel too.
    """
    df = df.copy()
    if "planet_star_dist" in df.columns and "ap_AU" in df.columns:
        df["planet_star_dist_norm"] = df["planet_star_dist"].values / df["ap_AU"].values
    return df

# Per-step state (input features at step t).
# stable and habitable are deliberately excluded: feeding binary flags back into
# the GRU during autoregressive inference creates hard attractor states (stable=1
# or stable=0) that prevent the model from discovering stability thresholds on its
# own. The model receives only continuous kinematic features and learns stability
# via the flag head targets (BCE loss), not via binary input feedback.
#
# moon_planet_dist_norm / moon_star_dist_norm replace the raw-AU distance columns:
# raw AU values mean something different in every system (the same 0.78 AU can be
# inside or outside the HZ depending on the star), forcing the model to relearn an
# absolute-scale interpretation per system. Normalizing by rhill_AU (stable <=> <=1.0)
# and by HZ band position (habitable <=> in [0,1]) makes both quantities universal
# across systems and is exactly analogous to how am_hill already expresses the moon's
# orbital semi-major axis as a Hill-radius fraction rather than raw AU.
#
# moon_temp_norm is a second, independently-derived encoding of the same habitability
# fact via the Stefan-Boltzmann equilibrium-temperature relation (habitable <=> in
# [0,1], same convention as moon_star_dist_norm). It is mathematically redundant with
# moon_star_dist_norm in the training LABELS (both come from the same true
# moon_star_dist), but the model predicts it through a SEPARATE nn.Linear head with
# its own weights -- nothing forces the two heads to agree at inference time. Their
# disagreement during autoregressive rollout is therefore a genuine, free reliability
# signal: a real ~6-step disagreement window was observed exactly at a discontinuity
# in the existing moon_star_dist_norm/habitable pair, confirming the two heads can
# and do diverge independently rather than always moving in lockstep.
STATE_COLS = [
    "moon_planet_dist_norm", "moon_star_dist_norm", "moon_temp_norm", "planet_star_dist",
    "moon_speed", "planet_speed",
    "t_frac",
]
STATE_DIM = len(STATE_COLS)   # 7

# Per-step targets (what the model predicts: next-step state minus flags)
# Flags get their own BCE head; distances/speeds get MSE head
TARGET_DIST_COLS  = ["moon_planet_dist_norm", "moon_star_dist_norm", "moon_temp_norm",
                     "planet_star_dist", "moon_speed", "planet_speed"]
TARGET_FLAG_COLS  = ["stable", "habitable", "habitable_from_temp"]
TARGET_COLS       = TARGET_DIST_COLS + TARGET_FLAG_COLS
OUT_DIM           = len(TARGET_COLS)   # 9


def load_parquet(path: str) -> "pd.DataFrame":
    if not _HAS_PANDAS:
        raise RuntimeError("pandas required — pip install pandas pyarrow")
    import pandas as pd
    return pd.read_parquet(path)


def fit_normalizer(df: "pd.DataFrame", cols: list[str]) -> object:
    """Fit a StandardScaler on the given columns and return it."""
    try:
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise RuntimeError("scikit-learn required — pip install scikit-learn")
    scaler = StandardScaler()
    scaler.fit(df[cols].values.astype(np.float32))
    return scaler


def save_normalizer(scaler, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(scaler, f)


def load_normalizer(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _safe_cols(df: "pd.DataFrame", cols: list[str]) -> list[str]:
    """Return only the columns that actually exist in df."""
    return [c for c in cols if c in df.columns]


if _HAS_TORCH:
    class MoonDataset(Dataset):
        """
        PyTorch Dataset — one item per simulation.

        Parameters
        ----------
        df          : full Parquet DataFrame (already filtered to train or val split)
        sys_scaler  : fitted StandardScaler for SYS_COLS
        state_scaler: fitted StandardScaler for STATE_COLS
        """

        def __init__(
            self,
            df: "pd.DataFrame",
            sys_scaler=None,
            state_scaler=None,
        ):
            if not _HAS_PANDAS:
                raise RuntimeError("pandas required")

            self.sim_ids = df["sim_id"].unique()
            self._df     = df
            self.sys_scaler   = sys_scaler
            self.state_scaler = state_scaler

            # Pre-group for fast __getitem__
            self._groups = {sid: grp for sid, grp in df.groupby("sim_id")}

        def __len__(self) -> int:
            return len(self.sim_ids)

        def __getitem__(self, idx: int):
            sid   = self.sim_ids[idx]
            grp   = self._groups[sid].sort_values("t_frac")

            # System params (first row — constant across sequence)
            sys_vals_raw = grp[_safe_cols(grp, SYS_COLS)].iloc[0].values.astype(np.float32)
            if self.sys_scaler is not None:
                sys_vals = self.sys_scaler.transform(sys_vals_raw.reshape(1, -1))[0]
            else:
                sys_vals = sys_vals_raw

            # Per-step state
            state_raw = grp[_safe_cols(grp, STATE_COLS)].values.astype(np.float32)
            if self.state_scaler is not None:
                state_norm = self.state_scaler.transform(state_raw)
            else:
                state_norm = state_raw

            # Targets = next-step values (shift by 1; last step duplicated)
            target_raw = grp[_safe_cols(grp, TARGET_COLS)].values.astype(np.float32)
            target_shifted = np.concatenate([target_raw[1:], target_raw[-1:]], axis=0)

            # Distance/speed regression is only meaningful while the moon remains
            # bound to the planet. Once stable=0, the raw trajectory is unbounded
            # ballistic drift post-escape with no relevance to the stability-mapping
            # task; dist_mask excludes those rows from the MSE term in compute_loss
            # while the flag head (BCE) stays fully supervised on every row.
            stable_idx = len(TARGET_DIST_COLS)   # "stable" is the first flag column
            dist_mask = target_shifted[:, stable_idx:stable_idx + 1].copy()   # (T, 1)

            return (
                torch.from_numpy(state_norm),          # (T, STATE_DIM)
                torch.from_numpy(sys_vals),             # (SYS_DIM,)
                torch.from_numpy(target_shifted),       # (T, OUT_DIM)
                torch.from_numpy(dist_mask),            # (T, 1)
            )


def make_splits(
    parquet_path: str,
    val_frac: float = 0.20,
    seed: int = 42,
    log_transform: bool = False,
) -> tuple["pd.DataFrame", "pd.DataFrame", object, object]:
    """
    Load Parquet, split by simulation_id (not by row), fit scalers on train split.
    Returns (train_df, val_df, sys_scaler, state_scaler).

    log_transform: apply _log_transform_sys_cols before fitting/returning. Defaults
    to False -- the log transform broke autoregressive habitable recall in practice
    (K-452b-v2 dropped from 0.386 to 0.016) despite fixing resolution-compression,
    which the dose-response test showed was not the dominant driver of recall problems.
    Pass True only when explicitly experimenting with it on a fresh training run.
    """
    df = load_parquet(parquet_path)

    # Drop rows missing required columns (from early incomplete sims).
    # Use dict.fromkeys to deduplicate while preserving order — t_frac appears
    # in both the explicit list and STATE_COLS, so without this df[existing]
    # would produce duplicate column names and raise an error.
    required = ["sim_id"] + SYS_COLS + STATE_COLS + TARGET_FLAG_COLS
    existing = list(dict.fromkeys(c for c in required if c in df.columns))
    df = df[existing].dropna()
    if log_transform:
        df = _log_transform_sys_cols(df)

    # Split by sim_id
    rng     = np.random.default_rng(seed)
    sim_ids = df["sim_id"].unique()
    rng.shuffle(sim_ids)
    n_val   = max(1, int(len(sim_ids) * val_frac))
    val_ids = set(sim_ids[:n_val])

    train_df = df[~df["sim_id"].isin(val_ids)].reset_index(drop=True)
    val_df   = df[ df["sim_id"].isin(val_ids)].reset_index(drop=True)

    sys_scaler   = fit_normalizer(train_df, _safe_cols(train_df, SYS_COLS))
    state_scaler = fit_normalizer(train_df, _safe_cols(train_df, STATE_COLS))

    return train_df, val_df, sys_scaler, state_scaler
