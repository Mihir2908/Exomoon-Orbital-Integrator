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
SYS_COLS = [
    "ms_solar", "rs_solar", "Ts", "mp_earth", "ap_AU", "ep",
    "mm_earth", "am_hill", "em", "moon_retrograde", "t_sim", "rhill_AU",
]
SYS_DIM = len(SYS_COLS)   # 12

# Per-step state (input features at step t)
STATE_COLS = [
    "moon_planet_dist", "moon_star_dist", "planet_star_dist",
    "moon_speed", "planet_speed",
    "stable", "habitable",
    "t_frac",
]
STATE_DIM = len(STATE_COLS)   # 8

# Per-step targets (what the model predicts: next-step state minus flags)
# Flags get their own BCE head; distances/speeds get MSE head
TARGET_DIST_COLS  = ["moon_planet_dist", "moon_star_dist", "planet_star_dist",
                     "moon_speed", "planet_speed"]
TARGET_FLAG_COLS  = ["stable", "habitable"]
TARGET_COLS       = TARGET_DIST_COLS + TARGET_FLAG_COLS
OUT_DIM           = len(TARGET_COLS)   # 7


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

            return (
                torch.from_numpy(state_norm),          # (T, STATE_DIM)
                torch.from_numpy(sys_vals),             # (SYS_DIM,)
                torch.from_numpy(target_shifted),       # (T, OUT_DIM)
            )


def make_splits(
    parquet_path: str,
    val_frac: float = 0.20,
    seed: int = 42,
) -> tuple["pd.DataFrame", "pd.DataFrame", object, object]:
    """
    Load Parquet, split by simulation_id (not by row), fit scalers on train split.
    Returns (train_df, val_df, sys_scaler, state_scaler).
    """
    df = load_parquet(parquet_path)

    # Drop rows missing required columns (from early incomplete sims).
    # Use dict.fromkeys to deduplicate while preserving order — t_frac appears
    # in both the explicit list and STATE_COLS, so without this df[existing]
    # would produce duplicate column names and raise an error.
    required = ["sim_id"] + SYS_COLS + STATE_COLS
    existing = list(dict.fromkeys(c for c in required if c in df.columns))
    df = df[existing].dropna()

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
