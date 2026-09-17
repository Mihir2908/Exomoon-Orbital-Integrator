"""
exomoon/ml/hnn_dataset_hill.py  —  Hill-frame training dataset for HNN_Hill.

Transforms ml_dataset.parquet rows into Hill-frame inputs and force targets.

Hill-frame axes (computed per-timestep from planet/star state vectors):
  x_h : radial      (star → planet direction)
  y_h : tangential  (planet orbital velocity direction)
  z_h : normal      (angular momentum direction = L_sp / |L_sp|)

HNN input:
  q_h_norm  =  (x_h, y_h, z_h) / rhill_AU        [3D, dimensionless]
  sys_enc   =  StandardScaler on log([ms_solar, mp_earth, mm_earth,
                                       ap_AU, rhill_AU, Omega_yr_inst])

HNN training target (= ∂V_θ/∂q_h_norm):
  t_hill  =  −mm_msun × a_conservative_hill × rhill / V_ref   [3D, dimensionless]

  where:
    a_conservative_hill  =  R^T ⊗ (a_moon_newton − a_planet_newton)  +  Ω² × (x_h, y_h, 0)
      R^T ⊗ (a_moon − a_planet) = relative Newton accel rotated to Hill frame
      + Ω² × (x_h, y_h, 0)     = centrifugal acceleration (encodes centrifugal potential)
      Planet accel ≈ a_star_on_planet (mm ≪ ms, moon pull on planet negligible)

  Coriolis (2Ω × v_hill) is NOT in the target — it is velocity-dependent and
  added analytically during leapfrog integration.

Sign convention:
  t_hill is POSITIVE in the outward radial direction when x_h > 0.
  (Gradient of an attractive potential points outward; force = −gradient points inward.)
  Verified: t_hill_x ≈ (mm/ms)(ap/rhill)/d_pm_n²  (= t_pm from Track 1) for
  radial positions, confirming dimensional consistency.

Training/validation split: by sim_id (same 80/20 as Track 1), no row-level leakage.
Filter: keep only sims that start stable+habitable (first row stable=1 AND habitable=1).
Trim per-sim at the first step where stable=0 AND habitable=0 simultaneously.
"""

import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

_G               = 4.0 * np.pi ** 2          # AU³ / (Msun · yr²)
_MERTH_OVER_MSUN = 5.976e24 / 1.989e30       # Earth mass in solar masses

# System parameter columns (raw, before log + scaler)
_SYS_COLS_HILL = ["ms_solar", "mp_earth", "mm_earth", "ap_AU", "rhill_AU"]
# 6th sys param (Omega_yr) is computed per-row, not read from parquet directly


# ── Internal: Hill-frame coordinate transform ─────────────────────────────────

def _compute_hill_targets(df: pd.DataFrame):
    """
    Vectorised Hill-frame computation over all rows in df.

    Returns
    -------
    q_h_norm      : (N, 3) float32  normalised Hill-frame moon position
    t_hill        : (N, 3) float32  target gradient ∂V_θ/∂q_h_norm
    Omega_inst    : (N,)   float64  instantaneous angular velocity [rad/yr]
    valid         : (N,)   bool     True where frame is non-degenerate
    """
    N = len(df)

    # ── Inertial positions [AU] and velocities [AU/yr] ─────────────────────
    r_s = df[["star_x",   "star_y",   "star_z"]].values.astype(np.float64)   # (N,3)
    r_p = df[["planet_x", "planet_y", "planet_z"]].values.astype(np.float64)
    r_m = df[["moon_x",   "moon_y",   "moon_z"]].values.astype(np.float64)
    v_s = df[["star_vx",   "star_vy",   "star_vz"]].values.astype(np.float64)
    v_p = df[["planet_vx", "planet_vy", "planet_vz"]].values.astype(np.float64)

    ms_flat    = df["ms_solar"].values.astype(np.float64)   # (N,) Msun
    mp_flat    = df["mp_earth"].values.astype(np.float64) * _MERTH_OVER_MSUN  # Msun
    mm_flat    = df["mm_earth"].values.astype(np.float64) * _MERTH_OVER_MSUN  # Msun
    rhill_flat = df["rhill_AU"].values.astype(np.float64)   # (N,) AU
    ap_flat    = df["ap_AU"].values.astype(np.float64)      # (N,) AU

    V_ref = _G * ms_flat * mp_flat / ap_flat                # (N,) Msun·AU²/yr² / AU = Msun·AU/yr²
    # Note: V_ref has units such that V_ref/rhill = force reference [Msun·AU/yr² / AU]
    # Force target = F_moon [Msun·AU/yr²] / (V_ref / rhill) = dimensionless

    # ── Hill-frame basis vectors ────────────────────────────────────────────
    r_sp = r_p - r_s                                        # (N,3) star→planet [AU]
    r_sp_sq = (r_sp ** 2).sum(axis=1)                       # (N,) |r_sp|² [AU²]
    r_sp_norm = np.sqrt(r_sp_sq)                            # (N,) |r_sp| [AU]

    valid = r_sp_norm > 1e-10

    # x̂_h = (r_p - r_s) / |r_p - r_s|  (radial)
    denom_r = np.where(valid, r_sp_norm, 1.0)
    x_hat = r_sp / denom_r[:, None]                         # (N,3)

    # Angular momentum vector L = r_sp × v_sp_rel
    v_sp = v_p - v_s                                        # (N,3) relative velocity
    L_sp = np.cross(r_sp, v_sp)                             # (N,3) AU²/yr
    L_norm = np.linalg.norm(L_sp, axis=1)                   # (N,)

    valid &= (L_norm > 1e-10)

    # ẑ_h = L_sp / |L_sp|  (orbital normal)
    denom_L = np.where(valid, L_norm, 1.0)
    z_hat = L_sp / denom_L[:, None]                         # (N,3)

    # ŷ_h = ẑ_h × x̂_h  (tangential, completes right-hand frame)
    y_hat = np.cross(z_hat, x_hat)                          # (N,3)

    # Instantaneous angular velocity Ω = |L_sp| / |r_sp|²
    Omega_inst = L_norm / np.maximum(r_sp_sq, 1e-20)        # (N,) rad/yr

    # ── Moon position in Hill frame ─────────────────────────────────────────
    r_mp = r_m - r_p                                        # (N,3) moon rel to planet [AU]

    # q_h = R^T · r_mp  where R = [x̂_h | ŷ_h | ẑ_h] (columns)
    q_h_x = (r_mp * x_hat).sum(axis=1)                     # (N,) [AU]
    q_h_y = (r_mp * y_hat).sum(axis=1)
    q_h_z = (r_mp * z_hat).sum(axis=1)

    q_h_norm_x = q_h_x / np.maximum(rhill_flat, 1e-20)     # (N,) dimensionless
    q_h_norm_y = q_h_y / np.maximum(rhill_flat, 1e-20)
    q_h_norm_z = q_h_z / np.maximum(rhill_flat, 1e-20)

    q_h_norm = np.stack([q_h_norm_x, q_h_norm_y, q_h_norm_z], axis=1)  # (N,3)

    # ── Newton acceleration on moon and planet (inertial frame) ───────────
    r_sm = r_m - r_s                                        # (N,3) moon - star [AU]
    r_pm = r_mp                                             # (N,3) = r_m - r_p [AU]

    d_sm = np.linalg.norm(r_sm, axis=1, keepdims=True)     # (N,1) [AU]
    d_pm = np.linalg.norm(r_pm, axis=1, keepdims=True)     # (N,1) [AU]
    d_sp_col = r_sp_norm[:, None]                          # (N,1) [AU] (reuse above)

    # a = −G · M / d³ · r̂  (per unit test-mass → acceleration [AU/yr²])
    a_star_on_moon   = -_G * ms_flat[:, None] / (d_sm ** 3) * r_sm   # (N,3)
    a_planet_on_moon = -_G * mp_flat[:, None] / (d_pm ** 3) * r_pm   # (N,3)
    a_moon_newton = a_star_on_moon + a_planet_on_moon                 # (N,3) [AU/yr²]

    # Planet acceleration (dominated by star; mm << ms so moon pull negligible)
    a_star_on_planet = -_G * ms_flat[:, None] / (d_sp_col ** 3) * r_sp  # (N,3)

    # Moon acceleration RELATIVE to planet — the Hill frame EOM base.
    # Hill frame is centred on the planet, so the conservative part of the EOM is:
    #   ü_hill = R^T(a_moon − a_planet) + Ω²(q_h_x, q_h_y, 0) + Coriolis
    # Omitting a_planet gives an erroneous +G·ms/ap² ≈ 230 AU/yr² term in x.
    a_moon_rel = a_moon_newton - a_star_on_planet                     # (N,3) [AU/yr²]

    # ── Rotate relative acceleration to Hill frame ──────────────────────────
    a_hill_x = (a_moon_rel * x_hat).sum(axis=1)            # (N,)
    a_hill_y = (a_moon_rel * y_hat).sum(axis=1)
    a_hill_z = (a_moon_rel * z_hat).sum(axis=1)

    # ── Centrifugal correction: a_cf = Ω² × (x_h, y_h, 0) ─────────────────
    # (Centrifugal acts only in the orbital plane; z component = 0.)
    Omega2 = Omega_inst ** 2                                # (N,) [rad²/yr²]
    cf_x = Omega2 * q_h_x                                  # (N,) [AU/yr²] via AU·rad²/yr²
    cf_y = Omega2 * q_h_y

    # Conservative Hill-frame acceleration (Newton rotated + centrifugal)
    a_cons_h_x = a_hill_x + cf_x                           # (N,)
    a_cons_h_y = a_hill_y + cf_y
    a_cons_h_z = a_hill_z                                   # (centrifugal z = 0)

    # ── Dimensionless target gradient: t_hill = ∂V_θ/∂q_h_norm ────────────
    # Derivation:  V_θ_dim = V_phys / V_ref,  q_h_norm = q_h / rhill
    # ∂V_θ_dim/∂q_h_norm = −(mm_msun · rhill / V_ref) · a_conservative_hill
    #
    # Sign check: for moon at +x_h, a_cons_h_x ≈ −G·mp/d_pm² (inward, negative).
    # → t_hill_x ≈ +(mm·rhill/V_ref)·G·mp/d_pm² = (mm/ms)·(ap/rhill)/d_pm_n² = t_pm ✓
    scale = mm_flat * rhill_flat / V_ref                    # (N,) dimensionless scale

    t_hill = np.stack([
        -scale * a_cons_h_x,
        -scale * a_cons_h_y,
        -scale * a_cons_h_z,
    ], axis=1)                                              # (N,3) dimensionless

    return (
        q_h_norm.astype(np.float32),
        t_hill.astype(np.float32),
        Omega_inst.astype(np.float64),
        valid,
    )


# ── Internal: parquet loader (same filter as hnn_dataset.py) ─────────────────

def _load_parquet_hill(path: str, val_frac: float, seed: int):
    """
    Load parquet, apply training filter, compute Hill-frame targets.

    Filter (identical to hnn_dataset.py):
      - Keep only sims whose FIRST row has stable=1 AND habitable=1
      - Trim each sim at the first step where stable=0 AND habitable=0 simultaneously

    Split by sim_id (not by row) to prevent sequence-level data leakage.

    Returns
    -------
    train_rows, val_rows : filtered DataFrames with added columns
      q_h_norm_x/y/z, t_hill_x/y/z, Omega_inst
    """
    df = pd.read_parquet(path)

    # ── Step 1: keep sims starting stable+habitable ─────────────────────────
    first = df.groupby("sim_id").first().reset_index()
    valid_sims = set(
        first.loc[(first["stable"] == 1) & (first["habitable"] == 1), "sim_id"]
    )
    df = df[df["sim_id"].isin(valid_sims)].copy()

    # ── Step 2: trim per-sim at first step where both stable=0 AND habitable=0
    def _trim(grp):
        bad = (grp["stable"] == 0) & (grp["habitable"] == 0)
        first_bad = bad.values.argmax() if bad.any() else len(grp)
        if not bad.any():
            first_bad = len(grp)
        return grp.iloc[:first_bad]

    df = df.groupby("sim_id", group_keys=False).apply(_trim).reset_index(drop=True)

    # ── Step 3: compute Hill-frame quantities ───────────────────────────────
    q_h_norm, t_hill, Omega_inst, valid = _compute_hill_targets(df)

    # Store as additional columns
    df["q_h_norm_x"] = q_h_norm[:, 0].astype(np.float32)
    df["q_h_norm_y"] = q_h_norm[:, 1].astype(np.float32)
    df["q_h_norm_z"] = q_h_norm[:, 2].astype(np.float32)
    df["t_hill_x"]   = t_hill[:, 0].astype(np.float32)
    df["t_hill_y"]   = t_hill[:, 1].astype(np.float32)
    df["t_hill_z"]   = t_hill[:, 2].astype(np.float32)
    df["Omega_inst"] = Omega_inst

    # ── Step 4: split by sim_id ─────────────────────────────────────────────
    all_sims = np.array(sorted(df["sim_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(all_sims)
    n_val = max(1, int(len(all_sims) * val_frac))
    val_sims  = set(all_sims[:n_val])
    train_sims = set(all_sims[n_val:])

    train_df = df[df["sim_id"].isin(train_sims)].reset_index(drop=True)
    val_df   = df[df["sim_id"].isin(val_sims)].reset_index(drop=True)

    return train_df, val_df


# ── Dataset class ─────────────────────────────────────────────────────────────

class HnnHillDataset(Dataset):
    """
    PyTorch Dataset for Hill-frame HNN training.

    Each item is a single timestep row:
      q_h_norm : (3,)  normalised Hill-frame moon position
      sys_enc  : (6,)  log-standardised system params [ms,mp,mm,ap,rhill,Omega]
      t_hill   : (3,)  target gradient ∂V_θ/∂q_h_norm
      stable   : (,)   1 if moon is still bound, 0 if escaped
    """

    def __init__(self, df: pd.DataFrame, sys_scaler: StandardScaler) -> None:
        self.q_h_norm = torch.tensor(
            df[["q_h_norm_x", "q_h_norm_y", "q_h_norm_z"]].values.astype(np.float32)
        )
        self.t_hill = torch.tensor(
            df[["t_hill_x", "t_hill_y", "t_hill_z"]].values.astype(np.float32)
        )
        self.stable = torch.tensor(
            df["stable"].values.astype(np.float32)
        )

        # Build sys_raw: [ms_solar, mp_earth, mm_earth, ap_AU, rhill_AU, Omega_inst]
        sys_raw = np.column_stack([
            df["ms_solar"].values,
            df["mp_earth"].values,
            df["mm_earth"].values,
            df["ap_AU"].values,
            df["rhill_AU"].values,
            df["Omega_inst"].values,
        ]).astype(np.float64)
        sys_log = np.log(np.maximum(sys_raw, 1e-20))
        self.sys_enc = torch.tensor(
            sys_scaler.transform(sys_log).astype(np.float32)
        )

    def __len__(self) -> int:
        return len(self.q_h_norm)

    def __getitem__(self, idx):
        return (
            self.q_h_norm[idx],
            self.sys_enc[idx],
            self.t_hill[idx],
            self.stable[idx],
        )


# ── Public API ────────────────────────────────────────────────────────────────

def make_hnn_hill_splits(
    path:      str,
    val_frac:  float = 0.20,
    seed:      int   = 42,
):
    """
    Load parquet, apply filter, compute Hill-frame targets, split by sim_id.

    Returns
    -------
    train_ds  : HnnHillDataset
    val_ds    : HnnHillDataset
    sys_scaler: fitted StandardScaler (on log-transformed sys_raw of training rows)
    """
    train_df, val_df = _load_parquet_hill(path, val_frac, seed)

    # Fit scaler on TRAINING rows only
    sys_raw_train = np.column_stack([
        train_df["ms_solar"].values,
        train_df["mp_earth"].values,
        train_df["mm_earth"].values,
        train_df["ap_AU"].values,
        train_df["rhill_AU"].values,
        train_df["Omega_inst"].values,
    ]).astype(np.float64)

    scaler = StandardScaler()
    scaler.fit(np.log(np.maximum(sys_raw_train, 1e-20)))

    train_ds = HnnHillDataset(train_df, scaler)
    val_ds   = HnnHillDataset(val_df,   scaler)

    return train_ds, val_ds, scaler


def save_hill_sys_scaler(scaler: StandardScaler, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "hnn_hill_sys_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)


def load_hill_sys_scaler(model_dir: str) -> StandardScaler:
    with open(os.path.join(model_dir, "hnn_hill_sys_scaler.pkl"), "rb") as f:
        return pickle.load(f)


def compute_hill_sys_enc(
    ms_solar:  float,
    mp_earth:  float,
    mm_earth:  float,
    ap_AU:     float,
    rhill_AU:  float,
    Omega_yr:  float,
    scaler:    StandardScaler,
) -> "np.ndarray":
    """
    Build a single normalised sys_enc vector for inference.

    Omega_yr should be the instantaneous angular velocity computed from the
    current planet position and velocity:
        r_sp = r_p - r_s
        v_sp = v_p - v_s
        L    = cross(r_sp, v_sp)
        Omega_yr = norm(L) / dot(r_sp, r_sp)
    """
    import numpy as _np
    sys_raw = _np.array([[ms_solar, mp_earth, mm_earth, ap_AU, rhill_AU, Omega_yr]],
                        dtype=np.float64)
    sys_log = _np.log(_np.maximum(sys_raw, 1e-20))
    return scaler.transform(sys_log).astype(np.float32)   # (1, 6)
