"""
exomoon/ml/hnn_dataset.py — Training dataset for the distance-based HNN.

Each sample is (d_n, sys_enc, t_grad) where:
  d_n     : (3,) physics-normalised pair distances [d_sp/ap, d_sm/ap, d_pm/rhill]
  sys_enc : (5,) StandardScaler-normalised [ms_solar, mp_earth, mm_earth, ap_AU, rhill_AU]
  t_grad  : (3,) dimensionless force targets [t_sp, t_sm, t_pm]

Why physics-based normalisation instead of StandardScaler on distances:
  d_pm spans ~9000× across the training set (tiny Roche-limit separations vs.
  wide Hill-radius orbits).  A global StandardScaler on raw distances would be
  dominated by large-separation outliers; close-in cells become invisible.
  Using d_pm / rhill_AU (per-simulation) collapses this to [0, 1] by construction.
  Similarly d_sp and d_sm / ap_AU collapse to O(1), matching orbital phase variation.
  No StandardScaler needed on distances — physics provides the correct scale.

Why dimensionless force targets (t_grad) instead of raw forces:
  Raw forces span ~10^9 across the training set.  F_ref = G·ms·mp/ap² collapses
  system-scale spread to O(1).  Each target is the pair force magnitude / F_ref:
    t_sp = G·ms·mp / (d_sp² × F_ref) = (ap/d_sp)²    — ~O(1) at periapsis
    t_sm = G·ms·mm / (d_sm² × F_ref) = (mm/mp)·(ap/d_sm)²
    t_pm = G·mp·mm / (d_pm² × F_ref) × (rhill/ap)    — see note below

  t_pm carries a correction factor of rhill/ap relative to the raw force/F_ref
  formula.  d_sp and d_sm are normalised by ap (matching the ap² in F_ref), so
  their force/F_ref targets are self-consistent with the chain rule at inference.
  d_pm is normalised by rhill instead; the chain rule at inference introduces a
  factor of 1/rhill rather than 1/ap, so the target must be scaled by rhill/ap:
    t_pm = (mm/ms)·(ap/rhill) / d_pm_n²
  Using (mm/ms)·(ap/d_pm)² here would give moon-planet forces wrong by ap/rhill
  (~10–100×) at inference.  t_pm is clipped at T_MAX=1000 near the Roche limit.

ISOLATION: zero imports from dataset.py, model.py, train.py, inference.py.
Never writes to models/, models_temphead/, eval_aux_mlp_output/.
"""

import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

_G = 4.0 * np.pi ** 2                      # G in (AU, yr, M☉) units
_MERTH_OVER_MSUN = 5.976e24 / 1.989e30     # ≈ 3.005e-6

_SYS_COLS = ["ms_solar", "mp_earth", "mm_earth", "ap_AU", "rhill_AU"]
_Q_COLS   = [
    "star_x",   "star_y",   "star_z",
    "planet_x", "planet_y", "planet_z",
    "moon_x",   "moon_y",   "moon_z",
]

D_DIM   = 3
SYS_DIM = len(_SYS_COLS)   # = 5: ms_solar, mp_earth, mm_earth, ap_AU, rhill_AU

T_MAX = 1000.0   # clip ceiling for t_pm near Roche limit


class HnnDataset(Dataset):
    """Per-timestep HNN training dataset."""

    def __init__(
        self,
        d_n:     np.ndarray,   # (N, 3) physics-normalised distances
        sys_enc: np.ndarray,   # (N, 3) StandardScaler-normalised masses
        t_grad:  np.ndarray,   # (N, 3) dimensionless force targets
        stable:  np.ndarray,   # (N,)   1.0 where d_pm_n <= 1 (moon within Hill sphere)
    ) -> None:
        self.d_n     = torch.from_numpy(d_n.astype(np.float32))
        self.sys_enc = torch.from_numpy(sys_enc.astype(np.float32))
        self.t_grad  = torch.from_numpy(t_grad.astype(np.float32))
        self.stable  = torch.from_numpy(stable.astype(np.float32))

    def __len__(self) -> int:
        return len(self.d_n)

    def __getitem__(self, idx: int):
        return self.d_n[idx], self.sys_enc[idx], self.t_grad[idx], self.stable[idx]


def _load_parquet(path: str) -> tuple:
    """
    Load parquet and compute physics-normalised distances + dimensionless force targets.

    Training filter applied here:
    1. Exclude simulations whose first timestep is not both stable AND habitable.
    2. Per simulation, exclude all rows from the first timestep onward where
       stable=0 AND habitable=0 simultaneously (trajectory preview stopping point).
    This eliminates deeply-escaped moon rows (d_pm_n >> 1, t_pm ~ 1e-17) that
    caused catastrophic MSRE loss in v1-v4.

    Returns (sim_ids, d_n, sys_raw, t_grads) as numpy arrays.
    """
    cols = ["sim_id"] + _Q_COLS + _SYS_COLS + ["stable", "habitable"]
    df   = pd.read_parquet(path, columns=cols)
    n_raw = len(df)

    # 1. Keep only simulations that start stable+habitable
    sim_first  = df.groupby("sim_id")[["stable", "habitable"]].first()
    valid_sims = sim_first[
        (sim_first["stable"] == 1) & (sim_first["habitable"] == 1)
    ].index
    n_sims_excluded = df["sim_id"].nunique() - len(valid_sims)
    df = df[df["sim_id"].isin(valid_sims)].copy()

    # 2. Per simulation, trim rows after the first step where stable=0 AND habitable=0.
    #    cumsum of the bad flag: once it becomes 1, all subsequent rows are >= 1.
    #    Keep only rows where cumsum == 0 (no bad step seen yet).
    df["_bad"] = ((df["stable"] == 0) & (df["habitable"] == 0)).astype(np.int8)
    df["_cum"] = df.groupby("sim_id")["_bad"].cumsum()
    df = df[df["_cum"] == 0].drop(columns=["_bad", "_cum", "stable", "habitable"])

    n_kept = len(df)
    print(f"  Loaded {n_raw:,} total rows; {n_sims_excluded} sims excluded (non-stable+habitable start)")
    print(f"  After stable+habitable trajectory filter: {n_kept:,} rows kept "
          f"({100*n_kept/n_raw:.1f}%)")

    ms_solar  = df["ms_solar"].to_numpy(dtype=np.float64)
    mp_earth  = df["mp_earth"].to_numpy(dtype=np.float64)
    mm_earth  = df["mm_earth"].to_numpy(dtype=np.float64)
    ap_AU     = df["ap_AU"].to_numpy(dtype=np.float64)
    rhill_AU  = df["rhill_AU"].to_numpy(dtype=np.float64)

    # Convert to solar masses for G=4π² unit system
    ms = ms_solar
    mp = mp_earth * _MERTH_OVER_MSUN
    mm = mm_earth * _MERTH_OVER_MSUN

    q = df[_Q_COLS].to_numpy(dtype=np.float64)   # (N, 9)  AU
    r_s = q[:, 0:3];  r_p = q[:, 3:6];  r_m = q[:, 6:9]

    # Pair distances (AU)
    d_sp = np.linalg.norm(r_p - r_s, axis=1)   # (N,)
    d_sm = np.linalg.norm(r_m - r_s, axis=1)   # (N,)
    d_pm = np.linalg.norm(r_m - r_p, axis=1)   # (N,)

    # Physics-based normalisation — no StandardScaler
    d_sp_n = d_sp / ap_AU       # O(1) by construction
    d_sm_n = d_sm / ap_AU
    d_pm_n = d_pm / rhill_AU    # in [0, ~1] for stable moons; bounded for escaped-but-habitable

    n = len(d_sp_n)
    print(f"  d_sp_n: {d_sp_n.min():.4f}–{d_sp_n.max():.4f}  "
          f"d_sm_n: {d_sm_n.min():.4f}–{d_sm_n.max():.4f}  "
          f"d_pm_n: {d_pm_n.min():.4f}–{d_pm_n.max():.4f}")

    # Dimensionless force targets: pair force magnitude / F_ref (F_ref = G·ms·mp/ap²).
    # t_sp and t_sm: force/F_ref is self-consistent because d_sp, d_sm use ap normalisation.
    # t_pm: force/F_ref × (rhill/ap) correction because d_pm uses rhill normalisation —
    #   the chain rule at inference introduces 1/rhill (not 1/ap) for the d_pm derivative.
    t_sp = 1.0 / d_sp_n ** 2                                  # (N,)  = (ap/d_sp)²
    t_sm = (mm / mp) / d_sm_n ** 2                            # (N,)  = (mm/mp)·(ap/d_sm)²
    t_pm = (mm / ms) * (ap_AU / rhill_AU) / d_pm_n ** 2      # (N,)  corrected for rhill normalisation

    t_grads = np.clip(
        np.stack([t_sp, t_sm, t_pm], axis=1),
        0.0, T_MAX,
    )   # (N, 3)

    print(f"  t_sp: {t_sp.mean():.3f} ± {t_sp.std():.3f}  "
          f"t_sm (pre-clip): {t_sm.mean():.4f}  "
          f"t_pm (pre-clip): {t_pm.mean():.4f}  "
          f"t_pm clipped fraction: {(t_pm > T_MAX).mean():.4f}")

    stable  = (d_pm_n <= 1.0).astype(np.float32)             # (N,) 1 where moon within Hill sphere
    d_n     = np.stack([d_sp_n, d_sm_n, d_pm_n], axis=1)   # (N, 3)
    sys_raw = df[_SYS_COLS].to_numpy(dtype=np.float64)       # (N, 5)
    sim_ids = df["sim_id"].to_numpy()

    bound_frac = stable.mean()
    print(f"  Bound (stable=1) rows: {100*bound_frac:.1f}%  "
          f"({int(stable.sum()):,} / {len(stable):,})  (t_pm log-MSE applied here only)")

    return sim_ids, d_n, sys_raw, t_grads, stable


def make_hnn_splits(
    path:         str,
    val_frac:     float = 0.20,
    seed:         int   = 42,
    log_sys_enc:  bool  = False,
) -> tuple:
    """
    Load parquet and split by sim_id into train/val HnnDatasets.

    Splitting by sim_id prevents timestep-level leakage between train and val.

    log_sys_enc=True: fit StandardScaler on log(sys_raw) instead of raw sys_raw.
        Required when training HNN with log_inputs=True.  In that regime the
        t_pm target decomposes as a LINEAR function of log-space inputs:
            ln(t_pm) = [ln(mm)−ln(ms)] + [ln(ap)−ln(rhill)] − 2·ln(d_pm_n)
        All sys_raw columns are strictly positive, so log is always well-defined.
        The same transform must be applied at inference:
            hnn_scaler.transform(np.log(sys_raw))  (not sys_raw).

    Returns
    -------
    train_ds   : HnnDataset
    val_ds     : HnnDataset
    sys_scaler : StandardScaler fitted on training sys_raw (or log(sys_raw))
    """
    sim_ids, d_n, sys_raw, t_grads, stable = _load_parquet(path)

    unique_sims = np.unique(sim_ids)
    rng         = np.random.default_rng(seed)
    rng.shuffle(unique_sims)

    n_val      = max(1, int(len(unique_sims) * val_frac))
    val_set    = set(unique_sims[:n_val].tolist())
    train_mask = np.array([s not in val_set for s in sim_ids])
    val_mask   = ~train_mask

    print(f"  Train rows: {train_mask.sum():,}  |  Val rows: {val_mask.sum():,}  (split by sim_id)")
    if log_sys_enc:
        print("  sys_enc: StandardScaler fitted on log(sys_raw)  [log_inputs mode]")

    sys_fit    = np.log(sys_raw[train_mask]) if log_sys_enc else sys_raw[train_mask]
    sys_scaler = StandardScaler().fit(sys_fit)

    def _make(mask: np.ndarray) -> HnnDataset:
        sys_input = np.log(sys_raw[mask]) if log_sys_enc else sys_raw[mask]
        return HnnDataset(
            d_n=d_n[mask],
            sys_enc=sys_scaler.transform(sys_input),
            t_grad=t_grads[mask],
            stable=stable[mask],
        )

    return _make(train_mask), _make(val_mask), sys_scaler


def make_hnn_splits_tidal(
    path:     str,
    val_frac: float = 0.20,
    seed:     int   = 42,
) -> tuple:
    """
    Like make_hnn_splits but uses the tidal coordinate for the second input slot.

    The second distance input changes from d_sm_n = d_sm/ap to
        δ_tidal_n = (d_sm - d_sp) / rhill_AU
    This is a signed quantity (negative when moon is on near side of planet relative
    to star) and avoids catastrophic cancellation in the tidal force computation.

    Because δ_tidal_n replaces d_sm_n in the input, the gradient targets change:
        slot 0: t_sp_new = 1/d_sp_n² + (mm/mp)/d_sm_n²
                (∂V/∂d_sp_n at constant δ_tidal_n — includes d_sm coupling)
        slot 1: t_tidal  = (mm/mp)/d_sm_n² × (rhill/ap)
                (∂V/∂δ_tidal_n — the small tidal perturbation, ~t_pm magnitude)
        slot 2: t_pm     = (mm/ms)·(ap/rhill)/d_pm_n² (unchanged)

    At inference the autograd chain q→(d_sp,d_sm,d_pm)→(d_sp_n,δ_tidal_n,d_pm_n)→V_θ
    correctly cancels the t_sp_new/t_tidal coupling (Jacobian identity), recovering
    the physical forces without catastrophic cancellation in the tidal term.

    Always uses log_sys_enc=True (required for log-linear decomposition of t_tidal).
    No Parquet regeneration needed — δ_tidal is computed on-the-fly from existing
    star/planet/moon x,y,z coordinate columns.

    Returns
    -------
    train_ds   : HnnDataset
    val_ds     : HnnDataset
    sys_scaler : StandardScaler fitted on log(sys_raw) of the training split
    """
    sim_ids, d_n_orig, sys_raw, t_grads_orig, stable = _load_parquet(path)

    # Reconstruct pair distances in AU from the position columns already loaded
    # into d_n_orig.  We need d_sp, d_sm in AU (not normalised) to compute δ_tidal.
    # d_sp_n = d_sp/ap  →  d_sp = d_sp_n × ap_AU
    # d_sm_n = d_sm/ap  →  d_sm = d_sm_n × ap_AU
    ap_AU    = sys_raw[:, 3]   # (N,)
    rhill_AU = sys_raw[:, 4]   # (N,)
    mp_earth = sys_raw[:, 1]
    mm_earth = sys_raw[:, 2]

    d_sp_n = d_n_orig[:, 0]   # (N,)
    d_sm_n = d_n_orig[:, 1]   # (N,) — original, needed for t targets
    d_pm_n = d_n_orig[:, 2]   # (N,)

    # δ_tidal_n: signed tidal displacement normalised by Hill radius
    # d_sm = d_sm_n × ap, d_sp = d_sp_n × ap  → δ_tidal = (d_sm - d_sp)
    # δ_tidal_n = (d_sm - d_sp) / rhill = (d_sm_n - d_sp_n) × (ap/rhill)
    delta_tidal_n = (d_sm_n - d_sp_n) * (ap_AU / rhill_AU)   # (N,) signed, |.| ≤ am_hill

    # New gradient targets under the tidal coordinate change of variables:
    #   ∂V_Newton_n/∂d_sp_n (at const δ_tidal_n) = 1/d_sp_n² + (mm/mp)/d_sm_n²
    #   ∂V_Newton_n/∂δ_tidal_n = (mm/mp)/d_sm_n² × (rhill/ap)
    mm_over_mp = mm_earth / mp_earth                          # (N,)
    t_sp_new   = 1.0 / d_sp_n**2 + mm_over_mp / d_sm_n**2   # (N,)
    t_tidal    = mm_over_mp / d_sm_n**2 * (rhill_AU / ap_AU) # (N,) small, ~t_pm magnitude
    t_pm       = t_grads_orig[:, 2]                           # (N,) unchanged

    d_n_tidal = np.stack([d_sp_n, delta_tidal_n, d_pm_n], axis=1)   # (N, 3)
    t_grads_tidal = np.clip(
        np.stack([t_sp_new, t_tidal, t_pm], axis=1),
        0.0, T_MAX,
    )   # (N, 3)

    print(f"\n  Tidal coord summary:")
    print(f"  delta_tidal_n: {delta_tidal_n.min():.4f} to {delta_tidal_n.max():.4f}  "
          f"(signed; negative = near-side moon)")
    print(f"  t_sp_new: {t_sp_new.mean():.4f} +/- {t_sp_new.std():.4f}  "
          f"(vs t_sp_old mean {(1/d_sp_n**2).mean():.4f})")
    print(f"  t_tidal:  {t_tidal.mean():.6f} +/- {t_tidal.std():.6f}  "
          f"(t_pm mean for comparison: {t_pm.mean():.6f})")

    # Split by sim_id (same as standard splits — no timestep leakage)
    unique_sims = np.unique(sim_ids)
    rng         = np.random.default_rng(seed)
    rng.shuffle(unique_sims)

    n_val      = max(1, int(len(unique_sims) * val_frac))
    val_set    = set(unique_sims[:n_val].tolist())
    train_mask = np.array([s not in val_set for s in sim_ids])
    val_mask   = ~train_mask

    print(f"  Train rows: {train_mask.sum():,}  |  Val rows: {val_mask.sum():,}  (split by sim_id)")
    print("  sys_enc: StandardScaler fitted on log(sys_raw)  [tidal_coords requires log sys]")

    sys_fit    = np.log(sys_raw[train_mask])   # always log for tidal_coords
    sys_scaler = StandardScaler().fit(sys_fit)

    def _make(mask: np.ndarray) -> HnnDataset:
        return HnnDataset(
            d_n=d_n_tidal[mask],
            sys_enc=sys_scaler.transform(np.log(sys_raw[mask])),
            t_grad=t_grads_tidal[mask],
            stable=stable[mask],
        )

    return _make(train_mask), _make(val_mask), sys_scaler


def _compute_v_newton_n(d_n: np.ndarray, sys_raw: np.ndarray) -> np.ndarray:
    """
    Analytical Newtonian gravitational potential in V_ref-normalised units.

    V_ref = 4π²·ms·mp / ap_AU  (same reference used for force targets t_grad).

    V_Newton_n = −( 1/d_sp_n  +  (mm/mp)/d_sm_n  +  (mm/ms)·(ap/rhill)/d_pm_n )

    Derivation:
        V_Newton = −4π²·(ms·mp/d_sp + ms·mm/d_sm + mp·mm/d_pm)
        V_Newton_n = V_Newton / V_ref
                   = −[ap/d_sp + (mm/mp)·ap/d_sm + (mm/ms)·ap/d_pm]
    Substituting d_sp = d_sp_n·ap, d_sm = d_sm_n·ap, d_pm = d_pm_n·rhill:
                   = −[1/d_sp_n + (mm/mp)/d_sm_n + (mm/ms)·(ap/rhill)/d_pm_n]

    Note: mm/mp = mm_earth/mp_earth (MERTH_OVER_MSUN cancels).
          mm/ms = mm_earth·MERTH_OVER_MSUN / ms_solar.

    Clipped to [−1000, 0] — gravitational potential is always negative; the cap
    handles any near-Roche rows (d_pm_n → 0) that survive the training filter.
    """
    d_sp_n    = d_n[:, 0]
    d_sm_n    = d_n[:, 1]
    d_pm_n    = d_n[:, 2]
    ms_solar  = sys_raw[:, 0]
    mp_earth  = sys_raw[:, 1]
    mm_earth  = sys_raw[:, 2]
    ap_AU     = sys_raw[:, 3]
    rhill_AU  = sys_raw[:, 4]

    mm_over_mp    = mm_earth / mp_earth
    mm_over_ms    = mm_earth * _MERTH_OVER_MSUN / ms_solar
    ap_over_rhill = ap_AU / rhill_AU

    v = -(1.0 / d_sp_n
          + mm_over_mp / d_sm_n
          + mm_over_ms * ap_over_rhill / d_pm_n)
    return np.clip(v, -1000.0, 0.0).astype(np.float64)


class HnnDatasetFull(Dataset):
    """Per-timestep HNN dataset that also carries V_Newton_n for Option C value supervision."""

    def __init__(
        self,
        d_n:        np.ndarray,
        sys_enc:    np.ndarray,
        t_grad:     np.ndarray,
        stable:     np.ndarray,   # (N,) 1.0 where d_pm_n <= 1
        v_newton_n: np.ndarray,
    ) -> None:
        self.d_n        = torch.from_numpy(d_n.astype(np.float32))
        self.sys_enc    = torch.from_numpy(sys_enc.astype(np.float32))
        self.t_grad     = torch.from_numpy(t_grad.astype(np.float32))
        self.stable     = torch.from_numpy(stable.astype(np.float32))
        self.v_newton_n = torch.from_numpy(v_newton_n.astype(np.float32))

    def __len__(self) -> int:
        return len(self.d_n)

    def __getitem__(self, idx: int):
        return self.d_n[idx], self.sys_enc[idx], self.t_grad[idx], self.stable[idx], self.v_newton_n[idx]


def make_hnn_splits_optionC(
    path:        str,
    val_frac:    float = 0.20,
    seed:        int   = 42,
    log_sys_enc: bool  = False,
) -> tuple:
    """
    Like make_hnn_splits but returns HnnDatasetFull instances (4-element batches)
    that include V_Newton_n for Option C value supervision.

    log_sys_enc: if True, StandardScaler is fitted on log(sys_raw).
        Must match the log_inputs flag passed to HNN (see make_hnn_splits docstring).

    Returns
    -------
    train_ds   : HnnDatasetFull
    val_ds     : HnnDatasetFull
    sys_scaler : StandardScaler fitted on training sys_raw (or log(sys_raw))
    """
    sim_ids, d_n, sys_raw, t_grads, stable = _load_parquet(path)
    v_newton_n = _compute_v_newton_n(d_n, sys_raw)

    unique_sims = np.unique(sim_ids)
    rng         = np.random.default_rng(seed)
    rng.shuffle(unique_sims)

    n_val      = max(1, int(len(unique_sims) * val_frac))
    val_set    = set(unique_sims[:n_val].tolist())
    train_mask = np.array([s not in val_set for s in sim_ids])
    val_mask   = ~train_mask

    print(f"  Train rows: {train_mask.sum():,}  |  Val rows: {val_mask.sum():,}  (split by sim_id)")
    print(f"  v_newton_n: {v_newton_n.min():.3f}–{v_newton_n.max():.3f}  "
          f"mean={v_newton_n.mean():.3f}")
    if log_sys_enc:
        print("  sys_enc: StandardScaler fitted on log(sys_raw)  [log_inputs mode]")

    sys_fit    = np.log(sys_raw[train_mask]) if log_sys_enc else sys_raw[train_mask]
    sys_scaler = StandardScaler().fit(sys_fit)

    def _make(mask: np.ndarray) -> HnnDatasetFull:
        sys_input = np.log(sys_raw[mask]) if log_sys_enc else sys_raw[mask]
        return HnnDatasetFull(
            d_n=d_n[mask],
            sys_enc=sys_scaler.transform(sys_input),
            t_grad=t_grads[mask],
            stable=stable[mask],
            v_newton_n=v_newton_n[mask],
        )

    return _make(train_mask), _make(val_mask), sys_scaler


# ── Serialisation helpers ───────────────────────────────────────────────────────

def save_sys_scaler(scaler: StandardScaler, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "hnn_sys_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)


def load_sys_scaler(model_dir: str) -> StandardScaler:
    path = os.path.join(model_dir, "hnn_sys_scaler.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)
