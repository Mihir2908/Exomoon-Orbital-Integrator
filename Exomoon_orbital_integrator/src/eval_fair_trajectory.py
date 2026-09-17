"""
eval_fair_trajectory.py — Fair trajectory comparison: per-cell adaptive dt, no dt_factor.

Compares HNN v12 vs Force MLP vs Ground-truth Numba leapfrog at 9 representative
(mm_idx, am_idx) cells from each system in ground_truth_grids/.

ALL integrators use the same per-cell timestep:
    dt = min(T_moon_cell / 100,  1 / 20_000)
where T_moon_cell is computed from that specific cell's am_AU and (mp + mm).

This matches exactly what run_simulation_for_years() uses in simulation.py
(dt_moon = orbprd_mm_mp / 100, dt_year = 1/20000, dt = min of the two).

No dt_factor anywhere — each cell uses its own physically correct timestep.

ISOLATION: does not touch models/, models_temphead/, eval_aux_mlp_output/.
"""

import os, sys, json, time, argparse
import numpy as np
import torch

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from exomoon.constants          import FOUR_PI2, rsun, merth, msun, au as AU_m
from exomoon.habitable_zone     import hz_bounds_au
from exomoon.params             import SystemParams
from exomoon.initial_conditions import initial_state
from exomoon.simulation         import run_simulation_for_years
from exomoon.ml.hnn_model             import HNN
from exomoon.ml.hnn_greydanus_model   import HNNGreydanus, is_greydanus_dir
from exomoon.ml.hnn_dataset           import load_sys_scaler, _MERTH_OVER_MSUN
from exomoon.ml.force_mlp_inference   import preview_trajectory
from exomoon.ml.hnn_model_hill        import HNN_Hill, is_hill_dir
from exomoon.ml.hnn_dataset_hill      import load_hill_sys_scaler, compute_hill_sys_enc

_STEPS_PER_YEAR = 20_000.0   # same constant as simulation.py
_FOUR_PI2       = 4.0 * np.pi ** 2


# ── CLI ───────────────────────────────────────────────────────────────────────

ap = argparse.ArgumentParser()
ap.add_argument("--hnn_dir",   default="models_hnn_dist_v12")
ap.add_argument("--fmlp_dir",  default="models_force_mlp")
ap.add_argument("--systems",   default=None,
                help="Substring filters, comma-separated. Default: all prograde.")
ap.add_argument("--n_out",     type=int, default=50,
                help="Output time-points for trajectory comparison (default 50)")
ap.add_argument("--max_steps", type=int, default=200_000,
                help="Hard cap on physics steps per HNN cell (default 200k)")
ap.add_argument("--parquet_dt", action="store_true",
                help="Use Parquet resampled dt (t_sim/999 steps) for HNN integration "
                     "instead of physics dt. Use this when evaluating Option D models "
                     "trained on the 1000-step Parquet grid.")
ap.add_argument("--newton", action="store_true",
                help="Run Newton sanity check: replace V_theta with analytical V_Newton_n "
                     "through the identical autograd→V_ref→leapfrog chain, and print "
                     "per-component force predictions vs analytical at t=0. "
                     "If Newton trajectories match GT, the inference chain is confirmed correct.")
ap.add_argument("--hill_coords", action="store_true",
                help="Use Hill-frame HNN (HNN_Hill from models_hnn_hill/) instead of "
                     "the inertial-frame Track 1 HNN (models_hnn_log/).  "
                     "The Track 1 model is untouched — this is a completely separate "
                     "inference path.  Requires a trained HNN_Hill checkpoint.")
ap.add_argument("--hill_dir", default="models_hnn_hill",
                help="Directory containing HNN_Hill checkpoint (default: models_hnn_hill)")
ap.add_argument("--include_retro", action="store_true",
                help="Also evaluate retrograde meta.json files (default: prograde only)")
ap.add_argument("--small_rhill_thresh", type=float, default=0.002,
                help="Systems with rhill_AU below this threshold also test denser "
                     "low-am_idx cells (default: 0.002 AU). Addresses TRAPPIST-1e "
                     "where standard am_idx=12/25/37 cells are all GT-escaped.")
args = ap.parse_args()

GT_DIR   = os.path.join(SRC, "ground_truth_grids")
HNN_DIR  = os.path.join(SRC, args.hnn_dir)
FMLP_DIR = os.path.join(SRC, args.fmlp_dir)

MM_RES = AM_RES = 50
CELL_INDICES = [
    (12,  3), (12,  6),
    (12, 12), (12, 25), (12, 37),
    (25, 12), (25, 25), (25, 37),
    (37, 12), (37, 25), (37, 37),
]
N_PRINT = 10


# ── Load HNN model (v12 or Greydanus, detected from hnn_config.json) ─────────

print(f"HNN  dir : {args.hnn_dir}")
print(f"FMLP dir : {args.fmlp_dir}")
if args.hill_coords:
    print(f"Hill dir : {args.hill_dir}  [--hill_coords active]")

_HNN_IS_GREYDANUS = is_greydanus_dir(HNN_DIR)
if _HNN_IS_GREYDANUS:
    hnn_model = HNNGreydanus.load(HNN_DIR)
    print("HNN Greydanus (Option E) loaded.")
else:
    hnn_model = HNN.load(HNN_DIR)
    print("HNN v12 loaded.")
hnn_model.eval()
hnn_scaler = load_sys_scaler(HNN_DIR)
print()

# ── Hill-frame HNN (loaded only when --hill_coords is active) ─────────────────

HILL_DIR        = os.path.join(SRC, args.hill_dir)
_hill_hnn_model  = None
_hill_hnn_scaler = None

if args.hill_coords:
    if not is_hill_dir(HILL_DIR):
        raise FileNotFoundError(
            f"No HNN_Hill checkpoint found at '{HILL_DIR}'. "
            "Train first:  py hnn_train_hill.py --data ml_dataset.parquet"
        )
    _hill_hnn_model  = HNN_Hill.load(HILL_DIR)
    _hill_hnn_model.eval()
    _hill_hnn_scaler = load_hill_sys_scaler(HILL_DIR)
    print(f"HNN_Hill loaded from {HILL_DIR}")


# ── HNN force computation via autograd ────────────────────────────────────────

def _hnn_accel(
    r_s: np.ndarray, r_p: np.ndarray, r_m: np.ndarray,
    ap_f32: torch.Tensor, rhill_f32: torch.Tensor,
    ms_solar: float, mp_msun: float, mm_msun: float,
    V_ref: float,
    sys_enc_t: torch.Tensor,
) -> tuple:
    """KDK force step: absolute positions → accelerations (AU/yr²).

    Chain rule: V(d_n) → dV/dq_abs → F = -dV/dq × V_ref → a = F/mass.
    V_ref = G*ms*mp/ap  [Msun*AU²/yr²].
    """
    q = torch.tensor(
        np.concatenate([r_s, r_p, r_m]).astype(np.float32),
        dtype=torch.float32,
    ).unsqueeze(0)                    # (1, 9)
    q_leaf = q.detach().requires_grad_(True)

    with torch.enable_grad():
        d_sp = (q_leaf[:, 3:6] - q_leaf[:, 0:3]).norm(dim=1, keepdim=True)
        d_sm = (q_leaf[:, 6:9] - q_leaf[:, 0:3]).norm(dim=1, keepdim=True)
        d_pm = (q_leaf[:, 6:9] - q_leaf[:, 3:6]).norm(dim=1, keepdim=True)
        if hnn_model.tidal_coords:
            # Option 2: slot 1 = δ_tidal_n = (d_sm - d_sp) / rhill (signed)
            d_n = torch.cat([d_sp / ap_f32, (d_sm - d_sp) / rhill_f32, d_pm / rhill_f32], dim=1)
        else:
            d_n = torch.cat([d_sp / ap_f32, d_sm / ap_f32, d_pm / rhill_f32], dim=1)
        V    = hnn_model(d_n, sys_enc_t)
        dV   = torch.autograd.grad(V.sum(), q_leaf, create_graph=False)[0]

    # F = -dV/dq * V_ref  [Msun*AU/yr²]
    F_raw = (-dV.detach().numpy()[0]).astype(np.float64) * V_ref
    a_s   = F_raw[0:3] / ms_solar   # AU/yr²
    a_p   = F_raw[3:6] / mp_msun
    a_m   = F_raw[6:9] / mm_msun
    return a_s, a_p, a_m


def _hnn_cell_trajectory(
    st:          dict,          # initial_state() output
    p_cell:      SystemParams,  # same SystemParams used to build st
    t_sim:       float,
    n_out:       int,
    max_steps:   int,
    parquet_dt:  bool = False,  # use Parquet resampled dt instead of physics dt
) -> dict:
    """KDK leapfrog with HNN forces, per-cell adaptive dt, no dt_factor.

    Default: dt = min(T_moon_cell / 100,  1/20_000)  — identical to simulation.py.
    With parquet_dt=True: dt = t_sim / 999  (matches 1000-step Parquet resampling
    used during Option D training).
    """
    rhill_AU = float(st["rhill_AU"])
    am_AU    = float(st["am_AU"])
    ap_AU    = float(p_cell.ap_AU)

    # Raw masses in solar-mass units (for force conversion and sys_enc)
    ms_solar = float(p_cell.ms_solar)
    mp_msun  = float(p_cell.mp_earth * _MERTH_OVER_MSUN)
    mm_msun  = float(p_cell.mm_earth * _MERTH_OVER_MSUN)

    # Per-cell T_moon and dt
    G_pm   = FOUR_PI2 * (mp_msun + mm_msun)    # AU³/yr²
    T_moon = 2.0 * np.pi * np.sqrt(am_AU ** 3 / G_pm)

    if parquet_dt:
        # Match the 1000-step Parquet grid used during Option D training
        n_phys = 999
        dt     = t_sim / n_phys
    else:
        # Physics dt — matches simulation.py exactly
        dt_raw = min(T_moon / 100.0, 1.0 / _STEPS_PER_YEAR)
        n_phys = min(int(np.ceil(t_sim / dt_raw)), max_steps)
        dt     = t_sim / n_phys     # snap to hit t_sim exactly

    stride  = max(1, n_phys // n_out)

    # Constant tensors for force computation
    ap_f32    = torch.tensor(ap_AU,    dtype=torch.float32)
    rhill_f32 = torch.tensor(rhill_AU, dtype=torch.float32)

    sys_raw    = np.array([[ms_solar, p_cell.mp_earth, p_cell.mm_earth, ap_AU, rhill_AU]])
    sys_fit    = np.log(sys_raw) if hnn_model.use_log_sys_enc else sys_raw
    sys_enc_t  = torch.tensor(
        hnn_scaler.transform(sys_fit).astype(np.float32)
    )   # (1, 5)

    # V_ref = G*ms*mp/ap  [Msun*AU²/yr²]  — same as batch_hnn_trajectories
    V_ref = FOUR_PI2 * ms_solar * mp_msun / ap_AU

    # Initial positions (float64 for integration accuracy)
    r_s = st["pos_ms"].copy().astype(np.float64)
    r_p = st["pos_mp"].copy().astype(np.float64)
    r_m = st["pos_mm"].copy().astype(np.float64)

    # Calibrate moon's initial velocity to circular orbit under H_θ (not H_Newton).
    # Separable v12: force is position-only → single iteration suffices.
    v_s, v_p, v_m = _v12_calibrate_init_vel(
        st, ap_f32, rhill_f32, ms_solar, mp_msun, mm_msun, V_ref, sys_enc_t,
    )

    # Output buffers
    cap    = n_phys // stride + 2
    t_arr  = np.zeros(cap)
    mpd    = np.zeros(cap)
    msd    = np.zeros(cap)
    t_arr[0] = 0.0
    mpd[0]   = np.linalg.norm(r_m - r_p)
    msd[0]   = np.linalg.norm(r_m - r_s)
    out_idx  = 1

    # Initial forces
    a_s, a_p, a_m = _hnn_accel(
        r_s, r_p, r_m, ap_f32, rhill_f32,
        ms_solar, mp_msun, mm_msun, V_ref, sys_enc_t
    )
    half_dt = 0.5 * dt

    for step in range(1, n_phys + 1):
        # Half kick
        v_s += half_dt * a_s
        v_p += half_dt * a_p
        v_m += half_dt * a_m

        # Full drift
        r_s += dt * v_s
        r_p += dt * v_p
        r_m += dt * v_m

        # New forces
        a_s, a_p, a_m = _hnn_accel(
            r_s, r_p, r_m, ap_f32, rhill_f32,
            ms_solar, mp_msun, mm_msun, V_ref, sys_enc_t
        )

        # Half kick
        v_s += half_dt * a_s
        v_p += half_dt * a_p
        v_m += half_dt * a_m

        if step % stride == 0 and out_idx < cap:
            t_arr[out_idx] = step * dt
            mpd[out_idx]   = np.linalg.norm(r_m - r_p)
            msd[out_idx]   = np.linalg.norm(r_m - r_s)
            out_idx += 1

    return {
        "ok":               True,
        "t_arr":            t_arr[:out_idx],
        "moon_planet_dist": mpd[:out_idx],
        "moon_star_dist":   msd[:out_idx],
        "dt":               dt,
        "T_moon":           T_moon,
        "n_phys":           n_phys,
    }


# ── Hill-frame HNN inference ──────────────────────────────────────────────────

def _compute_hill_frame_instant(r_s, r_p, r_m, v_s, v_p, v_m, rhill_AU):
    """
    Compute Hill-frame quantities from current inertial positions/velocities.

    Returns
    -------
    x_hat, y_hat, z_hat : (3,) unit vectors of Hill frame in inertial coords
    q_h_norm            : (3,) normalised moon position in Hill frame
    q_h_AU              : (3,) physical moon position in Hill frame [AU]
    Omega               : float  instantaneous angular velocity [rad/yr]
    R                   : (3,3) rotation matrix — R @ v_hill = v_inertial
    """
    r_sp  = r_p - r_s                               # star→planet [AU]
    r_sp_sq = float(np.dot(r_sp, r_sp))
    r_sp_norm = float(np.sqrt(r_sp_sq))
    x_hat = r_sp / max(r_sp_norm, 1e-10)

    v_sp  = v_p - v_s
    L_sp  = np.cross(r_sp, v_sp)                    # angular momentum vector
    L_norm = float(np.linalg.norm(L_sp))
    z_hat = L_sp / max(L_norm, 1e-10)

    Omega = L_norm / max(r_sp_sq, 1e-20)            # rad/yr

    y_hat = np.cross(z_hat, x_hat)

    R = np.stack([x_hat, y_hat, z_hat], axis=1)     # (3,3); columns = Hill basis

    # Moon position in Hill frame
    r_mp  = r_m - r_p
    q_h_x = float(np.dot(r_mp, x_hat))
    q_h_y = float(np.dot(r_mp, y_hat))
    q_h_z = float(np.dot(r_mp, z_hat))
    q_h_AU   = np.array([q_h_x, q_h_y, q_h_z])
    q_h_norm = q_h_AU / max(rhill_AU, 1e-20)

    return x_hat, y_hat, z_hat, q_h_norm, q_h_AU, Omega, R


def _hnn_hill_accel(
    r_s:      np.ndarray, r_p: np.ndarray, r_m: np.ndarray,
    v_s:      np.ndarray, v_p: np.ndarray, v_m: np.ndarray,
    ms_solar: float, mp_msun: float, mm_msun: float,
    rhill_AU: float, ap_AU:   float, V_ref:   float,
) -> tuple:
    """
    Hill-frame HNN forces → inertial accelerations for star, planet, moon.

    Integration is in the INERTIAL frame, so pseudo-forces (Coriolis, centrifugal)
    must NOT appear in the returned accelerations.

    After training fix, HNN_Hill gradient gives:
        a_cons_h ≈ R^T @ (a_moon_Newton − a_planet_Newton) + Ω²×(q_h_x, q_h_y, 0)

    To recover a_moon_inertial:
        1. Strip centrifugal:  a_rel = R @ (a_cons_h − Ω²×q_h_AU)  ← pure Newton relative
        2. Add planet inertial: a_m = a_planet + a_rel              ← moon inertial
        NO Coriolis (pseudo-force, only appears when integrating in rotating frame).
    """
    G = FOUR_PI2

    # ── Hill-frame geometry ────────────────────────────────────────────────
    _, _, _, q_h_norm, q_h_AU, Omega, R = _compute_hill_frame_instant(
        r_s, r_p, r_m, v_s, v_p, v_m, rhill_AU
    )

    # sys_enc for HNN_Hill: [ms, mp_earth, mm_earth, ap, rhill, Omega] (all log-scaled)
    sys_raw = np.array([[ms_solar,
                         mp_msun / _MERTH_OVER_MSUN,
                         mm_msun / _MERTH_OVER_MSUN,
                         ap_AU, rhill_AU, Omega]])
    sys_enc = torch.tensor(
        _hill_hnn_scaler.transform(np.log(np.maximum(sys_raw, 1e-20))).astype(np.float32)
    )

    q_h_t    = torch.tensor(q_h_norm.astype(np.float32)).unsqueeze(0)   # (1,3)
    q_h_leaf = q_h_t.detach().requires_grad_(True)

    with torch.enable_grad():
        V_theta = _hill_hnn_model(q_h_leaf, sys_enc)
        grad    = torch.autograd.grad(
            V_theta.sum(), q_h_leaf, create_graph=False
        )[0]   # (1,3) = ∂V_θ/∂q_h_norm

    grad_np = grad.detach().numpy()[0].astype(np.float64)   # (3,)

    # Conservative Hill-frame acceleration (includes centrifugal from training targets):
    #   a_cons_h ≈ R^T@(a_moon − a_planet) + Ω²×(q_h_x, q_h_y, 0)
    a_cons_h = -grad_np * V_ref / (mm_msun * rhill_AU)      # (3,) [AU/yr²]

    # Strip centrifugal to get pure relative Newton accel in Hill frame
    centrifugal_h = Omega**2 * np.array([q_h_AU[0], q_h_AU[1], 0.0])   # [AU/yr²]
    a_rel_inertial = R @ (a_cons_h - centrifugal_h)         # (a_moon−a_planet) [AU/yr²]

    # ── Planet and star: exact Newton ──────────────────────────────────────
    r_sp_v = r_p - r_s                                       # star→planet [AU]
    d_sp   = float(np.linalg.norm(r_sp_v))
    r_sm   = r_m - r_s; d_sm = float(np.linalg.norm(r_sm))
    r_pm   = r_m - r_p; d_pm = float(np.linalg.norm(r_pm))

    # Planet: attracted toward star (moon pull on planet negligible, mm << mp)
    a_planet_inertial = (-G * ms_solar / d_sp**3) * r_sp_v  # [AU/yr²]

    # Moon inertial acceleration (no pseudo-forces)
    a_m_inertial = a_planet_inertial + a_rel_inertial

    # Star: attracted toward planet and toward moon
    a_s = (G * mp_msun / d_sp**3) * r_sp_v + (G * mm_msun / d_sm**3) * r_sm

    # Planet: attracted toward star (dominant); moon pull negligible
    a_p = a_planet_inertial

    return a_s, a_p, a_m_inertial


def _hill_calibrate_init_vel(
    st:       dict,
    ms_solar: float, mp_msun: float, mm_msun: float,
    rhill_AU: float, ap_AU:   float, V_ref:   float,
) -> tuple:
    """
    Adjust moon's initial velocity so it starts on a circular orbit under HNN_Hill.

    Uses the conservative acceleration from HNN_Hill at the initial position.
    The Newtonian circular orbit speed (from initial_state) is already a good
    approximation (~2% tidal correction), so one iteration suffices.
    """
    r_s = st["pos_ms"].astype(np.float64)
    r_p = st["pos_mp"].astype(np.float64)
    r_m = st["pos_mm"].astype(np.float64)
    v_s = st["vel_ms"].copy().astype(np.float64)
    v_p = st["vel_mp"].copy().astype(np.float64)
    v_m = st["vel_mm"].copy().astype(np.float64)

    # Conservative HNN_Hill acceleration at initial position
    _, _, _, q_h_norm, q_h_AU, Omega, R = _compute_hill_frame_instant(
        r_s, r_p, r_m, v_s, v_p, v_m, rhill_AU
    )
    ms_earth_f = ms_solar
    mp_earth_f = mp_msun / _MERTH_OVER_MSUN
    mm_earth_f = mm_msun / _MERTH_OVER_MSUN
    sys_raw = np.array([[ms_earth_f, mp_earth_f, mm_earth_f, ap_AU, rhill_AU, Omega]])
    sys_enc = torch.tensor(
        _hill_hnn_scaler.transform(np.log(np.maximum(sys_raw, 1e-20))).astype(np.float32)
    )
    q_h_t    = torch.tensor(q_h_norm.astype(np.float32)).unsqueeze(0)
    q_h_leaf = q_h_t.detach().requires_grad_(True)
    with torch.enable_grad():
        V_theta = _hill_hnn_model(q_h_leaf, sys_enc)
        grad    = torch.autograd.grad(V_theta.sum(), q_h_leaf, create_graph=False)[0]
    grad_np = grad.detach().numpy()[0].astype(np.float64)
    a_cons_h = -grad_np * V_ref / (mm_msun * rhill_AU)      # (3,) [AU/yr²]

    a_cons_h_mag = float(np.linalg.norm(a_cons_h))
    r_pm_vec = r_m - r_p
    r_pm     = float(np.linalg.norm(r_pm_vec))

    if a_cons_h_mag > 1e-30 and r_pm > 1e-30:
        v_circ_hill = np.sqrt(r_pm * a_cons_h_mag)
        v_rel_newton = v_m - v_p
        v_rel_mag    = float(np.linalg.norm(v_rel_newton))
        if v_rel_mag > 1e-30:
            v_m = v_p + v_circ_hill * (v_rel_newton / v_rel_mag)

    return v_s, v_p, v_m


def _hnn_hill_cell_trajectory(
    st:         dict,
    p_cell:     "SystemParams",
    t_sim:      float,
    n_out:      int,
    max_steps:  int,
    parquet_dt: bool = False,
) -> dict:
    """
    KDK leapfrog using Hill-frame HNN forces for the moon.
    Star and planet use exact Newton forces.
    Coriolis is added analytically at each force evaluation step.

    Isolation: only uses _hill_hnn_model and _hill_hnn_scaler (loaded from
    models_hnn_hill/).  Track 1 model (models_hnn_log/) is untouched.
    """
    rhill_AU = float(st["rhill_AU"])
    am_AU    = float(st["am_AU"])
    ap_AU    = float(p_cell.ap_AU)
    ms_solar = float(p_cell.ms_solar)
    mp_msun  = float(p_cell.mp_earth * _MERTH_OVER_MSUN)
    mm_msun  = float(p_cell.mm_earth * _MERTH_OVER_MSUN)

    G_pm   = FOUR_PI2 * (mp_msun + mm_msun)
    T_moon = 2.0 * np.pi * np.sqrt(am_AU ** 3 / G_pm)

    if parquet_dt:
        n_phys = 999
        dt     = t_sim / n_phys
    else:
        dt_raw = min(T_moon / 100.0, 1.0 / _STEPS_PER_YEAR)
        n_phys = min(int(np.ceil(t_sim / dt_raw)), max_steps)
        dt     = t_sim / n_phys

    stride = max(1, n_phys // n_out)

    V_ref = FOUR_PI2 * ms_solar * mp_msun / ap_AU

    r_s = st["pos_ms"].copy().astype(np.float64)
    r_p = st["pos_mp"].copy().astype(np.float64)
    r_m = st["pos_mm"].copy().astype(np.float64)

    # Calibrate initial moon velocity to circular orbit under HNN_Hill
    v_s, v_p, v_m = _hill_calibrate_init_vel(
        st, ms_solar, mp_msun, mm_msun, rhill_AU, ap_AU, V_ref
    )

    cap      = n_phys // stride + 2
    t_arr    = np.zeros(cap)
    mpd      = np.zeros(cap)
    msd      = np.zeros(cap)
    t_arr[0] = 0.0
    mpd[0]   = np.linalg.norm(r_m - r_p)
    msd[0]   = np.linalg.norm(r_m - r_s)
    out_idx  = 1

    a_s, a_p, a_m = _hnn_hill_accel(
        r_s, r_p, r_m, v_s, v_p, v_m,
        ms_solar, mp_msun, mm_msun, rhill_AU, ap_AU, V_ref,
    )
    half_dt = 0.5 * dt

    for step in range(1, n_phys + 1):
        # Half kick
        v_s += half_dt * a_s
        v_p += half_dt * a_p
        v_m += half_dt * a_m

        # Full drift
        r_s += dt * v_s
        r_p += dt * v_p
        r_m += dt * v_m

        # New forces (Coriolis uses velocities after half-kick, before second half-kick)
        a_s, a_p, a_m = _hnn_hill_accel(
            r_s, r_p, r_m, v_s, v_p, v_m,
            ms_solar, mp_msun, mm_msun, rhill_AU, ap_AU, V_ref,
        )

        # Half kick
        v_s += half_dt * a_s
        v_p += half_dt * a_p
        v_m += half_dt * a_m

        if step % stride == 0 and out_idx < cap:
            t_arr[out_idx] = step * dt
            mpd[out_idx]   = np.linalg.norm(r_m - r_p)
            msd[out_idx]   = np.linalg.norm(r_m - r_s)
            out_idx += 1

    return {
        "ok":               True,
        "t_arr":            t_arr[:out_idx],
        "moon_planet_dist": mpd[:out_idx],
        "moon_star_dist":   msd[:out_idx],
        "dt":               dt,
        "T_moon":           T_moon,
        "n_phys":           n_phys,
    }


# ── End Hill-frame HNN functions ──────────────────────────────────────────────


def _v12_calibrate_init_vel(
    st:        dict,
    ap_f32:    torch.Tensor,
    rhill_f32: torch.Tensor,
    ms_solar:  float,
    mp_msun:   float,
    mm_msun:   float,
    V_ref:     float,
    sys_enc_t: torch.Tensor,
) -> tuple:
    """Adjust moon's initial velocity for circular orbit under H_θ (separable v12).

    Uses only the planet-moon gradient (∂V_θ/∂d_pm_n) for the circular orbit
    speed — NOT the total force on the moon.  Using the total force is wrong:
    for systems where a_star_on_moon ≳ a_planet_on_moon (e.g. K-1229b at
    am=0.27 Hill, where a_star≈236 and a_planet≈163 AU/yr²), the calibrated
    v_rel exceeds the planet's escape velocity and the moon escapes from step 1.
    Only the PLANET-MOON component provides the centripetal force for the
    moon's orbit around the planet; the star's direct force is mostly cancelled
    by the corresponding force on the planet (only the ~1% tidal residual matters).
    """
    r_s = st["pos_ms"].astype(np.float64)
    r_p = st["pos_mp"].astype(np.float64)
    r_m = st["pos_mm"].astype(np.float64)
    v_s = st["vel_ms"].copy().astype(np.float64)
    v_p = st["vel_mp"].copy().astype(np.float64)
    v_m = st["vel_mm"].copy().astype(np.float64)

    r_pm_vec = r_m - r_p
    r_pm     = float(np.linalg.norm(r_pm_vec))

    # Isolate ∂V_θ/∂d_pm_n via a leaf on the d_n vector.
    rhill_AU = rhill_f32.item()
    ap_AU    = ap_f32.item()
    d_sp_n   = float(np.linalg.norm(r_p - r_s)) / ap_AU
    d_sm_n   = float(np.linalg.norm(r_m - r_s)) / ap_AU
    d_pm_n   = float(np.linalg.norm(r_m - r_p)) / rhill_AU

    if hnn_model.tidal_coords:
        # Option 2: slot 1 is δ_tidal_n (signed), not d_sm_n
        delta_tidal_n = (d_sm_n - d_sp_n) * (ap_AU / rhill_AU)
        d_n_leaf = torch.tensor([[d_sp_n, delta_tidal_n, d_pm_n]], dtype=torch.float32,
                                  requires_grad=True)
    else:
        d_n_leaf = torch.tensor([[d_sp_n, d_sm_n, d_pm_n]], dtype=torch.float32,
                                  requires_grad=True)

    with torch.enable_grad():
        V  = hnn_model(d_n_leaf, sys_enc_t)
        dV = torch.autograd.grad(V.sum(), d_n_leaf, create_graph=False)[0]
    t_pm   = float(dV[0, 2].item())                           # ∂V_θ/∂d_pm_n
    a_m_pm = V_ref * max(t_pm, 0.0) / (rhill_AU * mm_msun)  # planet-moon acc [AU/yr²]

    if a_m_pm > 1e-30:
        v_rel_hnn    = np.sqrt(r_pm * a_m_pm)
        v_rel_newton = v_m - v_p
        v_rel_mag    = float(np.linalg.norm(v_rel_newton))
        if v_rel_mag > 1e-30:
            v_m = v_p + v_rel_hnn * (v_rel_newton / v_rel_mag)

    return v_s, v_p, v_m


# ── Force diagnostic at t=0 (HNN v12 only) ───────────────────────────────────

def _hnn_force_diagnostic(
    st:       dict,
    p_cell:   "SystemParams",
    rhill_AU: float,
) -> tuple:
    """One forward+backward at t=0 initial conditions; no integration.

    Differentiates V_theta w.r.t. d_n directly (leaf tensor, not q_abs) so the
    three components are cleanly separated.  Returns:
      (pred_tsp, pred_tsm, pred_tpm)   — model predictions of dV/d(d_n)
      (true_tsp, true_tsm, true_tpm)   — analytical targets
      (d_sp_n,   d_sm_n,   d_pm_n)     — normalised distances at t=0
    """
    ms_solar = float(p_cell.ms_solar)
    mp_earth = float(p_cell.mp_earth)
    mm_earth = float(p_cell.mm_earth)
    ap_AU    = float(p_cell.ap_AU)
    mp_msun  = mp_earth * _MERTH_OVER_MSUN
    mm_msun  = mm_earth * _MERTH_OVER_MSUN

    r_s = st["pos_ms"].astype(np.float64)
    r_p = st["pos_mp"].astype(np.float64)
    r_m = st["pos_mm"].astype(np.float64)
    d_sp_n = float(np.linalg.norm(r_p - r_s)) / ap_AU
    d_sm_n = float(np.linalg.norm(r_m - r_s)) / ap_AU
    d_pm_n = float(np.linalg.norm(r_m - r_p)) / rhill_AU

    sys_raw   = np.array([[ms_solar, mp_earth, mm_earth, ap_AU, rhill_AU]])
    sys_fit   = np.log(sys_raw) if hnn_model.use_log_sys_enc else sys_raw
    sys_enc_t = torch.tensor(hnn_scaler.transform(sys_fit).astype(np.float32))

    if hnn_model.tidal_coords:
        delta_tidal_n = (d_sm_n - d_sp_n) * (ap_AU / rhill_AU)
        d_n_leaf = torch.tensor([[d_sp_n, delta_tidal_n, d_pm_n]], dtype=torch.float32,
                                  requires_grad=True)
    else:
        d_n_leaf  = torch.tensor([[d_sp_n, d_sm_n, d_pm_n]], dtype=torch.float32,
                                  requires_grad=True)

    with torch.enable_grad():
        V  = hnn_model(d_n_leaf, sys_enc_t)
        dV = torch.autograd.grad(V.sum(), d_n_leaf, create_graph=False)[0]

    pred_tsp = float(dV[0, 0].item())
    pred_tsm = float(dV[0, 1].item())   # = pred_t_tidal for tidal_coords models
    pred_tpm = float(dV[0, 2].item())

    mm_over_mp    = mm_msun / mp_msun
    mm_over_ms    = mm_msun / ms_solar
    ap_over_rhill = ap_AU   / rhill_AU

    true_tsp = 1.0            / d_sp_n ** 2
    true_tsm = mm_over_mp     / d_sm_n ** 2
    true_tpm = mm_over_ms * ap_over_rhill / d_pm_n ** 2

    return (pred_tsp, pred_tsm, pred_tpm), (true_tsp, true_tsm, true_tpm), \
           (d_sp_n, d_sm_n, d_pm_n)


# ── Newton sanity-check: analytical V_Newton_n through same inference chain ───

def _newton_accel(
    r_s: np.ndarray, r_p: np.ndarray, r_m: np.ndarray,
    ap_f32:    torch.Tensor,
    rhill_f32: torch.Tensor,
    ms_solar:  float, mp_msun: float, mm_msun: float,
    V_ref:     float,
) -> tuple:
    """Forces via autograd through V_Newton_n = -(1/d_sp_n + (mm/mp)/d_sm_n
    + (mm/ms)*(ap/rhill)/d_pm_n).

    Identical inference chain to _hnn_accel — only the potential changes.
    Should recover Newtonian accelerations exactly (within float32 precision).
    """
    mm_over_mp    = float(mm_msun / mp_msun)
    mm_over_ms    = float(mm_msun / ms_solar)
    ap_over_rhill = float(ap_f32.item() / rhill_f32.item())

    q = torch.tensor(
        np.concatenate([r_s, r_p, r_m]).astype(np.float32),
        dtype=torch.float32,
    ).unsqueeze(0)
    q_leaf = q.detach().requires_grad_(True)

    with torch.enable_grad():
        d_sp = (q_leaf[:, 3:6] - q_leaf[:, 0:3]).norm(dim=1)
        d_sm = (q_leaf[:, 6:9] - q_leaf[:, 0:3]).norm(dim=1)
        d_pm = (q_leaf[:, 6:9] - q_leaf[:, 3:6]).norm(dim=1)
        d_sp_n = d_sp / ap_f32
        d_sm_n = d_sm / ap_f32
        d_pm_n = d_pm / rhill_f32
        V_n = -(1.0 / d_sp_n
                + mm_over_mp    / d_sm_n
                + mm_over_ms * ap_over_rhill / d_pm_n)
        dV  = torch.autograd.grad(V_n.sum(), q_leaf, create_graph=False)[0]

    F_raw = (-dV.detach().numpy()[0]).astype(np.float64) * V_ref
    a_s   = F_raw[0:3] / ms_solar
    a_p   = F_raw[3:6] / mp_msun
    a_m   = F_raw[6:9] / mm_msun
    return a_s, a_p, a_m


def _newton_cell_trajectory(
    st:         dict,
    p_cell:     "SystemParams",
    t_sim:      float,
    n_out:      int,
    max_steps:  int,
    parquet_dt: bool = False,
) -> dict:
    """KDK leapfrog with V_Newton_n forces (sanity check).

    No velocity calibration: initial_state() velocities are already correct for
    Newtonian gravity, so no adjustment is needed.  The resulting trajectory
    should match the GT Numba integrator closely (float32 vs float64 introduces
    small numerical differences, but nothing like the HNN escape failure).
    """
    rhill_AU = float(st["rhill_AU"])
    am_AU    = float(st["am_AU"])
    ap_AU    = float(p_cell.ap_AU)
    ms_solar = float(p_cell.ms_solar)
    mp_msun  = float(p_cell.mp_earth * _MERTH_OVER_MSUN)
    mm_msun  = float(p_cell.mm_earth * _MERTH_OVER_MSUN)

    G_pm   = FOUR_PI2 * (mp_msun + mm_msun)
    T_moon = 2.0 * np.pi * np.sqrt(am_AU ** 3 / G_pm)

    if parquet_dt:
        n_phys = 999
        dt     = t_sim / n_phys
    else:
        dt_raw = min(T_moon / 100.0, 1.0 / _STEPS_PER_YEAR)
        n_phys = min(int(np.ceil(t_sim / dt_raw)), max_steps)
        dt     = t_sim / n_phys

    stride    = max(1, n_phys // n_out)
    ap_f32    = torch.tensor(ap_AU,    dtype=torch.float32)
    rhill_f32 = torch.tensor(rhill_AU, dtype=torch.float32)
    V_ref     = FOUR_PI2 * ms_solar * mp_msun / ap_AU

    r_s = st["pos_ms"].copy().astype(np.float64)
    r_p = st["pos_mp"].copy().astype(np.float64)
    r_m = st["pos_mm"].copy().astype(np.float64)
    v_s = st["vel_ms"].copy().astype(np.float64)
    v_p = st["vel_mp"].copy().astype(np.float64)
    v_m = st["vel_mm"].copy().astype(np.float64)

    cap      = n_phys // stride + 2
    t_arr    = np.zeros(cap)
    mpd      = np.zeros(cap)
    msd      = np.zeros(cap)
    t_arr[0] = 0.0
    mpd[0]   = np.linalg.norm(r_m - r_p)
    msd[0]   = np.linalg.norm(r_m - r_s)
    out_idx  = 1

    a_s, a_p, a_m = _newton_accel(
        r_s, r_p, r_m, ap_f32, rhill_f32, ms_solar, mp_msun, mm_msun, V_ref
    )
    half_dt = 0.5 * dt

    for step in range(1, n_phys + 1):
        v_s += half_dt * a_s;  v_p += half_dt * a_p;  v_m += half_dt * a_m
        r_s += dt * v_s;       r_p += dt * v_p;       r_m += dt * v_m
        a_s, a_p, a_m = _newton_accel(
            r_s, r_p, r_m, ap_f32, rhill_f32, ms_solar, mp_msun, mm_msun, V_ref
        )
        v_s += half_dt * a_s;  v_p += half_dt * a_p;  v_m += half_dt * a_m

        if step % stride == 0 and out_idx < cap:
            t_arr[out_idx] = step * dt
            mpd[out_idx]   = np.linalg.norm(r_m - r_p)
            msd[out_idx]   = np.linalg.norm(r_m - r_s)
            out_idx += 1

    return {
        "ok":               True,
        "t_arr":            t_arr[:out_idx],
        "moon_planet_dist": mpd[:out_idx],
        "moon_star_dist":   msd[:out_idx],
        "dt":               dt,
        "T_moon":           T_moon,
        "n_phys":           n_phys,
    }


# ── Greydanus (Option E) force computation via autograd ──────────────────────

def _greydanus_accel(
    r_s: np.ndarray, r_p: np.ndarray, r_m: np.ndarray,
    v_s: np.ndarray, v_p: np.ndarray, v_m: np.ndarray,
    ap_f32: torch.Tensor, rhill_f32: torch.Tensor,
    ms_solar: float, mp_msun: float, mm_msun: float,
    V_ref: float,
    v_circ: float,
    sys_enc_t: torch.Tensor,
) -> tuple:
    """KDK force step for HNNGreydanus: absolute positions + current velocities → accelerations.

    Forces = -∂H_θ(d_n, v_n)/∂q — velocity-dependent because H_θ is joint.
    At each KDK half-step the correct current velocities must be passed in.
    """
    q = torch.tensor(
        np.concatenate([r_s, r_p, r_m]).astype(np.float32),
        dtype=torch.float32,
    ).unsqueeze(0)   # (1, 9)
    q_leaf = q.detach().requires_grad_(True)

    # Convert velocities to normalised canonical momenta: p_n_i = (m_i/mp) * v_n_i
    v_raw = np.concatenate([v_s, v_p, v_m]).astype(np.float32) / v_circ
    ms_mp = ms_solar / mp_msun
    mm_mp = mm_msun  / mp_msun
    mass_ratio_arr = np.array([ms_mp]*3 + [1.0]*3 + [mm_mp]*3, dtype=np.float32)
    p_n = torch.tensor(mass_ratio_arr * v_raw, dtype=torch.float32).unsqueeze(0)  # (1, 9)

    with torch.enable_grad():
        d_sp = (q_leaf[:, 3:6] - q_leaf[:, 0:3]).norm(dim=1, keepdim=True)
        d_sm = (q_leaf[:, 6:9] - q_leaf[:, 0:3]).norm(dim=1, keepdim=True)
        d_pm = (q_leaf[:, 6:9] - q_leaf[:, 3:6]).norm(dim=1, keepdim=True)
        d_n  = torch.cat([d_sp / ap_f32, d_sm / ap_f32, d_pm / rhill_f32], dim=1)
        H    = hnn_model(d_n, p_n, sys_enc_t)   # HNNGreydanus.forward — p_n input
        dH   = torch.autograd.grad(H.sum(), q_leaf, create_graph=False)[0]

    F_raw = (-dH.detach().numpy()[0]).astype(np.float64) * V_ref
    a_s   = F_raw[0:3] / ms_solar
    a_p   = F_raw[3:6] / mp_msun
    a_m   = F_raw[6:9] / mm_msun
    return a_s, a_p, a_m


def _greydanus_calibrate_init_vel(
    st:        dict,
    v_circ:    float,
    ap_f32:    "torch.Tensor",
    rhill_f32: "torch.Tensor",
    ms_solar:  float,
    mp_msun:   float,
    mm_msun:   float,
    V_ref:     float,
    sys_enc_t: "torch.Tensor",
    n_iter:    int = 3,
) -> tuple:
    """Adjust moon's initial velocity for a circular orbit under the HNN force.

    The Newtonian initial conditions set v_moon for circular orbit under H_Newton.
    H_θ ≠ H_Newton, so the moon is unbound under H_θ from step 0 — catastrophic escape.
    Fix: replace v_moon with v_moon_θ such that |a_m_HNN| * r_pm = v_rel_θ²
    (centripetal balance under HNN forces). Positions are unchanged.

    Iterates n_iter times because for Greydanus H_θ the force depends on velocity.
    """
    r_s = st["pos_ms"].astype(np.float64)
    r_p = st["pos_mp"].astype(np.float64)
    r_m = st["pos_mm"].astype(np.float64)
    v_s = st["vel_ms"].copy().astype(np.float64)
    v_p = st["vel_mp"].copy().astype(np.float64)
    v_m = st["vel_mm"].copy().astype(np.float64)

    r_pm_vec = r_m - r_p
    r_pm     = float(np.linalg.norm(r_pm_vec))

    for _ in range(n_iter):
        _, _, a_m = _greydanus_accel(
            r_s, r_p, r_m, v_s, v_p, v_m,
            ap_f32, rhill_f32, ms_solar, mp_msun, mm_msun, V_ref, v_circ, sys_enc_t,
        )
        a_m_mag = float(np.linalg.norm(a_m))
        if a_m_mag < 1e-30:
            break
        # Circular orbit speed under H_θ: v_rel_θ = sqrt(r_pm * |a_m_HNN|)
        v_rel_hnn = np.sqrt(r_pm * a_m_mag)
        # Keep the same direction as the Newtonian relative velocity
        v_rel_newton = v_m - v_p
        v_rel_mag    = float(np.linalg.norm(v_rel_newton))
        if v_rel_mag < 1e-30:
            break
        v_m = v_p + v_rel_hnn * (v_rel_newton / v_rel_mag)

    return v_s, v_p, v_m


def _greydanus_cell_trajectory(
    st:         dict,
    p_cell:     SystemParams,
    t_sim:      float,
    n_out:      int,
    max_steps:  int,
    parquet_dt: bool = False,
) -> dict:
    """KDK leapfrog using HNNGreydanus forces (velocity-dependent).

    Forces are re-evaluated with the current velocities at each KDK half-step,
    matching the Greydanus integration scheme for joint H_θ(q, v).
    """
    rhill_AU = float(st["rhill_AU"])
    am_AU    = float(st["am_AU"])
    ap_AU    = float(p_cell.ap_AU)

    ms_solar = float(p_cell.ms_solar)
    mp_msun  = float(p_cell.mp_earth * _MERTH_OVER_MSUN)
    mm_msun  = float(p_cell.mm_earth * _MERTH_OVER_MSUN)

    G_pm   = FOUR_PI2 * (mp_msun + mm_msun)
    T_moon = 2.0 * np.pi * np.sqrt(am_AU ** 3 / G_pm)

    if parquet_dt:
        n_phys = 999
        dt     = t_sim / n_phys
    else:
        dt_raw = min(T_moon / 100.0, 1.0 / _STEPS_PER_YEAR)
        n_phys = min(int(np.ceil(t_sim / dt_raw)), max_steps)
        dt     = t_sim / n_phys

    stride  = max(1, n_phys // n_out)

    ap_f32    = torch.tensor(ap_AU,    dtype=torch.float32)
    rhill_f32 = torch.tensor(rhill_AU, dtype=torch.float32)

    sys_raw   = np.array([[ms_solar, p_cell.mp_earth, p_cell.mm_earth, ap_AU, rhill_AU]])
    sys_enc_t = torch.tensor(hnn_scaler.transform(sys_raw).astype(np.float32))

    V_ref  = FOUR_PI2 * ms_solar * mp_msun / ap_AU
    v_circ = float(np.sqrt(_FOUR_PI2 * ms_solar / ap_AU))   # planet orbital speed

    r_s = st["pos_ms"].copy().astype(np.float64)
    r_p = st["pos_mp"].copy().astype(np.float64)
    r_m = st["pos_mm"].copy().astype(np.float64)

    # Calibrate moon's initial velocity to circular orbit under H_θ (not H_Newton).
    # Eliminates catastrophic escape caused by H_θ ≠ H_Newton binding energy mismatch.
    v_s, v_p, v_m = _greydanus_calibrate_init_vel(
        st, v_circ, ap_f32, rhill_f32,
        ms_solar, mp_msun, mm_msun, V_ref, sys_enc_t,
    )

    cap    = n_phys // stride + 2
    t_arr  = np.zeros(cap)
    mpd    = np.zeros(cap)
    msd    = np.zeros(cap)
    t_arr[0] = 0.0
    mpd[0]   = np.linalg.norm(r_m - r_p)
    msd[0]   = np.linalg.norm(r_m - r_s)
    out_idx  = 1

    # Initial forces using initial velocities
    a_s, a_p, a_m = _greydanus_accel(
        r_s, r_p, r_m, v_s, v_p, v_m,
        ap_f32, rhill_f32, ms_solar, mp_msun, mm_msun, V_ref, v_circ, sys_enc_t,
    )
    half_dt = 0.5 * dt

    for step in range(1, n_phys + 1):
        # First half kick: update velocities, then recompute forces with v_half
        v_s += half_dt * a_s
        v_p += half_dt * a_p
        v_m += half_dt * a_m

        # Full drift
        r_s += dt * v_s
        r_p += dt * v_p
        r_m += dt * v_m

        # New forces at new positions AND current (post-drift) velocities
        a_s, a_p, a_m = _greydanus_accel(
            r_s, r_p, r_m, v_s, v_p, v_m,
            ap_f32, rhill_f32, ms_solar, mp_msun, mm_msun, V_ref, v_circ, sys_enc_t,
        )

        # Second half kick
        v_s += half_dt * a_s
        v_p += half_dt * a_p
        v_m += half_dt * a_m

        if step % stride == 0 and out_idx < cap:
            t_arr[out_idx] = step * dt
            mpd[out_idx]   = np.linalg.norm(r_m - r_p)
            msd[out_idx]   = np.linalg.norm(r_m - r_s)
            out_idx += 1

    return {
        "ok":               True,
        "t_arr":            t_arr[:out_idx],
        "moon_planet_dist": mpd[:out_idx],
        "moon_star_dist":   msd[:out_idx],
        "dt":               dt,
        "T_moon":           T_moon,
        "n_phys":           n_phys,
    }


# ── Grid construction (mirrors hnn_inference.py _build_initial_states) ────────

def _build_grids(ms_solar, mp_earth, ap_AU, ep, rhill_AU):
    """Build mm_grid and am_grid exactly as batch_hnn_trajectories does."""
    mp_kg   = mp_earth * merth
    dp_si   = 5.5 * 1e3   # kg/m³ — default planet density
    dm_si   = 3.0 * 1e3   # default moon density (same as hnn_inference.py)
    rp_m    = (0.75 * mp_kg / (np.pi * dp_si)) ** (1/3)
    a_roche = 2.456 * rp_m * (dp_si / dm_si) ** (1/3) / AU_m
    am_min  = max(a_roche / rhill_AU, 1e-3)

    mm_max  = min(mp_earth, 3.0)
    mm_grid = np.exp(np.linspace(np.log(0.107), np.log(mm_max), MM_RES))
    am_grid = np.linspace(am_min, 1.0, AM_RES)
    return mm_grid, am_grid


# ── Helpers ───────────────────────────────────────────────────────────────────

def resample_to(arr: np.ndarray, n: int) -> np.ndarray:
    idx = np.round(np.linspace(0, len(arr) - 1, n)).astype(int)
    return arr[idx]


def pct_within(errs, thresh):
    return (np.abs(errs) < thresh).mean() * 100


def err_summary(errs):
    return (f"mean|e|={np.abs(errs).mean()*100:.1f}%  "
            f"max|e|={np.abs(errs).max()*100:.1f}%  "
            f"<15%={pct_within(errs*100,15):.0f}%  "
            f"<30%={pct_within(errs*100,30):.0f}%")


def stab_label(mpd, rhill):
    return "STABLE  " if np.nanmax(mpd) <= rhill else "ESCAPED "

def hab_label(msd, a_inner, a_outer):
    """Moon stays inside HZ for the full trajectory (same criterion as batch inference)."""
    return "HABITBL " if (np.nanmin(msd) >= a_inner and np.nanmax(msd) <= a_outer) else "UNHABIT "


# ── Collect meta files ────────────────────────────────────────────────────────

def _meta_filter(fname):
    if not fname.endswith(".meta.json"):
        return False
    if "retrograde" in fname:
        return args.include_retro
    return True  # prograde always included

all_meta = sorted(
    os.path.join(GT_DIR, f)
    for f in os.listdir(GT_DIR)
    if _meta_filter(f)
)

if args.systems:
    filters = [s.strip() for s in args.systems.split(",")]
    all_meta = [m for m in all_meta
                if any(flt in os.path.basename(m) for flt in filters)]

print(f"Systems  : {len(all_meta)}\n")

SEP  = "=" * 115


# ── Per-system loop ───────────────────────────────────────────────────────────

for meta_path in all_meta:
    with open(meta_path) as f:
        meta = json.load(f)

    sp_raw = meta["system_params"]
    t_sim  = float(meta["t_sim"])
    em     = float(meta["em"])
    retro  = bool(meta["moon_retrograde"])
    label  = (os.path.basename(meta_path)
              .replace("_prograde.meta.json", "")
              .replace("_retrograde.meta.json", "")
              .replace(".meta.json", ""))

    system_params = {k: float(sp_raw[k])
                     for k in ("ms_solar", "rs_solar", "Ts", "mp_earth", "ap_AU")}
    system_params["ep"] = float(sp_raw.get("ep", 0.0))

    ms_solar = system_params["ms_solar"]
    mp_earth = system_params["mp_earth"]
    ap_AU    = system_params["ap_AU"]
    ep       = system_params["ep"]

    mp_msun  = mp_earth * _MERTH_OVER_MSUN
    rhill_AU = ap_AU * (1 - ep) * (mp_msun / (3 * ms_solar)) ** (1/3)

    mm_grid, am_grid = _build_grids(ms_solar, mp_earth, ap_AU, ep, rhill_AU)

    rs_solar     = system_params.get("rs_solar", 1.0)
    Ts           = system_params.get("Ts", 5772.0)
    a_inner_au, a_outer_au = hz_bounds_au(Ts, rs_solar * rsun)

    # For tiny-rhill systems (e.g. TRAPPIST-1e rhill=0.00058 AU) the standard
    # am_idx 12/25/37 cells are nearly all GT-escaped.  Add denser low-am_idx
    # coverage so we have at least a few GT-stable cells to evaluate.
    if rhill_AU < args.small_rhill_thresh:
        cell_indices = list(CELL_INDICES) + [
            (12, 2), (12, 5), (12, 8),
            (25, 2), (25, 5), (25, 8),
            (37, 2), (37, 5), (37, 8),
        ]
    else:
        cell_indices = list(CELL_INDICES)

    print(SEP)
    print(f"  SYSTEM : {label}   t_sim={t_sim} yr   retro={retro}   em={em}")
    print(f"           ms={ms_solar}  mp={mp_earth} Mearth  ap={ap_AU} AU"
          f"  rhill={rhill_AU:.5f} AU")
    print(f"           HZ: [{a_inner_au:.4f}, {a_outer_au:.4f}] AU"
          f"  (Ts={Ts:.0f}K  rs={rs_solar:.3f}R☉)")
    print(SEP)

    for mm_idx, am_idx in cell_indices:
        mm_earth = float(mm_grid[mm_idx])
        am_hill  = float(am_grid[am_idx])
        am_AU    = am_hill * rhill_AU

        G_pm   = FOUR_PI2 * ((mp_earth + mm_earth) * _MERTH_OVER_MSUN)
        T_moon = 2.0 * np.pi * np.sqrt(am_AU ** 3 / G_pm)
        dt_exp = min(T_moon / 100.0, 1.0 / _STEPS_PER_YEAR)
        n_exp  = min(int(np.ceil(t_sim / dt_exp)), args.max_steps)

        print(f"\n  Cell ({mm_idx:>2},{am_idx:>2})"
              f"  mm={mm_earth:.4f} Mearth  am={am_hill:.3f} Hill"
              f"  am_AU={am_AU:.5f} AU  T_moon={T_moon:.4f} yr"
              f"  dt={dt_exp:.2e} yr  n_phys~{n_exp:,}")

        # ── Build SystemParams for this cell ─────────────────────────────────
        p_cell = SystemParams(
            ms_solar       = ms_solar,
            rs_solar       = system_params.get("rs_solar", 1.0),
            Ts             = system_params.get("Ts", 5772.0),
            mp_earth       = mp_earth,
            dp_cgs         = 5.5,
            ap_AU          = ap_AU,
            ep             = ep,
            mm_earth       = mm_earth,
            am_hill        = am_hill,
            em             = em,
            moon_retrograde= retro,
        )
        st = initial_state(p_cell)

        # ── Force diagnostic at t=0 (HNN v12 only, --newton flag) ───────────
        if args.newton and not _HNN_IS_GREYDANUS:
            (p_tsp, p_tsm, p_tpm), (a_tsp, a_tsm, a_tpm), (dsp_n, dsm_n, dpm_n) = \
                _hnn_force_diagnostic(st, p_cell, rhill_AU)
            def _fe(pred, true):
                if abs(true) > 1e-30:
                    return (f"pred={pred:>11.5f}  true={true:>11.5f}  "
                            f"ratio={pred/true:>8.3f}x  err={100*(pred-true)/true:>+8.1f}%")
                return f"pred={pred:>11.5f}  true={true:>11.5f}  (true~0)"
            print(f"\n  [Newton] Force diagnostic at t=0 "
                  f"(d_n=[{dsp_n:.4f}, {dsm_n:.4f}, {dpm_n:.4f}]):")
            print(f"    t_sp : {_fe(p_tsp, a_tsp)}")
            print(f"    t_sm : {_fe(p_tsm, a_tsm)}")
            print(f"    t_pm : {_fe(p_tpm, a_tpm)}")

        # ── Ground truth ─────────────────────────────────────────────────────
        t0    = time.perf_counter()
        sim_gt = run_simulation_for_years(p_cell, t_sim)
        gt_el = time.perf_counter() - t0

        traj         = sim_gt["traj"]
        gt_mpd_full  = np.linalg.norm(traj["xyzarr_mm"] - traj["xyzarr_mp"], axis=1)
        gt_msd_full  = np.linalg.norm(traj["xyzarr_mm"] - traj["xyzarr_ms"], axis=1)
        gt_mpd       = resample_to(gt_mpd_full, args.n_out)
        gt_msd       = resample_to(gt_msd_full, args.n_out)
        t_gt         = np.linspace(0.0, t_sim, args.n_out)

        # ── HNN trajectory (v12 / Greydanus / Hill-frame, per-cell dt) ───────────
        t0 = time.perf_counter()
        if args.hill_coords:
            hnn_r = _hnn_hill_cell_trajectory(
                st, p_cell, t_sim, args.n_out, args.max_steps,
                parquet_dt=args.parquet_dt,
            )
            hnn_label = "Hill"
        elif _HNN_IS_GREYDANUS:
            hnn_r = _greydanus_cell_trajectory(
                st, p_cell, t_sim, args.n_out, args.max_steps,
                parquet_dt=args.parquet_dt,
            )
            hnn_label = "Grey"
        else:
            hnn_r = _hnn_cell_trajectory(
                st, p_cell, t_sim, args.n_out, args.max_steps,
                parquet_dt=args.parquet_dt,
            )
            hnn_label = "HNN "
        hnn_el = time.perf_counter() - t0

        hnn_mpd = resample_to(np.array(hnn_r["moon_planet_dist"]), args.n_out)
        hnn_msd = resample_to(np.array(hnn_r["moon_star_dist"]),   args.n_out)
        print(f"         {hnn_label} dt={hnn_r['dt']:.2e} yr  "
              f"n_phys={hnn_r['n_phys']:,}  wall={hnn_el:.1f}s")

        # ── Newton sanity check (--newton flag) ───────────────────────────────
        if args.newton:
            t0_newt = time.perf_counter()
            newt_r  = _newton_cell_trajectory(
                st, p_cell, t_sim, args.n_out, args.max_steps,
                parquet_dt=args.parquet_dt,
            )
            newt_el  = time.perf_counter() - t0_newt
            newt_mpd = resample_to(np.array(newt_r["moon_planet_dist"]), args.n_out)
            newt_msd = resample_to(np.array(newt_r["moon_star_dist"]),   args.n_out)
            print(f"         Newt dt={newt_r['dt']:.2e} yr  "
                  f"n_phys={newt_r['n_phys']:,}  wall={newt_el:.1f}s")
        else:
            newt_mpd = np.full(args.n_out, np.nan)
            newt_msd = np.full(args.n_out, np.nan)

        # ── Force MLP (per-cell dt internally) ───────────────────────────────
        sp_full = dict(**system_params,
                       mm_earth       = mm_earth,
                       am_hill        = am_hill,
                       em             = em,
                       moon_retrograde= retro,
                       dp_cgs         = 5.5)
        t0      = time.perf_counter()
        fmlp_r  = preview_trajectory(sp_full, t_sim, model_dir=FMLP_DIR,
                                     n_out=args.n_out)
        fmlp_el = time.perf_counter() - t0

        if fmlp_r.get("ok"):
            fraw_mpd = np.array(fmlp_r["moon_planet_dist"])
            fraw_msd = np.array(fmlp_r["moon_star_dist"])
            if len(fraw_mpd) >= args.n_out:
                fmlp_mpd = resample_to(fraw_mpd, args.n_out)
                fmlp_msd = resample_to(fraw_msd, args.n_out)
            else:
                fmlp_mpd = np.full(args.n_out, fraw_mpd[-1])
                fmlp_msd = np.full(args.n_out, fraw_msd[-1])
                fmlp_mpd[:len(fraw_mpd)] = fraw_mpd
                fmlp_msd[:len(fraw_msd)] = fraw_msd
            print(f"         FMLP dt={fmlp_r.get('dt_yr',float('nan')):.2e} yr  "
                  f"n_phys~{fmlp_r.get('n_phys_steps',0):,}  wall={fmlp_el:.2f}s")
        else:
            fmlp_mpd = np.full(args.n_out, np.nan)
            fmlp_msd = np.full(args.n_out, np.nan)
            print(f"         FMLP FAILED: {fmlp_r.get('error')}")

        # ── Error arrays ──────────────────────────────────────────────────────
        eps = 1e-12
        hnn_mpd_err  = (hnn_mpd  - gt_mpd) / (gt_mpd + eps)
        fmlp_mpd_err = (fmlp_mpd - gt_mpd) / (gt_mpd + eps)
        hnn_msd_err  = (hnn_msd  - gt_msd) / (gt_msd + eps)
        fmlp_msd_err = (fmlp_msd - gt_msd) / (gt_msd + eps)
        newt_mpd_err = (newt_mpd - gt_mpd)  / (gt_mpd + eps)
        newt_msd_err = (newt_msd - gt_msd)  / (gt_msd + eps)

        # ── Detail table ──────────────────────────────────────────────────────
        if args.newton:
            print(f"\n    {'t_yr':>6}  {'GT mpd':>10}  {'HNN mpd':>10}  {'HNN%':>7}"
                  f"  {'Newt mpd':>10}  {'Newt%':>7}"
                  f"  {'FMLP mpd':>10}  {'FMLP%':>7}"
                  f"  |  {'GT msd':>8}  {'HNN msd':>8}  {'Newt msd':>8}  {'FMLP msd':>8}")
            print("    " + "-" * 125)
        else:
            print(f"\n    {'t_yr':>6}  {'GT mpd':>10}  {'HNN mpd':>10}  {'HNN%':>7}"
                  f"  {'FMLP mpd':>10}  {'FMLP%':>7}"
                  f"  |  {'GT msd':>8}  {'HNN msd':>8}  {'FMLP msd':>8}")
            print("    " + "-" * 97)
        pt_idx = np.round(np.linspace(0, args.n_out - 1, N_PRINT)).astype(int)
        for i in pt_idx:
            if args.newton:
                print(f"    {t_gt[i]:>6.2f}  "
                      f"{gt_mpd[i]:>10.6f}  {hnn_mpd[i]:>10.6f}  {hnn_mpd_err[i]*100:>+6.1f}%  "
                      f"{newt_mpd[i]:>10.6f}  {newt_mpd_err[i]*100:>+6.1f}%  "
                      f"{fmlp_mpd[i]:>10.6f}  {fmlp_mpd_err[i]*100:>+6.1f}%  "
                      f"|  {gt_msd[i]:>8.4f}  {hnn_msd[i]:>8.4f}  {newt_msd[i]:>8.4f}  {fmlp_msd[i]:>8.4f}")
            else:
                print(f"    {t_gt[i]:>6.2f}  "
                      f"{gt_mpd[i]:>10.6f}  {hnn_mpd[i]:>10.6f}  {hnn_mpd_err[i]*100:>+6.1f}%  "
                      f"{fmlp_mpd[i]:>10.6f}  {fmlp_mpd_err[i]*100:>+6.1f}%  "
                      f"|  {gt_msd[i]:>8.4f}  {hnn_msd[i]:>8.4f}  {fmlp_msd[i]:>8.4f}")

        # ── Summary ───────────────────────────────────────────────────────────
        print(f"\n    mpd  HNN : {err_summary(hnn_mpd_err)}")
        if args.newton and not np.all(np.isnan(newt_mpd)):
            print(f"    mpd Newt : {err_summary(newt_mpd_err)}")
        print(f"    mpd FMLP : {err_summary(fmlp_mpd_err)}")
        print(f"    msd  HNN : {err_summary(hnn_msd_err)}")
        if args.newton and not np.all(np.isnan(newt_msd)):
            print(f"    msd Newt : {err_summary(newt_msd_err)}")
        print(f"    msd FMLP : {err_summary(fmlp_msd_err)}")
        stab_str = (f"    Stability   GT={stab_label(gt_mpd_full, rhill_AU)}"
                    f"  HNN={stab_label(hnn_mpd, rhill_AU)}")
        if args.newton:
            stab_str += f"  Newt={stab_label(newt_mpd, rhill_AU)}"
        stab_str += (f"  FMLP={stab_label(fmlp_mpd, rhill_AU)}"
                     f"  (rhill={rhill_AU:.5f} AU)")
        print(stab_str)

        hab_str = (f"    Habitability GT={hab_label(gt_msd_full, a_inner_au, a_outer_au)}"
                   f"  HNN={hab_label(hnn_msd, a_inner_au, a_outer_au)}")
        if args.newton and not np.all(np.isnan(newt_msd)):
            hab_str += f"  Newt={hab_label(newt_msd, a_inner_au, a_outer_au)}"
        hab_str += (f"  FMLP={hab_label(fmlp_msd, a_inner_au, a_outer_au)}"
                    f"  (HZ=[{a_inner_au:.4f},{a_outer_au:.4f}] AU)")
        print(hab_str)

        timing_str = f"    Timing      GT={gt_el:.2f}s  HNN={hnn_el:.1f}s"
        if args.newton:
            timing_str += f"  Newt={newt_el:.1f}s"
        timing_str += f"  FMLP={fmlp_el:.2f}s"
        print(timing_str)

    print()
