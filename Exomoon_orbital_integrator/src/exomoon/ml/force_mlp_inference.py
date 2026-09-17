"""
exomoon/ml/force_mlp_inference.py — 2D trajectory preview using force regression MLPs.

Second trajectory-preview option alongside HNN v12 (models_hnn_dist_v12/).
HNN v12 is the primary completed model; this module provides an alternative
leapfrog integrator driven by the three direct force regression MLPs.

For a single (mm_earth, am_hill) configuration, integrates all three bodies
using forces predicted by the MLP ensemble, then returns:
  - t_arr             : time array (years), length n_out
  - moon_planet_dist  : |r_moon - r_planet| (AU), length n_out
  - moon_star_dist    : |r_moon - r_star|   (AU), length n_out
  - events            : list of (t_yr, message) for stopping-criteria annotations
  - rhill_AU, a_inner_au, a_outer_au : boundary values for plot annotations

Force-to-acceleration conversion table (derived from dimensionless targets):
  C_ref = G * ms_solar / ap_AU^2          (base acceleration scale, AU/yr^2)

  a_planet_from_star = C_ref * t_sp                            (toward star)
  a_star_from_planet = C_ref * (mp/ms) * t_sp                  (toward planet)
  a_moon_from_star   = C_ref * (mp/mm) * t_sm                  (toward star)
  a_star_from_moon   = C_ref * (mp/ms) * t_sm                  (toward moon)
  a_planet_from_moon = C_ref * (ap/rhill) * t_pm               (toward moon)
  a_moon_from_planet = C_ref * (ap/rhill) * (mp/mm) * t_pm     (toward planet)

Stopping criteria (same as ml_gru_2d_preview_plan memory):
  - Stop when stable=0 AND habitable=0 simultaneously
  - Annotate "moon escaped!"       at first step where moon_planet_dist > rhill_AU
  - Annotate "moon uninhabitable!" at first step where moon_star_dist outside HZ

ISOLATION: does NOT touch models/, models_temphead/, eval_aux_mlp_output/,
           models_hnn_dist_v12/, or any GRU/classification infrastructure.
"""

import os
import sys

import numpy as np
import torch

_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from exomoon.constants         import rsun, merth, msun, FOUR_PI2, au
from exomoon.params            import SystemParams
from exomoon.initial_conditions import initial_state
from exomoon.habitable_zone    import hz_bounds_au
from exomoon.ml.hnn_dataset    import load_sys_scaler
from exomoon.ml.force_mlp_model import ForceMLPEnsemble

_MEM = merth / msun      # Earth mass in solar masses  ≈ 3.003e-6
_G   = FOUR_PI2          # G in (AU, yr, M_sun) units = 4π^2


# ── Internal helpers ──────────────────────────────────────────────────────────

def _unit(v: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(v)
    return v / (d + 1e-300)


def _accel(
    r_s: np.ndarray, r_p: np.ndarray, r_m: np.ndarray,
    t_sp: float, t_sm: float, t_pm: float,
    C_ref: float, ap_rhill: float, mp_over_ms: float, mp_over_mm: float,
) -> tuple:
    """Convert dimensionless force magnitudes to body accelerations (AU/yr^2).

    See module docstring for the full derivation of each conversion factor.
    """
    hat_ps = _unit(r_s - r_p)   # planet → star
    hat_pm_ = _unit(r_m - r_p)  # planet → moon

    hat_sp_ = _unit(r_p - r_s)  # star → planet
    hat_sm_ = _unit(r_m - r_s)  # star → moon

    hat_ms_ = _unit(r_s - r_m)  # moon → star
    hat_mp_ = _unit(r_p - r_m)  # moon → planet

    a_p = (C_ref * t_sp) * hat_ps + (C_ref * ap_rhill * t_pm) * hat_pm_
    a_s = (C_ref * mp_over_ms * t_sp) * hat_sp_ + (C_ref * mp_over_ms * t_sm) * hat_sm_
    a_m = (C_ref * mp_over_mm * t_sm) * hat_ms_ + (C_ref * ap_rhill * mp_over_mm * t_pm) * hat_mp_

    return a_s, a_p, a_m


def _mlp_forces(
    r_s: np.ndarray, r_p: np.ndarray, r_m: np.ndarray,
    ap_AU: float, rhill_AU: float,
    sys_enc_t: torch.Tensor,          # (1, 5) pre-normalised
    ensemble: ForceMLPEnsemble,
) -> tuple:
    """Query the force MLP ensemble for current positions.

    Returns (t_sp, t_sm, t_pm) as Python floats.
    """
    d_sp = np.linalg.norm(r_p - r_s)
    d_sm = np.linalg.norm(r_m - r_s)
    d_pm = np.linalg.norm(r_m - r_p)

    d_sp_n = d_sp / ap_AU
    d_sm_n = d_sm / ap_AU
    d_pm_n = d_pm / rhill_AU

    d_n_t = torch.tensor([[d_sp_n, d_sm_n, d_pm_n]], dtype=torch.float32)
    with torch.no_grad():
        t = ensemble.forward(d_n_t, sys_enc_t)   # (1, 3)
    t_sp, t_sm, t_pm = t[0, 0].item(), t[0, 1].item(), t[0, 2].item()
    return t_sp, t_sm, t_pm


# ── Public API ────────────────────────────────────────────────────────────────

def preview_trajectory(
    system_params:   dict,
    t_sim:           float,
    model_dir:       str  = "models_force_mlp/",
    n_out:           int  = 1000,
    dt_steps_per_orbit: int = 100,
    max_phys_steps:  int  = 200_000,
) -> dict:
    """Run a 2D trajectory preview for one (mm_earth, am_hill) configuration.

    Parameters
    ----------
    system_params : dict
        Must contain: ms_solar, rs_solar, Ts, mp_earth, mm_earth, ap_AU, ep,
                      am_hill, em, moon_retrograde, dp_cgs
    t_sim         : simulation duration (years)
    model_dir     : directory containing force_mlp_config.json + weights + hnn_sys_scaler.pkl
    n_out         : number of output time-points (evenly subsampled)
    dt_steps_per_orbit : integration steps per moon orbital period
    max_phys_steps : hard cap on integration steps (safety limit)

    Returns
    -------
    dict with keys:
        ok, t_arr, moon_planet_dist, moon_star_dist,
        rhill_AU, a_inner_au, a_outer_au, events,
        [error, message] on failure
    """
    # ── Load model ────────────────────────────────────────────────────────────
    cfg_path = os.path.join(model_dir, ForceMLPEnsemble.CFG_FILE)
    if not os.path.exists(cfg_path):
        return {"ok": False, "error": "no_model",
                "message": f"No trained force MLP found at {model_dir}"}
    try:
        ensemble = ForceMLPEnsemble.load(model_dir)
        ensemble.eval()
        sys_scaler = load_sys_scaler(model_dir)
    except Exception as exc:
        return {"ok": False, "error": "load_error", "message": str(exc)}

    # ── Build SystemParams and initial state ──────────────────────────────────
    sp = SystemParams(
        ms_solar       = float(system_params["ms_solar"]),
        rs_solar       = float(system_params.get("rs_solar", 1.0)),
        Ts             = float(system_params.get("Ts", 5772.0)),
        mp_earth       = float(system_params["mp_earth"]),
        dp_cgs         = float(system_params.get("dp_cgs", 5.5)),
        ap_AU          = float(system_params["ap_AU"]),
        ep             = float(system_params.get("ep", 0.0)),
        mm_earth       = float(system_params["mm_earth"]),
        am_hill        = float(system_params["am_hill"]),
        em             = float(system_params.get("em", 0.0)),
        moon_retrograde= bool(system_params.get("moon_retrograde", False)),
    )

    st       = initial_state(sp)
    rhill_AU = float(st["rhill_AU"])
    am_AU    = float(st["am_AU"])

    # Habitable zone
    rs_m = sp.rs_solar * rsun
    a_inner_au, a_outer_au = hz_bounds_au(sp.Ts, rs_m)

    # Mass ratios (in solar masses) for acceleration conversion
    ms     = sp.ms_solar
    mp     = sp.mp_earth * _MEM
    mm     = sp.mm_earth * _MEM
    ap_AU  = sp.ap_AU

    C_ref      = _G * ms / ap_AU ** 2   # AU/yr^2  — base acceleration scale
    ap_rhill   = ap_AU / rhill_AU
    mp_over_ms = mp / ms
    mp_over_mm = mp / mm

    # sys_enc for MLP (same 5-dim vector as HNN training)
    sys_raw = np.array([[ms, sp.mp_earth, sp.mm_earth, ap_AU, rhill_AU]], dtype=np.float64)
    sys_enc = torch.tensor(
        sys_scaler.transform(sys_raw).astype(np.float32)
    )   # (1, 5)

    # ── Adaptive timestep ─────────────────────────────────────────────────────
    # Moon orbital period at periapsis of its orbit (am_AU*(1-em) at closest)
    G_pm    = _G * (mp + mm)
    T_moon  = 2.0 * np.pi * np.sqrt(am_AU ** 3 / G_pm)   # years
    dt      = T_moon / max(dt_steps_per_orbit, 10)
    n_phys  = min(int(np.ceil(t_sim / dt)), max_phys_steps)
    dt      = t_sim / n_phys                              # re-normalise so we hit t_sim exactly
    subsample = max(1, n_phys // n_out)

    # ── Initial positions / velocities ────────────────────────────────────────
    r_s = st["pos_ms"].copy()
    r_p = st["pos_mp"].copy()
    r_m = st["pos_mm"].copy()
    v_s = st["vel_ms"].copy()
    v_p = st["vel_mp"].copy()
    v_m = st["vel_mm"].copy()

    # ── Output arrays ─────────────────────────────────────────────────────────
    n_actual = n_phys // subsample + 1
    t_arr          = np.zeros(n_actual)
    mp_dist_arr    = np.zeros(n_actual)   # moon-planet distance
    ms_dist_arr    = np.zeros(n_actual)   # moon-star distance

    # ── Stopping criteria state ───────────────────────────────────────────────
    escaped      = False
    uninhabitable= False
    events: list = []
    stopped_at   = n_phys   # step index where we halt (inclusive)

    # Record initial state
    t_arr[0]       = 0.0
    mp_dist_arr[0] = float(np.linalg.norm(r_m - r_p))
    ms_dist_arr[0] = float(np.linalg.norm(r_m - r_s))
    out_idx = 1

    # ── KDK leapfrog ─────────────────────────────────────────────────────────
    # Initial acceleration
    t_sp, t_sm, t_pm = _mlp_forces(r_s, r_p, r_m, ap_AU, rhill_AU, sys_enc, ensemble)
    a_s, a_p, a_m = _accel(r_s, r_p, r_m, t_sp, t_sm, t_pm,
                            C_ref, ap_rhill, mp_over_ms, mp_over_mm)

    for step in range(1, n_phys + 1):
        # Half kick
        v_s += 0.5 * dt * a_s
        v_p += 0.5 * dt * a_p
        v_m += 0.5 * dt * a_m

        # Full drift
        r_s += dt * v_s
        r_p += dt * v_p
        r_m += dt * v_m

        # New accelerations
        t_sp, t_sm, t_pm = _mlp_forces(r_s, r_p, r_m, ap_AU, rhill_AU, sys_enc, ensemble)
        a_s, a_p, a_m = _accel(r_s, r_p, r_m, t_sp, t_sm, t_pm,
                                C_ref, ap_rhill, mp_over_ms, mp_over_mm)

        # Half kick
        v_s += 0.5 * dt * a_s
        v_p += 0.5 * dt * a_p
        v_m += 0.5 * dt * a_m

        # Distances
        d_pm = float(np.linalg.norm(r_m - r_p))
        d_ms = float(np.linalg.norm(r_m - r_s))
        t_yr = step * dt

        # Record output point
        if step % subsample == 0 and out_idx < n_actual:
            t_arr[out_idx]       = t_yr
            mp_dist_arr[out_idx] = d_pm
            ms_dist_arr[out_idx] = d_ms
            out_idx += 1

        # Stability event
        if not escaped and d_pm > rhill_AU:
            escaped = True
            events.append((t_yr, "moon escaped!"))

        # Habitability event
        if not uninhabitable and not (a_inner_au <= d_ms <= a_outer_au):
            uninhabitable = True
            events.append((t_yr, "moon uninhabitable!"))

        # Stop when BOTH stable=0 AND habitable=0
        if escaped and uninhabitable:
            stopped_at = step
            break

    # Trim output arrays to actual filled length
    t_arr       = t_arr[:out_idx]
    mp_dist_arr = mp_dist_arr[:out_idx]
    ms_dist_arr = ms_dist_arr[:out_idx]

    return {
        "ok":               True,
        "t_arr":            t_arr.tolist(),
        "moon_planet_dist": mp_dist_arr.tolist(),
        "moon_star_dist":   ms_dist_arr.tolist(),
        "rhill_AU":         rhill_AU,
        "a_inner_au":       a_inner_au,
        "a_outer_au":       a_outer_au,
        "events":           events,
        "n_phys_steps":     stopped_at,
        "dt_yr":            dt,
        "T_moon_yr":        T_moon,
    }
