"""
diag_inner_cells.py — Targeted single-cell diagnostic for HNN hinge4 inner cells.

Tests K-1229b prograde at am_idx=3 and am_idx=6 in the 50×50 grid
(am_hill≈0.094 and 0.153) — the cells that fail in batch inference but
were never tested in the original single-cell eval (which only used am_idx=12,25,37).

t_sim=0.5yr: inner T_moon≈3.7e-3yr → ~135 moon orbits.
If HNN escapes at these cells, it will do so within the first 10-20 orbits.

Runs GT (Numba, fast) and HNN hinge4 side by side.
Reports: dt, T_moon, n_phys, max_mpd/rhill, STABLE/ESCAPED.
"""

import os, sys, json, pickle, time
import numpy as np
import torch

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from exomoon.constants          import FOUR_PI2, rsun as _rsun, merth, msun, au as AU_m
from exomoon.habitable_zone     import hz_bounds_au
from exomoon.params             import SystemParams
from exomoon.initial_conditions import initial_state
from exomoon.simulation         import run_simulation_for_years
from exomoon.ml.hnn_model_hill  import HNN_Hill
from exomoon.ml.hnn_dataset_hill import load_hill_sys_scaler, compute_hill_sys_enc

_MERTH_OVER_MSUN = merth / msun
_STEPS_PER_YEAR  = 20_000.0

HINGE4_DIR = os.path.join(SRC, "models_hnn_hill_hinge4")

# ── K-1229b system ────────────────────────────────────────────────────────────
SYS = dict(ms_solar=0.54, rs_solar=0.51, Ts=3784.0,
           mp_earth=2.54, ap_AU=0.3006, ep=0.0)
T_SIM     = 0.5      # yr — enough to detect escape within first ~135 moon orbits
MM_RES    = 50
AM_RES    = 50
TEST_AM_IDX = [3, 6]   # the inner cells never tested before
MM_IDX    = 12

PLANET_DENSITY_SI  = 5500.0
PLANET_DENSITY_CGS = 5.5
MOON_DENSITY_CGS   = 3.0

# ── Grid construction (identical to batch_leapfrog.py / eval_fair_trajectory.py) ─
ms_solar  = SYS["ms_solar"]; mp_earth = SYS["mp_earth"]
ap_AU     = SYS["ap_AU"];    ep       = SYS["ep"]

mp_msun_sys  = mp_earth * _MERTH_OVER_MSUN
rhill_AU = ap_AU * (1 - ep) * (mp_msun_sys / (3 * ms_solar)) ** (1/3)

mp_kg   = mp_earth * merth
rp_m    = (0.75 * mp_kg / (np.pi * PLANET_DENSITY_SI)) ** (1/3)
a_roche = 2.456 * rp_m * (PLANET_DENSITY_CGS / MOON_DENSITY_CGS) ** (1/3) / AU_m
am_min  = max(a_roche / rhill_AU, 1e-3)
mm_max  = min(mp_earth, 3.0)

mm_grid = np.exp(np.linspace(np.log(0.107), np.log(mm_max), MM_RES))
am_grid = np.linspace(am_min, 1.0, AM_RES)

a_inner_au, a_outer_au = hz_bounds_au(SYS["Ts"], SYS["rs_solar"] * _rsun)

print(f"K-1229b: rhill={rhill_AU:.5f} AU  am_min={am_min:.4f} Hill  mm_max={mm_max:.3f} Mearth")
print(f"HZ: [{a_inner_au:.4f}, {a_outer_au:.4f}] AU  (planet at ap={ap_AU} AU)")
print(f"t_sim={T_SIM} yr\n")

# ── Load HNN hinge4 ───────────────────────────────────────────────────────────
hnn_model  = HNN_Hill.load(HINGE4_DIR)
hnn_model.eval()
hnn_scaler = load_hill_sys_scaler(HINGE4_DIR)
print(f"Loaded HNN hinge4 from {HINGE4_DIR}\n")


# ── Hill-frame force (identical to eval_fair_trajectory._hnn_hill_accel) ──────

def _compute_hill_frame(r_s, r_p, r_m, v_s, v_p, v_m, rhill):
    r_sp = r_p - r_s
    r_sp_sq = float(np.dot(r_sp, r_sp))
    r_sp_norm = float(np.sqrt(r_sp_sq))
    x_hat = r_sp / max(r_sp_norm, 1e-10)
    v_sp  = v_p - v_s
    L_sp  = np.cross(r_sp, v_sp)
    L_norm = float(np.linalg.norm(L_sp))
    z_hat = L_sp / max(L_norm, 1e-10)
    Omega = L_norm / max(r_sp_sq, 1e-20)
    y_hat = np.cross(z_hat, x_hat)
    R = np.stack([x_hat, y_hat, z_hat], axis=1)
    r_mp  = r_m - r_p
    q_h_x = float(np.dot(r_mp, x_hat))
    q_h_y = float(np.dot(r_mp, y_hat))
    q_h_z = float(np.dot(r_mp, z_hat))
    q_h_AU   = np.array([q_h_x, q_h_y, q_h_z])
    q_h_norm = q_h_AU / max(rhill, 1e-20)
    return x_hat, y_hat, z_hat, q_h_norm, q_h_AU, Omega, R


def _hnn_hill_accel(r_s, r_p, r_m, v_s, v_p, v_m,
                    ms_solar, mp_msun, mm_msun, rhill, ap, V_ref):
    G = FOUR_PI2
    _, _, _, q_h_norm, q_h_AU, Omega, R = _compute_hill_frame(
        r_s, r_p, r_m, v_s, v_p, v_m, rhill)

    sys_raw = np.array([[ms_solar,
                         mp_msun / _MERTH_OVER_MSUN,
                         mm_msun / _MERTH_OVER_MSUN,
                         ap, rhill, Omega]])
    sys_enc = torch.tensor(
        hnn_scaler.transform(np.log(np.maximum(sys_raw, 1e-20))).astype(np.float32))

    q_t    = torch.tensor(q_h_norm.astype(np.float32)).unsqueeze(0)
    q_leaf = q_t.detach().requires_grad_(True)
    with torch.enable_grad():
        V     = hnn_model(q_leaf, sys_enc)
        grad  = torch.autograd.grad(V.sum(), q_leaf, create_graph=False)[0]
    grad_np = grad.detach().numpy()[0].astype(np.float64)

    a_cons_h = -grad_np * V_ref / (mm_msun * rhill)
    cf_h = Omega**2 * np.array([q_h_AU[0], q_h_AU[1], 0.0])
    a_rel_inertial = R @ (a_cons_h - cf_h)

    r_sp_v = r_p - r_s
    d_sp   = float(np.linalg.norm(r_sp_v))
    a_planet = (-G * ms_solar / d_sp**3) * r_sp_v
    a_m_inertial = a_planet + a_rel_inertial

    r_sm = r_m - r_s; d_sm = float(np.linalg.norm(r_sm))
    a_s  = (G * mp_msun / d_sp**3) * r_sp_v + (G * mm_msun / d_sm**3) * r_sm
    return a_s, a_planet, a_m_inertial


def _hill_calibrate_vel(st, ms_solar, mp_msun, mm_msun, rhill, ap, V_ref):
    r_s = st["pos_ms"].astype(np.float64); r_p = st["pos_mp"].astype(np.float64)
    r_m = st["pos_mm"].astype(np.float64)
    v_s = st["vel_ms"].copy().astype(np.float64)
    v_p = st["vel_mp"].copy().astype(np.float64)
    v_m = st["vel_mm"].copy().astype(np.float64)

    _, _, _, q_h_norm, q_h_AU, Omega, R = _compute_hill_frame(
        r_s, r_p, r_m, v_s, v_p, v_m, rhill)
    sys_raw = np.array([[ms_solar, mp_msun/_MERTH_OVER_MSUN,
                         mm_msun/_MERTH_OVER_MSUN, ap, rhill, Omega]])
    sys_enc = torch.tensor(
        hnn_scaler.transform(np.log(np.maximum(sys_raw, 1e-20))).astype(np.float32))
    q_t    = torch.tensor(q_h_norm.astype(np.float32)).unsqueeze(0)
    q_leaf = q_t.detach().requires_grad_(True)
    with torch.enable_grad():
        V    = hnn_model(q_leaf, sys_enc)
        grad = torch.autograd.grad(V.sum(), q_leaf, create_graph=False)[0]
    grad_np = grad.detach().numpy()[0].astype(np.float64)
    a_cons_h = -grad_np * V_ref / (mm_msun * rhill)
    a_mag    = float(np.linalg.norm(a_cons_h))
    r_pm_vec = r_m - r_p
    r_pm     = float(np.linalg.norm(r_pm_vec))
    if a_mag > 1e-30 and r_pm > 1e-30:
        v_circ = np.sqrt(r_pm * a_mag)
        v_rel  = v_m - v_p
        v_rel_mag = float(np.linalg.norm(v_rel))
        if v_rel_mag > 1e-30:
            v_m = v_p + v_circ * (v_rel / v_rel_mag)
    return v_s, v_p, v_m


def run_hnn(st, p_cell, t_sim, label):
    rhill = float(st["rhill_AU"]); am    = float(st["am_AU"])
    ap    = float(p_cell.ap_AU);   ms    = float(p_cell.ms_solar)
    mp    = float(p_cell.mp_earth * _MERTH_OVER_MSUN)
    mm    = float(p_cell.mm_earth * _MERTH_OVER_MSUN)

    G_pm   = FOUR_PI2 * (mp + mm)
    T_moon = 2 * np.pi * np.sqrt(am**3 / G_pm)
    dt_raw = min(T_moon / 100.0, 1.0 / _STEPS_PER_YEAR)
    n_phys = int(np.ceil(t_sim / dt_raw))
    dt     = t_sim / n_phys
    V_ref  = FOUR_PI2 * ms * mp / ap

    r_s = st["pos_ms"].copy().astype(np.float64)
    r_p = st["pos_mp"].copy().astype(np.float64)
    r_m = st["pos_mm"].copy().astype(np.float64)
    v_s, v_p, v_m = _hill_calibrate_vel(st, ms, mp, mm, rhill, ap, V_ref)

    max_mpd = float(np.linalg.norm(r_m - r_p))
    a_s, a_p, a_m = _hnn_hill_accel(r_s, r_p, r_m, v_s, v_p, v_m,
                                      ms, mp, mm, rhill, ap, V_ref)
    half_dt = 0.5 * dt
    t0 = time.time()
    for step in range(1, n_phys + 1):
        v_s += half_dt*a_s; v_p += half_dt*a_p; v_m += half_dt*a_m
        r_s += dt*v_s;      r_p += dt*v_p;      r_m += dt*v_m
        a_s, a_p, a_m = _hnn_hill_accel(r_s, r_p, r_m, v_s, v_p, v_m,
                                          ms, mp, mm, rhill, ap, V_ref)
        v_s += half_dt*a_s; v_p += half_dt*a_p; v_m += half_dt*a_m
        mpd_now = float(np.linalg.norm(r_m - r_p))
        if mpd_now > max_mpd:
            max_mpd = mpd_now

    elapsed = time.time() - t0
    stable = max_mpd <= rhill
    print(f"  HNN  {label}: T_moon={T_moon:.4e}yr  dt={dt:.3e}yr  n_phys={n_phys:,}"
          f"  max_mpd={max_mpd:.4e}AU  rhill={rhill:.4e}AU"
          f"  ratio={max_mpd/rhill:.3f}  {'STABLE  ' if stable else 'ESCAPED '}  ({elapsed:.1f}s)")
    return stable


def run_gt(st, p_cell, t_sim, label):
    t0 = time.time()
    sim = run_simulation_for_years(p_cell, t_sim)
    elapsed = time.time() - t0
    traj = sim["traj"]
    rhill = float(st["rhill_AU"])
    moon_rel = traj["xyzarr_mm"] - traj["xyzarr_mp"]
    mpd_arr  = np.linalg.norm(moon_rel[:, :2], axis=1)
    max_mpd  = float(np.max(mpd_arr))
    stable   = max_mpd <= rhill
    print(f"  GT   {label}: max_mpd={max_mpd:.4e}AU  rhill={rhill:.4e}AU"
          f"  ratio={max_mpd/rhill:.3f}  {'STABLE  ' if stable else 'ESCAPED '}  ({elapsed:.1f}s)")
    return stable


# ── Main diagnostic loop ──────────────────────────────────────────────────────

print("=" * 80)
print(f"  DIAGNOSTIC: K-1229b prograde inner cells — GT vs HNN hinge4")
print(f"  am_idx = {TEST_AM_IDX}  (out of AM_RES={AM_RES})  mm_idx={MM_IDX}")
print("=" * 80)

for am_idx in TEST_AM_IDX:
    mm_val = float(mm_grid[MM_IDX])
    am_val = float(am_grid[am_idx])
    am_AU  = am_val * rhill_AU

    mp_msun = mp_earth * _MERTH_OVER_MSUN
    mm_msun = mm_val   * _MERTH_OVER_MSUN
    G_pm    = FOUR_PI2 * (mp_msun + mm_msun)
    T_moon  = 2 * np.pi * np.sqrt(am_AU**3 / G_pm)
    orbits  = T_SIM / T_moon

    label = f"am_idx={am_idx}  am_hill={am_val:.4f}  mm={mm_val:.3f}Mearth  T_moon={T_moon:.4e}yr  (~{orbits:.0f} orbits)"
    print(f"\n  Cell: {label}")

    p_cell = SystemParams(
        Ts=SYS["Ts"], rs_solar=SYS["rs_solar"], ms_solar=SYS["ms_solar"],
        mp_earth=mp_earth, ap_AU=ap_AU, ep=ep,
        mm_earth=mm_val, am_hill=am_val, em=0.0, moon_retrograde=False,
    )
    st = initial_state(p_cell)

    run_gt(st, p_cell, T_SIM, label)
    run_hnn(st, p_cell, T_SIM, label)

print("\n" + "=" * 80)
print("  DONE")
print("=" * 80)
