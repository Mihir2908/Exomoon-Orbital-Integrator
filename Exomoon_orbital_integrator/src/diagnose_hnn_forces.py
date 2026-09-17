"""
diagnose_hnn_forces.py

At t=0 (initial conditions, before any integration), compares:
  - HNN predicted force/acceleration for the centre cell (mm_idx=25, am_idx=25)
  - Analytical Newtonian force/acceleration

This confirms whether the model is producing wrong forces, or there is a code bug.
"""

import os, sys, json
import numpy as np
import torch

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from exomoon.constants   import FOUR_PI2, au, merth, msun, rsun as _rsun
from exomoon.ml.hnn_dataset  import (load_sys_scaler, load_q_scaler, load_a_scaler,
                                      _MERTH_OVER_MSUN, _G, _analytical_forces)
from exomoon.ml.hnn_model    import HNN
from exomoon.ml.batch_leapfrog import _build_initial_states
from exomoon.habitable_zone  import hz_bounds_au

_PLANET_DENSITY_CGS = 5.5
_PLANET_DENSITY_SI  = 5500.0
_MOON_DENSITY_CGS   = 3.0

HNN_DIR  = os.path.join(SRC, "models_hnn_128x64_forces")
GT_META  = os.path.join(SRC, "ground_truth_grids", "Kepler_452_b_v2_prograde.meta.json")
MM_RES = AM_RES = 50
MM_IDX = AM_IDX = 25

with open(GT_META) as f:
    gt_meta = json.load(f)
sp = gt_meta["system_params"]
system_params = {k: float(sp[k]) for k in ["ms_solar", "rs_solar", "Ts", "mp_earth", "ap_AU", "ep"]}
T_SIM = float(gt_meta["t_sim"])

ms_solar = system_params["ms_solar"]
mp_earth = system_params["mp_earth"]
ap_AU    = system_params["ap_AU"]
ep       = system_params["ep"]

# ── Grid ──────────────────────────────────────────────────────────────────────
mp_kg   = mp_earth * merth
rp_m    = (0.75 * mp_kg / (np.pi * _PLANET_DENSITY_SI)) ** (1.0 / 3.0)
a_roche = 2.456 * rp_m * (_PLANET_DENSITY_CGS / _MOON_DENSITY_CGS) ** (1.0 / 3.0) / au

ms_gp = ms_solar * FOUR_PI2
mp_gp = mp_earth * _MERTH_OVER_MSUN * FOUR_PI2
rhill_AU = ap_AU * (1 - ep) * (mp_gp / (3 * ms_gp)) ** (1.0/3.0)

am_min = max(a_roche / rhill_AU, 1e-3)
mm_max = max(min(mp_earth * 0.30, 0.5), 0.108)

mm_grid = np.exp(np.linspace(np.log(0.107), np.log(mm_max), MM_RES))
am_grid = np.linspace(am_min, 1.0, AM_RES)

# ── Centre cell initial state ──────────────────────────────────────────────────
(pos_mp0, pos_ms0, pos_mm0, vel_mp0, vel_ms0, vel_mm0,
 mu_ms, mu_mp, mu_mm_arr, rhill) = _build_initial_states(
    ms_solar=ms_solar, mp_earth=mp_earth,
    ap_AU=ap_AU, ep=ep, em=0.0,
    moon_retrograde=False,
    mm_grid=mm_grid, am_grid=am_grid,
)

N        = MM_RES * AM_RES
cell_idx = MM_IDX * AM_RES + AM_IDX

mm_cell  = float(mm_grid[MM_IDX])
am_cell  = float(am_grid[AM_IDX])
am_AU    = am_cell * rhill_AU

# Positions for the centre cell
pos_s = pos_ms0[cell_idx]  # (3,)
pos_p = pos_mp0[cell_idx]
pos_m = pos_mm0[cell_idx]

print("=" * 68)
print(f"HNN force diagnostics — K-452b-v2 prograde, centre cell")
print(f"mm_earth={mm_cell:.5f} M⊕  am_hill={am_cell:.4f}  am_AU={am_AU:.6f} AU")
print(f"rhill_AU={rhill_AU:.5f} AU")
print("=" * 68)
print()
print(f"Initial positions (AU):")
print(f"  Star:   {pos_s}")
print(f"  Planet: {pos_p}")
print(f"  Moon:   {pos_m}")
print(f"  moon-planet dist: {np.linalg.norm(pos_m - pos_p):.6f} AU  (should be {am_AU:.6f} AU)")

# ── Analytical force ───────────────────────────────────────────────────────────
ms  = ms_solar
mp  = mp_earth * _MERTH_OVER_MSUN
mm  = mm_cell  * _MERTH_OVER_MSUN

q_single = np.stack([pos_s, pos_p, pos_m], axis=0).reshape(1, 9)  # (1, 9)
F_true   = _analytical_forces(q_single,
                               np.array([ms]), np.array([mp]), np.array([mm]))  # (1, 9)
F_true   = F_true[0]   # (9,)

a_true_s = F_true[0:3] / ms
a_true_p = F_true[3:6] / mp
a_true_m = F_true[6:9] / mm

print()
print("Analytical (Newtonian) forces  [M☉·AU/yr²]:")
print(f"  Star:   F={F_true[0:3]}  |F|={np.linalg.norm(F_true[0:3]):.6f}")
print(f"  Planet: F={F_true[3:6]}  |F|={np.linalg.norm(F_true[3:6]):.6f}")
print(f"  Moon:   F={F_true[6:9]}  |F|={np.linalg.norm(F_true[6:9]):.6f}")
print()
print("Analytical accelerations  [AU/yr²]:")
print(f"  Star:   a={a_true_s}  |a|={np.linalg.norm(a_true_s):.4f}")
print(f"  Planet: a={a_true_p}  |a|={np.linalg.norm(a_true_p):.4f}")
print(f"  Moon:   a={a_true_m}  |a|={np.linalg.norm(a_true_m):.4f}")

# ── HNN predicted force ────────────────────────────────────────────────────────
model      = HNN.load(HNN_DIR, map_location="cpu")
model.eval()
sys_scaler = load_sys_scaler(HNN_DIR)
q_scaler   = load_q_scaler(HNN_DIR)
a_scaler   = load_a_scaler(HNN_DIR)

a_mean = torch.tensor(a_scaler.mean_,  dtype=torch.float64)
a_std  = torch.tensor(a_scaler.scale_, dtype=torch.float64)

# Build sys_enc for just this cell
sys_raw = np.array([[ms_solar, mp_earth, mm_cell]])  # (1, 3) — raw Earth/solar mass units
sys_enc = torch.tensor(sys_scaler.transform(sys_raw).astype(np.float32))  # (1, 3)

# Normalise position
q_raw  = torch.tensor(q_single.astype(np.float32))   # (1, 9)
q_mean = torch.tensor(q_scaler.mean_,  dtype=torch.float32)
q_std_ = torch.tensor(q_scaler.scale_, dtype=torch.float32)
q_norm = (q_raw - q_mean) / q_std_                   # (1, 9)

# HNN gradient
F_scaled_hnn = model.accel(q_norm, sys_enc, create_graph=False)  # (1, 9) — -dV/dq_norm

# Unscale to real forces
F_hnn_real = F_scaled_hnn.to(torch.float64) * a_std + a_mean    # (1, 9)
F_hnn_real = F_hnn_real[0].detach().numpy()   # (9,)

a_hnn_s = F_hnn_real[0:3] / ms
a_hnn_p = F_hnn_real[3:6] / mp
a_hnn_m = F_hnn_real[6:9] / mm

print()
print("HNN predicted forces  [M☉·AU/yr²]:")
print(f"  Star:   F={F_hnn_real[0:3]}  |F|={np.linalg.norm(F_hnn_real[0:3]):.6f}")
print(f"  Planet: F={F_hnn_real[3:6]}  |F|={np.linalg.norm(F_hnn_real[3:6]):.6f}")
print(f"  Moon:   F={F_hnn_real[6:9]}  |F|={np.linalg.norm(F_hnn_real[6:9]):.6f}")
print()
print("HNN predicted accelerations  [AU/yr²]:")
print(f"  Star:   a={a_hnn_s}  |a|={np.linalg.norm(a_hnn_s):.4f}")
print(f"  Planet: a={a_hnn_p}  |a|={np.linalg.norm(a_hnn_p):.4f}")
print(f"  Moon:   a={a_hnn_m}  |a|={np.linalg.norm(a_hnn_m):.4f}")

print()
print("Force error (HNN - True):")
err = F_hnn_real - F_true
print(f"  Star:   ΔF={err[0:3]}  |ΔF|={np.linalg.norm(err[0:3]):.6f}  ({np.linalg.norm(err[0:3])/np.linalg.norm(F_true[0:3])*100:.1f}%)")
print(f"  Planet: ΔF={err[3:6]}  |ΔF|={np.linalg.norm(err[3:6]):.6f}  ({np.linalg.norm(err[3:6])/np.linalg.norm(F_true[3:6])*100:.1f}%)")
print(f"  Moon:   ΔF={err[6:9]}  |ΔF|={np.linalg.norm(err[6:9]):.6f}  ({np.linalg.norm(err[6:9])/np.linalg.norm(F_true[6:9])*100:.1f}%)")
print()
print("Acceleration error (HNN - True):")
da_s = a_hnn_s - a_true_s
da_p = a_hnn_p - a_true_p
da_m = a_hnn_m - a_true_m
print(f"  Star:   Δa={da_s}  |Δa|={np.linalg.norm(da_s):.2f} AU/yr²  ({np.linalg.norm(da_s)/np.linalg.norm(a_true_s)*100:.1f}%)")
print(f"  Planet: Δa={da_p}  |Δa|={np.linalg.norm(da_p):.2f} AU/yr²  ({np.linalg.norm(da_p)/np.linalg.norm(a_true_p)*100:.1f}%)")
print(f"  Moon:   Δa={da_m}  |Δa|={np.linalg.norm(da_m):.2f} AU/yr²  ({np.linalg.norm(da_m)/np.linalg.norm(a_true_m)*100:.1f}%)")

# ── One manual leapfrog step ───────────────────────────────────────────────────
# Reproduce exactly what hnn_inference does for step 0, to verify the initial pos change
dt = T_SIM / max(4000, 1000)   # same dt_factor=50 formula
half_dt = dt * 0.5
print()
print(f"Manual first leapfrog step (dt={dt:.5f} yr):")

def _unpack(F_s_t):
    F_real_t = F_s_t.to(torch.float64) * a_std + a_mean
    a_s_ = (F_real_t[0:3] / ms).numpy()
    a_p_ = (F_real_t[3:6] / mp).numpy()
    a_m_ = (F_real_t[6:9] / mm).numpy()
    return a_s_, a_p_, a_m_

# Initial velocities for this cell
v_s0 = vel_ms0[cell_idx]
v_p0 = vel_mp0[cell_idx]
v_m0 = vel_mm0[cell_idx]

a_s0, a_p0, a_m0 = a_hnn_s, a_hnn_p, a_hnn_m   # HNN initial acceleration

# Half-kick
v_s1 = v_s0 + a_s0 * half_dt
v_p1 = v_p0 + a_p0 * half_dt
v_m1 = v_m0 + a_m0 * half_dt

# Full drift
q_s1 = pos_s + v_s1 * dt
q_p1 = pos_p + v_p1 * dt
q_m1 = pos_m + v_m1 * dt

mpd_after_step = np.linalg.norm(q_m1 - q_p1)
print(f"  Initial mpd:          {np.linalg.norm(pos_m - pos_p):.6f} AU")
print(f"  mpd after step 0:     {mpd_after_step:.6f} AU")
print(f"  Expected (true acc):  ", end="")

# What would one step with TRUE accelerations give?
v_s1t = v_s0 + a_true_s * half_dt
v_p1t = v_p0 + a_true_p * half_dt
v_m1t = v_m0 + a_true_m * half_dt
q_s1t = pos_s + v_s1t * dt
q_p1t = pos_p + v_p1t * dt
q_m1t = pos_m + v_m1t * dt
mpd_true_step = np.linalg.norm(q_m1t - q_p1t)
print(f"{mpd_true_step:.6f} AU")
print(f"  Ratio HNN/True:       {mpd_after_step/mpd_true_step:.4f}")
