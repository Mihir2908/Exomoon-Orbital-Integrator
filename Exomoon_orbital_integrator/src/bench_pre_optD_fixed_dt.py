"""
bench_pre_optD_fixed_dt.py

Pre-Option-D batch benchmark — corrected mm_max, fixed dt=5e-5 yr for all cells.

Three corrections vs the Aug 27 buggy benchmark:
  1. mm_max = min(mp_earth, 3.0) for BOTH GT and HNN  — same grid (mass bug fix)
  2. AuxMLPBinary eligibility mask applied
  3. BOTH stopping criterion: stopped when unstable AND uninhabitable (not time-limit only)

Timestep: fixed dt = 5e-5 yr for ALL 2500 cells.  No per-cell K logic.
System:   Kepler-1229b prograde
t_sim:    10.0 yr   |   n_phys: 200,000 steps   |   n_output_frames: 1000
"""

import json, os, pickle, sys, time
import numpy as np
import torch
import torch.nn as nn

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from exomoon.constants           import FOUR_PI2, au, merth, msun
from exomoon.constants           import rsun as _rsun
from exomoon.habitable_zone      import hz_bounds_au
from exomoon.ml.batch_leapfrog   import _build_initial_states, _accel_batch
from exomoon.ml.dataset          import SYS_COLS, LOG_SYS_COLS
from exomoon.ml.hnn_model_hill   import HNN_Hill, is_hill_dir
from exomoon.ml.hnn_dataset_hill import load_hill_sys_scaler, _MERTH_OVER_MSUN

_PLANET_DENSITY_CGS = 5.5
_PLANET_DENSITY_SI  = 5500.0
_MOON_DENSITY_CGS   = 3.0

# ── System parameters ──────────────────────────────────────────────────────────
META_PATH = os.path.join(SRC, "ground_truth_grids", "Kepler_1229_b_prograde.meta.json")
meta     = json.load(open(META_PATH))
SP       = meta["system_params"]
ms_solar = float(SP["ms_solar"])    # 0.54
rs_solar = float(SP["rs_solar"])    # 0.51
Ts       = float(SP["Ts"])          # 3784.0
mp_earth = float(SP["mp_earth"])    # 2.54
ap_AU    = float(SP["ap_AU"])       # 0.3006
ep       = float(SP.get("ep", 0.0))

# ── Grid — identical for GT and HNN (mass bug fixed) ──────────────────────────
MM_RES, AM_RES = 50, 50
N = MM_RES * AM_RES

ms_gp     = ms_solar * FOUR_PI2
mp_gp_tmp = mp_earth * (merth / msun) * FOUR_PI2
rhill_AU  = ap_AU * (1.0 - ep) * (mp_gp_tmp / (3.0 * ms_gp)) ** (1.0 / 3.0)

mp_kg   = mp_earth * merth
rp_m    = (0.75 * mp_kg / (np.pi * _PLANET_DENSITY_SI)) ** (1.0 / 3.0)
a_roche = 2.456 * rp_m * (_PLANET_DENSITY_CGS / _MOON_DENSITY_CGS) ** (1.0 / 3.0) / au
am_min  = max(a_roche / rhill_AU, 1e-3)
mm_max  = min(mp_earth, 3.0)          # FIXED: was min(mp*0.30, 0.5) for GT in Aug 27 run

mm_grid = np.exp(np.linspace(np.log(0.107), np.log(mm_max), MM_RES))
am_grid = np.linspace(am_min, 1.0, AM_RES)

a_inner_au, a_outer_au = hz_bounds_au(Ts, rs_solar * _rsun)

DT_PHYS = 5e-5        # fixed dt for ALL cells — no per-cell K
T_SIM   = 10.0        # years
N_STEPS = 1000        # number of stored output frames
N_PHYS  = max(int(np.ceil(T_SIM / DT_PHYS)), N_STEPS)   # 200,000
STRIDE  = max(1, N_PHYS // N_STEPS)
N_OUT   = N_PHYS // STRIDE
WARMUP  = 5            # steps excluded from stability evaluation

print("=" * 72)
print("  BENCH: Kepler-1229b prograde — Pre-Option-D fixed dt=5e-5 yr")
print(f"  System: ms={ms_solar} M☉  rs={rs_solar} R☉  Ts={Ts} K  "
      f"mp={mp_earth} M⊕  ap={ap_AU} AU  ep={ep}")
print(f"  rhill={rhill_AU:.5f} AU  a_inner={a_inner_au:.4f} AU  a_outer={a_outer_au:.4f} AU")
print(f"  Grid: {MM_RES}x{AM_RES}={N} cells")
print(f"  mm_grid: [{mm_grid[0]:.4f}, {mm_max:.4f}] M_earth (log-spaced)")
print(f"  am_grid: [{am_min:.4f}, {am_grid[-1]:.4f}] Hill frac (linear)")
print(f"  dt={DT_PHYS:.0e} yr  t_sim={T_SIM} yr  n_phys={N_PHYS:,}")
print(f"  stride={STRIDE}  n_output_frames={N_OUT}")
print("=" * 72)

# ── AuxMLPBinary eligibility mask ─────────────────────────────────────────────
MLP_DIR = os.path.join(SRC, "eval_aux_mlp_output", "binary")

class AuxMLPBinary(nn.Module):
    def __init__(self, input_dim=14, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, 2),
        )
    def forward(self, x): return self.net(x)

print("\n[MLP] Building eligibility mask ...")
_cfg = json.load(open(os.path.join(MLP_DIR, "model_config.json")))
_mlp = AuxMLPBinary(_cfg["input_dim"], _cfg["hidden"])
_mlp.load_state_dict(torch.load(os.path.join(MLP_DIR, "aux_mlp_binary.pt"),
                                 map_location="cpu", weights_only=True))
_mlp.eval()
_mlp_scl = pickle.load(open(os.path.join(MLP_DIR, "aux_mlp_scaler.pkl"), "rb"))

mm_c = np.repeat(mm_grid, AM_RES)
am_c = np.tile(am_grid, MM_RES)
X_mlp = np.column_stack([
    np.full(N, ms_solar), np.full(N, rs_solar),
    np.full(N, Ts),       np.full(N, mp_earth),
    np.full(N, ap_AU),    np.full(N, ep),
    mm_c, am_c,
    np.zeros(N), np.zeros(N),          # em=0, moon_retrograde=0
    np.full(N, T_SIM),                 # t_sim feature
    np.full(N, rhill_AU),
    np.full(N, a_inner_au),
    np.full(N, a_outer_au),
]).astype(np.float32)

_log_idx = [SYS_COLS.index(c) for c in LOG_SYS_COLS if c in SYS_COLS]
X_mlp[:, _log_idx] = np.log(np.clip(X_mlp[:, _log_idx], 1e-10, None))
Xs_mlp = _mlp_scl.transform(X_mlp).astype(np.float32)
with torch.no_grad():
    _logits = _mlp(torch.tensor(Xs_mlp))
_preds  = (_logits > 0).numpy()
eligible = _preds[:, 0] & _preds[:, 1]   # stable AND habitable both predicted True
n_elig   = int(eligible.sum())
print(f"  eligible: {n_elig}/{N} ({100*n_elig/N:.1f}%)")

# ── Common tensors ─────────────────────────────────────────────────────────────
F64        = torch.float64
rhill_tf   = torch.tensor(rhill_AU,   dtype=F64)
a_inner_tf = torch.tensor(a_inner_au, dtype=F64)
a_outer_tf = torch.tensor(a_outer_au, dtype=F64)

inelig_t = torch.tensor(~eligible.reshape(N, 1), dtype=torch.bool)
dt_t      = torch.full((N, 1), DT_PHYS, dtype=F64)   # same dt for all cells
half_dt_t = dt_t * 0.5


def _compute_maps(mpd_arr, msd_arr, n_out):
    """
    Classify each cell from (N, n_out) float32 distance arrays.
    Warm-up window excluded.  Eligible mask applied.
    Returns stable_2d, habitable_2d, both_2d (MM_RES, AM_RES bool) + valid_mm_range.
    """
    w        = min(WARMUP, n_out - 1)
    mpd_post = mpd_arr[:, w:]
    msd_post = msd_arr[:, w:]

    stable    = mpd_post.max(axis=1) <= rhill_AU
    habitable = ((msd_post.min(axis=1) >= a_inner_au) &
                 (msd_post.max(axis=1) <= a_outer_au))
    both = stable & habitable

    stable    &= eligible
    habitable &= eligible
    both      &= eligible

    stable_2d    = stable.reshape(MM_RES, AM_RES)
    habitable_2d = habitable.reshape(MM_RES, AM_RES)
    both_2d      = both.reshape(MM_RES, AM_RES)

    valid_rows     = np.where(both_2d.any(axis=1))[0]
    valid_mm_range = (
        [float(mm_grid[valid_rows[0]]), float(mm_grid[valid_rows[-1]])]
        if len(valid_rows) else None
    )
    return stable_2d, habitable_2d, both_2d, valid_mm_range


def _print_result_block(label, stable_2d, hab_2d, both_2d, valid_mm, elapsed, actual_steps):
    print(f"\n  [{label}] elapsed={elapsed:.1f} s  actual_steps={actual_steps:,}")
    print(f"  stable={stable_2d.mean():.4f}  habitable={hab_2d.mean():.4f}  both={both_2d.mean():.4f}")
    print(f"  valid_mm_range: {valid_mm}")
    valid_rows = np.where(both_2d.any(axis=1))[0]
    valid_am   = np.where(both_2d.any(axis=0))[0]
    if len(valid_rows):
        print(f"  mm rows with >=1 both cell: {len(valid_rows)}  "
              f"mm=[{mm_grid[valid_rows[0]]:.3f}, {mm_grid[valid_rows[-1]]:.3f}] M_earth")
    if len(valid_am):
        print(f"  am cols with >=1 both cell: {len(valid_am)}  "
              f"am=[{am_grid[valid_am[0]]:.3f}, {am_grid[valid_am[-1]]:.3f}] Hill frac")
    print(f"  Per-am_hill stable fraction (avg over mm dim):")
    am_stab = stable_2d.mean(axis=0)
    for j in range(AM_RES):
        bar = "#" * int(am_stab[j] * 20)
        print(f"    am_hill={am_grid[j]:.3f}  {am_stab[j]:.3f}  {bar}")


# ══════════════════════════════════════════════════════════════════════════════
# A) GT BATCH — fixed dt=5e-5 yr
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print(f"A) GT batch leapfrog  (fixed dt={DT_PHYS:.0e} yr, t_sim={T_SIM} yr)")
print(f"   n_phys={N_PHYS:,}  all {N} cells, {n_elig} eligible")

(pos_mp0, pos_ms0, pos_mm0, vel_mp0, vel_ms0, vel_mm0,
 mu_ms, mu_mp, mu_mm_arr, _rh) = _build_initial_states(
    ms_solar=ms_solar, mp_earth=mp_earth,
    ap_AU=ap_AU, ep=ep, em=0.0,
    moon_retrograde=False,
    mm_grid=mm_grid, am_grid=am_grid,
)

p_mp = torch.tensor(pos_mp0, dtype=F64)
p_ms = torch.tensor(pos_ms0, dtype=F64)
p_mm = torch.tensor(pos_mm0, dtype=F64)
v_mp = torch.tensor(vel_mp0, dtype=F64)
v_ms = torch.tensor(vel_ms0, dtype=F64)
v_mm = torch.tensor(vel_mm0, dtype=F64)
mu_mm_t = torch.tensor(mu_mm_arr[:, None], dtype=F64)

mpd_gt_buf = np.empty((N, N_OUT), dtype=np.float32)
msd_gt_buf = np.empty((N, N_OUT), dtype=np.float32)

t0_gt = time.perf_counter()
with torch.no_grad():
    elapsed_t  = torch.zeros(N, 1, dtype=F64)
    done_t     = inelig_t.clone()
    out_idx_gt = 0
    steps_gt   = 0
    for i in range(N_PHYS):
        steps_gt = i + 1
        active = (~done_t).to(F64)

        # Planet (KDK leapfrog)
        p2_mp = p_mp + v_mp * (half_dt_t * active)
        a_mp  = (_accel_batch(p2_mp, p_ms, mu_ms)
               + _accel_batch(p2_mp, p_mm, mu_mm_t))
        v_mp  = v_mp + a_mp * (dt_t * active)
        p_mp  = p2_mp + v_mp * (half_dt_t * active)

        # Star
        p2_ms = p_ms + v_ms * (half_dt_t * active)
        a_ms  = (_accel_batch(p2_ms, p_mp, mu_mp)
               + _accel_batch(p2_ms, p_mm, mu_mm_t))
        v_ms  = v_ms + a_ms * (dt_t * active)
        p_ms  = p2_ms + v_ms * (half_dt_t * active)

        # Moon
        p2_mm = p_mm + v_mm * (half_dt_t * active)
        a_mm  = (_accel_batch(p2_mm, p_mp, mu_mp)
               + _accel_batch(p2_mm, p_ms, mu_ms))
        v_mm  = v_mm + a_mm * (dt_t * active)
        p_mm  = p2_mm + v_mm * (half_dt_t * active)

        # Distances at every step (needed for stopping + output)
        diff_mp = p_mm - p_mp
        diff_ms = p_mm - p_ms
        mpd_s   = (diff_mp * diff_mp).sum(dim=1, keepdim=True).sqrt()
        msd_s   = (diff_ms * diff_ms).sum(dim=1, keepdim=True).sqrt()

        if i % STRIDE == 0 and out_idx_gt < N_OUT:
            mpd_gt_buf[:, out_idx_gt] = mpd_s.squeeze(1).float().numpy()
            msd_gt_buf[:, out_idx_gt] = msd_s.squeeze(1).float().numpy()
            out_idx_gt += 1

        elapsed_t += dt_t * active
        unstable      = mpd_s > rhill_tf
        uninhabitable = (msd_s < a_inner_tf) | (msd_s > a_outer_tf)
        done_t = done_t | (unstable & uninhabitable) | (elapsed_t >= T_SIM)
        if done_t.all():
            break

gt_elapsed = time.perf_counter() - t0_gt
mpd_gt = mpd_gt_buf[:, :out_idx_gt]
msd_gt = msd_gt_buf[:, :out_idx_gt]
gt_stable, gt_hab, gt_both, gt_vmm = _compute_maps(mpd_gt, msd_gt, out_idx_gt)
_print_result_block("GT", gt_stable, gt_hab, gt_both, gt_vmm, gt_elapsed, steps_gt)


# ══════════════════════════════════════════════════════════════════════════════
# B) HNN HINGE4 BATCH — fixed dt=5e-5 yr
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print(f"B) HNN hinge4 batch  (fixed dt={DT_PHYS:.0e} yr, t_sim={T_SIM} yr)")
print(f"   n_phys={N_PHYS:,}  all {N} cells, {n_elig} eligible")
print(f"   NOTE: HNN was trained at dt≈0.01yr/step; fixed dt=5e-5 is ~200x finer")
print(f"   than training resolution. Forces are smooth so integration is valid,")
print(f"   but out-of-distribution dt may affect HNN trajectory accuracy.")

HNN_DIR = os.path.join(SRC, "models_hnn_hill_hinge4")
assert is_hill_dir(HNN_DIR), f"HNN checkpoint not found at {HNN_DIR}"
hill_model  = HNN_Hill.load(HNN_DIR)
hill_model.eval()
hill_scaler = load_hill_sys_scaler(HNN_DIR)

mp_msun     = mp_earth * _MERTH_OVER_MSUN
ms_gp_hnn   = ms_solar * FOUR_PI2
mp_gp_hnn   = mp_msun  * FOUR_PI2

mm_per_cell  = np.repeat(mm_grid, AM_RES)          # (N,) M_earth
mm_msun_arr  = mm_per_cell * _MERTH_OVER_MSUN       # (N,) M_sun
V_ref_val    = FOUR_PI2 * ms_solar * mp_msun / ap_AU  # dominant energy scale

# Static portion of sys_enc: [ms_solar, mp_earth, mm_earth, ap_AU, rhill_AU]
_sys_static  = np.column_stack([
    np.full(N, ms_solar),
    np.full(N, mp_earth),
    mm_per_cell,
    np.full(N, ap_AU),
    np.full(N, rhill_AU),
])   # (N, 5) — Omega_yr appended per-step inside _hill_forces

F32 = torch.float32


def _hill_forces(q_ms_t, q_mp_t, q_mm_t, v_ms_t, v_mp_t):
    """
    Compute Hill-frame HNN forces for all N cells simultaneously.
    Returns (a_s, a_p, a_m) as F64 tensors (N, 3).
    """
    # Per-cell star-planet displacement and velocity
    r_sp_np  = (q_mp_t - q_ms_t).cpu().numpy()          # (N, 3) AU
    v_sp_np  = (v_mp_t - v_ms_t).cpu().numpy()          # (N, 3) AU/yr

    r_sp_norm = np.maximum(np.linalg.norm(r_sp_np, axis=1, keepdims=True), 1e-10)  # (N,1)
    r_sp_sq   = (r_sp_norm ** 2).squeeze(1)              # (N,)

    x_hat = r_sp_np / r_sp_norm                          # (N, 3)

    L_sp   = np.cross(r_sp_np, v_sp_np)                  # (N, 3)
    L_norm = np.maximum(np.linalg.norm(L_sp, axis=1, keepdims=True), 1e-10)  # (N,1)
    z_hat  = L_sp / L_norm                               # (N, 3)
    Omega  = L_norm.squeeze(1) / np.maximum(r_sp_sq, 1e-20)   # (N,) rad/yr
    y_hat  = np.cross(z_hat, x_hat)                      # (N, 3)

    # Moon position in per-cell Hill frame (dot product per row)
    r_mp_np  = (q_mm_t - q_mp_t).cpu().numpy()           # (N, 3) AU
    q_h_x    = (r_mp_np * x_hat).sum(axis=1)             # (N,)
    q_h_y    = (r_mp_np * y_hat).sum(axis=1)
    q_h_z    = (r_mp_np * z_hat).sum(axis=1)
    q_h_AU   = np.stack([q_h_x, q_h_y, q_h_z], axis=1)  # (N, 3) AU
    q_h_norm = q_h_AU / max(rhill_AU, 1e-20)             # (N, 3) dimensionless

    # sys_enc with per-step Omega
    sys_raw = np.column_stack([_sys_static, Omega])       # (N, 6)
    sys_log = np.log(np.maximum(sys_raw, 1e-20))
    sys_enc = torch.tensor(
        hill_scaler.transform(sys_log).astype(np.float32), dtype=F32
    )   # (N, 6)

    # HNN_Hill forward + autograd: V_theta(q_h_norm, sys_enc) -> (N,)
    q_h_leaf = torch.tensor(
        q_h_norm.astype(np.float32), dtype=F32
    ).detach().requires_grad_(True)

    with torch.enable_grad():
        V_theta = hill_model(q_h_leaf, sys_enc)           # (N,)
        grad    = torch.autograd.grad(
            V_theta.sum(), q_h_leaf, create_graph=False
        )[0]                                               # (N, 3) dV/dq_h_norm

    grad_np = grad.detach().cpu().numpy().astype(np.float64)

    # Conservative Hill acceleration: a_cons_h = -grad * V_ref / (mm_msun * rhill)
    a_cons_h = -grad_np * V_ref_val / (mm_msun_arr[:, None] * rhill_AU)  # (N,3) AU/yr²

    # Strip centrifugal (per-cell Omega)
    cf = (Omega[:, None] ** 2 *
          np.column_stack([q_h_AU[:, 0], q_h_AU[:, 1], np.zeros(N)]))    # (N,3)
    a_hill = a_cons_h - cf                                # (N,3) in Hill frame

    # Rotate to inertial
    a_rel_in = (a_hill[:, 0:1] * x_hat +
                a_hill[:, 1:2] * y_hat +
                a_hill[:, 2:3] * z_hat)                   # (N,3) AU/yr²

    # Planet acceleration from star (per-cell)
    a_p_arr = -FOUR_PI2 * ms_solar * r_sp_np / (r_sp_sq ** 1.5)[:, None]  # (N,3)

    # Moon inertial acceleration
    a_m = a_rel_in + a_p_arr                              # (N,3)

    # Star acceleration: from moon (per-cell mm_msun) + from planet
    r_sm_np = (q_mm_t - q_ms_t).cpu().numpy()            # (N,3)
    d_sm_cu = (r_sm_np ** 2).sum(axis=1, keepdims=True) ** 1.5  # (N,1)
    a_s     = (FOUR_PI2 * mm_msun_arr[:, None] / d_sm_cu * r_sm_np +
               FOUR_PI2 * mp_msun * r_sp_np / (r_sp_sq ** 1.5)[:, None])  # (N,3)

    return (
        torch.tensor(a_s,     dtype=F64),
        torch.tensor(a_p_arr, dtype=F64),
        torch.tensor(a_m,     dtype=F64),
    )


# ── Initial velocity calibration ───────────────────────────────────────────────
# Rescale v_mm so initial moon speed matches HNN circular speed — mirrors
# _hill_calibrate_init_vel from eval_fair_trajectory.py.
print("  [HNN] Calibrating initial velocities ...")

(pos_mp0h, pos_ms0h, pos_mm0h, vel_mp0h, vel_ms0h, vel_mm0h,
 _mu_ms_h, _mu_mp_h, _mu_mm_arr_h, _) = _build_initial_states(
    ms_solar=ms_solar, mp_earth=mp_earth,
    ap_AU=ap_AU, ep=ep, em=0.0,
    moon_retrograde=False,
    mm_grid=mm_grid, am_grid=am_grid,
)

q_ms = torch.tensor(pos_ms0h, dtype=F64)
q_mp = torch.tensor(pos_mp0h, dtype=F64)
q_mm = torch.tensor(pos_mm0h, dtype=F64)
v_ms = torch.tensor(vel_ms0h, dtype=F64)
v_mp = torch.tensor(vel_mp0h, dtype=F64)
v_mm = torch.tensor(vel_mm0h, dtype=F64)

r_sp0  = (q_mp - q_ms).cpu().numpy()
v_sp0  = (v_mp - v_ms).cpu().numpy()
r_sp_n0 = np.maximum(np.linalg.norm(r_sp0, axis=1, keepdims=True), 1e-10)
r_sp_sq0 = (r_sp_n0 ** 2).squeeze(1)
x_h0 = r_sp0 / r_sp_n0
L0   = np.cross(r_sp0, v_sp0)
L_n0 = np.maximum(np.linalg.norm(L0, axis=1, keepdims=True), 1e-10)
z_h0 = L0 / L_n0
y_h0 = np.cross(z_h0, x_h0)
Omega0 = L_n0.squeeze(1) / np.maximum(r_sp_sq0, 1e-20)

r_mp0   = (q_mm - q_mp).cpu().numpy()
q_h_AU0 = np.stack([(r_mp0 * x_h0).sum(1),
                     (r_mp0 * y_h0).sum(1),
                     (r_mp0 * z_h0).sum(1)], axis=1)
q_h_n0  = q_h_AU0 / max(rhill_AU, 1e-20)

sys_r0 = np.column_stack([_sys_static, Omega0])
sys_e0 = torch.tensor(
    hill_scaler.transform(np.log(np.maximum(sys_r0, 1e-20))).astype(np.float32),
    dtype=F32,
)
q_lf0 = torch.tensor(q_h_n0.astype(np.float32), dtype=F32).detach().requires_grad_(True)
with torch.enable_grad():
    V0    = hill_model(q_lf0, sys_e0)
    grad0 = torch.autograd.grad(V0.sum(), q_lf0, create_graph=False)[0]
g0_np      = grad0.detach().cpu().numpy().astype(np.float64)
a_ch0      = -g0_np * V_ref_val / (mm_msun_arr[:, None] * rhill_AU)   # (N,3) Hill
ach_mag0   = np.linalg.norm(a_ch0, axis=1)                             # (N,)
r_pm0      = np.linalg.norm(r_mp0, axis=1)                             # (N,)
v_circ0    = np.where((ach_mag0 > 1e-30) & (r_pm0 > 1e-30),
                      np.sqrt(r_pm0 * ach_mag0), 0.0)                  # (N,)
v_rel0     = (v_mm - v_mp).cpu().numpy()                               # (N,3)
v_rel0_mag = np.linalg.norm(v_rel0, axis=1)                           # (N,)
ok_cal     = (v_circ0 > 1e-30) & (v_rel0_mag > 1e-30)
v_rel0_dir = np.where(ok_cal[:, None], v_rel0 / v_rel0_mag[:, None], v_rel0)
v_mm_np0   = v_mm.cpu().numpy().copy()
v_mm_np0[ok_cal] = (v_mp.cpu().numpy() + v_circ0[:, None] * v_rel0_dir)[ok_cal]
v_mm = torch.tensor(v_mm_np0, dtype=F64)
print(f"  [HNN] Calibrated {ok_cal.sum()}/{N} cells")

# ── Initial HNN forces (before main loop — VV / Velocity-Verlet scheme) ────────
print("  [HNN] Computing initial forces ...")
a_s_h, a_p_h, a_m_h = _hill_forces(q_ms, q_mp, q_mm, v_ms, v_mp)

# ── HNN KDK (Velocity-Verlet) leapfrog loop ────────────────────────────────────
mpd_hnn_buf = np.empty((N, N_OUT), dtype=np.float32)
msd_hnn_buf = np.empty((N, N_OUT), dtype=np.float32)

a_inner_np = float(a_inner_au)
a_outer_np = float(a_outer_au)
rhill_np   = float(rhill_AU)

t0_hnn = time.perf_counter()
with torch.no_grad():
    elapsed_hnn  = torch.zeros(N, 1, dtype=F64)
    stopped      = inelig_t.clone()
    out_idx_hnn  = 0
    steps_hnn    = 0
    for step in range(N_PHYS):
        steps_hnn = step + 1
        mask = (~stopped).to(F64)   # (N,1)

        # Half kick
        v_ms = v_ms + a_s_h * (half_dt_t * mask)
        v_mp = v_mp + a_p_h * (half_dt_t * mask)
        v_mm = v_mm + a_m_h * (half_dt_t * mask)

        # Full drift
        q_ms = q_ms + v_ms * (dt_t * mask)
        q_mp = q_mp + v_mp * (dt_t * mask)
        q_mm = q_mm + v_mm * (dt_t * mask)

        # New forces at drifted positions
        a_s_h, a_p_h, a_m_h = _hill_forces(q_ms, q_mp, q_mm, v_ms, v_mp)

        # Half kick
        v_ms = v_ms + a_s_h * (half_dt_t * mask)
        v_mp = v_mp + a_p_h * (half_dt_t * mask)
        v_mm = v_mm + a_m_h * (half_dt_t * mask)

        # Distances
        diff_mp_h = q_mm - q_mp
        diff_ms_h = q_mm - q_ms
        mpd_col   = (diff_mp_h * diff_mp_h).sum(1, keepdim=True).sqrt()   # (N,1) F64
        msd_col   = (diff_ms_h * diff_ms_h).sum(1, keepdim=True).sqrt()

        if step % STRIDE == 0 and out_idx_hnn < N_OUT:
            mpd_hnn_buf[:, out_idx_hnn] = mpd_col.squeeze(1).float().numpy()
            msd_hnn_buf[:, out_idx_hnn] = msd_col.squeeze(1).float().numpy()
            out_idx_hnn += 1

        elapsed_hnn  += dt_t * mask
        unstable_h    = mpd_col > rhill_tf
        uninhab_h     = (msd_col < a_inner_tf) | (msd_col > a_outer_tf)
        stopped = stopped | (unstable_h & uninhab_h) | (elapsed_hnn >= T_SIM)
        if stopped.all():
            break

hnn_elapsed = time.perf_counter() - t0_hnn
mpd_hnn = mpd_hnn_buf[:, :out_idx_hnn]
msd_hnn = msd_hnn_buf[:, :out_idx_hnn]
hnn_stable, hnn_hab, hnn_both, hnn_vmm = _compute_maps(mpd_hnn, msd_hnn, out_idx_hnn)
_print_result_block("HNN", hnn_stable, hnn_hab, hnn_both, hnn_vmm, hnn_elapsed, steps_hnn)


# ══════════════════════════════════════════════════════════════════════════════
# C) COMPREHENSIVE COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("C) COMPREHENSIVE COMPARISON: GT vs HNN hinge4")
print("=" * 72)

# Speed
ratio = hnn_elapsed / gt_elapsed
print(f"\n  TIMING:")
print(f"    GT  elapsed : {gt_elapsed:.1f} s  ({gt_elapsed/60:.1f} min)  "
      f"actual_steps={steps_gt:,}  "
      f"ms/step={1000*gt_elapsed/steps_gt:.2f}")
print(f"    HNN elapsed : {hnn_elapsed:.1f} s  ({hnn_elapsed/60:.1f} min)  "
      f"actual_steps={steps_hnn:,}  "
      f"ms/step={1000*hnn_elapsed/steps_hnn:.2f}")
print(f"    Speed ratio : HNN/GT = {ratio:.2f}x")

# Fractions
print(f"\n  FRACTIONS (over full {MM_RES}x{AM_RES} grid):")
print(f"    {'Metric':<12} {'GT':>8} {'HNN':>8} {'Diff':>8}")
print(f"    {'stable':<12} {gt_stable.mean():>8.4f} {hnn_stable.mean():>8.4f} "
      f"{hnn_stable.mean()-gt_stable.mean():>+8.4f}")
print(f"    {'habitable':<12} {gt_hab.mean():>8.4f} {hnn_hab.mean():>8.4f} "
      f"{hnn_hab.mean()-gt_hab.mean():>+8.4f}")
print(f"    {'both':<12} {gt_both.mean():>8.4f} {hnn_both.mean():>8.4f} "
      f"{hnn_both.mean()-gt_both.mean():>+8.4f}")

# Confusion matrices (eligible cells only, flattened)
gt_s_flat  = gt_stable.ravel()   & eligible
hnn_s_flat = hnn_stable.ravel()  & eligible
gt_h_flat  = gt_hab.ravel()      & eligible
hnn_h_flat = hnn_hab.ravel()     & eligible
gt_b_flat  = gt_both.ravel()     & eligible
hnn_b_flat = hnn_both.ravel()    & eligible

def _confusion(gt_flag, hnn_flag, elig, label):
    gt  = gt_flag  & elig
    hnn = hnn_flag & elig
    TP = int((gt  &  hnn).sum())
    FN = int((gt  & ~hnn).sum())
    FP = int((~gt &  hnn).sum())
    TN = int((~gt & ~hnn).sum())
    prec   = TP / (TP + FP) if (TP + FP) > 0 else float("nan")
    recall = TP / (TP + FN) if (TP + FN) > 0 else float("nan")
    f1     = (2*prec*recall/(prec+recall)) if (prec+recall) > 0 else float("nan")
    print(f"\n  Confusion [{label}] (eligible cells only, n_elig={elig.sum()}):")
    print(f"    GT=T  HNN=T (TP): {TP:4d}    GT=T  HNN=F (FN): {FN:4d}")
    print(f"    GT=F  HNN=T (FP): {FP:4d}    GT=F  HNN=F (TN): {TN:4d}")
    print(f"    Precision={prec:.3f}  Recall={recall:.3f}  F1={f1:.3f}")

_confusion(gt_stable.ravel(), hnn_stable.ravel(), eligible, "STABLE")
_confusion(gt_hab.ravel(),    hnn_hab.ravel(),    eligible, "HABITABLE")
_confusion(gt_both.ravel(),   hnn_both.ravel(),   eligible, "BOTH")

# Side-by-side per-am_hill stable fractions
print(f"\n  SIDE-BY-SIDE — per-am_hill stable fraction (avg over mm dim):")
print(f"    {'am_hill':>8}  {'GT_stab':>8}  {'HNN_stab':>9}  {'Diff':>7}  bar_GT / bar_HNN")
gt_am_s  = gt_stable.mean(axis=0)
hnn_am_s = hnn_stable.mean(axis=0)
for j in range(AM_RES):
    bar_gt  = "#" * int(gt_am_s[j]  * 20)
    bar_hnn = "#" * int(hnn_am_s[j] * 20)
    diff = hnn_am_s[j] - gt_am_s[j]
    print(f"    {am_grid[j]:>8.3f}  {gt_am_s[j]:>8.3f}  {hnn_am_s[j]:>9.3f}  "
          f"{diff:>+7.3f}  {bar_gt:<20} / {bar_hnn}")

# Side-by-side per-am_hill habitable fractions
print(f"\n  SIDE-BY-SIDE — per-am_hill habitable fraction (avg over mm dim):")
print(f"    {'am_hill':>8}  {'GT_hab':>8}  {'HNN_hab':>9}  {'Diff':>7}")
gt_am_h  = gt_hab.mean(axis=0)
hnn_am_h = hnn_hab.mean(axis=0)
for j in range(AM_RES):
    diff = hnn_am_h[j] - gt_am_h[j]
    print(f"    {am_grid[j]:>8.3f}  {gt_am_h[j]:>8.3f}  {hnn_am_h[j]:>9.3f}  {diff:>+7.3f}")

print("\n" + "=" * 72)
print("  DONE.")
print("=" * 72)
