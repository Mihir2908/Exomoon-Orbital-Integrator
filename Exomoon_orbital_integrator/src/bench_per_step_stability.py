"""
bench_per_step_stability.py

Compares stride-based vs per-step stability classification for GT and HNN hinge4
on K-1229b prograde at dt=5e-5 yr.

Uses running max/min accumulators (updated every physics step, O(N) memory)
instead of a stride-sampled buffer. This catches any stability violation no
matter when it occurs.

Runs FOUR configurations:
  A) GT  — new benchmark conditions (eligible_mask + BOTH stopping)
  B) HNN — new benchmark conditions (eligible_mask + BOTH stopping)
  C) GT  — Aug27-equivalent conditions (no mask, no stopping, mm_max=2.54 both)
  D) HNN — Aug27-equivalent conditions (no mask, no stopping, mm_max=2.54 both)

Then reports per-am_hill fractions for all four, side-by-side with the previously
obtained stride-based results.
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

# ── System ────────────────────────────────────────────────────────────────────
META_PATH = os.path.join(SRC, "ground_truth_grids", "Kepler_1229_b_prograde.meta.json")
meta     = json.load(open(META_PATH))
SP       = meta["system_params"]
ms_solar = float(SP["ms_solar"])
rs_solar = float(SP["rs_solar"])
Ts       = float(SP["Ts"])
mp_earth = float(SP["mp_earth"])
ap_AU    = float(SP["ap_AU"])
ep       = float(SP.get("ep", 0.0))

# ── Grid ──────────────────────────────────────────────────────────────────────
MM_RES, AM_RES = 50, 50
N = MM_RES * AM_RES

ms_gp     = ms_solar * FOUR_PI2
mp_gp_tmp = mp_earth * (merth / msun) * FOUR_PI2
rhill_AU  = ap_AU * (1.0 - ep) * (mp_gp_tmp / (3.0 * ms_gp)) ** (1.0 / 3.0)

mp_kg   = mp_earth * merth
rp_m    = (0.75 * mp_kg / (np.pi * _PLANET_DENSITY_SI)) ** (1.0 / 3.0)
a_roche = 2.456 * rp_m * (_PLANET_DENSITY_CGS / _MOON_DENSITY_CGS) ** (1.0 / 3.0) / au
am_min  = max(a_roche / rhill_AU, 1e-3)
mm_max  = min(mp_earth, 3.0)

mm_grid = np.exp(np.linspace(np.log(0.107), np.log(mm_max), MM_RES))
am_grid = np.linspace(am_min, 1.0, AM_RES)

a_inner_au, a_outer_au = hz_bounds_au(Ts, rs_solar * _rsun)

DT_PHYS = 5e-5
T_SIM   = 10.0
N_PHYS  = int(np.ceil(T_SIM / DT_PHYS))   # 200,000
WARMUP_STEPS = 5   # first N steps excluded from classification

F64 = torch.float64
rhill_tf   = torch.tensor(rhill_AU,   dtype=F64)
a_inner_tf = torch.tensor(a_inner_au, dtype=F64)
a_outer_tf = torch.tensor(a_outer_au, dtype=F64)

print("=" * 72)
print("  PER-STEP STABILITY BENCHMARK  —  K-1229b prograde")
print(f"  dt=5e-5 yr  t_sim={T_SIM} yr  n_phys={N_PHYS:,}  warmup={WARMUP_STEPS} steps")
print(f"  rhill={rhill_AU:.5f} AU  a_inner={a_inner_au:.4f} AU  a_outer={a_outer_au:.4f} AU")
print(f"  mm_grid: [{mm_grid[0]:.4f}, {mm_max:.4f}] M_earth  am_grid: [{am_min:.4f}, 1.0000]")
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
    np.full(N, ms_solar), np.full(N, rs_solar), np.full(N, Ts),
    np.full(N, mp_earth), np.full(N, ap_AU),    np.full(N, ep),
    mm_c, am_c,
    np.zeros(N), np.zeros(N),
    np.full(N, T_SIM), np.full(N, rhill_AU),
    np.full(N, a_inner_au), np.full(N, a_outer_au),
]).astype(np.float32)
_log_idx = [SYS_COLS.index(c) for c in LOG_SYS_COLS if c in SYS_COLS]
X_mlp[:, _log_idx] = np.log(np.clip(X_mlp[:, _log_idx], 1e-10, None))
Xs_mlp = _mlp_scl.transform(X_mlp).astype(np.float32)
with torch.no_grad():
    _logits = _mlp(torch.tensor(Xs_mlp))
_preds  = (_logits > 0).numpy()
eligible = _preds[:, 0] & _preds[:, 1]
n_elig   = int(eligible.sum())
print(f"  eligible: {n_elig}/{N} ({100*n_elig/N:.1f}%)")
eligible_t = torch.tensor(eligible.reshape(N, 1), dtype=torch.bool)
inelig_t   = ~eligible_t

dt_t      = torch.full((N, 1), DT_PHYS, dtype=F64)
half_dt_t = dt_t * 0.5


def _classify_from_accumulators(max_mpd_np, min_msd_np, max_msd_np,
                                 eligible_arr, apply_mask=True):
    stable    = max_mpd_np <= rhill_AU
    habitable = (min_msd_np >= a_inner_au) & (max_msd_np <= a_outer_au)
    both      = stable & habitable
    if apply_mask:
        stable    &= eligible_arr
        habitable &= eligible_arr
        both      &= eligible_arr
    s2 = stable.reshape(MM_RES, AM_RES)
    h2 = habitable.reshape(MM_RES, AM_RES)
    b2 = both.reshape(MM_RES, AM_RES)
    valid_rows = np.where(b2.any(axis=1))[0]
    vmm = ([float(mm_grid[valid_rows[0]]), float(mm_grid[valid_rows[-1]])]
           if len(valid_rows) else None)
    return s2, h2, b2, vmm


def _print_block(label, s2, h2, b2, vmm, elapsed, steps):
    print(f"\n  [{label}]  elapsed={elapsed:.1f} s  steps={steps:,}")
    print(f"  stable={s2.mean():.4f}  habitable={h2.mean():.4f}  both={b2.mean():.4f}")
    print(f"  valid_mm_range: {vmm}")
    va = np.where(b2.any(axis=0))[0]
    if len(va):
        print(f"  am cols with >=1 both cell: {len(va)}  am=[{am_grid[va[0]]:.3f}, {am_grid[va[-1]]:.3f}]")
    print(f"  Per-am_hill stable (avg over mm):")
    am_s = s2.mean(axis=0)
    for j in range(AM_RES):
        bar = "#" * int(am_s[j] * 20)
        print(f"    am={am_grid[j]:.3f}  {am_s[j]:.3f}  {bar}")


def _confusion_block(gt_s2, hnn_s2, gt_b2, hnn_b2, elig, label_s, label_b):
    gt_sf  = gt_s2.ravel()  & elig
    hnn_sf = hnn_s2.ravel() & elig
    gt_bf  = gt_b2.ravel()  & elig
    hnn_bf = hnn_b2.ravel() & elig
    for (gf, hf, lbl) in [(gt_sf, hnn_sf, label_s), (gt_bf, hnn_bf, label_b)]:
        TP = int(( gf &  hf).sum())
        FN = int(( gf & ~hf).sum())
        FP = int((~gf &  hf).sum())
        TN = int((~gf & ~hf).sum())
        prec   = TP/(TP+FP) if (TP+FP)>0 else float("nan")
        recall = TP/(TP+FN) if (TP+FN)>0 else float("nan")
        f1     = 2*prec*recall/(prec+recall) if (prec+recall)>0 else float("nan")
        print(f"  [{lbl}]  TP={TP}  FN={FN}  FP={FP}  TN={TN}"
              f"  Prec={prec:.3f}  Recall={recall:.3f}  F1={f1:.3f}")


# ── HNN setup (shared) ────────────────────────────────────────────────────────
HNN_DIR    = os.path.join(SRC, "models_hnn_hill_hinge4")
hill_model = HNN_Hill.load(HNN_DIR)
hill_model.eval()
hill_scaler = load_hill_sys_scaler(HNN_DIR)

mp_msun       = mp_earth * _MERTH_OVER_MSUN
ms_gp_hnn     = ms_solar * FOUR_PI2
mp_gp_hnn     = mp_msun  * FOUR_PI2
mm_per_cell   = np.repeat(mm_grid, AM_RES)
mm_msun_arr   = mm_per_cell * _MERTH_OVER_MSUN
V_ref_val     = FOUR_PI2 * ms_solar * mp_msun / ap_AU
_sys_static   = np.column_stack([
    np.full(N, ms_solar), np.full(N, mp_earth),
    mm_per_cell, np.full(N, ap_AU), np.full(N, rhill_AU),
])
F32 = torch.float32


def _hill_forces(q_ms_t, q_mp_t, q_mm_t, v_ms_t, v_mp_t):
    r_sp_np   = (q_mp_t - q_ms_t).cpu().numpy()
    v_sp_np   = (v_mp_t - v_ms_t).cpu().numpy()
    r_sp_norm = np.maximum(np.linalg.norm(r_sp_np, axis=1, keepdims=True), 1e-10)
    r_sp_sq   = (r_sp_norm ** 2).squeeze(1)
    x_hat = r_sp_np / r_sp_norm
    L_sp  = np.cross(r_sp_np, v_sp_np)
    L_norm = np.maximum(np.linalg.norm(L_sp, axis=1, keepdims=True), 1e-10)
    z_hat  = L_sp / L_norm
    Omega  = L_norm.squeeze(1) / np.maximum(r_sp_sq, 1e-20)
    y_hat  = np.cross(z_hat, x_hat)
    r_mp_np = (q_mm_t - q_mp_t).cpu().numpy()
    q_h_x   = (r_mp_np * x_hat).sum(axis=1)
    q_h_y   = (r_mp_np * y_hat).sum(axis=1)
    q_h_z   = (r_mp_np * z_hat).sum(axis=1)
    q_h_AU  = np.stack([q_h_x, q_h_y, q_h_z], axis=1)
    q_h_norm = q_h_AU / max(rhill_AU, 1e-20)
    sys_raw  = np.column_stack([_sys_static, Omega])
    sys_log  = np.log(np.maximum(sys_raw, 1e-20))
    sys_enc  = torch.tensor(
        hill_scaler.transform(sys_log).astype(np.float32), dtype=F32)
    q_h_leaf = torch.tensor(
        q_h_norm.astype(np.float32), dtype=F32).detach().requires_grad_(True)
    with torch.enable_grad():
        V_theta = hill_model(q_h_leaf, sys_enc)
        grad    = torch.autograd.grad(V_theta.sum(), q_h_leaf, create_graph=False)[0]
    grad_np  = grad.detach().cpu().numpy().astype(np.float64)
    a_cons_h = -grad_np * V_ref_val / (mm_msun_arr[:, None] * rhill_AU)
    cf       = (Omega[:, None] ** 2 *
                np.column_stack([q_h_AU[:, 0], q_h_AU[:, 1], np.zeros(N)]))
    a_hill   = a_cons_h - cf
    a_rel_in = (a_hill[:, 0:1] * x_hat +
                a_hill[:, 1:2] * y_hat +
                a_hill[:, 2:3] * z_hat)
    a_p_arr  = -FOUR_PI2 * ms_solar * r_sp_np / (r_sp_sq ** 1.5)[:, None]
    a_m      = a_rel_in + a_p_arr
    r_sm_np  = (q_mm_t - q_ms_t).cpu().numpy()
    d_sm_cu  = (r_sm_np ** 2).sum(axis=1, keepdims=True) ** 1.5
    a_s      = (FOUR_PI2 * mm_msun_arr[:, None] / d_sm_cu * r_sm_np +
                FOUR_PI2 * mp_msun * r_sp_np / (r_sp_sq ** 1.5)[:, None])
    return (torch.tensor(a_s, dtype=F64),
            torch.tensor(a_p_arr, dtype=F64),
            torch.tensor(a_m, dtype=F64))


def _hnn_initial_vel(pos_mp0, pos_ms0, pos_mm0, vel_mp0, vel_ms0, vel_mm0):
    """Calibrate initial v_mm to match HNN circular speed."""
    q_ms = torch.tensor(pos_ms0, dtype=F64)
    q_mp = torch.tensor(pos_mp0, dtype=F64)
    q_mm = torch.tensor(pos_mm0, dtype=F64)
    v_ms = torch.tensor(vel_ms0, dtype=F64)
    v_mp = torch.tensor(vel_mp0, dtype=F64)
    v_mm = torch.tensor(vel_mm0, dtype=F64)
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
    sys_r0  = np.column_stack([_sys_static, Omega0])
    sys_e0  = torch.tensor(
        hill_scaler.transform(np.log(np.maximum(sys_r0, 1e-20))).astype(np.float32),
        dtype=F32)
    q_lf0   = torch.tensor(q_h_n0.astype(np.float32), dtype=F32).detach().requires_grad_(True)
    with torch.enable_grad():
        V0    = hill_model(q_lf0, sys_e0)
        grad0 = torch.autograd.grad(V0.sum(), q_lf0, create_graph=False)[0]
    g0_np    = grad0.detach().cpu().numpy().astype(np.float64)
    a_ch0    = -g0_np * V_ref_val / (mm_msun_arr[:, None] * rhill_AU)
    ach_mag0 = np.linalg.norm(a_ch0, axis=1)
    r_pm0    = np.linalg.norm(r_mp0, axis=1)
    v_circ0  = np.where((ach_mag0 > 1e-30) & (r_pm0 > 1e-30),
                         np.sqrt(r_pm0 * ach_mag0), 0.0)
    v_rel0     = (v_mm - v_mp).cpu().numpy()
    v_rel0_mag = np.linalg.norm(v_rel0, axis=1)
    ok_cal     = (v_circ0 > 1e-30) & (v_rel0_mag > 1e-30)
    v_rel0_dir = np.where(ok_cal[:, None], v_rel0 / v_rel0_mag[:, None], v_rel0)
    v_mm_np0   = v_mm.cpu().numpy().copy()
    v_mm_np0[ok_cal] = (v_mp.cpu().numpy() + v_circ0[:, None] * v_rel0_dir)[ok_cal]
    v_mm = torch.tensor(v_mm_np0, dtype=F64)
    print(f"    [HNN] Calibrated {ok_cal.sum()}/{N} cells")
    return q_ms, q_mp, q_mm, v_ms, v_mp, v_mm


# ═════════════════════════════════════════════════════════════════════════════
# Run GT — returns (max_mpd_np, min_msd_np, max_msd_np, elapsed, steps)
# ═════════════════════════════════════════════════════════════════════════════
def run_gt(use_mask, use_stopping, label):
    print(f"\n{'='*72}")
    print(f"  GT  [{label}]  use_mask={use_mask}  use_stopping={use_stopping}")
    (pos_mp0, pos_ms0, pos_mm0, vel_mp0, vel_ms0, vel_mm0,
     mu_ms, mu_mp, mu_mm_arr, _) = _build_initial_states(
        ms_solar=ms_solar, mp_earth=mp_earth,
        ap_AU=ap_AU, ep=ep, em=0.0,
        moon_retrograde=False,
        mm_grid=mm_grid, am_grid=am_grid)
    p_mp = torch.tensor(pos_mp0, dtype=F64)
    p_ms = torch.tensor(pos_ms0, dtype=F64)
    p_mm = torch.tensor(pos_mm0, dtype=F64)
    v_mp = torch.tensor(vel_mp0, dtype=F64)
    v_ms = torch.tensor(vel_ms0, dtype=F64)
    v_mm = torch.tensor(vel_mm0, dtype=F64)
    mu_mm_t = torch.tensor(mu_mm_arr[:, None], dtype=F64)

    # Per-step accumulators (initialised after warmup)
    max_mpd = torch.zeros(N, 1, dtype=F64)
    min_msd = torch.full((N, 1), 1e30, dtype=F64)
    max_msd = torch.zeros(N, 1, dtype=F64)
    warmup_done = False

    elapsed_t = torch.zeros(N, 1, dtype=F64)
    done_t    = inelig_t.clone() if use_mask else torch.zeros(N, 1, dtype=torch.bool)

    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(N_PHYS):
            active = (~done_t).to(F64)
            p2_mp = p_mp + v_mp * (half_dt_t * active)
            a_mp  = _accel_batch(p2_mp, p_ms, mu_ms) + _accel_batch(p2_mp, p_mm, mu_mm_t)
            v_mp  = v_mp + a_mp * (dt_t * active)
            p_mp  = p2_mp + v_mp * (half_dt_t * active)
            p2_ms = p_ms + v_ms * (half_dt_t * active)
            a_ms  = _accel_batch(p2_ms, p_mp, mu_mp) + _accel_batch(p2_ms, p_mm, mu_mm_t)
            v_ms  = v_ms + a_ms * (dt_t * active)
            p_ms  = p2_ms + v_ms * (half_dt_t * active)
            p2_mm = p_mm + v_mm * (half_dt_t * active)
            a_mm  = _accel_batch(p2_mm, p_mp, mu_mp) + _accel_batch(p2_mm, p_ms, mu_ms)
            v_mm  = v_mm + a_mm * (dt_t * active)
            p_mm  = p2_mm + v_mm * (half_dt_t * active)
            diff_mp = p_mm - p_mp
            diff_ms = p_mm - p_ms
            mpd_s   = (diff_mp * diff_mp).sum(dim=1, keepdim=True).sqrt()
            msd_s   = (diff_ms * diff_ms).sum(dim=1, keepdim=True).sqrt()
            elapsed_t += dt_t * active
            if i >= WARMUP_STEPS:
                if not warmup_done:
                    warmup_done = True
                torch.maximum(max_mpd, mpd_s, out=max_mpd)
                torch.minimum(min_msd, msd_s, out=min_msd)
                torch.maximum(max_msd, msd_s, out=max_msd)
            if use_stopping:
                unstable      = mpd_s > rhill_tf
                uninhabitable = (msd_s < a_inner_tf) | (msd_s > a_outer_tf)
                done_t = done_t | (unstable & uninhabitable) | (elapsed_t >= T_SIM)
            else:
                done_t = done_t | (elapsed_t >= T_SIM)
            if done_t.all():
                break
    elapsed = time.perf_counter() - t0
    steps   = i + 1
    return (max_mpd.squeeze(1).numpy(),
            min_msd.squeeze(1).numpy(),
            max_msd.squeeze(1).numpy(),
            elapsed, steps)


# ═════════════════════════════════════════════════════════════════════════════
# Run HNN — returns (max_mpd_np, min_msd_np, max_msd_np, elapsed, steps)
# ═════════════════════════════════════════════════════════════════════════════
def run_hnn(use_mask, use_stopping, label):
    print(f"\n{'='*72}")
    print(f"  HNN [{label}]  use_mask={use_mask}  use_stopping={use_stopping}")
    (pos_mp0, pos_ms0, pos_mm0, vel_mp0, vel_ms0, vel_mm0,
     _, _, _, _) = _build_initial_states(
        ms_solar=ms_solar, mp_earth=mp_earth,
        ap_AU=ap_AU, ep=ep, em=0.0,
        moon_retrograde=False,
        mm_grid=mm_grid, am_grid=am_grid)

    q_ms, q_mp, q_mm, v_ms, v_mp, v_mm = _hnn_initial_vel(
        pos_mp0, pos_ms0, pos_mm0, vel_mp0, vel_ms0, vel_mm0)

    print("    [HNN] Computing initial forces ...")
    a_s_h, a_p_h, a_m_h = _hill_forces(q_ms, q_mp, q_mm, v_ms, v_mp)

    max_mpd = torch.zeros(N, 1, dtype=F64)
    min_msd = torch.full((N, 1), 1e30, dtype=F64)
    max_msd = torch.zeros(N, 1, dtype=F64)
    warmup_done = False

    elapsed_hnn = torch.zeros(N, 1, dtype=F64)
    stopped     = inelig_t.clone() if use_mask else torch.zeros(N, 1, dtype=torch.bool)

    t0 = time.perf_counter()
    with torch.no_grad():
        for step in range(N_PHYS):
            mask = (~stopped).to(F64)
            v_ms = v_ms + a_s_h * (half_dt_t * mask)
            v_mp = v_mp + a_p_h * (half_dt_t * mask)
            v_mm = v_mm + a_m_h * (half_dt_t * mask)
            q_ms = q_ms + v_ms * (dt_t * mask)
            q_mp = q_mp + v_mp * (dt_t * mask)
            q_mm = q_mm + v_mm * (dt_t * mask)
            a_s_h, a_p_h, a_m_h = _hill_forces(q_ms, q_mp, q_mm, v_ms, v_mp)
            v_ms = v_ms + a_s_h * (half_dt_t * mask)
            v_mp = v_mp + a_p_h * (half_dt_t * mask)
            v_mm = v_mm + a_m_h * (half_dt_t * mask)
            diff_mp_h = q_mm - q_mp
            diff_ms_h = q_mm - q_ms
            mpd_col   = (diff_mp_h * diff_mp_h).sum(1, keepdim=True).sqrt()
            msd_col   = (diff_ms_h * diff_ms_h).sum(1, keepdim=True).sqrt()
            elapsed_hnn += dt_t * mask
            if step >= WARMUP_STEPS:
                if not warmup_done:
                    warmup_done = True
                torch.maximum(max_mpd, mpd_col, out=max_mpd)
                torch.minimum(min_msd, msd_col, out=min_msd)
                torch.maximum(max_msd, msd_col, out=max_msd)
            if use_stopping:
                unstable_h = mpd_col > rhill_tf
                uninhab_h  = (msd_col < a_inner_tf) | (msd_col > a_outer_tf)
                stopped = stopped | (unstable_h & uninhab_h) | (elapsed_hnn >= T_SIM)
            else:
                stopped = stopped | (elapsed_hnn >= T_SIM)
            if stopped.all():
                break
    elapsed = time.perf_counter() - t0
    steps   = step + 1
    return (max_mpd.squeeze(1).numpy(),
            min_msd.squeeze(1).numpy(),
            max_msd.squeeze(1).numpy(),
            elapsed, steps)


# ═════════════════════════════════════════════════════════════════════════════
# RUN ALL FOUR CONDITIONS
# ═════════════════════════════════════════════════════════════════════════════

# A) GT new-benchmark conditions
gt_a_mpd, gt_a_msd_min, gt_a_msd_max, gt_a_t, gt_a_s = run_gt(
    use_mask=True,  use_stopping=True,  label="NEW-BENCHMARK")
gt_a_s2, gt_a_h2, gt_a_b2, gt_a_vmm = _classify_from_accumulators(
    gt_a_mpd, gt_a_msd_min, gt_a_msd_max, eligible, apply_mask=True)
_print_block("GT  NEW-BENCH", gt_a_s2, gt_a_h2, gt_a_b2, gt_a_vmm, gt_a_t, gt_a_s)

# B) HNN new-benchmark conditions
hnn_a_mpd, hnn_a_msd_min, hnn_a_msd_max, hnn_a_t, hnn_a_s = run_hnn(
    use_mask=True,  use_stopping=True,  label="NEW-BENCHMARK")
hnn_a_s2, hnn_a_h2, hnn_a_b2, hnn_a_vmm = _classify_from_accumulators(
    hnn_a_mpd, hnn_a_msd_min, hnn_a_msd_max, eligible, apply_mask=True)
_print_block("HNN NEW-BENCH", hnn_a_s2, hnn_a_h2, hnn_a_b2, hnn_a_vmm, hnn_a_t, hnn_a_s)

# C) GT Aug27-equivalent (no mask, no stopping, mm_max=2.54 both)
gt_c_mpd, gt_c_msd_min, gt_c_msd_max, gt_c_t, gt_c_s = run_gt(
    use_mask=False, use_stopping=False, label="AUG27-EQUIV")
gt_c_s2, gt_c_h2, gt_c_b2, gt_c_vmm = _classify_from_accumulators(
    gt_c_mpd, gt_c_msd_min, gt_c_msd_max, eligible, apply_mask=False)
_print_block("GT  AUG27-EQ ", gt_c_s2, gt_c_h2, gt_c_b2, gt_c_vmm, gt_c_t, gt_c_s)

# D) HNN Aug27-equivalent (no mask, no stopping, mm_max=2.54 both)
hnn_d_mpd, hnn_d_msd_min, hnn_d_msd_max, hnn_d_t, hnn_d_s = run_hnn(
    use_mask=False, use_stopping=False, label="AUG27-EQUIV")
hnn_d_s2, hnn_d_h2, hnn_d_b2, hnn_d_vmm = _classify_from_accumulators(
    hnn_d_mpd, hnn_d_msd_min, hnn_d_msd_max, eligible, apply_mask=False)
_print_block("HNN AUG27-EQ ", hnn_d_s2, hnn_d_h2, hnn_d_b2, hnn_d_vmm, hnn_d_t, hnn_d_s)


# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("  SUMMARY — per-step classification vs previous stride-200 results")
print("=" * 72)
print(f"\n  {'Config':<28} {'stable':>8} {'hab':>8} {'both':>8}  note")
print(f"  {'-'*28} {'-------':>8} {'-------':>8} {'-------':>8}  ----")

# Previous stride results (from bench_pre_optD_fixed_dt.py output)
prev = {
    "GT  stride-200 NEW-BENCH":  (0.4840, 0.4328, 0.4260),
    "HNN stride-200 NEW-BENCH":  (0.1472, 0.4456, 0.1472),
    "HNN stride-200 AUG27*":     (None,   None,   0.284),   # Aug 27 actual (diff grid GT)
}
for k, (s, h, b) in prev.items():
    sv = f"{s:.4f}" if s is not None else "  n/a"
    hv = f"{h:.4f}" if h is not None else "  n/a"
    bv = f"{b:.4f}" if b is not None else "  n/a"
    note = "* GT had mm_max=0.5, comparison invalid" if "AUG27" in k else ""
    print(f"  {k:<28} {sv:>8} {hv:>8} {bv:>8}  {note}")

print()
for lbl, s2, h2, b2 in [
    ("GT  per-step NEW-BENCH",  gt_a_s2,  gt_a_h2,  gt_a_b2),
    ("HNN per-step NEW-BENCH",  hnn_a_s2, hnn_a_h2, hnn_a_b2),
    ("GT  per-step AUG27-EQ ",  gt_c_s2,  gt_c_h2,  gt_c_b2),
    ("HNN per-step AUG27-EQ ",  hnn_d_s2, hnn_d_h2, hnn_d_b2),
]:
    print(f"  {lbl:<28} {s2.mean():>8.4f} {h2.mean():>8.4f} {b2.mean():>8.4f}  per-step accumulator")

# Confusion matrices (per-step, new-benchmark conditions, eligible only)
print(f"\n  CONFUSION MATRICES — per-step NEW-BENCHMARK (eligible, n={n_elig}):")
_confusion_block(gt_a_s2, hnn_a_s2, gt_a_b2, hnn_a_b2, eligible, "STABLE", "BOTH")

print(f"\n  SIDE-BY-SIDE per-am_hill stable — per-step NEW-BENCHMARK:")
print(f"  {'am_hill':>8}  {'GT_ps':>7}  {'HNN_ps':>8}  {'GT_str':>7}  {'HNN_str':>8}  diff_ps")
gt_ps_am  = gt_a_s2.mean(axis=0)
hnn_ps_am = hnn_a_s2.mean(axis=0)
# Previous stride results per-am (from memory — GT=1.000 for am=0.054-0.488, HNN varies)
for j in range(AM_RES):
    diff = hnn_ps_am[j] - gt_ps_am[j]
    print(f"  {am_grid[j]:>8.3f}  {gt_ps_am[j]:>7.3f}  {hnn_ps_am[j]:>8.3f}  {diff:>+7.3f}")

print(f"\n  SIDE-BY-SIDE per-am_hill stable — per-step AUG27-EQUIV (no mask):")
print(f"  {'am_hill':>8}  {'GT_ps':>7}  {'HNN_ps':>8}  diff_ps")
gt_c_am  = gt_c_s2.mean(axis=0)
hnn_d_am = hnn_d_s2.mean(axis=0)
for j in range(AM_RES):
    diff = hnn_d_am[j] - gt_c_am[j]
    print(f"  {am_grid[j]:>8.3f}  {gt_c_am[j]:>7.3f}  {hnn_d_am[j]:>8.3f}  {diff:>+7.3f}")

print(f"\n  SIDE-BY-SIDE per-am_hill habitable — per-step NEW-BENCHMARK:")
print(f"  {'am_hill':>8}  {'GT_ps':>7}  {'HNN_ps':>8}  diff_ps")
gt_ps_ah  = gt_a_h2.mean(axis=0)
hnn_ps_ah = hnn_a_h2.mean(axis=0)
for j in range(AM_RES):
    diff = hnn_ps_ah[j] - gt_ps_ah[j]
    print(f"  {am_grid[j]:>8.3f}  {gt_ps_ah[j]:>7.3f}  {hnn_ps_ah[j]:>8.3f}  {diff:>+7.3f}")

print(f"\n  SIDE-BY-SIDE per-am_hill habitable — per-step AUG27-EQUIV (no mask):")
print(f"  {'am_hill':>8}  {'GT_ps':>7}  {'HNN_ps':>8}  diff_ps")
gt_c_ah  = gt_c_h2.mean(axis=0)
hnn_d_ah = hnn_d_h2.mean(axis=0)
for j in range(AM_RES):
    diff = hnn_d_ah[j] - gt_c_ah[j]
    print(f"  {am_grid[j]:>8.3f}  {gt_c_ah[j]:>7.3f}  {hnn_d_ah[j]:>8.3f}  {diff:>+7.3f}")

print(f"\n  SIDE-BY-SIDE per-am_hill BOTH — per-step NEW-BENCHMARK:")
print(f"  {'am_hill':>8}  {'GT_ps':>7}  {'HNN_ps':>8}  diff_ps")
gt_ps_ab  = gt_a_b2.mean(axis=0)
hnn_ps_ab = hnn_a_b2.mean(axis=0)
for j in range(AM_RES):
    diff = hnn_ps_ab[j] - gt_ps_ab[j]
    print(f"  {am_grid[j]:>8.3f}  {gt_ps_ab[j]:>7.3f}  {hnn_ps_ab[j]:>8.3f}  {diff:>+7.3f}")

print(f"\n  SIDE-BY-SIDE per-am_hill BOTH — per-step AUG27-EQUIV (no mask):")
print(f"  {'am_hill':>8}  {'GT_ps':>7}  {'HNN_ps':>8}  diff_ps")
gt_c_ab  = gt_c_b2.mean(axis=0)
hnn_d_ab = hnn_d_b2.mean(axis=0)
for j in range(AM_RES):
    diff = hnn_d_ab[j] - gt_c_ab[j]
    print(f"  {am_grid[j]:>8.3f}  {gt_c_ab[j]:>7.3f}  {hnn_d_ab[j]:>8.3f}  {diff:>+7.3f}")

print("\n" + "=" * 72)
print("  DONE.")
print("=" * 72)
