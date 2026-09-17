"""
Diagnostic: isolate whether stable=0 comes from K=10/16 integration
or from the eligible_mask filtering.

Tests K-1229b for t_sim=2 yr (short, completes ~60-200 s):
  A) K=16, no eligible_mask  -> should show inner cells stable
  B) K=16, with eligible_mask (same as bench_k16_quick)
  C) K=10, no eligible_mask  -> compare with A

Prints per-am_hill stable fraction so we can see WHERE instability appears.
"""
import json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import pickle

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from exomoon.constants      import FOUR_PI2, au, merth, msun
from exomoon.constants      import rsun as _rsun
from exomoon.habitable_zone import hz_bounds_au
from exomoon.ml.dataset     import SYS_COLS, LOG_SYS_COLS

MM_RES, AM_RES = 50, 50
N_CELLS        = MM_RES * AM_RES
MLP_DIR        = os.path.join(SRC, "eval_aux_mlp_output", "binary")
_PLANET_DENSITY_CGS = 5.5
_PLANET_DENSITY_SI  = 5500.0
_MOON_DENSITY_CGS   = 3.0


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


def _load_aux_mlp(mlp_dir):
    cfg = json.load(open(os.path.join(mlp_dir, "model_config.json")))
    mdl = AuxMLPBinary(input_dim=cfg["input_dim"], hidden=cfg["hidden"])
    mdl.load_state_dict(torch.load(os.path.join(mlp_dir, "aux_mlp_binary.pt"),
                                    map_location="cpu", weights_only=True))
    mdl.eval()
    scl = pickle.load(open(os.path.join(mlp_dir, "aux_mlp_scaler.pkl"), "rb"))
    return mdl, scl


def _apply_log(arr):
    arr = arr.copy()
    idx = [SYS_COLS.index(c) for c in LOG_SYS_COLS if c in SYS_COLS]
    arr[:, idx] = np.log(np.clip(arr[:, idx], 1e-10, None))
    return arr


def _compute_grid(sp, mm_res, am_res):
    ms_solar, mp_earth = float(sp["ms_solar"]), float(sp["mp_earth"])
    ap_AU, ep           = float(sp["ap_AU"]),   float(sp.get("ep", 0.0))
    rs_solar, Ts        = float(sp.get("rs_solar", 1.0)), float(sp.get("Ts", 5772.0))

    ms_gp    = ms_solar * FOUR_PI2
    mp_gp    = mp_earth * (merth / msun) * FOUR_PI2
    rhill_AU = ap_AU * (1.0 - ep) * (mp_gp / (3.0 * ms_gp)) ** (1.0 / 3.0)

    mp_kg   = mp_earth * merth
    rp_m    = (0.75 * mp_kg / (np.pi * _PLANET_DENSITY_SI)) ** (1.0 / 3.0)
    a_roche = 2.456 * rp_m * (_PLANET_DENSITY_CGS / _MOON_DENSITY_CGS) ** (1.0 / 3.0) / au

    am_min  = max(a_roche / rhill_AU, 1e-3)
    mm_max  = min(mp_earth, 3.0)
    mm_grid = np.exp(np.linspace(np.log(0.107), np.log(mm_max), mm_res))
    am_grid = np.linspace(am_min, 1.0, am_res)
    a_inner, a_outer = hz_bounds_au(Ts, rs_solar * _rsun)
    return mm_grid, am_grid, rhill_AU, mp_gp, a_inner, a_outer


def _mlp_mask(mlp_dir, sp, mm_grid, am_grid, rhill_AU, a_inner, a_outer, t_sim):
    N = len(mm_grid) * len(am_grid)
    mm_res, am_res = len(mm_grid), len(am_grid)
    if not os.path.isfile(os.path.join(mlp_dir, "aux_mlp_binary.pt")):
        return np.ones(N, dtype=bool)
    mdl, scl = _load_aux_mlp(mlp_dir)
    mm_c = np.repeat(mm_grid, am_res)
    am_c = np.tile(am_grid, mm_res)
    X = np.column_stack([
        np.full(N, sp["ms_solar"]), np.full(N, sp.get("rs_solar", 1.0)),
        np.full(N, sp.get("Ts", 5772.0)), np.full(N, sp["mp_earth"]),
        np.full(N, sp["ap_AU"]), np.full(N, sp.get("ep", 0.0)),
        mm_c, am_c, np.full(N, 0.0), np.full(N, 0.0),
        np.full(N, t_sim), np.full(N, rhill_AU),
        np.full(N, a_inner), np.full(N, a_outer),
    ]).astype(np.float32)
    X  = _apply_log(X).astype(np.float32)
    Xs = scl.transform(X).astype(np.float32)
    with torch.no_grad():
        logits = mdl(torch.tensor(Xs))
    preds = (logits > 0).numpy()
    elig  = preds[:, 0] & preds[:, 1]
    print(f"  [MLP] eligible {elig.sum()}/{N} ({100*elig.sum()/N:.1f}%)")
    # Per am_hill bucket
    elig_2d = elig.reshape(mm_res, am_res)
    am_elig = elig_2d.any(axis=0)   # (am_res,) — True if ANY mm row eligible at this am
    return elig


def _run_gt(sp, mm_grid, am_grid, t_sim, K, eligible_mask, n_steps=200):
    """Inline GT batch leapfrog — same code as batch_leapfrog but with tunable K."""
    from exomoon.ml.batch_leapfrog import _build_initial_states, _accel_batch
    from exomoon.constants import FOUR_PI2, merth, msun, au
    from exomoon.habitable_zone import hz_bounds_au
    from exomoon.constants import rsun as _rsun

    ms_solar = float(sp["ms_solar"])
    mp_earth = float(sp["mp_earth"])
    ap_AU    = float(sp["ap_AU"])
    ep       = float(sp.get("ep", 0.0))
    Ts       = float(sp.get("Ts", 5772.0))
    rs_solar = float(sp.get("rs_solar", 1.0))

    ms_gp_tmp = ms_solar * FOUR_PI2
    mp_gp_tmp = mp_earth * (merth / msun) * FOUR_PI2
    rhill_AU  = ap_AU * (1.0 - ep) * (mp_gp_tmp / (3.0 * ms_gp_tmp)) ** (1.0 / 3.0)

    mp_kg   = mp_earth * merth
    rp_m    = (0.75 * mp_kg / (np.pi * _PLANET_DENSITY_SI)) ** (1.0 / 3.0)
    a_roche = 2.456 * rp_m * (_PLANET_DENSITY_CGS / _MOON_DENSITY_CGS) ** (1.0 / 3.0) / au
    am_min  = max(a_roche / rhill_AU, 1e-3)

    a_inner_au, a_outer_au = hz_bounds_au(Ts, rs_solar * _rsun)

    (pos_mp0, pos_ms0, pos_mm0, vel_mp0, vel_ms0, vel_mm0,
     mu_ms, mu_mp, mu_mm_arr, rhill) = _build_initial_states(
        ms_solar=ms_solar, mp_earth=mp_earth, ap_AU=ap_AU, ep=ep, em=0.0,
        moon_retrograde=False, mm_grid=mm_grid, am_grid=am_grid,
    )

    mm_res, am_res = len(mm_grid), len(am_grid)
    N = mm_res * am_res
    dtype = torch.float64
    dev   = torch.device("cpu")

    p_mp = torch.tensor(pos_mp0, dtype=dtype)
    p_ms = torch.tensor(pos_ms0, dtype=dtype)
    p_mm = torch.tensor(pos_mm0, dtype=dtype)
    v_mp = torch.tensor(vel_mp0, dtype=dtype)
    v_ms = torch.tensor(vel_ms0, dtype=dtype)
    v_mm = torch.tensor(vel_mm0, dtype=dtype)
    mu_mm_t = torch.tensor(mu_mm_arr[:, None], dtype=dtype)

    am_per_cell    = np.tile(am_grid, mm_res)
    am_AU_per_cell = am_per_cell * rhill_AU
    T_moon_cell    = 2.0 * np.pi * np.sqrt(am_AU_per_cell**3 / (mu_mp + mu_mm_arr))
    dt_arr         = T_moon_cell / K
    dt_min_val     = float(dt_arr.min())
    n_phys         = max(int(np.ceil(t_sim / dt_min_val)), n_steps)
    stride         = max(1, n_phys // n_steps)
    n_out_pre      = n_phys // stride

    dt_t      = torch.tensor(dt_arr[:, None], dtype=dtype)
    half_dt_t = dt_t * 0.5

    rhill_tf   = torch.tensor(rhill_AU,   dtype=dtype)
    a_inner_tf = torch.tensor(a_inner_au, dtype=dtype)
    a_outer_tf = torch.tensor(a_outer_au, dtype=dtype)

    mpd_t = torch.empty(N, n_out_pre, dtype=torch.float32)
    msd_t = torch.empty(N, n_out_pre, dtype=torch.float32)

    print(f"  K={K}  dt_min={dt_min_val:.2e} yr  dt_max={dt_arr.max():.2e} yr"
          f"  n_phys={n_phys:,}  t_sim={t_sim:.3f} yr  n_out={n_out_pre}")

    with torch.no_grad():
        elapsed_t = torch.zeros(N, 1, dtype=dtype)
        done_t    = torch.zeros(N, 1, dtype=torch.bool)
        if eligible_mask is not None:
            inelig = torch.tensor(
                ~np.asarray(eligible_mask, dtype=bool).reshape(N, 1), dtype=torch.bool)
            done_t = done_t | inelig

        out_idx = 0
        t0 = time.perf_counter()
        for i in range(n_phys):
            active = (~done_t).to(dtype)

            p2_mp = p_mp + v_mp * (half_dt_t * active)
            a_mp  = (_accel_batch(p2_mp, p_ms, mu_ms)
                   + _accel_batch(p2_mp, p_mm, mu_mm_t))
            v_mp  = v_mp + a_mp * (dt_t * active)
            p_mp  = p2_mp + v_mp * (half_dt_t * active)

            p2_ms = p_ms + v_ms * (half_dt_t * active)
            a_ms  = (_accel_batch(p2_ms, p_mp, mu_mp)
                   + _accel_batch(p2_ms, p_mm, mu_mm_t))
            v_ms  = v_ms + a_ms * (dt_t * active)
            p_ms  = p2_ms + v_ms * (half_dt_t * active)

            p2_mm = p_mm + v_mm * (half_dt_t * active)
            a_mm  = (_accel_batch(p2_mm, p_mp, mu_mp)
                   + _accel_batch(p2_mm, p_ms, mu_ms))
            v_mm  = v_mm + a_mm * (dt_t * active)
            p_mm  = p2_mm + v_mm * (half_dt_t * active)

            diff_mp = p_mm - p_mp
            diff_ms = p_mm - p_ms
            mpd_s = (diff_mp * diff_mp).sum(dim=1, keepdim=True).sqrt()
            msd_s = (diff_ms * diff_ms).sum(dim=1, keepdim=True).sqrt()

            if i % stride == 0 and out_idx < n_out_pre:
                mpd_t[:, out_idx] = mpd_s.squeeze(1).float()
                msd_t[:, out_idx] = msd_s.squeeze(1).float()
                out_idx += 1

            elapsed_t += dt_t * active
            unstable      = mpd_s > (1.0 * rhill_tf)
            uninhabitable = (msd_s < a_inner_tf) | (msd_s > a_outer_tf)
            done_t = done_t | (unstable & uninhabitable) | (elapsed_t >= t_sim)
            if done_t.all():
                break

    elapsed = time.perf_counter() - t0
    n_out = out_idx
    mpd = mpd_t.numpy()[:, :n_out]
    msd = msd_t.numpy()[:, :n_out]

    w = min(5, n_out - 1)
    mpd_post = mpd[:, w:]
    msd_post = msd[:, w:]

    map_stable    = (mpd_post.max(axis=1) <= rhill_AU)
    map_habitable = ((msd_post.min(axis=1) >= a_inner_au) &
                     (msd_post.max(axis=1) <= a_outer_au))

    map_stable    = map_stable.reshape(mm_res, am_res)
    map_habitable = map_habitable.reshape(mm_res, am_res)

    if eligible_mask is not None:
        em_2d = np.asarray(eligible_mask, dtype=bool).reshape(mm_res, am_res)
        map_stable    &= em_2d
        map_habitable &= em_2d

    print(f"  Elapsed: {elapsed:.1f} s")
    print(f"  stable={map_stable.mean():.3f}  habitable={map_habitable.mean():.3f}")
    print(f"  Per-am_hill stable fraction (averaged over mm dimension):")
    am_stable = map_stable.mean(axis=0)  # (am_res,)
    for j in range(am_res):
        bar = '#' * int(am_stable[j] * 20)
        print(f"    am_hill={am_grid[j]:.3f}  {am_stable[j]:.3f}  {bar}")
    return map_stable, map_habitable


# ── Load system ────────────────────────────────────────────────────────────────
meta = json.load(open(os.path.join(SRC, "ground_truth_grids",
                                   "Kepler_1229_b_prograde.meta.json")))
sp   = {k: meta["system_params"][k] for k in ("ms_solar", "rs_solar", "Ts", "mp_earth", "ap_AU")}
sp["ep"] = meta["system_params"].get("ep", 0.0)

mm_grid, am_grid, rhill_AU, mp_gp, a_inner, a_outer = _compute_grid(sp, MM_RES, AM_RES)
T_SIM = 2.0   # short test: 2 yr

print(f"\n{'='*70}")
print(f"  DIAGNOSTIC: Kepler-1229b prograde  t_sim={T_SIM} yr")
print(f"  {sp}")
print(f"  rhill_AU={rhill_AU:.5f}  a_inner={a_inner:.4f}  a_outer={a_outer:.4f}")
print(f"{'='*70}")

# MLP mask (uses t_sim=2 yr for features)
eligible = _mlp_mask(MLP_DIR, sp, mm_grid, am_grid, rhill_AU, a_inner, a_outer, T_SIM)
n_elig   = int(eligible.sum())

print(f"\n=== A) K=16, NO eligible_mask ===")
_run_gt(sp, mm_grid, am_grid, T_SIM, K=16, eligible_mask=None)

print(f"\n=== B) K=16, WITH eligible_mask (n_elig={n_elig}) ===")
_run_gt(sp, mm_grid, am_grid, T_SIM, K=16, eligible_mask=eligible)

print(f"\n=== C) K=10, NO eligible_mask ===")
_run_gt(sp, mm_grid, am_grid, T_SIM, K=10, eligible_mask=None)

print(f"\n=== D) K=10, WITH eligible_mask (n_elig={n_elig}) ===")
_run_gt(sp, mm_grid, am_grid, T_SIM, K=10, eligible_mask=eligible)

print("\nDone.")
