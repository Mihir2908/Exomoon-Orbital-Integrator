"""
benchmark_hnn_hill_inference.py — GT batch leapfrog vs HNN hinge4 batch.

Two-layer architecture:
  Layer 1: AuxMLPBinary predicts which (mm_earth, am_hill) cells are stable+habitable.
  Layer 2: Only MLP-eligible cells get trajectory integration (GT or HNN).

Timestep: per-cell k=10 (Option D) — dt_i = T_moon_i / 10 for each cell.

Usage:
    cd Exomoon_orbital_integrator/src
    py benchmark_hnn_hill_inference.py

CRITICAL: does not touch models/, models_temphead/, eval_aux_mlp_output/.
"""

import json, os, pickle, sys, time
import numpy as np
import torch
import torch.nn as nn

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from exomoon.constants        import FOUR_PI2, au, merth, msun
from exomoon.constants        import rsun as _rsun
from exomoon.habitable_zone   import hz_bounds_au
from exomoon.ml.batch_leapfrog     import batch_leapfrog_trajectories
from exomoon.ml.hnn_inference_hill import batch_hnn_hill_trajectories
from exomoon.ml.dataset            import SYS_COLS, LOG_SYS_COLS

MM_RES, AM_RES = 50, 50
N_CELLS        = MM_RES * AM_RES
HINGE4_DIR     = os.path.join(SRC, "models_hnn_hill_hinge4")
MLP_DIR        = os.path.join(SRC, "eval_aux_mlp_output", "binary")
N_ORBITS       = 100
_PLANET_DENSITY_CGS = 5.5
_PLANET_DENSITY_SI  = 5500.0
_MOON_DENSITY_CGS   = 3.0


# ── Minimal AuxMLPBinary definition (mirrors eval_aux_mlp.py) ────────────────

class AuxMLPBinary(nn.Module):
    def __init__(self, input_dim=14, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


def _load_aux_mlp(mlp_dir):
    cfg_path  = os.path.join(mlp_dir, "model_config.json")
    pt_path   = os.path.join(mlp_dir, "aux_mlp_binary.pt")
    scl_path  = os.path.join(mlp_dir, "aux_mlp_scaler.pkl")
    with open(cfg_path) as f:
        cfg = json.load(f)
    model = AuxMLPBinary(input_dim=cfg["input_dim"], hidden=cfg["hidden"])
    model.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
    model.eval()
    with open(scl_path, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def _apply_log(arr):
    arr = arr.copy()
    log_idx = [SYS_COLS.index(c) for c in LOG_SYS_COLS if c in SYS_COLS]
    arr[:, log_idx] = np.log(np.clip(arr[:, log_idx], 1e-10, None))
    return arr


def _compute_grid(system_params, mm_res, am_res):
    """Replicate grid construction from batch_leapfrog / hnn_inference_hill (post Fix A)."""
    ms_solar = float(system_params["ms_solar"])
    mp_earth = float(system_params["mp_earth"])
    ap_AU    = float(system_params["ap_AU"])
    ep       = float(system_params.get("ep", 0.0))
    rs_solar = float(system_params.get("rs_solar", 1.0))
    Ts       = float(system_params.get("Ts", 5772.0))

    ms_gp = ms_solar * FOUR_PI2
    mp_gp = mp_earth * (merth / msun) * FOUR_PI2
    rhill_AU = ap_AU * (1.0 - ep) * (mp_gp / (3.0 * ms_gp)) ** (1.0 / 3.0)

    mp_kg   = mp_earth * merth
    rp_m    = (0.75 * mp_kg / (np.pi * _PLANET_DENSITY_SI)) ** (1.0 / 3.0)
    a_roche = 2.456 * rp_m * (_PLANET_DENSITY_CGS / _MOON_DENSITY_CGS) ** (1.0 / 3.0) / au

    am_min = max(a_roche / rhill_AU, 1e-3)
    mm_max = min(mp_earth, 3.0)

    mm_grid = np.exp(np.linspace(np.log(0.107), np.log(mm_max), mm_res))
    am_grid = np.linspace(am_min, 1.0, am_res)

    a_inner_au, a_outer_au = hz_bounds_au(Ts, rs_solar * _rsun)

    return mm_grid, am_grid, rhill_AU, mp_gp, a_inner_au, a_outer_au


def _t_sim_from_orbits(mm_grid, am_grid, rhill_AU, mp_gp, n_orbits, am_res, mm_res):
    """T_moon of the median cell × n_orbits — same formula as the batch functions."""
    am_ref_AU = float(am_grid[am_res // 2]) * rhill_AU
    mm_mid    = float(mm_grid[mm_res // 2])
    mu_mm_ref = mm_mid * (merth / msun) * FOUR_PI2
    T_moon_ref = 2.0 * np.pi * np.sqrt(am_ref_AU ** 3 / (mp_gp + mu_mm_ref))
    return float(n_orbits) * T_moon_ref


def _mlp_eligible_mask(mlp_dir, system_params, mm_grid, am_grid,
                        rhill_AU, a_inner_au, a_outer_au, t_sim,
                        moon_retrograde=False, em=0.0):
    """
    Run AuxMLPBinary on all N=mm_res*am_res cells.
    Returns boolean (N,) mask: True = MLP predicts both stable AND habitable.
    Returns all-True mask (no filtering) if MLP checkpoint not found.
    """
    N = len(mm_grid) * len(am_grid)
    mm_res, am_res = len(mm_grid), len(am_grid)

    if not os.path.isfile(os.path.join(mlp_dir, "aux_mlp_binary.pt")):
        print(f"  [MLP] checkpoint not found at {mlp_dir} — no eligibility filtering")
        return np.ones(N, dtype=bool)

    model, scaler = _load_aux_mlp(mlp_dir)

    ms_solar = float(system_params["ms_solar"])
    rs_solar = float(system_params.get("rs_solar", 1.0))
    Ts       = float(system_params.get("Ts", 5772.0))
    mp_earth = float(system_params["mp_earth"])
    ap_AU    = float(system_params["ap_AU"])
    ep       = float(system_params.get("ep", 0.0))

    # cell k = mm_idx*am_res + am_idx
    mm_per_cell = np.repeat(mm_grid, am_res)   # (N,)
    am_per_cell = np.tile(am_grid,   mm_res)   # (N,)

    X = np.column_stack([
        np.full(N, ms_solar),
        np.full(N, rs_solar),
        np.full(N, Ts),
        np.full(N, mp_earth),
        np.full(N, ap_AU),
        np.full(N, ep),
        mm_per_cell,
        am_per_cell,
        np.full(N, em),
        np.full(N, float(moon_retrograde)),
        np.full(N, t_sim),
        np.full(N, rhill_AU),
        np.full(N, a_inner_au),
        np.full(N, a_outer_au),
    ]).astype(np.float32)   # (N, 14) in SYS_COLS order

    X = _apply_log(X).astype(np.float32)
    X_scaled = scaler.transform(X).astype(np.float32)

    with torch.no_grad():
        logits = model(torch.tensor(X_scaled))  # (N, 2)
    preds = (logits > 0).numpy()                # (N, 2): [:,0]=stable, [:,1]=habitable

    eligible = preds[:, 0] & preds[:, 1]
    n_elig   = eligible.sum()
    print(f"  [MLP] eligible cells: {n_elig}/{N}  ({100*n_elig/N:.1f}%)")
    return eligible


def _load_meta(fname):
    path = os.path.join(SRC, "ground_truth_grids", fname)
    with open(path) as f:
        return json.load(f)


def _bench_system(name, meta_fname, moon_retrograde=False):
    meta = _load_meta(meta_fname)
    sp   = meta["system_params"]
    system_params = {k: sp[k] for k in ("ms_solar", "rs_solar", "Ts", "mp_earth", "ap_AU")}
    system_params["ep"] = sp.get("ep", 0.0)

    print(f"\n{'=' * 70}")
    print(f"  SYSTEM: {name}")
    print(f"  {system_params}")
    print(f"{'=' * 70}")

    # ── Pre-compute shared grid and t_sim_eff ─────────────────────────────
    mm_grid, am_grid, rhill_AU, mp_gp, a_inner_au, a_outer_au = _compute_grid(
        system_params, MM_RES, AM_RES,
    )
    t_sim_eff = _t_sim_from_orbits(
        mm_grid, am_grid, rhill_AU, mp_gp, N_ORBITS, AM_RES, MM_RES,
    )
    print(f"  t_sim_eff = {t_sim_eff:.4f} yr  ({t_sim_eff * 365.25:.1f} days)"
          f"  [n_orbits={N_ORBITS} × T_moon_ref]")

    # ── Layer 1: MLP eligibility mask ─────────────────────────────────────
    eligible = _mlp_eligible_mask(
        MLP_DIR, system_params, mm_grid, am_grid,
        rhill_AU, a_inner_au, a_outer_au, t_sim_eff,
        moon_retrograde=moon_retrograde,
    )
    n_elig = int(eligible.sum())

    # ── Layer 2A: GT batch leapfrog (eligible cells only) ─────────────────
    print(f"\nA) GT batch leapfrog  (eligible {n_elig}/{N_CELLS}, per-cell k=10 dt)")
    t0 = time.perf_counter()
    r_gt = batch_leapfrog_trajectories(
        system_params=system_params, t_sim=t_sim_eff,
        mm_resolution=MM_RES, am_resolution=AM_RES, n_steps=1000,
        n_orbits=None,
        eligible_mask=eligible,
    )
    t_gt = time.perf_counter() - t0
    t_grid_gt = r_gt.get("t_grid")
    t_sim_gt  = float(t_grid_gt[-1]) if (t_grid_gt is not None and len(t_grid_gt) > 0) else t_sim_eff
    print(f"   n_phys={r_gt['n_phys']:,}  dt_min={r_gt['dt_phys']:.2e} yr  n_out={r_gt['n_out']}")
    print(f"   Effective t_sim = {t_sim_gt:.4f} yr  ({t_sim_gt * 365.25:.1f} days)")
    print(f"   Elapsed: {t_gt:.1f} s  ({t_gt / N_CELLS * 1e3:.2f} ms/cell  |  "
          f"{t_gt / max(n_elig, 1) * 1e3:.2f} ms/eligible-cell)")
    both_gt = np.array(r_gt["map_both"])
    print(f"   stable={np.array(r_gt['map_stable']).mean():.3f}  "
          f"habitable={np.array(r_gt['map_habitable']).mean():.3f}  "
          f"both={both_gt.mean():.3f}  "
          f"both/eligible={both_gt.sum() / max(n_elig, 1):.3f}")

    # ── Layer 2B: HNN hinge4 batch (eligible cells only) ──────────────────
    print(f"\nB) HNN hinge4 batch   (eligible {n_elig}/{N_CELLS}, per-cell k=10 dt)")
    t0 = time.perf_counter()
    r_hnn = batch_hnn_hill_trajectories(
        system_params=system_params, t_sim=t_sim_eff,
        mm_resolution=MM_RES, am_resolution=AM_RES, n_steps=1000,
        model_dir=HINGE4_DIR,
        n_orbits=None,
        eligible_mask=eligible,
    )
    t_hnn = time.perf_counter() - t0

    if not r_hnn["ok"]:
        print(f"   ERROR: {r_hnn.get('error')} — {r_hnn.get('message')}")
        return

    t_grid_hnn = r_hnn.get("t_grid")
    t_sim_hnn  = float(t_grid_hnn[-1]) if (t_grid_hnn is not None and len(t_grid_hnn) > 0) else t_sim_eff
    print(f"   n_phys={r_hnn['n_phys']:,}  dt_min={r_hnn['dt_phys']:.2e} yr  n_out={r_hnn['n_out']}")
    print(f"   Effective t_sim = {t_sim_hnn:.4f} yr  ({t_sim_hnn * 365.25:.1f} days)")
    print(f"   Elapsed: {t_hnn:.1f} s  ({t_hnn / N_CELLS * 1e3:.2f} ms/cell  |  "
          f"{t_hnn / max(n_elig, 1) * 1e3:.2f} ms/eligible-cell)")
    both_hnn = np.array(r_hnn["map_both"])
    print(f"   stable={np.array(r_hnn['map_stable']).mean():.3f}  "
          f"habitable={np.array(r_hnn['map_habitable']).mean():.3f}  "
          f"both={both_hnn.mean():.3f}  "
          f"both/eligible={both_hnn.sum() / max(n_elig, 1):.3f}")

    ratio = t_hnn / t_gt
    print(f"\n   Speed ratio HNN/GT : {ratio:.2f}x  "
          f"({'faster' if ratio < 1 else 'slower'} than GT)")
    print(f"   GT  valid_mm_range : {r_gt.get('valid_mm_range')}")
    print(f"   HNN valid_mm_range : {r_hnn.get('valid_mm_range')}")


if __name__ == "__main__":
    _bench_system("Kepler-452b-v2 prograde",  "Kepler_452_b_v2_prograde.meta.json",  moon_retrograde=False)
    _bench_system("Kepler-1229b prograde",     "Kepler_1229_b_prograde.meta.json",    moon_retrograde=False)

    print("\nDone.")
