"""Quick sanity check: K=16 per-cell on K-1229b only (~5 min)."""
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
    cfg  = json.load(open(os.path.join(mlp_dir, "model_config.json")))
    mdl  = AuxMLPBinary(input_dim=cfg["input_dim"], hidden=cfg["hidden"])
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
    ap_AU, ep          = float(sp["ap_AU"]), float(sp.get("ep", 0.0))
    rs_solar, Ts       = float(sp.get("rs_solar", 1.0)), float(sp.get("Ts", 5772.0))

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
    a_inner, a_outer = hz_bounds_au(Ts, rs_solar * _rsun)
    return mm_grid, am_grid, rhill_AU, mp_gp, a_inner, a_outer


def _t_sim(mm_grid, am_grid, rhill_AU, mp_gp, n_orbits, am_res, mm_res):
    am_ref = float(am_grid[am_res // 2]) * rhill_AU
    mm_mid = float(mm_grid[mm_res // 2])
    mu_mm  = mm_mid * (merth / msun) * FOUR_PI2
    T_ref  = 2.0 * np.pi * np.sqrt(am_ref**3 / (mp_gp + mu_mm))
    return float(n_orbits) * T_ref


def _mlp_mask(mlp_dir, sp, mm_grid, am_grid, rhill_AU, a_inner, a_outer, t_sim,
              moon_retrograde=False, em=0.0):
    N = len(mm_grid) * len(am_grid)
    mm_res, am_res = len(mm_grid), len(am_grid)
    if not os.path.isfile(os.path.join(mlp_dir, "aux_mlp_binary.pt")):
        return np.ones(N, dtype=bool)
    mdl, scl = _load_aux_mlp(mlp_dir)
    mm_c = np.repeat(mm_grid, am_res)
    am_c = np.tile(am_grid,   mm_res)
    X = np.column_stack([
        np.full(N, sp["ms_solar"]), np.full(N, sp.get("rs_solar", 1.0)),
        np.full(N, sp.get("Ts", 5772.0)), np.full(N, sp["mp_earth"]),
        np.full(N, sp["ap_AU"]), np.full(N, sp.get("ep", 0.0)),
        mm_c, am_c, np.full(N, em), np.full(N, float(moon_retrograde)),
        np.full(N, t_sim), np.full(N, rhill_AU),
        np.full(N, a_inner), np.full(N, a_outer),
    ]).astype(np.float32)
    X = _apply_log(X).astype(np.float32)
    X_sc = scl.transform(X).astype(np.float32)
    with torch.no_grad():
        logits = mdl(torch.tensor(X_sc))
    preds   = (logits > 0).numpy()
    elig    = preds[:, 0] & preds[:, 1]
    print(f"  [MLP] eligible: {elig.sum()}/{N} ({100*elig.sum()/N:.1f}%)")
    return elig


meta = json.load(open(os.path.join(SRC, "ground_truth_grids",
                                   "Kepler_1229_b_prograde.meta.json")))
sp   = {k: meta["system_params"][k] for k in ("ms_solar", "rs_solar", "Ts", "mp_earth", "ap_AU")}
sp["ep"] = meta["system_params"].get("ep", 0.0)

print(f"\n{'='*70}")
print(f"  SANITY CHECK: Kepler-1229b prograde  (K_ORBIT=16)")
print(f"  {sp}")
print(f"{'='*70}")

mm_grid, am_grid, rhill_AU, mp_gp, a_inner, a_outer = _compute_grid(sp, MM_RES, AM_RES)
t_sim_eff = _t_sim(mm_grid, am_grid, rhill_AU, mp_gp, N_ORBITS, AM_RES, MM_RES)
print(f"  t_sim_eff = {t_sim_eff:.4f} yr  ({t_sim_eff*365.25:.1f} days)")

eligible = _mlp_mask(MLP_DIR, sp, mm_grid, am_grid, rhill_AU, a_inner, a_outer, t_sim_eff)
n_elig   = int(eligible.sum())

print(f"\nA) GT batch leapfrog  (K=16, eligible {n_elig}/{N_CELLS})")
t0 = time.perf_counter()
r_gt = batch_leapfrog_trajectories(
    system_params=sp, t_sim=t_sim_eff,
    mm_resolution=MM_RES, am_resolution=AM_RES, n_steps=1000,
    n_orbits=None, eligible_mask=eligible,
)
t_gt = time.perf_counter() - t0
print(f"   n_phys={r_gt['n_phys']:,}  dt={r_gt['dt_phys']:.2e} yr  n_out={r_gt['n_out']}")
print(f"   Elapsed: {t_gt:.1f} s  ({t_gt/N_CELLS*1e3:.2f} ms/cell)")
both_gt = np.array(r_gt["map_both"])
print(f"   stable={np.array(r_gt['map_stable']).mean():.3f}  "
      f"habitable={np.array(r_gt['map_habitable']).mean():.3f}  "
      f"both={both_gt.mean():.3f}  both/eligible={both_gt.sum()/max(n_elig,1):.3f}")
print(f"   GT valid_mm_range: {r_gt.get('valid_mm_range')}")

print(f"\nB) HNN hinge4  (K=16, eligible {n_elig}/{N_CELLS})")
t0 = time.perf_counter()
r_hnn = batch_hnn_hill_trajectories(
    system_params=sp, t_sim=t_sim_eff,
    mm_resolution=MM_RES, am_resolution=AM_RES, n_steps=1000,
    model_dir=HINGE4_DIR, n_orbits=None, eligible_mask=eligible,
)
t_hnn = time.perf_counter() - t0
if r_hnn["ok"]:
    both_hnn = np.array(r_hnn["map_both"])
    print(f"   n_phys={r_hnn['n_phys']:,}  dt={r_hnn['dt_phys']:.2e} yr  n_out={r_hnn['n_out']}")
    print(f"   Elapsed: {t_hnn:.1f} s  ({t_hnn/N_CELLS*1e3:.2f} ms/cell)")
    print(f"   stable={np.array(r_hnn['map_stable']).mean():.3f}  "
          f"habitable={np.array(r_hnn['map_habitable']).mean():.3f}  "
          f"both={both_hnn.mean():.3f}  both/eligible={both_hnn.sum()/max(n_elig,1):.3f}")
    print(f"   Speed ratio HNN/GT: {t_hnn/t_gt:.2f}x")
    print(f"   HNN valid_mm_range: {r_hnn.get('valid_mm_range')}")
else:
    print(f"   ERROR: {r_hnn.get('error')}")

print("\nDone.")
