"""
benchmark_hnn_unified.py — Unified HNN architecture timing sweep.

All architectures use the SAME code path:
  - Models instantiated directly as nn.Module objects (no save/load/monkey-patching)
  - Integration loop runs inline (no batch_hnn_trajectories wrapper)
  - Both dt_factor=50 and dt_factor=10 measured directly, no extrapolation

ISOLATION: does not touch any MLP infrastructure.
"""

import os, sys, json, time, math, tempfile
import numpy as np
import torch
import torch.nn as nn

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from exomoon.params             import SystemParams
from exomoon.initial_conditions import initial_state
from exomoon.habitable_zone     import hz_bounds_au

GT_META_PATH = os.path.join(SRC, "ground_truth_grids",
                             "Kepler_452_b_v2_prograde.meta.json")
with open(GT_META_PATH) as f:
    gt_meta = json.load(f)

sp = gt_meta["system_params"]
SYSTEM_PARAMS = dict(
    ms_solar=sp["ms_solar"], rs_solar=sp["rs_solar"],
    Ts=sp["Ts"], mp_earth=sp["mp_earth"],
    ap_AU=sp["ap_AU"], ep=sp.get("ep", 0.0),
)
T_SIM    = float(gt_meta["t_sim"])
MM_RES   = AM_RES = 50
N_CELLS  = MM_RES * AM_RES
_MERTH   = 5.976e24 / 1.989e30   # Earth/Sun mass ratio
_G       = 4.0 * math.pi ** 2    # AU³/(yr²·M☉)
F32      = torch.float32
F64      = torch.float64


# ── Minimal HNN: any layer sizes, same accel() interface ──────────────────────

def make_hnn(layer_sizes: list) -> nn.Module:
    """Build an HNN MLP with Tanh activations from a list of layer widths.
    layer_sizes example: [15, 128, 64, 1]  →  15→128→64→1
    """
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        if i < len(layer_sizes) - 2:
            layers.append(nn.Tanh())
    net = nn.Sequential(*layers)

    class _HNN(nn.Module):
        def forward(self, q, sys_p):
            return net(torch.cat([q, sys_p], dim=-1)).squeeze(-1)

        def accel(self, q, sys_p):
            with torch.enable_grad():
                q_in = q.detach().requires_grad_(True)
                V    = self.forward(q_in, sys_p)
                dV   = torch.autograd.grad(V.sum(), q_in)[0]
            return -dV.detach()

    m = _HNN()
    m.net = net
    return m


# ── Build initial states for all 2500 cells ───────────────────────────────────

def build_initial_states(system_params, t_sim, mm_res, am_res):
    ms  = system_params["ms_solar"]
    rs  = system_params["rs_solar"]
    Ts  = system_params["Ts"]
    mp  = system_params["mp_earth"]
    ap  = system_params["ap_AU"]
    ep  = system_params["ep"]

    # Hill radius: a_p*(1-e_p)*(M_p/(3*M_s))^(1/3) in solar-mass units
    mp_msun  = mp * _MERTH
    rhill    = ap * (1.0 - ep) * (mp_msun / (3.0 * ms)) ** (1.0 / 3.0)
    a_in, a_out = hz_bounds_au(Ts, rs * 6.957e8)

    mm_min   = 0.107
    mm_max   = min(mp * 0.30, 0.5)
    mm_grid  = np.geomspace(mm_min, mm_max, mm_res)

    roche_frac = 0.012 / rhill if rhill > 0 else 0.05
    am_grid  = np.linspace(roche_frac, 1.0, am_res)

    # Reference moon period for dt
    ref_p = SystemParams(
        ms_solar=ms, rs_solar=rs, Ts=Ts, mp_earth=mp,
        ap_AU=ap, ep=ep, mm_earth=mm_grid[mm_res // 2],
        am_hill=am_grid[am_res // 2], em=0.0,
        moon_retrograde=False, dp_cgs=5.51,
    )
    ref_state = initial_state(ref_p)
    am_AU_ref = am_grid[am_res // 2] * rhill
    mu_mp     = _G * (mp * _MERTH + mm_grid[mm_res // 2] * _MERTH)
    T_moon    = 2 * math.pi * math.sqrt(am_AU_ref ** 3 / mu_mp)
    dt_base   = min(T_moon / 100.0, 1.0 / 20000.0)

    # Build per-cell initial positions and velocities
    q_ms = np.zeros((N_CELLS, 3), dtype=np.float64)
    q_mp = np.zeros((N_CELLS, 3), dtype=np.float64)
    q_mm = np.zeros((N_CELLS, 3), dtype=np.float64)
    v_ms = np.zeros((N_CELLS, 3), dtype=np.float64)
    v_mp = np.zeros((N_CELLS, 3), dtype=np.float64)
    v_mm = np.zeros((N_CELLS, 3), dtype=np.float64)
    sys_raw = np.zeros((N_CELLS, 6), dtype=np.float64)

    idx = 0
    for mm_e in mm_grid:
        for am_h in am_grid:
            p = SystemParams(
                ms_solar=ms, rs_solar=rs, Ts=Ts, mp_earth=mp,
                ap_AU=ap, ep=ep, mm_earth=mm_e, am_hill=am_h,
                em=0.0, moon_retrograde=False, dp_cgs=5.51,
            )
            s = initial_state(p)
            q_ms[idx] = s["pos_ms"]; q_mp[idx] = s["pos_mp"]; q_mm[idx] = s["pos_mm"]
            v_ms[idx] = s["vel_ms"]; v_mp[idx] = s["vel_mp"]; v_mm[idx] = s["vel_mm"]
            sys_raw[idx] = [ms, mp, mm_e, rhill, a_in, a_out]
            idx += 1

    return (q_ms, q_mp, q_mm, v_ms, v_mp, v_mm, sys_raw, dt_base,
            mm_grid, am_grid, rhill, a_in, a_out)


# ── Integration loop (identical for every model) ──────────────────────────────

def run_integration(model, q_ms, q_mp, q_mm, v_ms, v_mp, v_mm,
                    sys_enc, dt, n_phys, n_out):
    """Velocity-Verlet leapfrog using model.accel(). Returns elapsed seconds."""
    half_dt  = dt / 2.0

    q_ms = torch.tensor(q_ms, dtype=F64)
    q_mp = torch.tensor(q_mp, dtype=F64)
    q_mm = torch.tensor(q_mm, dtype=F64)
    v_ms = torch.tensor(v_ms, dtype=F64)
    v_mp = torch.tensor(v_mp, dtype=F64)
    v_mm = torch.tensor(v_mm, dtype=F64)

    def _cat_q():
        return torch.cat([q_ms, q_mp, q_mm], dim=1).to(F32)

    def _accel():
        a = model.accel(_cat_q(), sys_enc).to(F64)
        return a[:, 0:3], a[:, 3:6], a[:, 6:9]

    stride  = max(1, n_phys // n_out)
    out_idx = 0
    traj_mm = torch.zeros(n_out, N_CELLS, 3, dtype=F64)

    t0 = time.perf_counter()

    with torch.no_grad():
        a_s, a_p, a_m = _accel()
        for step in range(n_phys):
            v_ms = v_ms + a_s * half_dt
            v_mp = v_mp + a_p * half_dt
            v_mm = v_mm + a_m * half_dt
            q_ms = q_ms + v_ms * dt
            q_mp = q_mp + v_mp * dt
            q_mm = q_mm + v_mm * dt
            a_s, a_p, a_m = _accel()
            v_ms = v_ms + a_s * half_dt
            v_mp = v_mp + a_p * half_dt
            v_mm = v_mm + a_m * half_dt
            if step % stride == 0 and out_idx < n_out:
                traj_mm[out_idx] = q_mm
                out_idx += 1

    return time.perf_counter() - t0


# ── Architectures: all specified as layer_sizes lists ─────────────────────────

ARCHS = [
    ("15→16→1",          [15, 16, 1]),
    ("15→32→1",          [15, 32, 1]),
    ("15→32→32→1",       [15, 32, 32, 1]),
    ("15→64→64→1",       [15, 64, 64, 1]),
    ("15→128→64→1",      [15, 128, 64, 1]),
    ("15→128→128→1",     [15, 128, 128, 1]),
]

DT_FACTORS = [50, 10]

print(f"Unified HNN architecture sweep — identical code path for all models")
print(f"No save/load/monkey-patching. Integration loop runs inline.")
print(f"System: K-452b-v2  |  {N_CELLS} cells  |  t_sim={T_SIM} yr")
print(f"batch_leapfrog reference: 463.52s  (dt_factor=1, n_phys≈200,000, 2.30 ms/step)")
print()
print("Building initial states for all 2500 cells...")
(q_ms0, q_mp0, q_mm0, v_ms0, v_mp0, v_mm0,
 sys_raw, dt_base, mm_grid, am_grid,
 rhill, a_in, a_out) = build_initial_states(SYSTEM_PARAMS, T_SIM, MM_RES, AM_RES)
print(f"  dt_base = {dt_base:.6f} yr")
print()

# results[label][dt_factor] = (elapsed, n_phys, ms_per_step)
results = {label: {} for label, _ in ARCHS}

for label, layer_sizes in ARCHS:
    model  = make_hnn(layer_sizes)
    n_param = sum(p.numel() for p in model.parameters())

    # StandardScaler fitted on dummy data (timing only — weights are random)
    from sklearn.preprocessing import StandardScaler
    dummy = np.array([
        [1.0, 3.0,   0.2, 0.018, 0.99, 1.80],
        [0.5, 10.0,  0.5, 0.005, 0.20, 0.40],
        [1.5, 100.0, 1.0, 0.050, 1.50, 2.80],
    ])
    scaler = StandardScaler().fit(dummy)
    sys_enc = torch.tensor(
        scaler.transform(sys_raw).astype(np.float32), dtype=F32
    )

    print(f"  {label}  (params={n_param:,})")
    for dt_factor in DT_FACTORS:
        dt     = dt_base * dt_factor
        n_phys = max(1, int(math.ceil(T_SIM / dt)))

        elapsed = run_integration(
            model,
            q_ms0.copy(), q_mp0.copy(), q_mm0.copy(),
            v_ms0.copy(), v_mp0.copy(), v_mm0.copy(),
            sys_enc, dt, n_phys, n_out=1000,
        )
        ms_step = elapsed / n_phys * 1000
        results[label][dt_factor] = (elapsed, n_phys, ms_step)
        print(f"    dt_factor={dt_factor:>2}  n_phys={n_phys:>7}  "
              f"elapsed={elapsed:>7.2f}s  {ms_step:.2f} ms/step")
    print()

# ── Summary table ─────────────────────────────────────────────────────────────
W = 96
print("─" * W)
print(f"  {'Architecture':<22}  {'Params':>7}  "
      f"{'dt50 elapsed':>13}  {'ms/step':>8}  "
      f"{'dt10 elapsed':>13}  {'ms/step':>8}  {'dt10/dt50':>10}")
print("─" * W)
for label, layer_sizes in ARCHS:
    n_param = sum(p.numel() for p in make_hnn(layer_sizes).parameters())
    e50, n50, ms50 = results[label][50]
    e10, n10, ms10 = results[label][10]
    ratio = e10 / e50
    print(f"  {label:<22}  {n_param:>7,}  "
          f"{e50:>12.2f}s  {ms50:>8.2f}  "
          f"{e10:>12.2f}s  {ms10:>8.2f}  {ratio:>9.2f}×")
print("─" * W)
print(f"  {'batch_leapfrog':<22}  {'—':>7}  "
      f"{'—':>13}  {'—':>8}  "
      f"{'463.52s':>13}  {'2.30':>8}  "
      f"{'(ref)':>10}  ← dt_factor=1, n_phys≈200,000")
print("─" * W)
