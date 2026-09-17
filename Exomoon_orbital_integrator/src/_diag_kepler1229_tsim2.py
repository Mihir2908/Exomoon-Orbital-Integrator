import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch

from exomoon.params import SystemParams
from exomoon.initial_conditions import initial_state
from exomoon.simulation import run_simulation_for_years
from exomoon.habitable_zone import hz_bounds_au
from exomoon.constants import rsun
from exomoon.ml.model import MoonRNN, SYS_DIM
from exomoon.ml.dataset import load_normalizer
from exomoon.ml.inference import predict_stability_map

p = SystemParams(
    Ts=3784.0, rs_solar=0.51, ms_solar=0.54,
    mp_earth=2.54, ap_AU=0.301, ep=0.0,
    mm_earth=1.02, am_hill=0.11, em=0.0, moon_retrograde=False,
)
t_sim = 2.0

st = initial_state(p)
rhill = st["rhill_AU"]
a_in, a_out = hz_bounds_au(p.Ts, p.rs_solar * rsun)
print(f"rhill_AU={rhill:.6f}  a_inner_au={a_in:.6f}  a_outer_au={a_out:.6f}")
print()

# ── ground truth via real leapfrog integrator ──────────────────────────────
sim = run_simulation_for_years(p, t_sim)
traj = sim["traj"]
moon_planet_dist = np.linalg.norm(traj["xyzarr_mm"] - traj["xyzarr_mp"], axis=1)
moon_star_dist    = np.linalg.norm(traj["xyzarr_mm"] - traj["xyzarr_ms"], axis=1)
print(f"TRUE sim: max moon_planet_dist={moon_planet_dist.max():.6f}  (rhill={rhill:.6f})")
print(f"TRUE sim: stable for whole sim: {bool(np.all(moon_planet_dist <= rhill))}")
print(f"TRUE sim: moon_star_dist range [{moon_star_dist.min():.6f}, {moon_star_dist.max():.6f}]")
print(f"TRUE sim: habitable for whole sim: {bool(np.all((moon_star_dist >= a_in) & (moon_star_dist <= a_out)))}")
print()

# ── grid sweep for this system ──────────────────────────────────────────────
system_params = {"ms_solar": p.ms_solar, "rs_solar": p.rs_solar, "Ts": p.Ts,
                  "mp_earth": p.mp_earth, "ap_AU": p.ap_AU, "ep": p.ep}
result = predict_stability_map(
    system_params=system_params, t_sim=t_sim, moon_retrograde=False, em=0.0,
    mm_resolution=30, am_resolution=30, model_dir="models", n_steps=1000, rnn_type="gru",
)
if not result.get("ok"):
    print("GRID SWEEP FAILED:", result)
else:
    mm_grid = np.array(result["mm_grid"])
    am_grid = np.array(result["am_grid"])
    map_both = np.array(result["map_both"])
    map_stable = np.array(result["map_stable"])
    map_habitable = np.array(result["map_habitable"])
    print(f"map_stable sum: {map_stable.sum()}/{map_stable.size}  "
          f"map_habitable sum: {map_habitable.sum()}/{map_habitable.size}  "
          f"map_both sum: {map_both.sum()}/{map_both.size}")
    print(f"valid_mm_range: {result['valid_mm_range']}")
    # closest grid point to mm_earth=1.02, am_hill=0.11
    mi = int(np.argmin(np.abs(mm_grid - 1.02)))
    ai = int(np.argmin(np.abs(am_grid - 0.11)))
    print(f"Closest grid point: mm={mm_grid[mi]:.4f} am={am_grid[ai]:.4f}  "
          f"stable={map_stable[mi, ai]} habitable={map_habitable[mi, ai]}")
print()

# ── single-candidate rollout at the EXACT point (mm=1.02, am=0.11) ─────────
model = MoonRNN.load("models", rnn_type="gru")
model.eval()
sys_scaler, state_scaler = load_normalizer(os.path.join("models", "normalizer.pkl"))

pos_mp, pos_ms, pos_mm = st["pos_mp"], st["pos_ms"], st["pos_mm"]
vel_mp, vel_mm = st["vel_mp"], st["vel_mm"]
moon_planet_dist0 = float(np.linalg.norm(pos_mm - pos_mp))
planet_star_dist0 = float(np.linalg.norm(pos_mp - pos_ms))
moon_star_dist0   = float(np.linalg.norm(pos_mm - pos_ms))
moon_speed0       = float(np.linalg.norm(vel_mm))
planet_speed0     = float(np.linalg.norm(vel_mp))
hz_width = max(a_out - a_in, 1e-6)

state_raw = np.array([[
    moon_planet_dist0 / rhill,
    np.arcsinh((moon_star_dist0 - a_in) / hz_width),
    planet_star_dist0, moon_speed0, planet_speed0, 0.0,
]], dtype=np.float32)
sys_raw = np.array([[
    p.ms_solar, p.rs_solar, p.Ts, p.mp_earth, p.ap_AU, p.ep,
    p.mm_earth, p.am_hill, p.em, float(p.moon_retrograde),
    t_sim, rhill, a_in, a_out,
]], dtype=np.float32)

state_norm = state_scaler.transform(state_raw).astype(np.float32)
sys_norm   = sys_scaler.transform(sys_raw).astype(np.float32)
state_t = torch.from_numpy(state_norm).unsqueeze(1)
sys_t   = torch.from_numpy(sys_norm).unsqueeze(1)

n_steps = 1000
h = torch.zeros(model.layers, 1, model.hidden)
t_fracs = np.linspace(0.0, 1.0, n_steps)

dn_hist, sn_hist, sl_hist, hl_hist = [], [], [], []
with torch.no_grad():
    for step in range(n_steps):
        rnn_in = torch.cat([state_t, sys_t], dim=-1)
        rnn_out, h = model.rnn(rnn_in, h)
        raw_dist = model.head_dist(rnn_out)
        dist_pred = torch.cat([
            torch.relu(raw_dist[..., 0:1]), raw_dist[..., 1:2], torch.relu(raw_dist[..., 2:5]),
        ], dim=-1)
        flag_logits = model.head_flags(rnn_out)
        dist_np = dist_pred.squeeze(1).numpy()
        flags_np = flag_logits.squeeze(1).numpy()
        dn_hist.append(dist_np[0, 0]); sn_hist.append(dist_np[0, 1])
        sl_hist.append(flags_np[0, 0]); hl_hist.append(flags_np[0, 1])
        next_state_raw = np.concatenate([dist_np, np.full((1, 1), t_fracs[step])], axis=1).astype(np.float32)
        state_t = torch.from_numpy(state_scaler.transform(next_state_raw)).unsqueeze(1)

dn_hist = np.array(dn_hist); sn_hist = np.array(sn_hist)
sl_hist = np.array(sl_hist); hl_hist = np.array(hl_hist)
print(f"min stable_logit={sl_hist.min():.3f} at step {sl_hist.argmin()}  "
      f"min habitable_logit={hl_hist.min():.3f} at step {hl_hist.argmin()}")
print()
print("step | moon_planet_dist_norm | moon_star_dist_norm | stable_logit | habitable_logit")
for i in range(0, n_steps, 50):
    print(f"{i:4d} | {dn_hist[i]:.4f} | {sn_hist[i]:.4f} | {sl_hist[i]:+.3f} | {hl_hist[i]:+.3f}")
