"""
Diagnostic: compare the GRU's autoregressive rollout against the real leapfrog
simulation for a single, deliberately easy/stable configuration (small am_hill,
well inside the Hill sphere). If "no valid orbits found" is happening even here,
it tells us whether the problem is genuine autoregressive drift in the model's
continuous distance predictions, or something else.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

from exomoon.params import SystemParams
from exomoon.initial_conditions import initial_state
from exomoon.simulation import run_simulation_for_years
from exomoon.ml.model import MoonRNN, SYS_DIM, N_DIST, N_FLAGS
from exomoon.ml.dataset import load_normalizer

MODEL_DIR = "models"
RNN_TYPE  = "gru"

# A deliberately easy, should-be-obviously-stable config
p = SystemParams(
    Ts=5772.0, rs_solar=1.0, ms_solar=1.0,
    mp_earth=1.0, ap_AU=1.0, ep=0.0,
    mm_earth=0.107, am_hill=0.05, em=0.0, moon_retrograde=False,
)
t_sim = 10.0

# ── ground truth via real leapfrog integrator ──────────────────────────────
sim = run_simulation_for_years(p, t_sim)
rhill = sim["state"]["rhill_AU"]
traj = sim["traj"]
moon_planet_dist_true = np.linalg.norm(traj["xyzarr_mm"] - traj["xyzarr_mp"], axis=1)
print(f"rhill_AU = {rhill:.6f}")
print(f"TRUE simulation: max moon_planet_dist over {len(moon_planet_dist_true)} steps = {moon_planet_dist_true.max():.6f}")
print(f"TRUE simulation: stable (max_dist <= rhill) = {moon_planet_dist_true.max() <= rhill}")
print()

# ── GRU rollout for the same config ────────────────────────────────────────
model = MoonRNN.load(MODEL_DIR, rnn_type=RNN_TYPE)
model.eval()
print(f"model.state_dim = {model.state_dim}  (6 = new architecture)")

sys_scaler, state_scaler = load_normalizer(os.path.join(MODEL_DIR, "normalizer.pkl"))

st = initial_state(p)
pos_mp, pos_ms, pos_mm = st["pos_mp"], st["pos_ms"], st["pos_mm"]
vel_mp, vel_mm = st["vel_mp"], st["vel_mm"]

moon_planet_dist0 = float(np.linalg.norm(pos_mm - pos_mp))
planet_star_dist0 = float(np.linalg.norm(pos_mp - pos_ms))
moon_star_dist0   = float(np.linalg.norm(pos_mm - pos_ms))
moon_speed0       = float(np.linalg.norm(vel_mm))
planet_speed0     = float(np.linalg.norm(vel_mp))

state_raw = np.array([[moon_planet_dist0, moon_star_dist0, planet_star_dist0,
                        moon_speed0, planet_speed0, 0.0]], dtype=np.float32)
sys_raw = np.array([[p.ms_solar, p.rs_solar, p.Ts, p.mp_earth, p.ap_AU, p.ep,
                      p.mm_earth, p.am_hill, p.em, float(p.moon_retrograde),
                      t_sim, rhill]], dtype=np.float32)

state_norm = state_scaler.transform(state_raw).astype(np.float32)
sys_norm   = sys_scaler.transform(sys_raw).astype(np.float32)

state_t = torch.from_numpy(state_norm).unsqueeze(1)   # (1,1,6)
sys_t   = torch.from_numpy(sys_norm).unsqueeze(1)      # (1,1,12)

n_steps = 1000
h = torch.zeros(model.layers, 1, model.hidden) if model.rnn_type == "gru" else \
    (torch.zeros(model.layers, 1, model.hidden), torch.zeros(model.layers, 1, model.hidden))

t_fracs = np.linspace(0.0, 1.0, n_steps)
dist_history = []
stable_logit_history = []

with torch.no_grad():
    for step in range(n_steps):
        rnn_in = torch.cat([state_t, sys_t], dim=-1)
        rnn_out, h = model.rnn(rnn_in, h)
        dist_pred = torch.relu(model.head_dist(rnn_out))
        flag_logits = model.head_flags(rnn_out)

        dist_np  = dist_pred.squeeze(1).numpy()
        flags_np = flag_logits.squeeze(1).numpy()

        dist_history.append(dist_np[0, 0])   # moon_planet_dist
        stable_logit_history.append(flags_np[0, 0])

        next_state_raw = np.concatenate([dist_np, np.full((1, 1), t_fracs[step])], axis=1).astype(np.float32)
        next_state_norm = state_scaler.transform(next_state_raw)
        state_t = torch.from_numpy(next_state_norm).unsqueeze(1)

dist_history = np.array(dist_history)
stable_logit_history = np.array(stable_logit_history)

print()
print(f"GRU rollout: max predicted moon_planet_dist = {dist_history.max():.6f}  (rhill={rhill:.6f})")
print(f"GRU rollout: first step where predicted dist > rhill: "
      f"{np.argmax(dist_history > rhill) if (dist_history > rhill).any() else 'never'}")
print(f"GRU rollout: first step where stable_logit < 0: "
      f"{np.argmax(stable_logit_history < 0) if (stable_logit_history < 0).any() else 'never'}")
print()
print("Sampled trajectory (every 50 steps): step, pred_dist, stable_logit")
for i in range(0, n_steps, 50):
    print(f"  step {i:4d}: dist={dist_history[i]:.6f}  stable_logit={stable_logit_history[i]:+.3f}")
