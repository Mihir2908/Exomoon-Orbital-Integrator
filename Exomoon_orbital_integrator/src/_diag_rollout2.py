"""
Same as _diag_rollout.py but tracks BOTH stable_logit and habitable_logit across
the full 1000-step rollout, to find the exact step (if any) where either first
goes negative -- this is what trips ever_unstable/ever_uninhabited in inference.py.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

from exomoon.params import SystemParams
from exomoon.initial_conditions import initial_state
from exomoon.ml.model import MoonRNN
from exomoon.ml.dataset import load_normalizer

MODEL_DIR = "models"

p = SystemParams(
    Ts=5772.0, rs_solar=1.0, ms_solar=1.0,
    mp_earth=1.0, ap_AU=1.0, ep=0.0,
    mm_earth=0.107, am_hill=0.05, em=0.0, moon_retrograde=False,
)
t_sim = 10.0

model = MoonRNN.load(MODEL_DIR, rnn_type="gru")
model.eval()
sys_scaler, state_scaler = load_normalizer(os.path.join(MODEL_DIR, "normalizer.pkl"))

st = initial_state(p)
pos_mp, pos_ms, pos_mm = st["pos_mp"], st["pos_ms"], st["pos_mm"]
vel_mp, vel_mm = st["vel_mp"], st["vel_mm"]
rhill = st["rhill_AU"]

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

state_t = torch.from_numpy(state_norm).unsqueeze(1)
sys_t   = torch.from_numpy(sys_norm).unsqueeze(1)

n_steps = 1000
h = torch.zeros(model.layers, 1, model.hidden)
t_fracs = np.linspace(0.0, 1.0, n_steps)

dist_hist  = []
mstar_hist = []
stable_logit_hist = []
habit_logit_hist  = []

with torch.no_grad():
    for step in range(n_steps):
        rnn_in = torch.cat([state_t, sys_t], dim=-1)
        rnn_out, h = model.rnn(rnn_in, h)
        dist_pred = torch.relu(model.head_dist(rnn_out))
        flag_logits = model.head_flags(rnn_out)

        dist_np  = dist_pred.squeeze(1).numpy()
        flags_np = flag_logits.squeeze(1).numpy()

        dist_hist.append(dist_np[0, 0])
        mstar_hist.append(dist_np[0, 1])
        stable_logit_hist.append(flags_np[0, 0])
        habit_logit_hist.append(flags_np[0, 1])

        next_state_raw = np.concatenate([dist_np, np.full((1, 1), t_fracs[step])], axis=1).astype(np.float32)
        next_state_norm = state_scaler.transform(next_state_raw)
        state_t = torch.from_numpy(next_state_norm).unsqueeze(1)

stable_logit_hist = np.array(stable_logit_hist)
habit_logit_hist  = np.array(habit_logit_hist)
dist_hist  = np.array(dist_hist)
mstar_hist = np.array(mstar_hist)

print(f"rhill_AU = {rhill:.6f}")
first_unstable = np.argmax(stable_logit_hist < 0) if (stable_logit_hist < 0).any() else None
first_uninhab  = np.argmax(habit_logit_hist < 0) if (habit_logit_hist < 0).any() else None
print(f"First step stable_logit < 0:    {first_unstable}")
print(f"First step habitable_logit < 0: {first_uninhab}")
print(f"min stable_logit over rollout:    {stable_logit_hist.min():.4f}  (at step {stable_logit_hist.argmin()})")
print(f"min habitable_logit over rollout: {habit_logit_hist.min():.4f}  (at step {habit_logit_hist.argmin()})")
print()
print("step | dist | moon_star_dist | stable_logit | habitable_logit")
for i in range(0, n_steps, 25):
    print(f"{i:4d} | {dist_hist[i]:.5f} | {mstar_hist[i]:.5f} | {stable_logit_hist[i]:+.3f} | {habit_logit_hist[i]:+.3f}")
