"""
Quick diagnostic: check what the new model predicts step-by-step for ONE
candidate in the K-452b-v2 prograde system. Print moon_star_dist_norm and
habitable_logit at each step to find when/why ever_uninhabited fires.
"""
import os, sys, json
import numpy as np

SRC = os.path.dirname(__file__)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import torch
from exomoon.ml.model   import MoonRNN, SYS_DIM, STATE_DIM
from exomoon.ml.dataset import load_normalizer, SYS_COLS, STATE_COLS
from exomoon.params     import SystemParams
from exomoon.initial_conditions import initial_state
from exomoon.habitable_zone import hz_bounds_au, moon_effective_temp_K
from exomoon.constants  import rsun, merth, msun, au

ARCSINH_1_OUTER = float(np.arcsinh(1.05))
MODEL_DIR = "models/"

# K-452b-v2 prograde system params (from ground truth meta)
meta_path = "ground_truth_grids/Kepler_452_b_v2_prograde.meta.json"
with open(meta_path) as f:
    meta = json.load(f)
print("System meta:", json.dumps(meta, indent=2))

sp = meta["system_params"]
ms, rs, Ts = sp["ms_solar"], sp["rs_solar"], sp["Ts"]
mp, ap, ep = sp["mp_earth"], sp["ap_AU"], sp["ep"]
t_sim = meta["t_sim"]

rs_m = rs * rsun
a_inner_au, a_outer_au = hz_bounds_au(Ts, rs_m)
hz_width = max(a_outer_au - a_inner_au, 1e-6)
print(f"\nHZ: [{a_inner_au:.4f}, {a_outer_au:.4f}] AU  (width={hz_width:.4f})")
print(f"t_sim={t_sim}, ms={ms}, rs={rs}, Ts={Ts}, mp={mp}, ap={ap}, ep={ep}")

# Load model and scalers
model = MoonRNN.load(MODEL_DIR, rnn_type="gru")
model.eval()
sys_scaler, state_scaler = load_normalizer(os.path.join(MODEL_DIR, "normalizer.pkl"))

print(f"\nstate_scaler means: {dict(zip(STATE_COLS, state_scaler.mean_.tolist()))}")
print(f"state_scaler stds:  {dict(zip(STATE_COLS, state_scaler.scale_.tolist()))}")
print(f"\nsys_scaler means: {dict(zip(SYS_COLS, sys_scaler.mean_.tolist()))}")
print(f"sys_scaler stds:  {dict(zip(SYS_COLS, sys_scaler.scale_.tolist()))}")

# Pick ONE candidate: am_hill=0.3, mm_earth=0.3 (should be stable+habitable prograde)
mm_test, am_test = 0.3, 0.3
p = SystemParams(Ts=Ts, rs_solar=rs, ms_solar=ms, mp_earth=mp, ap_AU=ap, ep=ep,
                 mm_earth=mm_test, am_hill=am_test, em=0.0, moon_retrograde=False)
st = initial_state(p)
rhill = st["rhill_AU"]

pos_mp, pos_ms, pos_mm = st["pos_mp"], st["pos_ms"], st["pos_mm"]
vel_mp, vel_mm = st["vel_mp"], st["vel_mm"]

mpd = float(np.linalg.norm(pos_mm - pos_mp))
psd = float(np.linalg.norm(pos_mp - pos_ms))
msd = float(np.linalg.norm(pos_mm - pos_ms))
mspd = float(np.linalg.norm(vel_mm))
plspd = float(np.linalg.norm(vel_mp))

Tm_K   = moon_effective_temp_K(Ts, rs_m, msd * au)
T_hot  = moon_effective_temp_K(Ts, rs_m, a_inner_au * au)
T_cold = moon_effective_temp_K(Ts, rs_m, a_outer_au * au)
temp_width = max(T_hot - T_cold, 1e-6)

mpdn_raw = mpd / rhill
msdn_raw = float(np.arcsinh((msd - a_inner_au) / hz_width))
tmpn_raw = float(np.arcsinh((Tm_K - T_cold) / temp_width))

state_raw = np.array([[mpdn_raw, msdn_raw, tmpn_raw, psd, mspd, plspd, 0.0]], dtype=np.float32)
sys_raw   = np.array([[ms, rs, Ts, mp, ap, ep, mm_test, am_test, 0.0, 0.0,
                        t_sim, rhill, a_inner_au, a_outer_au]], dtype=np.float32)

print(f"\nInitial raw state: mpdn={mpdn_raw:.4f} msdn={msdn_raw:.4f} tmpn={tmpn_raw:.4f} psd={psd:.4f} mspd={mspd:.6f} plspd={plspd:.6f}")
print(f"moon_star_dist={msd:.4f} AU  (in HZ: {a_inner_au:.4f}-{a_outer_au:.4f})")

state_norm = state_scaler.transform(state_raw)
sys_norm   = sys_scaler.transform(sys_raw)

print(f"\nNormalized initial state: {dict(zip(STATE_COLS, state_norm[0].tolist()))}")

state_t = torch.from_numpy(state_norm).unsqueeze(1)
sys_exp = torch.from_numpy(sys_norm).unsqueeze(1)
h = torch.zeros(model.layers, 1, model.hidden)

n_steps = 1000
t_fracs = np.linspace(0.0, 1.0, n_steps)
warmup = 10

print(f"\nStep | mpdn_raw | msdn_raw | tmpn_raw | flag_hab | flag_hab_temp | dist_uninh | flag_uninh | both_uninh")
ever_uninhabited = False

with torch.no_grad():
    for step in range(min(n_steps, 100)):
        rnn_in = torch.cat([state_t, sys_exp], dim=-1)
        rnn_out, h = model.rnn(rnn_in, h)

        raw_dist = model.head_dist(rnn_out)
        dist_pred = torch.cat([
            torch.relu(raw_dist[..., 0:1]),
            raw_dist[..., 1:3],
            torch.relu(raw_dist[..., 3:6]),
        ], dim=-1)
        flag_logits = model.head_flags(rnn_out)

        dist_np  = dist_pred.squeeze(1).cpu().numpy()[0]
        flags_np = flag_logits.squeeze(1).cpu().numpy()[0]

        mpdn_p = dist_np[0]
        msdn_p = dist_np[1]
        tmpn_p = dist_np[2]
        flag_hab = flags_np[1]
        flag_hab_temp = flags_np[2]

        dist_uninh = bool((msdn_p < 0) or (msdn_p > ARCSINH_1_OUTER))
        flag_uninh = bool(flag_hab < 0)
        both = dist_uninh and flag_uninh

        if step < 30 or step % 20 == 0:
            print(f"{step:4d} | {mpdn_p:8.4f} | {msdn_p:8.4f} | {tmpn_p:8.4f} | {flag_hab:8.4f} | {flag_hab_temp:13.4f} | {str(dist_uninh):10s} | {str(flag_uninh):10s} | {str(both)}")

        if step >= warmup and both:
            if not ever_uninhabited:
                print(f"\n*** FIRST uninhabited at step {step}: msdn={msdn_p:.4f}, flag_hab={flag_hab:.4f}")
            ever_uninhabited = True

        next_raw = np.array([[dist_np[0], dist_np[1], dist_np[2], dist_np[3], dist_np[4], dist_np[5], t_fracs[step]]], dtype=np.float32)
        next_norm = state_scaler.transform(next_raw)
        state_t = torch.from_numpy(next_norm).unsqueeze(1)

print(f"\never_uninhabited after 100 steps: {ever_uninhabited}")
