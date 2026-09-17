"""
One-off diagnostic: pick a few "clean" false-negative grid cells (gt_stable=True,
gt_habitable=True for the whole duration, ML predicts not-habitable) from the
Kepler-1229 b prograde grid, and print the full per-step habitable_logit and
moon_star_dist_norm trajectory to see the shape of the failure: single early
catastrophic dip, periodic dips tied to moon orbital phase, or one-way drift.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from exomoon.params import SystemParams
from exomoon.initial_conditions import initial_state
from exomoon.habitable_zone import hz_bounds_au, moon_effective_temp_K
from exomoon.constants import rsun, au
from exomoon.ml.model import MoonRNN
from exomoon.ml.dataset import load_normalizer, SYS_COLS, LOG_SYS_COLS

MODEL_DIR = "models_temphead_ss_prefix"
WARMUP_STEPS = 10
N_STEPS = 1000
MOON_RETROGRADE = False

base = "ground_truth_grids/Kepler_1229_b_prograde"
npz = np.load(f"{base}.npz")
with open(f"{base}.meta.json") as f:
    meta = json.load(f)
sys_params = meta["system_params"]
mm_grid = npz["mm_grid"]; am_grid = npz["am_grid"]
gt_stable = npz["map_stable"]; gt_habitable = npz["map_habitable"]

clean_fn = gt_stable & gt_habitable   # candidates: also need ml says not-habitable, checked below

model = MoonRNN.load(MODEL_DIR, rnn_type="gru"); model.eval()
sys_scaler, state_scaler = load_normalizer(os.path.join(MODEL_DIR, "normalizer.pkl"))
a_in, a_out = hz_bounds_au(sys_params["Ts"], sys_params["rs_solar"] * rsun)
hz_width = max(a_out - a_in, 1e-6)

# pick a few candidate (i,j) cells where ground truth is cleanly stable+habitable,
# spread across the grid (not just the first few in (i,j) order -- that corner is
# biased toward minimum mm_earth/am_hill and unrepresentative).
candidates = list(zip(*np.where(clean_fn)))
print(f"Total clean stable+habitable cells: {len(candidates)}")
rng = np.random.default_rng(0)  # same seed -> same picked cells
rng.shuffle(candidates)

picked = 0
for (i, j) in candidates:
    if picked >= 6:
        break
    mm = float(mm_grid[i]); am = float(am_grid[j])
    p = SystemParams(Ts=sys_params["Ts"], rs_solar=sys_params["rs_solar"],
                      ms_solar=sys_params["ms_solar"], mp_earth=sys_params["mp_earth"],
                      ap_AU=sys_params["ap_AU"], ep=sys_params.get("ep", 0.0),
                      mm_earth=mm, am_hill=am, em=0.0, moon_retrograde=MOON_RETROGRADE)
    st = initial_state(p)
    rhill = st["rhill_AU"]
    pos_mp, pos_ms, pos_mm = st["pos_mp"], st["pos_ms"], st["pos_mm"]
    vel_mp, vel_mm = st["vel_mp"], st["vel_mm"]
    mpd = float(np.linalg.norm(pos_mm - pos_mp))
    psd = float(np.linalg.norm(pos_mp - pos_ms))
    msd = float(np.linalg.norm(pos_mm - pos_ms))
    msp = float(np.linalg.norm(vel_mm)); psp = float(np.linalg.norm(vel_mp))
    Tm = moon_effective_temp_K(sys_params["Ts"], sys_params["rs_solar"] * rsun, msd * au)
    Th = moon_effective_temp_K(sys_params["Ts"], sys_params["rs_solar"] * rsun, a_in * au)
    Tc = moon_effective_temp_K(sys_params["Ts"], sys_params["rs_solar"] * rsun, a_out * au)
    tw = max(Th - Tc, 1e-6)

    state_raw = np.array([[mpd / rhill, np.arcsinh((msd - a_in) / hz_width),
                            np.arcsinh((Tm - Tc) / tw), psd / sys_params["ap_AU"], msp, psp, 0.0]], dtype=np.float32)
    sys_raw = np.array([[sys_params["ms_solar"], sys_params["rs_solar"], sys_params["Ts"],
                          sys_params["mp_earth"], sys_params["ap_AU"], sys_params.get("ep", 0.0),
                          mm, am, 0.0, float(MOON_RETROGRADE), meta["t_sim"], rhill, a_in, a_out]], dtype=np.float32)
    state_t = torch.from_numpy(state_scaler.transform(state_raw)).unsqueeze(1)
    for _col in LOG_SYS_COLS:
        _j = SYS_COLS.index(_col)
        sys_raw[:, _j] = np.log(np.maximum(sys_raw[:, _j], 1e-12))
    sys_t = torch.from_numpy(sys_scaler.transform(sys_raw)).unsqueeze(1)
    h = torch.zeros(model.layers, 1, model.hidden)
    t_fracs = np.linspace(0, 1, N_STEPS)

    sn_hist, hl_hist, mp_hist, sl_hist = [], [], [], []
    with torch.no_grad():
        for step in range(N_STEPS):
            rnn_in = torch.cat([state_t, sys_t], dim=-1)
            rnn_out, h = model.rnn(rnn_in, h)
            raw = model.head_dist(rnn_out)
            dist_pred = torch.cat([torch.relu(raw[..., 0:1]), raw[..., 1:3],
                                    torch.relu(raw[..., 3:6])], dim=-1)
            flags = model.head_flags(rnn_out)
            dn = dist_pred.squeeze(1).numpy(); fn = flags.squeeze(1).numpy()
            sn_hist.append(dn[0, 1]); hl_hist.append(fn[0, 1])
            mp_hist.append(dn[0, 0]); sl_hist.append(fn[0, 0])
            nxt = np.concatenate([dn, np.full((1, 1), t_fracs[step])], axis=1).astype(np.float32)
            state_t = torch.from_numpy(state_scaler.transform(nxt)).unsqueeze(1)

    sn_hist = np.array(sn_hist); hl_hist = np.array(hl_hist)
    mp_hist = np.array(mp_hist); sl_hist = np.array(sl_hist)
    ever_uninhab = bool(np.any(hl_hist[WARMUP_STEPS:] < 0))
    if not ever_uninhab:
        continue   # this cell isn't actually a false negative, skip

    picked += 1
    print(f"\n=== mm_earth={mm:.3f} am_hill={am:.3f} (rhill={rhill:.5f}) ===")
    print(f"min habitable_logit={hl_hist.min():.3f} at step {hl_hist.argmin()}  "
          f"min stable_logit={sl_hist.min():.3f} at step {sl_hist.argmin()}")
    neg_steps = np.where(hl_hist[WARMUP_STEPS:] < 0)[0] + WARMUP_STEPS
    stable_neg_steps = np.where(sl_hist[WARMUP_STEPS:] < 0)[0] + WARMUP_STEPS
    print(f"steps with negative habitable_logit: {len(neg_steps)}/{N_STEPS - WARMUP_STEPS}   "
          f"steps with negative stable_logit: {len(stable_neg_steps)}/{N_STEPS - WARMUP_STEPS}")
    # moon_planet_dist_norm (dn[...,0]) is the raw mpd/rhill ratio -- "safe" stable range is [0,1].
    # moon_star_dist_norm (dn[...,1]) is arcsinh((msd-a_in)/hz_width) -- "safe" habitable range is [0,0.881].
    mp_excursion = np.maximum(mp_hist[WARMUP_STEPS:] - 1.0, 0.0)  # how far past the stable threshold, in ratio units
    sn_excursion = np.maximum(np.abs(sn_hist[WARMUP_STEPS:] - 0.4405) - 0.4405, 0.0)  # distance outside [0,0.881] band
    print(f"  moon_planet_dist_norm (mpd/rhill, safe<=1.0): max={mp_hist[WARMUP_STEPS:].max():.3f}  "
          f"max excursion past threshold={mp_excursion.max():.3f} ratio-units")
    print(f"  moon_star_dist_norm (arcsinh, safe in [0,0.881]): min={sn_hist[WARMUP_STEPS:].min():.3f} max={sn_hist[WARMUP_STEPS:].max():.3f}  "
          f"max excursion outside band={sn_excursion.max():.3f} arcsinh-units (real-space ratio sinh={np.sinh(sn_excursion.max()):.2f}x hz_width)")
    print("step | moon_planet_dist_norm | stable_logit | moon_star_dist_norm | habitable_logit")
    for step in range(0, N_STEPS, 25):
        flag = " <-- HAB-NEG" if hl_hist[step] < 0 else ""
        flag2 = " <-- STAB-NEG" if sl_hist[step] < 0 else ""
        print(f"{step:4d} | {mp_hist[step]:+.4f}{flag2:13s} | {sl_hist[step]:+.3f} | {sn_hist[step]:+.4f} | {hl_hist[step]:+.3f}{flag}")
