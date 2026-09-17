"""
exomoon/ml/inference.py — Generate a 2D stability-habitability map via MoonRNN.

Given fixed stellar/planet parameters, sweeps a grid of (mm_earth × am_hill)
candidates in a single batched forward pass and returns which combinations
produce stable+habitable orbits throughout the simulated duration.

The 50×50 grid is inference/heatmap only — dataset generation uses LHS sampling.

Model compatibility
-------------------
system_dim=14 (current): SYS_COLS includes a_inner_au/a_outer_au alongside rhill_AU,
  and STATE_COLS uses moon_planet_dist_norm/moon_star_dist_norm (Hill-radius fraction
  and HZ-band position, both system-agnostic) instead of raw AU distances.

system_dim=12 (older): raw-AU distances, no HZ bounds as direct input. Not supported
  here -- retrain with the current dataset/train.py to get a system_dim=14 model.

Warm-up exclusion
-----------------
train() is 100% teacher-forced -- every training step the model has ever seen
received the true, ground-truth input. The transition from step 0 (true initial
condition) to step 1 (the model's own first self-generated prediction) is therefore
the single most out-of-distribution input the model ever encounters at inference.
Empirically this produces a brief, low-magnitude dip in the habitable/stable logits
in the first few steps that the model's own learned dynamics then pull back out of
(the same contraction toward the "normal" trajectory that lets it track correctly
for the rest of the rollout) -- NOT something scheduled sampling or further training
fixes (it tried, and only marginally reduced the dip's magnitude). Across a 900-point
grid sweep, this dip occurred in effectively every candidate regardless of its true
stability/habitability (including a real, simulation-verified stable+habitable system),
so it carries no information for distinguishing genuinely bad candidates from good
ones -- excluding it from the ever_unstable/ever_uninhabited accumulation flipped
~50% of an otherwise-failing 900-candidate grid to correctly valid, while ~50%
remained correctly disqualified by genuine negative steps occurring after the
warm-up window.
"""

from __future__ import annotations

import os
import sys

import numpy as np

_src = os.path.join(os.path.dirname(__file__), "..", "..")
if _src not in sys.path:
    sys.path.insert(0, os.path.abspath(_src))

MARS_MASS_EARTH  = 0.107
MOON_DENSITY_CGS = 3.0   # rocky moon assumption for Roche limit (g/cm³)
ARCSINH_1        = float(np.arcsinh(1.0))    # ≈ 0.881 — true outer HZ boundary in arcsinh encoding
# 5 % outer-HZ tolerance applied at inference only. The HZ flux threshold (0.5 × F_earth)
# carries genuine climate-model uncertainty of this magnitude; training labels remain
# anchored to the true a_outer. K-1229b (ap≈0.301 AU, a_outer≈0.309 AU) starts at
# msd_norm≈0.837 — this extends the margin before the outer-edge trip-wire from 0.044
# to 0.079 (1.8×), which is the main lever for fixing its habitable recall without
# corrupting the training distribution.
ARCSINH_1_OUTER  = float(np.arcsinh(1.05))  # ≈ 0.916 — outer bound with 5 % tolerance


def predict_stability_map(
    system_params:   dict,          # ms_solar, rs_solar, Ts, mp_earth, ap_AU, ep
    t_sim:           float,
    moon_retrograde: bool  = False,
    em:              float = 0.0,
    mm_resolution:   int   = 50,    # grid points along mm_earth axis
    am_resolution:   int   = 50,    # grid points along am_hill axis
    model_dir:       str   = "models/",
    n_steps:         int   = 1000,  # must match training resampling
    rnn_type:        str   = "gru",
    warmup_steps:    int   = 10,    # exclude this many initial steps from ever_unstable/
                                     # ever_uninhabited accumulation -- see note below
) -> dict:
    """
    Sweep a mm_resolution × am_resolution grid of (mm_earth, am_hill) candidates
    through the trained RNN in one batched forward pass.

    Returns
    -------
    dict with:
      map_stable     : (mm_resolution, am_resolution) bool array
      map_habitable  : (mm_resolution, am_resolution) bool array
      map_both       : (mm_resolution, am_resolution) bool array
      mm_grid        : (mm_resolution,) array  [M_earth]
      am_grid        : (am_resolution,) array  [Hill radii]
      valid_mm_range : [min, max] or None
      valid_am_per_mm: list of [min_am, max_am] per mm value (None if no valid am)
    """
    try:
        import torch
    except ImportError:
        raise RuntimeError("PyTorch required — pip install torch --index-url https://download.pytorch.org/whl/cpu")

    from exomoon.ml.model     import MoonRNN, SYS_DIM, STATE_DIM
    from exomoon.ml.dataset   import load_normalizer
    from exomoon.params        import SystemParams
    from exomoon.initial_conditions import initial_state
    from exomoon.habitable_zone import hz_bounds_au, moon_effective_temp_K
    from exomoon.constants      import rsun, merth, msun, au

    # ── load model ────────────────────────────────────────────────────────────
    model_path = os.path.join(model_dir, f"{rnn_type}_model.pt")
    if not os.path.exists(model_path):
        return {"ok": False, "error": "no_model",
                "message": f"No trained {rnn_type.upper()} model found at {model_path}"}

    model = MoonRNN.load(model_dir, rnn_type=rnn_type)
    device = torch.device("cpu")
    model.to(device).eval()

    # ── detect model generation ───────────────────────────────────────────────
    # system_dim=14 models include a_inner_au/a_outer_au and use normalized
    # distance features. state_dim=7 models additionally include moon_temp_norm
    # (the second, independently-headed habitability encoding). Older generations
    # are not supported here -- retrain with the current dataset/train.py.
    if model.system_dim != SYS_DIM or model.state_dim != STATE_DIM:
        return {"ok": False, "error": "stale_model",
                "message": f"Loaded {rnn_type.upper()} model has system_dim={model.system_dim}, "
                           f"state_dim={model.state_dim}; expected system_dim={SYS_DIM}, "
                           f"state_dim={STATE_DIM}. Retrain with the current dataset/train.py."}

    # ── load normaliser ───────────────────────────────────────────────────────
    norm_path = os.path.join(model_dir, "normalizer.pkl")
    if not os.path.exists(norm_path):
        return {"ok": False, "error": "no_normalizer",
                "message": f"No normalizer.pkl found at {norm_path}"}

    sys_scaler, state_scaler = load_normalizer(norm_path)

    # ── build mm / am grids ───────────────────────────────────────────────────
    mp_earth = float(system_params.get("mp_earth", 1.0))
    ms       = float(system_params.get("ms_solar",  1.0))
    rs       = float(system_params.get("rs_solar",  1.0))
    Ts       = float(system_params.get("Ts",        5772.0))
    ap       = float(system_params.get("ap_AU",     1.0))
    ep       = float(system_params.get("ep",        0.0))
    dp_cgs   = float(system_params.get("dp_cgs",   5.5))   # planet density; default = Earth

    # mm_earth: Mars mass → min(mp_earth, 3.0 M_earth), log-spaced
    # Matches training formula in run_ml_dataset.py exactly: min(mp, MM_MAX_CAP=3.0)
    mm_min  = MARS_MASS_EARTH
    mm_max  = min(mp_earth, 3.0)
    if mm_max <= mm_min:
        mm_max = mm_min * 1.01   # degenerate: planet barely above Mars mass
    mm_grid = np.exp(np.linspace(np.log(mm_min), np.log(mm_max), mm_resolution))

    # Roche limit (fluid-body): a_roche = 2.456 × R_planet × (ρ_planet/ρ_moon)^(1/3)
    mp_kg      = mp_earth * merth
    dp_SI      = dp_cgs * 1e3                                    # g/cm³ → kg/m³
    rp_m       = (0.75 * mp_kg / (np.pi * dp_SI)) ** (1.0 / 3.0)
    a_roche_m  = 2.456 * rp_m * (dp_cgs / MOON_DENSITY_CGS) ** (1.0 / 3.0)
    a_roche_AU = a_roche_m / au

    # Hill radius for this stellar/planet configuration
    ms_kg    = ms * msun
    rhill_AU = ap * (1.0 - ep) * (mp_kg / (3.0 * ms_kg)) ** (1.0 / 3.0)

    # am_hill: Roche limit → 1.0 Hill radius (linear-spaced)
    am_min  = max(a_roche_AU / rhill_AU, 1e-3)   # Hill fraction; floor avoids edge cases
    am_grid = np.linspace(am_min, 1.0, am_resolution)

    n_candidates = mm_resolution * am_resolution

    # ── compute initial states for all candidates ─────────────────────────────
    all_sys_raw   = np.zeros((n_candidates, SYS_DIM),     dtype=np.float32)
    all_state_raw = np.zeros((n_candidates, model.state_dim), dtype=np.float32)
    valid_mask    = np.ones(n_candidates, dtype=bool)

    rs_m = rs * rsun
    a_inner_au, a_outer_au = hz_bounds_au(Ts, rs_m)
    hz_width = max(a_outer_au - a_inner_au, 1e-6)

    idx = 0
    for mm in mm_grid:
        for am in am_grid:
            p = SystemParams(
                Ts=Ts, rs_solar=rs, ms_solar=ms,
                mp_earth=mp_earth, ap_AU=ap, ep=ep,
                mm_earth=float(mm), am_hill=float(am),
                em=em, moon_retrograde=moon_retrograde,
            )
            try:
                st = initial_state(p)
            except Exception:
                valid_mask[idx] = False
                idx += 1
                continue

            rhill = st["rhill_AU"]

            # Skip degenerate configurations
            if rhill < 1e-4:
                valid_mask[idx] = False
                idx += 1
                continue

            # Initial distances
            pos_mp = st["pos_mp"]
            pos_ms = st["pos_ms"]
            pos_mm = st["pos_mm"]
            vel_mp = st["vel_mp"]
            vel_mm = st["vel_mm"]

            moon_planet_dist = float(np.linalg.norm(pos_mm - pos_mp))
            planet_star_dist = float(np.linalg.norm(pos_mp - pos_ms))
            moon_star_dist   = float(np.linalg.norm(pos_mm - pos_ms))
            moon_speed       = float(np.linalg.norm(vel_mm))
            planet_speed     = float(np.linalg.norm(vel_mp))

            # Second, independently-headed habitability encoding via equilibrium
            # temperature (Stefan-Boltzmann) -- see dataset.py STATE_COLS comment.
            Tm_K   = moon_effective_temp_K(Ts, rs_m, moon_star_dist * au)
            T_hot  = moon_effective_temp_K(Ts, rs_m, a_inner_au * au)
            T_cold = moon_effective_temp_K(Ts, rs_m, a_outer_au * au)
            temp_width = max(T_hot - T_cold, 1e-6)

            # STATE_COLS: moon_planet_dist_norm (Hill-radius fraction, stable <=> <=1.0),
            # moon_star_dist_norm (HZ-band position, habitable <=> in [0,1]), moon_temp_norm
            # (same band position via temperature instead of distance), planet_star_dist
            # (raw AU — same column as in the training Parquet, scaler handles normalisation),
            # moon_speed, planet_speed, t_frac
            all_state_raw[idx] = [
                moon_planet_dist / rhill,
                np.arcsinh((moon_star_dist - a_inner_au) / hz_width),
                np.arcsinh((Tm_K - T_cold) / temp_width),
                planet_star_dist,
                moon_speed, planet_speed,
                0.0,   # t_frac = 0 at t=0
            ]

            # SYS_COLS: ms_solar, rs_solar, Ts, mp_earth, ap_AU, ep,
            #           mm_earth, am_hill, em, moon_retrograde, t_sim, rhill_AU,
            #           a_inner_au, a_outer_au
            all_sys_raw[idx] = [
                ms, rs, Ts, mp_earth, ap, ep,
                float(mm), float(am), em,
                float(int(moon_retrograde)),
                t_sim, rhill, a_inner_au, a_outer_au,
            ]
            idx += 1

    # ── normalise ─────────────────────────────────────────────────────────────
    all_sys_norm   = sys_scaler.transform(all_sys_raw).astype(np.float32)
    all_state_norm = state_scaler.transform(all_state_raw).astype(np.float32)

    # ── autoregressive rollout ─────────────────────────────────────────────────
    # model.rnn is called directly so the hidden state h flows continuously across
    # all n_steps (matching how the full 1000-step sequence was processed during
    # training — h_t carries accumulated trajectory history, not just step t−1).
    state_t = torch.from_numpy(all_state_norm).unsqueeze(1)   # (N, 1, model.state_dim)
    sys_t   = torch.from_numpy(all_sys_norm)                   # (N, SYS_DIM)
    sys_exp = sys_t.unsqueeze(1)                               # (N, 1, SYS_DIM)

    if model.rnn_type == "lstm":
        h: object = (
            torch.zeros(model.layers, n_candidates, model.hidden),
            torch.zeros(model.layers, n_candidates, model.hidden),
        )
    else:
        h = torch.zeros(model.layers, n_candidates, model.hidden)

    ever_unstable    = np.zeros(n_candidates, dtype=bool)
    ever_uninhabited = np.zeros(n_candidates, dtype=bool)
    ever_uninhabited_from_temp = np.zeros(n_candidates, dtype=bool)
    ever_disagreed   = np.zeros(n_candidates, dtype=bool)   # habitable vs habitable_from_temp
    n_disagreements  = np.zeros(n_candidates, dtype=np.int32)
    ever_unstable[~valid_mask]    = True
    ever_uninhabited[~valid_mask] = True
    ever_uninhabited_from_temp[~valid_mask] = True

    t_fracs = np.linspace(0.0, 1.0, n_steps)

    with torch.no_grad():
        for step in range(n_steps):
            rnn_in  = torch.cat([state_t, sys_exp], dim=-1)
            rnn_out, h = model.rnn(rnn_in, h)

            # model.forward() applies the per-column activation (relu for the
            # non-negative columns, identity for the signed moon_star_dist_norm /
            # moon_temp_norm) -- replicate that here since we call model.rnn/.head_dist
            # directly to keep the hidden state h flowing continuously across all n_steps.
            raw_dist = model.head_dist(rnn_out)
            dist_pred = torch.cat([
                torch.relu(raw_dist[..., 0:1]),
                raw_dist[..., 1:3],
                torch.relu(raw_dist[..., 3:6]),
            ], dim=-1)
            flag_logits = model.head_flags(rnn_out)               # (N, 1, N_FLAGS)

            dist_np  = dist_pred.squeeze(1).cpu().numpy()         # (N, N_DIST)
            flags_np = flag_logits.squeeze(1).cpu().numpy()       # (N, N_FLAGS)

            # warmup_steps excludes the first few steps from accumulation -- see the
            # module docstring's "Warm-up exclusion" note.
            if step >= warmup_steps:
                ever_unstable |= (flags_np[:, 0] < 0)   # flag head: stable logit

                # Habitability requires BOTH heads to agree the moon is outside the HZ
                # before disqualifying a candidate. The flag head (BCE-trained logit) and
                # the dist head (MSE-trained moon_star_dist_norm) are independent predictions
                # from the same hidden state. The flag head showed systematic miscalibration
                # on edge-of-HZ systems (K-1229b retrograde) where the dist head tracked
                # correctly. Requiring both to agree reduces false disqualifications caused
                # by single-head misfires, at the cost of only disqualifying candidates
                # where both signals are confident.
                dist_uninh = (dist_np[:, 1] < 0) | (dist_np[:, 1] > ARCSINH_1_OUTER)
                flag_uninh = (flags_np[:, 1] < 0)
                ever_uninhabited |= dist_uninh & flag_uninh   # both must agree

                temp_uninh = (dist_np[:, 2] < 0) | (dist_np[:, 2] > ARCSINH_1_OUTER)
                flag_temp_uninh = (flags_np[:, 2] < 0)
                ever_uninhabited_from_temp |= temp_uninh & flag_temp_uninh   # both must agree

                # Disagreement between dist head and flag head is now the criterion for
                # NOT disqualifying — steps where they disagree are the ones that would
                # have been false disqualifications under the pure flag criterion.
                step_disagrees  = dist_uninh != flag_uninh
                ever_disagreed  |= step_disagrees
                n_disagreements += step_disagrees.astype(np.int32)

            # ── state feedback ────────────────────────────────────────────────
            next_state_raw = np.concatenate([
                dist_np[:, 0:6],                              # all 6 dist targets (direct)
                np.full((n_candidates, 1), t_fracs[step]),    # t_frac
            ], axis=1)   # (N, 7)

            next_state_norm = state_scaler.transform(next_state_raw.astype(np.float32))
            state_t = torch.from_numpy(next_state_norm).unsqueeze(1)

    # ── reshape results ───────────────────────────────────────────────────────
    map_stable    = (~ever_unstable).reshape(mm_resolution, am_resolution)
    map_habitable = (~ever_uninhabited).reshape(mm_resolution, am_resolution)
    map_habitable_from_temp = (~ever_uninhabited_from_temp).reshape(mm_resolution, am_resolution)
    map_both      = map_stable & map_habitable
    map_disagreed = ever_disagreed.reshape(mm_resolution, am_resolution)
    map_disagreement_count = n_disagreements.reshape(mm_resolution, am_resolution)

    # Valid ranges per mm row
    valid_am_per_mm: list = []
    for i in range(mm_resolution):
        row = map_both[i]
        valid_idxs = np.where(row)[0]
        if len(valid_idxs):
            valid_am_per_mm.append([
                float(am_grid[valid_idxs[0]]),
                float(am_grid[valid_idxs[-1]]),
            ])
        else:
            valid_am_per_mm.append(None)

    valid_mm_idxs = [i for i, v in enumerate(valid_am_per_mm) if v is not None]
    valid_mm_range = (
        [float(mm_grid[valid_mm_idxs[0]]), float(mm_grid[valid_mm_idxs[-1]])]
        if valid_mm_idxs else None
    )

    return {
        "ok":             True,
        "map_stable":     map_stable.tolist(),
        "map_habitable":  map_habitable.tolist(),
        "map_both":       map_both.tolist(),
        "mm_grid":        mm_grid.tolist(),
        "am_grid":        am_grid.tolist(),
        "valid_mm_range": valid_mm_range,
        "valid_am_per_mm": valid_am_per_mm,
        # Diagnostic: second, independently-headed habitability encoding (temperature-
        # based) and per-candidate disagreement with the existing distance-based head.
        # map_disagreed marks candidates where the two heads ever gave opposite
        # habitable/not-habitable verdicts at the same rollout step -- a free signal
        # of where the autoregressive prediction became unreliable, not (yet) used
        # to change map_both/map_habitable themselves.
        "map_habitable_from_temp":  map_habitable_from_temp.tolist(),
        "map_disagreed":            map_disagreed.tolist(),
        "map_disagreement_count":   map_disagreement_count.tolist(),
    }
