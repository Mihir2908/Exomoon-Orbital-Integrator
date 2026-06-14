"""
exomoon/ml/inference.py — Generate a 2D stability-habitability map via MoonRNN.

Given fixed stellar/planet parameters, sweeps a grid of (mm_earth × am_hill)
candidates in a single batched forward pass and returns which combinations
produce stable+habitable orbits throughout the simulated duration.

The 50×50 grid is inference/heatmap only — dataset generation uses LHS sampling.
"""

from __future__ import annotations

import os
import sys
import pickle

import numpy as np

_src = os.path.join(os.path.dirname(__file__), "..", "..")
if _src not in sys.path:
    sys.path.insert(0, os.path.abspath(_src))

MARS_MASS_EARTH  = 0.107
MOON_DENSITY_CGS = 3.0   # rocky moon assumption for Roche limit (g/cm³)


def predict_stability_map(
    system_params:   dict,          # ms_solar, rs_solar, Ts, mp_earth, ap_AU, ep
    t_sim:           float,
    moon_retrograde: bool  = False,
    em:              float = 0.0,
    mm_resolution:   int   = 50,    # grid points along mm_earth axis
    am_resolution:   int   = 50,    # grid points along am_hill axis
    model_dir:       str   = "models/",
    n_steps:         int   = 1000,  # must match training resampling
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

    from exomoon.ml.model     import MoonRNN, SYS_DIM, STATE_DIM, N_DIST, N_FLAGS
    from exomoon.ml.dataset   import SYS_COLS, STATE_COLS, load_normalizer
    from exomoon.params        import SystemParams
    from exomoon.initial_conditions import initial_state
    from exomoon.habitable_zone import hz_bounds_au
    from exomoon.constants      import rsun, merth, msun, au

    # ── load model ────────────────────────────────────────────────────────────
    model_path = os.path.join(model_dir, "gru_model.pt")
    if not os.path.exists(model_path):
        return {"ok": False, "error": "no_model",
                "message": f"No trained model found at {model_path}"}

    model = MoonRNN.load(model_dir)
    device = torch.device("cpu")
    model.to(device).eval()

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

    # mm_earth: Mars mass → planet mass (log-spaced)
    mm_min  = MARS_MASS_EARTH
    mm_max  = max(mp_earth, mm_min * 1.5)   # floor ensures mm_max > mm_min
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

    all_sys_raw   = np.zeros((n_candidates, SYS_DIM),   dtype=np.float32)
    all_state_raw = np.zeros((n_candidates, STATE_DIM), dtype=np.float32)
    rhill_vals    = np.zeros(n_candidates, dtype=np.float32)
    a_inner_vals  = np.zeros(n_candidates, dtype=np.float32)
    a_outer_vals  = np.zeros(n_candidates, dtype=np.float32)
    valid_mask    = np.ones(n_candidates, dtype=bool)

    rs_m = rs * rsun
    a_inner_au, a_outer_au = hz_bounds_au(Ts, rs_m)

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
            am_AU = st["am_AU"]

            # Skip degenerate configurations
            if rhill < 1e-4 or am_AU > 0.95 * rhill:
                valid_mask[idx] = False
                idx += 1
                continue

            rhill_vals[idx]   = rhill
            a_inner_vals[idx] = a_inner_au
            a_outer_vals[idx] = a_outer_au

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

            stable_0    = int(moon_planet_dist <= rhill)
            habitable_0 = int(a_inner_au <= moon_star_dist <= a_outer_au)

            # STATE_COLS: moon_planet_dist, moon_star_dist, planet_star_dist,
            #             moon_speed, planet_speed, stable, habitable, t_frac
            all_state_raw[idx] = [
                moon_planet_dist, moon_star_dist, planet_star_dist,
                moon_speed, planet_speed,
                stable_0, habitable_0,
                0.0,   # t_frac = 0 at t=0
            ]

            # SYS_COLS: ms_solar, rs_solar, Ts, mp_earth, ap_AU, ep,
            #           mm_earth, am_hill, em, moon_retrograde, t_sim, rhill_AU
            all_sys_raw[idx] = [
                ms, rs, Ts, mp_earth, ap, ep,
                float(mm), float(am), em,
                float(int(moon_retrograde)),
                t_sim, rhill,
            ]
            idx += 1

    # ── normalise ─────────────────────────────────────────────────────────────
    all_sys_norm   = sys_scaler.transform(all_sys_raw).astype(np.float32)
    all_state_norm = state_scaler.transform(all_state_raw).astype(np.float32)

    # ── autoregressive rollout ─────────────────────────────────────────────────
    # We roll out n_steps steps for all candidates simultaneously.
    # At each step we update t_frac in the state and feed the previous output back.
    # Shape: (n_candidates, 1, STATE_DIM)
    state_t = torch.from_numpy(all_state_norm).unsqueeze(1)   # (N, 1, STATE_DIM)
    sys_t   = torch.from_numpy(all_sys_norm)                   # (N, SYS_DIM)

    # Track whether each candidate has ever violated stability/habitability
    ever_unstable    = np.zeros(n_candidates, dtype=bool)
    ever_uninhabited = np.zeros(n_candidates, dtype=bool)
    ever_unstable[~valid_mask]    = True
    ever_uninhabited[~valid_mask] = True

    t_fracs = np.linspace(0.0, 1.0, n_steps)

    with torch.no_grad():
        for step in range(n_steps):
            dist_pred, flag_logits = model(state_t, sys_t)
            # dist_pred: (N, 1, N_DIST)   flag_logits: (N, 1, N_FLAGS)

            # Threshold flags
            flags = (flag_logits.squeeze(1) > 0).cpu().numpy()   # (N, 2)
            ever_unstable    |= ~flags[:, 0]
            ever_uninhabited |= ~flags[:, 1]

            # Build next state: [dist0..4, stable, habitable, t_frac]
            dist_np  = dist_pred.squeeze(1).cpu().numpy()         # (N, N_DIST)
            next_state_raw = np.concatenate(
                [dist_np, flags.astype(np.float32), np.full((n_candidates, 1), t_fracs[step])],
                axis=1,
            )   # (N, STATE_DIM)
            # Re-normalise for next step
            next_state_norm = state_scaler.transform(next_state_raw.astype(np.float32))
            state_t = torch.from_numpy(next_state_norm).unsqueeze(1)

    # ── reshape results ───────────────────────────────────────────────────────
    map_stable    = (~ever_unstable).reshape(mm_resolution, am_resolution)
    map_habitable = (~ever_uninhabited).reshape(mm_resolution, am_resolution)
    map_both      = map_stable & map_habitable

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
    }
