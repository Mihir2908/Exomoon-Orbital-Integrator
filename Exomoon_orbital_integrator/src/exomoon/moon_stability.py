import numpy as np
from exomoon.params import SystemParams
from exomoon.simulation import run_simulation_for_years

def assess_moon_stability(p: SystemParams, years: float, escape_factor: float = 1.0):
    """
    Stable if max ||r_moon - r_planet|| over [0, years] <= escape_factor * Hill radius.
    Adds escape_time (years) if an escape occurred.
    """
    sim = run_simulation_for_years(p, float(years))
    st = sim["state"]; traj = sim["traj"]

    moon_rel = traj["xyzarr_mm"] - traj["xyzarr_mp"]
    r_rel = np.linalg.norm(moon_rel[:, :2], axis=1)
    max_r = float(np.max(r_rel))
    rhill = float(st.get("rhill_AU")) if st.get("rhill_AU") is not None else None

    threshold = (escape_factor * rhill) if rhill is not None else float("inf")
    stable = (rhill is not None) and (max_r <= threshold)

    escape_time = None
    escape_index = None
    if not stable and rhill is not None:
        # first index where r_rel > threshold
        idxs = np.where(r_rel > threshold)[0]
        if idxs.size:
            j = int(idxs[0])
            # Time mapping: index j corresponds to (j+1)*dt (see integrator loop)
            dt = float(sim["dt"])
            t_prev = j * dt
            r_prev = r_rel[j-1] if j > 0 else r_rel[j]
            r_curr = r_rel[j]
            # Linear interpolation (only if growth across boundary)
            if j > 0 and r_curr > r_prev:
                frac = (threshold - r_prev) / (r_curr - r_prev)
                frac = max(0.0, min(1.0, frac))
                escape_time = t_prev + frac * dt
            else:
                escape_time = (j + 1) * dt
            escape_index = j

    return {
        "stable": bool(stable),
        "max_r_rel": max_r,
        "rhill_AU": rhill,
        "escape_factor": float(escape_factor),
        "t_end": float(sim["t_end"]),
        "escape_time": escape_time,
        "escape_index": escape_index,
        "threshold": threshold if rhill is not None else None,
    }

def analyze_moon_escape(p: SystemParams, years: float, escape_factor: float = 1.0):
    """
    Detailed escape analysis. Always returns:
      stable, threshold, escape_time, escape_index, max_r_rel, rhill_AU, dt, t_end
    """
    res = assess_moon_stability(p, years, escape_factor)
    # expose dt and times array info for clients
    sim = run_simulation_for_years(p, float(years))
    res["dt"] = float(sim["dt"])
    return res