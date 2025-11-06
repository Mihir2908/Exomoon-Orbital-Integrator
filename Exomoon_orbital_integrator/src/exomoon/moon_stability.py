import numpy as np
from exomoon.params import SystemParams
from exomoon.simulation import run_simulation_for_years

def assess_moon_stability(p: SystemParams, years: float, escape_factor: float = 1.0):
    """
    Stable if max ||r_moon - r_planet|| over [0, years] <= escape_factor * Hill radius.
    Returns: dict with stable flag and metrics.
    """
    sim = run_simulation_for_years(p, float(years))
    st = sim["state"]; traj = sim["traj"]

    moon_rel = traj["xyzarr_mm"] - traj["xyzarr_mp"]
    r_rel = np.linalg.norm(moon_rel[:, :2], axis=1)
    max_r = float(np.max(r_rel))
    rhill = float(st.get("rhill_AU")) if st.get("rhill_AU") is not None else None

    threshold = (escape_factor * rhill) if rhill is not None else float("inf")
    stable = (rhill is not None) and (max_r <= threshold)

    return {
        "stable": bool(stable),
        "max_r_rel": max_r,
        "rhill_AU": rhill,
        "escape_factor": float(escape_factor),
        "t_end": float(sim["t_end"]),
    }