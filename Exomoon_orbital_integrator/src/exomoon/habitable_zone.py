import numpy as np
from .constants import stefboltz, F_earth, au

def hz_bounds_au(Ts_K: float, rs_m: float):
    L_star = 4 * np.pi * rs_m**2 * stefboltz * Ts_K**4
    F_inner = 1.1 * F_earth
    F_outer = 0.5 * F_earth
    a_inner_m = np.sqrt(L_star / (4 * np.pi * F_inner))
    a_outer_m = np.sqrt(L_star / (4 * np.pi * F_outer))
    return a_inner_m / au, a_outer_m / au


def moon_effective_temp_K(Ts_K, rs_m, moon_star_dist_m):
    """
    Moon equilibrium blackbody temperature at distance moon_star_dist_m from a star
    of temperature Ts_K and radius rs_m. Same albedo=0.3 / effective-emissivity=2.448
    greybody assumption used elsewhere in the app (agent_service.py's moon_teff_K,
    trajectoryMath.ts's moonEffectiveTempK) -- kept here so the ML pipeline shares
    one implementation instead of a third inline copy of the formula.
    """
    L_star = 4 * np.pi * rs_m**2 * stefboltz * Ts_K**4
    F = L_star / (4 * np.pi * moon_star_dist_m**2)
    return ((0.7 * F) / (2.448 * stefboltz)) ** 0.25