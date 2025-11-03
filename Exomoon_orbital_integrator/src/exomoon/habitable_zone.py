import numpy as np
from .constants import stefboltz, F_earth, au

def hz_bounds_au(Ts_K: float, rs_m: float):
    L_star = 4 * np.pi * rs_m**2 * stefboltz * Ts_K**4
    F_inner = 1.1 * F_earth
    F_outer = 0.5 * F_earth
    a_inner_m = np.sqrt(L_star / (4 * np.pi * F_inner))
    a_outer_m = np.sqrt(L_star / (4 * np.pi * F_outer))
    return a_inner_m / au, a_outer_m / au