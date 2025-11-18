import numpy as np
from numba import njit

@njit(fastmath=True, cache=True)
def _accel(pos_a, pos_b, mu_b):
    r0 = pos_a[0] - pos_b[0]
    r1 = pos_a[1] - pos_b[1]
    r2 = pos_a[2] - pos_b[2]
    r2n = r0*r0 + r1*r1 + r2*r2
    r = r2n ** 0.5
    r3 = r * r2n
    return np.array([-mu_b * r0 / r3,
                     -mu_b * r1 / r3,
                     -mu_b * r2 / r3], dtype=np.float64)

@njit(fastmath=True, cache=True)
def _leapfrog_integrate(pos_mp, pos_ms, pos_mm,
                        vel_mp, vel_ms, vel_mm,
                        ms, mp, mm, t_end, dt):
    n_steps = max(1, int(np.ceil(t_end / dt)))
    xyz_mp = np.empty((n_steps, 3), dtype=np.float64)
    xyz_ms = np.empty((n_steps, 3), dtype=np.float64)
    xyz_mm = np.empty((n_steps, 3), dtype=np.float64)
    vel_mp_arr = np.empty((n_steps, 3), dtype=np.float64)
    vel_ms_arr = np.empty((n_steps, 3), dtype=np.float64)
    vel_mm_arr = np.empty((n_steps, 3), dtype=np.float64)

    p_mp = pos_mp.copy(); p_ms = pos_ms.copy(); p_mm = pos_mm.copy()
    v_mp = vel_mp.copy(); v_ms = vel_ms.copy(); v_mm = vel_mm.copy()
    half_dt = 0.5 * dt

    for i in range(n_steps):
        p2_mp = p_mp + v_mp * half_dt
        a_mp = _accel(p2_mp, p_ms, ms) + _accel(p2_mp, p_mm, mm)
        v_mp = v_mp + a_mp * dt
        p_mp = p2_mp + v_mp * half_dt

        p2_ms = p_ms + v_ms * half_dt
        a_ms = _accel(p2_ms, p_mp, mp) + _accel(p2_ms, p_mm, mm)
        v_ms = v_ms + a_ms * dt
        p_ms = p2_ms + v_ms * half_dt

        p2_mm = p_mm + v_mm * half_dt
        a_mm = _accel(p2_mm, p_mp, mp) + _accel(p2_mm, p_ms, ms)
        v_mm = v_mm + a_mm * dt
        p_mm = p2_mm + v_mm * half_dt

        xyz_mp[i] = p_mp; xyz_ms[i] = p_ms; xyz_mm[i] = p_mm
        vel_mp_arr[i] = v_mp; vel_ms_arr[i] = v_ms; vel_mm_arr[i] = v_mm

    return xyz_mp, xyz_ms, xyz_mm, vel_mp_arr, vel_ms_arr, vel_mm_arr

def leapfrog_integrate(state: dict, t_end: float, dt: float):
    pos_mp = state["pos_mp"].astype(np.float64)
    pos_ms = state["pos_ms"].astype(np.float64)
    pos_mm = state["pos_mm"].astype(np.float64)
    vel_mp = state["vel_mp"].astype(np.float64)
    vel_ms = state["vel_ms"].astype(np.float64)
    vel_mm = state["vel_mm"].astype(np.float64)
    ms = float(state["ms"]); mp = float(state["mp"]); mm = float(state["mm"])
    t_end = float(t_end); dt = float(dt)

    xyz_mp, xyz_ms, xyz_mm, vel_mp_arr, vel_ms_arr, vel_mm_arr = _leapfrog_integrate(
        pos_mp, pos_ms, pos_mm, vel_mp, vel_ms, vel_mm, ms, mp, mm, t_end, dt
    )

    return {
        "xyzarr_mp": xyz_mp,
        "xyzarr_ms": xyz_ms,
        "xyzarr_mm": xyz_mm,
        "velarr_mp": vel_mp_arr,
        "velarr_ms": vel_ms_arr,
        "velarr_mm": vel_mm_arr,
    }