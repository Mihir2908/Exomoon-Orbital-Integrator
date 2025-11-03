import numpy as np

def _accel(pos_a: np.ndarray, pos_b: np.ndarray, mu_b: float) -> np.ndarray:
    r = pos_a - pos_b
    r3 = np.linalg.norm(r) ** 3
    return -mu_b * r / r3

def leapfrog_integrate(state: dict, t_end: float, dt: float):
    pos_mp = state["pos_mp"].copy()
    pos_ms = state["pos_ms"].copy()
    pos_mm = state["pos_mm"].copy()
    vel_mp = state["vel_mp"].copy()
    vel_ms = state["vel_ms"].copy()
    vel_mm = state["vel_mm"].copy()
    ms = state["ms"]; mp = state["mp"]; mm = state["mm"]

    xyzlist_mp, xyzlist_ms, xyzlist_mm = [], [], []

    t = 0.0
    while t < t_end:
        t += dt

        # Planet
        pos2_mp = pos_mp + vel_mp * (dt / 2)
        a_mp = _accel(pos2_mp, pos_ms, ms) + _accel(pos2_mp, pos_mm, mm)
        vel_mp = vel_mp + a_mp * dt
        pos_mp = pos2_mp + vel_mp * (dt / 2)

        # Star
        pos2_ms = pos_ms + vel_ms * (dt / 2)
        a_ms = _accel(pos2_ms, pos_mp, mp) + _accel(pos2_ms, pos_mm, mm)
        vel_ms = vel_ms + a_ms * dt
        pos_ms = pos2_ms + vel_ms * (dt / 2)

        # Moon
        pos2_mm = pos_mm + vel_mm * (dt / 2)
        a_mm = _accel(pos2_mm, pos_mp, mp) + _accel(pos2_mm, pos_ms, ms)
        vel_mm = vel_mm + a_mm * dt
        pos_mm = pos2_mm + vel_mm * (dt / 2)

        xyzlist_mp.append(pos_mp.copy())
        xyzlist_ms.append(pos_ms.copy())
        xyzlist_mm.append(pos_mm.copy())

    return dict(
        xyzarr_mp=np.array(xyzlist_mp),
        xyzarr_ms=np.array(xyzlist_ms),
        xyzarr_mm=np.array(xyzlist_mm),
    )