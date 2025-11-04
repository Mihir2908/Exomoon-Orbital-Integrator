import numpy as np
from exomoon.constants import rsun, msun, merth, FOUR_PI2
from exomoon.params import SystemParams

def initial_state(p: SystemParams):
    # Planet radius (km) if needed
    rp_km = (0.75 * p.mp_earth * merth / (p.dp_cgs * 1e3)) ** (1.0 / 3.0) / 1e3
    rs_m = p.rs_solar * rsun

    # Dimensionless gravitational parameters
    ms = p.ms_solar * FOUR_PI2
    mp = p.mp_earth * (merth / msun) * FOUR_PI2
    mm = p.mm_earth * (merth / msun) * FOUR_PI2

    # Hill radius and moon-planet distance (AU)
    rhill = p.ap_AU * (1.0 - p.ep) * ((mp / (3 * ms)) ** (1.0 / 3.0))
    am_AU = p.am_hill * rhill

    

    # Planet-moon barycenter about system barycenter
    xpm = p.ap_AU * (1 - p.ep) * ms / (ms + mp + mm)
    ypm = zpm = 0.0
    vxpm = vzpm = 0.0
    vypm = ms / ((1 - p.ep) * (mp + mm + ms) * p.ap_AU) ** 0.5

    # Star
    xs = -p.ap_AU * (1 - p.ep) * (mp + mm) / (ms + mp + mm)
    ys = zs = 0.0
    vxs = vzs = 0.0
    vys = -(mp + mm) / ((1 - p.ep) * (mp + mm + ms) * p.ap_AU) ** 0.5

    # Planet relative to PM barycenter, then to system barycenter
    yp = zp = 0.0
    xp = -am_AU * (1 - p.em) * mm / (mp + mm)
    xp = xpm + xp
    vxp = vzp = 0.0
    vyp = -mm / ((1 - p.em) * (mp + mm) * am_AU) ** 0.5
    vyp = vypm + vyp

    # Moon relative to PM barycenter, then to system barycenter
    ym = zm = 0.0
    xm = am_AU * (1 - p.em) * mp / (mp + mm)
    xm = xpm + xm
    vxm = vzm = 0.0
    vym = mp / ((1 - p.em) * (mp + mm) * am_AU) ** 0.5
    vym = vypm + vym

    pos_mp = np.array([xp, yp, zp], dtype=float)
    pos_ms = np.array([xs, ys, zs], dtype=float)
    pos_mm = np.array([xm, ym, zm], dtype=float)
    vel_mp = np.array([vxp, vyp, vzp], dtype=float)
    vel_ms = np.array([vxs, vys, vzs], dtype=float)
    vel_mm = np.array([vxm, vym, vzm], dtype=float)

    return dict(
        rp_km=rp_km, rs_m=rs_m,
        ms=ms, mp=mp, mm=mm,
        am_AU=am_AU,
        rhill_AU=rhill,
        pos_mp=pos_mp, pos_ms=pos_ms, pos_mm=pos_mm,
        vel_mp=vel_mp, vel_ms=vel_ms, vel_mm=vel_mm
    )
