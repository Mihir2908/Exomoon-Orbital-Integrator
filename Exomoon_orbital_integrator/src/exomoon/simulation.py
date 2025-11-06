import numpy as np
from exomoon.params import SystemParams
from exomoon.initial_conditions import initial_state
import exomoon.integrator as integrator
#from exomoon.integrator import leapfrog_integrate
from exomoon.habitable_zone import hz_bounds_au

def run_simulation(p: SystemParams):
    st = initial_state(p)

    # Periods (dimensionless units consistent with your scaling)
    orbprd_mm_ms = 2.0 * np.pi * p.ap_AU**1.5 / (st["mp"] + st["ms"]) ** 0.5
    orbprd_mm_mp = 2.0 * np.pi * st["am_AU"]**1.5 / (st["mp"] + st["mm"]) ** 0.5

    dt = orbprd_mm_mp / 1_000.0
    t_end = orbprd_mm_ms

    traj = integrator.leapfrog_integrate(st, t_end, dt)
    a_inner_au, a_outer_au = hz_bounds_au(p.Ts, st["rs_m"])

    return dict(
        params=p,
        state=st,
        traj=traj,
        dt=dt,
        t_end=t_end,
        a_inner_au=a_inner_au,
        a_outer_au=a_outer_au,
    )

def run_simulation_for_years(p: SystemParams, years: float):
    st = initial_state(p)

    # Periods (dimensionless units consistent with your scaling)
    orbprd_mm_ms = 2.0 * np.pi * p.ap_AU**1.5 / (st["mp"] + st["ms"]) ** 0.5
    orbprd_mm_mp = 2.0 * np.pi * st["am_AU"]**1.5 / (st["mp"] + st["mm"]) ** 0.5

    dt = max(orbprd_mm_mp / 1_000.0, float(years) / 50_000.0)
    t_end = float(years)

    #dt = orbprd_mm_mp / 1_000.0
    #t_end = orbprd_mm_ms

    traj = integrator.leapfrog_integrate(st, t_end, dt)
    a_inner_au, a_outer_au = hz_bounds_au(p.Ts, st["rs_m"])

    return dict(
        params=p,
        state=st,
        traj=traj,
        dt=dt,
        t_end=t_end,
        a_inner_au=a_inner_au,
        a_outer_au=a_outer_au,
    )