import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from exomoon.params import SystemParams
from exomoon.initial_conditions import initial_state
from exomoon.simulation import run_simulation_for_years
from exomoon.habitable_zone import hz_bounds_au
from exomoon.constants import rsun

p = SystemParams(
    Ts=5772.0, rs_solar=1.0, ms_solar=1.0,
    mp_earth=1.0, ap_AU=1.0, ep=0.0,
    mm_earth=0.107, am_hill=0.05, em=0.0, moon_retrograde=False,
)
t_sim = 10.0

st = initial_state(p)
pos_mp, pos_ms, pos_mm = st["pos_mp"], st["pos_ms"], st["pos_mm"]
print("pos_mp (planet):", pos_mp)
print("pos_ms (star):  ", pos_ms)
print("pos_mm (moon):  ", pos_mm)
print("planet_star_dist0 =", np.linalg.norm(pos_mp - pos_ms))
print("moon_star_dist0   =", np.linalg.norm(pos_mm - pos_ms))
print("rhill_AU =", st["rhill_AU"])
print()

a_in, a_out = hz_bounds_au(p.Ts, p.rs_solar * rsun)
print(f"HZ bounds: a_inner_au={a_in:.6f}  a_outer_au={a_out:.6f}")
print()

sim = run_simulation_for_years(p, t_sim)
traj = sim["traj"]
planet_star_dist = np.linalg.norm(traj["xyzarr_mp"] - traj["xyzarr_ms"], axis=1)
moon_star_dist   = np.linalg.norm(traj["xyzarr_mm"] - traj["xyzarr_ms"], axis=1)
print(f"TRUE sim: planet_star_dist min={planet_star_dist.min():.6f} max={planet_star_dist.max():.6f}")
print(f"TRUE sim: moon_star_dist   min={moon_star_dist.min():.6f} max={moon_star_dist.max():.6f}")
print(f"TRUE sim: is moon_star_dist within HZ for the whole sim? "
      f"{bool(np.all((moon_star_dist >= a_in) & (moon_star_dist <= a_out)))}")
