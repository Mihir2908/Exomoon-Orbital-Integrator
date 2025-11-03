import os, sys, importlib

# Ensure src/ is importable regardless of current working dir
SRC_DIR = os.path.dirname(__file__)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Debug: verify the modules being imported are from your src path
import exomoon
print("exomoon package file:", exomoon.__file__)
sim_mod = importlib.import_module("exomoon.simulation")
print("exomoon.simulation file:", sim_mod.__file__)

from exomoon.params import SystemParams
from exomoon.simulation import run_simulation
import exomoon.plotting.anim as anim
#from exomoon.plotting.anim import build_animation

def main():
    p = SystemParams()
    sim = run_simulation(p)

    fig = anim.build_animation(
        traj=sim["traj"],
        a_inner_au=sim["a_inner_au"],
        a_outer_au=sim["a_outer_au"],
        open_in_browser=True
    )
    fig.show(renderer="browser")
    fig.write_html("orbit_anim.html")

if __name__ == "__main__":
    main()