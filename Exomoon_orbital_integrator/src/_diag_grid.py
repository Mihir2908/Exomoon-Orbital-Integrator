"""
Diagnostic: run the actual predict_stability_map() grid sweep (the same function
agent_service.py /ml/predict calls) and inspect where/why every candidate ends up
flagged unstable, to find out at what am_hill the model starts predicting escape.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from exomoon.ml.inference import predict_stability_map

system_params = {
    "ms_solar": 1.0, "rs_solar": 1.0, "Ts": 5772.0,
    "mp_earth": 1.0, "ap_AU": 1.0, "ep": 0.0,
}

for retro in (False, True):
    print(f"=== moon_retrograde={retro} ===")
    result = predict_stability_map(
        system_params=system_params,
        t_sim=10.0,
        moon_retrograde=retro,
        em=0.0,
        mm_resolution=20,
        am_resolution=20,
        model_dir="models",
        n_steps=1000,
        rnn_type="gru",
    )
    if not result.get("ok"):
        print("FAILED:", result)
        continue

    map_both = np.array(result["map_both"])
    map_stable = np.array(result["map_stable"])
    map_habitable = np.array(result["map_habitable"])
    am_grid = np.array(result["am_grid"])
    mm_grid = np.array(result["mm_grid"])

    print(f"am_grid range: {am_grid.min():.4f} -> {am_grid.max():.4f}")
    print(f"mm_grid range: {mm_grid.min():.4f} -> {mm_grid.max():.4f}")
    print(f"map_stable sum:    {map_stable.sum()} / {map_stable.size}")
    print(f"map_habitable sum: {map_habitable.sum()} / {map_habitable.size}")
    print(f"map_both sum:      {map_both.sum()} / {map_both.size}")
    print(f"valid_mm_range: {result['valid_mm_range']}")

    # Show stability as a function of am_hill for the smallest mm_earth row (easiest case)
    print("Row 0 (smallest mm_earth) stable flags across am_grid:")
    print("  am_hill:", np.round(am_grid, 3))
    print("  stable: ", map_stable[0].astype(int))
    print("  habitable:", map_habitable[0].astype(int))
    print()
