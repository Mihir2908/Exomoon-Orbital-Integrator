"""
One-off post-processing: apply the same post-escape freeze logic now baked into
run_ml_dataset.py's _freeze_post_escape() to the already-sampled ml_dataset.parquet,
so the existing 2979-sim dataset doesn't need to be regenerated from scratch.
"""
import shutil
import numpy as np
import pandas as pd

SRC = "ml_dataset.parquet"
BACKUP = "ml_dataset_raw_backup.parquet"

shutil.copyfile(SRC, BACKUP)
print(f"Backed up original to {BACKUP}")

df = pd.read_parquet(SRC)
print(f"Loaded {len(df):,} rows, {df['sim_id'].nunique():,} sims")


def freeze_group(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("t_frac").reset_index(drop=True)
    rhill = g["rhill_AU"].values
    a_in  = g["a_inner_au"].values
    a_out = g["a_outer_au"].values

    stable_before_freeze = g["moon_planet_dist"].values <= rhill
    unstable_idx = np.where(~stable_before_freeze)[0]

    if len(unstable_idx) > 0:
        freeze_idx = unstable_idx[0]
        for col in ("moon_planet_dist", "moon_star_dist", "moon_speed"):
            vals = g[col].values.copy()
            vals[freeze_idx:] = vals[freeze_idx]
            g[col] = vals

    g["stable"]    = (g["moon_planet_dist"].values <= rhill).astype(int)
    g["habitable"] = (
        (g["moon_star_dist"].values >= a_in) & (g["moon_star_dist"].values <= a_out)
    ).astype(int)
    return g


print("Applying freeze per simulation (this may take a minute)...")
df = df.groupby("sim_id", group_keys=False).apply(freeze_group)

df.to_parquet(SRC, index=False)
print(f"Wrote corrected dataset back to {SRC} ({len(df):,} rows)")
