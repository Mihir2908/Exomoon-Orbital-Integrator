"""
One-off post-processing: add moon_planet_dist_norm / moon_star_dist_norm columns
to the already-sampled ml_dataset.parquet, matching the normalization now baked
into run_ml_dataset.py's _freeze_post_escape(). a_inner_au/a_outer_au/rhill_AU
already exist in the parquet (from eda.py's existing ML columns block), so this
is a pure derived-column addition -- no resampling needed.
"""
import shutil
import numpy as np
import pandas as pd

SRC = "ml_dataset.parquet"
BACKUP = "ml_dataset_pre_norm_backup.parquet"

shutil.copyfile(SRC, BACKUP)
print(f"Backed up to {BACKUP}")

df = pd.read_parquet(SRC)
print(f"Loaded {len(df):,} rows, {df['sim_id'].nunique():,} sims")

rhill = df["rhill_AU"].values
a_in  = df["a_inner_au"].values
a_out = df["a_outer_au"].values
hz_width = np.clip(a_out - a_in, 1e-6, None)

df["moon_planet_dist_norm"] = df["moon_planet_dist"].values / np.clip(rhill, 1e-9, None)
df["moon_star_dist_norm"]   = (df["moon_star_dist"].values - a_in) / hz_width

df.to_parquet(SRC, index=False)
print(f"Wrote {len(df):,} rows with moon_planet_dist_norm/moon_star_dist_norm added -> {SRC}")
print()
print(df[["moon_planet_dist_norm", "moon_star_dist_norm"]].describe())
