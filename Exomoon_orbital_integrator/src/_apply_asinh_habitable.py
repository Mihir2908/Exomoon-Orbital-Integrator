"""
One-off post-processing: replace moon_star_dist_norm with
arcsinh((moon_star_dist - a_inner_au) / hz_width) in the already-sampled
ml_dataset.parquet. Raw moon_star_dist/a_inner_au/a_outer_au columns survive
alongside the engineered column, so this is a pure derived-column recompute --
no resampling/resimulation needed. moon_planet_dist_norm is untouched.
"""
import shutil
import numpy as np
import pandas as pd

SRC = "ml_dataset.parquet"
BACKUP = "ml_dataset_pre_asinh_backup.parquet"

shutil.copyfile(SRC, BACKUP)
print(f"Backed up to {BACKUP}")

df = pd.read_parquet(SRC)
print(f"Loaded {len(df):,} rows, {df['sim_id'].nunique():,} sims")

a_in  = df["a_inner_au"].values
a_out = df["a_outer_au"].values
hz_width = np.clip(a_out - a_in, 1e-6, None)

old = df["moon_star_dist_norm"].copy()
df["moon_star_dist_norm"] = np.arcsinh((df["moon_star_dist"].values - a_in) / hz_width)

df.to_parquet(SRC, index=False)
print(f"Wrote {len(df):,} rows with arcsinh-compressed moon_star_dist_norm -> {SRC}")
print()
print("Before (HZ-ratio, uncompressed):")
print(old.describe())
print()
print("After (arcsinh-compressed):")
print(df["moon_star_dist_norm"].describe())
