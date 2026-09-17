"""
One-off post-processing: add moon_temp_norm / habitable_from_temp columns to the
already-sampled ml_dataset.parquet. moon_star_dist/Ts/rs_solar/a_inner_au/a_outer_au
already exist, so this is a pure derived-column addition -- no resampling needed.
"""
import shutil
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from exomoon.habitable_zone import moon_effective_temp_K
from exomoon.constants import rsun, au

SRC = "ml_dataset.parquet"
BACKUP = "ml_dataset_pre_temphead_backup.parquet"

shutil.copyfile(SRC, BACKUP)
print(f"Backed up to {BACKUP}")

df = pd.read_parquet(SRC)
print(f"Loaded {len(df):,} rows, {df['sim_id'].nunique():,} sims")

rs_m  = df["rs_solar"].values * rsun
Ts_K  = df["Ts"].values
a_in  = df["a_inner_au"].values
a_out = df["a_outer_au"].values

Tm_K   = moon_effective_temp_K(Ts_K, rs_m, df["moon_star_dist"].values * au)
T_hot  = moon_effective_temp_K(Ts_K, rs_m, a_in * au)
T_cold = moon_effective_temp_K(Ts_K, rs_m, a_out * au)
temp_width = np.clip(T_hot - T_cold, 1e-6, None)

df["habitable_from_temp"] = ((Tm_K <= T_hot) & (Tm_K >= T_cold)).astype(int)
df["moon_temp_norm"] = np.arcsinh((Tm_K - T_cold) / temp_width)

df.to_parquet(SRC, index=False)
print(f"Wrote {len(df):,} rows with moon_temp_norm/habitable_from_temp added -> {SRC}")
print()
print("Consistency check (should match 'habitable' almost exactly):")
mismatch = (df["habitable"] != df["habitable_from_temp"]).sum()
print(f"  habitable vs habitable_from_temp label mismatch: {mismatch}/{len(df)} rows")
print()
print(df[["moon_temp_norm", "habitable_from_temp"]].describe())
