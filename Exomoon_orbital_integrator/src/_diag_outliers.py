import pandas as pd
import numpy as np

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)

df = pd.read_parquet("ml_dataset.parquet")
print("Total rows:", len(df))
print("Total sims:", df["sim_id"].nunique())
print()

dist_speed_cols = ["moon_planet_dist", "moon_star_dist", "planet_star_dist", "moon_speed", "planet_speed"]
print("=== Overall describe (raw units) ===")
print(df[dist_speed_cols].describe(percentiles=[.5, .9, .99, .999]))
print()

# Per-simulation max values — find the worst offenders
per_sim_max = df.groupby("sim_id")[dist_speed_cols].max()
per_sim_first = df.groupby("sim_id")[["am_hill", "mm_earth", "mp_earth", "ms_solar", "ap_AU", "rhill_AU", "t_sim"]].first()
merged = per_sim_max.join(per_sim_first)

print("=== Top 15 sims by max moon_planet_dist ===")
top_mpd = merged.sort_values("moon_planet_dist", ascending=False).head(15)
print(top_mpd[["moon_planet_dist", "rhill_AU", "am_hill", "mm_earth", "mp_earth", "ap_AU", "ms_solar", "t_sim"]])
print()

print("=== Top 15 sims by max moon_speed ===")
top_speed = merged.sort_values("moon_speed", ascending=False).head(15)
print(top_speed[["moon_speed", "rhill_AU", "am_hill", "mm_earth", "mp_earth", "ap_AU", "ms_solar", "t_sim"]])
print()

# Ratio of max distance to rhill_AU -- how far beyond the Hill sphere did the moon end up
merged["escape_ratio"] = merged["moon_planet_dist"] / merged["rhill_AU"]
print("=== escape_ratio = max(moon_planet_dist) / rhill_AU  -- describe ===")
print(merged["escape_ratio"].describe(percentiles=[.5, .9, .99, .999]))
print()

print("=== Top 15 sims by escape_ratio ===")
top_ratio = merged.sort_values("escape_ratio", ascending=False).head(15)
print(top_ratio[["escape_ratio", "moon_planet_dist", "rhill_AU", "am_hill", "mm_earth", "mp_earth", "ap_AU", "t_sim"]])
print()

# Correlation between am_hill and the per-sim outlier severity
print("=== Correlation: am_hill vs escape_ratio, am_hill vs max moon_speed ===")
print(merged[["am_hill", "escape_ratio", "moon_speed"]].corr())
print()

# How many sims have wildly large values (orders of magnitude beyond a "sane" orbit)
n_extreme = (merged["escape_ratio"] > 10).sum()
n_total = len(merged)
print(f"Sims with escape_ratio > 10 (moon ended up >10x its own Hill radius away): {n_extreme} / {n_total}")
n_extreme_speed = (merged["moon_speed"] > merged["moon_speed"].median() * 50).sum()
print(f"Sims with max moon_speed > 50x median: {n_extreme_speed} / {n_total}")
