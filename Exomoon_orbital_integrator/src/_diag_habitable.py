import pandas as pd
import numpy as np

df = pd.read_parquet("ml_dataset.parquet")

print("Overall habitable rate:", df["habitable"].mean())
print("Overall stable rate:   ", df["stable"].mean())
print()

per_sim = df.groupby("sim_id")[["habitable", "stable", "ap_AU", "a_inner_au", "a_outer_au"]].first()
per_sim["in_hz"] = (per_sim["ap_AU"] >= per_sim["a_inner_au"]) & (per_sim["ap_AU"] <= per_sim["a_outer_au"])
print("Fraction of SIMULATIONS where the planet's ap_AU itself starts inside the HZ:", per_sim["in_hz"].mean())
print()

# habitable rate among rows where the planet is actually in the HZ at all
in_hz_sims = per_sim[per_sim["in_hz"]].index
df_in_hz = df[df["sim_id"].isin(in_hz_sims)]
print(f"Sims where planet ap_AU is in HZ: {len(in_hz_sims)} / {per_sim.shape[0]}")
print("habitable rate within those sims:", df_in_hz["habitable"].mean())
print()
print(per_sim[["ap_AU", "a_inner_au", "a_outer_au"]].describe())
