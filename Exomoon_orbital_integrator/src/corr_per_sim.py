"""
Per-simulation correlation matrix.

Collapses the ~1000-step per-sim parquet rows into one row per simulation:
  - SYS_COLS (14 system params, constant per sim — take first row)
  - frac_stable, frac_habitable, frac_temp_habitable (fraction of timesteps
    where the binary flag = 1)

Outputs:
  corr_per_sim.csv         — full correlation matrix
  corr_full.png            — full heatmap (all columns × all columns)
  corr_focused.png         — focused: SYS_COLS × outcome fractions only
  (also prints ranked correlators to stdout)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PARQUET = "ml_dataset.parquet"
OUT_DIR = "."   # save alongside this script in src/

SYS_COLS = [
    "ms_solar", "rs_solar", "Ts",
    "mp_earth", "ap_AU", "ep",
    "mm_earth", "am_hill", "am_AU", "em",
    "moon_retrograde", "t_sim",
    "rhill_AU", "a_inner_au", "a_outer_au",
]
FLAG_COLS  = ["stable", "habitable", "habitable_from_temp"]
STATE_NORM = ["moon_star_dist_norm", "moon_planet_dist_norm", "moon_temp_norm"]

# ── 1. Load & collapse ──────────────────────────────────────────────────────
print("Loading parquet ...")
df = pd.read_parquet(PARQUET, columns=["sim_id"] + SYS_COLS + FLAG_COLS + STATE_NORM)
print(f"  {len(df):,} rows, {df['sim_id'].nunique()} simulations")

# SYS_COLS are constant within a sim — first row suffices
sys_per_sim = df.groupby("sim_id")[SYS_COLS].first()

# Flag fractions: mean of 0/1 column = fraction of steps where flag is True
frac_per_sim = df.groupby("sim_id")[FLAG_COLS].mean()
frac_per_sim.columns = [f"frac_{c}" for c in FLAG_COLS]

# State aggregates: mean and std of normalized trajectory features per sim
state_mean = df.groupby("sim_id")[STATE_NORM].mean()
state_mean.columns = [f"mean_{c}" for c in STATE_NORM]
state_std  = df.groupby("sim_id")[STATE_NORM].std()
state_std.columns  = [f"std_{c}"  for c in STATE_NORM]

per_sim = pd.concat([sys_per_sim, frac_per_sim, state_mean, state_std], axis=1)
print(f"  Per-sim table: {per_sim.shape[0]} rows x {per_sim.shape[1]} cols\n")

# ── 2. Correlation matrix ────────────────────────────────────────────────────
corr = per_sim.corr(method="pearson")
corr.to_csv(os.path.join(OUT_DIR, "corr_per_sim.csv"))
print("Saved corr_per_sim.csv")

# ── 3. Focused printout: SYS_COLS → outcome fracs ───────────────────────────
frac_cols  = [c for c in per_sim.columns if c.startswith("frac_")]
state_cols = [c for c in per_sim.columns if c.startswith("mean_") or c.startswith("std_")]
row_order  = SYS_COLS + state_cols
focused = corr.loc[row_order, frac_cols]

print("\n-- Correlations: SYS_COLS + state aggregates -> outcome fractions --")
print(focused.to_string(float_format=lambda x: f"{x:+.3f}"))

for fc in frac_cols:
    ranked = focused[fc].abs().sort_values(ascending=False)
    print(f"\n  Top correlators with {fc} (by |r|):")
    for col, val in ranked.items():
        print(f"    {col:<35s}  r = {focused.loc[col, fc]:+.3f}")

# ── 4. Full heatmap ──────────────────────────────────────────────────────────
n = len(corr)
fig, ax = plt.subplots(figsize=(max(n * 0.9, 14), max(n * 0.8, 12)))
mask = np.eye(n, dtype=bool)   # hide diagonal (self-correlation = 1)
sns.heatmap(
    corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
    vmin=-1, vmax=1, square=True, ax=ax,
    linewidths=0.4, mask=mask,
    annot_kws={"size": 7},
)
ax.set_title(
    "Per-simulation correlation matrix\n"
    "(SYS_COLS + outcome fractions, N=2979 sims)",
    fontsize=13, pad=12,
)
plt.tight_layout()
out_full = os.path.join(OUT_DIR, "corr_full.png")
plt.savefig(out_full, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved {out_full}")

# ── 5. Focused heatmap: SYS_COLS + state aggregates x outcome fracs ─────────
fig, ax = plt.subplots(figsize=(7, 14))
sns.heatmap(
    focused, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
    vmin=-1, vmax=1, square=True, ax=ax,
    linewidths=0.5,
    annot_kws={"size": 9},
)
ax.axhline(len(SYS_COLS), color="black", linewidth=2)
ax.set_title(
    "SYS_COLS + state aggregates -> outcome fraction correlations\n"
    "(N=2979 sims; line separates system params from trajectory stats)",
    fontsize=11, pad=10,
)
plt.tight_layout()
out_focused = os.path.join(OUT_DIR, "corr_focused.png")
plt.savefig(out_focused, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out_focused}")
