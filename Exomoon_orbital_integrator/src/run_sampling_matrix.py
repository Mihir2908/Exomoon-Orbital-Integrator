"""
run_sampling_matrix.py

Sampling matrix diagnostic: 2D histograms for all 21 pairwise combinations of
the 7 key LHS-sampled parameters in ml_dataset.parquet.

Bins are equal-width in sampling space (linear params: equal linear width;
log params: equal log width). Under ideal independent LHS, every bin (row/column)
should contain N/3 sims and every cell should contain N/9 sims. Deviations from
N/9 per cell reveal structural imbalances caused by conditional sampling (ap_AU
conditional HZ mechanism) or adaptive bounds (am_hill Roche limit, mm_earth cap)
— not LHS geometry artifacts.

Outputs:
  sampling_matrix_raw.png        -- raw unique sim counts per cell
  sampling_matrix_normalized.png -- counts / expected_per_cell (ratio; 1.0 = LHS ideal)
  sampling_matrix_undersampled.txt -- cells with raw count below 50% of expected
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas required: pip install pandas pyarrow")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PARQUET = os.path.join(os.path.dirname(__file__), "ml_dataset.parquet")
OUT_RAW  = os.path.join(os.path.dirname(__file__), "sampling_matrix_raw.png")
OUT_NORM = os.path.join(os.path.dirname(__file__), "sampling_matrix_normalized.png")
OUT_TXT  = os.path.join(os.path.dirname(__file__), "sampling_matrix_undersampled.txt")

N_BINS = 3
UNDERSAMPLE_THRESHOLD = 0.5   # flag cells below this ratio of expected count
OVERSAMPLE_THRESHOLD  = 2.0   # flag cells above this ratio

# ---------------------------------------------------------------------------
# Parameter specs: (lo, hi, scale)
# Bins are computed as N_BINS equal-width intervals in sampling space.
# linear -> equal linear width; log -> equal log width.
# ---------------------------------------------------------------------------
PARAM_SPECS = {
    "ms_solar": (0.08,  2.0,   "linear"),
    "ep":       (0.0,   0.35,  "linear"),
    "ap_AU":    (0.01,  3.5,   "linear"),
    "mp_earth": (0.5,   300.0, "log"),
    "mm_earth": (0.107, 3.0,   "log"),
    "am_hill":  (0.0,   1.0,   "linear"),
    "em":       (0.0,   0.25,  "linear"),
}


def _make_bins(lo: float, hi: float, scale: str, n: int = N_BINS) -> dict:
    """Compute equal-width bin edges and labels for one parameter."""
    if scale == "log":
        edges = np.exp(np.linspace(np.log(lo), np.log(hi), n + 1))
    else:
        edges = np.linspace(lo, hi, n + 1)

    labels = []
    for i in range(n):
        lo_v, hi_v = edges[i], edges[i + 1]
        # Format: use 3 sig figs, strip trailing zeros
        lo_s = f"{lo_v:.3g}"
        hi_s = f"{hi_v:.3g}"
        labels.append(f"{lo_s}-{hi_s}")

    return {"edges": edges.tolist(), "labels": labels}


PARAM_BINS = {name: _make_bins(*spec) for name, spec in PARAM_SPECS.items()}

PARAMS = list(PARAM_BINS.keys())
PAIRS  = list(combinations(PARAMS, 2))   # 21 pairs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def compute_matrix(sims: "pd.DataFrame", px: str, py: str) -> np.ndarray:
    """Return 2D count matrix shape (n_bins_y, n_bins_x)."""
    edges_x = PARAM_BINS[px]["edges"]
    edges_y = PARAM_BINS[py]["edges"]
    n_x = len(edges_x) - 1
    n_y = len(edges_y) - 1

    x_idx = np.clip(np.digitize(sims[px].values, edges_x) - 1, 0, n_x - 1)
    y_idx = np.clip(np.digitize(sims[py].values, edges_y) - 1, 0, n_y - 1)

    mat = np.zeros((n_y, n_x), dtype=int)
    for xi, yi in zip(x_idx, y_idx):
        mat[yi, xi] += 1
    return mat


def plot_matrix(ax, mat, labels_x, labels_y, cmap, vmin, vmax, fmt):
    n_y, n_x = mat.shape
    im = ax.imshow(mat, cmap=cmap, aspect="auto", origin="lower",
                   vmin=vmin, vmax=vmax)
    ax.set_xticks(range(n_x))
    ax.set_xticklabels(labels_x, fontsize=5.5)
    ax.set_yticks(range(n_y))
    ax.set_yticklabels(labels_y, fontsize=5.5)
    threshold = (vmin + vmax) / 2
    for r in range(n_y):
        for c in range(n_x):
            val = mat[r, c]
            txt = fmt.format(val)
            color = "white" if val < threshold else "black"
            ax.text(c, r, txt, ha="center", va="center", fontsize=5.5,
                    color=color, fontweight="bold")
    return im


def make_figure(pairs, sims, N_sims, mode="raw"):
    n_pairs = len(pairs)
    n_cols = 7
    n_rows = (n_pairs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(30, 13),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.45})
    axes_flat = axes.flatten()

    expected_per_cell = N_sims / (N_BINS * N_BINS)

    for i, (px, py) in enumerate(pairs):
        ax = axes_flat[i]
        mat = compute_matrix(sims, px, py)
        labels_x = PARAM_BINS[px]["labels"]
        labels_y = PARAM_BINS[py]["labels"]

        if mode == "raw":
            vmax = mat.max() if mat.max() > 0 else 1
            im = plot_matrix(ax, mat, labels_x, labels_y,
                             cmap="YlOrRd", vmin=0, vmax=vmax, fmt="{:d}")
            cbar_label = "sim count"
        else:
            mat_f = mat.astype(float) / expected_per_cell
            im = plot_matrix(ax, mat_f, labels_x, labels_y,
                             cmap="RdYlGn", vmin=0.0, vmax=2.0, fmt="{:.2f}")
            cbar_label = f"ratio  (expected≈{expected_per_cell:.0f})"

        cbar = fig.colorbar(im, ax=ax, shrink=0.72, pad=0.04)
        cbar.ax.tick_params(labelsize=5)
        cbar.set_label(cbar_label, fontsize=5)

        ax.set_xlabel(px, fontsize=6.5, labelpad=2)
        ax.set_ylabel(py, fontsize=6.5, labelpad=2)
        ax.set_title(f"{px}  x  {py}", fontsize=6.5, pad=3)

    for j in range(n_pairs, len(axes_flat)):
        axes_flat[j].set_visible(False)

    if mode == "raw":
        mode_label = "Raw Unique Sim Counts  (equal-width bins in sampling space)"
    else:
        mode_label = (f"Normalized Counts  (observed / expected≈{expected_per_cell:.0f};  "
                      f"1.0 = LHS ideal;  deviations = structural sampling bias)")
    fig.suptitle(f"Stage C Dataset Sampling Matrix — {mode_label}\n"
                 f"N = {N_sims} unique sims  |  {N_BINS} equal-width bins per parameter  "
                 f"|  ml_dataset.parquet",
                 fontsize=10, y=1.01)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Loading {PARQUET} ...")
    df = pd.read_parquet(PARQUET)
    sims = df.groupby("sim_id").first().reset_index()
    N_sims = len(sims)
    print(f"  {len(df):,} timestep rows  |  {N_sims} unique sims")

    missing = [p for p in PARAMS if p not in sims.columns]
    if missing:
        sys.exit(f"Missing columns in parquet: {missing}")

    expected_per_cell = N_sims / (N_BINS * N_BINS)
    expected_per_bin  = N_sims / N_BINS

    print(f"\n  Equal-width bins in sampling space  ({N_BINS} bins per parameter)")
    print(f"  Expected sims per bin  (row/column total): ~{expected_per_bin:.0f}")
    print(f"  Expected sims per cell (joint combination): ~{expected_per_cell:.0f}")
    print(f"\n  Bin edges:")
    for p in PARAMS:
        lo, hi, scale = PARAM_SPECS[p]
        edges = PARAM_BINS[p]["edges"]
        labels = PARAM_BINS[p]["labels"]
        print(f"    {p:12s}  ({scale:6s})  edges: {[f'{e:.3g}' for e in edges]}  =>  {labels}")

    print(f"\n  Actual parameter ranges in dataset:")
    for p in PARAMS:
        lo, hi = sims[p].min(), sims[p].max()
        print(f"    {p:12s}  [{lo:.4g}, {hi:.4g}]")

    # Per-bin row totals — sanity check that each bin has ~expected_per_bin sims
    print(f"\n  Per-bin sim counts  (should be ~{expected_per_bin:.0f} each):")
    for p in PARAMS:
        edges = PARAM_BINS[p]["edges"]
        n = N_BINS
        counts = []
        for i in range(n):
            mask = (sims[p] >= edges[i]) & (sims[p] < edges[i + 1])
            if i == n - 1:
                mask = (sims[p] >= edges[i]) & (sims[p] <= edges[i + 1])
            counts.append(int(mask.sum()))
        labels = PARAM_BINS[p]["labels"]
        row = "  ".join(f"{lbl}: {cnt}" for lbl, cnt in zip(labels, counts))
        print(f"    {p:12s}  {row}")

    # --- Figure 1: raw counts ---
    print("\nGenerating raw count figure ...")
    fig1 = make_figure(PAIRS, sims, N_sims, mode="raw")
    fig1.savefig(OUT_RAW, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"  Saved: {OUT_RAW}")

    # --- Figure 2: normalized ---
    print("Generating normalized figure ...")
    fig2 = make_figure(PAIRS, sims, N_sims, mode="normalized")
    fig2.savefig(OUT_NORM, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {OUT_NORM}")

    # --- Text summary ---
    low_thresh  = expected_per_cell * UNDERSAMPLE_THRESHOLD
    high_thresh = expected_per_cell * OVERSAMPLE_THRESHOLD
    lines = [
        f"Sampling matrix diagnostic  |  N={N_sims} sims  |  ml_dataset.parquet",
        f"Equal-width bins in sampling space  |  {N_BINS} bins per parameter",
        f"Expected per cell (independent LHS): ~{expected_per_cell:.0f} sims",
        f"Under-flag: < {low_thresh:.0f} sims ({UNDERSAMPLE_THRESHOLD:.0f}x expected)  |  "
        f"Over-flag: > {high_thresh:.0f} sims ({OVERSAMPLE_THRESHOLD:.0f}x expected)",
        "",
    ]

    any_flag = False
    for px, py in PAIRS:
        mat = compute_matrix(sims, px, py)
        labels_x = PARAM_BINS[px]["labels"]
        labels_y = PARAM_BINS[py]["labels"]
        n_y, n_x = mat.shape

        pair_lines = []
        for r in range(n_y):
            for c in range(n_x):
                count = mat[r, c]
                ratio = count / expected_per_cell
                if count < low_thresh or count > high_thresh:
                    tag = "UNDER" if count < low_thresh else "OVER"
                    lx = labels_x[c]
                    ly = labels_y[r]
                    pair_lines.append(
                        f"  [{tag}]  {px}={lx}  x  {py}={ly}"
                        f"   count={count}  expected≈{expected_per_cell:.0f}  ratio={ratio:.2f}"
                    )

        if pair_lines:
            lines.append(f"{px} x {py}:")
            lines.extend(pair_lines)
            lines.append("")
            any_flag = True

    if not any_flag:
        lines.append("No cells exceeded under/over thresholds — dataset matches LHS ideal.")

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {OUT_TXT}")
    print("\nDone.")


if __name__ == "__main__":
    main()
