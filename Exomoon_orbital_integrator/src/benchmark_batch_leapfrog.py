"""
benchmark_batch_leapfrog.py — Validate and time the batched PyTorch leapfrog.

Three checks:
  1. CORRECTNESS: compare batch leapfrog vs Numba integrator for a single
     reference cell (K-452b-v2 prograde, am_hill≈0.40, mm_earth≈mid-grid).
     Both run at the same adaptive dt chosen by batch_leapfrog_trajectories.
     Pass criterion: final position error < 0.1% of Hill radius.

  2. SPEED: time the full batch run (2500 cells).
     Target: < 60 seconds on CPU.

  3. COVERAGE: compare map_stable / map_habitable from batch leapfrog vs the
     ground-truth .npz for K-452b-v2 prograde.  System params and grid
     resolution are read from the .meta.json — no hardcoding.

CRITICAL — no MLP infrastructure touched by this script.
"""

import os, sys, time, json
import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from exomoon.ml.batch_leapfrog import batch_leapfrog_trajectories
from exomoon.integrator import leapfrog_integrate
from exomoon.initial_conditions import initial_state
from exomoon.params import SystemParams

# ── Load ground-truth meta (system params + resolution) ───────────────────────
GT_META_PATH = os.path.join(SRC, "ground_truth_grids",
                             "Kepler_452_b_v2_prograde.meta.json")
with open(GT_META_PATH) as f:
    gt_meta = json.load(f)

sp = gt_meta["system_params"]
SYSTEM_PARAMS = {
    "ms_solar": sp["ms_solar"],
    "rs_solar": sp["rs_solar"],
    "Ts":       sp["Ts"],
    "mp_earth": sp["mp_earth"],
    "ap_AU":    sp["ap_AU"],
    "ep":       sp.get("ep", 0.0),
}
T_SIM          = float(gt_meta["t_sim"])
MM_RESOLUTION  = int(gt_meta["mm_resolution"])
AM_RESOLUTION  = int(gt_meta["am_resolution"])
FULL_MM_RES    = 50
FULL_AM_RES    = 50

print(f"System (from meta.json): {SYSTEM_PARAMS}")
print(f"t_sim={T_SIM}yr  GT grid: {MM_RESOLUTION}×{AM_RESOLUTION}")
print(f"Full batch grid: {FULL_MM_RES}×{FULL_AM_RES}")


# ── Helper ─────────────────────────────────────────────────────────────────────

def pr_f1(gt_mask: np.ndarray, ml_mask: np.ndarray, name=""):
    tp    = int((gt_mask & ml_mask).sum())
    n_gt  = int(gt_mask.sum())
    n_ml  = int(ml_mask.sum())
    rec   = tp / n_gt  if n_gt > 0 else None
    prec  = tp / n_ml  if n_ml > 0 else None
    f1    = (2 * rec * prec / (rec + prec)) if (rec and prec and rec + prec > 0) else None
    return rec, prec, f1


def fmt(v, width=7):
    return f"{v:.4f}" if v is not None else f"{'--':>{width}}"


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 2: SPEED — run 50×50 grid (timing is the primary purpose here)
# (Run first so we can use the adaptive n_phys/dt for CHECK 1 Numba comparison)
# ══════════════════════════════════════════════════════════════════════════════

print()
print("=" * 72)
print(f"CHECK 2: Speed — {FULL_MM_RES}×{FULL_AM_RES} batch ({FULL_MM_RES*FULL_AM_RES} cells)")
print("=" * 72)

t_batch_start = time.perf_counter()
result = batch_leapfrog_trajectories(
    system_params=SYSTEM_PARAMS,
    t_sim=T_SIM,
    moon_retrograde=False,
    em=0.0,
    mm_resolution=FULL_MM_RES,
    am_resolution=FULL_AM_RES,
    n_steps=1000,
    escape_factor=1.0,
)
t_batch_end = time.perf_counter()

assert result["ok"], "batch_leapfrog_trajectories returned ok=False"

n_phys   = result["n_phys"]
dt_phys  = result["dt_phys"]
n_out    = result["n_out"]
elapsed  = result["elapsed_s"]
rhill_AU = result["rhill_AU"]

print(f"\n  Adaptive timestep: dt={dt_phys:.5f} yr  n_phys={n_phys}  n_out={n_out}")
print(f"  Total elapsed: {elapsed:.2f} s")
print(f"  Per-cell:      {elapsed / (FULL_MM_RES * FULL_AM_RES) * 1000:.2f} ms")
print(f"  Per step:      {elapsed / n_phys * 1000:.4f} ms")

TARGET = 60.0
status2 = "PASS ✓ (< 60 s)" if elapsed < TARGET else f"FAIL ✗ (target {TARGET} s)"
print(f"  Target: < {TARGET:.0f} s  →  {status2}")

mem_traj_MB = (result["traj_moon"].nbytes +
               result["traj_planet"].nbytes +
               result["traj_star"].nbytes) / 1e6
mem_dist_MB = (result["moon_planet_dist"].nbytes +
               result["moon_star_dist"].nbytes) / 1e6
print(f"\n  Memory — trajectory arrays: {mem_traj_MB:.1f} MB")
print(f"  Memory — distance arrays:   {mem_dist_MB:.1f} MB")

mm_grid = np.array(result["mm_grid"])
am_grid = np.array(result["am_grid"])

# Stable-cell fraction sanity check
map_stable = np.array(result["map_stable"])
stable_frac = map_stable.mean()
print(f"\n  map_stable fraction: {stable_frac:.3f}  (should be > 0.05)")
if stable_frac > 0.05:
    print("  Stability sanity:  PASS ✓  (adaptive dt resolves orbits correctly)")
else:
    print("  Stability sanity:  FAIL ✗  (all cells showing unstable — dt issue?)")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 1: CORRECTNESS — batch vs Numba at the SAME adaptive dt
# ══════════════════════════════════════════════════════════════════════════════

print()
print("=" * 72)
print("CHECK 1: Correctness — batch leapfrog vs Numba (same adaptive dt)")
print("=" * 72)

# Pick a reference cell at am_hill ≈ 0.40 — should be in the stable regime
REF_AM_HILL = 0.40
REF_MM_EARTH = float(mm_grid[FULL_MM_RES // 2])   # middle mm row

i_mm = int(np.argmin(np.abs(mm_grid - REF_MM_EARTH)))
i_am = int(np.argmin(np.abs(am_grid - REF_AM_HILL)))
k    = i_mm * FULL_AM_RES + i_am

mm_actual = float(mm_grid[i_mm])
am_actual = float(am_grid[i_am])
print(f"\n  Reference: mm_earth={REF_MM_EARTH:.4f}, am_hill={REF_AM_HILL:.3f}")
print(f"  Grid cell: mm={mm_actual:.4f} M⊕  am={am_actual:.4f} Hill  (k={k})")
print(f"  Numba runs at same dt_phys={dt_phys:.5f} yr  ({n_phys} steps)")

p_ref = SystemParams(
    Ts=SYSTEM_PARAMS["Ts"], rs_solar=SYSTEM_PARAMS["rs_solar"],
    ms_solar=SYSTEM_PARAMS["ms_solar"], mp_earth=SYSTEM_PARAMS["mp_earth"],
    ap_AU=SYSTEM_PARAMS["ap_AU"], ep=SYSTEM_PARAMS["ep"],
    mm_earth=mm_actual, am_hill=am_actual,
    em=0.0, moon_retrograde=False,
)
st_ref = initial_state(p_ref)

t_numba_start = time.perf_counter()
traj_ref = leapfrog_integrate(st_ref, t_end=T_SIM, dt=dt_phys)
t_numba_end = time.perf_counter()

moon_numba = traj_ref["xyzarr_mm"][-1]                        # (3,) last step
moon_batch = result["traj_moon"][k, -1].astype(np.float64)    # last stored frame

pos_err      = np.linalg.norm(moon_batch - moon_numba)
pos_err_frac = pos_err / rhill_AU

# mpd at reference cell
mpd_ref_cell = result["moon_planet_dist"][k]   # (n_out,) float32
print(f"\n  Final moon position:")
print(f"    Numba:  {moon_numba}")
print(f"    Batch:  {moon_batch}")
print(f"    |Δ|   = {pos_err:.2e} AU  ({pos_err_frac*100:.4f}% of Hill radius)")
print(f"  mpd at ref cell — max: {mpd_ref_cell.max():.4f} AU  (rhill={rhill_AU:.4f} AU)")
print(f"    stable? {bool(mpd_ref_cell.max() <= rhill_AU)}")

PASS_THRESH = 0.001   # 0.1% of Hill radius
status1 = "PASS ✓" if pos_err_frac < PASS_THRESH else "FAIL ✗"
print(f"\n  Threshold: {PASS_THRESH*100:.1f}% of Hill radius  →  {status1}")
print(f"  Numba single-cell time: {(t_numba_end - t_numba_start)*1000:.1f} ms")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 3: COVERAGE vs GROUND TRUTH
# ══════════════════════════════════════════════════════════════════════════════

print()
print("=" * 72)
print(f"CHECK 3: Coverage vs ground-truth .npz ({MM_RESOLUTION}×{AM_RESOLUTION} grid)")
print("=" * 72)

gt_path = os.path.join(SRC, "ground_truth_grids", "Kepler_452_b_v2_prograde.npz")
if not os.path.exists(gt_path):
    print(f"  Ground-truth grid not found: {gt_path}")
    print("  Skipping CHECK 3.")
else:
    # Ground truth uses GT resolution; batch result uses FULL resolution.
    # Re-run at GT resolution for an apples-to-apples map comparison.
    print(f"\n  Re-running batch at GT resolution {MM_RESOLUTION}×{AM_RESOLUTION}...")
    gt_result = batch_leapfrog_trajectories(
        system_params=SYSTEM_PARAMS,
        t_sim=T_SIM,
        moon_retrograde=False,
        em=0.0,
        mm_resolution=MM_RESOLUTION,
        am_resolution=AM_RESOLUTION,
        n_steps=1000,
        escape_factor=1.0,
    )
    print(f"  GT-res run: n_phys={gt_result['n_phys']}  dt={gt_result['dt_phys']:.5f} yr"
          f"  elapsed={gt_result['elapsed_s']:.2f}s")

    npz  = np.load(gt_path, allow_pickle=True)
    gt_s = npz["map_stable"].astype(bool)
    gt_h = npz["map_habitable"].astype(bool)
    gt_b = gt_s & gt_h

    ml_s = np.array(gt_result["map_stable"])
    ml_h = np.array(gt_result["map_habitable"])
    ml_b = ml_s & ml_h

    mm_grid_gt = np.array(gt_result["mm_grid"])
    am_grid_gt = np.array(gt_result["am_grid"])

    print(f"\n  Grid fractions:")
    print(f"    GT  stable={gt_s.mean():.3f}  hab={gt_h.mean():.3f}  both={gt_b.mean():.3f}")
    print(f"    Batch stable={ml_s.mean():.3f}  hab={ml_h.mean():.3f}  both={ml_b.mean():.3f}")

    print(f"\n  Stable map (vs GT stable):")
    rec, prec, f1 = pr_f1(gt_s, ml_s)
    print(f"    recall={fmt(rec)}  prec={fmt(prec)}  F1={fmt(f1)}")

    print(f"\n  Stable+habitable map (vs GT both):")
    rec, prec, f1 = pr_f1(gt_b, ml_b)
    print(f"    recall={fmt(rec)}  prec={fmt(prec)}  F1={fmt(f1)}")

    # Per-row stable+habitable count
    print(f"\n  Per-row (mm_earth) stable+habitable count [batch | GT]:")
    for i in range(MM_RESOLUTION):
        b_count  = int(ml_b[i].sum())
        gt_count = int(gt_b[i].sum())
        bar_b    = "█" * b_count
        bar_gt   = "▒" * gt_count
        if b_count > 0 or gt_count > 0:
            print(f"    mm={mm_grid_gt[i]:.3f} M⊕: batch={b_count:2d} [{bar_b:<{AM_RESOLUTION}}]"
                  f"  GT={gt_count:2d} [{bar_gt}]")

print()
print("Done.")
