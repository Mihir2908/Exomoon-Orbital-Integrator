"""
eval_force_mlp_trajectory.py — Trajectory-level comparison:
  Ground-truth Numba leapfrog  vs  HNN v12  vs  Force regression MLP

For each system in ground_truth_grids/ and a small set of representative
(mm_earth, am_hill) cells, runs all three integrators and reports:
  - moon_planet_dist and moon_star_dist at 20 evenly-spaced time points
  - Mean absolute error on mpd vs ground truth (AU and % of rhill)
  - Stability classification agreement (stable / escaped)

Representative cells chosen as 3x3 = 9 grid indices:
  mm_idx in {12, 25, 37}  x  am_idx in {12, 25, 37}
  (low/mid/high thirds of each axis)

HNN runs all 2500 cells in one batched pass; individual cells are extracted.
Force MLP runs one cell at a time via preview_trajectory().

ISOLATION: does not touch models/, models_temphead/, eval_aux_mlp_output/.
"""

import os, sys, json, time
import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from exomoon.ml.hnn_inference        import batch_hnn_trajectories
from exomoon.ml.force_mlp_inference  import preview_trajectory
from exomoon.params                  import SystemParams
from exomoon.simulation              import run_simulation_for_years

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--hnn_dir",  default="models_hnn_dist_v12")
ap.add_argument("--fmlp_dir", default="models_force_mlp")
ap.add_argument("--systems",  default=None,
                help="Comma-separated list of system labels (substring match). "
                     "E.g. 'Kepler_452_b_v2,Kepler_1229'. Default: all prograde gates.")
ap.add_argument("--n_out",    type=int, default=200,
                help="Output time points for force MLP preview (default 200)")
args = ap.parse_args()

GT_DIR   = os.path.join(SRC, "ground_truth_grids")
HNN_DIR  = os.path.join(SRC, args.hnn_dir)
FMLP_DIR = os.path.join(SRC, args.fmlp_dir)

MM_RES = AM_RES = 50
CELL_INDICES = [(12, 12), (12, 25), (12, 37),
                (25, 12), (25, 25), (25, 37),
                (37, 12), (37, 25), (37, 37)]   # (mm_idx, am_idx)

N_PRINT = 10   # evenly-spaced time points printed in detail table

SEP  = "=" * 100
DSEP = "-" * 100


# ── Collect meta files ────────────────────────────────────────────────────────

all_meta = sorted(
    os.path.join(GT_DIR, f)
    for f in os.listdir(GT_DIR)
    if f.endswith(".meta.json") and "prograde" in f
)

if args.systems:
    filters = [s.strip() for s in args.systems.split(",")]
    all_meta = [m for m in all_meta
                if any(flt in os.path.basename(m) for flt in filters)]

print(f"HNN  dir : {args.hnn_dir}")
print(f"FMLP dir : {args.fmlp_dir}")
print(f"Systems  : {len(all_meta)}")
print()


# ── Helpers ───────────────────────────────────────────────────────────────────

def resample(arr, n):
    """Resample 1-D array to n evenly-spaced indices."""
    idx = np.round(np.linspace(0, len(arr) - 1, n)).astype(int)
    return arr[idx]


def stability_label(mpd, rhill):
    return "STABLE " if np.max(mpd) <= rhill else "ESCAPED"


def pct_within(err_pct_arr, thresh):
    return (np.abs(err_pct_arr) < thresh).mean() * 100


# ── Per-system loop ───────────────────────────────────────────────────────────

for meta_path in all_meta:
    with open(meta_path) as f:
        meta = json.load(f)

    sp_raw      = meta["system_params"]
    t_sim       = float(meta["t_sim"])
    em          = float(meta["em"])
    retro       = bool(meta["moon_retrograde"])
    sys_label   = os.path.basename(meta_path).replace("_prograde.meta.json", "")

    system_params = {k: float(sp_raw[k])
                     for k in ("ms_solar", "rs_solar", "Ts", "mp_earth", "ap_AU")}
    system_params["ep"] = float(sp_raw.get("ep", 0.0))

    print(SEP)
    print(f"  SYSTEM : {sys_label}   t_sim={t_sim} yr   retro={retro}   em={em}")
    print(SEP)

    # ── Step 1: HNN batch run (all 2500 cells, extract needed) ───────────────
    print(f"\n  [HNN]  Running batch inference ({MM_RES}x{AM_RES} cells)...", flush=True)
    t0 = time.perf_counter()
    hnn_res = batch_hnn_trajectories(
        system_params = system_params,
        t_sim         = t_sim,
        mm_resolution = MM_RES,
        am_resolution = AM_RES,
        n_steps       = args.n_out,
        model_dir     = HNN_DIR,
        dt_factor     = 50,
    )
    hnn_elapsed = time.perf_counter() - t0
    if not hnn_res["ok"]:
        print(f"  HNN failed: {hnn_res.get('error')}")
        continue

    mm_grid  = np.array(hnn_res["mm_grid"])
    am_grid  = np.array(hnn_res["am_grid"])
    rhill_AU = float(hnn_res["rhill_AU"])
    t_hnn    = np.array(hnn_res["t_grid"])           # (n_out,)
    mpd_hnn  = np.array(hnn_res["moon_planet_dist"]) # (N_cells, n_out)
    msd_hnn  = np.array(hnn_res["moon_star_dist"])   # (N_cells, n_out)
    print(f"         Done in {hnn_elapsed:.1f}s  |  rhill={rhill_AU:.5f} AU")

    # ── Per-cell comparison ───────────────────────────────────────────────────
    for mm_idx, am_idx in CELL_INDICES:
        mm_earth = float(mm_grid[mm_idx])
        am_hill  = float(am_grid[am_idx])
        am_AU    = am_hill * rhill_AU
        cell_flat = mm_idx * AM_RES + am_idx

        print()
        print(f"  Cell ({mm_idx:>2},{am_idx:>2})  mm={mm_earth:.4f} Mearth  "
              f"am={am_hill:.3f} Hill  am_AU={am_AU:.5f} AU")

        # ── Ground truth (Numba leapfrog) ─────────────────────────────────────
        p_gt = SystemParams(
            ms_solar       = system_params["ms_solar"],
            rs_solar       = system_params.get("rs_solar", 1.0),
            Ts             = system_params.get("Ts", 5772.0),
            mp_earth       = system_params["mp_earth"],
            dp_cgs         = 5.5,
            ap_AU          = system_params["ap_AU"],
            ep             = system_params.get("ep", 0.0),
            mm_earth       = mm_earth,
            am_hill        = am_hill,
            em             = em,
            moon_retrograde= retro,
        )
        t0 = time.perf_counter()
        sim_gt = run_simulation_for_years(p_gt, t_sim)
        gt_elapsed = time.perf_counter() - t0

        traj      = sim_gt["traj"]
        mp_pos    = traj["xyzarr_mp"]
        mm_pos    = traj["xyzarr_mm"]
        ms_pos    = traj["xyzarr_ms"]
        gt_mpd_full = np.linalg.norm(mm_pos - mp_pos, axis=1)
        gt_msd_full = np.linalg.norm(mm_pos - ms_pos, axis=1)

        gt_mpd = resample(gt_mpd_full, args.n_out)
        gt_msd = resample(gt_msd_full, args.n_out)
        t_gt   = np.linspace(0.0, t_sim, args.n_out)

        # ── Force MLP single-cell run ─────────────────────────────────────────
        sp_full = dict(**system_params,
                       mm_earth       = mm_earth,
                       am_hill        = am_hill,
                       em             = em,
                       moon_retrograde= retro,
                       dp_cgs         = 5.5)
        t0 = time.perf_counter()
        fmlp_res = preview_trajectory(
            system_params   = sp_full,
            t_sim           = t_sim,
            model_dir       = FMLP_DIR,
            n_out           = args.n_out,
        )
        fmlp_elapsed = time.perf_counter() - t0

        if not fmlp_res["ok"]:
            print(f"    Force MLP failed: {fmlp_res.get('error')}")
            continue

        fmlp_mpd = np.array(fmlp_res["moon_planet_dist"])
        fmlp_msd = np.array(fmlp_res["moon_star_dist"])
        fmlp_t   = np.array(fmlp_res["t_arr"])

        # Resample force MLP output to match n_out grid (it may be shorter if stopped early)
        if len(fmlp_mpd) < args.n_out:
            fmlp_mpd_rs = np.full(args.n_out, fmlp_mpd[-1])
            fmlp_msd_rs = np.full(args.n_out, fmlp_msd[-1])
            fmlp_mpd_rs[:len(fmlp_mpd)] = fmlp_mpd
            fmlp_msd_rs[:len(fmlp_msd)] = fmlp_msd
        else:
            fmlp_mpd_rs = resample(fmlp_mpd, args.n_out)
            fmlp_msd_rs = resample(fmlp_msd, args.n_out)

        # Extract HNN cell
        hnn_mpd_cell = mpd_hnn[cell_flat]   # (n_out,)
        hnn_msd_cell = msd_hnn[cell_flat]

        # ── Errors vs ground truth ────────────────────────────────────────────
        hnn_mpd_err  = (hnn_mpd_cell  - gt_mpd) / (gt_mpd + 1e-12)
        fmlp_mpd_err = (fmlp_mpd_rs   - gt_mpd) / (gt_mpd + 1e-12)
        hnn_msd_err  = (hnn_msd_cell  - gt_msd) / (gt_msd + 1e-12)
        fmlp_msd_err = (fmlp_msd_rs   - gt_msd) / (gt_msd + 1e-12)

        # ── Detail table (N_PRINT evenly-spaced points) ───────────────────────
        print(f"    {'t_yr':>6}  {'GT mpd':>10}  {'HNN mpd':>10}  "
              f"{'HNNerr%':>8}  {'FMLP mpd':>10}  {'FMLPerr%':>8}"
              f"  {'GT msd':>8}  {'HNN msd':>8}  {'FMLP msd':>8}")
        print("    " + "-" * 90)
        print_idx = np.round(np.linspace(0, args.n_out - 1, N_PRINT)).astype(int)
        for i in print_idx:
            t_yr   = t_gt[i]
            gt_m   = gt_mpd[i]
            h_m    = hnn_mpd_cell[i]
            f_m    = fmlp_mpd_rs[i]
            h_err  = hnn_mpd_err[i]  * 100
            f_err  = fmlp_mpd_err[i] * 100
            gt_s   = gt_msd[i]
            h_s    = hnn_msd_cell[i]
            f_s    = fmlp_msd_rs[i]
            print(f"    {t_yr:>6.2f}  {gt_m:>10.6f}  {h_m:>10.6f}  "
                  f"{h_err:>+7.1f}%  {f_m:>10.6f}  {f_err:>+7.1f}%"
                  f"  {gt_s:>8.4f}  {h_s:>8.4f}  {f_s:>8.4f}")

        # ── Summary ───────────────────────────────────────────────────────────
        def _summary(err):
            return (f"mean|e|={np.abs(err).mean()*100:.1f}%  "
                    f"max|e|={np.abs(err).max()*100:.1f}%  "
                    f"<15%={pct_within(err*100,15):.0f}%  "
                    f"<30%={pct_within(err*100,30):.0f}%")

        gt_stab   = stability_label(gt_mpd_full, rhill_AU)
        hnn_stab  = stability_label(hnn_mpd_cell, rhill_AU)
        fmlp_stab = stability_label(fmlp_mpd_rs,  rhill_AU)

        print()
        print(f"    mpd summary  HNN : {_summary(hnn_mpd_err)}")
        print(f"    mpd summary FMLP : {_summary(fmlp_mpd_err)}")
        print(f"    msd summary  HNN : {_summary(hnn_msd_err)}")
        print(f"    msd summary FMLP : {_summary(fmlp_msd_err)}")
        print(f"    Stability  GT={gt_stab}  HNN={hnn_stab}  "
              f"FMLP={fmlp_stab}  (rhill={rhill_AU:.5f})")
        print(f"    Timing     GT={gt_elapsed:.1f}s  FMLP={fmlp_elapsed:.2f}s  "
              f"HNN={hnn_elapsed/len(CELL_INDICES):.2f}s/cell (amortised)")

    print()
