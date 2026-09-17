"""
OPTION D BACKUP — extracted from batch_leapfrog.py 2026-09-12.

Option D used a per-cell timestep: dt_i = T_moon_i / K_ORBIT.
K=10 gave GT stable=0.000 on ALL systems.
K=16 gave GT stable=0.005 on K-1229b (expected 0.484).
CATEGORICALLY RULED OUT — GT physics breaks at these coarse timesteps
for fast inner-moon cells in compact systems (K-1229b, TRAPPIST-1e).

DO NOT USE. Preserved here only for reference.
"""

# ── Option D block that was in batch_leapfrog.py lines 251-283 ─────────────
#
#    K_ORBIT = 16
#    if n_orbits is not None:
#        am_ref_AU  = float(am_grid[am_resolution // 2]) * rhill_AU
#        mm_mid     = float(mm_grid[mm_resolution // 2])
#        mu_mm_ref  = mm_mid * (merth / msun) * FOUR_PI2
#        T_moon_ref = 2.0 * np.pi * np.sqrt(am_ref_AU**3 / (mu_mp + mu_mm_ref))
#        t_sim      = float(n_orbits) * T_moon_ref
#
#    # cell k = mm_idx*am_res + am_idx → mm_grid[mm_idx], am_grid[am_idx]
#    am_per_cell    = np.tile(am_grid, mm_resolution)           # (N,) Hill-fraction
#    am_AU_per_cell = am_per_cell * rhill_AU                    # (N,) [AU]
#    T_moon_cell    = 2.0 * np.pi * np.sqrt(
#        am_AU_per_cell**3 / (mu_mp + mu_mm_arr)
#    )                                                           # (N,) orbital period [yr]
#    dt_arr         = T_moon_cell / K_ORBIT                     # (N,) per-cell dt [yr]
#    dt_min_val     = float(dt_arr.min())                       # innermost cell — sets loop count
#    n_phys         = max(int(np.ceil(t_sim / dt_min_val)), n_steps)
#    stride         = max(1, n_phys // n_steps)
#    n_out          = n_phys // stride
#
#    dt_t      = torch.tensor(dt_arr[:, None], dtype=dtype, device=dev)   # (N,1)
#    half_dt_t = dt_t * 0.5                                                # (N,1)
#    print(f"  GT batch leapfrog: per-cell k={K_ORBIT}  "
#          f"dt_min={dt_min_val:.2e} yr  dt_max={dt_arr.max():.2e} yr  "
#          f"n_phys={n_phys:,}  N={N}")
#
# ── Benchmark results proving Option D is invalid ──────────────────────────
#
#  Task bqlowylmd (K=10): GT stable=0.000 on K-452b_v2 AND K-1229b
#  Task b8vq7k5z8 (K=16): GT stable=0.005 on K-1229b (expected ~0.484)
#  Pre-Option-D fixed dt=5e-5: GT stable=0.484 on K-1229b ✓
