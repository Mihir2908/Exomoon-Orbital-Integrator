// JS equivalents of Python physics utilities

export const STEFAN_BOLTZMANN = 5.6696e-8;  // W m^-2 K^-4  (constants.py)
const F_EARTH = 1370;                        // W/m^2
export const AU_M   = 1.495979e11;           // m per AU  (constants.py)
export const RSUN_M = 6.9598e8;              // m per solar radius  (constants.py)
const MERTH_KG = 5.976e24;                   // kg  (constants.py merth)
const RERTH_M  = 6.378e6;                    // m  (constants.py rerth)
const MSUN_KG  = 1.989e30;                   // kg  (constants.py msun)

// ── Body radius (mirrors initial_conditions.py formula) ───────────────────────
// r_m = (0.75 * mass_kg / density_kgm3)^(1/3)
export function bodyRadiusM(massEarth: number, densityCgs: number): number {
  return Math.cbrt(0.75 * massEarth * MERTH_KG / (densityCgs * 1e3));
}
export function bodyRadiusAU(massEarth: number, densityCgs: number): number {
  return bodyRadiusM(massEarth, densityCgs) / AU_M;
}
export function bodyRadiusEarth(massEarth: number, densityCgs: number): number {
  return bodyRadiusM(massEarth, densityCgs) / RERTH_M;
}
// Inverse: density (g/cm³) given radius (Earth radii) and mass (Earth masses)
export function densityFromRadiusEarth(massEarth: number, radiusEarth: number): number {
  const r_m = radiusEarth * RERTH_M;
  return 0.75 * massEarth * MERTH_KG / (r_m ** 3 * 1e3);
}

// ── Moon effective temperature ────────────────────────────────────────────────
// Tm = ((0.7 * F) / (2.448 * σ))^0.25   where F = L_star / (4π r²)
export function moonEffectiveTempK(Ts_K: number, rs_solar: number, moonStarDistAU: number): number {
  if (moonStarDistAU <= 0) return 0;
  const rs_m = rs_solar * RSUN_M;
  const L = 4 * Math.PI * rs_m * rs_m * STEFAN_BOLTZMANN * Ts_K ** 4;
  const r_m = moonStarDistAU * AU_M;
  const F = L / (4 * Math.PI * r_m * r_m);
  return ((0.7 * F) / (2.448 * STEFAN_BOLTZMANN)) ** 0.25;
}

// ── Kepler's 3rd law: T² = a³ / (m1+m2)  [AU / yr / M_sun] ──────────────────
export function orbitalPeriodYr(semiMajorAxisAU: number, totalMassSolar: number): number {
  if (semiMajorAxisAU <= 0 || totalMassSolar <= 0) return 0;
  return Math.sqrt(semiMajorAxisAU ** 3 / totalMassSolar);
}
export const EARTH_MASS_SOLAR = MERTH_KG / MSUN_KG; // ≈ 3.003e-6

// Mirrors habitable_zone.py hz_bounds_au()
export function hzBoundsAu(Ts_K: number, rs_solar: number): { inner: number; outer: number } {
  const rs_m = rs_solar * RSUN_M;
  const L_star = 4 * Math.PI * rs_m * rs_m * STEFAN_BOLTZMANN * Math.pow(Ts_K, 4);
  const a_inner_m = Math.sqrt(L_star / (4 * Math.PI * 1.1 * F_EARTH));
  const a_outer_m = Math.sqrt(L_star / (4 * Math.PI * 0.5 * F_EARTH));
  return { inner: a_inner_m / AU_M, outer: a_outer_m / AU_M };
}

// Mirrors eda.py var_info() lookup
export interface VarInfo { label: string; unit: string }

export const VAR_INFO: Record<string, VarInfo> = {
  star_x:           { label: 'Star X position',      unit: 'AU' },
  star_y:           { label: 'Star Y position',      unit: 'AU' },
  star_z:           { label: 'Star Z position',      unit: 'AU' },
  planet_x:         { label: 'Planet X position',    unit: 'AU' },
  planet_y:         { label: 'Planet Y position',    unit: 'AU' },
  planet_z:         { label: 'Planet Z position',    unit: 'AU' },
  moon_x:           { label: 'Moon X position',      unit: 'AU' },
  moon_y:           { label: 'Moon Y position',      unit: 'AU' },
  moon_z:           { label: 'Moon Z position',      unit: 'AU' },
  star_vx:          { label: 'Star Vx',              unit: 'AU/yr' },
  star_vy:          { label: 'Star Vy',              unit: 'AU/yr' },
  star_vz:          { label: 'Star Vz',              unit: 'AU/yr' },
  planet_vx:        { label: 'Planet Vx',            unit: 'AU/yr' },
  planet_vy:        { label: 'Planet Vy',            unit: 'AU/yr' },
  planet_vz:        { label: 'Planet Vz',            unit: 'AU/yr' },
  moon_vx:          { label: 'Moon Vx',              unit: 'AU/yr' },
  moon_vy:          { label: 'Moon Vy',              unit: 'AU/yr' },
  moon_vz:          { label: 'Moon Vz',              unit: 'AU/yr' },
  moon_planet_dist: { label: 'Moon–Planet distance', unit: 'AU' },
  planet_star_dist: { label: 'Planet–Star distance', unit: 'AU' },
  moon_star_dist:   { label: 'Moon–Star distance',   unit: 'AU' },
  moon_speed:       { label: 'Moon speed',           unit: 'AU/yr' },
  planet_speed:     { label: 'Planet speed',         unit: 'AU/yr' },
  star_speed:       { label: 'Star speed',           unit: 'AU/yr' },
  moon_rel_x:       { label: 'Moon ΔX (rel. planet)', unit: 'AU' },
  moon_rel_y:       { label: 'Moon ΔY (rel. planet)', unit: 'AU' },
};

export function varInfo(col: string): VarInfo {
  return VAR_INFO[col] ?? { label: col, unit: '' };
}

// All EDA-plottable columns (excludes t_years and rel_ columns)
export const EDA_COLUMNS = [
  'moon_planet_dist', 'planet_star_dist', 'moon_star_dist',
  'moon_speed', 'planet_speed', 'star_speed',
  'moon_x', 'moon_y', 'moon_z',
  'planet_x', 'planet_y', 'planet_z',
  'star_x', 'star_y', 'star_z',
  'moon_vx', 'moon_vy', 'moon_vz',
  'planet_vx', 'planet_vy', 'planet_vz',
  'star_vx', 'star_vy', 'star_vz',
] as const;

export const DEFAULT_EDA_VARS = ['moon_planet_dist', 'planet_star_dist', 'moon_speed'];
