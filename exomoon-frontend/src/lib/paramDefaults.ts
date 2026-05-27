import type { SystemParams } from './types';

// Mirrors exomoon/params.py SystemParams defaults
export const DEFAULT_PARAMS: SystemParams = {
  Ts: 5772,
  rs_solar: 1.0,
  ms_solar: 1.0,
  mp_earth: 1.0,
  dp_cgs: 5.51,
  ap_AU: 1.0,
  ep: 0.0,
  mm_earth: 0.01,
  am_hill: 0.3,
  em: 0.0,
  moon_retrograde: false,
};

// Slider ranges matching run_dash.py control definitions
export const PARAM_RANGES = {
  Ts:        { min: 2000,  max: 20000, step: 1,      unit: 'K' },
  rs_solar:  { min: 0.05,  max: 30.0,  step: 0.01,   unit: 'R☉' },
  ms_solar:  { min: 0.05,  max: 50.0,  step: 0.01,   unit: 'M☉' },
  mp_earth:  { min: 0.01,  max: 10.0,  step: 0.01,   unit: 'M⊕' },
  dp_cgs:    { min: 0.2,   max: 40.0,  step: 0.01,   unit: 'g/cc' },
  ap_AU:     { min: 0.005, max: 10.0,  step: 0.005,  unit: 'AU' },
  ep:        { min: 0.0,   max: 0.99,  step: 0.001,  unit: '' },
  mm_earth:  { min: 0.01,  max: 10.0,  step: 0.01,   unit: 'M⊕' },
  am_hill:   { min: 0.01,  max: 0.95,  step: 0.01,   unit: '× Rhill' },
  em:        { min: 0.0,   max: 1.0,   step: 0.001,  unit: '' },
} as const;

export const PARAM_LABELS: Record<keyof Omit<SystemParams, 'moon_retrograde'>, string> = {
  Ts:       'Stellar temperature Ts',
  rs_solar: 'Star radius',
  ms_solar: 'Star mass',
  mp_earth: 'Planet mass',
  dp_cgs:   'Planet density',
  ap_AU:    'Planet semi-major axis ap',
  ep:       'Planet eccentricity ep',
  mm_earth: 'Moon mass',
  am_hill:  'Moon a (fraction of Hill radius)',
  em:       'Moon eccentricity em',
};

// Mirrors exoplanet_archive.py estimate_density_gcc
export function estimateDensityGcc(mpEarth: number | null, radeEarth: number | null): number | null {
  if (!mpEarth || !radeEarth || radeEarth <= 0) return null;
  // rho = 5.514 * (mp / rade^3) in g/cc (Earth density = 5.514 g/cc)
  return 5.514 * (mpEarth / Math.pow(radeEarth, 3));
}
