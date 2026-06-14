export interface SystemParams {
  Ts: number;           // Stellar temperature (K)
  rs_solar: number;     // Star radius (solar radii)
  ms_solar: number;     // Star mass (solar masses)
  mp_earth: number;     // Planet mass (Earth masses)
  dp_cgs: number;       // Planet density (g/cc)
  ap_AU: number;        // Planet semi-major axis (AU)
  ep: number;           // Planet eccentricity
  mm_earth: number;     // Moon mass (Earth masses)
  am_hill: number;      // Moon semi-major axis (fraction of Hill radius)
  em: number;           // Moon eccentricity
  moon_retrograde: boolean;
}

export interface TrajectoryFrame {
  t_years: number;
  star_x: number; star_y: number; star_z: number;
  planet_x: number; planet_y: number; planet_z: number;
  moon_x: number; moon_y: number; moon_z: number;
  star_vx: number; star_vy: number; star_vz: number;
  planet_vx: number; planet_vy: number; planet_vz: number;
  moon_vx: number; moon_vy: number; moon_vz: number;
  moon_planet_dist: number;
  planet_star_dist: number;
  moon_star_dist?:  number;
  moon_speed: number;
  planet_speed: number;
  star_speed: number;
  moon_rel_x?: number; moon_rel_y?: number; moon_rel_z?: number;
  planet_rel_x?: number; planet_rel_y?: number; planet_rel_z?: number;
}

export interface SimulationMeta {
  dt: number;
  t_end: number;
  a_inner_au: number;
  a_outer_au: number;
  rhill_AU: number | null;
}

export interface JobStatus {
  ok: boolean;
  job_id: string;
  status: 'RUNNING' | 'SUCCEEDED' | 'FAILED' | 'TIMED_OUT' | 'UNKNOWN';
  elapsed_seconds: number;
  urls: Record<string, string>;
  error?: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  streaming?: boolean;
  jobId?: string;
  timestamp: number;
}

export interface ChatRequest {
  message: string;
  simdata?: string | null;
  params: Record<string, unknown>;
  years?: number | null;
  escape_factor: number;
}

export interface AgentHealthResponse {
  ok: boolean;
  service: string;
  aws_enabled: boolean;
  claude_enabled: boolean;
}

export type JobStatusState = 'idle' | 'running' | 'succeeded' | 'failed';

export interface MlPrediction {
  mmGrid: number[];                          // mm_earth values [M_earth], log-spaced
  amGrid: number[];                          // am_hill values [Hill radii], linear-spaced
  mapStable: boolean[][];                    // [mm_resolution][am_resolution] — stable throughout
  mapHabitable: boolean[][];                 // [mm_resolution][am_resolution] — habitable throughout
  mapBoth: boolean[][];                      // [mm_resolution][am_resolution] — stable AND habitable
  validMmRange: [number, number] | null;     // [min_mm, max_mm] in M_earth, or null if none
  validAmPerMm: ([number, number] | null)[]; // per-mm valid [min_am, max_am] in Hill radii
}
