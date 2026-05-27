import type { TrajectoryFrame, SimulationMeta } from './types';

const MAX_FRAMES = 10000;

export interface ParseResult {
  frames: TrajectoryFrame[];
  meta: SimulationMeta;
}

// Parse traj.csv text into typed TrajectoryFrame array.
// Downsamples to MAX_FRAMES if needed (matching Python np.linspace approach).
export function parseTrajectoryCsv(
  csvText: string,
  summaryJson?: Record<string, unknown>
): ParseResult {
  const lines = csvText.trim().split('\n');
  if (lines.length < 2) {
    return { frames: [], meta: buildMeta([], summaryJson) };
  }

  const headers = lines[0].split(',').map(h => h.trim());
  const colIdx = Object.fromEntries(headers.map((h, i) => [h, i]));

  const allRows: TrajectoryFrame[] = [];
  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(',');
    if (cols.length < headers.length) continue;
    const g = (col: string): number => parseFloat(cols[colIdx[col]] ?? '0') || 0;

    allRows.push({
      t_years:          g('t_years'),
      star_x:           g('star_x'),   star_y:    g('star_y'),   star_z:    g('star_z'),
      planet_x:         g('planet_x'), planet_y:  g('planet_y'), planet_z:  g('planet_z'),
      moon_x:           g('moon_x'),   moon_y:    g('moon_y'),   moon_z:    g('moon_z'),
      star_vx:          g('star_vx'),  star_vy:   g('star_vy'),  star_vz:   g('star_vz'),
      planet_vx:        g('planet_vx'),planet_vy: g('planet_vy'),planet_vz: g('planet_vz'),
      moon_vx:          g('moon_vx'),  moon_vy:   g('moon_vy'),  moon_vz:   g('moon_vz'),
      moon_planet_dist: g('moon_planet_dist'),
      planet_star_dist: g('planet_star_dist'),
      moon_speed:       g('moon_speed'),
      planet_speed:     g('planet_speed'),
      star_speed:       g('star_speed'),
      moon_rel_x:       colIdx['moon_rel_x'] !== undefined ? g('moon_rel_x') : undefined,
      moon_rel_y:       colIdx['moon_rel_y'] !== undefined ? g('moon_rel_y') : undefined,
      moon_rel_z:       colIdx['moon_rel_z'] !== undefined ? g('moon_rel_z') : undefined,
    });
  }

  // Downsample evenly if over MAX_FRAMES
  const frames = downsample(allRows, MAX_FRAMES);
  return { frames, meta: buildMeta(frames, summaryJson) };
}

function downsample<T>(arr: T[], maxLen: number): T[] {
  if (arr.length <= maxLen) return arr;
  const result: T[] = [];
  const step = (arr.length - 1) / (maxLen - 1);
  for (let i = 0; i < maxLen; i++) {
    result.push(arr[Math.round(i * step)]);
  }
  return result;
}

function buildMeta(frames: TrajectoryFrame[], summary?: Record<string, unknown>): SimulationMeta {
  const dt = frames.length >= 2 ? frames[1].t_years - frames[0].t_years : 0.001;
  const t_end = frames.length > 0 ? frames[frames.length - 1].t_years : 0;
  return {
    dt:         (summary?.dt as number)          ?? dt,
    t_end:      (summary?.t_end as number)        ?? t_end,
    a_inner_au: (summary?.a_inner_au as number)   ?? 0.75,
    a_outer_au: (summary?.a_outer_au as number)   ?? 1.75,
    rhill_AU:   (summary?.rhill_AU as number)     ?? null,
  };
}
