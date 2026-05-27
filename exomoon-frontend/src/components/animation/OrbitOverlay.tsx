'use client';
import React from 'react';
import { useSimulationStore } from '@/hooks/useSimulationStore';
import {
  moonEffectiveTempK, orbitalPeriodYr, EARTH_MASS_SOLAR,
} from '@/lib/trajectoryMath';
import type { TrajectoryFrame, SimulationMeta } from '@/lib/types';

interface OrbitOverlayProps {
  frame: TrajectoryFrame | null;
  frameIndex: number;
  totalFrames: number;
  meta: SimulationMeta | null;
}

function fmt(v: number, d = 4) { return v.toFixed(d); }

export function OrbitOverlay({ frame, frameIndex, totalFrames, meta }: OrbitOverlayProps) {
  const { params } = useSimulationStore();
  if (!frame || !meta) return null;

  const moonPlanetDist = frame.moon_planet_dist ?? 0;
  const planetStarDist = frame.planet_star_dist ?? 0;
  const moonSpeed      = frame.moon_speed ?? 0;
  const planetSpeed    = frame.planet_speed ?? 0;
  const rhill          = meta.rhill_AU ?? 0;
  const simTime        = frame.t_years ?? 0;

  // Moon–star distance (not stored directly in frame, computed from positions)
  const dx = frame.moon_x - frame.star_x;
  const dy = frame.moon_y - frame.star_y;
  const dz = frame.moon_z - frame.star_z;
  const moonStarDist = Math.sqrt(dx * dx + dy * dy + dz * dz);

  // Stability
  const escaped  = rhill > 0 && moonPlanetDist > rhill;
  const hillFrac = rhill > 0 ? moonPlanetDist / rhill : 0;

  // Habitability: moon within HZ band?
  const inHZ = moonStarDist >= meta.a_inner_au && moonStarDist <= meta.a_outer_au;

  // Moon effective surface temperature
  const moonTempK = moonEffectiveTempK(params.Ts, params.rs_solar, moonStarDist);

  // Orbital periods (Kepler's 3rd law in AU/yr/M_sun)
  const mpSolar   = params.mp_earth * EARTH_MASS_SOLAR;
  const mmSolar   = params.mm_earth * EARTH_MASS_SOLAR;
  const aMoonAU   = params.am_hill * (rhill || 0.01);  // moon semi-major axis (AU)
  const T_planet  = orbitalPeriodYr(params.ap_AU, params.ms_solar + mpSolar);
  const T_moon    = orbitalPeriodYr(aMoonAU, mpSolar + mmSolar);

  // Orbit counters (how many complete orbits in elapsed sim time)
  const planetOrbits = T_planet > 0 ? Math.floor(simTime / T_planet) : 0;
  const moonOrbits   = T_moon   > 0 ? Math.floor(simTime / T_moon)   : 0;

  return (
    <div className="absolute top-3 left-3 pointer-events-none space-y-1.5">
      {/* Stability badge */}
      <div className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-xs font-semibold tracking-wide ${
        escaped
          ? 'bg-red-900/70 text-red-300 border border-red-700/50'
          : 'bg-green-900/70 text-green-300 border border-green-700/50'
      }`}>
        <span className={`w-1.5 h-1.5 rounded-full ${escaped ? 'bg-red-400' : 'bg-green-400'}`} />
        {escaped ? 'Moon Escaped' : 'Stable'}
      </div>

      {/* Habitability badge */}
      <div className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-xs font-semibold tracking-wide ${
        inHZ
          ? 'bg-emerald-900/70 text-emerald-300 border border-emerald-700/50'
          : 'bg-orange-900/70 text-orange-300 border border-orange-700/50'
      }`}>
        <span className={`w-1.5 h-1.5 rounded-full ${inHZ ? 'bg-emerald-400' : 'bg-orange-400'}`} />
        {inHZ ? 'Habitable' : 'Uninhabitable'}
      </div>

      {/* Data readouts — no backdrop-blur so objects show through */}
      <div className="bg-black/45 rounded px-2.5 py-2 space-y-1 border border-gray-700/40 min-w-[230px]">
        <Row label="Time"   value={`${fmt(simTime, 3)} yr`} />
        <Row label="Frame"  value={`${frameIndex + 1} / ${totalFrames}`} />

        <Sep />
        <Row label="Moon–Planet"  value={`${fmt(moonPlanetDist, 4)} AU`} dim={`${fmt(hillFrac * 100, 1)}% R_Hill`} highlight={escaped ? 'red' : 'green'} />
        <Row label="Planet–Star"  value={`${fmt(planetStarDist, 4)} AU`} highlight={inHZ ? 'green' : 'red'} />
        <Row label="Moon–Star"    value={`${fmt(moonStarDist, 4)} AU`}   highlight={inHZ ? 'green' : 'red'} />

        <Sep />
        <Row label="Moon speed"   value={`${fmt(moonSpeed, 3)} AU/yr`} />
        <Row label="Planet speed" value={`${fmt(planetSpeed, 3)} AU/yr`} />

        <Sep />
        <Row label="Moon Teff"    value={`${moonTempK.toFixed(0)} K`} highlight={inHZ ? 'green' : 'red'} />
        {rhill > 0 && <Row label="R_Hill" value={`${fmt(rhill, 4)} AU`} />}

        <Sep />
        <Row label="T_planet" value={T_planet > 0 ? `${fmt(T_planet, 3)} yr` : '—'} />
        <Row label="T_moon"   value={T_moon   > 0 ? `${fmt(T_moon,   4)} yr` : '—'} />

        <Sep />
        <Row label="Planet orbits" value={`${planetOrbits}`} />
        <Row label="Moon orbits"   value={`${moonOrbits}`} />
      </div>
    </div>
  );
}

function Sep() {
  return <div className="border-t border-gray-700/40 my-1" />;
}

function Row({ label, value, dim, highlight }: { label: string; value: string; dim?: string; highlight?: 'green' | 'red' }) {
  const valueColor = highlight === 'green' ? 'text-green-400' : highlight === 'red' ? 'text-red-400' : 'text-blue-300';
  return (
    <div className="flex items-baseline justify-between gap-4 text-xs">
      <span className="text-gray-500 shrink-0">{label}</span>
      <span className={`${valueColor} font-mono text-right`}>
        {value}
        {dim && <span className="text-gray-500 ml-1">{dim}</span>}
      </span>
    </div>
  );
}
