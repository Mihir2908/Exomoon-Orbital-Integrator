'use client';
import React from 'react';
import { useSimulationStore } from '@/hooks/useSimulationStore';
import { ParameterSlider } from './ParameterSlider';
import { PARAM_RANGES, PARAM_LABELS } from '@/lib/paramDefaults';
import { bodyRadiusEarth, densityFromRadiusEarth } from '@/lib/trajectoryMath';

export function PlanetPanel() {
  const { params, setParam, jobStatus } = useSimulationStore();
  const isRunning = jobStatus === 'running';

  const plRadeEarth = bodyRadiusEarth(params.mp_earth, params.dp_cgs);

  const handlePlRade = (r: number) => {
    const d = densityFromRadiusEarth(params.mp_earth, r);
    if (d > 0.2 && d <= 40) setParam('dp_cgs', d);
  };

  return (
    <div className="p-3 space-y-2.5">
      <ParameterSlider
        label={PARAM_LABELS['mp_earth']} unit={PARAM_RANGES['mp_earth'].unit}
        value={params.mp_earth}
        min={PARAM_RANGES['mp_earth'].min} max={PARAM_RANGES['mp_earth'].max} step={PARAM_RANGES['mp_earth'].step}
        onChange={v => setParam('mp_earth', v)} disabled={isRunning}
      />
      <ParameterSlider
        label={PARAM_LABELS['dp_cgs']} unit={PARAM_RANGES['dp_cgs'].unit}
        value={params.dp_cgs}
        min={PARAM_RANGES['dp_cgs'].min} max={PARAM_RANGES['dp_cgs'].max} step={PARAM_RANGES['dp_cgs'].step}
        onChange={v => setParam('dp_cgs', v)} disabled={isRunning}
      />
      <ParameterSlider
        label="Planet radius rp" unit="R⊕"
        value={plRadeEarth} min={0.1} max={15} step={0.01}
        onChange={handlePlRade} disabled={isRunning}
      />
      {(['ap_AU', 'ep'] as const).map(key => (
        <ParameterSlider
          key={key}
          label={PARAM_LABELS[key]} unit={PARAM_RANGES[key].unit}
          value={params[key] as number}
          min={PARAM_RANGES[key].min} max={PARAM_RANGES[key].max} step={PARAM_RANGES[key].step}
          onChange={v => setParam(key, v)} disabled={isRunning}
        />
      ))}
    </div>
  );
}
