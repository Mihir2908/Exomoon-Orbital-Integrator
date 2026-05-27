'use client';
import React from 'react';
import { useSimulationStore } from '@/hooks/useSimulationStore';
import { ParameterSlider } from './ParameterSlider';
import { PARAM_RANGES, PARAM_LABELS } from '@/lib/paramDefaults';
import { bodyRadiusEarth, densityFromRadiusEarth } from '@/lib/trajectoryMath';

export function MoonPanel() {
  const { params, setParam, dmCgs, setDmCgs, jobStatus } = useSimulationStore();
  const isRunning = jobStatus === 'running';

  const rmEarth = bodyRadiusEarth(params.mm_earth, dmCgs);

  const handleRmEarth = (r: number) => {
    const d = densityFromRadiusEarth(params.mm_earth, r);
    if (d > 0.2 && d <= 40) setDmCgs(d);
  };

  return (
    <div className="p-3 space-y-2.5">
      <ParameterSlider
        label={PARAM_LABELS['mm_earth']} unit={PARAM_RANGES['mm_earth'].unit}
        value={params.mm_earth}
        min={PARAM_RANGES['mm_earth'].min} max={PARAM_RANGES['mm_earth'].max} step={PARAM_RANGES['mm_earth'].step}
        onChange={v => setParam('mm_earth', v)} disabled={isRunning}
      />
      <ParameterSlider
        label="Moon density dm" unit="g/cc"
        value={dmCgs} min={0.2} max={40} step={0.01}
        onChange={v => setDmCgs(v)} disabled={isRunning}
      />
      <ParameterSlider
        label="Moon radius rm" unit="R⊕"
        value={rmEarth} min={0.05} max={5} step={0.01}
        onChange={handleRmEarth} disabled={isRunning}
      />
      {(['am_hill', 'em'] as const).map(key => (
        <ParameterSlider
          key={key}
          label={PARAM_LABELS[key]} unit={PARAM_RANGES[key].unit}
          value={params[key] as number}
          min={PARAM_RANGES[key].min} max={PARAM_RANGES[key].max} step={PARAM_RANGES[key].step}
          onChange={v => setParam(key, v)} disabled={isRunning}
        />
      ))}

      <div className="space-y-1">
        <label className="text-xs text-gray-400">Moon orbit direction</label>
        <div className="flex gap-3">
          {(['Prograde', 'Retrograde'] as const).map(dir => (
            <label key={dir} className="flex items-center gap-1.5 text-xs text-gray-300 cursor-pointer">
              <input
                type="radio"
                name="moon_dir_panel"
                value={dir}
                checked={params.moon_retrograde === (dir === 'Retrograde')}
                onChange={() => setParam('moon_retrograde', dir === 'Retrograde')}
                disabled={isRunning}
                className="accent-blue-500"
              />
              {dir}
            </label>
          ))}
        </div>
      </div>
    </div>
  );
}
