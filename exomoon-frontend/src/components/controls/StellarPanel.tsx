'use client';
import React from 'react';
import { useSimulationStore } from '@/hooks/useSimulationStore';
import { ParameterSlider } from './ParameterSlider';
import { PARAM_RANGES, PARAM_LABELS } from '@/lib/paramDefaults';
import { cn } from '@/lib/utils';

export function StellarPanel() {
  const { params, setParam, jobStatus } = useSimulationStore();
  const isRunning = jobStatus === 'running';

  return (
    <div className="p-3 space-y-2.5">
      <div className="space-y-1">
        <label className="text-xs text-gray-400">
          {PARAM_LABELS['Ts']} <span className="text-gray-600">(K)</span>
        </label>
        <input
          type="number"
          min={2000} max={20000} step={1}
          value={params.Ts}
          disabled={isRunning}
          onChange={e => setParam('Ts', parseFloat(e.target.value) || 5772)}
          className={cn(
            'w-full px-2 py-1 text-xs rounded bg-gray-800 border border-gray-700',
            'text-blue-300 font-mono focus:outline-none focus:border-blue-500',
            isRunning && 'opacity-40 cursor-not-allowed'
          )}
        />
      </div>

      {(['rs_solar', 'ms_solar'] as const).map(key => (
        <ParameterSlider
          key={key}
          label={PARAM_LABELS[key]}
          unit={PARAM_RANGES[key].unit}
          value={params[key] as number}
          min={PARAM_RANGES[key].min}
          max={PARAM_RANGES[key].max}
          step={PARAM_RANGES[key].step}
          onChange={v => setParam(key, v)}
          disabled={isRunning}
        />
      ))}
    </div>
  );
}
