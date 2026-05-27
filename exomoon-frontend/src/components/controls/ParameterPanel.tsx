'use client';
import React from 'react';
import { Download, Play } from 'lucide-react';
import { useSimulationStore } from '@/hooks/useSimulationStore';
import { ParameterSlider } from './ParameterSlider';
import { NasaSearch } from './NasaSearch';
import { JobStatusBanner } from './JobStatusBanner';
import { PARAM_RANGES, PARAM_LABELS } from '@/lib/paramDefaults';
import { bodyRadiusEarth, densityFromRadiusEarth } from '@/lib/trajectoryMath';
import { cn } from '@/lib/utils';

interface ParameterPanelProps {
  onRun: () => void;
  onExportCsv: () => void;
  isRunning?: boolean;
}

function ColHeader({ color, children }: { color: string; children: React.ReactNode }) {
  return (
    <p className={cn(
      'text-[10px] font-semibold uppercase tracking-widest pb-1 mb-2 border-b border-gray-800',
      color,
    )}>
      {children}
    </p>
  );
}

export function ParameterPanel({ onRun, onExportCsv, isRunning }: ParameterPanelProps) {
  const { params, simYears, setParam, setSimYears, dmCgs, setDmCgs } = useSimulationStore();

  const plRadeEarth = bodyRadiusEarth(params.mp_earth, params.dp_cgs);
  const rmEarth     = bodyRadiusEarth(params.mm_earth, dmCgs);

  const handlePlRade = (r: number) => {
    const d = densityFromRadiusEarth(params.mp_earth, r);
    if (d > 0.2 && d <= 40) setParam('dp_cgs', d);
  };
  const handleRmEarth = (r: number) => {
    const d = densityFromRadiusEarth(params.mm_earth, r);
    if (d > 0.2 && d <= 40) setDmCgs(d);
  };

  return (
    <div className="p-3 space-y-3">

      {/* NASA search — full width */}
      <NasaSearch />

      <div className="border-t border-gray-800" />

      {/* ── 3-column parameter grid ── */}
      <div className="grid grid-cols-3 gap-x-5 items-start">

        {/* ── Stellar ── */}
        <div className="space-y-2.5">
          <ColHeader color="text-yellow-400/80">Stellar</ColHeader>

          {/* Stellar temperature — wide range, use number input */}
          <div className="space-y-1">
            <label className="text-xs text-gray-400">
              {PARAM_LABELS['Ts']} <span className="text-gray-600">(K)</span>
            </label>
            <input
              type="number"
              min={2000} max={20000} step={1}
              value={params.Ts}
              onChange={e => setParam('Ts', parseFloat(e.target.value) || 5772)}
              className={cn(
                'w-full px-2 py-1 text-xs rounded bg-gray-800 border border-gray-700',
                'text-blue-300 font-mono focus:outline-none focus:border-blue-500'
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

        {/* ── Planet ── */}
        <div className="space-y-2.5">
          <ColHeader color="text-blue-400/80">Planet</ColHeader>

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

        {/* ── Moon ── */}
        <div className="space-y-2.5">
          <ColHeader color="text-red-400/80">Moon</ColHeader>

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

          {/* Moon direction */}
          <div className="space-y-1">
            <label className="text-xs text-gray-400">Moon orbit direction</label>
            <div className="flex gap-3">
              {(['Prograde', 'Retrograde'] as const).map(dir => (
                <label key={dir} className="flex items-center gap-1.5 text-xs text-gray-300 cursor-pointer">
                  <input
                    type="radio"
                    name="moon_dir"
                    value={dir}
                    checked={params.moon_retrograde === (dir === 'Retrograde')}
                    onChange={() => setParam('moon_retrograde', dir === 'Retrograde')}
                    className="accent-blue-500"
                  />
                  {dir}
                </label>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="border-t border-gray-800" />

      {/* Simulation duration — full width */}
      <div className="space-y-1">
        <label className="text-xs text-gray-400">
          Simulation duration <span className="text-gray-600">(years, 0 = 1 orbit)</span>
        </label>
        <input
          type="number"
          min={0} step="any"
          value={simYears}
          onChange={e => setSimYears(parseFloat(e.target.value) || 0)}
          className={cn(
            'w-full px-2 py-1 text-xs rounded bg-gray-800 border border-gray-700',
            'text-blue-300 font-mono focus:outline-none focus:border-blue-500'
          )}
        />
      </div>

      <div className="border-t border-gray-800" />

      {/* Job status */}
      <JobStatusBanner />

      {/* Action buttons */}
      <button
        onClick={onRun}
        disabled={isRunning}
        className={cn(
          'w-full py-2 px-4 rounded text-sm font-medium flex items-center justify-center gap-2',
          'bg-blue-600 hover:bg-blue-500 text-white transition-colors',
          isRunning && 'opacity-60 cursor-not-allowed'
        )}
      >
        <Play size={14} />
        {isRunning ? 'Running…' : 'Run Simulation'}
      </button>

      <button
        onClick={onExportCsv}
        className={cn(
          'w-full py-1.5 px-4 rounded text-sm font-medium flex items-center justify-center gap-2',
          'bg-gray-700 hover:bg-gray-600 text-gray-200 transition-colors'
        )}
      >
        <Download size={14} />
        Export CSV
      </button>
    </div>
  );
}
