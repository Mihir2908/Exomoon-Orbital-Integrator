'use client';
import React from 'react';
import { useSimulationStore } from '@/hooks/useSimulationStore';
import { VAR_INFO, EDA_COLUMNS } from '@/lib/trajectoryMath';
import { cn } from '@/lib/utils';

export function EdaControls() {
  const {
    edaVars, edaPlotType, edaNormalize,
    showHzOverlay, showHillOverlay,
    setEdaVars, setEdaPlotType, setEdaNormalize,
    setShowHzOverlay, setShowHillOverlay,
  } = useSimulationStore();

  const toggle = (col: string) => {
    if (edaVars.includes(col)) {
      setEdaVars(edaVars.filter(v => v !== col));
    } else {
      setEdaVars([...edaVars, col]);
    }
  };

  return (
    <div className="flex flex-wrap items-center gap-3 px-4 py-2 bg-gray-900 border-b border-gray-800">
      {/* Variable chips */}
      <div className="flex flex-wrap gap-1">
        {EDA_COLUMNS.map(col => {
          const info = VAR_INFO[col];
          const active = edaVars.includes(col);
          return (
            <button
              key={col}
              onClick={() => toggle(col)}
              title={info?.label ?? col}
              className={cn(
                'px-2 py-0.5 text-xs rounded font-mono transition-colors border',
                active
                  ? 'bg-blue-700/60 border-blue-500 text-blue-200'
                  : 'bg-gray-800 border-gray-700 text-gray-400 hover:border-gray-500'
              )}
            >
              {info?.label ?? col}
            </button>
          );
        })}
      </div>

      <div className="ml-auto flex items-center gap-3 shrink-0">
        {/* Plot type */}
        <div className="flex items-center gap-1">
          {(['line', 'scatter'] as const).map(t => (
            <button
              key={t}
              onClick={() => setEdaPlotType(t)}
              className={cn(
                'px-2 py-0.5 text-xs rounded capitalize transition-colors',
                edaPlotType === t
                  ? 'bg-gray-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              )}
            >
              {t}
            </button>
          ))}
        </div>

        {/* Normalize toggle */}
        <label className="flex items-center gap-1.5 text-xs text-gray-400 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={edaNormalize}
            onChange={e => setEdaNormalize(e.target.checked)}
            className="accent-blue-500 w-3 h-3"
          />
          Normalize
        </label>

        {/* HZ overlay toggle */}
        <label className="flex items-center gap-1.5 text-xs cursor-pointer select-none">
          <input
            type="checkbox"
            checked={showHzOverlay}
            onChange={e => setShowHzOverlay(e.target.checked)}
            className="accent-green-500 w-3 h-3"
          />
          <span className="text-green-400">HZ</span>
        </label>

        {/* Hill sphere overlay toggle */}
        <label className="flex items-center gap-1.5 text-xs cursor-pointer select-none">
          <input
            type="checkbox"
            checked={showHillOverlay}
            onChange={e => setShowHillOverlay(e.target.checked)}
            className="accent-purple-400 w-3 h-3"
          />
          <span className="text-purple-400">Hill</span>
        </label>
      </div>
    </div>
  );
}
