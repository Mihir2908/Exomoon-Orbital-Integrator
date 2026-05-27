'use client';
import React from 'react';
import { Download } from 'lucide-react';
import { useSimulationStore } from '@/hooks/useSimulationStore';
import { bodyRadiusAU } from '@/lib/trajectoryMath';
import { EdaControls } from './EdaControls';
import { EdaPlot } from './EdaPlot';

export function EdaPanel() {
  const {
    trajectoryFrames, simMeta,
    edaVars, edaPlotType, edaNormalize,
    showHzOverlay, showHillOverlay,
    params, dmCgs,
  } = useSimulationStore();

  const handleExportCsv = () => {
    if (!trajectoryFrames || trajectoryFrames.length === 0) return;
    const cols = Object.keys(trajectoryFrames[0]);
    const header = cols.join(',');
    const rows = trajectoryFrames.map(f =>
      cols.map(c => (f as unknown as Record<string, unknown>)[c] ?? '').join(',')
    );
    const blob = new Blob([[header, ...rows].join('\n')], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'trajectory.csv'; a.click();
    URL.revokeObjectURL(url);
  };

  // Overlay bounds — computed from params + simMeta, all in AU
  const planetRadiusAU = bodyRadiusAU(params.mp_earth, params.dp_cgs);
  const rocheLimitAU   = 2.44 * planetRadiusAU * Math.cbrt(params.dp_cgs / dmCgs);

  const overlayBounds = {
    hzInner:   simMeta?.a_inner_au  ?? null,
    hzOuter:   simMeta?.a_outer_au  ?? null,
    hillInner: 4 * rocheLimitAU,
    hillOuter: simMeta?.rhill_AU    ?? null,
  };

  return (
    <div className="flex flex-col h-full bg-gray-950">
      <div className="flex items-center px-4 py-2 border-b border-gray-800 shrink-0">
        <h2 className="text-xs font-semibold text-gray-400 tracking-widest uppercase">EDA</h2>
      </div>

      <EdaControls />

      <div className="flex-1 min-h-0 p-2">
        {trajectoryFrames && trajectoryFrames.length > 0 ? (
          <EdaPlot
            frames={trajectoryFrames}
            variables={edaVars}
            plotType={edaPlotType}
            normalize={edaNormalize}
            showHz={showHzOverlay}
            showHill={showHillOverlay}
            overlayBounds={overlayBounds}
          />
        ) : (
          <div className="flex items-center justify-center h-full text-gray-600 text-sm">
            Run a simulation to enable EDA plots
          </div>
        )}
      </div>

      {/* Export CSV footer */}
      <div className="shrink-0 px-3 py-2 border-t border-gray-800">
        <button
          onClick={handleExportCsv}
          disabled={!trajectoryFrames || trajectoryFrames.length === 0}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium
                     bg-gray-700 hover:bg-gray-600 text-gray-200 transition-colors
                     disabled:opacity-40 disabled:cursor-not-allowed"
        >
          <Download size={12} />
          Export CSV
        </button>
      </div>
    </div>
  );
}
