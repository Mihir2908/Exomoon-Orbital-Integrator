'use client';
import React, { useRef, useState, useCallback } from 'react';
import dynamic from 'next/dynamic';
import { Download } from 'lucide-react';
import { VAR_INFO } from '@/lib/trajectoryMath';
import type { TrajectoryFrame } from '@/lib/types';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false, loading: () => (
  <div className="flex items-center justify-center h-full text-gray-600 text-sm">Loading plot…</div>
)});

interface OverlayBounds {
  hzInner:   number | null;
  hzOuter:   number | null;
  hillInner: number;
  hillOuter: number | null;
}

interface EdaPlotProps {
  frames:       TrajectoryFrame[];
  variables:    string[];
  plotType:     'line' | 'scatter';
  normalize:    boolean;
  showHz:       boolean;
  showHill:     boolean;
  overlayBounds: OverlayBounds;
}

const COLORS = ['#60a5fa', '#34d399', '#f87171', '#fbbf24', '#a78bfa', '#fb923c', '#38bdf8'];

// Build a Plotly rect shape for a shaded horizontal band with dotted borders
function bandShape(
  y0: number, y1: number,
  fillColor: string, lineColor: string,
): object {
  return {
    type: 'rect',
    xref: 'paper',
    x0: 0, x1: 1,
    yref: 'y',
    y0, y1,
    fillcolor: fillColor,
    line: { color: lineColor, width: 1, dash: 'dot' },
    layer: 'below',
  };
}

export function EdaPlot({
  frames, variables, plotType, normalize,
  showHz, showHill, overlayBounds,
}: EdaPlotProps) {
  const gdRef = useRef<HTMLDivElement | null>(null);
  const [saving, setSaving] = useState(false);
  const handleExport = useCallback(async () => {
    const gd = gdRef.current;
    if (!gd) return;
    setSaving(true);
    try {
      // Import the pre-built browser bundle directly. plotly.js/dist/plotly.js is already
      // fully bundled — no unresolved buffer/ or glslify deps — and exports the Plotly
      // object as module.exports (exposed as .default via webpack ESM dynamic import).
      // @ts-ignore – no .d.ts for subpath; webpack resolves plotly.js/dist/plotly.js at runtime
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const Plotly = ((await import('plotly.js/dist/plotly')) as any).default as any;
      if (!Plotly?.toImage) {
        console.warn('Plotly.toImage not available');
        return;
      }
      const url: string = await Plotly.toImage(gd, {
        format: 'png',
        width: 1000,
        height: 480,
        scale: 2,
      });
      const a = document.createElement('a');
      a.download = 'eda_plot.png';
      a.href = url;
      a.click();
    } catch (e) {
      console.error('EDA export failed:', e);
    } finally {
      setSaving(false);
    }
  }, []);

  if (frames.length === 0 || variables.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-gray-600 text-sm">
        Select variables above to plot
      </div>
    );
  }

  const timeArr = frames.map(f => f.t_years ?? 0);

  // Normalization ranges for overlay alignment
  const psArr  = frames.map(f => f.planet_star_dist  ?? 0);
  const mpArr  = frames.map(f => f.moon_planet_dist  ?? 0);
  const psMn   = Math.min(...psArr), psMx = Math.max(...psArr);
  const psSpan = psMx - psMn || 1;
  const mpMn   = Math.min(...mpArr), mpMx = Math.max(...mpArr);
  const mpSpan = mpMx - mpMn || 1;

  const normPs   = (v: number) => (v - psMn) / psSpan;
  const normMp   = (v: number) => (v - mpMn) / mpSpan;

  const traces = variables.map((col, i) => {
    let yArr = frames.map(f => (f as unknown as Record<string, number>)[col] ?? 0);
    if (normalize) {
      const mn = Math.min(...yArr);
      const mx = Math.max(...yArr);
      const span = mx - mn || 1;
      yArr = yArr.map(v => (v - mn) / span);
    }
    const info = VAR_INFO[col];
    return {
      x: timeArr,
      y: yArr,
      mode: plotType === 'scatter' ? ('markers' as const) : ('lines' as const),
      type: 'scatter' as const,
      name: info?.label ?? col,
      marker: { color: COLORS[i % COLORS.length], size: plotType === 'scatter' ? 3 : undefined },
      line: plotType === 'line' ? { color: COLORS[i % COLORS.length], width: 1.5 } : undefined,
    };
  });

  // Build overlay shapes
  const shapes: object[] = [];

  if (showHz && overlayBounds.hzInner !== null && overlayBounds.hzOuter !== null) {
    const lo = normalize ? normPs(overlayBounds.hzInner) : overlayBounds.hzInner;
    const hi = normalize ? normPs(overlayBounds.hzOuter) : overlayBounds.hzOuter;
    shapes.push(bandShape(lo, hi, 'rgba(0,200,80,0.08)', 'rgba(0,200,80,0.55)'));
  }

  if (showHill && overlayBounds.hillOuter !== null) {
    const lo = normalize ? normMp(overlayBounds.hillInner) : overlayBounds.hillInner;
    const hi = normalize ? normMp(overlayBounds.hillOuter) : overlayBounds.hillOuter;
    shapes.push(bandShape(lo, hi, 'rgba(180,100,255,0.08)', 'rgba(180,100,255,0.55)'));
  }

  return (
    <div className="relative w-full h-full">
      <Plot
        onInitialized={(_, gd) => { gdRef.current = gd as HTMLDivElement; }}
        data={traces}
        layout={{
        paper_bgcolor: 'transparent',
        plot_bgcolor: '#0d1117',
        font: { color: '#9ca3af', family: 'monospace', size: 11 },
        xaxis: {
          title: { text: 'Time (yr)', font: { size: 11 } },
          gridcolor: '#1f2937',
          zerolinecolor: '#374151',
          color: '#6b7280',
        },
        yaxis: {
          title: { text: normalize ? 'Normalized value' : '', font: { size: 11 } },
          gridcolor: '#1f2937',
          zerolinecolor: '#374151',
          color: '#6b7280',
        },
        legend: { bgcolor: 'rgba(0,0,0,0.4)', bordercolor: '#374151', borderwidth: 1 },
        margin: { t: 20, r: 20, b: 50, l: 55 },
        hovermode: 'x unified',
        hoverlabel: { bgcolor: '#1f2937', bordercolor: '#374151', font: { color: '#e5e7eb' } },
        shapes,
      }}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: '100%', height: '100%' }}
      useResizeHandler
      />
      <button
        onClick={handleExport}
        disabled={saving}
        className="absolute top-1 right-2 flex items-center gap-1 px-1.5 py-0.5 rounded
                   bg-gray-800/80 hover:bg-gray-700 text-gray-400 hover:text-gray-200
                   text-[10px] transition-colors disabled:opacity-40 z-10"
        title="Export as PNG"
      >
        <Download size={10} />
        {saving ? 'Saving…' : 'PNG'}
      </button>
    </div>
  );
}
