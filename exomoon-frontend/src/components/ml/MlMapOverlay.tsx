'use client';
import React, { useState, useRef, useCallback, useEffect } from 'react';
import dynamic from 'next/dynamic';
import { X, ChevronDown, ChevronUp, Loader2, Brain, Play, Download } from 'lucide-react';
import { useSimulationStore } from '@/hooks/useSimulationStore';
import { cn } from '@/lib/utils';
import type { MlPrediction } from '@/lib/types';

// Dynamic Plotly import — never SSR (Three.js canvas page is client-only anyway)
const Plot = dynamic(() => import('react-plotly.js'), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-full text-gray-600 text-xs">Loading chart…</div>
  ),
});

const AGENT_URL = process.env.NEXT_PUBLIC_AGENT_URL ?? 'http://127.0.0.1:8000';

// ── Types ────────────────────────────────────────────────────────────────────

interface MlMapOverlayProps {
  onClose: () => void;
  containerRef: React.RefObject<HTMLDivElement | null>;
  /** Called after the store has been updated with predicted params; should trigger a sim run. */
  onApplyAndRun: () => void;
}

interface TrainStatus {
  status: string;        // 'idle' | 'running' | 'complete' | 'error' | 'failed'
  epoch?: number;
  total_epochs?: number;
  train_loss?: number;
  val_loss?: number;
  flag_acc?: number;
  elapsed_s?: number;
  error?: string;
}

interface TrainingHistory {
  train_loss:           number[];
  val_loss:             number[];
  flag_accuracy:        number[];        // validation
  flag_accuracy_train?: number[];        // training (added in later runs)
  dist_mae:             number[];
  epochs:               number;
  hyperparams?:         Record<string, unknown>;
}

// ── Section header ────────────────────────────────────────────────────────────

function SectionHeader({
  title, open, onToggle,
}: { title: string; open: boolean; onToggle: () => void }) {
  return (
    <button
      onClick={onToggle}
      className="w-full flex items-center justify-between px-3 py-1.5
                 hover:bg-white/5 transition-colors select-none"
    >
      <span className="text-[10px] text-gray-400 font-semibold tracking-widest uppercase">
        {title}
      </span>
      {open
        ? <ChevronUp   size={11} className="text-gray-500 shrink-0" />
        : <ChevronDown size={11} className="text-gray-500 shrink-0" />}
    </button>
  );
}

// ── Main overlay ──────────────────────────────────────────────────────────────

export function MlMapOverlay({ onClose, containerRef, onApplyAndRun }: MlMapOverlayProps) {
  // ── Store ────────────────────────────────────────────────────────────────────
  const {
    params, simYears,
    mlPrediction, mlMassIdx,
    setMlPrediction, setMlMassIdx,
    setParam,
  } = useSimulationStore();

  // ── Drag ─────────────────────────────────────────────────────────────────────
  const panelRef = useRef<HTMLDivElement>(null);
  const [dragPos, setDragPos] = useState<{ left: number; top: number } | null>(null);

  const handleDragStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    const panel = panelRef.current;
    const container = containerRef.current;
    if (!panel || !container) return;
    const pr = panel.getBoundingClientRect();
    const ox = e.clientX - pr.left;
    const oy = e.clientY - pr.top;
    function onMove(ev: MouseEvent) {
      const cr = containerRef.current?.getBoundingClientRect();
      const pr2 = panelRef.current?.getBoundingClientRect();
      if (!cr || !pr2) return;
      setDragPos({
        left: Math.max(0, Math.min(ev.clientX - cr.left - ox, cr.width  - pr2.width)),
        top:  Math.max(0, Math.min(ev.clientY - cr.top  - oy, cr.height - pr2.height)),
      });
    }
    function onUp() {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup',   onUp);
    }
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup',   onUp);
  }, [containerRef]);

  // ── Section collapse ─────────────────────────────────────────────────────────
  const [sec1Open, setSec1Open] = useState(true);
  const [sec2Open, setSec2Open] = useState(false);
  const [sec3Open, setSec3Open] = useState(false);

  // ═══════════════════════════════════════════════════════════════════════════
  // Section 3: Training history  (defined first — used by fetchTrainStatus)
  // ═══════════════════════════════════════════════════════════════════════════
  const [history, setHistory] = useState<TrainingHistory | null>(null);

  const fetchHistory = useCallback(async () => {
    try {
      const r    = await fetch(`${AGENT_URL}/ml/train/history`);
      const data = await r.json();
      // ok === false means "file not found" — leave history null
      if (data.ok !== false) setHistory(data as TrainingHistory);
    } catch { /* network error — silently ignore */ }
  }, []);

  // ═══════════════════════════════════════════════════════════════════════════
  // Section 2: Model Training
  // ═══════════════════════════════════════════════════════════════════════════
  const [trainModelType, setTrainModelType] = useState<'gru' | 'lstm'>('gru');
  const [trainHidden,    setTrainHidden]    = useState(256);
  const [trainLayers,    setTrainLayers]    = useState(2);
  const [trainLr,        setTrainLr]        = useState(0.001);
  const [trainBatch,     setTrainBatch]     = useState(64);
  const [trainEpochs,    setTrainEpochs]    = useState(30);
  const [trainDataPath,  setTrainDataPath]  = useState('ml_dataset.parquet');
  const [trainOutDir,    setTrainOutDir]    = useState('models/');

  const [trainStatus, setTrainStatus] = useState<TrainStatus | null>(null);
  const pollRef      = useRef<ReturnType<typeof setInterval> | null>(null);
  const lossGdRef    = useRef<HTMLDivElement | null>(null);
  const accGdRef     = useRef<HTMLDivElement | null>(null);
  const heatmapGdRef = useRef<HTMLDivElement | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
  }, []);

  const downloadPlot = useCallback(async (gd: HTMLDivElement | null, filename: string) => {
    if (!gd) return;
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
        width: 900,
        height: 500,
        scale: 2,
      });
      const a = document.createElement('a');
      a.download = `${filename}.png`;
      a.href = url;
      a.click();
    } catch (e) {
      console.error('Plot download failed:', e);
    }
  }, []);

  const fetchTrainStatus = useCallback(async () => {
    try {
      const r    = await fetch(`${AGENT_URL}/ml/train/status`);
      const data = await r.json() as TrainStatus;
      setTrainStatus(data);
      if (data.status === 'complete' || data.status === 'error' || data.status === 'failed') {
        stopPolling();
        fetchHistory();          // refresh Section 3 charts
      }
    } catch { /* network error — keep polling */ }
  }, [stopPolling, fetchHistory]);

  const handleTrain = useCallback(async () => {
    try {
      const r = await fetch(`${AGENT_URL}/ml/train`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data_path:  trainDataPath,
          out_dir:    trainOutDir,
          epochs:     trainEpochs,
          batch_size: trainBatch,
          lr:         trainLr,
          hidden:     trainHidden,
          layers:     trainLayers,
          rnn_type:   trainModelType,
        }),
      });
      const data = await r.json();
      if (data.ok) {
        setTrainStatus({ status: 'running', epoch: 0, total_epochs: trainEpochs });
        stopPolling();
        pollRef.current = setInterval(fetchTrainStatus, 3000);
      } else {
        setTrainStatus({ status: 'error', error: data.detail ?? 'Train request failed' });
      }
    } catch (e: unknown) {
      setTrainStatus({ status: 'error', error: e instanceof Error ? e.message : 'Network error' });
    }
  }, [
    trainDataPath, trainOutDir, trainEpochs, trainBatch,
    trainLr, trainHidden, trainLayers, trainModelType,
    fetchTrainStatus, stopPolling,
  ]);

  // ═══════════════════════════════════════════════════════════════════════════
  // Section 1: Prediction
  // ═══════════════════════════════════════════════════════════════════════════
  const [predLoading, setPredLoading] = useState(false);
  const [predError,   setPredError]   = useState<string | null>(null);

  const handlePredict = useCallback(async () => {
    setPredLoading(true);
    setPredError(null);
    try {
      const res = await fetch(`${AGENT_URL}/ml/predict`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          system_params: {
            ms_solar: params.ms_solar,
            rs_solar: params.rs_solar,
            Ts:       params.Ts,
            mp_earth: params.mp_earth,
            dp_cgs:   params.dp_cgs,
            ap_AU:    params.ap_AU,
            ep:       params.ep,
          },
          t_sim:           simYears > 0 ? simYears : 1,
          moon_retrograde: params.moon_retrograde,
          em:              params.em,
          mm_resolution:   50,
          am_resolution:   50,
        }),
      });
      const data = await res.json();
      if (data.ok) {
        const pred: MlPrediction = {
          mmGrid:       data.mm_grid        as number[],
          amGrid:       data.am_grid        as number[],
          mapStable:    data.map_stable     as boolean[][],
          mapHabitable: data.map_habitable  as boolean[][],
          mapBoth:      data.map_both       as boolean[][],
          validMmRange: data.valid_mm_range as [number, number] | null,
          validAmPerMm: data.valid_am_per_mm as ([number, number] | null)[],
        };
        setMlPrediction(pred);
      } else {
        setPredError(data.message ?? data.error ?? 'Prediction failed');
      }
    } catch (e: unknown) {
      setPredError(e instanceof Error ? e.message : 'Network error');
    } finally {
      setPredLoading(false);
    }
  }, [params, simYears, setMlPrediction]);

  const handleApplyAndRun = useCallback(() => {
    if (!mlPrediction) return;
    const newMm  = mlPrediction.mmGrid[mlMassIdx];
    const amRange = mlPrediction.validAmPerMm[mlMassIdx];
    setParam('mm_earth', newMm);
    if (amRange) {
      setParam('am_hill', (amRange[0] + amRange[1]) / 2);
    }
    onApplyAndRun();
  }, [mlPrediction, mlMassIdx, setParam, onApplyAndRun]);

  // ── Lifecycle ─────────────────────────────────────────────────────────────────
  useEffect(() => {
    fetchHistory();
    return () => stopPolling();
  }, [fetchHistory, stopPolling]);

  // ── Derived heatmap data ──────────────────────────────────────────────────────
  // Transpose mapBoth from [mm][am] → [am][mm] so Plotly x=mm, y=am
  const heatmapZ = mlPrediction
    ? mlPrediction.amGrid.map((_, j) =>
        mlPrediction.mmGrid.map((_, i) => (mlPrediction.mapBoth[i][j] ? 1 : 0))
      )
    : null;

  const curMm      = mlPrediction?.mmGrid[mlMassIdx] ?? null;
  const curAmRange = mlPrediction?.validAmPerMm[mlMassIdx] ?? null;
  const validMmR   = mlPrediction?.validMmRange ?? null;
  const isTraining = trainStatus?.status === 'running';

  // ── Render ────────────────────────────────────────────────────────────────────
  return (
    <div
      ref={panelRef}
      className={cn(
        'absolute z-10 flex flex-col bg-gray-900/94 rounded-lg border border-violet-700/30 overflow-hidden',
        !dragPos && 'top-16 right-3',
      )}
      style={{
        width: 410,
        maxHeight: 'calc(100% - 80px)',
        ...(dragPos ? { left: dragPos.left, top: dragPos.top } : {}),
      }}
    >
      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <div
        onMouseDown={handleDragStart}
        className="flex items-center justify-between px-3 py-1.5 bg-violet-950/40
                   border-b border-gray-700/40 shrink-0 cursor-grab active:cursor-grabbing select-none"
      >
        <div className="flex items-center gap-2">
          <Brain size={12} className="text-violet-400" />
          <span className="text-xs text-violet-400/80 font-medium tracking-wide uppercase">
            ML Stability Predictor
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          {dragPos && (
            <button
              onMouseDown={e => e.stopPropagation()}
              onClick={() => setDragPos(null)}
              className="text-gray-600 hover:text-gray-300 transition-colors text-[10px]"
              title="Reset position"
            >↩</button>
          )}
          <button
            onMouseDown={e => e.stopPropagation()}
            onClick={onClose}
            className="text-gray-500 hover:text-white transition-colors"
          >
            <X size={14} />
          </button>
        </div>
      </div>

      {/* ── Scrollable body ──────────────────────────────────────────────────── */}
      <div className="flex-1 min-h-0 overflow-y-auto text-xs text-gray-300 divide-y divide-gray-800/60">

        {/* ═══ Section 1: Prediction ══════════════════════════════════════════ */}
        <SectionHeader title="Prediction" open={sec1Open} onToggle={() => setSec1Open(v => !v)} />
        {sec1Open && (
          <div className="p-3 space-y-3">
            {/* Run ML Prediction button */}
            <button
              onClick={handlePredict}
              disabled={predLoading}
              className={cn(
                'w-full flex items-center justify-center gap-1.5 py-1.5 rounded text-xs font-medium transition-colors',
                'bg-violet-600 hover:bg-violet-500 text-white',
                predLoading && 'opacity-60 cursor-not-allowed',
              )}
            >
              {predLoading
                ? <Loader2 size={11} className="animate-spin" />
                : <Brain size={11} />}
              {predLoading ? 'Running inference…' : 'Run ML Prediction'}
            </button>

            {predError && (
              <p className="text-red-400 text-[11px] text-center px-1">{predError}</p>
            )}

            {/* Heatmap — shown once prediction exists */}
            {mlPrediction && heatmapZ && (
              <>
                <div className="flex items-center justify-between">
                  <p className="text-gray-500 text-[10px] uppercase tracking-wide">
                    Moon Mass × Orbit Stability
                  </p>
                  <button
                    onClick={() => downloadPlot(heatmapGdRef.current, 'ml_stability_heatmap')}
                    className="flex items-center gap-0.5 px-1.5 py-0.5 rounded bg-gray-800/60
                               hover:bg-gray-700 text-gray-500 hover:text-gray-300 text-[9px] transition-colors"
                    title="Export as PNG"
                  >
                    <Download size={9} /> PNG
                  </button>
                </div>
              <div className="rounded overflow-hidden border border-gray-800/60" style={{ height: 210 }}>
                <Plot
                  onInitialized={(_, gd) => { heatmapGdRef.current = gd as HTMLDivElement; }}
                  data={[{
                    type:        'heatmap' as const,
                    x:           mlPrediction.mmGrid,
                    y:           mlPrediction.amGrid,
                    z:           heatmapZ,
                    colorscale:  [[0, '#1a2535'], [1, '#0d9488']],
                    showscale:   false,
                    hoverongaps: false,
                    zmin: 0, zmax: 1,
                    hovertemplate:
                      'mm: %{x:.4f} M⊕<br>am: %{y:.3f} Hill<br>stable+hab: %{z}<extra></extra>',
                  }]}
                  layout={{
                    paper_bgcolor: 'transparent',
                    plot_bgcolor:  '#0d1117',
                    font:   { color: '#9ca3af', family: 'monospace', size: 10 },
                    margin: { t: 8, r: 8, b: 40, l: 48 },
                    xaxis: {
                      title:    { text: 'Moon mass (M⊕)', font: { size: 10 } },
                      type:     'log' as const,
                      gridcolor: '#1f2937',
                      color:    '#6b7280',
                    },
                    yaxis: {
                      title:    { text: 'am (Hill radii)', font: { size: 10 } },
                      gridcolor: '#1f2937',
                      color:    '#6b7280',
                    },
                    // Vertical dashed line at currently selected mass
                    shapes: curMm != null ? [{
                      type:  'line'  as const,
                      xref:  'x'    as const,
                      yref:  'paper' as const,
                      x0: curMm, x1: curMm,
                      y0: 0,     y1: 1,
                      line: { color: '#c084fc', width: 1.5, dash: 'dash' as const },
                    }] : [],
                    uirevision: 'ml-map',
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  style={{ width: '100%', height: '100%' }}
                  useResizeHandler
                />
              </div>
              </>
            )}

            {/* Mass slider */}
            {mlPrediction && (
              <>
                <div className="space-y-1.5">
                  <div className="flex items-center justify-between">
                    <span className="text-gray-500">Moon mass</span>
                    <span className="text-violet-300 font-mono">
                      {curMm !== null ? curMm.toFixed(5) : '—'} M⊕
                    </span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={mlPrediction.mmGrid.length - 1}
                    step={1}
                    value={mlMassIdx}
                    onChange={e => setMlMassIdx(parseInt(e.target.value, 10))}
                    className="w-full cursor-pointer accent-violet-500"
                  />
                  <div className="flex justify-between text-gray-600 text-[10px] font-mono">
                    <span>{mlPrediction.mmGrid[0].toFixed(3)}</span>
                    <span>{mlPrediction.mmGrid[mlPrediction.mmGrid.length - 1].toFixed(3)} M⊕</span>
                  </div>
                </div>

                {/* Valid range readout */}
                <div className="rounded bg-gray-800/60 p-2 space-y-1 text-[11px]">
                  {validMmR ? (
                    <p className="text-teal-400/90 font-mono">
                      Valid mass:&nbsp;
                      {validMmR[0].toFixed(4)}–{validMmR[1].toFixed(4)} M⊕
                    </p>
                  ) : (
                    <p className="text-orange-400/80">No stable+habitable configurations found</p>
                  )}
                  {curAmRange ? (
                    <p className="text-cyan-400/90 font-mono">
                      Valid orbit:&nbsp;
                      {curAmRange[0].toFixed(3)}–{curAmRange[1].toFixed(3)} Hill radii
                    </p>
                  ) : (
                    <p className="text-gray-500">No valid orbit at this mass</p>
                  )}
                </div>

                {/* Apply & Run */}
                <button
                  onClick={handleApplyAndRun}
                  disabled={!curAmRange}
                  className={cn(
                    'w-full flex items-center justify-center gap-1.5 py-1.5 rounded text-xs font-medium transition-colors',
                    curAmRange
                      ? 'bg-teal-600 hover:bg-teal-500 text-white'
                      : 'bg-gray-700/60 text-gray-500 cursor-not-allowed',
                  )}
                  title={
                    curAmRange
                      ? 'Set predicted params and run simulation'
                      : 'No valid orbit at this mass — move slider to a teal region'
                  }
                >
                  <Play size={10} />
                  Apply &amp; Run
                </button>
              </>
            )}
          </div>
        )}

        {/* ═══ Section 2: Model Training ══════════════════════════════════════ */}
        <SectionHeader title="Model Training" open={sec2Open} onToggle={() => setSec2Open(v => !v)} />
        {sec2Open && (
          <div className="p-3 space-y-2.5">

            {/* Hyperparameter grid */}
            <div className="grid grid-cols-2 gap-x-3 gap-y-2">
              <HpLabel label="Model type">
                <select
                  value={trainModelType}
                  onChange={e => setTrainModelType(e.target.value as 'gru' | 'lstm')}
                  disabled={isTraining}
                  className={selectCls}
                >
                  <option value="gru">GRU</option>
                  <option value="lstm">LSTM</option>
                </select>
              </HpLabel>
              <HpLabel label="Hidden size">
                <select
                  value={trainHidden}
                  onChange={e => setTrainHidden(Number(e.target.value))}
                  disabled={isTraining}
                  className={selectCls}
                >
                  <option value={128}>128</option>
                  <option value={256}>256</option>
                  <option value={512}>512</option>
                </select>
              </HpLabel>
              <HpLabel label="Layers">
                <select
                  value={trainLayers}
                  onChange={e => setTrainLayers(Number(e.target.value))}
                  disabled={isTraining}
                  className={selectCls}
                >
                  <option value={1}>1</option>
                  <option value={2}>2</option>
                  <option value={3}>3</option>
                </select>
              </HpLabel>
              <HpLabel label="Learning rate">
                <input
                  type="number" min={1e-5} max={0.1} step="any"
                  value={trainLr}
                  onChange={e => setTrainLr(parseFloat(e.target.value) || 1e-3)}
                  disabled={isTraining}
                  className={inputCls}
                />
              </HpLabel>
              <HpLabel label="Batch size">
                <input
                  type="number" min={8} max={512} step={1}
                  value={trainBatch}
                  onChange={e => setTrainBatch(parseInt(e.target.value) || 64)}
                  disabled={isTraining}
                  className={inputCls}
                />
              </HpLabel>
              <HpLabel label="Epochs">
                <input
                  type="number" min={1} max={500} step={1}
                  value={trainEpochs}
                  onChange={e => setTrainEpochs(parseInt(e.target.value) || 30)}
                  disabled={isTraining}
                  className={inputCls}
                />
              </HpLabel>
            </div>

            <HpLabel label="Dataset path">
              <input
                type="text"
                value={trainDataPath}
                onChange={e => setTrainDataPath(e.target.value)}
                disabled={isTraining}
                className={inputCls}
                placeholder="ml_dataset.parquet"
              />
            </HpLabel>
            <HpLabel label="Output dir">
              <input
                type="text"
                value={trainOutDir}
                onChange={e => setTrainOutDir(e.target.value)}
                disabled={isTraining}
                className={inputCls}
                placeholder="models/"
              />
            </HpLabel>

            {/* Train button */}
            <button
              onClick={handleTrain}
              disabled={isTraining}
              className={cn(
                'w-full flex items-center justify-center gap-1.5 py-1.5 rounded text-xs font-medium transition-colors',
                'bg-amber-600 hover:bg-amber-500 text-white',
                isTraining && 'opacity-60 cursor-not-allowed',
              )}
            >
              {isTraining ? <Loader2 size={11} className="animate-spin" /> : '⚡'}
              {isTraining ? 'Training…' : 'Train Model'}
            </button>

            {/* Training progress / status */}
            {trainStatus && (
              <div className={cn(
                'rounded p-2 text-[11px] font-mono space-y-0.5',
                trainStatus.status === 'complete'
                  ? 'bg-green-900/30 text-green-400'
                  : trainStatus.status === 'error' || trainStatus.status === 'failed'
                  ? 'bg-red-900/30 text-red-400'
                  : 'bg-gray-800/60 text-gray-300',
              )}>
                {trainStatus.status === 'running' && (
                  <>
                    {trainStatus.epoch != null && (
                      <p>Epoch {trainStatus.epoch} / {trainStatus.total_epochs ?? '?'}</p>
                    )}
                    {trainStatus.train_loss != null && (
                      <p>
                        loss: {trainStatus.train_loss.toFixed(4)} ·{' '}
                        val: {(trainStatus.val_loss ?? 0).toFixed(4)}
                      </p>
                    )}
                    {trainStatus.flag_acc != null && (
                      <p>flag_acc: {(trainStatus.flag_acc * 100).toFixed(1)}%</p>
                    )}
                    {trainStatus.elapsed_s != null && (
                      <p className="text-gray-500">{trainStatus.elapsed_s.toFixed(0)}s elapsed</p>
                    )}
                  </>
                )}
                {trainStatus.status === 'complete' && <p>✓ Training complete — model saved</p>}
                {(trainStatus.status === 'error' || trainStatus.status === 'failed') && (
                  <p>✗ {trainStatus.error ?? 'Training failed'}</p>
                )}
              </div>
            )}
          </div>
        )}

        {/* ═══ Section 3: Model Performance ═══════════════════════════════════ */}
        <SectionHeader title="Model Performance" open={sec3Open} onToggle={() => setSec3Open(v => !v)} />
        {sec3Open && (
          <div className="p-3 space-y-2">
            {!history ? (
              <p className="text-gray-600 text-center py-4 leading-relaxed">
                No training history found.
                <br />
                Train a model to see performance charts.
              </p>
            ) : (
              <>
                {/* Loss curves */}
                <div className="flex items-center justify-between">
                  <p className="text-gray-500 text-[10px] uppercase tracking-wide">Loss Curves</p>
                  <button
                    onClick={() => downloadPlot(lossGdRef.current, 'ml_loss_curves')}
                    className="flex items-center gap-0.5 px-1.5 py-0.5 rounded bg-gray-800/60
                               hover:bg-gray-700 text-gray-500 hover:text-gray-300 text-[9px] transition-colors"
                    title="Export as PNG"
                  >
                    <Download size={9} /> PNG
                  </button>
                </div>
                <div className="rounded overflow-hidden" style={{ height: 165 }}>
                  <Plot
                    onInitialized={(_, gd) => { lossGdRef.current = gd as HTMLDivElement; }}
                    data={[
                      {
                        x:    history.train_loss.map((_, i) => i + 1),
                        y:    history.train_loss,
                        mode: 'lines' as const,
                        name: 'Train',
                        line: { color: '#60a5fa', width: 1.5 },
                      },
                      {
                        x:    history.val_loss.map((_, i) => i + 1),
                        y:    history.val_loss,
                        mode: 'lines' as const,
                        name: 'Val',
                        line: { color: '#f87171', width: 1.5, dash: 'dash' as const },
                      },
                    ]}
                    layout={{
                      paper_bgcolor: 'transparent',
                      plot_bgcolor:  '#0d1117',
                      font:   { color: '#9ca3af', family: 'monospace', size: 10 },
                      margin: { t: 8, r: 10, b: 36, l: 48 },
                      xaxis: {
                        title:    { text: 'Epoch', font: { size: 10 } },
                        gridcolor: '#1f2937', color: '#6b7280',
                      },
                      yaxis: {
                        title:    { text: 'Loss', font: { size: 10 } },
                        gridcolor: '#1f2937', color: '#6b7280',
                      },
                      legend: {
                        bgcolor:  'rgba(0,0,0,0)',
                        x: 0.98, xanchor: 'right',
                        y: 0.98, font: { size: 10 },
                      },
                    }}
                    config={{ displayModeBar: false, responsive: true }}
                    style={{ width: '100%', height: '100%' }}
                    useResizeHandler
                  />
                </div>

                {/* Flag accuracy */}
                <div className="flex items-center justify-between mt-2">
                  <p className="text-gray-500 text-[10px] uppercase tracking-wide">
                    Stable / Habitable Flag Accuracy
                  </p>
                  <button
                    onClick={() => downloadPlot(accGdRef.current, 'ml_flag_accuracy')}
                    className="flex items-center gap-0.5 px-1.5 py-0.5 rounded bg-gray-800/60
                               hover:bg-gray-700 text-gray-500 hover:text-gray-300 text-[9px] transition-colors"
                    title="Export as PNG"
                  >
                    <Download size={9} /> PNG
                  </button>
                </div>
                <div className="rounded overflow-hidden" style={{ height: 135 }}>
                  <Plot
                    onInitialized={(_, gd) => { accGdRef.current = gd as HTMLDivElement; }}
                    data={[
                      ...(history.flag_accuracy_train ? [{
                        x:    history.flag_accuracy_train.map((_, i) => i + 1),
                        y:    history.flag_accuracy_train.map(v => v * 100),
                        mode: 'lines' as const,
                        name: 'Train',
                        line: { color: '#34d399', width: 1.5 },
                      }] : []),
                      {
                        x:    history.flag_accuracy.map((_, i) => i + 1),
                        y:    history.flag_accuracy.map(v => v * 100),
                        mode: 'lines' as const,
                        name: 'Val',
                        line: { color: '#f87171', width: 1.5, dash: 'dash' as const },
                      },
                    ]}
                    layout={{
                      paper_bgcolor: 'transparent',
                      plot_bgcolor:  '#0d1117',
                      font:   { color: '#9ca3af', family: 'monospace', size: 10 },
                      margin: { t: 8, r: 10, b: 36, l: 48 },
                      xaxis: {
                        title:    { text: 'Epoch', font: { size: 10 } },
                        gridcolor: '#1f2937', color: '#6b7280',
                      },
                      yaxis: {
                        title:    { text: 'Accuracy (%)', font: { size: 10 } },
                        gridcolor: '#1f2937', color: '#6b7280',
                        range: [0, 100],
                      },
                      legend: {
                        bgcolor:  'rgba(0,0,0,0)',
                        x: 0.02, xanchor: 'left',
                        y: 0.98, font: { size: 10 },
                      },
                      showlegend: true,
                    }}
                    config={{ displayModeBar: false, responsive: true }}
                    style={{ width: '100%', height: '100%' }}
                    useResizeHandler
                  />
                </div>

                {/* Hyperparams summary */}
                {history.hyperparams && (
                  <div className="rounded bg-gray-800/40 p-2 text-[10px] text-gray-500 font-mono flex flex-wrap gap-x-3 gap-y-0.5">
                    {Object.entries(history.hyperparams).map(([k, v]) => (
                      <span key={k}>{k}: {String(v)}</span>
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Tiny shared helpers ───────────────────────────────────────────────────────

const selectCls =
  'w-full px-2 py-1 text-xs rounded bg-gray-800 border border-gray-700 ' +
  'text-gray-200 focus:outline-none focus:border-violet-500 disabled:opacity-50';

const inputCls =
  'w-full px-2 py-1 text-xs rounded bg-gray-800 border border-gray-700 ' +
  'text-gray-200 font-mono focus:outline-none focus:border-violet-500 disabled:opacity-50';

function HpLabel({
  label, children,
}: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-0.5">
      <span className="text-gray-500 text-[10px] uppercase tracking-wide">{label}</span>
      {children}
    </label>
  );
}
