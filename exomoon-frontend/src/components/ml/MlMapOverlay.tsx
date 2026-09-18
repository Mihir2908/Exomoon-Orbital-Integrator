'use client';
import React, { useState, useRef, useCallback, useEffect, useMemo } from 'react';
import dynamic from 'next/dynamic';
import { X, ChevronDown, ChevronUp, Loader2, Brain, Play, Download, Zap } from 'lucide-react';
import { useSimulationStore } from '@/hooks/useSimulationStore';
import { cn } from '@/lib/utils';
import { MiniOrbitView } from '@/components/animation/MiniOrbitView';
import type { MlPrediction, TrajectoryFrame, SimulationMeta } from '@/lib/types';

const Plot = dynamic(() => import('react-plotly.js'), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-full text-gray-600 text-xs">Loading chart…</div>
  ),
});

const AGENT_URL = process.env.NEXT_PUBLIC_AGENT_URL ?? 'http://127.0.0.1:8000';
// Heavy GPU calls (trajectory preview) go direct to avoid Next.js proxy timeout (~30s vs GPU ~150s+)
const AGENT_DIRECT = process.env.NEXT_PUBLIC_AGENT_DIRECT_URL ?? 'http://127.0.0.1:8000';

const CELL_PX       = 40;  // pixels per grid cell in trajectory preview canvas
const LEFT_LABEL_PX = 38;  // width of am (y-axis) label strip drawn on canvas left
const BOT_LABEL_PX  = 22;  // height of mm (x-axis) label strip drawn on canvas bottom
const PREVIEW_FRAMES = 120; // synthetic orbit frames for cell detail MiniOrbitView

/** Generate synthetic circular orbit frames for MiniOrbitView trajectory preview. */
function generateSyntheticOrbitFrames(amAU: number): TrajectoryFrame[] {
  return Array.from({ length: PREVIEW_FRAMES }, (_, i) => {
    const theta = (2 * Math.PI * i) / PREVIEW_FRAMES;
    return {
      t_years: i / PREVIEW_FRAMES,
      star_x: 0, star_y: 0, star_z: 0,
      planet_x: 0, planet_y: 0, planet_z: 0,
      moon_x: amAU * Math.cos(theta),
      moon_y: amAU * Math.sin(theta),
      moon_z: 0,
      star_vx: 0, star_vy: 0, star_vz: 0,
      planet_vx: 0, planet_vy: 0, planet_vz: 0,
      moon_vx: 0, moon_vy: 0, moon_vz: 0,
      moon_planet_dist: amAU,
      planet_star_dist: 1.0,
      moon_speed: 0,
      planet_speed: 0,
      star_speed: 0,
    };
  });
}

// ── Types ─────────────────────────────────────────────────────────────────────

interface MlMapOverlayProps {
  onClose: () => void;
  containerRef: React.RefObject<HTMLDivElement | null>;
  onApplyAndRun: () => void;
  frameIndex: number;
}

interface TrainingHistory {
  train_loss:           number[];
  val_loss:             number[];
  flag_accuracy?:       number[];
  flag_accuracy_train?: number[];
  dist_mae?:            number[];
  mono_loss?:           number[];
  epochs:               number;
  hyperparams?:         Record<string, unknown>;
}

interface TrajResult {
  ok:              boolean;
  map_stable:      boolean[][];
  map_habitable:   boolean[][];
  map_both:        boolean[][];
  confidence_map?: ('HIGH' | 'LOW' | null)[][];
  mm_grid:         number[];
  am_grid:         number[];
  wall_s?:         number;
  from_cache?:     boolean;
  cache_key?:      string;   // server-computed key — sent back on cell clicks to avoid key mismatch
}

// ── Section header ─────────────────────────────────────────────────────────────

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

// ── Layer tab ─────────────────────────────────────────────────────────────────

function LayerTab({
  label, active, onClick,
}: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={cn(
        'flex-1 py-1 text-[10px] font-medium rounded transition-colors',
        active
          ? 'bg-violet-700/70 text-violet-200'
          : 'text-gray-500 hover:text-gray-300 hover:bg-white/5',
      )}
    >
      {label}
    </button>
  );
}

// ── Read-only hyperparameter row ──────────────────────────────────────────────

function ReadField({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-start justify-between gap-3 text-[11px]">
      <span className="text-gray-500 uppercase tracking-wide text-[10px] shrink-0 pt-px">{label}</span>
      <span className="text-gray-300 font-mono text-right">{value}</span>
    </div>
  );
}

// ── Main overlay ───────────────────────────────────────────────────────────────

export function MlMapOverlay({ onClose, containerRef, onApplyAndRun, frameIndex }: MlMapOverlayProps) {
  const {
    params, simYears,
    mlPrediction, mlMassIdx,
    setMlPrediction, setMlMassIdx,
    setParam,
    setPreviewCellFrames,
    setTrajectoryData,
  } = useSimulationStore();

  // ── Drag ───────────────────────────────────────────────────────────────────
  const panelRef = useRef<HTMLDivElement>(null);
  const [dragPos, setDragPos] = useState<{ left: number; top: number } | null>(null);

  const handleDragStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    const panel     = panelRef.current;
    const container = containerRef.current;
    if (!panel || !container) return;
    const pr = panel.getBoundingClientRect();
    const ox = e.clientX - pr.left;
    const oy = e.clientY - pr.top;
    function onMove(ev: MouseEvent) {
      const cr  = containerRef.current?.getBoundingClientRect();
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

  // ── Section collapse ───────────────────────────────────────────────────────
  const [sec1Open, setSec1Open] = useState(false);   // Model Training (collapsed by default)
  const [sec2Open, setSec2Open] = useState(true);    // Prediction (open by default)
  const [sec3Open, setSec3Open] = useState(false);   // Model Performance

  // ── Section 1: Model Training layer tab ───────────────────────────────────
  const [trainLayer, setTrainLayer] = useState<'mlp' | 'hnn'>('mlp');

  // ── Section 2: Prediction ─────────────────────────────────────────────────
  const [predLayer, setPredLayer] = useState<'mlp' | 'trajectory'>('mlp');
  const [gridSize,  setGridSize]  = useState<30 | 50>(30);

  // Layer 1 state
  const [predLoading, setPredLoading] = useState(false);
  const [predError,   setPredError]   = useState<string | null>(null);

  // Layer 2 state
  const [trajEngine,        setTrajEngine]        = useState<'gt_leapfrog' | 'hnn_hinge4'>('gt_leapfrog');
  const [trajLoading,       setTrajLoading]        = useState(false);
  const [trajError,         setTrajError]          = useState<string | null>(null);
  const [trajResult,        setTrajResult]         = useState<TrajResult | null>(null);
  const [selectedCell,      setSelectedCell]       = useState<{ mmIdx: number; amIdx: number } | null>(null);
  const [selectedCellFrames, setSelectedCellFrames] = useState<TrajectoryFrame[] | null>(null);
  const [cellTrajLoading,   setCellTrajLoading]    = useState(false);
  const [cellTrajError,     setCellTrajError]      = useState<string | null>(null);

  // ── Section 3: Model Performance ──────────────────────────────────────────
  const [perfLayer, setPerfLayer] = useState<'mlp' | 'hnn'>('mlp');
  const [history,   setHistory]   = useState<TrainingHistory | null>(null);
  const [histError, setHistError] = useState<string | null>(null);

  // Plotly refs for PNG export
  const heatmapGdRef  = useRef<HTMLDivElement | null>(null);
  const lossGdRef     = useRef<HTMLDivElement | null>(null);
  const accGdRef      = useRef<HTMLDivElement | null>(null);

  // Canvas ref for trajectory grid
  const gridCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const downloadPlot = useCallback(async (gd: HTMLDivElement | null, filename: string) => {
    if (!gd) return;
    try {
      // @ts-ignore — no .d.ts for the dist subpath; webpack resolves at runtime
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const Plotly = ((await import('plotly.js/dist/plotly')) as any).default as any;
      if (!Plotly?.toImage) return;
      const url: string = await Plotly.toImage(gd, { format: 'png', width: 900, height: 500, scale: 2 });
      const a = document.createElement('a');
      a.download = `${filename}.png`;
      a.href = url;
      a.click();
    } catch (e) { console.error('Plot download failed:', e); }
  }, []);

  const fetchHistory = useCallback(async (layer: 'mlp' | 'hnn') => {
    setHistError(null);
    try {
      const url = layer === 'hnn'
        ? `${AGENT_URL}/ml/train/history?model_type=hnn`
        : `${AGENT_URL}/ml/train/history`;
      const r    = await fetch(url);
      const data = await r.json();
      if (data.ok === false) {
        setHistory(null);
        setHistError(data.message ?? 'History not found on server');
      } else if (Array.isArray(data.train_loss) && Array.isArray(data.val_loss)) {
        setHistory(data as TrainingHistory);
      } else {
        setHistory(null);
        setHistError('Unexpected response format from server');
      }
    } catch (e: unknown) {
      setHistory(null);
      setHistError(e instanceof Error ? e.message : 'Network error — is the agent service running?');
    }
  }, []);

  // ── Prediction: Layer 1 MLP ───────────────────────────────────────────────
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
          t_sim:           simYears > 0 ? simYears : 10.0,
          moon_retrograde: params.moon_retrograde,
          em:              params.em,
          mm_resolution:   gridSize,
          am_resolution:   gridSize,
        }),
      });
      const data = await res.json();
      if (data.ok) {
        const pred: MlPrediction = {
          mmGrid:       data.mm_grid         as number[],
          amGrid:       data.am_grid         as number[],
          mapStable:    data.map_stable      as boolean[][],
          mapHabitable: data.map_habitable   as boolean[][],
          mapBoth:      data.map_both        as boolean[][],
          validMmRange: data.valid_mm_range  as [number, number] | null,
          validAmPerMm: data.valid_am_per_mm as ([number, number] | null)[],
        };
        setMlPrediction(pred);
        setTrajResult(null);
        setSelectedCell(null);
        setSelectedCellFrames(null);
        setPreviewCellFrames(null, null);
      } else {
        setPredError(data.message ?? data.error ?? 'Prediction failed');
      }
    } catch (e: unknown) {
      setPredError(e instanceof Error ? e.message : 'Network error');
    } finally {
      setPredLoading(false);
    }
  }, [params, simYears, gridSize, setMlPrediction, setPreviewCellFrames]);

  const handleApplyAndRun = useCallback(() => {
    if (!mlPrediction) return;
    const newMm   = mlPrediction.mmGrid[mlMassIdx];
    const amRange = mlPrediction.validAmPerMm[mlMassIdx];
    setParam('mm_earth', newMm);
    if (amRange) setParam('am_hill', (amRange[0] + amRange[1]) / 2);
    onApplyAndRun();
  }, [mlPrediction, mlMassIdx, setParam, onApplyAndRun]);

  // ── Prediction: Layer 2 Trajectory ────────────────────────────────────────
  const handleTrajBatch = useCallback(async (forceRefresh = false) => {
    setTrajLoading(true);
    setTrajError(null);
    setSelectedCell(null);
    setSelectedCellFrames(null);
    setPreviewCellFrames(null, null);
    try {
      const res = await fetch(`${AGENT_DIRECT}/trajectory/preview`, {
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
          // HNN gate eval was standardised at 10yr; default to 10 when no sim has been run
          t_sim:           simYears > 0 ? simYears : (trajEngine === 'hnn_hinge4' ? 10.0 : 1.0),
          moon_retrograde: params.moon_retrograde,
          em:              params.em,
          mm_resolution:   gridSize,
          am_resolution:   gridSize,
          escape_factor:   1.0,
          mode:            trajEngine,
          model_version:   'hinge4_v1',
          force_refresh:   forceRefresh,
        }),
      });
      const rawText = await res.text();
      let data: Record<string, unknown>;
      try {
        data = JSON.parse(rawText) as Record<string, unknown>;
      } catch {
        throw new Error(`Server error (HTTP ${res.status}): ${rawText.slice(0, 300)}`);
      }
      if (data.ok || data.map_both) {
        const result = data as TrajResult;
        // Compute confidence_map client-side for HNN mode:
        // HIGH = MLP eligible AND HNN agrees (map_both true)
        // LOW  = MLP eligible AND HNN disagrees (map_both false)
        if (trajEngine === 'hnn_hinge4' && mlPrediction) {
          const N_MM = result.mm_grid.length;
          const N_AM = result.am_grid.length;
          result.confidence_map = Array.from({ length: N_MM }, (_, mi) =>
            Array.from({ length: N_AM }, (_, ai) => {
              const mlBoth = mlPrediction.mapBoth[mi]?.[ai] ?? false;
              if (!mlBoth) return null;
              return (result.map_both[mi]?.[ai] ?? false) ? 'HIGH' as const : 'LOW' as const;
            })
          );
        }
        setTrajResult(result);
      } else {
        setTrajError(data.message ?? data.error ?? 'Trajectory batch failed');
      }
    } catch (e: unknown) {
      setTrajError(e instanceof Error ? e.message : 'Network error');
    } finally {
      setTrajLoading(false);
    }
  }, [params, simYears, gridSize, trajEngine, mlPrediction, setPreviewCellFrames]);

  const handleCellApplyAndRun = useCallback(() => {
    if (!trajResult || !selectedCell || !selectedCellFrames) return;
    setParam('mm_earth', trajResult.mm_grid[selectedCell.mmIdx]);
    setParam('am_hill',  trajResult.am_grid[selectedCell.amIdx]);
    // Inject HNN/GT frames directly — no new simulation needed.
    // Compute meta inline (avoids TDZ: useMemo values declared after this callback).
    const tSim = simYears > 0 ? simYears : (trajEngine === 'hnn_hinge4' ? 10.0 : 1.0);
    const M_EARTH_MSUN = 3.003e-6;
    const localRhill = params.ap_AU * (1 - params.ep) *
      Math.cbrt(params.mp_earth * M_EARTH_MSUN / (3 * params.ms_solar));
    const Lrel = params.rs_solar ** 2 * (params.Ts / 5778) ** 4;
    const meta: SimulationMeta = {
      dt:         tSim / Math.max(selectedCellFrames.length - 1, 1),
      t_end:      tSim,
      a_inner_au: Math.sqrt(Lrel / 1.1),
      a_outer_au: Math.sqrt(Lrel / 0.5),
      rhill_AU:   localRhill,
    };
    setTrajectoryData(selectedCellFrames, meta);
  }, [trajResult, selectedCell, selectedCellFrames, setParam, setTrajectoryData,
      simYears, trajEngine, params]);

  // ── Trajectory grid callbacks ──────────────────────────────────────────────
  const downloadGridPNG = useCallback(() => {
    const c = gridCanvasRef.current;
    if (!c) return;
    const a = document.createElement('a');
    a.download = 'trajectory_grid.png';
    a.href = c.toDataURL('image/png');
    a.click();
  }, []);

  const handleGridClick = useCallback(async (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!trajResult) return;
    const canvas = gridCanvasRef.current;
    if (!canvas) return;
    const rect   = canvas.getBoundingClientRect();
    const scaleX = canvas.width  / rect.width;
    const scaleY = canvas.height / rect.height;
    const rawX   = (e.clientX - rect.left) * scaleX;
    const rawY   = (e.clientY - rect.top)  * scaleY;
    // Ignore clicks in the axis label strips
    const cellX  = rawX - LEFT_LABEL_PX;
    if (cellX < 0) return;
    const N_AM   = trajResult.am_grid.length;
    if (rawY >= N_AM * CELL_PX) return;
    const mmIdx  = Math.min(Math.max(Math.floor(cellX / CELL_PX), 0), trajResult.mm_grid.length - 1);
    const amIdx  = Math.max(0, Math.min(N_AM - 1 - Math.floor(rawY / CELL_PX), N_AM - 1));
    // Only MLP-valid cells (green) are interactive
    const isMLPValid = mlPrediction?.mapBoth[mmIdx]?.[amIdx] ?? false;
    if (!isMLPValid) return;

    // Toggle off: clicking same cell again deselects
    const isToggleOff = selectedCell?.mmIdx === mmIdx && selectedCell?.amIdx === amIdx;
    if (isToggleOff) {
      setSelectedCell(null);
      setSelectedCellFrames(null);
      setPreviewCellFrames(null, null);
      return;
    }

    setSelectedCell({ mmIdx, amIdx });
    setMlMassIdx(mmIdx);

    const amHill      = trajResult.am_grid[amIdx];
    const rocheFrac   = trajResult.am_grid[0];
    const M_EARTH_MSUN = 3.003e-6;
    const rhillAULocal = params.ap_AU * (1 - params.ep) *
      Math.cbrt(params.mp_earth * M_EARTH_MSUN / (3 * params.ms_solar));

    // Clear old frames and start loading actual physics trajectory
    setSelectedCellFrames(null);
    setPreviewCellFrames(null, rocheFrac, rhillAULocal);
    setCellTrajError(null);
    setCellTrajLoading(true);

    try {
      const res = await fetch(`${AGENT_URL}/trajectory/cell_preview`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          system_params: {
            ms_solar: params.ms_solar,
            rs_solar: params.rs_solar,
            Ts:       params.Ts,
            mp_earth: params.mp_earth,
            ap_AU:    params.ap_AU,
            ep:       params.ep,
          },
          mm_idx:          mmIdx,
          am_idx:          amIdx,
          mm_resolution:   trajResult.mm_grid.length,   // actual grid size from batch result
          am_resolution:   trajResult.am_grid.length,
          mm_earth:        trajResult.mm_grid[mmIdx],
          am_hill:         amHill,
          t_sim:           simYears > 0 ? simYears : (trajEngine === 'hnn_hinge4' ? 10.0 : 1.0),
          moon_retrograde: params.moon_retrograde,
          em:              params.em,
          escape_factor:   1.0,
          mode:            trajEngine,
          model_version:   'hinge4_v1',
          cache_key:       trajResult.cache_key,   // exact key from server — skips reconstruction
        }),
      });
      if (res.status === 503) {
        const errData = await res.json().catch(() => ({})) as Record<string, unknown>;
        setCellTrajError(String(errData.detail ?? 'GPU service unavailable — try again shortly'));
        return;
      }
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const frames: TrajectoryFrame[] = data.frames;
      setSelectedCellFrames(frames);
      setPreviewCellFrames(frames, rocheFrac, rhillAULocal);

      // Route frames through trajectoryFrames so useOrbitScene drives animation
      // — same mechanism as the non-ML "Run" path (external frameIndex, no internal RAF).
      const tSim = simYears > 0 ? simYears : (trajEngine === 'hnn_hinge4' ? 10.0 : 1.0);
      const Lrel = params.rs_solar ** 2 * (params.Ts / 5778) ** 4;
      const meta: SimulationMeta = {
        dt:         tSim / Math.max(frames.length - 1, 1),
        t_end:      tSim,
        a_inner_au: Math.sqrt(Lrel / 1.1),
        a_outer_au: Math.sqrt(Lrel / 0.5),
        rhill_AU:   rhillAULocal,
      };
      setTrajectoryData(frames, meta);
    } catch (err) {
      console.error('[MlMapOverlay] Cell trajectory fetch failed:', err);
      setCellTrajError('Failed to load trajectory');
    } finally {
      setCellTrajLoading(false);
    }
  }, [trajResult, selectedCell, mlPrediction, params, simYears, trajEngine, setMlMassIdx, setPreviewCellFrames, setTrajectoryData]);

  // ── Lifecycle ──────────────────────────────────────────────────────────────
  useEffect(() => {
    setHistory(null);
    setHistError(null);
    fetchHistory(perfLayer);
  }, [perfLayer, fetchHistory]);

  // Physics constants — declared BEFORE canvas effects so dep arrays don't hit TDZ
  const rhillAU = useMemo(() => {
    const M_EARTH_MSUN = 3.003e-6;
    return params.ap_AU * (1 - params.ep) *
      Math.cbrt(params.mp_earth * M_EARTH_MSUN / (3 * params.ms_solar));
  }, [params.ap_AU, params.ep, params.mp_earth, params.ms_solar]);

  const hzInnerAU = useMemo(() => {
    const Lrel = params.rs_solar ** 2 * (params.Ts / 5778) ** 4;
    return Math.sqrt(Lrel / 1.1);
  }, [params.rs_solar, params.Ts]);

  const hzOuterAU = useMemo(() => {
    const Lrel = params.rs_solar ** 2 * (params.Ts / 5778) ** 4;
    return Math.sqrt(Lrel / 0.5);
  }, [params.rs_solar, params.Ts]);

  // ── Grid canvas draw effect ────────────────────────────────────────────────
  useEffect(() => {
    const canvas = gridCanvasRef.current;
    if (!canvas || !trajResult) return;
    const N_MM = trajResult.mm_grid.length;
    const N_AM = trajResult.am_grid.length;
    canvas.width  = N_MM * CELL_PX + LEFT_LABEL_PX;
    canvas.height = N_AM * CELL_PX + BOT_LABEL_PX;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const halfCell  = CELL_PX / 2;
    const scalePx   = (halfCell - 4) / 1.0;
    const rocheFrac = trajResult.am_grid[0];
    const hzInFrac  = rhillAU > 0 ? hzInnerAU / rhillAU : 9999;
    const hzOutFrac = rhillAU > 0 ? hzOuterAU / rhillAU : 9999;
    const isHNN     = trajEngine === 'hnn_hinge4';
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Background for label strips
    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw cells
    for (let mi = 0; mi < N_MM; mi++) {
      for (let ai = 0; ai < N_AM; ai++) {
        const row  = N_AM - 1 - ai;
        const x0   = mi  * CELL_PX + LEFT_LABEL_PX;
        const y0   = row * CELL_PX;
        const cx   = x0 + halfCell;
        const cy   = y0 + halfCell;
        // MLP Layer 1 classification drives green/grey coloring (canonical source)
        const mlBoth = mlPrediction?.mapBoth[mi]?.[ai] ?? false;
        const conf   = trajResult.confidence_map?.[mi]?.[ai] ?? null;
        // Background: HNN LOW = dark red, HNN HIGH or GT = dark green, invalid = black
        const isLow  = isHNN && mlBoth && conf === 'LOW';
        ctx.fillStyle = mlBoth ? (isLow ? '#2d0a0a' : '#0d3830') : '#0d1117';
        ctx.fillRect(x0, y0, CELL_PX, CELL_PX);
        // Grid lines
        ctx.strokeStyle = '#1a2535';
        ctx.lineWidth = 0.5;
        ctx.strokeRect(x0 + 0.25, y0 + 0.25, CELL_PX - 0.5, CELL_PX - 0.5);
        // HZ shading (green annulus)
        if (hzInFrac < hzOutFrac) {
          const ri = Math.min(hzInFrac  * scalePx, halfCell - 2);
          const ro = Math.min(hzOutFrac * scalePx, halfCell - 2);
          if (ro > 0 && ro > ri) {
            ctx.save();
            ctx.globalAlpha = 0.12;
            ctx.beginPath();
            ctx.arc(cx, cy, ro, 0, Math.PI * 2);
            ctx.arc(cx, cy, Math.max(ri, 0), 0, Math.PI * 2, true);
            ctx.fillStyle = '#22c55e';
            ctx.fill();
            ctx.restore();
          }
        }
        // Hill sphere ring (white dashed — outer physical limit)
        ctx.save();
        ctx.strokeStyle = 'rgba(255,255,255,0.35)';
        ctx.lineWidth = 0.5;
        ctx.setLineDash([2, 2]);
        ctx.beginPath();
        ctx.arc(cx, cy, scalePx, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.restore();
        // Roche limit ring (red dashed)
        if (rocheFrac * scalePx > 0) {
          ctx.save();
          ctx.strokeStyle = 'rgba(239,68,68,0.55)';
          ctx.lineWidth = 0.5;
          ctx.setLineDash([1, 2]);
          ctx.beginPath();
          ctx.arc(cx, cy, rocheFrac * scalePx, 0, Math.PI * 2);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.restore();
        }
        // Moon orbit circle — red for HNN LOW, teal for valid, grey for invalid
        const moonR = trajResult.am_grid[ai] * scalePx;
        if (moonR > 0 && moonR <= scalePx + 1) {
          ctx.save();
          ctx.strokeStyle = !mlBoth
            ? 'rgba(100,116,139,0.35)'
            : isLow ? 'rgba(239,68,68,0.70)' : 'rgba(20,184,166,0.65)';
          ctx.lineWidth = 0.7;
          ctx.beginPath();
          ctx.arc(cx, cy, moonR, 0, Math.PI * 2);
          ctx.stroke();
          ctx.restore();
        }
        // HNN LOW hatching — red diagonal lines reinforce the red background
        if (isLow) {
          ctx.save();
          ctx.globalAlpha = 0.20;
          ctx.strokeStyle = 'rgba(239,68,68,1)';
          ctx.lineWidth = 0.5;
          ctx.beginPath();
          for (let h = 0; h < CELL_PX * 2; h += 5) {
            ctx.moveTo(x0 + h - CELL_PX, y0);
            ctx.lineTo(x0, y0 + h);
          }
          ctx.stroke();
          ctx.restore();
        }
        // Planet dot
        ctx.fillStyle = '#e2e8f0';
        ctx.beginPath();
        ctx.arc(cx, cy, 1.5, 0, Math.PI * 2);
        ctx.fill();
        // Selection highlight
        if (selectedCell?.mmIdx === mi && selectedCell?.amIdx === ai) {
          ctx.save();
          ctx.strokeStyle = '#c084fc';
          ctx.lineWidth = 2;
          ctx.setLineDash([]);
          ctx.strokeRect(x0 + 1, y0 + 1, CELL_PX - 2, CELL_PX - 2);
          ctx.restore();
        }
      }
    }

    // ── Axis tick labels drawn directly on canvas ─────────────────────────────
    ctx.font = '9px monospace';
    ctx.fillStyle = '#4b5563';

    // am (y-axis): draw ticks every 5 rows in the left strip
    const AM_TICK_STEP = Math.max(1, Math.floor(N_AM / 10));
    for (let ai = 0; ai < N_AM; ai += AM_TICK_STEP) {
      const row = N_AM - 1 - ai;
      const labelY = row * CELL_PX + halfCell + 3;
      ctx.textAlign = 'right';
      ctx.fillText(trajResult.am_grid[ai].toFixed(2), LEFT_LABEL_PX - 3, labelY);
    }

    // mm (x-axis): draw ticks every 5 cols in the bottom strip
    const MM_TICK_STEP = Math.max(1, Math.floor(N_MM / 10));
    const botY = N_AM * CELL_PX + 14;
    for (let mi = 0; mi < N_MM; mi += MM_TICK_STEP) {
      const labelX = mi * CELL_PX + LEFT_LABEL_PX + halfCell;
      ctx.textAlign = 'center';
      ctx.fillText(trajResult.mm_grid[mi].toFixed(2), labelX, botY);
    }

    // Axis label text
    ctx.fillStyle = '#374151';
    ctx.textAlign = 'left';
    ctx.font = '8px monospace';
    ctx.fillText('am↑', 2, 10);
    ctx.fillText('mm→', LEFT_LABEL_PX + 2, N_AM * CELL_PX + 20);

  }, [trajResult, selectedCell, mlPrediction, rhillAU, hzInnerAU, hzOuterAU, trajEngine]);


  // ── Derived heatmap data ───────────────────────────────────────────────────
  const heatmapZ = mlPrediction
    ? mlPrediction.amGrid.map((_, j) =>
        mlPrediction.mmGrid.map((_, i) => (mlPrediction.mapBoth[i][j] ? 1 : 0))
      )
    : null;

  const curMm      = mlPrediction?.mmGrid[mlMassIdx] ?? null;
  const curAmRange = mlPrediction?.validAmPerMm[mlMassIdx] ?? null;
  const validMmR   = mlPrediction?.validMmRange ?? null;

  // ── Render ─────────────────────────────────────────────────────────────────
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
      {/* Header */}
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

      {/* Scrollable body */}
      <div className="flex-1 min-h-0 overflow-y-auto text-xs text-gray-300 divide-y divide-gray-800/60">

        {/* ══ Section 1: Model Training ═══════════════════════════════════════ */}
        <SectionHeader title="Model Training" open={sec1Open} onToggle={() => setSec1Open(v => !v)} />
        {sec1Open && (
          <div className="p-3 space-y-3">
            <div className="flex gap-1 bg-gray-800/50 rounded p-0.5">
              <LayerTab label="Layer 1 — MLP" active={trainLayer === 'mlp'} onClick={() => setTrainLayer('mlp')} />
              <LayerTab label="Layer 2 — HNN" active={trainLayer === 'hnn'} onClick={() => setTrainLayer('hnn')} />
            </div>

            {trainLayer === 'mlp' && (
              <div className="space-y-2 rounded bg-gray-800/40 p-2.5">
                <ReadField label="Model type"    value="MLP (Multi-Layer Perceptron)" />
                <ReadField label="Hidden layers" value="3 × 64 neurons" />
                <ReadField label="Output"        value="Binary — stable / habitable" />
                <ReadField label="Loss function" value="BCEWithLogitsLoss" />
              </div>
            )}

            {trainLayer === 'hnn' && (
              <div className="space-y-2 rounded bg-gray-800/40 p-2.5">
                <ReadField label="Model type"    value="Hamiltonian Neural Network (HNN)" />
                <ReadField label="Hidden layers" value="3 × 256 neurons" />
                <ReadField label="Activation"    value="Tanh" />
                <ReadField label="Loss function" value="Log-MSE + Sign Hinge (λ=50)" />
                <ReadField label="Optimizer"     value="Adam + ReduceLROnPlateau" />
                <ReadField label="Epochs"        value="50" />
              </div>
            )}
          </div>
        )}

        {/* ══ Section 2: Prediction ═══════════════════════════════════════════ */}
        <SectionHeader title="Prediction" open={sec2Open} onToggle={() => setSec2Open(v => !v)} />
        {sec2Open && (
          <div className="p-3 space-y-3">
            {/* Layer tabs */}
            <div className="flex gap-1 bg-gray-800/50 rounded p-0.5">
              <LayerTab
                label="Layer 1 — MLP"
                active={predLayer === 'mlp'}
                onClick={() => setPredLayer('mlp')}
              />
              <LayerTab
                label="Layer 2 — Trajectory"
                active={predLayer === 'trajectory'}
                onClick={() => setPredLayer('trajectory')}
              />
            </div>

            {/* ─── Layer 1: MLP Classification ─── */}
            {predLayer === 'mlp' && (
              <>
                {/* Grid size selector */}
                <div className="flex items-center gap-2">
                  <span className="text-gray-500 text-[10px] uppercase tracking-wide shrink-0">
                    Grid size
                  </span>
                  <div className="flex gap-1 ml-auto">
                    {([30, 50] as const).map(s => (
                      <button
                        key={s}
                        onClick={() => setGridSize(s)}
                        disabled={predLoading}
                        className={cn(
                          'px-2 py-0.5 rounded text-[10px] font-mono transition-colors',
                          gridSize === s
                            ? 'bg-violet-700/70 text-violet-200'
                            : 'bg-gray-800 text-gray-500 hover:text-gray-300 disabled:opacity-40',
                        )}
                      >
                        {s}×{s}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Run button */}
                <button
                  onClick={handlePredict}
                  disabled={predLoading}
                  className={cn(
                    'w-full flex items-center justify-center gap-1.5 py-1.5 rounded text-xs font-medium transition-colors',
                    'bg-violet-600 hover:bg-violet-500 text-white',
                    predLoading && 'opacity-60 cursor-not-allowed',
                  )}
                >
                  {predLoading ? <Loader2 size={11} className="animate-spin" /> : <Brain size={11} />}
                  {predLoading ? 'Running inference…' : 'Run ML Prediction'}
                </button>

                {predError && (
                  <p className="text-red-400 text-[11px] text-center px-1">{predError}</p>
                )}

                {/* Heatmap */}
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
                            title:     { text: 'Moon mass (M⊕)', font: { size: 10 } },
                            type:      'log' as const,
                            gridcolor: '#1f2937', color: '#6b7280',
                          },
                          yaxis: {
                            title:     { text: 'am (Hill radii)', font: { size: 10 } },
                            gridcolor: '#1f2937', color: '#6b7280',
                          },
                          shapes: curMm != null ? [{
                            type: 'line' as const, xref: 'x' as const, yref: 'paper' as const,
                            x0: curMm, x1: curMm, y0: 0, y1: 1,
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

                {/* Mass slider + readout + Apply & Run */}
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

                    <div className="rounded bg-gray-800/60 p-2 space-y-1 text-[11px]">
                      {validMmR ? (
                        <p className="text-teal-400/90 font-mono">
                          Valid mass:&nbsp;{validMmR[0].toFixed(4)}–{validMmR[1].toFixed(4)} M⊕
                        </p>
                      ) : (
                        <p className="text-orange-400/80">No stable+habitable configurations found</p>
                      )}
                      {curAmRange ? (
                        <p className="text-cyan-400/90 font-mono">
                          Valid orbit:&nbsp;{curAmRange[0].toFixed(3)}–{curAmRange[1].toFixed(3)} Hill radii
                        </p>
                      ) : (
                        <p className="text-gray-500">No valid orbit at this mass</p>
                      )}
                    </div>

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
              </>
            )}

            {/* ─── Layer 2: Trajectory Preview ─── */}
            {predLayer === 'trajectory' && (
              <>
                {!mlPrediction ? (
                  <p className="text-gray-500 text-center text-[11px] py-2 leading-relaxed">
                    Run Layer 1 MLP Classification first to identify eligible cells for trajectory preview.
                  </p>
                ) : (
                  <>
                    {/* Engine selector */}
                    <div className="space-y-1.5">
                      <span className="text-gray-500 text-[10px] uppercase tracking-wide">Physics engine</span>
                      {([
                        { key: 'gt_leapfrog' as const, label: 'Ground Truth Physics Integrator', note: '~2.3s' },
                        { key: 'hnn_hinge4'  as const, label: 'HNN Physics ML Model',            note: '~470s + cached' },
                      ]).map(({ key, label, note }) => (
                        <button
                          key={key}
                          onClick={() => setTrajEngine(key)}
                          disabled={trajLoading}
                          className={cn(
                            'w-full flex items-center justify-between px-2.5 py-1.5 rounded text-[11px] border transition-colors text-left',
                            trajEngine === key
                              ? 'border-violet-600/60 bg-violet-900/20 text-violet-300'
                              : 'border-gray-700/60 bg-gray-800/40 text-gray-400 hover:border-gray-600',
                            trajLoading && 'opacity-50 cursor-not-allowed',
                          )}
                        >
                          <span>{label}</span>
                          <span className="text-gray-500 font-mono text-[10px] shrink-0 ml-2">{note}</span>
                        </button>
                      ))}
                    </div>

                    {/* Run trajectory batch */}
                    <button
                      onClick={() => handleTrajBatch(false)}
                      disabled={trajLoading}
                      className={cn(
                        'w-full flex items-center justify-center gap-1.5 py-1.5 rounded text-xs font-medium transition-colors',
                        'bg-violet-600 hover:bg-violet-500 text-white',
                        trajLoading && 'opacity-60 cursor-not-allowed',
                      )}
                    >
                      {trajLoading
                        ? <Loader2 size={11} className="animate-spin" />
                        : <Zap size={11} />}
                      {trajLoading
                        ? (trajEngine === 'hnn_hinge4' ? 'Running HNN inference… (~470s)' : 'Running batch integrator…')
                        : 'Run Trajectory Batch'}
                    </button>
                    {/* Force-refresh: bypasses S3 cache and re-calls EC2 for fresh HNN result */}
                    {trajResult && !trajLoading && (
                      <button
                        onClick={() => handleTrajBatch(true)}
                        className="w-full flex items-center justify-center gap-1 py-1 rounded text-[10px] font-medium transition-colors
                                   border border-amber-700/50 bg-amber-950/30 text-amber-400 hover:bg-amber-900/40"
                        title="Bypass S3 cache and force a fresh EC2 call — use if results look wrong"
                      >
                        ↺ Re-run (bypass cache)
                      </button>
                    )}

                    {trajError && (
                      <p className="text-red-400 text-[11px] text-center px-1">{trajError}</p>
                    )}

                    {/* Trajectory results — per-cell orbit canvas grid */}
                    {trajResult && (
                      <>
                        <div className="flex items-center justify-between">
                          <div className="space-y-0.5">
                            <p className="text-gray-500 text-[10px] uppercase tracking-wide">
                              Trajectory Results
                            </p>
                            {trajResult.from_cache && (
                              <p className="text-gray-600 text-[9px]">✓ from cache</p>
                            )}
                            {trajResult.wall_s != null && !trajResult.from_cache && (
                              <p className="text-gray-600 text-[9px]">{trajResult.wall_s.toFixed(1)}s</p>
                            )}
                          </div>
                          <button
                            onClick={downloadGridPNG}
                            className="flex items-center gap-0.5 px-1.5 py-0.5 rounded bg-gray-800/60
                                       hover:bg-gray-700 text-gray-500 hover:text-gray-300 text-[9px] transition-colors"
                            title="Export full grid as PNG"
                          >
                            <Download size={9} /> PNG
                          </button>
                        </div>

                        {/* Confidence legend — HNN mode only */}
                        {trajResult.confidence_map && (
                          <div className="flex items-center gap-4 text-[10px]">
                            <div className="flex items-center gap-1">
                              <div className="w-2.5 h-2.5 rounded-sm" style={{ background: '#0d3830' }} />
                              <span className="text-gray-400">HIGH — MLP+HNN agree</span>
                            </div>
                            <div className="flex items-center gap-1">
                              <div className="w-2.5 h-2.5 rounded-sm" style={{ background: '#2d0a0a' }} />
                              <span className="text-gray-400">LOW — HNN disagrees</span>
                            </div>
                          </div>
                        )}

                        {/* Grid — axis labels drawn on canvas, scroll together */}
                        <p className="text-[9px] text-gray-600 text-center">
                          mm → (Moon mass, M⊕) · am ↑ (Hill radii) · click cell to expand
                        </p>
                        <div
                          className="rounded overflow-auto border border-gray-800/60 bg-[#0d1117]"
                          style={{ maxHeight: 320 }}
                        >
                          <canvas
                            ref={gridCanvasRef}
                            onClick={handleGridClick}
                            className="cursor-pointer block"
                            style={{ imageRendering: 'pixelated' }}
                          />
                        </div>

                        {/* Selected cell — expanded orbit view */}
                        {selectedCell && (
                          <div className="rounded bg-gray-800/60 p-2.5 space-y-2.5">
                            <div className="flex items-center justify-between">
                              <span className="flex items-center gap-1 text-gray-500 uppercase text-[10px] tracking-wide">
                                Cell Detail
                                {cellTrajLoading && <Loader2 size={10} className="animate-spin text-violet-400" />}
                              </span>
                              <button
                                onClick={() => setSelectedCell(null)}
                                className="text-gray-600 hover:text-gray-400 text-[10px]"
                              >✕</button>
                            </div>

                            {/* MiniOrbitView — driven by same frameIndex as 3D canvas (non-ML mechanism) */}
                            {selectedCellFrames && trajResult ? (
                              <MiniOrbitView
                                frames={selectedCellFrames}
                                frameIndex={frameIndex}
                                className="relative mb-1"
                                showHillSphereRings={true}
                                rocheInnerFrac={trajResult.am_grid[0]}
                              />
                            ) : (
                              <div className="flex items-center justify-center rounded bg-gray-800/40 text-[10px]"
                                   style={{ height: 170 }}>
                                {cellTrajLoading
                                  ? <span className="text-gray-600">{trajEngine === 'hnn_hinge4' ? 'Running HNN…' : 'Running GT integrator…'}</span>
                                  : cellTrajError
                                    ? <span className="text-amber-400 text-center px-2">{cellTrajError}</span>
                                    : <span className="text-gray-600">Click a cell to load trajectory</span>}
                              </div>
                            )}

                            <div className="space-y-1.5 text-[11px]">
                              <p className="font-mono text-violet-300">
                                mm: {trajResult.mm_grid[selectedCell.mmIdx].toFixed(5)} M⊕
                              </p>
                              <p className="font-mono text-cyan-300">
                                am: {trajResult.am_grid[selectedCell.amIdx].toFixed(4)} Hill radii
                              </p>
                              {(() => {
                                // Cell is only reachable because MLP Layer 1 said valid
                                const trajBoth = trajResult.map_both[selectedCell.mmIdx]?.[selectedCell.amIdx] ?? false;
                                const conf     = trajResult.confidence_map?.[selectedCell.mmIdx]?.[selectedCell.amIdx];
                                const isGT     = trajEngine === 'gt_leapfrog';
                                return (
                                  <>
                                    <div className="text-[10px] text-teal-400">
                                      ✓ MLP Layer 1: Stable + habitable
                                    </div>
                                    {isGT && !trajBoth && (
                                      <div className="text-[10px] text-red-400">
                                        ⚠ Trajectory: Unstable or Escaped
                                      </div>
                                    )}
                                    {conf && (
                                      <span className={cn(
                                        'inline-block text-[10px] px-1.5 py-0.5 rounded mt-0.5',
                                        conf === 'HIGH'
                                          ? 'bg-teal-900/50 text-teal-400'
                                          : 'bg-red-950/60 text-red-400',
                                      )}>
                                        HNN {conf} confidence
                                      </span>
                                    )}
                                  </>
                                );
                              })()}
                            </div>

                            {/* Ring legend */}
                            <div className="flex flex-wrap gap-x-3 gap-y-1 text-[9px] text-gray-500">
                              <span><span className="text-red-500">─ ─</span> Roche limit</span>
                              <span><span className="text-gray-300 opacity-50">─ ─</span> Hill sphere</span>
                              <span><span className="text-green-500">■</span> HZ</span>
                              <span><span className="text-teal-500">○</span> Moon orbit</span>
                            </div>

                            <button
                              onClick={handleCellApplyAndRun}
                              disabled={!selectedCellFrames}
                              className="w-full flex items-center justify-center gap-1.5 py-1 rounded
                                         bg-teal-600 hover:bg-teal-500 text-white text-xs font-medium transition-colors
                                         disabled:opacity-40 disabled:cursor-not-allowed"
                              title={selectedCellFrames ? 'Render this cell\'s HNN trajectory in the 3D orbit view' : 'Load the cell trajectory first'}
                            >
                              <Play size={10} />
                              {trajEngine === 'hnn_hinge4' ? 'Export HNN trajectory → main view' : 'Export GT trajectory → main view'}
                            </button>
                          </div>
                        )}
                      </>
                    )}
                  </>
                )}
              </>
            )}
          </div>
        )}

        {/* ══ Section 3: Model Performance ════════════════════════════════════ */}
        <SectionHeader title="Model Performance" open={sec3Open} onToggle={() => setSec3Open(v => !v)} />
        {sec3Open && (
          <div className="p-3 space-y-2">
            {/* Layer selector */}
            <div className="flex gap-1 bg-gray-800/50 rounded p-0.5">
              <LayerTab label="MLP / Layer 1" active={perfLayer === 'mlp'} onClick={() => setPerfLayer('mlp')} />
              <LayerTab label="HNN / Layer 2" active={perfLayer === 'hnn'} onClick={() => setPerfLayer('hnn')} />
            </div>

            {!history ? (
              <div className="text-center py-4 space-y-1">
                <p className="text-gray-600 leading-relaxed">
                  No training history found for {perfLayer === 'mlp' ? 'MLP' : 'HNN'}.
                </p>
                {histError && (
                  <p className="text-red-500/70 text-[10px] font-mono px-2 leading-relaxed">
                    {histError}
                  </p>
                )}
              </div>
            ) : (
              <>
                {/* Loss curves */}
                <div className="flex items-center justify-between">
                  <p className="text-gray-500 text-[10px] uppercase tracking-wide">Loss Curves</p>
                  <button
                    onClick={() => downloadPlot(lossGdRef.current, `${perfLayer}_loss_curves`)}
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
                        title:     { text: 'Epoch', font: { size: 10 } },
                        gridcolor: '#1f2937', color: '#6b7280',
                      },
                      yaxis: {
                        title:     { text: 'Loss', font: { size: 10 } },
                        gridcolor: '#1f2937', color: '#6b7280',
                      },
                      legend: {
                        bgcolor: 'rgba(0,0,0,0)',
                        x: 0.98, xanchor: 'right',
                        y: 0.98, font: { size: 10 },
                      },
                    }}
                    config={{ displayModeBar: false, responsive: true }}
                    style={{ width: '100%', height: '100%' }}
                    useResizeHandler
                  />
                </div>

                {/* Flag accuracy — MLP only */}
                {perfLayer === 'mlp' && history.flag_accuracy && (
                  <>
                    <div className="flex items-center justify-between mt-2">
                      <p className="text-gray-500 text-[10px] uppercase tracking-wide">
                        Stable / Habitable Flag Accuracy
                      </p>
                      <button
                        onClick={() => downloadPlot(accGdRef.current, 'mlp_flag_accuracy')}
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
                            title:     { text: 'Epoch', font: { size: 10 } },
                            gridcolor: '#1f2937', color: '#6b7280',
                          },
                          yaxis: {
                            title:     { text: 'Accuracy (%)', font: { size: 10 } },
                            gridcolor: '#1f2937', color: '#6b7280',
                            range: [0, 100],
                          },
                          legend: {
                            bgcolor: 'rgba(0,0,0,0)',
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
                  </>
                )}

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
