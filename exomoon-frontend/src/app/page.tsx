'use client';
import React, { useRef, useState, useCallback, useEffect, memo } from 'react';
import {
  BarChart2, X, GripHorizontal, Sun, Globe, Moon,
  CheckCircle, Zap, Loader2, ChevronDown, ChevronUp, Maximize2, Brain,
} from 'lucide-react';
import { AppShell } from '@/components/layout/AppShell';
import { OrbitCanvas } from '@/components/animation/OrbitCanvas';
import { OrbitOverlay } from '@/components/animation/OrbitOverlay';
import { AnimationControls } from '@/components/animation/AnimationControls';
import { MiniOrbitView } from '@/components/animation/MiniOrbitView';
import { useOrbitScene } from '@/components/animation/useOrbitScene';
import type { BodyRadiiAU } from '@/components/animation/useOrbitScene';
import { EdaPanel } from '@/components/eda/EdaPanel';
import { MlMapOverlay } from '@/components/ml/MlMapOverlay';
import { ChatPanel } from '@/components/chat/ChatPanel';
import { StellarPanel } from '@/components/controls/StellarPanel';
import { PlanetPanel } from '@/components/controls/PlanetPanel';
import { MoonPanel } from '@/components/controls/MoonPanel';
import { NasaSearch } from '@/components/controls/NasaSearch';
import { useSimulationStore } from '@/hooks/useSimulationStore';
import { useJobPoller } from '@/hooks/useJobPoller';
import { agentApi, paramsToAgentFormat } from '@/lib/agentApi';
import { bodyRadiusAU } from '@/lib/trajectoryMath';
import { cn } from '@/lib/utils';
import type { ParamStatus } from '@/hooks/useSimulationStore';
import type { TrajectoryFrame, SimulationMeta } from '@/lib/types';

const RSUN_AU = 6.9598e8 / 1.495979e11;

// ── Stability + habitability badges (used in fullscreen mode) ────────────────
function FullscreenBadges({
  frame, meta,
}: { frame: TrajectoryFrame; meta: SimulationMeta }) {
  const moonPlanetDist = frame.moon_planet_dist ?? 0;
  const rhill = meta.rhill_AU ?? 0;
  const escaped = rhill > 0 && moonPlanetDist > rhill;
  const dx = frame.moon_x - frame.star_x;
  const dy = frame.moon_y - frame.star_y;
  const dz = frame.moon_z - frame.star_z;
  const moonStarDist = Math.sqrt(dx * dx + dy * dy + dz * dz);
  const inHZ = moonStarDist >= meta.a_inner_au && moonStarDist <= meta.a_outer_au;

  return (
    <div className="absolute top-3 left-3 z-10 space-y-1.5 pointer-events-none">
      <div className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-xs font-semibold tracking-wide ${
        escaped
          ? 'bg-red-900/70 text-red-300 border border-red-700/50'
          : 'bg-green-900/70 text-green-300 border border-green-700/50'
      }`}>
        <span className={`w-1.5 h-1.5 rounded-full ${escaped ? 'bg-red-400' : 'bg-green-400'}`} />
        {escaped ? 'Moon Escaped' : 'Stable'}
      </div>
      <div className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-xs font-semibold tracking-wide ${
        inHZ
          ? 'bg-emerald-900/70 text-emerald-300 border border-emerald-700/50'
          : 'bg-orange-900/70 text-orange-300 border border-orange-700/50'
      }`}>
        <span className={`w-1.5 h-1.5 rounded-full ${inHZ ? 'bg-emerald-400' : 'bg-orange-400'}`} />
        {inHZ ? 'Habitable' : 'Uninhabitable'}
      </div>
    </div>
  );
}

// ── Shared overlay props ─────────────────────────────────────────────────────
interface ObjectOverlayBaseProps {
  onClose: () => void;
  containerRef: React.RefObject<HTMLDivElement | null>;
}

// ── Reusable drag+resize logic ───────────────────────────────────────────────
function useDraggableResizable(containerRef: React.RefObject<HTMLDivElement | null>) {
  const [dragPos, setDragPos] = useState<{ left: number; top: number } | null>(null);
  const [panelWidth, setPanelWidth] = useState<number | null>(null);
  const panelRef = useRef<HTMLDivElement>(null);

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

  const handleResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const panel = panelRef.current;
    if (!panel) return;
    const startX = e.clientX;
    const startW = panel.getBoundingClientRect().width;
    function onMove(ev: MouseEvent) {
      setPanelWidth(Math.max(200, Math.min(600, startW + (ev.clientX - startX))));
    }
    function onUp() {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup',   onUp);
    }
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup',   onUp);
  }, []);

  const reset = useCallback(() => { setDragPos(null); setPanelWidth(null); }, []);

  return { panelRef, dragPos, panelWidth, handleDragStart, handleResizeStart, reset };
}

// ── Object overlay shell ─────────────────────────────────────────────────────
function ObjectOverlayShell({
  panelRef, dragPos, panelWidth, handleDragStart, handleResizeStart, reset,
  onClose, title, accentClass, headerBg, children,
}: {
  panelRef: React.RefObject<HTMLDivElement | null>;
  dragPos: { left: number; top: number } | null;
  panelWidth: number | null;
  handleDragStart: (e: React.MouseEvent) => void;
  handleResizeStart: (e: React.MouseEvent) => void;
  reset: () => void;
  onClose: () => void;
  title: React.ReactNode;
  accentClass: string;
  headerBg: string;
  children: React.ReactNode;
}) {
  return (
    <div
      ref={panelRef}
      className={cn(
        'absolute z-10 flex flex-col bg-gray-900/92 rounded-lg border overflow-hidden',
        accentClass,
        !dragPos    && 'top-36 left-1/2 -translate-x-1/2',
        !panelWidth && 'w-64',
      )}
      style={{
        ...(dragPos    ? { left: dragPos.left, top: dragPos.top } : {}),
        ...(panelWidth ? { width: panelWidth }                    : {}),
      }}
    >
      <div
        onMouseDown={handleDragStart}
        className={cn(
          'flex items-center justify-between px-3 py-1.5 border-b border-gray-700/40 shrink-0',
          'cursor-grab active:cursor-grabbing select-none', headerBg,
        )}
      >
        <div className="flex items-center gap-2">{title}</div>
        <div className="flex items-center gap-1.5">
          {(dragPos || panelWidth) && (
            <button onMouseDown={e => e.stopPropagation()} onClick={reset}
              className="text-gray-600 hover:text-gray-300 transition-colors text-[10px]"
              title="Reset position and size">↩</button>
          )}
          <button onMouseDown={e => e.stopPropagation()} onClick={onClose}
            className="text-gray-500 hover:text-white transition-colors">
            <X size={14} />
          </button>
        </div>
      </div>
      <div className="relative flex-1 min-h-0 overflow-auto">
        {children}
        <div onMouseDown={handleResizeStart}
          className="absolute top-0 right-0 bottom-0 w-1 cursor-ew-resize hover:bg-white/10 transition-colors"
          title="Drag to resize" />
      </div>
    </div>
  );
}

// ── Three memo'd object overlays ─────────────────────────────────────────────
const StellarOverlay = memo(function StellarOverlay({ onClose, containerRef }: ObjectOverlayBaseProps) {
  const dr = useDraggableResizable(containerRef);
  return (
    <ObjectOverlayShell {...dr} onClose={onClose} accentClass="border-yellow-700/30" headerBg="bg-yellow-950/40"
      title={<><Sun size={12} className="text-yellow-400" /><span className="text-xs text-yellow-400/80 font-medium tracking-wide uppercase">Stellar</span></>}>
      <StellarPanel />
    </ObjectOverlayShell>
  );
});

const PlanetOverlay = memo(function PlanetOverlay({ onClose, containerRef }: ObjectOverlayBaseProps) {
  const dr = useDraggableResizable(containerRef);
  return (
    <ObjectOverlayShell {...dr} onClose={onClose} accentClass="border-blue-700/30" headerBg="bg-blue-950/40"
      title={<><Globe size={12} className="text-blue-400" /><span className="text-xs text-blue-400/80 font-medium tracking-wide uppercase">Planet</span></>}>
      <PlanetPanel />
    </ObjectOverlayShell>
  );
});

const MoonOverlay = memo(function MoonOverlay({ onClose, containerRef }: ObjectOverlayBaseProps) {
  const dr = useDraggableResizable(containerRef);
  return (
    <ObjectOverlayShell {...dr} onClose={onClose} accentClass="border-red-700/30" headerBg="bg-red-950/40"
      title={<><Moon size={12} className="text-red-400" /><span className="text-xs text-red-400/80 font-medium tracking-wide uppercase">Moon</span></>}>
      <MoonPanel />
    </ObjectOverlayShell>
  );
});

// ── EDA overlay ──────────────────────────────────────────────────────────────
const EdaOverlay = memo(function EdaOverlay({ onClose, containerRef }: ObjectOverlayBaseProps) {
  const [dragPos, setDragPos] = useState<{ left: number; top: number } | null>(null);
  const panelRef = useRef<HTMLDivElement>(null);

  const handleDragStart = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    e.preventDefault();
    const panel = panelRef.current;
    const container = containerRef.current;
    if (!container || !panel) return;
    const pRect = panel.getBoundingClientRect();
    const offsetX = e.clientX - pRect.left;
    const offsetY = e.clientY - pRect.top;
    function onMove(ev: MouseEvent) {
      const cr = containerRef.current?.getBoundingClientRect();
      const pr = panelRef.current?.getBoundingClientRect();
      if (!cr || !pr) return;
      setDragPos({
        left: Math.max(0, Math.min(ev.clientX - cr.left - offsetX, cr.width  - pr.width)),
        top:  Math.max(0, Math.min(ev.clientY - cr.top  - offsetY, cr.height - pr.height)),
      });
    }
    function onUp() {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup',   onUp);
    }
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup',   onUp);
  }, [containerRef]);

  return (
    <div ref={panelRef}
      className={cn(
        'absolute z-10 flex flex-col w-[min(820px,68%)] max-h-[55%]',
        'bg-gray-900/90 rounded-lg border border-gray-700/40 overflow-hidden',
        !dragPos && 'top-3 left-1/2 -translate-x-1/2',
      )}
      style={dragPos ? { left: dragPos.left, top: dragPos.top } : undefined}
    >
      <div onMouseDown={handleDragStart}
        className="flex items-center justify-between px-3 py-1.5 border-b border-gray-700/40 shrink-0
                   cursor-grab active:cursor-grabbing select-none"
      >
        <div className="flex items-center gap-2">
          <GripHorizontal size={12} className="text-gray-500" />
          <span className="text-xs text-gray-400 font-medium tracking-wide uppercase">EDA — Exploratory Data Analysis</span>
        </div>
        <div className="flex items-center gap-2">
          {dragPos && (
            <button onMouseDown={e => e.stopPropagation()} onClick={() => setDragPos(null)}
              className="text-gray-600 hover:text-gray-300 transition-colors text-[10px]"
              title="Reset to default position">↩</button>
          )}
          <button onMouseDown={e => e.stopPropagation()} onClick={onClose}
            className="text-gray-500 hover:text-white transition-colors">
            <X size={14} />
          </button>
        </div>
      </div>
      <div className="flex-1 min-h-0 overflow-auto"><EdaPanel /></div>
    </div>
  );
});

// ── FAB status icon ──────────────────────────────────────────────────────────
function StatusIcon({ status, isRunning }: { status: ParamStatus; isRunning: boolean }) {
  if (isRunning)          return <Loader2    size={10} className="animate-spin text-blue-400 shrink-0" />;
  if (status === 'clean') return <CheckCircle size={10} className="text-green-400 shrink-0" />;
  if (status === 'dirty') return <Zap         size={10} className="text-amber-400 shrink-0" />;
  return null;
}

// ────────────────────────────────────────────────────────────────────────────

export default function HomePage() {
  useJobPoller();

  const [showEda,        setShowEda]        = useState(false);
  const [showStar,       setShowStar]       = useState(false);
  const [showPlanet,     setShowPlanet]     = useState(false);
  const [showMoon,       setShowMoon]       = useState(false);
  const [showMl,         setShowMl]         = useState(false);
  const [showFullscreen, setShowFullscreen] = useState(false);
  const [stripCollapsed, setStripCollapsed] = useState(false);

  // Status message with smooth fade-out
  const [statusMsg,  setStatusMsg]  = useState<string | null>(null);
  const [statusFade, setStatusFade] = useState(false);
  const statusTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Playback panel drag + resize
  const [playbackPos,   setPlaybackPos]   = useState<{ left: number; top: number } | null>(null);
  const [playbackWidth, setPlaybackWidth] = useState<number | null>(null);
  const containerRef     = useRef<HTMLDivElement>(null);
  const playbackPanelRef = useRef<HTMLDivElement>(null);

  const {
    params, simYears, setSimYears,
    trajectoryFrames, simMeta,
    setJob, jobStatus,
    dmCgs,
    starStatus, planetStatus, moonStatus,
  } = useSimulationStore();

  const isRunning = jobStatus === 'running';

  const bodyRadii: BodyRadiiAU = {
    star:   params.rs_solar * RSUN_AU,
    planet: bodyRadiusAU(params.mp_earth, params.dp_cgs),
    moon:   bodyRadiusAU(params.mm_earth, dmCgs),
  };

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sceneControls = useOrbitScene(canvasRef, trajectoryFrames, simMeta, bodyRadii);

  const currentFrame = trajectoryFrames
    ? trajectoryFrames[sceneControls.frameIndex] ?? null
    : null;

  // ── Esc to exit fullscreen ───────────────────────────────────────────────
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && showFullscreen) setShowFullscreen(false);
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [showFullscreen]);

  // ── Status message helpers ───────────────────────────────────────────────
  const clearStatusTimers = useCallback(() => {
    if (statusTimerRef.current) { clearTimeout(statusTimerRef.current); statusTimerRef.current = null; }
  }, []);

  const showStatus = useCallback((text: string) => {
    clearStatusTimers();
    setStatusFade(false);
    setStatusMsg(text);
  }, [clearStatusTimers]);

  const scheduleStatusFade = useCallback(() => {
    clearStatusTimers();
    statusTimerRef.current = setTimeout(() => {
      setStatusFade(true);
      statusTimerRef.current = setTimeout(() => {
        setStatusMsg(null);
        setStatusFade(false);
      }, 1000);
    }, 7000);
  }, [clearStatusTimers]);

  useEffect(() => {
    if      (jobStatus === 'running')   showStatus('Simulation running…');
    else if (jobStatus === 'succeeded') { showStatus('Simulation complete — scene updated'); scheduleStatusFade(); }
    else if (jobStatus === 'failed')    { showStatus('Job failed or timed out');             scheduleStatusFade(); }
  }, [jobStatus, showStatus, scheduleStatusFade]);

  const handleRun = useCallback(async () => {
    showStatus('Starting simulation…');
    try {
      const result = await agentApi.submitJob(paramsToAgentFormat(params), simYears);
      if (result.ok && result.job_id) {
        setJob(result.job_id);
      } else {
        showStatus('Job submission failed');
        scheduleStatusFade();
      }
    } catch {
      showStatus('Failed to reach agent service');
      scheduleStatusFade();
    }
  }, [params, simYears, setJob, showStatus, scheduleStatusFade]);

  // ── Playback drag ────────────────────────────────────────────────────────
  const handlePlaybackDragStart = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    e.preventDefault();
    const container = containerRef.current;
    const panel = playbackPanelRef.current;
    if (!container || !panel) return;
    const pRect = panel.getBoundingClientRect();
    const offsetX = e.clientX - pRect.left;
    const offsetY = e.clientY - pRect.top;
    function onMove(ev: MouseEvent) {
      const cr = container!.getBoundingClientRect();
      const pr = panel!.getBoundingClientRect();
      setPlaybackPos({
        left: Math.max(0, Math.min(ev.clientX - cr.left - offsetX, cr.width  - pr.width)),
        top:  Math.max(0, Math.min(ev.clientY - cr.top  - offsetY, cr.height - pr.height)),
      });
    }
    function onUp() { window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp); }
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup',   onUp);
  }, []);

  // ── Playback resize ──────────────────────────────────────────────────────
  const handlePlaybackResizeStart = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    const panel = playbackPanelRef.current;
    if (!panel) return;
    const startX = e.clientX;
    const startW = panel.getBoundingClientRect().width;
    function onMove(ev: MouseEvent) { setPlaybackWidth(Math.max(280, Math.min(900, startW + (ev.clientX - startX)))); }
    function onUp() { window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp); }
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup',   onUp);
  }, []);

  const handleCloseEda    = useCallback(() => setShowEda(false),    []);
  const handleCloseStar   = useCallback(() => setShowStar(false),   []);
  const handleClosePlanet = useCallback(() => setShowPlanet(false), []);
  const handleCloseMoon   = useCallback(() => setShowMoon(false),   []);
  const handleCloseMl     = useCallback(() => setShowMl(false),     []);

  const mainContent = (
    <div
      ref={containerRef}
      className={cn(
        'relative w-full h-full overflow-hidden',
        showFullscreen && 'fixed inset-0 z-50 bg-gray-950',
      )}
    >
      {/* Canvas — always fills full area */}
      <OrbitCanvas
        canvasRef={canvasRef}
        hasFrames={!!trajectoryFrames}
        className="absolute inset-0 w-full h-full"
      />

      {/* ── Normal-mode-only UI ──────────────────────────────────────────────── */}
      {!showFullscreen && (
        <>
          {/* Data readouts — top-left */}
          <div className="absolute top-0 left-0 z-10 pointer-events-none">
            <OrbitOverlay
              frame={currentFrame}
              frameIndex={sceneControls.frameIndex}
              totalFrames={sceneControls.totalFrames}
              meta={simMeta}
            />
          </div>

          {/* ── Top-center collapsible control strip ─────────────────────────── */}
          <div className={cn(
            'absolute top-3 left-1/2 -translate-x-1/2 z-20 w-72',
            'bg-black/50 border border-gray-700/40 backdrop-blur-sm rounded-lg px-3',
            stripCollapsed ? 'py-1.5' : 'py-2',
          )}>
            {/* NASA search row — always visible; chevron toggle on right */}
            <div className="flex items-center gap-1.5">
              <div className="flex-1 min-w-0">
                <NasaSearch hideLabel />
              </div>
              <button
                onClick={() => setStripCollapsed(v => !v)}
                className="shrink-0 text-gray-500 hover:text-gray-300 transition-colors"
                title={stripCollapsed ? 'Expand controls' : 'Collapse controls'}
              >
                {stripCollapsed ? <ChevronDown size={14} /> : <ChevronUp size={14} />}
              </button>
            </div>

            {/* Collapsible section */}
            {!stripCollapsed && (
              <>
                {/* Sim years + Run */}
                <div className="flex items-center gap-2 mt-1.5">
                  <span className="text-xs text-gray-500 shrink-0">Sim years</span>
                  <input
                    type="number"
                    min={0} step="any"
                    value={simYears}
                    onChange={e => setSimYears(parseFloat(e.target.value) || 0)}
                    disabled={isRunning}
                    className={cn(
                      'w-20 px-2 py-0.5 text-xs rounded bg-gray-800 border border-gray-700',
                      'text-blue-300 font-mono focus:outline-none focus:border-blue-500',
                      isRunning && 'opacity-40 cursor-not-allowed'
                    )}
                  />
                  <span className="text-xs text-gray-600 shrink-0">(0=1 orbit)</span>
                  <button
                    onClick={handleRun}
                    disabled={isRunning}
                    className={cn(
                      'ml-auto flex items-center gap-1 px-2.5 py-0.5 rounded text-xs font-medium transition-colors',
                      'bg-blue-600 hover:bg-blue-500 text-white',
                      isRunning && 'opacity-50 cursor-not-allowed'
                    )}
                  >
                    {isRunning ? <Loader2 size={10} className="animate-spin" /> : '▶'}
                    {isRunning ? 'Running' : 'Run'}
                  </button>
                </div>

                {/* Status message */}
                {statusMsg && (
                  <p className={cn(
                    'text-xs text-center font-mono mt-1 transition-opacity duration-1000',
                    jobStatus === 'succeeded' ? 'text-green-400'
                      : jobStatus === 'failed' ? 'text-red-400'
                      : 'text-yellow-400',
                    statusFade && 'opacity-0',
                  )}>
                    {statusMsg}
                  </p>
                )}
              </>
            )}
          </div>

          {/* ── Top-right: legend + FABs ──────────────────────────────────────── */}
          <div className="absolute top-3 right-3 z-10 flex flex-col items-end gap-1.5">
            {trajectoryFrames && (
              <div className="pointer-events-none bg-black/45 rounded px-2.5 py-2 space-y-1.5 border border-gray-700/40">
                <LegendRow color="#FFDD00" label="Star" />
                <LegendRow color="#4488FF" label="Planet" />
                <LegendRow color="#FF5555" label="Moon" />
              </div>
            )}
            <button onClick={() => setShowStar(v => !v)}
              className={cn('flex items-center gap-1.5 px-2.5 py-1.5 rounded text-xs font-medium border transition-colors',
                showStar ? 'bg-yellow-600/30 text-yellow-300 border-yellow-600/50'
                         : 'bg-black/50 text-gray-400 hover:text-yellow-300 hover:bg-gray-800/70 border-gray-700/60')}
              title="Toggle stellar parameters">
              <Sun size={12} className="text-yellow-400 shrink-0" />
              <span>Star</span>
              <StatusIcon status={starStatus} isRunning={isRunning} />
            </button>
            <button onClick={() => setShowPlanet(v => !v)}
              className={cn('flex items-center gap-1.5 px-2.5 py-1.5 rounded text-xs font-medium border transition-colors',
                showPlanet ? 'bg-blue-600/30 text-blue-300 border-blue-600/50'
                           : 'bg-black/50 text-gray-400 hover:text-blue-300 hover:bg-gray-800/70 border-gray-700/60')}
              title="Toggle planet parameters">
              <Globe size={12} className="text-blue-400 shrink-0" />
              <span>Planet</span>
              <StatusIcon status={planetStatus} isRunning={isRunning} />
            </button>
            <button onClick={() => setShowMoon(v => !v)}
              className={cn('flex items-center gap-1.5 px-2.5 py-1.5 rounded text-xs font-medium border transition-colors',
                showMoon ? 'bg-red-600/30 text-red-300 border-red-600/50'
                         : 'bg-black/50 text-gray-400 hover:text-red-300 hover:bg-gray-800/70 border-gray-700/60')}
              title="Toggle moon parameters">
              <Moon size={12} className="text-red-400 shrink-0" />
              <span>Moon</span>
              <StatusIcon status={moonStatus} isRunning={isRunning} />
            </button>
            <button onClick={() => setShowEda(v => !v)}
              className={cn('flex items-center gap-1.5 px-2.5 py-1.5 rounded text-xs font-medium border transition-colors',
                showEda ? 'bg-blue-600/80 text-white border-blue-500/60'
                        : 'bg-black/50 text-gray-400 hover:text-white hover:bg-gray-800/70 border-gray-700/60')}
              title="Toggle EDA panel">
              <BarChart2 size={12} />
              EDA
            </button>
            <button onClick={() => setShowMl(v => !v)}
              className={cn('flex items-center gap-1.5 px-2.5 py-1.5 rounded text-xs font-medium border transition-colors',
                showMl ? 'bg-violet-600/80 text-white border-violet-500/60'
                       : 'bg-black/50 text-gray-400 hover:text-violet-300 hover:bg-gray-800/70 border-gray-700/60')}
              title="Toggle ML Stability Predictor">
              <Brain size={12} className={showMl ? 'text-white' : 'text-violet-400'} />
              ML
            </button>
          </div>

          {/* Object parameter overlays */}
          {showStar   && <StellarOverlay onClose={handleCloseStar}   containerRef={containerRef} />}
          {showPlanet && <PlanetOverlay  onClose={handleClosePlanet} containerRef={containerRef} />}
          {showMoon   && <MoonOverlay    onClose={handleCloseMoon}   containerRef={containerRef} />}
          {showEda    && <EdaOverlay     onClose={handleCloseEda}    containerRef={containerRef} />}
          {showMl     && (
            <MlMapOverlay
              onClose={handleCloseMl}
              containerRef={containerRef}
              onApplyAndRun={handleRun}
            />
          )}

          {/* Fullscreen enter button — bottom-left */}
          <button
            onClick={() => setShowFullscreen(true)}
            className="absolute bottom-3 left-3 z-20 flex items-center gap-1.5 px-2 py-1.5
                       rounded text-xs text-gray-500 hover:text-white border border-gray-700/50
                       bg-black/40 hover:bg-gray-800/70 transition-colors"
            title="Enter fullscreen (Esc to exit)"
          >
            <Maximize2 size={12} />
          </button>
        </>
      )}

      {/* ── Fullscreen-mode-only UI ──────────────────────────────────────────── */}
      {showFullscreen && (
        <>
          {/* Stability + habitability badges — top-left */}
          {currentFrame && simMeta && (
            <FullscreenBadges frame={currentFrame} meta={simMeta} />
          )}

          {/* Exit button — top-right */}
          <button
            onClick={() => setShowFullscreen(false)}
            className="absolute top-3 right-3 z-10 flex items-center gap-1.5 px-2.5 py-1.5
                       rounded text-xs text-gray-400 hover:text-white border border-gray-700/50
                       bg-black/50 hover:bg-gray-800/70 transition-colors"
            title="Exit fullscreen"
          >
            <X size={12} />
            <span>Exit</span>
            <span className="text-gray-600 text-[10px]">Esc</span>
          </button>
        </>
      )}

      {/* ── Always-visible: MiniOrbitView + playback ────────────────────────── */}
      <MiniOrbitView frames={trajectoryFrames} frameIndex={sceneControls.frameIndex} />

      {sceneControls.totalFrames > 0 && (
        <div
          ref={playbackPanelRef}
          className={cn(
            'absolute z-20',
            !playbackPos   && 'bottom-3 left-1/2 -translate-x-1/2',
            !playbackWidth && 'w-[min(580px,78%)]',
          )}
          style={{
            ...(playbackPos   ? { left: playbackPos.left, top: playbackPos.top } : {}),
            ...(playbackWidth ? { width: playbackWidth }                          : {}),
          }}
        >
          <div
            onMouseDown={handlePlaybackDragStart}
            className="relative flex items-center justify-center w-full py-0.5
                       cursor-grab active:cursor-grabbing select-none
                       rounded-t bg-gray-800/60 border border-b-0 border-gray-700/50"
            title="Drag to reposition"
          >
            <GripHorizontal size={12} className="text-gray-500" />
            {(playbackPos || playbackWidth) && (
              <button
                onMouseDown={e => e.stopPropagation()}
                onClick={() => { setPlaybackPos(null); setPlaybackWidth(null); }}
                className="absolute right-2 text-gray-600 hover:text-gray-300 transition-colors text-[10px]"
                title="Reset position and size"
              >↩</button>
            )}
          </div>
          <div className="relative">
            <AnimationControls {...sceneControls} className="rounded-t-none" />
            <div
              onMouseDown={handlePlaybackResizeStart}
              className="absolute bottom-0 right-0 w-4 h-4 cursor-ew-resize flex items-end justify-end p-0.5"
              title="Drag to resize"
            >
              <svg width="8" height="8" viewBox="0 0 8 8" className="opacity-40">
                <polygon points="0,8 8,0 8,8" fill="#9ca3af" />
              </svg>
            </div>
          </div>
        </div>
      )}
    </div>
  );

  return (
    <AppShell
      main={mainContent}
      chat={<ChatPanel />}
    />
  );
}

function LegendRow({ color, label }: { color: string; label: string }) {
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="inline-block w-5 h-0.5 rounded-full shrink-0" style={{ backgroundColor: color }} />
      <span className="text-gray-400">{label}</span>
    </div>
  );
}
