'use client';
import React, { useEffect, useRef } from 'react';
import { useSimulationStore } from '@/hooks/useSimulationStore';
import type { TrajectoryFrame } from '@/lib/types';

interface MiniOrbitViewProps {
  frames: TrajectoryFrame[] | null;
  frameIndex: number;
  /** Override the default absolute-positioned wrapper class for inline use. */
  className?: string;
  /** When true: draw Hill sphere (white dashed outer) + Roche limit (red dashed inner). */
  showHillSphereRings?: boolean;
  /** Roche limit as fraction of rhill (e.g. am_grid[0]). Used when showHillSphereRings=true. */
  rocheInnerFrac?: number;
}

// TRAIL: number of past frames drawn as the orbit arc behind the current position.
// With n_steps=5000 from cell_preview (~50 frames/orbit), TRAIL=150 → 3 visible orbit arcs.
const TRAIL = 150;
const CSS_W = 200;
const CSS_H = 170;

export function MiniOrbitView({
  frames, frameIndex, className, showHillSphereRings, rocheInnerFrac,
}: MiniOrbitViewProps) {
  const { params, simMeta, mlPrediction, mlMassIdx, previewRhillAU } = useSimulationStore();
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // ── Canvas draw ──────────────────────────────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !frames || frames.length === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    if (canvas.width  !== Math.round(CSS_W * dpr) ||
        canvas.height !== Math.round(CSS_H * dpr)) {
      canvas.width  = Math.round(CSS_W * dpr);
      canvas.height = Math.round(CSS_H * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const W = CSS_W, H = CSS_H;
    const mp   = params.mp_earth;
    const mm   = params.mm_earth;
    const mtot = mp + mm;

    // ── Determine trail frames — always driven by external frameIndex ─────────
    // ML/HNN cell preview (showHillSphereRings=true): accumulate full history
    // from frame 0 to current — same as the 3D canvas trail. Reveals escape
    // spirals even for freeze-post-escape trajectories (the frozen escape
    // position is visible as a dot at rhill distance).
    // Non-ML path: last TRAIL frames only (compact arc, matches existing UX).
    let trailFrames: TrajectoryFrame[];
    let curFrame: TrajectoryFrame;

    const startIdx = showHillSphereRings ? 0 : Math.max(0, frameIndex - TRAIL);
    const endIdx   = Math.min(frameIndex + 1, frames.length);
    trailFrames    = frames.slice(startIdx, endIdx);
    curFrame       = trailFrames[trailFrames.length - 1];

    if (!curFrame) return;

    const relPoints = trailFrames.map(f => {
      const bx = (mp * f.planet_x + mm * f.moon_x) / mtot;
      const by = (mp * f.planet_y + mm * f.moon_y) / mtot;
      return {
        px: f.planet_x - bx, py: f.planet_y - by,
        mx: f.moon_x   - bx, my: f.moon_y   - by,
      };
    });
    const curBx = (mp * curFrame.planet_x + mm * curFrame.moon_x) / mtot;
    const curBy = (mp * curFrame.planet_y + mm * curFrame.moon_y) / mtot;
    const curRel = {
      px: curFrame.planet_x - curBx, py: curFrame.planet_y - curBy,
      mx: curFrame.moon_x   - curBx, my: curFrame.moon_y   - curBy,
    };

    const rhill = simMeta?.rhill_AU ?? previewRhillAU ?? null;

    // maxR: ensure rings always fit on canvas
    let maxR = 0;
    for (const p of relPoints) {
      maxR = Math.max(maxR, Math.abs(p.mx), Math.abs(p.my),
                           Math.abs(p.px), Math.abs(p.py));
    }
    if (maxR === 0) maxR = 1e-6;

    if (rhill) {
      // Hill sphere is the outermost ring — always on canvas
      maxR = Math.max(maxR, rhill);
    } else if (mlPrediction) {
      const amRange = mlPrediction.validAmPerMm[mlMassIdx];
      if (amRange && rhill) maxR = Math.max(maxR, amRange[1] * rhill);
    }

    const pad   = 18;
    const scale = (Math.min(W, H) / 2 - pad) / maxR;
    const cx    = W / 2, cy = H / 2;

    const toScreen = (x: number, y: number) => ({
      sx: cx + x * scale, sy: cy - y * scale,
    });

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#050a14';
    ctx.fillRect(0, 0, W, H);

    // Barycenter crosshair
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.lineWidth   = 0.5;
    ctx.beginPath(); ctx.moveTo(cx - 6, cy); ctx.lineTo(cx + 6, cy); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx, cy - 6); ctx.lineTo(cx, cy + 6); ctx.stroke();

    // Planet trail
    ctx.strokeStyle = 'rgba(68,136,255,0.5)';
    ctx.lineWidth   = 1;
    ctx.beginPath();
    relPoints.forEach((p, i) => {
      const { sx, sy } = toScreen(p.px, p.py);
      i === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
    });
    ctx.stroke();

    // Moon trail
    ctx.strokeStyle = 'rgba(255,85,85,0.5)';
    ctx.lineWidth   = 1;
    ctx.beginPath();
    relPoints.forEach((p, i) => {
      const { sx, sy } = toScreen(p.mx, p.my);
      i === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
    });
    ctx.stroke();

    // Current positions
    const { sx: psx, sy: psy } = toScreen(curRel.px, curRel.py);
    ctx.fillStyle = '#4488FF';
    ctx.beginPath(); ctx.arc(psx, psy, 4, 0, Math.PI * 2); ctx.fill();

    const { sx: msx, sy: msy } = toScreen(curRel.mx, curRel.my);
    // Escaped if 2D moon–planet distance > rhill (same convention as agent_service _traj_to_frames)
    const dx = curRel.mx - curRel.px, dy = curRel.my - curRel.py;
    const moonPlanetDist2D = Math.sqrt(dx * dx + dy * dy);
    const moonEscaped = rhill !== null && moonPlanetDist2D > rhill;
    ctx.fillStyle = moonEscaped ? '#FF8800' : '#FF5555';  // orange = escaped, red = stable
    ctx.beginPath(); ctx.arc(msx, msy, 3, 0, Math.PI * 2); ctx.fill();
    // Escaped halo — makes borderline escapes (moon at Hill sphere edge) unambiguous
    if (moonEscaped) {
      ctx.strokeStyle = 'rgba(255,136,0,0.55)';
      ctx.lineWidth   = 1.5;
      ctx.beginPath(); ctx.arc(msx, msy, 6, 0, Math.PI * 2); ctx.stroke();
    }

    // ── Z-axis status flags ───────────────────────────────────────────────────
    // HNN runs in a rotating Hill frame — Z drift can cause 3D escape or
    // uninhabitability while XY looks stable/habitable. Flag these explicitly.
    // Non-ML sims start with Z=0, vz=0 (coplanar physics → Z stays 0 forever),
    // so dist3D = dist2D always there and this flag never fires naturally.
    const dist3D   = curFrame.moon_planet_dist;                  // 3D (from _traj_to_frames)
    const zEscaped = rhill !== null && dist3D > rhill && !moonEscaped;

    let zUninhabitable = false;
    if (curFrame.moon_star_dist !== undefined && simMeta) {
      const msd3D = curFrame.moon_star_dist;
      const msd2D = Math.sqrt(
        (curFrame.moon_x - curFrame.star_x) ** 2 +
        (curFrame.moon_y - curFrame.star_y) ** 2,
      );
      const aIn = simMeta.a_inner_au, aOut = simMeta.a_outer_au;
      zUninhabitable = (msd3D < aIn || msd3D > aOut) && (msd2D >= aIn && msd2D <= aOut);
    }

    // ── ML valid-orbit annulus — always drawn when mlPrediction exists ─────────
    if (mlPrediction && rhill) {
      const amRange = mlPrediction.validAmPerMm[mlMassIdx];
      if (amRange) {
        const amInnerPx = amRange[0] * rhill * scale;
        const amOuterPx = amRange[1] * rhill * scale;
        ctx.save();
        // Outer ML bound — violet/purple
        ctx.strokeStyle  = 'rgba(139,92,246,0.70)';
        ctx.lineWidth    = 1.2;
        ctx.setLineDash([4, 3]);
        ctx.beginPath(); ctx.arc(psx, psy, amOuterPx, 0, Math.PI * 2); ctx.stroke();
        // Inner ML bound — cyan
        ctx.strokeStyle = 'rgba(6,182,212,0.70)';
        ctx.lineWidth   = 1.0;
        ctx.beginPath(); ctx.arc(psx, psy, amInnerPx, 0, Math.PI * 2); ctx.stroke();
        ctx.restore();
      }
    }

    // ── Hill sphere + Roche limit rings (only in preview mode) ────────────────
    if (showHillSphereRings && rhill) {
      const hillPx  = rhill * scale;
      const rochePx = (rocheInnerFrac ?? 0) * rhill * scale;
      ctx.save();
      // Hill sphere — white dashed (outermost)
      ctx.strokeStyle = 'rgba(255,255,255,0.65)';
      ctx.lineWidth   = 1.2;
      ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.arc(psx, psy, hillPx, 0, Math.PI * 2); ctx.stroke();
      // Roche limit — red dashed (innermost)
      if (rochePx > 0) {
        ctx.strokeStyle = 'rgba(239,68,68,0.70)';
        ctx.lineWidth   = 1.0;
        ctx.beginPath(); ctx.arc(psx, psy, rochePx, 0, Math.PI * 2); ctx.stroke();
      }
      ctx.restore();
    }

    // Z-axis flag text — amber badge at canvas bottom-left
    if (zEscaped || zUninhabitable) {
      const flagText = (zEscaped && zUninhabitable)
        ? 'Z-axis moon escaped and uninhabitable'
        : zEscaped ? 'Z-axis moon escaped' : 'Z-axis moon uninhabitable';
      ctx.fillStyle = 'rgba(251,191,36,0.92)';  // amber-400 — Z-drift, not XY escape
      ctx.font      = 'bold 7px monospace';
      ctx.fillText(flagText, 4, H - 5);
    }

    ctx.fillStyle = 'rgba(200,200,200,0.7)';
    ctx.font      = '9px monospace';
    ctx.fillText('Planet–Moon bary frame', 4, 11);

  }, [frames, frameIndex, params.mp_earth, params.mm_earth, simMeta, previewRhillAU,
      mlPrediction, mlMassIdx, showHillSphereRings, rocheInnerFrac]);

  if (!frames || frames.length === 0) return null;

  const rhill    = simMeta?.rhill_AU ?? previewRhillAU ?? null;
  const amRange  = (mlPrediction && rhill) ? mlPrediction.validAmPerMm[mlMassIdx] : null;

  return (
    <div className={className ?? 'absolute bottom-16 right-3 z-10 pointer-events-none'}
         style={{ width: CSS_W }}>
      {/* Legend */}
      {(amRange || (showHillSphereRings && rhill)) && (
        <div className="mb-0.5 px-1.5 py-1 rounded border border-gray-700/40
                        bg-gray-900/85 text-[9px] text-gray-400 space-y-0.5">
          {showHillSphereRings && rhill && (
            <>
              <div className="flex items-center gap-1.5">
                <svg width="18" height="5">
                  <line x1="0" y1="2.5" x2="18" y2="2.5"
                    stroke="rgba(255,255,255,0.65)" strokeWidth="1.2" strokeDasharray="4,3" />
                </svg>
                <span>Hill sphere {rhill.toFixed(4)} AU</span>
              </div>
              {(rocheInnerFrac ?? 0) > 0 && (
                <div className="flex items-center gap-1.5">
                  <svg width="18" height="5">
                    <line x1="0" y1="2.5" x2="18" y2="2.5"
                      stroke="rgba(239,68,68,0.70)" strokeWidth="1.0" strokeDasharray="4,3" />
                  </svg>
                  <span>Roche {((rocheInnerFrac ?? 0) * rhill).toFixed(4)} AU</span>
                </div>
              )}
            </>
          )}
          {amRange && rhill && (
            <>
              <div className="flex items-center gap-1.5">
                <svg width="18" height="5">
                  <line x1="0" y1="2.5" x2="18" y2="2.5"
                    stroke="rgba(139,92,246,0.80)" strokeWidth="1.2" strokeDasharray="4,3" />
                </svg>
                <span>MLP valid orbit max {(amRange[1] * rhill).toFixed(4)} AU</span>
              </div>
              <div className="flex items-center gap-1.5">
                <svg width="18" height="5">
                  <line x1="0" y1="2.5" x2="18" y2="2.5"
                    stroke="rgba(6,182,212,0.80)" strokeWidth="1.2" strokeDasharray="4,3" />
                </svg>
                <span>MLP valid orbit min {(amRange[0] * rhill).toFixed(4)} AU</span>
              </div>
            </>
          )}
        </div>
      )}
      <div className="rounded border border-gray-700/50 overflow-hidden bg-[#050a14]/80"
           style={{ height: CSS_H }}>
        <canvas ref={canvasRef} style={{ width: CSS_W, height: CSS_H, display: 'block' }} />
      </div>
    </div>
  );
}
