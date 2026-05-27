'use client';
import React, { useEffect, useRef } from 'react';
import { useSimulationStore } from '@/hooks/useSimulationStore';
import type { TrajectoryFrame } from '@/lib/types';

interface MiniOrbitViewProps {
  frames: TrajectoryFrame[] | null;
  frameIndex: number;
}

const TRAIL = 300;
const CSS_W = 200;
const CSS_H = 170;

export function MiniOrbitView({ frames, frameIndex }: MiniOrbitViewProps) {
  const { params } = useSimulationStore();
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !frames || frames.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Scale canvas backing store to device pixel ratio to prevent blur.
    // The HTML element stays at CSS_W × CSS_H logical pixels; only the
    // internal resolution is increased so text and lines are crisp.
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(CSS_W * dpr) || canvas.height !== Math.round(CSS_H * dpr)) {
      canvas.width  = Math.round(CSS_W * dpr);
      canvas.height = Math.round(CSS_H * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const W = CSS_W;
    const H = CSS_H;
    ctx.clearRect(0, 0, W, H);

    const mp = params.mp_earth;
    const mm = params.mm_earth;
    const mtot = mp + mm;

    const startIdx = Math.max(0, frameIndex - TRAIL);
    const trailFrames = frames.slice(startIdx, frameIndex + 1);

    const relPoints = trailFrames.map(f => {
      const bx = (mp * f.planet_x + mm * f.moon_x) / mtot;
      const by = (mp * f.planet_y + mm * f.moon_y) / mtot;
      return {
        px: f.planet_x - bx, py: f.planet_y - by,
        mx: f.moon_x   - bx, my: f.moon_y   - by,
      };
    });

    let maxR = 0;
    for (const p of relPoints) {
      maxR = Math.max(maxR, Math.abs(p.mx), Math.abs(p.my), Math.abs(p.px), Math.abs(p.py));
    }
    if (maxR === 0) return;

    const pad = 18;
    const scale = (Math.min(W, H) / 2 - pad) / maxR;
    const cx = W / 2;
    const cy = H / 2;

    const toScreen = (x: number, y: number) => ({
      sx: cx + x * scale,
      sy: cy - y * scale,
    });

    // Background
    ctx.fillStyle = '#050a14';
    ctx.fillRect(0, 0, W, H);

    // Barycenter crosshair
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(cx - 6, cy); ctx.lineTo(cx + 6, cy); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx, cy - 6); ctx.lineTo(cx, cy + 6); ctx.stroke();

    // Planet trail
    ctx.strokeStyle = 'rgba(68,136,255,0.5)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    relPoints.forEach((p, i) => {
      const { sx, sy } = toScreen(p.px, p.py);
      i === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
    });
    ctx.stroke();

    // Moon trail
    ctx.strokeStyle = 'rgba(255,85,85,0.5)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    relPoints.forEach((p, i) => {
      const { sx, sy } = toScreen(p.mx, p.my);
      i === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
    });
    ctx.stroke();

    // Current positions
    const cur = relPoints[relPoints.length - 1];
    if (!cur) return;

    const { sx: psx, sy: psy } = toScreen(cur.px, cur.py);
    ctx.fillStyle = '#4488FF';
    ctx.beginPath(); ctx.arc(psx, psy, 4, 0, Math.PI * 2); ctx.fill();

    const { sx: msx, sy: msy } = toScreen(cur.mx, cur.my);
    ctx.fillStyle = '#FF5555';
    ctx.beginPath(); ctx.arc(msx, msy, 3, 0, Math.PI * 2); ctx.fill();

    // Label
    ctx.fillStyle = 'rgba(200,200,200,0.7)';
    ctx.font = '9px monospace';
    ctx.fillText('Planet–Moon bary frame', 4, 11);

  }, [frames, frameIndex, params.mp_earth, params.mm_earth]);

  if (!frames || frames.length === 0) return null;

  return (
    <div className="absolute bottom-16 right-3 z-10 pointer-events-none"
         style={{ width: CSS_W, height: CSS_H }}>
      <div className="rounded border border-gray-700/50 overflow-hidden bg-[#050a14]/80">
        {/* style width/height keeps CSS layout size fixed; canvas attr controls resolution */}
        <canvas
          ref={canvasRef}
          style={{ width: CSS_W, height: CSS_H, display: 'block' }}
        />
      </div>
    </div>
  );
}
