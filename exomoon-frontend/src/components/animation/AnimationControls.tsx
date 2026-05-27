'use client';
import React from 'react';
import { Play, Pause, SkipBack, SkipForward } from 'lucide-react';
import type { SceneControls } from './useOrbitScene';
import { cn } from '@/lib/utils';

interface AnimationControlsProps extends SceneControls {
  className?: string;
}

const SPEED_OPTIONS = [0.25, 0.5, 1, 2, 4, 8, 16];

export function AnimationControls({
  frameIndex,
  totalFrames,
  isPlaying,
  speedMultiplier,
  setFrameIndex,
  setIsPlaying,
  setSpeedMultiplier,
  className,
}: AnimationControlsProps) {
  if (totalFrames === 0) return null;

  const pct = totalFrames > 1 ? (frameIndex / (totalFrames - 1)) * 100 : 0;

  const stepBack = () => {
    setIsPlaying(false);
    setFrameIndex(Math.max(0, frameIndex - 1));
  };
  const stepFwd = () => {
    setIsPlaying(false);
    setFrameIndex(Math.min(totalFrames - 1, frameIndex + 1));
  };

  return (
    <div className={cn('bg-black/60 backdrop-blur-sm border border-gray-700/40 rounded-lg px-3 py-2 space-y-2', className)}>
      {/* Scrubber */}
      <div className="flex items-center gap-2">
        <span className="text-xs text-gray-500 font-mono w-10 shrink-0 text-right">{frameIndex + 1}</span>
        <input
          type="range"
          min={0}
          max={totalFrames - 1}
          value={frameIndex}
          onChange={e => {
            setIsPlaying(false);
            setFrameIndex(parseInt(e.target.value, 10));
          }}
          className="flex-1 h-1 accent-blue-500 cursor-pointer"
        />
        <span className="text-xs text-gray-500 font-mono w-10 shrink-0">{totalFrames}</span>
      </div>

      {/* Controls row */}
      <div className="flex items-center justify-between gap-3">
        {/* Playback buttons */}
        <div className="flex items-center gap-1">
          <IconBtn onClick={stepBack} title="Step back">
            <SkipBack size={13} />
          </IconBtn>
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className={cn(
              'flex items-center justify-center w-7 h-7 rounded-full text-white transition-colors',
              isPlaying ? 'bg-blue-600 hover:bg-blue-500' : 'bg-gray-700 hover:bg-gray-600'
            )}
            title={isPlaying ? 'Pause' : 'Play'}
          >
            {isPlaying ? <Pause size={13} /> : <Play size={13} />}
          </button>
          <IconBtn onClick={stepFwd} title="Step forward">
            <SkipForward size={13} />
          </IconBtn>
        </div>

        {/* Speed selector */}
        <div className="flex items-center gap-1.5">
          <span className="text-xs text-gray-500">Speed</span>
          <div className="flex gap-0.5">
            {SPEED_OPTIONS.map(s => (
              <button
                key={s}
                onClick={() => setSpeedMultiplier(s)}
                className={cn(
                  'px-1.5 py-0.5 text-xs rounded font-mono transition-colors',
                  speedMultiplier === s
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                )}
              >
                {s}×
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function IconBtn({ onClick, title, children }: { onClick: () => void; title: string; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      title={title}
      className="flex items-center justify-center w-6 h-6 rounded text-gray-400 hover:text-white hover:bg-gray-700 transition-colors"
    >
      {children}
    </button>
  );
}
