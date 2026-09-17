'use client';
import React, { useState } from 'react';

interface OrbitCanvasProps {
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
  hasFrames: boolean;
  webGLError?: string | null;
  className?: string;
}

export function OrbitCanvas({ canvasRef, hasFrames, webGLError, className }: OrbitCanvasProps) {
  const [dismissed, setDismissed] = useState(false);

  return (
    <div className={`relative w-full h-full ${className ?? ''}`}>
      <canvas
        ref={canvasRef}
        className="w-full h-full block"
        style={{ touchAction: 'none' }}
      />
      {webGLError && !dismissed && (
        <div className="absolute top-2 left-2 right-2 flex items-start gap-2 bg-gray-900/95 border border-amber-500/40 rounded px-3 py-2 text-xs">
          <span className="text-amber-400 mt-0.5 shrink-0">⚠</span>
          <span className="text-gray-300 flex-1">{webGLError}</span>
          <button
            onClick={() => setDismissed(true)}
            className="text-gray-500 hover:text-gray-300 shrink-0 leading-none"
          >✕</button>
        </div>
      )}
      {!hasFrames && !webGLError && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="text-center text-gray-600">
            <div className="text-4xl mb-2">✦</div>
            <p className="text-sm">Run a simulation to see the orbital animation</p>
          </div>
        </div>
      )}
    </div>
  );
}
