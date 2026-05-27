'use client';
import React from 'react';

interface OrbitCanvasProps {
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
  hasFrames: boolean;
  className?: string;
}

export function OrbitCanvas({ canvasRef, hasFrames, className }: OrbitCanvasProps) {
  return (
    <div className={`relative w-full h-full ${className ?? ''}`}>
      <canvas
        ref={canvasRef}
        className="w-full h-full block"
        style={{ touchAction: 'none' }}
      />
      {!hasFrames && (
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
