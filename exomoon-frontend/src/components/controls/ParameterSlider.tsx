'use client';
import React from 'react';
import { cn } from '@/lib/utils';

interface ParameterSliderProps {
  label: string;
  unit?: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  disabled?: boolean;
}

export function ParameterSlider({
  label, unit, value, min, max, step, onChange, disabled,
}: ParameterSliderProps) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between items-center">
        <label className="text-xs text-gray-400">{label}</label>
        <span className="text-xs font-mono text-blue-300">
          {value.toFixed(step < 0.01 ? 3 : step < 0.1 ? 2 : step < 1 ? 2 : 0)}
          {unit ? ` ${unit}` : ''}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={e => onChange(parseFloat(e.target.value))}
        className={cn(
          'w-full h-1.5 appearance-none rounded-full cursor-pointer',
          'bg-gray-700 accent-blue-500',
          disabled && 'opacity-40 cursor-not-allowed'
        )}
      />
    </div>
  );
}
