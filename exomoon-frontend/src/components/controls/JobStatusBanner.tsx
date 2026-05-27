'use client';
import React from 'react';
import { useSimulationStore } from '@/hooks/useSimulationStore';
import { cn } from '@/lib/utils';
import { Loader2, CheckCircle, XCircle, Clock } from 'lucide-react';

export function JobStatusBanner() {
  const { jobId, jobStatus, jobElapsedSeconds } = useSimulationStore();

  if (!jobId || jobStatus === 'idle') return null;

  const config = {
    running:   { icon: <Loader2 size={14} className="animate-spin" />, color: 'text-yellow-400 border-yellow-800 bg-yellow-950', label: `Running… ${jobElapsedSeconds}s` },
    succeeded: { icon: <CheckCircle size={14} />, color: 'text-green-400 border-green-800 bg-green-950', label: 'Simulation complete — scene updated' },
    failed:    { icon: <XCircle size={14} />, color: 'text-red-400 border-red-800 bg-red-950', label: 'Job failed or timed out' },
  }[jobStatus] ?? { icon: <Clock size={14} />, color: 'text-gray-400 border-gray-700 bg-gray-900', label: 'Unknown' };

  return (
    <div className={cn('flex items-center gap-2 px-3 py-1.5 rounded border text-xs font-mono', config.color)}>
      {config.icon}
      <span>{config.label}</span>
      <span className="text-gray-500 ml-auto truncate max-w-32" title={jobId}>{jobId}</span>
    </div>
  );
}
