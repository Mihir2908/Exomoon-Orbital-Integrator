'use client';
import { useEffect, useRef } from 'react';
import { useSimulationStore } from './useSimulationStore';
import { agentApi } from '@/lib/agentApi';
import { parseTrajectoryCsv } from '@/lib/csvParser';

const POLL_INTERVAL_MS = 5000;

export function useJobPoller() {
  const {
    jobId, jobStatus,
    updateJobStatus, setTrajectoryData, setSimdata,
  } = useSimulationStore();

  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!jobId || jobStatus !== 'running') {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      return;
    }

    const poll = async () => {
      try {
        const data = await agentApi.getJobStatus(jobId);
        updateJobStatus(data.status, data.elapsed_seconds, data.urls);

        if (data.status === 'SUCCEEDED') {
          clearInterval(intervalRef.current!);
          intervalRef.current = null;

          // Fetch traj.csv → parse → update Three.js scene
          const csvUrl = data.urls?.['traj.csv'];
          const summaryUrl = data.urls?.['summary.json'];

          let summaryJson: Record<string, unknown> = {};
          if (summaryUrl) {
            try {
              const sr = await fetch(summaryUrl);
              summaryJson = await sr.json();
            } catch { /* ignore */ }
          }

          if (csvUrl) {
            try {
              const csvText = await fetch(csvUrl).then(r => r.text());
              const { frames, meta } = parseTrajectoryCsv(csvText, summaryJson);
              setTrajectoryData(frames, meta);
            } catch (e) {
              console.error('[JobPoller] CSV parse error:', e);
            }
          }

          // Cache simdata in agent service session for follow-up chat queries
          agentApi.retrieveSimdata(jobId).catch(console.warn);

        } else if (data.status === 'FAILED' || data.status === 'TIMED_OUT') {
          clearInterval(intervalRef.current!);
          intervalRef.current = null;
        }
      } catch (e) {
        if (e instanceof TypeError && String(e).includes('fetch')) {
          console.warn(`[JobPoller] Cannot reach agent at http://127.0.0.1:8000 — is Terminal 2 still running?`, e);
        } else {
          console.warn('[JobPoller] poll error:', e);
        }
      }
    };

    intervalRef.current = setInterval(poll, POLL_INTERVAL_MS);
    poll(); // immediate first check

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [jobId, jobStatus]);
}
