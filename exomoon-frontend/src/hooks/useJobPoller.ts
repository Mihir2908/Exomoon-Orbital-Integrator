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
    console.log(`[JobPoller] effect fired — jobId=${jobId} jobStatus=${jobStatus}`);
    if (!jobId || jobStatus !== 'running') {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      return;
    }

    const poll = async () => {
      try {
        console.log(`[JobPoller] polling ${jobId}...`);
        const data = await agentApi.getJobStatus(jobId);
        console.log(`[JobPoller] status=${data.status} urls=`, data.urls);
        updateJobStatus(data.status, data.elapsed_seconds, data.urls);

        if (data.status === 'SUCCEEDED') {
          clearInterval(intervalRef.current!);
          intervalRef.current = null;

          // meta comes directly from the agent service status response (no S3 fetch needed)
          const summaryJson: Record<string, unknown> = (data.meta ?? {}) as Record<string, unknown>;

          // Fetch traj.csv → parse → update Three.js scene
          const csvUrl = data.urls?.['traj.csv'];
          console.log(`[JobPoller] SUCCEEDED — csvUrl=${csvUrl}`);

          if (csvUrl) {
            try {
              const csvText = await fetch(csvUrl).then(r => r.text());
              console.log(`[JobPoller] CSV fetched — ${csvText.length} chars, parsing...`);
              const { frames, meta } = parseTrajectoryCsv(csvText, summaryJson);
              console.log(`[JobPoller] parsed ${frames.length} frames, calling setTrajectoryData`);
              setTrajectoryData(frames, meta);
              console.log(`[JobPoller] setTrajectoryData done`);
            } catch (e) {
              console.error('[JobPoller] CSV parse error:', e);
            }
          } else {
            console.warn('[JobPoller] SUCCEEDED but no traj.csv URL in response');
          }

          // Cache simdata in agent service session for follow-up chat queries
          agentApi.retrieveSimdata(jobId).catch(console.warn);

        } else if (data.status === 'FAILED' || data.status === 'TIMED_OUT') {
          clearInterval(intervalRef.current!);
          intervalRef.current = null;
        }
      } catch (e) {
        if (e instanceof TypeError && String(e).includes('fetch')) {
          console.warn(`[JobPoller] Cannot reach agent service via /api/agent — is the agent service running on port 8000?`, e);
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
