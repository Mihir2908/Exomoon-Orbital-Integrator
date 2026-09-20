'use client';
import { useCallback, useRef } from 'react';
import { useSimulationStore } from './useSimulationStore';
import type { MlPrediction, TrajPreview } from '@/lib/types';

// Chat stream is long-running (Claude tool loops + extended thinking can take 60-180s).
// Use the direct agent URL to bypass the Next.js rewrite proxy's ~30s timeout.
const AGENT_URL =
  process.env.NEXT_PUBLIC_AGENT_DIRECT_URL ??
  process.env.NEXT_PUBLIC_AGENT_URL ??
  'http://127.0.0.1:8000';

export function useChatStream() {
  const {
    params, simYears, simdataB64, dmCgs,
    mlPrediction, trajPreview,
    addChatMessage, appendToLastAssistant, finalizeChatMessage,
    setSimdataB64, setMlPrediction, setTrajPreview, updateJobStatus, setJob, setParams,
    setPreviewCellFrames, setChatCellFrames,
  } = useSimulationStore();

  const abortRef = useRef<AbortController | null>(null);

  const sendMessage = useCallback(async (userText: string) => {
    if (!userText.trim()) return;

    // Add user message immediately
    addChatMessage({ role: 'user', content: userText, id: crypto.randomUUID() });

    // Placeholder for streaming assistant message
    const assistantId = crypto.randomUUID();
    addChatMessage({ role: 'assistant', content: '', id: assistantId, streaming: true });

    abortRef.current?.abort();
    abortRef.current = new AbortController();

    try {
      // Build a compact summary of the current ML prediction to send to the agent
      // (only metadata, not the full 2500-element arrays — those stay on the frontend)
      const mlPredSummary = mlPrediction ? {
        valid_mm_range:   mlPrediction.validMmRange,
        mm_grid:          mlPrediction.mmGrid,
        am_grid:          mlPrediction.amGrid,
        valid_am_per_mm:  mlPrediction.validAmPerMm,
        // omit mapStable/mapHabitable/mapBoth — too large for the request body
      } : null;

      const body = {
        message:          userText,
        simdata:          simdataB64,
        params:           { ...params, dm_cgs: dmCgs },
        years:            simYears,
        escape_factor:    1.0,
        ml_prediction:    mlPredSummary,
        // Send traj preview key + grid arrays so trajectory_cell_query works even when
        // the batch was run from the panel (not via a chatbot trajectory_preview call).
        traj_preview_key: trajPreview?.cache_key ?? null,
        traj_mm_grid:     trajPreview?.mm_grid    ?? null,
        traj_am_grid:     trajPreview?.am_grid    ?? null,
      };

      const response = await fetch(`${AGENT_URL}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: abortRef.current.signal,
      });

      if (!response.ok || !response.body) {
        finalizeChatMessage(assistantId, `Error: ${response.status} ${response.statusText}`);
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split('\n\n');
        buffer = events.pop() ?? '';

        for (const eventBlock of events) {
          const dataLine = eventBlock.split('\n').find(l => l.startsWith('data:'));
          if (!dataLine) continue;

          const raw = dataLine.slice(5).trim();
          if (!raw) continue;

          try {
            const payload = JSON.parse(raw) as {
              type: string;
              token?: string;
              job_id?: string;
              simdata?: string;
              urls?: Record<string, string>;
              status?: string;
              elapsed_seconds?: number;
              effective_params?: Record<string, number | boolean> | null;
              ml_prediction?: {
                mm_grid: number[];
                am_grid: number[];
                map_stable: boolean[][];
                map_habitable: boolean[][];
                map_both: boolean[][];
                valid_mm_range: [number, number] | null;
                valid_am_per_mm: ([number, number] | null)[];
              } | null;
              cell_frames?: import('@/lib/types').TrajectoryFrame[] | null;
              cell_rhill_au?: number | null;
              cell_roche_frac?: number | null;
              cell_html_2d_url?: string | null;
              cell_html_3d_url?: string | null;
              cell_mm_earth?: number | null;
              cell_am_hill?: number | null;
              traj_preview?: {
                ok: boolean;
                map_stable: boolean[][];
                map_habitable: boolean[][];
                map_both: boolean[][];
                mm_grid: number[];
                am_grid: number[];
                wall_s?: number;
                from_cache?: boolean;
                cache_key?: string;
                valid_mm_range?: [number, number] | null;
                valid_am_per_mm?: ([number, number] | null)[];
              } | null;
            };

            if (payload.type === 'token' && payload.token) {
              appendToLastAssistant(payload.token);
            } else if (payload.type === 'meta') {
              console.log(`[ChatStream] meta event — job_id=${payload.job_id}`);
              if (payload.job_id) {
                // Job started — register with store so useJobPoller polls and updates orbit view
                setJob(payload.job_id);
                console.log(`[ChatStream] setJob(${payload.job_id}) called`);
              }
            } else if (payload.type === 'done') {
              if (payload.simdata) setSimdataB64(payload.simdata);
              // Sync sliders with whatever params the agent actually ran the simulation with
              if (payload.effective_params) setParams(payload.effective_params);
              // Layer 2 trajectory preview — pushed as traj_preview (separate from ml_prediction).
              // Storing separately keeps mlPrediction (MLP Layer 1) clean so MlMapOverlay's
              // confidence_map computation remains valid (MLP vs trajectory comparison).
              if (payload.traj_preview) {
                setTrajPreview(payload.traj_preview as TrajPreview);
              }
              // Layer 1 MLP prediction — only from ml_predict tool calls
              if (payload.ml_prediction) {
                const p = payload.ml_prediction;
                const mlPred: MlPrediction = {
                  mmGrid:        p.mm_grid,
                  amGrid:        p.am_grid,
                  mapStable:     p.map_stable,
                  mapHabitable:  p.map_habitable,
                  mapBoth:       p.map_both,
                  validMmRange:  p.valid_mm_range,
                  validAmPerMm:  p.valid_am_per_mm,
                };
                setMlPrediction(mlPred);
              }
              // If the agent ran trajectory_cell_query, push frames to both mini orbit views
              if (payload.cell_frames && payload.cell_frames.length > 0) {
                const rhillAU   = payload.cell_rhill_au   ?? null;
                const rocheFrac = payload.cell_roche_frac ?? null;
                const mmEarth   = payload.cell_mm_earth   ?? null;
                const amHill    = payload.cell_am_hill    ?? null;
                // External MiniOrbitView + 3D canvas (previewCellFrames in store)
                setPreviewCellFrames(payload.cell_frames, rocheFrac, rhillAU);
                // Cell details panel inside MlMapOverlay (chatCellFrames in store)
                setChatCellFrames(payload.cell_frames, rhillAU, rocheFrac, mmEarth, amHill);
              }
              finalizeChatMessage(assistantId);
            }
          } catch {
            // non-JSON SSE line, ignore
          }
        }
      }
    } catch (err: unknown) {
      if (err instanceof Error && err.name === 'AbortError') return;
      finalizeChatMessage(assistantId, 'Connection error. Is the agent service running?');
    }
  }, [params, simYears, simdataB64, dmCgs, mlPrediction, trajPreview, addChatMessage, appendToLastAssistant, finalizeChatMessage, setSimdataB64, setMlPrediction, setTrajPreview, updateJobStatus, setJob, setParams, setPreviewCellFrames, setChatCellFrames]);

  const abort = useCallback(() => abortRef.current?.abort(), []);

  return { sendMessage, abort };
}
