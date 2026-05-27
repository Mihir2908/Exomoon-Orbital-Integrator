'use client';
import { useCallback, useRef } from 'react';
import { useSimulationStore } from './useSimulationStore';

const AGENT_URL =
  process.env.NEXT_PUBLIC_AGENT_URL ??
  (typeof window !== 'undefined' ? window.location.origin : 'http://127.0.0.1:8000');

export function useChatStream() {
  const {
    params, simYears, simdataB64, dmCgs,
    addChatMessage, appendToLastAssistant, finalizeChatMessage,
    setSimdataB64, updateJobStatus,
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
      const body = {
        message: userText,
        simdata: simdataB64,
        params: { ...params, dm_cgs: dmCgs },  // include moon density for habitability context
        years: simYears,
        escape_factor: 1.0,
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
            };

            if (payload.type === 'token' && payload.token) {
              appendToLastAssistant(payload.token);
            } else if (payload.type === 'meta' && payload.job_id) {
              // Job started — store job ID so poller picks it up
              updateJobStatus('running', 0, {});
            } else if (payload.type === 'done') {
              if (payload.simdata) setSimdataB64(payload.simdata);
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
  }, [params, simYears, simdataB64, dmCgs, addChatMessage, appendToLastAssistant, finalizeChatMessage, setSimdataB64, updateJobStatus]);

  const abort = useCallback(() => abortRef.current?.abort(), []);

  return { sendMessage, abort };
}
