import type { SystemParams, JobStatus, AgentHealthResponse, ChatRequest } from './types';

const AGENT_URL =
  process.env.NEXT_PUBLIC_AGENT_URL ??
  'http://127.0.0.1:8000';

// Convert SystemParams to the flat dict the agent expects
export function paramsToAgentFormat(p: SystemParams): Record<string, unknown> {
  return {
    Ts:             p.Ts,
    rs_solar:       p.rs_solar,
    ms_solar:       p.ms_solar,
    mp_earth:       p.mp_earth,
    dp_cgs:         p.dp_cgs,
    ap_AU:          p.ap_AU,
    ep:             p.ep,
    mm_earth:       p.mm_earth,
    am_hill:        p.am_hill,
    em:             p.em,
    moon_retrograde: p.moon_retrograde,
  };
}

export const agentApi = {
  health: async (): Promise<AgentHealthResponse> => {
    const r = await fetch(`${AGENT_URL}/health`, { method: 'GET' });
    return r.json();
  },

  getJobStatus: async (jobId: string): Promise<JobStatus> => {
    const r = await fetch(`${AGENT_URL}/job/${jobId}/status`, { method: 'GET' });
    return r.json();
  },

  retrieveSimdata: async (jobId: string): Promise<{ ok: boolean; simdata_cached: boolean }> => {
    const r = await fetch(`${AGENT_URL}/job/${jobId}/retrieve_simdata`, { method: 'GET' });
    return r.json();
  },

  fetchPlanet: async (name: string): Promise<{ ok: boolean; data: Record<string, unknown> | null }> => {
    const r = await fetch(`${AGENT_URL}/tool/fetch_exoplanet`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    });
    return r.json();
  },

  // Direct job submission — bypasses Claude (mirrors Dash Run button / Mode 1)
  submitJob: async (
    params: Record<string, unknown>,
    years: number,
    escapeFactor = 1.0,
  ): Promise<{ ok: boolean; job_id?: string; status?: string; error?: string }> => {
    const r = await fetch(`${AGENT_URL}/job/submit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: '', params, years, escape_factor: escapeFactor }),
    });
    return r.json();
  },

  // Returns a ReadableStream for SSE parsing — caller owns the reader
  chatStream: (req: ChatRequest): Promise<Response> => {
    return fetch(`${AGENT_URL}/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    });
  },
};

export { AGENT_URL };
