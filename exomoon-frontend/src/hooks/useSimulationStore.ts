'use client';
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { DEFAULT_PARAMS } from '@/lib/paramDefaults';
import { DEFAULT_EDA_VARS as EDA_DEFAULTS } from '@/lib/trajectoryMath';
import type {
  SystemParams, TrajectoryFrame, SimulationMeta,
  ChatMessage, JobStatusState, MlPrediction,
} from '@/lib/types';

export type ParamStatus = 'none' | 'clean' | 'dirty';

const STELLAR_KEYS = new Set(['Ts', 'rs_solar', 'ms_solar']);
const PLANET_KEYS  = new Set(['mp_earth', 'dp_cgs', 'ap_AU', 'ep']);
const MOON_KEYS    = new Set(['mm_earth', 'am_hill', 'em', 'moon_retrograde']);

interface SimulationStore {
  // ── Parameters ──────────────────────────────────────────────────────────────
  params: SystemParams;
  simYears: number;
  setParam: (key: keyof SystemParams, value: number | boolean) => void;
  setParams: (params: Partial<SystemParams>) => void;
  setSimYears: (y: number) => void;

  // ── Per-object param status (drives FAB status icons) ───────────────────────
  starStatus:   ParamStatus;
  planetStatus: ParamStatus;
  moonStatus:   ParamStatus;
  setAllParamsClean: () => void;

  // ── Job lifecycle ────────────────────────────────────────────────────────────
  jobId: string | null;
  jobStatus: JobStatusState;
  jobElapsedSeconds: number;
  presignedUrls: Record<string, string>;
  setJob: (jobId: string) => void;
  updateJobStatus: (status: string, elapsed: number, urls?: Record<string, string>) => void;
  clearJob: () => void;

  // ── Trajectory data ──────────────────────────────────────────────────────────
  trajectoryFrames: TrajectoryFrame[] | null;
  simMeta: SimulationMeta | null;
  setTrajectoryData: (frames: TrajectoryFrame[], meta: SimulationMeta) => void;
  clearTrajectoryData: () => void;

  // ── Simdata (opaque base64 for chat context) ─────────────────────────────────
  simdataB64: string | null;
  setSimdata: (s: string | null) => void;

  // ── EDA ──────────────────────────────────────────────────────────────────────
  edaVars: string[];
  edaPlotType: 'line' | 'scatter';
  edaNormalize: boolean;
  showHzOverlay: boolean;
  showHillOverlay: boolean;
  setEdaVars: (vars: string[]) => void;
  setEdaPlotType: (t: 'line' | 'scatter') => void;
  setEdaNormalize: (n: boolean) => void;
  setShowHzOverlay: (v: boolean) => void;
  setShowHillOverlay: (v: boolean) => void;

  // ── Moon density (visual/UI only — not sent to simulation backend) ───────────
  dmCgs: number;  // moon density in g/cm³, default Earth density
  setDmCgs: (v: number) => void;

  // ── Simdata alias ─────────────────────────────────────────────────────────────
  setSimdataB64: (s: string | null) => void;

  // ── ML Prediction ────────────────────────────────────────────────────────────
  mlPrediction: MlPrediction | null;  // 2D stability-habitability map from inference
  mlMassIdx: number;                  // current slider index into mlPrediction.mmGrid
  setMlPrediction: (p: MlPrediction | null) => void;
  setMlMassIdx: (i: number) => void;

  // ── Chat ─────────────────────────────────────────────────────────────────────
  chatMessages: ChatMessage[];
  addChatMessage: (msg: Omit<ChatMessage, 'timestamp'> & { id: string }) => void;
  appendToLastAssistant: (token: string) => void;
  finalizeChatMessage: (id: string, errorContent?: string) => void;
  finalizeLastAssistant: () => void;
  clearChat: () => void;
}

export const useSimulationStore = create<SimulationStore>()(
  persist(
    (set, get) => ({
      // ── Parameters ───────────────────────────────────────────────────────────
      params: { ...DEFAULT_PARAMS },
      simYears: 0,
      setParam: (key, value) =>
        set(s => ({
          params: { ...s.params, [key]: value },
          ...(STELLAR_KEYS.has(key as string) ? { starStatus:   'dirty' as ParamStatus } : {}),
          ...(PLANET_KEYS.has(key  as string) ? { planetStatus: 'dirty' as ParamStatus } : {}),
          ...(MOON_KEYS.has(key    as string) ? { moonStatus:   'dirty' as ParamStatus } : {}),
        })),
      setParams: (partial) =>
        set(s => ({
          params: { ...s.params, ...partial },
          starStatus:   'dirty' as ParamStatus,
          planetStatus: 'dirty' as ParamStatus,
        })),
      setSimYears: (y) => set({ simYears: y }),

      // ── Per-object param status ───────────────────────────────────────────────
      starStatus:   'none',
      planetStatus: 'none',
      moonStatus:   'none',
      setAllParamsClean: () => set({ starStatus: 'clean', planetStatus: 'clean', moonStatus: 'clean' }),

      // ── Job lifecycle ─────────────────────────────────────────────────────────
      jobId: null,
      jobStatus: 'idle',
      jobElapsedSeconds: 0,
      presignedUrls: {},
      setJob: (jobId) => set({
        jobId,
        jobStatus: 'running',
        jobElapsedSeconds: 0,
        presignedUrls: {},
      }),
      updateJobStatus: (status, elapsed, urls) => {
        const mapped: JobStatusState =
          status === 'SUCCEEDED' ? 'succeeded' :
          status === 'FAILED' || status === 'TIMED_OUT' ? 'failed' :
          'running';
        set({
          jobStatus: mapped,
          jobElapsedSeconds: elapsed,
          presignedUrls: urls ?? get().presignedUrls,
          ...(mapped === 'succeeded' ? {
            starStatus:   'clean' as ParamStatus,
            planetStatus: 'clean' as ParamStatus,
            moonStatus:   'clean' as ParamStatus,
          } : {}),
        });
      },
      clearJob: () => set({ jobId: null, jobStatus: 'idle', jobElapsedSeconds: 0, presignedUrls: {} }),

      // ── Trajectory data ───────────────────────────────────────────────────────
      trajectoryFrames: null,
      simMeta: null,
      setTrajectoryData: (frames, meta) => set({ trajectoryFrames: frames, simMeta: meta }),
      clearTrajectoryData: () => set({ trajectoryFrames: null, simMeta: null }),

      // ── Simdata ───────────────────────────────────────────────────────────────
      simdataB64: null,
      setSimdata: (s) => set({ simdataB64: s }),
      setSimdataB64: (s) => set({ simdataB64: s }),

      // ── Moon density (UI/visual only) ─────────────────────────────────────────
      dmCgs: 5.5,
      setDmCgs: (v) => set({ dmCgs: v, moonStatus: 'dirty' as ParamStatus }),

      // ── EDA ───────────────────────────────────────────────────────────────────
      edaVars: [...EDA_DEFAULTS],
      edaPlotType: 'line',
      edaNormalize: false,
      showHzOverlay: false,
      showHillOverlay: false,
      setEdaVars: (vars) => set({ edaVars: vars }),
      setEdaPlotType: (t) => set({ edaPlotType: t }),
      setEdaNormalize: (n) => set({ edaNormalize: n }),
      setShowHzOverlay: (v) => set({ showHzOverlay: v }),
      setShowHillOverlay: (v) => set({ showHillOverlay: v }),

      // ── ML Prediction ─────────────────────────────────────────────────────────
      mlPrediction: null,
      mlMassIdx: 0,
      setMlPrediction: (p) => set({ mlPrediction: p, mlMassIdx: 0 }),
      setMlMassIdx: (i) => set({ mlMassIdx: i }),

      // ── Chat ──────────────────────────────────────────────────────────────────
      chatMessages: [],
      addChatMessage: (msg) => {
        const full: ChatMessage = { ...msg, timestamp: Date.now() };
        set(s => ({ chatMessages: [...s.chatMessages, full] }));
      },
      appendToLastAssistant: (token) => {
        set(s => {
          const msgs = [...s.chatMessages];
          for (let i = msgs.length - 1; i >= 0; i--) {
            if (msgs[i].role === 'assistant') {
              msgs[i] = { ...msgs[i], content: msgs[i].content + token };
              break;
            }
          }
          return { chatMessages: msgs };
        });
      },
      finalizeChatMessage: (id, errorContent) => {
        set(s => ({
          chatMessages: s.chatMessages.map(m =>
            m.id === id
              ? { ...m, streaming: false, ...(errorContent !== undefined ? { content: errorContent } : {}) }
              : m
          ),
        }));
      },
      finalizeLastAssistant: () => {
        set(s => {
          const msgs = [...s.chatMessages];
          for (let i = msgs.length - 1; i >= 0; i--) {
            if (msgs[i].role === 'assistant') {
              msgs[i] = { ...msgs[i], streaming: false };
              break;
            }
          }
          return { chatMessages: msgs };
        });
      },
      clearChat: () => set({ chatMessages: [] }),
    }),
    {
      name: 'exomoon-sim-store',
      // Only persist params and EDA prefs across page refreshes
      partialize: (s) => ({
        params: s.params,
        simYears: s.simYears,
        edaVars: s.edaVars,
        edaPlotType: s.edaPlotType,
        edaNormalize: s.edaNormalize,
        showHzOverlay: s.showHzOverlay,
        showHillOverlay: s.showHillOverlay,
        dmCgs: s.dmCgs,
      }),
    }
  )
);
