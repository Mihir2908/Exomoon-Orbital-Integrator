# Exomoon Orbital Integrator — CLAUDE.md

**Last Updated:** May 13, 2026
**Status:** Production-stable (core functionality verified, multi-turn queries working)
**Pending Issues:** CSV file retrieval via chatbot UI

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Physics Foundation](#physics-foundation)
4. [Core Modules](#core-modules)
5. [API Reference](#api-reference)
6. [Data Flow Pipeline](#data-flow-pipeline)
7. [AWS Workflow](#aws-workflow)
8. [Agent Service Workflows](#agent-service-workflows)
9. [Deployment Guide](#deployment-guide)
10. [Environment Variables](#environment-variables)
11. [Pending Issues](#pending-issues)
12. [Next Steps / Roadmap](#next-steps--roadmap)

---

## Executive Summary

The **Exomoon Orbital Integrator** is a full-stack distributed system built around **three distinct services**, each with its own compute, connected through HTTP/SSE and shared S3 storage:

---

### Service 1: Plotly Dash UI Frontend (App Runner, port 8050)

The sole user-facing surface. Responsibilities:
- Configure simulation parameters (sliders, dropdowns, NASA archive fetch)
- Trigger backend simulations via the "Run Simulation" button (calls Step Functions directly)
- Host the chat drawer — the only entry point to the Agent Service
- Poll job status and reconstruct the orbital animation **locally** from `traj.csv` (never fetches `animation.html` from S3)
- Render orbit animations and EDA plots (page 2 at `/eda`)
- Export `traj.csv` via the "Export CSV" button

---

### Service 2: Step Functions Simulation Backend (Step Functions + ECS/Fargate, `run_job.py`)

A pure simulation compute pipeline — knows nothing about the UI or the agent. Responsibilities:
- Triggered by either Dash (native UI path) or the Agent Service (chat path) — both write `params.json` to S3 and call `sf.start_execution()`
- Step Functions orchestrates: reads `params.json` from S3 inputs → spawns ECS/Fargate task running `run_job.py` → writes outputs to S3
- `run_job.py` runs the Numba leapfrog physics engine and generates all four outputs:
  - `traj.csv` — trajectory data (positions + velocities); fetched by Dash to rebuild animation
  - `summary.json` — metadata (HZ bounds, `rhill_AU`, stability); fetched by Dash
  - `traj.pkl` — base64-packed simdata; fetched by Agent Service for chat follow-ups
  - `animation.html` — standalone Plotly animation; S3-only artifact, never sent to Dash

---

### Service 3: Agent Service Backend (ECS/Fargate behind NLB, port 8000, `agent_service.py`)

Activated only when the user engages the chatbot on Dash. Responsibilities:
- Receives chat messages from Dash via `POST /chat/stream` (SSE)
- Runs Claude multi-turn reasoning to decide: fetch exoplanet data, trigger a simulation, or answer from cached simdata
- When a simulation is needed: writes `params.json` to S3 and calls `sf.start_execution()` (same Step Functions backend) — returns immediately, no blocking
- Manages `SessionCache` (in-memory per session): stores packed simdata after job completion for instant follow-up queries
- Exposes `/job/{job_id}/status` and `/job/{job_id}/retrieve_simdata` for Dash polling
- Streams token-by-token responses back to Dash via SSE

---

### Service Connectivity

```
                    ┌──────────────────────────────┐
                    │   Plotly Dash UI (App Runner) │
                    │         port 8050             │
                    └────────────┬─────────────────┘
                                 │
               ┌─────────────────┼──────────────────────┐
               │ Native UI path  │ Chat path             │
               │ (direct)        │ (via chatbot)         │
               ▼                 ▼                       │
    ┌──────────────────┐  ┌──────────────────────┐      │
    │  Step Functions  │  │   Agent Service       │      │
    │  Simulation      │◄─│   (ECS/Fargate,NLB)   │      │
    │  Backend         │  │   port 8000           │      │
    │  (ECS/Fargate)   │  └──────────────────────┘      │
    └────────┬─────────┘           │                     │
             │                     │ (both paths share)  │
             └─────────────────────┘                     │
                          │                              │
                          ▼                              │
              ┌───────────────────────┐                  │
              │   S3 (shared store)   │                  │
              │  inputs/{job_id}/     │                  │
              │  outputs/{job_id}/    │──────────────────┘
              │    traj.csv  ◄── Dash fetches
              │    summary.json ◄── Dash fetches
              │    traj.pkl ◄── Agent Service fetches
              │    animation.html (S3-only artifact)
              └───────────────────────┘
```

**Key rules**:
- Dash ↔ Agent Service: HTTP/SSE only (chat path)
- Dash ↔ Step Functions: boto3 direct (native UI path)
- Agent Service ↔ Step Functions: boto3 direct (when chat triggers a sim)
- Step Functions ↔ Agent Service: **never** — they only share S3
- S3 is the single shared data store between all three services

**Key Architectural Decision**: Non-blocking job submission + local simdata caching enables responsive chat with instant feedback while jobs compute in background (~5–60 seconds per simulation).

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         EXOMOON ORBITAL INTEGRATOR                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   FRONTEND LAYER                            │   │
│  │  ┌──────────────────────────────────────────────────────┐   │   │
│  │  │  Dash Web UI (run_dash.py:8050)                      │   │   │
│  │  │  - Simulation controls (params, years, escape_factor)│   │   │
│  │  │  - Chat drawer (SSE streaming, real-time updates)    │   │   │
│  │  │  - Polling callback (job status every 5 seconds)     │   │   │
│  │  │  - Animation + EDA visualization                     │   │   │
│  │  └──────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│            │                                                         │
│            │ REST/SSE: POST /chat, GET /stream                      │
│            ▼                                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   AGENT SERVICE LAYER                       │   │
│  │  ┌──────────────────────────────────────────────────────┐   │   │
│  │  │  FastAPI Agent Service (agent_service.py:8000)       │   │   │
│  │  │  Core Components:                                    │   │   │
│  │  │  • SessionCache: in-memory simdata + params cache    │   │   │
│  │  │  • Claude SDK: multi-turn LLM reasoning              │   │   │
│  │  │  • Tool Executor: run_simulation, export_csv, etc    │   │   │
│  │  │  • Job Orchestrator: Step Functions integration      │   │   │
│  │  └──────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│            │                                                         │
│            │ boto3: start_execution, describe_execution             │
│            ▼                                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   AWS BACKEND LAYER                         │   │
│  │  ┌──────────────────────────────────────────────────────┐   │   │
│  │  │  AWS Step Functions State Machine                    │   │   │
│  │  │  ├─ Input: params.json from S3                       │   │   │
│  │  │  ├─ Spawn: ECS/Batch task (run_job.py)              │   │   │
│  │  │  └─ Output: traj.csv, summary.json, traj.pkl,       │   │   │
│  │  │            animation.html (S3-only, not sent to UI)  │   │   │
│  │  └──────────────────────────────────────────────────────┘   │   │
│  │                                                               │   │
│  │  ┌──────────────────────────────────────────────────────┐   │   │
│  │  │  S3 Storage (EXOMOON_BUCKET)                         │   │   │
│  │  │  ├─ inputs/{job_id}/params.json                      │   │   │
│  │  │  ├─ outputs/{job_id}/                                │   │   │
│  │  │  │  ├─ traj.pkl (serialized simulation)              │   │   │
│  │  │  │  ├─ traj.csv  ← Dash fetches this                 │   │   │
│  │  │  │  ├─ summary.json ← Dash fetches this              │   │   │
│  │  │  │  ├─ animation.html ← S3 only (NOT sent to Dash)   │   │   │
│  │  │  │  ├─ job_metadata.json (execution_arn)             │   │   │
│  │  │  │  └─ COMPLETE (marker file)                        │   │   │
│  │  │  └─ Retention: 24h presigned URLs                    │   │   │
│  │  └──────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│            │                                                         │
│            │ Batch Job Container (docker image)                     │
│            ▼                                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   SIMULATION ENGINE                         │   │
│  │  ┌──────────────────────────────────────────────────────┐   │   │
│  │  │  Physics Modules (exomoon/*)                         │   │   │
│  │  │  • simulation.py: run_simulation_for_years()         │   │   │
│  │  │  • integrator.py: Numba-compiled leapfrog (3-body)   │   │   │
│  │  │  • initial_conditions.py: orbital bootstrap          │   │   │
│  │  │  • moon_stability.py: escape detection + analysis    │   │   │
│  │  │  • eda.py: pack_sim/unpack_sim (serialization)       │   │   │
│  │  └──────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### CRITICAL: Animation Reconstruction Flow

**animation.html is generated by the backend (`run_job.py`) and uploaded to S3, but Dash NEVER fetches it.**

Instead, when the Native UI path job completes, `load_s3_results` in `run_dash.py` (line ~1303):
1. Fetches `traj.csv` from S3 → parses position/velocity columns into numpy arrays
2. Fetches `summary.json` from S3 → extracts HZ bounds, rhill_AU, stability info
3. Reconstructs a local `sim` dict from those arrays
4. Calls `build_animation()` locally to render the Plotly animation in the browser

The `animation.html` on S3 exists as a standalone artifact (for direct download/sharing), but the Dash UI always rebuilds the animation client-side from trajectory data.

### User-Perspective Data Flow

```
DATA FLOW (User Perspective):
1. User types query in Dash chat + submits
2. Dash SSE streams: "Chat message received..." (instant)
3. Agent service receives message → calls _chat_with_claude()
4. Claude decides: run sim? retrieve from cache? answer directly?
5. If SIM NEEDED: calls _start_backend_job() → returns {job_id, status: "submitted"} immediately
6. Agent responds: "Started job [job_id]" via SSE
7. Dash polling callback wakes every 5s → checks /job/{job_id}/status
8. When status == "SUCCEEDED":
   - Polls gets job URLs
   - Calls /job/{job_id}/retrieve_simdata → caches in agent session
   - Chat updates: "✅ Results ready! Analyzing..."
9. User asks follow-up: "What's the escape time?" → Claude reads cached simdata → instant response
```

---

## Physics Foundation

### Gravitational Acceleration

All three bodies experience mutual gravitational attraction. Acceleration on body A due to body B:

$$\vec{a}_A = -\frac{\mu_B \vec{r}}{|\vec{r}|^3}$$

Where:
- $\mu_B = GM_B$ is the gravitational parameter of body B
- $\vec{r} = \vec{r}_A - \vec{r}_B$ is the position vector from B to A
- Units: G = 4π² (scaled to AU, years convention; see `constants.py`)

**Code Implementation** (`integrator.py`):
```python
@njit(fastmath=True, cache=True)
def _accel(pos_a, pos_b, mu_b):
    r0, r1, r2 = pos_a[0] - pos_b[0], pos_a[1] - pos_b[1], pos_a[2] - pos_b[2]
    r2n = r0*r0 + r1*r1 + r2*r2
    r = r2n ** 0.5
    r3 = r * r2n
    return np.array([-mu_b * r0 / r3, -mu_b * r1 / r3, -mu_b * r2 / r3])
```

**Numba Compilation**: `@njit(fastmath=True, cache=True)` compiles to native machine code at first call, enabling 100–1000x speedup for tight numerical loops. Cache persists across runs in `__pycache__`.

---

### Leapfrog Integration (Symplectic Integrator)

Symplectic integrator — conserves energy exceptionally well over long timescales, critical for orbital mechanics.

**Algorithm** (per timestep Δt):
1. **Position half-kick**: r += v · Δt/2
2. **Velocity kick**: v += a(r) · Δt
3. **Position half-kick**: r += v · Δt/2

Applied to all three bodies in series per timestep.

**Code** (`integrator.py`):
```python
for i in range(n_steps):
    # Planet
    p2_mp = p_mp + v_mp * half_dt
    a_mp = _accel(p2_mp, p_ms, ms) + _accel(p2_mp, p_mm, mm)
    v_mp = v_mp + a_mp * dt
    p_mp = p2_mp + v_mp * half_dt
    
    # Moon (same pattern)
    # Star (same pattern)
    
    xyz_mp[i] = p_mp  # Store trajectory
```

**Timestep selection**: `dt = min(T_moon/100, 1/20000)` — balances moon orbital resolution and year-scale accuracy.

---

### Hill Radius & Moon Stability

**Hill Radius** (sphere of gravitational dominance for planet relative to star):

$$R_{Hill} = a_p (1-e_p) \left(\frac{M_p}{3M_*}\right)^{1/3}$$

**Moon Stability**: Moon is stable if:
$$\max_t |\vec{r}_{moon} - \vec{r}_{planet}| \leq escape\_factor \times R_{Hill}$$

If violated, escape occurs; code logs `escape_time` via linear interpolation.

**Code** (`moon_stability.py`):
```python
moon_rel = traj["xyzarr_mm"] - traj["xyzarr_mp"]
r_rel = np.linalg.norm(moon_rel[:, :2], axis=1)  # 2D norm
stable = np.max(r_rel) <= escape_factor * rhill_AU
```

---

### Habitable Zone

Star's habitable zone defined by radiative balance from stellar luminosity.

$$L_* = 4\pi R_*^2 \sigma T_*^4$$

**Inner bound** (greenhouse runaway): $a_{inner} = \sqrt{L_* / (4\pi \cdot 1.1 \cdot F_\oplus)}$

**Outer bound** (max greenhouse): $a_{outer} = \sqrt{L_* / (4\pi \cdot 0.5 \cdot F_\oplus)}$

Where $F_\oplus = 1370$ W/m² (Earth's insolation).

**Code** (`habitable_zone.py`):
```python
def hz_bounds_au(Ts_K: float, rs_m: float):
    L_star = 4 * np.pi * rs_m**2 * stefboltz * Ts_K**4
    a_inner_m = np.sqrt(L_star / (4 * np.pi * 1.1 * F_earth))
    a_outer_m = np.sqrt(L_star / (4 * np.pi * 0.5 * F_earth))
    return a_inner_m / au, a_outer_m / au
```

---

## Core Modules

### `simulation.py` & `integrator.py`: Physics Engine

#### `run_simulation_for_years(p: SystemParams, years: float) → dict`

1. Initialize orbital state via `initial_state(p)` (barycenter coordinates, velocities)
2. Compute moon orbital period: $T_{mm} = 2\pi \sqrt{a_m^3 / (M_p + M_m)}$
3. Set `dt = min(T_mm/100, 1/20000)`
4. Call `leapfrog_integrate(state, t_end=years, dt)` → returns trajectory dict
5. Compute HZ bounds via `hz_bounds_au()`
6. Return: `{params, state, traj, dt, t_end, a_inner_au, a_outer_au}`

**Output Structure**:
```python
{
    "traj": {
        "xyzarr_mp": np.array([N, 3]),  # Planet positions over time
        "xyzarr_ms": np.array([N, 3]),  # Star positions
        "xyzarr_mm": np.array([N, 3]),  # Moon positions
        "velarr_mp": np.array([N, 3]),  # Planet velocities
        "velarr_ms": np.array([N, 3]),
        "velarr_mm": np.array([N, 3]),
    },
    "dt": float,        # Timestep (years)
    "t_end": float,     # Final time (years)
    "a_inner_au": float,
    "a_outer_au": float,
}
```

---

### `eda.py`: Data Serialization

**Purpose**: Convert simulations to/from compact base64-encoded JSON for transport via HTTP/S3.

#### `pack_sim(sim: dict) → str`
Serializes trajectory dict to base64 string. Converts numpy arrays → list → JSON → base64.

#### `unpack_sim(packed: str) → dict`
Deserializes base64 string back to dict with numpy arrays.

#### `traj_to_frame(sim: dict) → DataFrame`
Converts trajectory to columnar format for CSV export.

```python
def traj_to_frame(sim: dict):
    traj = sim["traj"]
    dt = sim["dt"]
    n = len(traj["xyzarr_mp"])
    t = np.arange(n) * dt  # Time array (years)
    
    rel_mm_mp = traj["xyzarr_mm"] - traj["xyzarr_mp"]  # Moon relative to planet
    moon_planet_dist = np.linalg.norm(rel_mm_mp[:, :2], axis=1)
    
    data = {
        "t_years": t,
        "star_x": traj["xyzarr_ms"][:, 0],
        # ... all positions and velocities ...
        "moon_planet_dist": moon_planet_dist,
    }
    return pd.DataFrame(data)  # or dict-of-arrays fallback
```

**CSV Output Columns**: `t_years`, `star_x/y/z`, `planet_x/y/z`, `moon_x/y/z`, `moon_planet_dist`, `planet_star_dist`, plus all velocity components.

---

### `agent_service.py`: Core Chatbot Engine

**Purpose**: FastAPI service (port 8000) handling Claude-powered multi-turn conversations, job orchestration, and simdata caching. Triggered by the Dash chatbot frontend. Depending on the scenario, itself triggers the AWS Step Functions backend.

#### `SessionCache` Class

```python
class SessionCache:
    def __init__(self):
        self.last_job_id: Optional[str] = None
        self.last_output_prefix: Optional[str] = None
        self.cached_simdata: Optional[str] = None  # Base64-encoded packed sim
        self.cached_params: Dict[str, Any] = {}
    
    def update_job(self, job_id: str, output_prefix: str):
        self.last_job_id = job_id
        self.last_output_prefix = output_prefix
        self.cached_simdata = None  # Clear old data when new job starts
    
    def set_simdata(self, simdata: str, params: Dict[str, Any]):
        self.cached_simdata = simdata
        self.cached_params = params
    
    def get_cached(self) -> tuple[Optional[str], Dict]:
        return self.cached_simdata, self.cached_params
```

All tool handlers (`export_csv`, `stability_from_simdata`, `eda_plot`) fall back to `_session.get_cached()` when no simdata is provided directly.

#### `_start_backend_job()` — KEY: Non-Blocking

Submits job to AWS Step Functions and returns **immediately** (~200ms). No blocking poll.

```python
def _start_backend_job(params, years, check_stability=False, escape_factor=1.0):
    job_id = f"agent-{uuid.uuid4().hex[:12]}"
    inp_prefix = f"inputs/{job_id}"
    out_prefix = f"outputs/{job_id}"
    
    # Upload params to S3
    s3.put_object(Bucket=BUCKET, Key=f"{inp_prefix}/params.json",
                  Body=json.dumps(params_dict).encode())
    
    # Submit to Step Functions (ASYNC - returns immediately)
    exec_resp = sf.start_execution(
        stateMachineArn=STATE_MACHINE_ARN,
        name=job_id,
        input=json.dumps({
            "inputS3Prefix": f"s3://{BUCKET}/{inp_prefix}",
            "outputS3Prefix": f"s3://{BUCKET}/{out_prefix}",
        })
    )
    
    _session.update_job(job_id, out_prefix)
    
    # ✅ Return immediately - NO polling
    return {"ok": True, "job_id": job_id, "status": "submitted"}
```

#### `/job/{job_id}/retrieve_simdata` Endpoint

Called by Dash when job completes. Fetches `traj.pkl` from S3, caches in session.

```python
@app.get("/job/{job_id}/retrieve_simdata")
def retrieve_job_simdata(job_id: str):
    obj = s3.get_object(Bucket=BUCKET, Key=f"outputs/{job_id}/traj.pkl")
    simdata = obj["Body"].read().decode()
    _session.set_simdata(simdata, {})
    return {"ok": True, "simdata_cached": True, "message": f"Cached ({len(simdata)} chars)"}
```

#### Tools Available to Claude

| Tool | Purpose | AWS Used? |
|------|---------|-----------|
| `fetch_exoplanet(name)` | NASA archive lookup | No |
| `run_simulation(params, years, escape_factor)` | Submit AWS job | Yes (S3 + Step Fn) |
| `stability_from_simdata(simdata, years, escape_factor)` | Stability analysis from cache | No |
| `export_csv(simdata)` | CSV export | No (reads cache) |
| `eda_plot(simdata, variables)` | EDA plot HTML | No |

---

### `run_dash.py`: Interactive Dashboard

**Purpose**: Multi-mode dashboard enabling simulations via (1) native Dash UI controls, (2) chat agent queries, (3) NASA archive parameter fetch.

**Parameter Configuration UI** (Left panel):
- NASA dropdown: type 3+ chars → autocomplete → "Fetch from NASA" button
- Sliders: `Ts`, `rs_solar`, `ms_solar`, `mp_earth`, `dp_cgs`, `ap_AU`, `ep`, `mm_earth`, `am_hill`, `em`
- Radio: `moon_dir` (prograde/retrograde)
- Input: `sim_years` (0 = 1 planet orbit)
- Buttons: "Run Simulation", "Export CSV"
- Link: "Open EDA Plots →"

#### Mode 1: Native Dash UI (Direct Execution)

**AWS Path** (`run_cb` in `run_dash.py`, triggered by Run button or `kick="run"`):
1. Build `params.json` → upload to S3 `inputs/{job_id}/`
2. `sf.start_execution()` → Step Functions
3. Return placeholder figure, enable `status-interval` poller (5s)
4. `poll_job_status` callback: checks S3 for `traj.csv` presence, then `sf.describe_execution()`
5. On SUCCEEDED: `load_s3_results` callback fires
   - Reads `traj.csv` from S3 → numpy arrays
   - Reads `summary.json` from S3 → HZ bounds, rhill_AU
   - Reconstructs local `sim` dict
   - Calls `build_animation()` locally → Plotly figure rendered in browser
   - Packs sim → `dcc.Store("simdata")`
   - **animation.html is NOT fetched from S3 — animation is always rebuilt locally**

**Local Path** (no AWS):
- `run_simulation_for_years(p, yrs)` or `run_simulation(p)`
- `build_animation(traj, a_inner, a_outer)` → figure
- `pack_sim(sim)` → `dcc.Store("simdata")`

#### Mode 2: Chat Agent (Conversational Execution)

User message → `send_chat_message` callback → `POST {AGENT_SERVICE_URL}/chat/stream` (SSE).

**SSE event types**: `meta` (mode + job_id), `token` (streaming text), `done` (final payload).

**Job tracking flow**:
1. `send_chat_message` extracts `job_id` from `done` payload → stored in `dcc.Store("chat-job-id")`
2. `track_chat_job_id` stores job_id from chat history metadata
3. `toggle_job_poller` enables `chat-job-poller` interval (5s) when job_id exists
4. `poll_agent_job_status` fires every 5s:
   - `GET {AGENT_SERVICE_URL}/job/{job_id}/status`
   - On SUCCEEDED: calls `GET {AGENT_SERVICE_URL}/job/{job_id}/retrieve_simdata` → agent caches simdata
   - Updates chat messages with status (⏳ Running / ✅ Complete / ❌ Failed)
5. `stop_polling_on_completion` disables poller when terminal state detected (✅/❌ in message)

**Session state**: Chat agent path uses agent service `SessionCache` (remote). Separate from native UI's `dcc.Store("simdata")`. Both can coexist without conflict.

#### Mode 3: NASA Archive (Parameter Fetch)

`populate_from_url_or_nasa` callback handles:
- URL query string `?pl=Kepler-442+b&run=1` → auto-populates + optionally auto-runs
- `pl_picker` dropdown search (3+ chars → `search_planets()` → typeahead)
- "Fetch from NASA" button → `fetch_system_by_planet()` → fills all controls

`?run=1` only triggers autorun if no simdata exists yet (prevents re-runs on navigation).

#### Animation & Visualization

`build_animation(traj, a_inner_au, a_outer_au, dt, t_end)` in `exomoon/plotting/anim.py`:
- Left panel: Full system orbit (Star, Planet, Moon trajectories over time)
- Right inset: Moon relative to planet (zoom perspective)
- Annotations: Hill radius (red circle), moon semi-major axis, habitable zones (green bands)
- Slider: Scrub through timesteps or auto-play
- Trail effect: Orbital paths with fading history
- `frame_duration=0`, `transition_duration=0`, `speed_factor` parameter for playback speed

#### EDA Page (`/eda` route)

- Dropdown: select variables (multi-select): `moon_planet_dist`, `planet_star_dist`, speeds, positions
- Plot type: Line / Scatter
- Normalize checkbox: divide by max for multi-variable comparison
- `eda_plot` callback: `unpack_sim()` → `traj_to_frame()` → Plotly figure
- Default vars auto-selected: `moon_planet_dist`, `planet_star_dist`, `moon_speed`
- `uirevision="eda"` preserves zoom/pan state between updates

#### Export CSV (Native UI)

```python
@app.callback(Output("download-csv", "data"), Input("export-btn", "n_clicks"), State("simdata", "data"))
def export_csv(n_clicks, packed):
    sim = unpack_sim(packed)
    frame = traj_to_frame(sim)
    csv_bytes = to_csv_bytes(frame)
    return dcc.send_bytes(csv_bytes, "exomoon_simulation.csv")
```

---

### `mcp_server.py`: Model Context Protocol (MCP) Server

**Purpose**: Expose physics simulation tools as MCP resources for Claude Desktop. Local only — no HTTP, no AWS, no web service needed.

**Transport**: stdio (subprocess of Claude Desktop)

**Config** (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "exomoon": {
      "command": "python -m exomoon.mcp_server",
      "env": {"PYTHONPATH": "/app/src"}
    }
  }
}
```

**Tools Exposed**: `run_simulation`, `stability_from_simdata`, `export_csv`, `eda_plot`

**Key Difference vs Agent Service**:
- Agent Service: Web-based, FastAPI, HTTP endpoints, session caching, AWS-backed
- MCP Server: Desktop-based, stdio transport, local computation only, no session state

---

## API Reference

### `POST /chat/stream` — Streaming SSE

**Body**:
```json
{
  "message": "Is a 0.01 Earth-mass moon stable around Kepler-442 b?",
  "simdata": null,
  "params": {},
  "years": 10.0,
  "escape_factor": 1.0
}
```

**Response** (Server-Sent Events):
```
data: {"type": "meta", "payload": {"mode": "run_sim", "job_id": "agent-abc123"}}
data: {"type": "token", "payload": {"token": "Started "}}
data: {"type": "token", "payload": {"token": "job "}}
...
data: {"type": "done", "payload": {"message": "...", "job_id": "...", "simdata": "..."}}
```

---

### `GET /job/{job_id}/status`

**Response** (RUNNING):
```json
{
  "ok": true,
  "job_id": "agent-abc123",
  "status": "RUNNING",
  "elapsed_seconds": 15,
  "urls": {}
}
```

**Response** (SUCCEEDED):
```json
{
  "ok": true,
  "job_id": "agent-abc123",
  "status": "SUCCEEDED",
  "elapsed_seconds": 45,
  "urls": {
    "traj.csv": "https://bucket.s3.amazonaws.com/.../traj.csv?X-Amz-Signature=...",
    "animation.html": "https://...",
    "summary.json": "https://..."
  }
}
```

---

### `GET /job/{job_id}/retrieve_simdata`

**Response**:
```json
{
  "ok": true,
  "job_id": "agent-abc123",
  "simdata_cached": true,
  "message": "Cached (125432 chars)"
}
```

---

## Data Flow Pipeline

### Complete Multi-Turn Query Flow (Chat Agent Path)

```
USER: "Is moon at Kepler-442 b stable over 100 years?"
         │
         ▼ POST /chat/stream (SSE)
┌─────────────────────────────────────────────────────────────────┐
│ Agent Service receives ChatRequest                               │
│ • message, simdata=None, params={}, years=100                   │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼ Claude Tool Call Decision
┌─────────────────────────────────────────────────────────────────┐
│ Claude evaluates → calls run_simulation tool                     │
│ → _start_backend_job() fires                                     │
│   1. Upload params.json → S3 inputs/                            │
│   2. sf.start_execution() → ASYNC                               │
│   3. _session.update_job(job_id, out_prefix)                    │
│   4. RETURN {"status": "submitted"} in ~200ms                   │
└──────────────────────┬──────────────────────────────────────────┘
                       │ SSE → Dash: "Started job agent-xyz789"
                       │
                       ▼ AWS Backend (parallel)
┌─────────────────────────────────────────────────────────────────┐
│ Step Functions → ECS/Fargate → run_job.py                       │
│ → run_simulation_for_years(p, 100)                              │
│ → Numba leapfrog integration                                     │
│ → Generate: traj.csv, summary.json, traj.pkl, animation.html   │
│ → Upload all to S3 outputs/                                     │
│ ⏱️ Duration: 10–60 seconds                                      │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼ Dash Polling (5s interval via chat-job-poller)
┌─────────────────────────────────────────────────────────────────┐
│ Poll #N: status = SUCCEEDED                                      │
│ ✅ → Call /job/.../retrieve_simdata                             │
│ → _session.set_simdata(packed_sim, {})                          │
│ Chat updates: "✅ Results ready!"                               │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼ Follow-up query (instant)
┌─────────────────────────────────────────────────────────────────┐
│ User: "What's the escape time?"                                  │
│ → Claude: stability_from_simdata tool                           │
│ → _session.get_cached() → unpack_sim() → instant result        │
│ ✅ Response in <100ms, no re-simulation                         │
└─────────────────────────────────────────────────────────────────┘
```

### Native UI Job Flow (AWS Path)

```
Run Button Click
         │
         ▼ run_cb callback
├─ params.json → S3 inputs/{job_id}/
├─ sf.start_execution() → Step Functions
├─ Return placeholder figure, enable status-interval (5s)
         │
         ▼ poll_job_status (5s)
├─ Check S3 for traj.csv existence (head_object)
├─ OR sf.describe_execution() → check status
├─ On SUCCEEDED: load_s3_results fires
         │
         ▼ load_s3_results
├─ s3.get_object → traj.csv → DataFrame → numpy arrays
├─ s3.get_object → summary.json → HZ bounds, rhill_AU
├─ Reconstruct sim dict locally
├─ build_animation() locally → Plotly figure (animation.html NOT fetched)
└─ pack_sim(sim) → dcc.Store("simdata")
```

### SessionCache Lifecycle

```
Session Created: last_job_id=None, cached_simdata=None

Message 1: "Run simulation for 10 years"
 → _start_backend_job() → _session.update_job("agent-abc", ...)
   cached_simdata = None (CLEARED)

Polling: SUCCEEDED → /retrieve_simdata
 → _session.set_simdata(packed_sim, {})
   cached_simdata = "eyJkdCI6..." (base64)

Message 2: "What's the max moon distance?"
 → stability_from_simdata tool
 → _session.get_cached() → cached_simdata → instant

Message 3: "Export as CSV"
 → export_csv tool → _session.get_cached() → instant, no re-sim

Message 4: "Run new sim with different params"
 → _start_backend_job() → _session.update_job("agent-xyz", ...)
   cached_simdata = None (CLEARED again)
```

---

## AWS Workflow

### Step Functions State Machine Flow

```
START
 ├─ Input: {inputS3Prefix, outputS3Prefix}
 ▼
┌─────────────────────────────────────────┐
│ Task: SpawnBatchJob (ECS/Fargate)        │
│ • Docker: exomoon-batch:latest           │
│ • Env: INPUT_S3, OUTPUT_S3, params       │
│ • Command: python run_job.py            │
│ • Timeout: 3600s                        │
└───────────────────┬─────────────────────┘
                    │
                    ▼ Container execution
┌─────────────────────────────────────────┐
│ run_job.py                               │
│ 1. Download params.json from S3         │
│ 2. Load SystemParams                    │
│ 3. run_simulation_for_years(p, years)   │
│ 4. Build trajectory DataFrame           │
│ 5. Generate outputs:                    │
│    • traj.csv (positions + velocities)  │
│    • summary.json (metadata)            │
│    • traj.pkl (base64 packed)           │
│    • animation.html (standalone Plotly) │
│ 6. Upload all to S3 outputs/            │
│ 7. Create COMPLETE marker               │
│ ⏱️ Duration: 10–60s                     │
└───────────────────┬─────────────────────┘
                    ▼
END (SUCCEEDED)
```

**S3 Output Structure**:
```
outputs/{job_id}/
├── traj.csv           ← Dash fetches (positions + velocities)
├── summary.json       ← Dash fetches (HZ bounds, stability, rhill_AU)
├── traj.pkl           ← Agent service fetches (packed simdata for chat)
├── animation.html     ← S3 only (standalone, NOT sent to Dash UI)
├── job_metadata.json  (execution_arn, timestamps)
└── COMPLETE           (marker file for polling)
```

### AWS Hosting Architecture

```
Internet Boundary
├─────────────────────────────────────────────┤

[App Runner] (PUBLIC)
    ↓ serves Dash UI (port 8050)
    
[NLB] ← Public DNS:
├─ exomoon-agent-nlb-451ecd523536521c.elb.eu-west-2.amazonaws.com:8000

[VPC Boundary]
├─────────────────────────────────────────────┤

[ECS Task / Fargate] (PRIVATE)
    ├─ 10.0.x.x:8000 (internal IP)
    ├─ agent_service.py
    └─ IAM role → S3, Step Functions access
```

| Service | Role | Details |
|---------|------|---------|
| **App Runner** | Frontend Host | Runs Dash UI (run_dash.py) on public internet; port 8050 |
| **NLB** | Entry Point | Accepts HTTP:8000; routes to private ECS; health checks every 30s |
| **VPC** | Network Isolation | Private subnet for ECS; NAT gateway for outbound (Claude API calls) |
| **ECS/Fargate** | Compute | Runs agent_service.py; auto-scales; IAM role for AWS access |
| **ECR** | Image Registry | Docker images: `exomoon-agent`, `exomoon-batch` |
| **Security Groups** | Firewall | NLB: 0.0.0.0/0:8000; ECS: NLB:8000 only |

---

## Agent Service Workflows

### Scenario 1: Local Stability Check (No Backend Job)

**Trigger**: User has existing simdata, asks stability question.

```
Dash → POST /chat/stream (simdata: "eyJkdCI6...")
Agent → Claude → stability_from_simdata tool
      → unpack_sim(simdata) → assess_moon_stability()
      → Returns: {stable: true, max_r_rel: 0.25}
      → NO AWS calls
✅ Instant response (<100ms)
```

### Scenario 2: New Simulation → Backend Job

**Trigger**: No simdata, or simdata covers insufficient time span.

```
Dash → POST /chat/stream (simdata: null)
Agent → Claude → run_simulation tool
      → _start_backend_job() → S3 + Step Functions (~200ms return)
      → SSE: "Started job agent-abc123"

[Parallel] Step Functions → ECS → run_job.py (10–60s)

Dash polling (5s) → SUCCEEDED detected
      → GET /retrieve_simdata → _session.set_simdata()
      → Chat: "✅ Results ready!"

Follow-up queries → reads SessionCache → instant
```

### Scenario 3: NASA Archive Fetch (No Sim)

```
Dash → POST /chat/stream ("Fetch Kepler-442 b params")
Agent → Claude → fetch_exoplanet tool
      → HTTP GET exoplanetarchive.ipac.caltech.edu
      → Returns: {Ts: 5326K, Rs: 0.77R☉, Mp: 8.3M⊕, ...}
      → NO AWS, NO simdata
✅ Response in 1–5s (external API call only)
```

### Scenario 4: MCP Desktop (Local Only)

```
Claude Desktop → MCP stdio → mcp_server.py
              → run_sim_years() → Numba integrator (local)
              → Returns figure + metadata
NO network, NO AWS, NO Agent Service
✅ 10–60s pure local computation
```

### Client-Server Architecture Summary

| Scenario | Client | Server | AWS Used | Response Time |
|----------|--------|--------|----------|---------------|
| Local Stability | Dash Browser | Agent Service | None | <100ms |
| Backend Job | Dash Browser | Agent → Step Functions | S3, Step Fn, ECS | 5–180s |
| NASA Archive | Dash Browser | Agent → External API | None | 1–5s |
| MCP Desktop | Claude Desktop | MCP Server (local) | None | 10–60s |

**Identity summary**:
- **Agent Service**: Server to Dash; client to AWS + Claude API
- **Dash**: Server (serves UI); client (calls Agent Service)
- **MCP Server**: Server only; no clients except Claude Desktop
- **`_session`**: Module-level singleton in `agent_service.py` — single-user design

---

## Deployment Guide

### Prerequisites

- AWS Account: S3 bucket, Step Functions state machine, ECS/Fargate, IAM roles
- Docker buildx (multi-platform builds)
- AWS CLI configured with credentials
- Python 3.12+

### 1. Environment Setup

```bash
export AWS_REGION=eu-west-2
export EXOMOON_BUCKET=my-exomoon-bucket
export STATE_MACHINE_ARN=arn:aws:states:eu-west-2:123456789:stateMachine:ExomoonStateMachine
export ANTHROPIC_API_KEY=sk-ant-...
```

### 2. Build Docker Images

Three separate images:

```bash
# Frontend (Dash UI)
docker build -f Dockerfile -t your-registry/exomoon-dash:latest .

# Backend Agent Service
docker build -f agent.Dockerfile -t your-registry/exomoon-agent:latest .

# Batch Job Runner (run_job.py)
# Uses job.Dockerfile (or batch.yaml config)
docker build -f job.Dockerfile -t your-registry/exomoon-batch:latest .
```

All share `requirements_docker_1.txt`.

### 3. Docker Compose (Local Multi-Container)

`compose.yaml` runs agent (8000) + dash (8050) with shared `exomoon` network. Agent receives AWS credentials via volume mount of `~/.aws/credentials`.

```bash
docker-compose up -d
# Dash UI: http://localhost:8050
# Agent service: http://localhost:8000
```

### 4. AWS Deployment

```bash
# Update ECS service (agent)
aws ecs update-service --cluster exomoon-cluster --service exomoon-agent \
  --force-new-deployment

# Get NLB DNS
aws elbv2 describe-load-balancers --names exomoon-agent-nlb \
  --query 'LoadBalancers[0].DNSName' --output text
# → exomoon-agent-nlb-451ecd523536521c.elb.eu-west-2.amazonaws.com

# Push batch image to ECR
docker push 123456789.dkr.ecr.eu-west-2.amazonaws.com/exomoon-batch:v1.0
```

### 5. Health Check

```bash
curl http://localhost:8000/docs         # Swagger UI
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "simdata": null, "params": {}}'
# Open browser: http://localhost:8050
```

---

## Environment Variables

| Variable | Example | Description |
|----------|---------|-------------|
| `AWS_ENABLED` | `1` | Enable AWS backend (S3, Step Functions). `0` for local dev. |
| `AWS_REGION` | `eu-west-2` | AWS region for all services. Must match bucket + state machine. |
| `EXOMOON_BUCKET` | `my-exomoon-bucket` | S3 bucket for inputs/outputs. Agent writes `inputs/{job_id}/`, `outputs/{job_id}/`. |
| `STATE_MACHINE_ARN` | `arn:aws:states:eu-west-2:...` | Full ARN of Step Functions state machine. |
| `ANTHROPIC_API_KEY` | `sk-ant-v0-...` | Claude API key. Required for agent service. |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-20250514` | Claude model ID. Controls LLM capability and latency. |
| `CLAUDE_ENABLED` | `1` | Gates Claude client init. `0` to run agent without LLM (debug). |
| `AGENT_SERVICE_URL` | `http://exomoon-agent-nlb-451ecd.elb.eu-west-2.amazonaws.com:8000` | Dash uses this for `/chat`, `/job/{id}/status`, `/job/{id}/retrieve_simdata`. Local: `http://127.0.0.1:8000`. |
| `PYTHONPATH` | `/app/src` | Python module search path. Docker sets this for `import exomoon.*`. |
| `HOST` | `0.0.0.0` | Dash server bind address. |
| `PORT` | `8050` | Dash port. Agent uses 8000. |
| `DEBUG` | `1` | Dash debug mode + hot-reload. Set `0` for production. |
| `AWS_SHARED_CREDENTIALS_FILE` | `/root/.aws/credentials` | Path to AWS credentials. Docker mounts from `~/.aws/credentials:ro`. |

**Production `.env`**:
```bash
AWS_ENABLED=1
AWS_REGION=eu-west-2
EXOMOON_BUCKET=exomoon-prod-bucket
STATE_MACHINE_ARN=arn:aws:states:eu-west-2:123456789:stateMachine:ExomoonProductionSM
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-20250514
CLAUDE_ENABLED=1
AGENT_SERVICE_URL=http://exomoon-agent-nlb-production.elb.eu-west-2.amazonaws.com:8000
DEBUG=0
```

**Local dev `.env`**:
```bash
AWS_ENABLED=0
ANTHROPIC_API_KEY=sk-ant-...
AGENT_SERVICE_URL=http://127.0.0.1:8000
DEBUG=1
```

---

## Pending Issues

### CSV File Retrieval via Chatbot

**Status**: ⚠️ Partially Working

**What works**:
- CSV generated server-side in `run_job.py` (`traj_to_frame()` + `to_csv_bytes()`)
- Uploaded to S3: `s3://bucket/outputs/{job_id}/traj.csv`
- Presigned URL generated in `/job/{job_id}/status` response (valid 24h)
- `export_csv` tool can deserialize cached simdata (no re-simulation)

**What's broken**:
- No mechanism to surface the presigned URL in the chat interface
- No download link shown to user in chat
- User cannot click anything in chat to save CSV

**Root Cause**: `export_csv` tool generates CSV bytes but agent responses are text-only SSE streams. The URL is available but not returned to the user.

**Solution Options**:

1. **Option A (Quick, ~15 min)**: `export_csv` handler returns S3 presigned URL in response. Chat renders as markdown link `[Download traj.csv](url)`.
   - Code change: `agent_service.py` export_csv handler, return URL instead of bytes.

2. **Option B (Robust)**: New `/csv/{job_id}` endpoint on agent service streaming `application/csv`. Dash chat renders `<a href="/csv/{job_id}">` link.

3. **Option C (Cleanest)**: Agent extracts presigned URL from `/job/{job_id}/status` `urls` dict and returns it in chat response.

**Recommendation**: Option A first, then Option C for polish.

---

## Next Steps / Roadmap

### High Priority

1. **Visual Enhancements**
   - 3D interactive system viewer (Plotly 3D scatter or Three.js integration)
   - Orbital animation with real-time physics overlay (energy conservation, angular momentum display per frame)
   - Habitability heatmap: planet position vs. time with HZ shading
   - Improved animation trail quality and frame interpolation for smoother playback

2. **Performance Optimization**
   - Profile Numba integrator for hot spots (candidate: `_accel` inner loop, per-step array allocation patterns)
   - Add result caching layer: same `SystemParams` hash → return stored simulation without re-running
   - Consider GPU acceleration (CuPy/CUDA) for large parameter sweeps or N-body ensembles
   - Optimize `pack_sim`/`unpack_sim`: JSON+base64 is slow for large trajectories; consider `numpy.save` + zlib for 10x+ compression

3. **CSV Download Fix**
   - Implement Option A (1–2 lines in `agent_service.py` export_csv handler)
   - Test with live chatbot query
   - Estimated: 15 minutes

4. **Production Hardening**
   - Request rate limiting on agent endpoints
   - Session timeout (30 min inactivity → clear `SessionCache`)
   - Comprehensive error logging + CloudWatch alerting
   - CloudWatch dashboards for job success rates

### Medium Priority

1. **User Experience**
   - Simulation progress indicator (% complete via Step Functions `describe_execution` events)
   - Batch multiple queries into single simulation job
   - Parameter presets (e.g., "Earth-like system", "Kepler-442b", "Compact binary")

2. **Multi-User Support**
   - Replace in-memory `SessionCache` with Redis
   - User authentication (API keys or OAuth)
   - Per-user job isolation and usage tracking

### Lower Priority

1. **Exoplanet Archive Integration**
   - Pre-compute stability analysis for full known exoplanet catalog
   - Searchable database of stable moon configurations
   - Weekly archive sync pipeline

2. **Advanced Physics**
   - Tidal moon migration model
   - Resonance effects (moon-moon interactions for multi-moon systems)
   - Relativistic corrections for compact systems

---

## Key Code Locations

| Concern | File | Location |
|---------|------|----------|
| Non-blocking job submission | `agent_service.py` | ~line 358 (`_start_backend_job`) |
| SessionCache | `agent_service.py` | ~line 72 |
| Simdata retrieval endpoint | `agent_service.py` | ~line 1050 (`retrieve_job_simdata`) |
| Chat SSE callback | `run_dash.py` | ~line 423 (`send_chat_message`) |
| Job polling callback | `run_dash.py` | ~line 680 (`poll_agent_job_status`) |
| S3 results loader (animation reconstruction) | `run_dash.py` | ~line 1303 (`load_s3_results`) |
| Native UI run callback | `run_dash.py` | ~line 1001 (`run_cb`) |
| Physics engine | `integrator.py` | ~line 15 (`_leapfrog_integrate`) |
| Serialization | `eda.py` | ~line 56 (`pack_sim` / `unpack_sim`) |
| Animation builder | `exomoon/plotting/anim.py` | `build_animation()` |
| Stability analysis | `exomoon/moon_stability.py` | `assess_moon_stability()` |
| Input params schema | `exomoon/params.py` | `SystemParams` |
| Habitable zone | `exomoon/habitable_zone.py` | `hz_bounds_au()` |
| NASA archive | `exomoon/exoplanet_archive.py` | `fetch_system_by_planet()`, `search_planets()` |
