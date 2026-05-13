# Exomoon Orbital Integrator: Complete System Context

**Last Updated:** May 13, 2026  
**Status:** Production-stable (core functionality verified, multi-turn queries working)  
**Pending Issues:** CSV file retrieval via chatbot UI

---

## Executive Summary

The **Exomoon Orbital Integrator** is a full-stack distributed system combining:
- **Physics Engine**: Numba-compiled 3-body orbital integrator (star-planet-moon)
- **AWS Backend**: Step Functions + S3 for scalable job orchestration
- **AI Chat Agent**: Claude-powered multi-turn conversational interface
- **Interactive Dashboard**: Real-time Dash UI with animations, polling, and live updates

**Key Architectural Decision**: Non-blocking job submission + local simdata caching enables responsive chat with instant feedback while jobs compute in background (~5-60 seconds per simulation).

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Physics Foundation](#physics-foundation)
3. [Core Modules](#core-modules)
4. [API Reference](#api-reference)
5. [Data Flow Pipeline](#data-flow-pipeline)
6. [AWS Workflow](#aws-workflow)
7. [Agent Service Workflows](#agent-service-workflows)
8. [Deployment Guide](#deployment-guide)
9. [Environment Variables](#environment-variables)
10. [Pending Issues](#pending-issues)
11. [Next Steps](#next-steps)

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
│  │  │  Core Components:                                   │   │   │
│  │  │  • SessionCache: in-memory simdata + params cache  │   │   │
│  │  │  • Claude SDK: multi-turn LLM reasoning            │   │   │
│  │  │  • Tool Executor: run_simulation, export_csv, etc  │   │   │
│  │  │  • Job Orchestrator: Step Functions integration    │   │   │
│  │  └──────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│            │                                                         │
│            │ boto3: start_execution, describe_execution             │
│            ▼                                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   AWS BACKEND LAYER                         │   │
│  │  ┌──────────────────────────────────────────────────────┐   │   │
│  │  │  AWS Step Functions State Machine                   │   │   │
│  │  │  ├─ Input: params.json from S3                      │   │   │
│  │  │  ├─ Spawn: ECS/Batch task (run_job.py)            │   │   │
│  │  │  └─ Output: traj.csv, animation.html, traj.pkl     │   │   │
│  │  └──────────────────────────────────────────────────────┘   │   │
│  │                                                               │   │
│  │  ┌──────────────────────────────────────────────────────┐   │   │
│  │  │  S3 Storage (EXOMOON_BUCKET)                        │   │   │
│  │  │  ├─ inputs/{job_id}/params.json                     │   │   │
│  │  │  ├─ outputs/{job_id}/                               │   │   │
│  │  │  │  ├─ traj.pkl (serialized simulation)            │   │   │
│  │  │  │  ├─ traj.csv (trajectory data)                  │   │   │
│  │  │  │  ├─ animation.html (Plotly animation)           │   │   │
│  │  │  │  ├─ job_metadata.json (execution_arn)           │   │   │
│  │  │  │  └─ COMPLETE (marker file)                       │   │   │
│  │  │  └─ Retention: 24h presigned URLs                  │   │   │
│  │  └──────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│            │                                                         │
│            │ Batch Job Container (docker image)                     │
│            ▼                                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   SIMULATION ENGINE                         │   │
│  │  ┌──────────────────────────────────────────────────────┐   │   │
│  │  │  Physics Modules (exomoon/*)                        │   │   │
│  │  │  • simulation.py: run_simulation_for_years()        │   │   │
│  │  │  • integrator.py: Numba-compiled leapfrog (3-body)  │   │   │
│  │  │  • initial_conditions.py: orbital bootstrap        │   │   │
│  │  │  • moon_stability.py: escape detection + analysis  │   │   │
│  │  │  • eda.py: pack_sim/unpack_sim (serialization)    │   │   │
│  │  └──────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘

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

All three bodies (star, planet, moon) experience mutual gravitational attraction. The acceleration $\vec{a}$ on body A due to body B follows Newton's law:

$$\vec{a}_A = -\frac{\mu_B \vec{r}}{|\vec{r}|^3}$$

Where:
- $\mu_B = GM_B$ is the gravitational parameter of body B
- $\vec{r} = \vec{r}_A - \vec{r}_B$ is the position vector from B to A
- $|\vec{r}|$ is the distance between bodies
- Units: G = 4π² (scaled to AU, years convention; see `constants.py`)

**Code Implementation** (`integrator.py`):
```python
@njit(fastmath=True, cache=True)
def _accel(pos_a, pos_b, mu_b):
    """Compute gravitational acceleration of body A due to body B."""
    r0, r1, r2 = pos_a[0] - pos_b[0], pos_a[1] - pos_b[1], pos_a[2] - pos_b[2]
    r2n = r0*r0 + r1*r1 + r2*r2
    r = r2n ** 0.5
    r3 = r * r2n
    return np.array([-mu_b * r0 / r3, -mu_b * r1 / r3, -mu_b * r2 / r3])
```

**Numba Compilation**: `@njit(fastmath=True, cache=True)` compiles to native machine code at first call, enabling 100-1000x speedup for tight numerical loops.

---

### Leapfrog Integration (Symplectic Integrator)

The leapfrog method is a **symplectic integrator** that conserves energy exceptionally well over long timescales, critical for orbital mechanics.

**Algorithm** (per timestep $\Delta t$):

1. **Position KickHalf**: $\vec{r} \gets \vec{r} + \vec{v} \cdot \frac{\Delta t}{2}$
2. **Velocity Kick**: $\vec{v} \gets \vec{v} + \vec{a}(\vec{r}) \cdot \Delta t$
3. **Position KickHalf**: $\vec{r} \gets \vec{r} + \vec{v} \cdot \frac{\Delta t}{2}$

Applied to all three bodies in series per timestep:

$$\vec{r}_{mp}^{n+1} = \vec{r}_{mp}^n + \frac{\Delta t}{2}\vec{v}_{mp}^n + \frac{\Delta t}{2}\vec{v}_{mp}^{n+1}$$

Where acceleration at step $n$ includes gravity from **both** star and moon.

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

---

### Hill Radius & Moon Stability

**Hill Radius** (sphere of gravitational dominance for planet relative to star):

$$R_{Hill} = a_p \left(1 - e_p\right) \left(\frac{M_p}{3M_*}\right)^{1/3}$$

Where:
- $a_p$ = planet semi-major axis (AU)
- $e_p$ = planet orbital eccentricity
- $M_p$ = planet mass, $M_*$ = star mass

**Moon Stability**: Moon is **stable** if:
$$\max_t \left|\vec{r}_{moon} - \vec{r}_{planet}\right| \leq escape\_factor \times R_{Hill}$$

over the simulation interval. If violated, escape occurs; code logs `escape_time` via linear interpolation.

**Code** (`moon_stability.py`):
```python
moon_rel = traj["xyzarr_mm"] - traj["xyzarr_mp"]
r_rel = np.linalg.norm(moon_rel[:, :2], axis=1)  # 2D norm
stable = np.max(r_rel) <= escape_factor * rhill_AU
```

---

### Habitable Zone

Star's habitable zone (where liquid water could exist) defined by radiative balance:

**Stellar Luminosity**:
$$L_* = 4\pi R_*^2 \sigma T_*^4$$

**Inner Bound** (greenhouse runaway limit):
$$a_{inner} = \sqrt{\frac{L_*}{4\pi F_{inner}}} \quad (F_{inner} = 1.1 \times F_{\oplus})$$

**Outer Bound** (maximum greenhouse limit):
$$a_{outer} = \sqrt{\frac{L_*}{4\pi F_{outer}}} \quad (F_{outer} = 0.5 \times F_{\oplus})$$

Where $F_{\oplus} = 1370$ W/m² (Earth's insolation).

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

**Purpose**: Execute N-body orbital simulations with adaptive timestep selection.

**Key Functions**:

#### `run_simulation_for_years(p: SystemParams, years: float) → dict`

Runs simulation for fixed wallclock duration (years).

**Logic**:
1. Initialize orbital state via `initial_state(p)` (barycenter coordinates, velocities)
2. Compute moon orbital period: $T_{mm} = 2\pi \sqrt{\frac{a_m^3}{(M_p + M_m)}}$ (simplified Kepler)
3. Set timestep: $dt = \min(T_{mm} / 100, 1 / 20000)$ (balance moon resolution + year accuracy)
4. Call `leapfrog_integrate(state, t_end=years, dt)` → returns trajectory dict
5. Compute habitable zone bounds via `hz_bounds_au()`
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
    "dt": float,  # Timestep (years)
    "t_end": float,  # Final time (years)
    "a_inner_au": float,  # Habitable zone inner bound
    "a_outer_au": float,  # Habitable zone outer bound
}
```

---

### `eda.py`: Data Serialization

**Purpose**: Convert simulations to/from compact base64-encoded JSON for transport via HTTP/S3.

#### `pack_sim(sim: dict) → str`

Serializes trajectory dict to base64 string.

```python
def pack_sim(sim: dict) -> str:
    traj = sim["traj"]
    payload = {
        "dt": sim["dt"],
        "t_end": sim["t_end"],
        "a_inner_au": sim.get("a_inner_au"),
        "a_outer_au": sim.get("a_outer_au"),
        "xyz_mp": traj["xyzarr_mp"].tolist(),  # Convert numpy → JSON-serializable
        "xyz_ms": traj["xyzarr_ms"].tolist(),
        "xyz_mm": traj["xyzarr_mm"].tolist(),
        "vel_mp": traj.get("velarr_mp").tolist() if traj.get("velarr_mp") is not None else None,
        # ... velocities for ms, mm ...
    }
    raw = json.dumps(payload).encode()
    return base64.b64encode(raw).decode()  # → "eyJkdCI6IDAuMDAx..."
```

#### `unpack_sim(packed: str) → dict`

Deserializes base64 string back to dict with numpy arrays.

```python
def unpack_sim(packed: str) -> dict:
    raw = base64.b64decode(packed.encode())
    payload = json.loads(raw.decode())
    traj = {
        "xyzarr_mp": np.array(payload["xyz_mp"], dtype=float),  # JSON → numpy
        "xyzarr_ms": np.array(payload["xyz_ms"], dtype=float),
        "xyzarr_mm": np.array(payload["xyz_mm"], dtype=float),
        "velarr_mp": np.array(payload["vel_mp"], dtype=float) if payload.get("vel_mp") else None,
        # ...
    }
    return {"dt": payload["dt"], "t_end": payload["t_end"], "traj": traj, ...}
```

#### `traj_to_frame(sim: dict) → dict | DataFrame`

Converts trajectory to columnar format for CSV export.

```python
def traj_to_frame(sim: dict):
    traj = sim["traj"]
    dt = sim["dt"]
    n = len(traj["xyzarr_mp"])
    t = np.arange(n) * dt  # Time array (years)
    
    # Compute derived quantities
    rel_mm_mp = traj["xyzarr_mm"] - traj["xyzarr_mp"]  # Moon relative to planet
    moon_planet_dist = np.linalg.norm(rel_mm_mp[:, :2], axis=1)
    
    data = {
        "t_years": t,
        "star_x": traj["xyzarr_ms"][:, 0],
        # ... all positions and velocities ...
        "moon_planet_dist": moon_planet_dist,
    }
    
    if _HAS_PANDAS:
        return pd.DataFrame(data)
    return data  # dict-of-arrays fallback
```

---

### `agent_service.py`: Core Chatbot Engine

**Purpose**: FastAPI service handling Claude-powered multi-turn conversations, job orchestration, and simdata caching.

#### `SessionCache` Class

In-memory session state for a user conversation.

```python
class SessionCache:
    """Store conversation state: last job_id, output_prefix, cached simdata."""
    def __init__(self):
        self.last_job_id: Optional[str] = None
        self.last_output_prefix: Optional[str] = None
        self.cached_simdata: Optional[str] = None  # Base64-encoded packed sim
        self.cached_params: Dict[str, Any] = {}
    
    def update_job(self, job_id: str, output_prefix: str):
        """Called when a new job is submitted."""
        self.last_job_id = job_id
        self.last_output_prefix = output_prefix
        self.cached_simdata = None  # Clear old data
    
    def set_simdata(self, simdata: str, params: Dict[str, Any]):
        """Called when job completes and simdata is retrieved from S3."""
        self.cached_simdata = simdata
        self.cached_params = params
    
    def get_cached(self) -> tuple[Optional[str], Dict]:
        """Fallback for tools to use cached simdata if available."""
        return self.cached_simdata, self.cached_params
```

**Usage Pattern**: When a tool like `export_csv` needs simdata but none is provided by user:
```python
simdata_to_use = req.simdata  # User-supplied
if not simdata_to_use:
    cached_sim, _ = _session.get_cached()
    if cached_sim:
        simdata_to_use = cached_sim  # Fall back to cached
```

---

#### `_start_backend_job()` Function

**KEY FIX (Responsiveness)**: Submits job to AWS Step Functions and returns **immediately** with `status: "submitted"`. No blocking polling.

```python
def _start_backend_job(params: Dict[str, Any], years: Optional[float],
                       check_stability: bool = False,
                       escape_factor: float = 1.0) -> Dict[str, Any]:
    """Start a Step Functions job to run the simulation."""
    if not (sf and s3 and STATE_MACHINE_ARN and BUCKET):
        return {"ok": False, "error": "AWS backend not configured"}

    # Generate unique job ID
    job_id = f"agent-{uuid.uuid4().hex[:12]}"
    inp_prefix = f"inputs/{job_id}"
    out_prefix = f"outputs/{job_id}"

    # Build params dict
    params_dict = {
        "Ts": float(params.get("Ts", 5772)),
        "years": float(years) if years else 0.0,
        # ... remaining params ...
    }

    try:
        # Upload params to S3 input prefix
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

        # Store execution ARN for status queries
        job_metadata = {
            "job_id": job_id,
            "execution_arn": exec_resp["executionArn"],
            "output_prefix": out_prefix,
        }
        s3.put_object(Bucket=BUCKET, Key=f"{out_prefix}/job_metadata.json",
                      Body=json.dumps(job_metadata).encode())

        # Update session cache
        _session.update_job(job_id, out_prefix)

        # ✅ Return immediately - NO polling
        return {
            "ok": True,
            "job_id": job_id,
            "execution_arn": exec_resp["executionArn"],
            "status": "submitted",  # Client can poll via /job/{job_id}/status
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}
```

**What Changed**: Removed 60-second polling loop that was blocking all responses.

---

#### `/job/{job_id}/retrieve_simdata` Endpoint

**KEY FIX (Data Persistence)**: Called by Dash when job completes. Fetches simdata from S3 and caches in session.

```python
@app.get("/job/{job_id}/retrieve_simdata")
def retrieve_job_simdata(job_id: str):
    """Retrieve and cache simdata from completed job. Called by Dash polling."""
    if not (s3 and BUCKET):
        return {"ok": False, "simdata_cached": False}

    try:
        output_prefix = f"outputs/{job_id}"
        simdata_key = f"{output_prefix}/traj.pkl"

        # Fetch simdata from S3
        obj = s3.get_object(Bucket=BUCKET, Key=simdata_key)
        simdata = obj["Body"].read().decode()

        # Cache in session for follow-up query tools
        _session.set_simdata(simdata, {})

        return {
            "ok": True,
            "job_id": job_id,
            "simdata_cached": True,
            "message": f"Cached ({len(simdata)} chars)"
        }
    except Exception as e:
        return {
            "ok": False,
            "job_id": job_id,
            "simdata_cached": False,
            "error": str(e)
        }
```

---

#### Tool: `export_csv` Handler

**KEY FIX (No Re-simulation)**: Uses `unpack_sim()` to deserialize cached data directly instead of re-running simulation.

```python
if tool_name == "export_csv":
    simdata_to_use = req.simdata  # User-provided simdata, if any
    if not simdata_to_use:
        cached_sim, _ = _session.get_cached()
        if cached_sim:
            simdata_to_use = cached_sim  # Fall back to cache
    
    if not simdata_to_use:
        return {"error": "No simdata available"}
    
    try:
        # Deserialize cached simdata (NO re-simulation)
        sim = unpack_sim(simdata_to_use)
        
        # Build trajectory frame
        frame = traj_to_frame(sim)
        
        # Export to CSV bytes
        csv_bytes = to_csv_bytes(frame)
        
        # Write to file (in Dash or streaming context)
        csv_url = f".../{job_id}/traj.csv"
        return {"ok": True, "csv_url": csv_url}
    except Exception as e:
        return {"error": f"CSV export failed: {str(e)}"}
```

---

### `run_dash.py`: Interactive Dashboard

**Purpose**: Multi-mode dashboard enabling simulations via (1) native Dash UI controls, (2) chat agent queries, and (3) NASA archive parameter fetch. Provides real-time animation, CSV export, and detailed EDA analysis.

**Execution Modes**:

#### Mode 1: Native Dash UI (Direct)
User configures parameters via sliders/dropdowns → clicks "Run Simulation" → simulation executes (local or AWS) → animation + results displayed immediately.

**Parameter Configuration UI** (Left panel):
```
NASA Exoplanet Archive
├─ Dropdown: search + select planet by name
└─ "Fetch from NASA" button → auto-populates stellar + planet params

System Parameters (Sliders/Inputs):
├─ Ts (stellar temp): 2000–20000 K (number input)
├─ rs_solar, ms_solar: 0.05–30/50 R☉/M☉ (sliders)
├─ mp_earth, dp_cgs: planet mass + density (sliders)
├─ ap_AU, ep: planet orbit semi-major axis + eccentricity (sliders)
├─ mm_earth, am_hill, em: moon mass, a-fraction, eccentricity (sliders)
├─ moon_dir: prograde/retrograde toggle (radio buttons)
└─ sim_years: simulation duration (0 = 1 planet orbit, number input)

Buttons:
├─ "Run Simulation" → triggers run-btn n_clicks callback
└─ "Export CSV" → triggers export-btn callback (early download before/after run)
```

**Simulation Execution Flow** (Native UI mode):
```
Run Simulation Button Click
 │
 ├─ Reads all parameter values from UI controls
 │
 ├─ (AWS Path) if AWS_ENABLED:
 │  ├─ Serialize params → params.json
 │  ├─ Upload to S3 inputs/{job_id}/
 │  ├─ Call sf.start_execution() → Step Functions
 │  ├─ Enable status poller (status-interval)
 │  └─ Return placeholder figure while job runs
 │
 └─ (Local Path) else:
    ├─ Call run_simulation_for_years(p, years)
    ├─ Build trajectory frame
    ├─ Call build_animation(traj, a_inner, a_outer)
    ├─ Pack sim → base64 → store in simdata Dash Store
    └─ Return animated figure immediately
```

#### Mode 2: Chat Agent (Conversational)
User types query in chat drawer → Claude agent decides tools → can trigger `run_simulation` tool → job submitted via agent service → polling callback in chat monitors status → simdata cached for follow-ups.

**Chat Job Status Polling Callback**:
```python
@app.callback(
    Output("chat-messages", "children", allow_duplicate=True),
    Input("chat-job-poller", "n_intervals"),
    [State("chat-job-id", "data"), State("chat-history", "data"), State("chat-messages", "children")],
    prevent_initial_call=True,
)
def poll_agent_job_status(n_intervals, job_id, chat_hist, chat_msgs):
    """
    Poll agent /job/{job_id}/status endpoint every 5 seconds.
    When job completes, call /job/{job_id}/retrieve_simdata to cache results.
    """
    if not job_id:
        return no_update

    try:
        # Query job status
        status_url = f"{AGENT_SERVICE_URL}/job/{job_id}/status"
        result = requests.get(status_url, timeout=10).json()
        status = result.get("status")

        status_msg = f"⏳ Job status: {status} ({result.get('elapsed_seconds', '?')}s)"

        if status == "SUCCEEDED":
            # ✅ Call retrieve_simdata endpoint to trigger caching in agent session
            retrieve_url = f"{AGENT_SERVICE_URL}/job/{job_id}/retrieve_simdata"
            retrieve_resp = requests.get(retrieve_url, timeout=30)
            retrieve_result = retrieve_resp.json()
            
            if retrieve_result.get("simdata_cached"):
                status_msg = f"✅ Job complete! Results ready for querying."
                # Simdata now cached in agent session for follow-up queries
            else:
                status_msg = "⚠️ Job done but couldn't cache results"

            # Stop polling
            return [*chat_msgs, dcc.Markdown(status_msg)]

        elif status == "FAILED":
            return [*chat_msgs, dcc.Markdown("❌ Job failed")]

        elif status == "TIMED_OUT":
            return [*chat_msgs, dcc.Markdown("⏱️ Job timed out")]

        else:
            # Still running: append status message
            return [*chat_msgs, dcc.Markdown(status_msg)]

    except Exception as e:
        return [*chat_msgs, dcc.Markdown(f"❌ Error: {str(e)}")]
```

#### Mode 3: NASA Archive (Parameter Fetch)
User types planet name in search box → fetches system params from archive → auto-populates UI controls → can auto-run if `?run=1` in query string.

---

**Animation & Visualization** (Main Page):

When simulation completes (local) or job succeeds (AWS), animation renders showing:
- **Left panel**: Full system orbit (Star, Planet, Moon trajectories over time)
- **Right inset**: Moon relative to planet (zoom perspective)
- **Annotations**: Hill radius (red circle), moon semi-major axis, habitable zones (green bands)
- **Slider**: Scrub through timesteps or auto-play
- **Trail effect**: Orbital paths with fading history

Code reference: `build_animation(traj, a_inner_au, a_outer_au)` in [exomoon/plotting/anim.py](exomoon/plotting/anim.py)

---

**Export CSV Feature** (Native UI):

```python
@app.callback(
    Output("download-csv", "data"),
    Input("export-btn", "n_clicks"),
    State("simdata", "data"),
    prevent_initial_call=True,
)
def export_csv_from_simdata(n_clicks, simdata):
    """Export trajectory to CSV when user clicks button."""
    if not simdata:
        return None  # No data yet
    
    try:
        sim = unpack_sim(simdata)
        frame = traj_to_frame(sim)
        csv_bytes = to_csv_bytes(frame)
        
        return dcc.send_bytes(
            csv_bytes,
            filename=f"exomoon_trajectory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
    except Exception as e:
        return None
```

**CSV Output Structure**: Columns include `t_years`, `star_x/y/z`, `planet_x/y/z`, `moon_x/y/z`, `moon_planet_dist`, `planet_star_dist`, and all velocity components (positions + velocities at each timestep).

---

**EDA Page Features**:

```
EDA Analysis Page (route: /eda)
├─ Dropdown: Select variables to plot (moon_planet_dist, planet_star_dist, speeds, positions)
├─ Plot type toggle: Time series (line) vs. histogram vs. 2D projection
├─ Normalize checkbox: Scale to 0–1 range for multi-axis comparison
│
├─ Main plot: Time-series figure showing selected variables over simulation duration
│  ├─ X-axis: Time (years)
│  ├─ Y-axis: Variable value (AU, AU/yr, etc.)
│  └─ Secondary Y-axis: Auto-scale for multi-variable plots
│
├─ Interactivity:
│  ├─ Hover: Show exact values + timestamps
│  ├─ Zoom: Click-drag to zoom time window
│  ├─ Pan: Shift-drag to pan
│  ├─ Box select: Click legend to show/hide variables
│  └─ Legend: Click to isolate/compare variables
│
└─ "Back to Simulation" button → return to main page
```

**Code Reference**: 
```python
@app.callback(
    Output("eda-graph", "figure"),
    [Input("eda-vars", "value"), Input("eda-plot-type", "value"), Input("eda-normalize", "value")],
    State("simdata", "data"),
    prevent_initial_call=True,
)
def generate_eda_plot(selected_vars, plot_type, normalize, simdata):
    """Build Plotly figure from selected variables."""
    if not simdata or not selected_vars:
        return {}
    
    sim = unpack_sim(simdata)
    frame = traj_to_frame(sim)  # Dict-of-arrays or DataFrame
    
    # Build multi-trace figure with secondary y-axis if needed
    # Normalize if checkbox set
    # Return Plotly figure object
```

---

### `mcp_server.py`: Model Context Protocol (MCP) Server

**Purpose**: Expose physics simulation tools as **Model Context Protocol** resource for Claude Desktop (local AI integration). Enables direct invocation of tools without HTTP roundtrip (used in `claude_desktop_config.json`).

**Trigger Points**:
- **When**: User interacts with Claude Desktop (local AI app) + MCP server extension registered
- **How**: `claude_desktop_config.json` specifies:
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
- **Result**: Claude Desktop gains direct access to tools: `run_simulation`, `stability_analysis`, `export_csv`, `eda_plot`, etc.

**Key Tools Exposed**:
```python
@mcp.tool()
def run_simulation(params: dict, years: float) -> dict:
    """Run orbital simulation for N years. Returns packed simdata."""
    
@mcp.tool()
def stability_from_simdata(simdata: str, years: float, escape_factor: float) -> dict:
    """Assess moon stability from cached simulation data."""
    
@mcp.tool()
def export_csv(simdata: str) -> bytes:
    """Export trajectory to CSV format."""
    
@mcp.tool()
def eda_plot(simdata: str, variables: list) -> str:
    """Generate EDA analysis HTML plot."""
```

**Difference from Agent Service**:
- **Agent Service** (`agent_service.py`): Web-based, FastAPI, full HTTP endpoints, session caching
- **MCP Server** (`mcp_server.py`): Desktop-based, stdio transport, local-only, direct Python execution

**When Used**:
- Claude Desktop with Exomoon extension installed → user asks physics questions → Claude calls MCP tools directly (no web service needed)
- Useful for offline analysis, local testing, or tightly-coupled AI workflows

---

## API Reference

### `/chat` Endpoint (Streaming)

**Method**: `POST`  
**Body** (JSON):
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

data: {"type": "token", "payload": "Started "}
data: {"type": "token", "payload": "job "}
...

data: {"type": "done", "payload": {"message": "...", "job_id": "...", ...}}
```

---

### `/job/{job_id}/status` Endpoint

**Method**: `GET`  
**Response** (JSON):
```json
{
  "ok": true,
  "job_id": "agent-abc123",
  "status": "RUNNING",
  "elapsed_seconds": 15,
  "urls": {}
}
```

When `status == "SUCCEEDED"`:
```json
{
  "ok": true,
  "job_id": "agent-abc123",
  "status": "SUCCEEDED",
  "elapsed_seconds": 45,
  "urls": {
    "traj.csv": "https://bucket.s3.amazonaws.com/.../traj.csv?X-Amz-Signature=...",
    "animation.html": "https://bucket.s3.amazonaws.com/.../animation.html?...",
    "summary.json": "https://bucket.s3.amazonaws.com/.../summary.json?..."
  }
}
```

---

### `/job/{job_id}/retrieve_simdata` Endpoint

**Method**: `GET`  
**Response** (JSON):
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

### Complete Multi-Turn Query Flow

```
USER INTERACTION:
┌─────────────────────────────────────────────────────────────────┐
│ User types: "Is moon at Kepler-442 b stable over 100 years?"    │
│ Clicks: Send                                                     │
└────────────┬────────────────────────────────────────────────────┘
             │
             ▼ POST /chat (SSE)
        ┌────────────────────────────────────────────────┐
        │ Agent Service receives ChatRequest              │
        │ • message = "Is moon... stable..."             │
        │ • simdata = None (no prior results)            │
        │ • params = {} (system defaults)                │
        └────────────┬───────────────────────────────────┘
                     │
                     ▼ Claude Tool Call Decision
        ┌────────────────────────────────────────────────┐
        │ Claude evaluates tools:                        │
        │ • fetch_exoplanet? (need planet name)          │
        │ • run_simulation? (YES - need years=100)       │
        │ • stability_from_simdata? (no cached data yet) │
        └────────────┬───────────────────────────────────┘
                     │
                     ▼ Call _start_backend_job()
        ┌────────────────────────────────────────────────┐
        │ 1. Generate job_id = "agent-xyz789"            │
        │ 2. Upload params.json → S3 inputs/             │
        │ 3. sf.start_execution() → ASYNC                │
        │ 4. _session.update_job(job_id, out_prefix)     │
        │ 5. RETURN {"status": "submitted", ...}         │
        │ ⏱️ Time to response: ~200ms (NO POLLING)       │
        └────────────┬───────────────────────────────────┘
                     │
         ✅ SSE STREAM TO DASH:
         │ "Started job agent-xyz789 (Step Functions)"
             │
             ▼ AWS Backend (parallel execution)
        ┌────────────────────────────────────────────────┐
        │ Step Functions State Machine                   │
        │ → Spawn ECS/Batch task                         │
        │ → run_job.py downloads params.json             │
        │ → run_simulation_for_years(p, 100)            │
        │ → Physics: leapfrog_integrate (Numba JIT)     │
        │ → Generate: traj.csv, animation.html, traj.pkl │
        │ → Upload outputs to S3                        │
        │ → Create COMPLETE marker                       │
        │ ⏱️ Duration: 10-60 seconds (depending on years) │
        └────────────────────────────────────────────────┘
                     │
             ▼ Dash Polling (5s interval)
        ┌────────────────────────────────────────────────┐
        │ Poll #1: status = RUNNING (elapsed: 5s)        │
        │ Poll #2: status = RUNNING (elapsed: 10s)       │
        │ ...                                            │
        │ Poll #10: status = SUCCEEDED (elapsed: 45s)    │
        │ ✅ DETECTED → Call /job/.../retrieve_simdata   │
        └────────────┬───────────────────────────────────┘
                     │
                     ▼ Retrieve Simdata
        ┌────────────────────────────────────────────────┐
        │ 1. Fetch traj.pkl from S3                      │
        │ 2. _session.set_simdata(packed_sim, params)    │
        │ 3. Return {"simdata_cached": true}             │
        │ ✅ Chat updates: "Results ready"               │
        └────────────┬───────────────────────────────────┘
                     │
        ✅ SSE TO DASH (STREAMING):
        │ "✅ Job complete! Analyzing results..."
             │
             ▼ Claude Analyzes Cached Simdata
        ┌────────────────────────────────────────────────┐
        │ Claude reads cached simdata:                   │
        │ • unpack_sim(simdata) → sim dict               │
        │ • Extract: max_r_rel, rhill_AU, escape_time   │
        │ • Compute: "Moon IS stable (r_max < 0.4 Rh)"  │
        │ • Stream response word-by-word                │
        │ ⏱️ Time to full response: ~500ms total         │
        └────────────┬───────────────────────────────────┘
                     │
      ✅ SSE STREAMS: "Moon IS stable..."
                     │
             ▼ User Asks Follow-up
        ┌────────────────────────────────────────────────┐
        │ User: "What's the escape time?"                │
        │ • Claude tool: "stability_from_simdata"        │
        │ • Reads CACHED simdata (instant)               │
        │ • No re-simulation needed                      │
        │ ✅ Response in <100ms                          │
        │ "Escape time: 345 years"                       │
        └────────────────────────────────────────────────┘
```

---

## Dual Execution Paths: Native UI vs. Chat Agent

The system supports two independent paths for running simulations:

### Path 1: Native Dash UI (Direct Execution)
```
User Interface → Parameter Controls (sliders/dropdowns) → Run Button
                                                            ↓
                                            Local Execution or AWS Job
                                                            ↓
                                    Immediate Animation + Visualization
```

**Characteristics**:
- Direct, synchronous (local) or fire-and-forget async (AWS)
- No AI reasoning; user manually configures parameters
- Results displayed in animation viewer + EDA page
- Export CSV via "Export CSV" button
- Fastest path for known good parameters

**Use Case**: User has parameters in mind (or from NASA archive preset), clicks Run, watches animation.

---

### Path 2: Chat Agent (Conversational Execution)
```
Chat Query → Claude Reasoning + Tool Dispatch → Agent Service
                                                        ↓
            (Claude may call: run_simulation tool, stability tool, export_csv tool, eda_plot tool)
                                                        ↓
                                    Async Job Submission to AWS
                                                        ↓
                                    5-sec Polling in Dash (UI feedback)
                                                        ↓
                                    Results Cached in Agent Session
                                                        ↓
                                    Follow-up Queries (instant, no re-sim)
```

**Characteristics**:
- Conversational, multi-turn reasoning
- Claude decides which tools to call based on queries
- Always async (even local agent uses Step Functions backend)
- Polling callback in chat keeps UI updated
- Simdata cached in agent session for instant follow-ups
- Best for exploratory analysis ("What if...?", "Is this stable?")

**Use Case**: User asks "Run a sim", gets job_id, asks follow-ups like "What's the max distance?" (instant from cache).

---

## Key Difference: Both Paths Are Independent

- **Native UI path** uses local Dash Stores (`dcc.Store`) for session state
- **Chat agent path** uses remote Agent Service `SessionCache` for session state
- **No conflict**: User can run via UI, then ask chat questions about OTHER sims (different cache)

---

### Step Functions State Machine Flow

```
START
 │
 ├─ Input: {inputS3Prefix, outputS3Prefix}
 │
 ▼
┌──────────────────────────────────────────┐
│ Task: DownloadInputs                     │
│ • Fetch params.json from inputS3Prefix   │
└──────────┬───────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│ Task: SpawnBatchJob (ECS/Fargate)        │
│ • Docker image: exomoon-batch:latest     │
│ • Env vars: INPUT_S3, OUTPUT_S3, params  │
│ • Command: python run_job.py             │
│ • Timeout: 3600s (1 hour)                │
└──────────┬───────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│ Container: run_job.py                    │
│ ┌────────────────────────────────────┐   │
│ │ 1. Download params.json from S3    │   │
│ │ 2. Load: SystemParams (star, ...) │   │
│ │ 3. Call: run_simulation_for_years()│   │
│ │ 4. Build: trajectory DataFrame    │   │
│ │ 5. Generate outputs:              │   │
│ │    • traj.csv (trajectory data)   │   │
│ │    • animation.html (Plotly)      │   │
│ │    • summary.json (metadata)      │   │
│ │    • traj.pkl (base64 packed)     │   │
│ │ 6. Upload all to S3 outputs/      │   │
│ │ 7. Create COMPLETE marker         │   │
│ │ ⏱️ Duration: 10-60s (Y-dependent)  │   │
│ └────────────────────────────────────┘   │
└──────────┬───────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│ Task: WaitForCompletion                  │
│ • Poll S3 for COMPLETE marker            │
│ • Timeout: 3600s                         │
└──────────┬───────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│ Task: GeneratePresignedURLs (optional)   │
│ • Create 24h-valid S3 download links     │
└──────────┬───────────────────────────────┘
           │
           ▼
END (SUCCEEDED)

S3 OUTPUT STRUCTURE:
outputs/agent-xyz789/
├── traj.csv                (trajectory: positions + velocities)
├── animation.html          (Plotly interactive animation)
├── summary.json            (timing, escape_time, stability bool)
├── traj.pkl                (base64-encoded packed simulation)
├── job_metadata.json       (execution_arn, timestamps)
└── COMPLETE                (marker file for polling)
```

---

## AWS Infrastructure: Agent Service Hosting

### Public Domain Access Through NLB to Private ECS

The agent service deployment spans the public internet boundary to provide Dash UI access while keeping compute isolated:

```
Internet Boundary
├─────────────────────────────────────────────┤

[App Runner] (PUBLIC)
    ↓ HTTP request to public DNS
[NLB] ← Public DNS name
├─ exomoon-agent-nlb-451ecd523536521c.elb.eu-west-2.amazonaws.com:8000
│  (Publicly routable, internet-facing)
    ↓ Routes through VPC boundary
 
[VPC Boundary]
├─────────────────────────────────────────────┤
    ↓ Private routing via security groups
[Private ECS Task] (Fargate)
    ├─ 10.0.x.x:8000 (internal IP)
    ├─ Agent Service listening (agent_service.py)
    └─ Can access S3, Step Functions (IAM role)
```

**Key Components**:

| Service | Role | Details |
|---------|------|---------|
| **App Runner** | Frontend Host | Runs Dash UI (run_dash.py) on public internet; serves HTML/CSS/JS on port 8050 |
| **Network Load Balancer (NLB)** | Entry Point | Accepts inbound HTTP on port 8000; routes to private ECS tasks; health checks every 30s |
| **VPC** | Network Isolation | Private subnet contains ECS tasks; NAT gateway for outbound internet (Claude API calls) |
| **ECS (Fargate)** | Compute | Runs agent_service.py container; auto-scales based on CPU/memory; IAM role for AWS service access |
| **ECR** | Image Registry | Stores exomoon-agent Docker image; pulled by ECS on task startup |
| **Security Groups** | Firewall | NLB SG allows 0.0.0.0/0:8000; ECS task SG allows NLB:8000 only |

**Traffic Flow**:
```
Browser on User's Machine
    ↓ HTTPS to App Runner (public)
    ├─ Dash serves UI (port 8050)
    └─ Browser JS makes requests to:
        ↓ HTTPS to NLB public DNS (port 8000)
        ├─ POST /chat/stream
        ├─ GET /job/{job_id}/status
        └─ GET /job/{job_id}/retrieve_simdata
            ↓ NLB routes via security group rules
            ↓ Inside VPC (private)
            ↓ ECS task processes request
            └─ Returns response back through NLB
```

---

## Agent Service: Client-Server Scenarios

The agent service operates in four distinct patterns depending on the client and data availability:

### Scenario 1: Local Stability Check (No Backend Job)

```
Browser (Dash UI)
run_dash.py listening on 127.0.0.1:8050
         │
         │ POST /chat/stream (SSE)
         │ {message: "Is moon stable?", simdata: "eyJkdCI6IDAuMDAxLCJ0cmFqIjp7..."}
         ↓
Agent Service (FastAPI)
agent_service.py listening on 0.0.0.0:8000
         │
         ├─ _chat_with_claude() [PRIMARY]
         │  ├─ Claude API → decides to call stability_from_simdata tool
         │  ├─ _execute_tool() runs locally:
         │  │  ├─ unpack_sim(simdata) → deserialize trajectory
         │  │  ├─ assess_moon_stability(traj, years=10, escape_factor=1.0)
         │  │  └─ Returns: {stable: true, max_r_rel: 0.25, escape_time: null}
         │  │
         │  └─ Claude formulates response: "Yes, the moon is stable..."
         │
         └─ NO AWS calls (no S3, no Step Functions)
                 │
                 ↓ SSE token stream back to browser
                 │
         Message rendered: "Yes, the moon is stable (max distance 0.25 AU)"
```

**Key Points**:
- ✅ Instant response (no job submission)
- ✅ Uses cached simdata from prior run
- ✅ No cloud cost
- ✅ Suitable for exploratory queries

---

### Scenario 2: Insufficient Simdata → Backend Job (AWS Path)

```
Browser (Dash UI)
         │ POST /chat/stream
         │ {message: "100-year stability check", simdata: "...covers 10 years"}
         ↓
Agent Service
         │
         ├─ _chat_with_claude()
         │  ├─ Claude API → calls stability_from_simdata tool
         │  ├─ Tool notes: {needs_rerun: true, reason: "simdata only covers 10 years, need 100"}
         │  ├─ Claude sees this, calls start_backend_job tool:
         │  │  └─ start_backend_job({years: 100, ...other_params...})
         │  │
         │  └─ _start_backend_job() executes:
         │     ├─ 1. PUT params.json → s3://bucket/inputs/agent-abc123/params.json
         │     ├─ 2. Call sf.start_execution(STATE_MACHINE_ARN, input={...})
         │     │   └─ Returns: executionArn = "arn:aws:states:..."
         │     ├─ 3. PUT job_metadata.json → s3://bucket/outputs/agent-abc123/
         │     │   └─ Stores: {execution_arn, job_id, start_time, params}
         │     └─ Returns: {ok: True, job_id: "agent-abc123", status: "submitted"}
         │
         └─ Claude responds: "Starting 100-year simulation (ID: agent-abc123)"
                 │
                 ↓ SSE: "📊 Job agent-abc123 submitted. Checking status..."
                 │
         Browser receives job_id → enables polling callback
                 │
                 │ GET /job/agent-abc123/status (every 5s)
                 ↓
Agent Service
         ├─ get_job_status()
         │  ├─ 1. GET job_metadata.json from S3 → extract execution_arn
         │  ├─ 2. sf.describe_execution(execution_arn)
         │  └─ Returns: {status: "RUNNING", elapsed_seconds: 45, output: {...}}
         │
         └─ Responds: {status: "RUNNING", elapsed_seconds: 45, execution_arn: "..."}
                 │
         Browser UI: "⏳ Job running... 45 seconds elapsed"
                 │ (keeps polling every 5s)
                 ↓
    [Parallel: Step Functions Job Execution in ECS Container]
    ├─ run_job.py starts
    ├─ Downloads params.json from S3
    ├─ Runs simulation for 100 years (may take 30-180s)
    ├─ Generates outputs:
    │  ├─ traj.csv (positions + velocities over time)
    │  ├─ traj.pkl (base64-encoded packed simdata)
    │  ├─ summary.json (metadata: escape_time, stability)
    │  └─ animation.html (Plotly animation)
    ├─ Uploads all to s3://bucket/outputs/agent-abc123/
    └─ Step Functions marks SUCCEEDED
                 │
                 ↓ (Browser still polling)
                 │
Agent Service
         ├─ get_job_status() checks again
         │  ├─ sf.describe_execution() → status: SUCCEEDED
         │  ├─ Generates presigned URLs (24h expiry):
         │  │  ├─ traj.csv
         │  │  ├─ summary.json
         │  │  └─ animation.html
         │  └─ Returns: {status: "SUCCEEDED", elapsed_seconds: 95, urls: {...}}
         │
         └─ Responds: {status: "SUCCEEDED", urls: {...}}
                 │
         Browser stops polling
                 │ Optional: GET /job/agent-abc123/retrieve_simdata
                 ↓
Agent Service
         ├─ retrieve_job_simdata()
         │  ├─ GET traj.pkl from S3/outputs/agent-abc123/
         │  ├─ _session.set_simdata(traj_pkl, params)
         │  │  └─ Cache stored in-memory for instant follow-ups
         │  └─ Returns: {ok: True, simdata_cached: True}
         │
         └─ Responds: {simdata_cached: True}
                 │
         Browser downloads files from presigned URLs
         Chat updates: "✅ 100-year simulation complete!"
```

**Key Points**:
- ⏱️ ~5-180s depending on simulation duration
- 🔄 Polling provides real-time feedback
- 💾 Results cached in agent session for instant follow-ups
- 💰 AWS billing: ECS + S3 storage/transfer
- 🔗 User can interact with results immediately after job completes

---

### Scenario 3: Chat Query Without Simdata (NASA Archive Fetch)

```
Browser (Dash UI)
         │ POST /chat/stream
         │ {message: "Fetch Kepler-442 b system parameters", simdata: null}
         ↓
Agent Service
         │
         ├─ _chat_with_claude()
         │  ├─ Claude API → calls fetch_exoplanet tool
         │  │
         │  ├─ _execute_tool("fetch_exoplanet", "Kepler-442 b")
         │  │  ├─ fetch_system_by_planet("Kepler-442 b")
         │  │  │  └─ HTTP GET exoplanetarchive.ipac.caltech.edu API
         │  │  │     └─ Returns: {Ts: 5326K, Rs: 0.77R☉, Mp: 8.3M⊕, ...}
         │  │  └─ Returns: {system_name: "Kepler-442", planet_name: "Kepler-442 b", Ts: 5326, ...}
         │  │
         │  └─ Claude formulates: "Kepler-442 b orbits a K-dwarf (5326K)..."
         │
         └─ NO backend job, NO simdata cache
                 │
                 ↓ SSE: "Found Kepler-442 b (host: Kepler-442).
                 │      Stellar Ts=5326 K, planet mass=8.3 M⊕, ap=0.40 AU"
                 │
         Browser renders response in chat drawer
```

**Key Points**:
- ✅ Instant (external API call only)
- 📡 No AWS services involved
- 📊 Provides reference data for parameter selection
- 🎯 Can be followed up with "Run simulation with these params"

---

### Scenario 4: MCP Server Integration (Claude Desktop - Local Only)

```
Claude Desktop App
(User's machine: Windows, macOS, Linux)
         │ MCP protocol (stdio)
         │ Tool: run_sim_years with params={Ts: 5326, mp: 8.3, ...}, years=50
         ↓
MCP Server Process
mcp_server.py (FastMCP)
listening on stdio (subprocess of Claude Desktop)
         │
         ├─ @mcp.tool() run_sim_years()
         │  ├─ Calls run_simulation_for_years(p, years=50) [LOCAL]
         │  │  └─ Numba-compiled integrator runs on host machine
         │  ├─ Builds animation.html (Plotly, written to temp file)
         │  ├─ Generates summary.json
         │  └─ Returns: {figure_path: "/tmp/...", t_end: 50.0, rhill_AU: 0.18, ...}
         │
         └─ NO network calls, NO AWS, NO Agent Service
                 │
                 ↓ MCP response (JSON)
                 │
         Claude Desktop displays results in chat
```

**Key Points**:
- ⚡ Fastest (pure local computation)
- 🔒 No cloud costs, no data upload
- 🎯 Suitable for offline analysis
- 📦 No dependency on web services
- ✅ Perfect for private testing

---

## Client-Server Architecture Summary

| Scenario | Client | Server | AWS Used | Response Time | Use Case |
|----------|--------|--------|----------|---------------|----------|
| **1. Local Stability** | Dash Browser | Agent Service | ❌ None | < 100ms | Quick queries on cached data |
| **2. Backend Job** | Dash Browser | Agent Service → Step Functions | ✅ S3, Step Fn, ECS | 5-180s | Long simulations, new parameters |
| **3. NASA Archive** | Dash Browser | Agent Service → External API | ❌ None | 1-5s | Parameter discovery |
| **4. MCP Desktop** | Claude Desktop | MCP Server (local) | ❌ None | 10-60s | Offline analysis |

**Server Identities**:
- **Agent Service**: Always a server (when called from Dash); always a client to AWS + Claude API
- **Dash**: Dual role - server (serves UI), client (talks to Agent Service)
- **MCP Server**: Only a server; no network, no clients other than Claude Desktop
- **AWS Services**: Servers accessed by Agent Service as client (S3, Step Functions, etc.)

---

### Chat Message → Tool Dispatch Flow

```
INPUT: ChatRequest
├── message: "Query about moon stability"
├── simdata: null (or base64 from prior job)
├── params: {} 
├── years: 10.0
└── escape_factor: 1.0

↓

_chat_with_claude() Main Loop:
├─ 1. Call Claude API with message + tool specs
│  │
│  ├─ Tool specs include:
│  │  • fetch_exoplanet(name) → planet from NASA archive
│  │  • run_simulation(params, years, escape_factor)
│  │  • stability_from_simdata(simdata, years, escape_factor)
│  │  • export_csv(simdata) → CSV file handle
│  │  • eda_plot(simdata, variables) → plot HTML
│  │
│  └─ Claude decides: which tools to call?
│
├─ 2. Handle tool calls in _execute_tool():
│  │
│  ├─ If "run_simulation":
│  │  ├─ Call _start_backend_job() → job_id + status="submitted"
│  │  ├─ Return immediately (NO polling)
│  │  └─ Response: "Job agent-xyz789 submitted. Check status later."
│  │
│  ├─ If "stability_from_simdata":
│  │  ├─ Try user-provided simdata
│  │  ├─ If none: _session.get_cached() → use stored simdata
│  │  ├─ Call unpack_sim() → sim dict
│  │  ├─ Call assess_moon_stability() → analysis
│  │  └─ Return: stability bool, escape_time, max_r_rel
│  │
│  ├─ If "export_csv":
│  │  ├─ Same fallback chain for simdata
│  │  ├─ Call unpack_sim() → sim dict (NO re-run!)
│  │  ├─ Call traj_to_frame(sim) → DataFrame
│  │  ├─ Call to_csv_bytes(frame) → CSV bytes
│  │  └─ Write to file / S3 presigned URL
│  │
│  └─ If "eda_plot":
│       ├─ Same fallback chain
│       ├─ Build Plotly figure
│       └─ Return HTML as string
│
├─ 3. Loop: Claude reads tool results → calls more tools?
│  │
│  └─ Example sequence:
│     1. Claude: "run_simulation({...})" → Job submitted
│     2. Claude reads: "ok, job_id=agent-xyz"
│     3. Claude: "Let me wait and check status"
│     4. (But NO polling in agent; return to user)
│
└─ 4. Final message: Stream token-by-token to Dash via SSE

OUTPUT: StreamingResponse
├── meta: {mode, job_id}
├── token: word 1
├── token: word 2
├── ...
└── done: {full_message, job_id, ...}
```

### SessionCache Lifecycle

```
Session Created:
┌─────────────────────────────────────────┐
│ _session = SessionCache()                │
│ • last_job_id = None                    │
│ • cached_simdata = None                 │
│ • cached_params = {}                    │
└─────────────────────────────────────────┘

Message 1: "Run simulation for 10 years"
│ ├─ Agent calls _start_backend_job()
│ └─ _session.update_job("agent-abc", "outputs/agent-abc")
│    • last_job_id = "agent-abc"
│    • cached_simdata = None (CLEARED)
│
│ Response: "Started job agent-abc"

Polling (Dash every 5s):
│ ├─ Status: RUNNING
│ ├─ Status: RUNNING
│ └─ Status: SUCCEEDED
│    ├─ Dash calls /job/agent-abc/retrieve_simdata
│    ├─ Endpoint fetches traj.pkl from S3
│    └─ _session.set_simdata(packed_sim, {})
│       • cached_simdata = "eyJkdCI6IDAuMDAx..." (base64)
│       • cached_params = {}
│
│ Chat updates: "✅ Results ready!"

Message 2: "What's the max moon distance?"
│ ├─ Claude tool: "stability_from_simdata()"
│ ├─ Agent: simdata not in request
│ │  ├─ _session.get_cached() → returns cached_simdata
│ │  ├─ unpack_sim() → sim dict
│ │  └─ Extract max_r_rel from trajectory
│ │
│ └─ Response: "Max distance: 0.38 AU" (instant, no re-sim!)

Message 3: "Export as CSV"
│ ├─ Claude tool: "export_csv()"
│ ├─ Agent: simdata not in request
│ │  ├─ _session.get_cached() → returns cached_simdata
│ │  ├─ unpack_sim() → sim dict
│ │  ├─ traj_to_frame() → DataFrame
│ │  └─ to_csv_bytes() → CSV data
│ │
│ └─ Response: "CSV ready at s3://.../traj.csv" (instant, no re-sim!)

Message 4: "Run new sim with different params"
│ ├─ Agent calls _start_backend_job() with NEW params
│ └─ _session.update_job("agent-xyz", "outputs/agent-xyz")
│    • last_job_id = "agent-xyz" (UPDATED)
│    • cached_simdata = None (CLEARED again)
│
│ → Cycle repeats for new job
```

---

## Deployment Guide

### Prerequisites

- AWS Account with:
  - S3 bucket (for inputs/outputs)
  - Step Functions state machine (for job orchestration)
  - ECS/Fargate or Batch (for compute jobs)
  - IAM roles with S3 + Step Functions permissions
  
- Docker buildx (multi-platform builds)
- AWS CLI configured with credentials
- Python 3.12+

---

### 1. AWS Credential Setup

**Set environment variables**:
```bash
export AWS_REGION=eu-west-2
export EXOMOON_BUCKET=my-exomoon-bucket
export STATE_MACHINE_ARN=arn:aws:states:eu-west-2:123456789:stateMachine:ExomoonStateMachine
export ANTHROPIC_API_KEY=sk-ant-... # Claude API key
```

**OR store in `.env` file** (loaded by docker-compose):
```
AWS_REGION=eu-west-2
EXOMOON_BUCKET=my-exomoon-bucket
STATE_MACHINE_ARN=arn:aws:states:eu-west-2:123456789012:stateMachine:ExomoonStateMachine
ANTHROPIC_API_KEY=sk-ant-...
```

---

### 2. Build Docker Images

**Frontend (Dash UI)**:
```bash
docker build -f Dockerfile -t your-registry/exomoon-dash:latest .
docker tag your-registry/exomoon-dash:latest your-registry/exomoon-dash:v1.0
docker push your-registry/exomoon-dash:latest
docker push your-registry/exomoon-dash:v1.0
```

**Backend Agent Service**:
```bash
docker build -f agent.Dockerfile -t your-registry/exomoon-agent:latest .
docker tag your-registry/exomoon-agent:latest your-registry/exomoon-agent:v1.0
docker push your-registry/exomoon-agent:latest
docker push your-registry/exomoon-agent:v1.0
```

**Batch Job Runner**:
```bash
docker build -f job.Dockerfile -t your-registry/exomoon-batch:latest .
docker tag your-registry/exomoon-batch:latest your-registry/exomoon-batch:v1.0
docker push your-registry/exomoon-batch:latest
docker push your-registry/exomoon-batch:v1.0
```

---

### 3. Docker Compose (Local Multi-Container)

**File**: `compose.yaml`

```yaml
version: '3.8'
services:
  agent:
    build:
      context: .
      dockerfile: agent.Dockerfile
    image: exomoon-agent:latest
    ports:
      - "8000:8000"
    environment:
      - PYTHONPATH=/app/src
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - ANTHROPIC_MODEL=claude-sonnet-4-20250514
      - AWS_ENABLED=1
      - AWS_REGION=${AWS_REGION}
      - EXOMOON_BUCKET=${EXOMOON_BUCKET}
      - STATE_MACHINE_ARN=${STATE_MACHINE_ARN}
      - AWS_SHARED_CREDENTIALS_FILE=/root/.aws/credentials
    volumes:
      - ~/.aws/credentials:/root/.aws/credentials:ro
    networks:
      - exomoon

  dash:
    build:
      context: .
      dockerfile: Dockerfile.dash
    image: exomoon-dash:latest
    ports:
      - "8050:8050"
    environment:
      - PYTHONPATH=/app/src
      - HOST=0.0.0.0
      - PORT=8050
      - DEBUG=1
      - AGENT_SERVICE_URL=http://agent:8000
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - AWS_ENABLED=1
      - AWS_REGION=${AWS_REGION}
      - EXOMOON_BUCKET=${EXOMOON_BUCKET}
      - STATE_MACHINE_ARN=${STATE_MACHINE_ARN}
      - AWS_SHARED_CREDENTIALS_FILE=/root/.aws/credentials
    volumes:
      - ~/.aws/credentials:/root/.aws/credentials:ro
    depends_on:
      - agent
    networks:
      - exomoon

networks:
  exomoon:
    driver: bridge
```

**Run locally**:
```bash
docker-compose up -d
# Dash UI at http://localhost:8050
# Agent service at http://localhost:8000
```

---

### 4. AWS Service Deployment

**Agent Service** (ECS/Fargate):
```bash
# Create ECS task definition JSON
aws ecs register-task-definition --cli-input-json file://task-def-agent.json

# Update service
aws ecs update-service --cluster exomoon-cluster --service exomoon-agent \
  --force-new-deployment

# Get Network LB DNS
aws elbv2 describe-load-balancers --names exomoon-agent-nlb \
  --query 'LoadBalancers[0].DNSName' --output text
# → exomoon-agent-nlb-abc123.elb.eu-west-2.amazonaws.com
```

**Dashboard** (ECS/Fargate):
```bash
aws ecs register-task-definition --cli-input-json file://task-def-dash.json
aws ecs update-service --cluster exomoon-cluster --service exomoon-dash \
  --force-new-deployment
```

**Batch Job Registry** (ECR):
```bash
aws ecr get-login-password --region eu-west-2 | \
  docker login --username AWS --password-stdin 123456789.dkr.ecr.eu-west-2.amazonaws.com

docker tag exomoon-batch:latest 123456789.dkr.ecr.eu-west-2.amazonaws.com/exomoon-batch:v1.0
docker push 123456789.dkr.ecr.eu-west-2.amazonaws.com/exomoon-batch:v1.0
```

---

### 5. Update Step Functions State Machine

**Upload new batch job image ARN**:
```bash
aws stepfunctions update-state-machine \
  --state-machine-arn arn:aws:states:eu-west-2:123456789:stateMachine:ExomoonStateMachine \
  --definition '{
    "Comment": "Exomoon orbital simulation job",
    "StartAt": "SpawnBatchJob",
    "States": {
      "SpawnBatchJob": {
        "Type": "Task",
        "Resource": "arn:aws:states:::ecs:runTask.sync",
        "Parameters": {
          "LaunchType": "FARGATE",
          "Cluster": "exomoon-cluster",
          "TaskDefinition": "exomoon-batch:1",
          "NetworkConfiguration": {...},
          "Overrides": {
            "ContainerOverrides": [{
              "Name": "exomoon-batch",
              "Environment": [
                {"Name": "INPUT_S3", "Value.$": "$.inputS3Prefix"},
                {"Name": "OUTPUT_S3", "Value.$": "$.outputS3Prefix"}
              ]
            }]
          }
        },
        "End": true
      }
    }
  }'
```

---

### 6. Health Check & Monitoring

**Test agent service**:
```bash
curl -X GET http://your-agent-nlb.elb.eu-west-2.amazonaws.com:8000/docs
# Should return Swagger UI
```

**Test chat endpoint**:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "simdata": null, "params": {}}'
```

**Dash UI**:
```
Open browser: http://localhost:8050
Check console for [ENV] debug logs
Verify AGENT_SERVICE_URL points to LB endpoint
```

---

## Environment Variables

All environment variables are **required** for production. Defaults allow local testing.

| Variable | Type | Example | Impact |
|----------|------|---------|--------|
| `AWS_ENABLED` | bool (0\|1) | `1` | Enables AWS backend (S3, Step Functions). Set `0` for local dev. |
| `AWS_REGION` | string | `eu-west-2` | AWS region for all services (S3, Step Functions, ECS). Must match bucket + state machine region. |
| `EXOMOON_BUCKET` | string | `my-exomoon-bucket` | S3 bucket name for inputs/outputs. Agent writes `inputs/{job_id}/`, `outputs/{job_id}/` prefixes. Required if `AWS_ENABLED=1`. |
| `STATE_MACHINE_ARN` | ARN string | `arn:aws:states:eu-west-2:123456789:stateMachine:ExomoonStateMachine` | Full ARN of Step Functions state machine. Agent calls `sf.start_execution()` with this. Required if `AWS_ENABLED=1`. |
| `ANTHROPIC_API_KEY` | string | `sk-ant-v0-...` | Claude API key from console.anthropic.com. Agent uses for multi-turn reasoning. |
| `ANTHROPIC_MODEL` | string | `claude-sonnet-4-20250514` | Claude model ID. Controls LLM capability (reasoning, tool use, latency). Current production: Sonnet 4. |
| `CLAUDE_ENABLED` | bool (0\|1) | `1` | Gates Claude client initialization. Set `0` to run agent without LLM (debugging only). |
| `AGENT_SERVICE_URL` | URL string | `http://exomoon-agent-nlb-451ecd.elb.eu-west-2.amazonaws.com:8000` | Dash uses this to call agent endpoints (`/chat`, `/job/{job_id}/status`, `/job/{job_id}/retrieve_simdata`). Local: `http://127.0.0.1:8000`. Production: NLB DNS. |
| `HTTP_PROXY` / `HTTPS_PROXY` | URL string | `http://proxy.corp:8080` | (Optional) Proxy for outbound requests (Claude API, NASA archive). Leave unset if direct internet access. |
| `PYTHONPATH` | path string | `/app/src` | Python module search path. Docker images set to enable `import exomoon.*`. |
| `HOST` | IP address | `0.0.0.0` | Dash server bind address. `0.0.0.0` for Docker (listen all interfaces). |
| `PORT` | int | `8050` | Dash server port (frontend). `8000` for agent service. |
| `DEBUG` | bool (0\|1) | `1` | Dash debug mode. Enables hot-reload. Set `0` for production. |
| `AWS_SHARED_CREDENTIALS_FILE` | path | `/root/.aws/credentials` | Path to AWS credentials file. Docker mounts from `~/.aws/credentials:ro`. |

**Example `.env` for Production**:
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

**Example for Local Testing**:
```bash
AWS_ENABLED=0
ANTHROPIC_API_KEY=sk-ant-... # Still needed for MCP server / local agent
AGENT_SERVICE_URL=http://127.0.0.1:8000
DEBUG=1
```

---

## Pending Issues

### CSV File Retrieval via Chatbot

**Current Status**: ⚠️ **Partially Working**

**What Works**:
- ✅ CSV file generated server-side in `run_job.py` (`traj_to_frame()` + `to_csv_bytes()`)
- ✅ Uploaded to S3: `s3://bucket/outputs/{job_id}/traj.csv`
- ✅ Presigned URL generated in `/job/{job_id}/status` endpoint (valid 24h)
- ✅ Tool handler `export_csv` can deserialize cached simdata (no re-simulation)
- ✅ CSV data is available in agent (`to_csv_bytes()` returns bytes)

**What's Broken**:
- ❌ No mechanism to **stream** CSV file from agent service to Dash UI
- ❌ No download link shown in chat interface
- ❌ User cannot click a button in chat to save/download CSV

**Root Cause**:
The `export_csv` tool generates CSV bytes but has no HTTP endpoint to deliver them to the browser. Agent response is text-only (streaming SSE).

**Solution Path** (Priority: Medium):

1. **Option A: Presigned URL (Quick)**
   - `export_csv` tool returns S3 presigned URL instead of bytes
   - Chat shows clickable link: "[Download traj.csv](#)"
   - User clicks → browser downloads from S3
   - Requires: Minor tool handler modification (1 line change)

2. **Option B: Agent Endpoint (Robust)**
   - Add `/csv/{job_id}` endpoint to agent service
   - Streams CSV bytes as `application/csv` attachment
   - Dash chat renders `<a href="/csv/{job_id}">` link
   - Requires: New endpoint + chat link renderer

3. **Option C: Direct S3 Download (Cleanest)**
   - User queries: "Export CSV"
   - Agent finds CSV in S3 via `/job/{job_id}/status` URLs
   - Returns presigned URL in response
   - Chat renders as markdown link
   - Requires: Agent logic to extract URL from status response

**Recommendation**: Implement **Option A** first (1-line fix), then add **Option C** for polish.

**Code Change Needed** (Option A):

In `agent_service.py`, `export_csv` tool handler:
```python
if tool_name == "export_csv":
    # ... existing code ...
    
    # Instead of writing bytes:
    # csv_url = s3.generate_presigned_url("get_object", ...)
    
    # Return URL to user
    return {
        "ok": True,
        "csv_url": f"https://bucket.s3.amazonaws.com/.../traj.csv?X-Amz-Signature=...",
        "message": "CSV ready for download (24h validity)"
    }
```

Chat renders: "CSV ready for download → [Download](url)"

---

## Next Steps

### Immediate (Priority: High)

1. **CSV Download Fix**
   - Implement Option A (presigned URL in tool response)
   - Update `export_csv` handler (1-2 lines)
   - Test with live chatbot query
   - Estimated time: 15 minutes

2. **Production Hardening**
   - Add request rate limiting to agent endpoints
   - Implement session timeout (30 min inactivity → clear cache)
   - Add comprehensive error logging + alerting
   - Set up CloudWatch dashboards for job success rates

### Medium-Term (Priority: Medium)

1. **Performance Optimization**
   - Profile Numba integrator for hot spots
   - Consider GPU acceleration for large N-body systems
   - Add result caching layer (same parameters → return stored result)

2. **User Experience**
   - Add simulation progress indicator (% complete via Step Functions describe_execution)
   - Batch multiple queries into single simulation job
   - Allow parameter presets (e.g., "Earth-like system", "Kepler-442b")

3. **Multi-User Support**
   - Replace in-memory `SessionCache` with Redis
   - Implement user authentication (API keys)
   - Isolate jobs per user
   - Track usage metrics

### Long-Term (Priority: Low)

1. **Exoplanet Archive Integration**
   - Pre-compute stability analysis for known exoplanets
   - Build searchable database of stable moon configurations
   - Real-time archive updates (weekly sync)

2. **Advanced Physics**
   - Tide-moon migration model
   - Resonance effects (moon-moon interactions)
   - Relativistic corrections for compact systems

3. **Visualization Enhancements**
   - 3D interactive system viewer
   - Orbital animation with real-time physics overlay
   - Habitability heatmap (position vs. time)

---

## Appendix: Key Code References

**Critical Logic Paths**:
- **Non-blocking job submission**: `agent_service.py` line ~358 (`_start_backend_job`)
- **Simdata caching**: `agent_service.py` line ~72 (`SessionCache` class)
- **Simdata retrieval**: `agent_service.py` line ~1050 (`retrieve_job_simdata` endpoint)
- **Polling integration**: `run_dash.py` line ~710 (`poll_agent_job_status` callback)
- **Physics engine**: `integrator.py` line ~15 (`_leapfrog_integrate`)
- **Serialization**: `eda.py` line ~56 (`pack_sim` / `unpack_sim`)

**Data Flow References**:
- Input params: `exomoon/params.py`
- Initial state: `exomoon/initial_conditions.py`
- Simulation output: `exomoon/simulation.py` return dict
- CSV generation: `exomoon/eda.py` (`traj_to_frame` → `to_csv_bytes`)
- Stability analysis: `exomoon/moon_stability.py` (`assess_moon_stability`)

