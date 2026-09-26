import os
import json
import re
import uuid
import time
import signal
import pathlib
import threading
import traceback
import hashlib
import requests as _requests
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import numpy as np
import boto3
import botocore
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from exomoon.params import SystemParams
from exomoon.constants import FOUR_PI2, merth, msun
from exomoon.eda import unpack_sim, traj_to_frame, to_csv_bytes, pack_sim
from exomoon.simulation import run_simulation, run_simulation_for_years
from exomoon.exoplanet_archive import fetch_system_by_planet
# Import the FunctionTool wrappers and unwrap to raw callables via .fn
# (fastmcp @mcp.tool() returns FunctionTool objects, not plain functions)
from exomoon.mcp_server import (
    env_info  as _mcp_env_info,
    dash_url  as _mcp_dash_url,
    export_csv as _mcp_export_csv,
    eda_plot  as _mcp_eda_plot,
)
env_info   = getattr(_mcp_env_info,   'fn', _mcp_env_info)
_dash_url  = getattr(_mcp_dash_url,   'fn', _mcp_dash_url)
_mcp_export_csv_fn = getattr(_mcp_export_csv, 'fn', _mcp_export_csv)
_mcp_eda_plot_fn   = getattr(_mcp_eda_plot,   'fn', _mcp_eda_plot)

# NEW: Claude SDK
try:
    import anthropic
except Exception:
    anthropic = None


def _force_exit(sig, frame):
    """Force-exit immediately so Ctrl+C isn't blocked by in-flight Claude calls."""
    print("\n[SHUTDOWN] Signal received — exiting immediately.", flush=True)
    os._exit(0)


@asynccontextmanager
async def lifespan(app):
    # Re-install SIGINT/SIGTERM after uvicorn has set its own handlers.
    # This ensures Ctrl+C kills the process immediately rather than waiting
    # for synchronous thread-pool tasks (Claude tool loops) to finish.
    signal.signal(signal.SIGINT,  _force_exit)
    signal.signal(signal.SIGTERM, _force_exit)
    yield


app = FastAPI(title="Exomoon Agent Service", version="0.1.0", lifespan=lifespan)

# FRONTEND_ORIGIN may be a comma-separated list of allowed origins.
# "*" (the default) allows all origins.  Always add localhost for local dev.
_cors_env = os.getenv("FRONTEND_ORIGIN", "*")
if _cors_env == "*":
    _allowed_origins: list[str] = ["*"]
else:
    _allowed_origins = [o.strip() for o in _cors_env.split(",") if o.strip()]
    _local_origins = [
        "http://localhost:3000", "http://localhost:3001",
        "http://127.0.0.1:3000", "http://127.0.0.1:3001",
    ]
    for _o in _local_origins:
        if _o not in _allowed_origins:
            _allowed_origins.append(_o)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chrome Private Network Access — pure ASGI middleware (avoids BaseHTTPMiddleware +
# StreamingResponse interaction that can bubble streaming body exceptions up as HTTP 500).
from starlette.types import ASGIApp as _ASGIApp, Receive as _Receive, Scope as _Scope, Send as _Send

class _PNAMiddleware:
    def __init__(self, app: _ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: _Scope, receive: _Receive, send: _Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def _send_with_pna(message: dict) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"access-control-allow-private-network", b"true"))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, _send_with_pna)

app.add_middleware(_PNAMiddleware)

# Global exception handler — logs the full traceback for ANY unhandled exception
# that reaches FastAPI's default 500 handler, so we can see exactly what escaped.
from fastapi import Request as _Request
from fastapi.responses import JSONResponse as _JSONResponse
import traceback as _tb_global

@app.exception_handler(Exception)
async def _global_exception_handler(_req: _Request, exc: Exception) -> _JSONResponse:
    _tb_str = _tb_global.format_exc()
    print(f"[GLOBAL_EXC] Unhandled exception on {_req.method} {_req.url.path}: {exc}", flush=True)
    print(_tb_str, flush=True)
    return _JSONResponse(status_code=500, content={"detail": str(exc), "type": type(exc).__name__})

# Serve static outputs (EDA PNGs, animation.html) at GET /outputs/<filename>
_OUTPUTS_DIR = pathlib.Path("outputs")
_OUTPUTS_DIR.mkdir(exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(_OUTPUTS_DIR)), name="outputs")

AWS_ENABLED = os.getenv("AWS_ENABLED", "0") == "1"
BUCKET = os.getenv("EXOMOON_BUCKET")
STATE_MACHINE_ARN = os.getenv("STATE_MACHINE_ARN")
AWS_REGION = os.getenv("AWS_REGION", "eu-west-2")

# NEW: Claude config
ANTHROPIC_API_KEY_RAW = os.getenv("ANTHROPIC_API_KEY", "").strip()

# DEBUG: Check if it's JSON (from Secrets Manager)
if ANTHROPIC_API_KEY_RAW.startswith("{"):
    try:
        import json
        secret_json = json.loads(ANTHROPIC_API_KEY_RAW)
        ANTHROPIC_API_KEY = secret_json.get("ANTHROPIC_API_KEY", secret_json.get("api_key", ""))
        print(f"[STARTUP] Parsed API key from JSON secret", flush=True)
    except Exception as e:
        ANTHROPIC_API_KEY = ANTHROPIC_API_KEY_RAW
        print(f"[STARTUP] Failed to parse JSON secret: {e}", flush=True)
else:
    ANTHROPIC_API_KEY = ANTHROPIC_API_KEY_RAW

ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")

CLAUDE_ENABLED = os.getenv("CLAUDE_ENABLED", "0") == "1"

claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if CLAUDE_ENABLED else None

# Startup diagnostics
print(f"[STARTUP] CLAUDE_ENABLED={CLAUDE_ENABLED}, AWS_ENABLED={AWS_ENABLED}", flush=True)
print(f"[STARTUP] anthropic available={anthropic is not None}, client created={claude is not None}", flush=True)

# NumPy version guard — Numba 0.60 requires NumPy 1.26.x (not 2.x).
# NumPy 2.0+ causes Numba's LLVM compilation path to hang on Windows when
# the cache is cold (i.e. after __pycache__ is deleted or first install).
_np_ver = tuple(int(x) for x in np.__version__.split(".")[:2])
if _np_ver >= (2, 0):
    print(
        f"[STARTUP] WARNING: NumPy {np.__version__} is incompatible with Numba 0.60 on Windows — "
        f"Numba JIT will hang on cold cache. "
        f"Fix: pip install 'numpy==1.26.4' then delete __pycache__ dirs and restart.",
        flush=True,
    )

s3 = boto3.client("s3", region_name=AWS_REGION) if AWS_ENABLED and BUCKET else None
sf = boto3.client("stepfunctions", region_name=AWS_REGION) if AWS_ENABLED and STATE_MACHINE_ARN else None

# ── GPU trajectory-preview service (EC2 g4dn.xlarge, hnn_gpu_service.py) ──────
GPU_SERVICE_URL       = os.getenv("GPU_SERVICE_URL", "http://52.56.252.104:8001")
GPU_SERVICE_TIMEOUT_S = int(os.getenv("GPU_SERVICE_TIMEOUT_S", "2400"))
# Inference cache — separate S3 bucket so existing nbody-time-series-storage is untouched
INFERENCE_CACHE_BUCKET = os.getenv("INFERENCE_CACHE_BUCKET", "exomoon-ml-inference-cache")
# Bump MODEL_VERSION when HNN weights are updated; old cache entries are automatically orphaned
HNN_MODEL_VERSION = os.getenv("HNN_MODEL_VERSION", "hinge4_v1")
# S3 client for inference cache — independent of AWS_ENABLED so cache works even in local mode
# (credentials still required; _read_cache/_write_cache catch ClientError if unavailable)
_s3_cache = boto3.client("s3", region_name=AWS_REGION)

# ── In-RAM trajectory cache — keeps full (N, n_out, 3) arrays after each batch run ──────────
# Keyed by the same cache key as _inference_cache_key(). LRU-capped at _MAX_TRAJ_RAM entries.
# Populated after every successful _forward_to_gpu call; looked up by /trajectory/cell_preview.
_traj_ram_cache: Dict[str, dict] = {}
_traj_ram_lock  = threading.Lock()
_MAX_TRAJ_RAM   = 3   # keep at most 3 batch results in RAM (~90 MB each)

# Developer/user mode flag — toggled by keyword in chat messages (never exposed to user)
_developer_mode: bool = False


# NEW: Session cache to track last job + simdata across multiple chat messages
class SessionCache:
    """Store conversation state: last job_id, output_prefix, cached simdata."""
    def __init__(self):
        self.last_job_id: Optional[str] = None
        self.last_output_prefix: Optional[str] = None
        self.cached_simdata: Optional[str] = None
        self.cached_params: Dict[str, Any] = {}
        self.last_animation_url: Optional[str] = None
        self.last_ml_prediction: Optional[Dict[str, Any]] = None
        self._ml_fresh: bool = False  # True only for the turn in which ml_predict was called
        # Trajectory batch state — populated when trajectory_preview tool hits cache
        self.last_traj_key: Optional[str] = None
        self.last_traj_mm_grid: list = []
        self.last_traj_am_grid: list = []
        # Per-cell trajectory frames — populated when trajectory_cell_query tool is called
        self.last_cell_frames: Optional[list] = None
        self.last_cell_rhill_au: Optional[float] = None
        self.last_cell_roche_frac: Optional[float] = None
        self.last_cell_html_2d_url: Optional[str] = None
        self.last_cell_html_3d_url: Optional[str] = None
        self.last_cell_mm_earth: Optional[float] = None
        self.last_cell_am_hill: Optional[float] = None
        self._cell_frames_fresh: bool = False
        self.last_effective_params: Optional[Dict[str, Any]] = None  # params used for last chat-triggered job
        self._job_fresh: bool = False  # True only for the turn in which start_backend_job was called
        self.last_traj_preview: Optional[Dict[str, Any]] = None  # Layer 2 trajectory batch result (separate from Layer 1 MLP)
        self._traj_preview_fresh: bool = False  # True for the turn trajectory_preview hits cache
        # Full conversation history (user + assistant + tool_result messages, including thinking blocks).
        # Prepended to messages on each new turn so Claude remembers the whole conversation.
        self.conversation_history: list = []

    def update_job(self, job_id: str, output_prefix: str):
        """Called when a new job is started."""
        self.last_job_id = job_id
        self.last_output_prefix = output_prefix
        self.cached_simdata = None  # Clear old simdata when new job starts
    
    def set_simdata(self, simdata: str, params: Dict[str, Any]):
        """Cache simdata from completed job or user-provided."""
        self.cached_simdata = simdata
        self.cached_params = params
    
    def get_cached(self) -> tuple[Optional[str], Dict[str, Any]]:
        """Return cached simdata and params if available."""
        return self.cached_simdata, self.cached_params
    
    def try_retrieve_job_results(self, max_retries: int = 5, retry_delay: float = 2.0) -> Optional[str]:
        """
        Poll S3 for job completion then retrieve simdata.
        Returns simdata if successful, None otherwise.
        """
        if not (self.last_job_id and self.last_output_prefix and s3 and BUCKET):
            return None
        
        try:
            import time
            # Poll for completion marker
            for attempt in range(max_retries):
                try:
                    marker_key = f"{self.last_output_prefix}/COMPLETE"
                    s3.head_object(Bucket=BUCKET, Key=marker_key)
                    print(f"[SESSION] Job complete on attempt {attempt+1}", flush=True)
                    break
                except botocore.exceptions.ClientError as e:
                    if e.response['Error']['Code'] == '404':
                        if attempt < max_retries - 1:
                            print(f"[SESSION] Polling... (attempt {attempt+1}/{max_retries})", flush=True)
                            time.sleep(retry_delay)
                        else:
                            return None
                    else:
                        raise
            
            # Retrieve traj.pkl
            key = f"{self.last_output_prefix}/traj.pkl"
            print(f"[SESSION] Retrieving simdata from s3://{BUCKET}/{key}", flush=True)
            obj = s3.get_object(Bucket=BUCKET, Key=key)
            simdata = obj['Body'].read().decode('utf-8')
            print(f"[SESSION] Retrieved ({len(simdata)} chars)", flush=True)
            self.cached_simdata = simdata
            return simdata
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] != '404':
                print(f"[SESSION] S3 error: {e}", flush=True)
            return None
        except Exception as e:
            print(f"[SESSION] Error: {e}", flush=True)
            return None

# Per-session cache dict — keyed by session_id (UUID from browser localStorage).
# Replaces the old single global `_session` that caused cross-user contamination.
# LRU-capped at 200 entries (oldest evicted when limit is reached).
_sessions: Dict[str, SessionCache] = {}

# Reverse map: job_id → session_key — so retrieve_simdata can find the right session
# without requiring the frontend to pass session_id on that endpoint.
_job_to_session: Dict[str, str] = {}


def _get_or_create_session(session_id: Optional[str]) -> SessionCache:
    key = session_id or "default"
    if key not in _sessions:
        if len(_sessions) >= 200:
            oldest = next(iter(_sessions))
            del _sessions[oldest]
        _sessions[key] = SessionCache()
    return _sessions[key]


# ── Local job store (used when AWS_ENABLED=0) ──────────────────────────────────
# Maps job_id → {status, started, elapsed, csv_bytes, summary, simdata, error}
LOCAL_JOBS: Dict[str, Dict] = {}
_LOCAL_AGENT_BASE = os.getenv("AGENT_BASE_URL", "http://localhost:8000")


def _run_local_job(job_id: str, params_dict: Dict, years: float, session: Optional[SessionCache] = None) -> None:
    """Background thread: run simulation locally, store result in LOCAL_JOBS."""
    LOCAL_JOBS[job_id]["started"] = time.time()
    try:
        # Build SystemParams from the flat dict (ignore unknown keys)
        import dataclasses
        known = {f.name for f in dataclasses.fields(SystemParams)}
        p = SystemParams(**{k: v for k, v in params_dict.items() if k in known})
        sim = run_simulation_for_years(p, years) if years > 0 else run_simulation(p)

        frame = traj_to_frame(sim)
        csv_bytes = to_csv_bytes(frame)
        simdata = pack_sim(sim)

        summary = {
            "t_end": sim["t_end"],
            "dt": sim["dt"],
            "rhill_AU": sim["state"].get("rhill_AU"),
            "n_steps": len(sim["traj"]["xyzarr_mp"]),
            "years_requested": years,
            "a_inner_au": sim["a_inner_au"],
            "a_outer_au": sim["a_outer_au"],
        }

        LOCAL_JOBS[job_id].update({
            "status": "SUCCEEDED",
            "csv_bytes": csv_bytes,
            "summary": summary,
            "simdata": simdata,
            "elapsed": time.time() - LOCAL_JOBS[job_id]["started"],
        })
        if session is not None:
            session.set_simdata(simdata, params_dict)
        print(f"[LOCAL-JOB] {job_id} completed ({len(csv_bytes)} csv bytes)", flush=True)
    except Exception as exc:
        LOCAL_JOBS[job_id].update({"status": "FAILED", "error": str(exc)})
        print(f"[LOCAL-JOB] {job_id} FAILED: {exc}", flush=True)
        traceback.print_exc()


class ChatRequest(BaseModel):
    """User message + context (simdata, params, duration, escape threshold)."""
    message: str
    simdata: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    years: Optional[float] = None
    escape_factor: float = 1.0
    ml_prediction: Optional[Dict[str, Any]] = None  # summary from frontend ML predictor (no full arrays)
    # Per-browser-session ID — UUID generated by the frontend on first visit (localStorage).
    # Isolates conversation state, simdata cache, and trajectory grids between users/tabs.
    session_id: Optional[str] = None
    # Layer 2 trajectory preview state — sent from frontend so trajectory_cell_query works
    # even when the batch was run from the panel (not from a chatbot trajectory_preview call).
    traj_preview_key: Optional[str] = None       # cache_key from TrajPreview in Zustand store
    traj_mm_grid: Optional[list] = None          # mm_grid from TrajPreview (small array, ~30 floats)
    traj_am_grid: Optional[list] = None          # am_grid from TrajPreview


class StabilityRequest(BaseModel):
    """Request to assess moon stability from existing simdata (no rerun)."""
    simdata: str
    params: Dict[str, Any]
    years: Optional[float] = None
    escape_factor: float = 1.0


class PlanetRequest(BaseModel):
    """Exoplanet name lookup."""
    name: str


class ToolRequest(BaseModel):
    """Generic tool invocation (params, optional duration, variables, plot config)."""
    params: Dict[str, Any] = Field(default_factory=dict)
    years: Optional[float] = None
    variables: Optional[list[str]] = None
    columns: Optional[list[str]] = None
    plot_type: str = "line"
    normalize: bool = False


def _to_params(d: Dict[str, Any]) -> SystemParams:
    """
    Convert dict to SystemParams.
    Duplicated here (also in mcp_server.py) because agent needs to run independently
    without calling mcp_server functions that may be slow or not available in cloud.
    """
    base = SystemParams()
    def f(k: str, default: float) -> float:
        v = d.get(k, default)
        return default if v is None or v == "" else float(v)

    moon_dir = str(d.get("moon_dir", "")).strip().lower()
    moon_retrograde = bool(d.get("moon_retrograde", False)) or moon_dir in ("retro", "retrograde", "r", "1", "true", "yes")

    return SystemParams(
        Ts=f("Ts", base.Ts),
        rs_solar=f("rs_solar", base.rs_solar),
        ms_solar=f("ms_solar", base.ms_solar),
        mp_earth=f("mp_earth", base.mp_earth),
        dp_cgs=f("dp_cgs", base.dp_cgs),
        ap_AU=f("ap_AU", base.ap_AU),
        ep=f("ep", base.ep),
        mm_earth=f("mm_earth", base.mm_earth),
        am_hill=f("am_hill", base.am_hill),
        em=f("em", base.em),
        moon_retrograde=moon_retrograde,
    )


def _hill_radius_au(p: SystemParams) -> float:
    """
    Compute Hill radius (AU) from system params.
    Duplicated here (also in initial_conditions.py) because agent must compute
    stability thresholds fast without importing the full simulation stack.
    Formula: a_p * (1-e_p) * (M_p / (3*M_*))^(1/3)
    """
    ms = p.ms_solar * FOUR_PI2
    mp = p.mp_earth * (merth / msun) * FOUR_PI2
    return float(p.ap_AU * (1.0 - p.ep) * ((mp / (3.0 * ms)) ** (1.0 / 3.0)))


def _assess_stability_from_simdata(simdata: str, params: Dict[str, Any], years: Optional[float], escape_factor: float) -> Dict[str, Any]:
    """
    Check moon stability from *existing* simdata (no rerun).
    Returns: ok, stable, max_r_rel, rhill_AU, threshold, escape_time, needs_rerun.
    
    Key feature (Option A): If simdata covers requested duration, compute stability locally.
    Otherwise flag needs_rerun=True (agent later handles fallback to Step Functions).
    """
    sim = unpack_sim(simdata)
    t_end = float(sim["t_end"])
    dt = float(sim["dt"])

    # Check if simdata is sufficient for requested duration
    if years is not None and t_end + 1e-12 < float(years):
        return {
            "ok": False,
            "message": f"Existing simdata covers {t_end:.6g} years, requested {float(years):.6g} years.",
            "needs_rerun": True,
            "t_end": t_end,
        }

    # Extract moon-planet distance in xy-plane (matches moon_stability convention)
    traj = sim["traj"]
    rel = traj["xyzarr_mm"] - traj["xyzarr_mp"]
    r_rel = np.linalg.norm(rel[:, :2], axis=1)

    # Compute Hill radius and escape threshold
    p = _to_params(params or {})
    rhill = _hill_radius_au(p)
    threshold = float(escape_factor) * rhill
    max_r = float(np.max(r_rel)) if len(r_rel) else 0.0
    stable = bool(max_r <= threshold)

    # If unstable, estimate escape time via linear interpolation
    escape_time = None
    escape_index = None
    if not stable:
        idxs = np.where(r_rel > threshold)[0]
        if idxs.size:
            j = int(idxs[0])
            t_prev = j * dt
            r_prev = r_rel[j - 1] if j > 0 else r_rel[j]
            r_curr = r_rel[j]
            if j > 0 and r_curr > r_prev:
                frac = (threshold - r_prev) / (r_curr - r_prev)
                frac = max(0.0, min(1.0, float(frac)))
                escape_time = float(t_prev + frac * dt)
            else:
                escape_time = float((j + 1) * dt)
            escape_index = j

    return {
        "ok": True,
        "stable": stable,
        "max_r_rel": max_r,
        "rhill_AU": rhill,
        "threshold": threshold,
        "escape_factor": float(escape_factor),
        "escape_time": escape_time,
        "escape_index": escape_index,
        "t_end": t_end,
        "dt": dt,
        "needs_rerun": False,
    }


def _extract_years(msg: str) -> Optional[float]:
    """Parse 'N year' or 'N years' from user message."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*year", msg.lower())
    return float(m.group(1)) if m else None


def _extract_planet(msg: str) -> Optional[str]:
    """Heuristic: extract planet name after 'for' or 'on' in message."""
    # Try "for Kepler-442 b" or "on Kepler-442 b"
    m = re.search(r"\b(?:for|on)\s+([a-z0-9\-\+\.\s]+?)(?:\s+for|\s*$)", msg.strip(), re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fallback: extract anything after "planet"
    m = re.search(r"\bplanet\s+([a-z0-9\-\+\.\s]+?)(?:\s|$)", msg.strip(), re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def _start_backend_job(params: Dict[str, Any], years: Optional[float], check_stability: bool = False, escape_factor: float = 1.0, session: Optional[SessionCache] = None, session_key: str = "default") -> Dict[str, Any]:
    """
    Start a Step Functions job to run the simulation (with optional stability check).
    Returns: job_id, execution_arn, output_prefix, or error dict.
    """
    if not (sf and s3 and STATE_MACHINE_ARN and BUCKET):
        return {
            "ok": False,
            "error": "AWS backend not configured (SF/S3 unavailable).",
        }

    job_id = f"agent-{uuid.uuid4().hex[:12]}"
    inp_prefix = f"inputs/{job_id}"
    out_prefix = f"outputs/{job_id}"

    # Build params dict for Step Functions
    p = _to_params(params or {})
    params_dict = {
        "Ts": p.Ts,
        "rs_solar": p.rs_solar,
        "ms_solar": p.ms_solar,
        "mp_earth": p.mp_earth,
        "dp_cgs": p.dp_cgs,
        "ap_AU": p.ap_AU,
        "ep": p.ep,
        "mm_earth": p.mm_earth,
        "am_hill": p.am_hill,
        "em": p.em,
        "moon_retrograde": p.moon_retrograde,
        "years": float(years) if years else 0.0,
        "check_stability": check_stability,
        "escape_factor": float(escape_factor),
    }

    try:
        # Upload params.json to input prefix
        s3.put_object(
            Bucket=BUCKET,
            Key=f"{inp_prefix}/params.json",
            Body=json.dumps(params_dict).encode(),
        )

        # Start Step Functions execution
        exec_resp = sf.start_execution(
            stateMachineArn=STATE_MACHINE_ARN,
            name=job_id,
            input=json.dumps({
                "inputS3Prefix": f"s3://{BUCKET}/{inp_prefix}",
                "outputS3Prefix": f"s3://{BUCKET}/{out_prefix}",
            })
        )

        # NEW: Store execution_arn in S3 for later retrieval
        job_metadata = {
            "job_id": job_id,
            "execution_arn": exec_resp["executionArn"],
            "output_prefix": out_prefix,
            "bucket": BUCKET,
            "region": AWS_REGION,
        }
        s3.put_object(
            Bucket=BUCKET,
            Key=f"{out_prefix}/job_metadata.json",
            Body=json.dumps(job_metadata).encode(),
        )

        # Update session cache with job info (do NOT poll here - return immediately)
        if session is not None:
            session.update_job(job_id, out_prefix)
            _job_to_session[job_id] = session_key
        print(f"[AGENT] Job {job_id} started (execution_arn: {exec_resp['executionArn']})", flush=True)
        
        # Return immediately - client can check status via get_job_status endpoint
        return {
            "ok": True,
            "job_id": job_id,
            "execution_arn": exec_resp["executionArn"],
            "output_prefix": out_prefix,
            "bucket": BUCKET,
            "status": "submitted",
        }
    except Exception as e:
        return {
            "ok": False,
            "error": f"Failed to start job: {str(e)}",
        }


def _tool_specs() -> list[dict]:
    """Claude tool definitions (function calling schema)."""
    return [
        {
            "name": "fetch_exoplanet",
            "description": "Fetch exoplanet system parameters by planet name from NASA archive.",
            "input_schema": {
                "type": "object",
                "properties": {"name": {"type": "string", "description": "Planet name (e.g., 'Kepler-442 b')"}},
                "required": ["name"],
            },
        },
        {
            "name": "stability_from_simdata",
            "description": "Assess moon stability from existing simdata without rerunning simulation.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "years": {"type": "number", "description": "Duration to check (years)"},
                    "escape_factor": {"type": "number", "description": "Escape threshold multiplier (default 1.0)"},
                },
                "required": [],
            },
        },
        {
            "name": "start_backend_job",
            "description": (
                "Start a Step Functions job to run a new simulation (with optional stability check). "
                "Pass `params` to override any system parameters the user requested — e.g. if the user says "
                "'change moon mass to 0.1 and run', pass {\"mm_earth\": 0.1} and the job runs with that value. "
                "Any key not included in `params` inherits from the current UI configuration. "
                "Valid keys: Ts, rs_solar, ms_solar, mp_earth, dp_cgs, ap_AU, ep, mm_earth, am_hill, em, moon_retrograde."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "years": {"type": "number", "description": "Simulation duration (years)"},
                    "check_stability": {"type": "boolean", "description": "Include stability check in job (default true)"},
                    "escape_factor": {"type": "number", "description": "Escape threshold multiplier (default 1.0)"},
                    "params": {
                        "type": "object",
                        "description": "Parameter overrides — any subset of system params to change from current UI values. E.g. {\"mm_earth\": 0.5, \"am_hill\": 0.3}.",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "get_trajectory_at_time",
            "description": "Query moon/planet/star positions, velocities, and distances at a specific simulation time.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "years": {"type": "number", "description": "Simulation time to query (years, 0 to t_end)"},
                },
                "required": ["years"],
            },
        },
        {
            "name": "export_csv",
            "description": "Export trajectory data to CSV (positions, velocities, distances).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "years": {"type": "number", "description": "Simulation duration (optional, years)"},
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific columns to export (optional)"
                    },
                },
                "required": [],
            },
        },
        {
            "name": "get_trajectory_range",
            "description": "Query trajectory snapshots every N years over a time range (e.g., every 0.5 years from year 0 to year 10).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "t_start": {"type": "number", "description": "Start time (years)"},
                    "t_end": {"type": "number", "description": "End time (years)"},
                    "step": {"type": "number", "description": "Interval between snapshots (years)"},
                },
                "required": ["t_start", "t_end", "step"],
            },
        },
        {
            "name": "env_info",
            "description": "Debug: get Python interpreter path and module resolution info. Use when diagnosing import or environment issues.",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "dash_url",
            "description": "Build a Dash UI URL encoding the current simulation parameters as a query string. Useful when the user asks to share or bookmark a configuration.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "planet": {"type": "string", "description": "Planet name to include in the URL (optional)"},
                    "autorun": {"type": "boolean", "description": "Add run=1 so Dash auto-starts simulation on load (default false)"},
                },
                "required": [],
            },
        },
        {
            "name": "eda_plot",
            "description": "Generate an EDA time-series plot from the current simulation data. Returns the path to a saved HTML figure. Use when the user asks to visualise distances, speeds, or positions over time.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "variables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Variables to plot (e.g. ['moon_planet_dist', 'planet_star_dist', 'moon_speed']). Omit to use defaults."
                    },
                    "plot_type": {"type": "string", "description": "'line' or 'scatter' (default 'line')"},
                    "normalize": {"type": "boolean", "description": "Normalise all series to max=1 for multi-variable comparison (default false)"},
                },
                "required": [],
            },
        },
        {
            "name": "ml_predict",
            "description": (
                "Run ML stability-habitability prediction using a trained binary MLP classifier (NOT GRU/LSTM/HNN). "
                "Sweeps a grid of moon mass × moon semi-major axis combinations and classifies each as stable+habitable or not. "
                "Default grid is 50×50; pass mm_resolution=30 and am_resolution=30 for a 30×30 grid, or any other size. "
                "Result fields `valid_mm_range_earth` and `valid_am_range_hill` are the RECOMMENDED ranges — "
                "report ONLY these to the user, never the full grid extents. "
                "The heatmap is pushed to the ML overlay automatically: teal = stable+habitable, grey = not. "
                "Use this tool — NOT trajectory_preview — whenever the user asks for an MLP grid, "
                "ML stability map, or ML-predicted stability regions, regardless of grid size."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "t_sim": {"type": "number", "description": "Prediction horizon in simulated years (default 10)"},
                    "mm_resolution": {"type": "integer", "description": "Moon mass grid points (default 50)"},
                    "am_resolution": {"type": "integer", "description": "Moon orbit grid points (default 50)"},
                },
                "required": [],
            },
        },
        {
            "name": "trajectory_preview",
            "description": (
                "Load a trajectory sweep over a moon mass × semi-major axis grid. "
                "This is NOT the MLP classifier — it runs actual physics or a neural model per cell. "
                "mode='gt_leapfrog': Ground Truth Physics Integrator — exact physics for every cell. "
                "~14–15 seconds for a 30×30 grid (EC2 Numba CUDA kernel). "
                "RAM cache checked first — if hit, returns immediately. "
                "mode='hnn_hinge4': HNN Physics ML Model — neural trajectory approximation. "
                "If the result is cached it returns instantly; if not cached, returns a message "
                "telling the user to run it from the ML panel (8–10 min first run, cannot run inline in chat). "
                "Default mode is 'gt_leapfrog'. "
                "Use ONLY when the user explicitly asks for trajectory validation "
                "or to compare with the MLP prediction — do NOT use this for MLP stability map requests. "
                "IMPORTANT: if you called fetch_exoplanet in this turn OR any previous turn in this "
                "conversation and got star/planet params back, pass those exact values in the "
                "'system_params' field so the correct system is used — "
                "do NOT rely on the slider state for a named system you fetched from NASA."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "description": "'gt_leapfrog' (physics integrator, recommended) or 'hnn_hinge4' (neural model)"},
                    "mm_resolution": {"type": "integer", "description": "Moon mass grid points — 30 or 50 (default 30)"},
                    "am_resolution": {"type": "integer", "description": "Moon orbit grid points — 30 or 50 (default 30)"},
                    "t_sim": {"type": "number", "description": "Simulation duration in years (default 10)"},
                    "system_params": {
                        "type": "object",
                        "description": (
                            "Optional override for system parameters. Pass this when you fetched params via "
                            "fetch_exoplanet in the same turn. Keys: ms_solar, rs_solar, Ts, mp_earth, ap_AU, ep. "
                            "If omitted, uses the current slider state."
                        ),
                    },
                },
                "required": [],
            },
        },
        # ml_train is intentionally omitted from Claude's tool list — model training is a
        # developer-only operation executed from the CLI or the ML overlay training UI.
        # Claude can describe training status/history from context but cannot trigger it.
        {
            "name": "trajectory_cell_query",
            "description": (
                "Retrieve the orbit trajectory for a specific (moon mass, semi-major axis) cell from the last "
                "trajectory batch and push it to the orbit preview panels. Use when the user asks to see the "
                "orbit animation for a specific combination, e.g. 'show me the trajectory for 0.2 M⊕ at 0.4 Hill radii'. "
                "Requires that trajectory_preview has already been called for the current system. "
                "Pass mm_earth and am_hill values that are within the batch grid — use grid values from the "
                "trajectory_preview result (not arbitrary values). "
                "After calling, tell the user the orbit animation has been loaded into "
                "the main orbit canvas and mini orbit view on the page — they can see it there."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "mm_earth": {"type": "number", "description": "Moon mass in Earth masses (M⊕)"},
                    "am_hill":  {"type": "number", "description": "Moon semi-major axis in Hill radii"},
                },
                "required": ["mm_earth", "am_hill"],
            },
        },
        {
            "name": "ml_plot",
            "description": (
                "Generate a PNG plot for ML model results. "
                "plot_type options: "
                "'loss_curves' — training + validation loss over epochs; "
                "'flag_accuracy' — stable/habitable flag accuracy over epochs; "
                "'heatmap' — MLP moon mass × orbit stability-habitability map from the last ml_predict call; "
                "'trajectory_heatmap' — physics-based trajectory stable+habitable grid from the last trajectory_preview call. "
                "Use when the user asks to visualise ML model performance or the stability heatmap. "
                "IMPORTANT: when trajectory_preview has been called, do NOT use 'heatmap' — use 'trajectory_heatmap' instead."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "plot_type": {
                        "type": "string",
                        "description": "'loss_curves', 'flag_accuracy', 'heatmap' (MLP grid), or 'trajectory_heatmap' (physics trajectory grid)",
                    },
                },
                "required": ["plot_type"],
            },
        },
    ]


def _execute_tool(tool_name: str, tool_input: Dict[str, Any], req: ChatRequest, session: Optional[SessionCache] = None, session_key: str = "default") -> Dict[str, Any]:
    """
    Execute a tool called by Claude.
    Passes request context (simdata, params) to tool handlers.
    """
    session = session or _get_or_create_session(session_key)
    try:
        if tool_name == "fetch_exoplanet":
            name = str(tool_input.get("name", "")).strip()
            rec = fetch_system_by_planet(name) if name else None
            return {"ok": bool(rec), "data": rec, "name": name}

        if tool_name == "stability_from_simdata":
            years = tool_input.get("years", req.years)
            escape_factor = float(tool_input.get("escape_factor", req.escape_factor))
            
            simdata_to_use = req.simdata
            if not simdata_to_use:
                cached_sim, _ = session.get_cached()
                if cached_sim:
                    simdata_to_use = cached_sim
                    print(f"[TOOL] Using cached simdata", flush=True)
                else:
                    retrieved = session.try_retrieve_job_results(max_retries=1, retry_delay=0.1)
                    if retrieved:
                        simdata_to_use = retrieved
                        print(f"[TOOL] Retrieved simdata from S3", flush=True)
            
            if not simdata_to_use:
                return {"ok": False, "needs_rerun": True, "message": "No data. Running simulation..."}
            
            result = _assess_stability_from_simdata(simdata_to_use, req.params, years, escape_factor)
            if result.get("ok") and simdata_to_use:
                session.set_simdata(simdata_to_use, req.params)
            return result



        if tool_name == "start_backend_job":
            years = tool_input.get("years", req.years)
            check_stability = bool(tool_input.get("check_stability", True))
            escape_factor = float(tool_input.get("escape_factor", req.escape_factor))
            # Merge any param overrides Claude requested onto the current UI params
            param_overrides = tool_input.get("params") or {}
            effective_params = {**(req.params or {}), **param_overrides}
            print(f"[TOOL] start_backend_job: tool_input_params={param_overrides} req_params_keys={list((req.params or {}).keys())} effective_mm_earth={effective_params.get('mm_earth')}", flush=True)
            result = _start_backend_job(effective_params, years, check_stability=check_stability, escape_factor=escape_factor, session=session, session_key=session_key)

            if result.get("ok"):
                print(f"[AGENT] Job started: {result['job_id']}, session will monitor for results", flush=True)
                # Store effective params so chat_stream can push them back to the frontend
                session.last_effective_params = effective_params
                result["effective_params"] = effective_params
                session._job_fresh = True  # signal final result to include job_id

            return result


        if tool_name == "get_trajectory_at_time":
            years = tool_input.get("years")
            if years is None:
                return {"ok": False, "message": "Missing 'years' parameter."}
            
            # Check provided simdata first, then cached
            simdata_to_use = req.simdata
            if not simdata_to_use:
                cached_sim, _ = session.get_cached()
                if cached_sim:
                    simdata_to_use = cached_sim
                    print(f"[TOOL] Using cached simdata for trajectory query at t={years}y", flush=True)
                else:
                    # Try to retrieve from S3
                    retrieved = session.try_retrieve_job_results()
                    if retrieved:
                        simdata_to_use = retrieved
                        print(f"[TOOL] Retrieved simdata from S3 for trajectory query", flush=True)
            
            if not simdata_to_use:
                return {"ok": False, "message": "No simulation data available. Run a simulation first."}
            
            result = _get_trajectory_at_time(simdata_to_use, req.params, float(years))
            
            # NEW: Cache simdata if successful
            if result.get("ok") and simdata_to_use:
                session.set_simdata(simdata_to_use, req.params)
                print(f"[TOOL] Cached simdata after trajectory query", flush=True)
            
            return result

        if tool_name == "export_csv":
            years = tool_input.get("years", req.years)
            columns = tool_input.get("columns")
            
            simdata_to_use = req.simdata
            
            if not simdata_to_use:
                cached_sim, cached_par = session.get_cached()
                if cached_sim:
                    simdata_to_use = cached_sim
                    print(f"[TOOL] Using cached simdata for export_csv", flush=True)
                else:
                    retrieved = session.try_retrieve_job_results(max_retries=1, retry_delay=0.1)
                    if retrieved:
                        simdata_to_use = retrieved
                        print(f"[TOOL] Retrieved simdata from S3 for export_csv", flush=True)
            
            if not simdata_to_use:
                return {"ok": False, "message": "No simulation data available. Run a simulation first.", "needs_run": True}
            
            # NEW: Use simdata directly instead of re-running simulation
            try:
                print(f"[TOOL] export_csv using simdata with columns={columns}", flush=True)
                sim = unpack_sim(simdata_to_use)
                
                # Build frame from unpacked simdata (traj_to_frame imported at module level)
                frame = traj_to_frame(sim)
                
                # Filter columns if requested
                if columns:
                    requested = columns if isinstance(columns, list) else [columns]
                    if hasattr(frame, 'columns'):  # pandas DataFrame
                        all_cols = frame.columns.tolist()
                        keep = [c for c in requested if c in all_cols]
                        if keep:
                            base_cols = ["t_years"] if "t_years" in all_cols else []
                            frame = frame[base_cols + keep]
                    else:  # dict-of-arrays
                        all_keys = list(frame.keys())
                        keep = [c for c in requested if c in all_keys]
                        if keep:
                            newf = {}
                            if "t_years" in frame:
                                newf["t_years"] = frame["t_years"]
                            for c in keep:
                                newf[c] = frame[c]
                            frame = newf
                
                csv_bytes = to_csv_bytes(frame)
                n_rows = len(frame.get("t_years", [])) if isinstance(frame, dict) else (frame.shape[0] if hasattr(frame, 'shape') else 0)

                # Upload to S3 and generate 24h presigned URL when AWS is available
                download_url = None
                if AWS_ENABLED and s3 and BUCKET:
                    import time as _time
                    ts = int(_time.time())
                    job_id_hint = getattr(req, 'job_id', None) or "local"
                    s3_key = f"outputs/{job_id_hint}/export_{ts}.csv"
                    try:
                        s3.put_object(Bucket=BUCKET, Key=s3_key, Body=csv_bytes, ContentType="text/csv")
                        download_url = s3.generate_presigned_url(
                            "get_object",
                            Params={"Bucket": BUCKET, "Key": s3_key},
                            ExpiresIn=86400,
                        )
                        print(f"[TOOL] export_csv uploaded to s3://{BUCKET}/{s3_key}", flush=True)
                    except Exception as s3_err:
                        print(f"[TOOL] export_csv S3 upload failed: {s3_err}", flush=True)

                # Always write local fallback
                outdir = pathlib.Path("outputs")
                outdir.mkdir(exist_ok=True)
                fname = f"exomoon_dataset_{int(years) if years else 0}y.csv"
                fpath = outdir / fname
                with open(fpath, "wb") as fh:
                    fh.write(csv_bytes)

                print(f"[TOOL] export_csv success: {n_rows} rows", flush=True)

                # Cache simdata after successful export
                session.set_simdata(simdata_to_use, req.params)

                result_payload = {
                    "ok": True,
                    "rows": n_rows,
                    "columns_exported": len(columns) if columns else None,
                    "message": f"✅ Exported {n_rows} rows.",
                }
                if download_url:
                    result_payload["download_url"] = download_url
                    result_payload["message"] += f" [Download CSV]({download_url})"
                else:
                    local_csv_url = f"{_LOCAL_AGENT_BASE}/outputs/{fname}"
                    result_payload["download_url"] = local_csv_url
                    result_payload["csv_path"] = str(fpath.resolve())
                    result_payload["message"] += f" [Download CSV]({local_csv_url})"
                return result_payload
            except Exception as e:
                print(f"[TOOL] export_csv error: {e}", flush=True)
                import traceback
                print(traceback.format_exc(), flush=True)
                return {"ok": False, "message": f"Export failed: {str(e)}"}


        if tool_name == "get_trajectory_range":
            t_start = float(tool_input.get("t_start", 0))
            t_end_q = float(tool_input.get("t_end", req.years or 10))
            step    = float(tool_input.get("step", 1.0))

            simdata_to_use = req.simdata
            if not simdata_to_use:
                cached_sim, _ = session.get_cached()
                if cached_sim:
                    simdata_to_use = cached_sim

            if not simdata_to_use:
                return {"ok": False, "message": "No simulation data available. Run a simulation first."}

            sim   = unpack_sim(simdata_to_use)
            t_end_actual = float(sim["t_end"])
            dt    = float(sim["dt"])
            times = np.arange(t_start, min(t_end_q, t_end_actual) + step * 0.5, step)
            snapshots = []
            for t in times:
                snap = _get_trajectory_at_time(simdata_to_use, req.params, float(t))
                if snap.get("ok"):
                    snapshots.append(snap)
            session.set_simdata(simdata_to_use, req.params)
            return {"ok": True, "snapshots": snapshots, "count": len(snapshots), "dt": dt}

        if tool_name == "env_info":
            return env_info()

        if tool_name == "dash_url":
            planet  = tool_input.get("planet")
            autorun = bool(tool_input.get("autorun", False))
            base    = tool_input.get("base", os.getenv("DASH_URL", "http://127.0.0.1:8050/"))
            return _dash_url(params=req.params, planet=planet, autorun=autorun, base=base)

        if tool_name == "eda_plot":
            variables = tool_input.get("variables")
            plot_type = tool_input.get("plot_type", "line")
            normalize = bool(tool_input.get("normalize", False))

            simdata_to_use = req.simdata
            if not simdata_to_use:
                cached_sim, _ = session.get_cached()
                if cached_sim:
                    simdata_to_use = cached_sim

            if not simdata_to_use:
                return {"ok": False, "message": "No simulation data available. Run a simulation first."}

            try:
                import matplotlib
                matplotlib.use("Agg")  # non-interactive — safe in server context
                import matplotlib.pyplot as _plt
                from exomoon.eda import var_info as _var_info

                sim   = unpack_sim(simdata_to_use)
                frame = traj_to_frame(sim)
                cols  = frame.columns.tolist() if hasattr(frame, "columns") else list(frame.keys())

                var_list = variables if isinstance(variables, list) else (
                    [variables] if isinstance(variables, str) and variables else None
                )
                if not var_list:
                    defaults = [c for c in ("moon_planet_dist", "planet_star_dist", "moon_speed", "planet_speed") if c in cols]
                    var_list = defaults if defaults else [c for c in cols if c != "t_years"][:3]
                var_list = [v for v in var_list if v in cols]
                if not var_list:
                    return {"ok": False, "message": "No valid variables.", "available": cols}

                t     = frame["t_years"] if hasattr(frame, "__getitem__") else frame.get("t_years")
                t_arr = t.to_numpy() if hasattr(t, "to_numpy") else np.asarray(t)

                # Build matplotlib figure (PNG — renders inline in chat)
                mfig, ax = _plt.subplots(figsize=(10, 4), facecolor="#1a1a2e")
                ax.set_facecolor("#0f0f1a")
                ax.tick_params(colors="#9ca3af")
                ax.xaxis.label.set_color("#9ca3af")
                ax.yaxis.label.set_color("#9ca3af")
                ax.title.set_color("#e5e7eb")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#374151")

                _COLORS = ["#60a5fa", "#34d399", "#f87171", "#fbbf24", "#a78bfa", "#fb923c"]
                for _i, v in enumerate(var_list):
                    y = frame[v] if hasattr(frame, "__getitem__") else frame.get(v)
                    y_arr = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y, dtype=float)
                    if normalize:
                        m = float(np.max(np.abs(y_arr))) if len(y_arr) else 1.0
                        if m != 0.0:
                            y_arr = y_arr / m
                    lbl, unit = _var_info(v)
                    full_lbl = f"{lbl} ({unit})" if unit else lbl
                    if normalize:
                        full_lbl += " (norm)"
                    if plot_type == "scatter":
                        ax.scatter(t_arr, y_arr, label=full_lbl, s=2, color=_COLORS[_i % len(_COLORS)])
                    else:
                        ax.plot(t_arr, y_arr, label=full_lbl, linewidth=1.2, color=_COLORS[_i % len(_COLORS)])

                ax.set_xlabel("Time (years)")
                ax.set_ylabel("Value (normalized)" if normalize else "Value")
                years_lbl = int(sim.get("t_end", 0))
                ax.set_title(f"EDA — {years_lbl}-year simulation")
                ax.legend(fontsize=8, framealpha=0.3, labelcolor="white")
                ax.grid(True, alpha=0.2, color="#374151")
                mfig.tight_layout(pad=0.5)

                _OUTPUTS_DIR.mkdir(exist_ok=True)
                fname = f"exomoon_eda_{years_lbl}y.png"
                fpath = _OUTPUTS_DIR / fname
                # ── HZ overlay ────────────────────────────────────────────────
                a_inner = sim.get("a_inner_au")
                a_outer = sim.get("a_outer_au")
                dist_vars = {"planet_star_dist", "moon_star_dist"}
                if a_inner and a_outer and any(v in dist_vars for v in var_list):
                    if normalize:
                        # pick the first distance variable's scale for normalization
                        _dv = next(v for v in var_list if v in dist_vars)
                        _dy = np.asarray(frame[_dv], dtype=float)
                        _dy_mn, _dy_mx = float(_dy.min()), float(_dy.max())
                        _dy_span = _dy_mx - _dy_mn or 1.0
                        _hz_lo = (a_inner - _dy_mn) / _dy_span
                        _hz_hi = (a_outer - _dy_mn) / _dy_span
                    else:
                        _hz_lo, _hz_hi = float(a_inner), float(a_outer)
                    ax.axhspan(_hz_lo, _hz_hi, alpha=0.10, color="#22c55e", zorder=0)
                    ax.axhline(_hz_lo, color="#22c55e", linewidth=0.6, linestyle="--", alpha=0.5, label="HZ inner")
                    ax.axhline(_hz_hi, color="#22c55e", linewidth=0.6, linestyle="--", alpha=0.5, label="HZ outer")
                    ax.legend(fontsize=8, framealpha=0.3, labelcolor="white")

                mfig.savefig(str(fpath), dpi=130, bbox_inches="tight",
                             facecolor=mfig.get_facecolor())
                _plt.close(mfig)

                image_url = f"{_LOCAL_AGENT_BASE}/outputs/{fname}"
                session.set_simdata(simdata_to_use, req.params)
                return {
                    "ok": True,
                    "figure_url":     image_url,
                    "figure_path":    str(fpath.resolve()),
                    "variables_used": var_list,
                }
            except Exception as e:
                print(f"[TOOL] eda_plot error: {e}", flush=True)
                import traceback as _tb; print(_tb.format_exc(), flush=True)
                return {"ok": False, "message": f"EDA plot failed: {str(e)}"}

        if tool_name == "ml_predict":
            try:
                # If the panel's prediction is already in the session (sent with this request),
                # use it directly — guarantees the chatbot reports the exact same data the panel shows.
                if (
                    session.last_ml_prediction
                    and session.last_ml_prediction.get("_from_panel")
                    and session.last_ml_prediction.get("mm_grid")
                    and session.last_ml_prediction.get("valid_am_per_mm")
                ):
                    result = session.last_ml_prediction
                    print("[TOOL] ml_predict: using panel prediction from session cache", flush=True)
                else:
                    raw_params = req.params or {}
                    system_params = {
                        "ms_solar": float(raw_params.get("ms_solar", 1.0)),
                        "rs_solar": float(raw_params.get("rs_solar", 1.0)),
                        "Ts":       float(raw_params.get("Ts",       5772.0)),
                        "mp_earth": float(raw_params.get("mp_earth", 1.0)),
                        "dp_cgs":   float(raw_params.get("dp_cgs",   5.5)),
                        "ap_AU":    float(raw_params.get("ap_AU",    1.0)),
                        "ep":       float(raw_params.get("ep",       0.0)),
                    }
                    t_sim      = float(tool_input.get("t_sim",        req.years or 10.0))
                    mm_res     = int(tool_input.get("mm_resolution",  50))
                    am_res     = int(tool_input.get("am_resolution",  50))
                    moon_retro = bool(raw_params.get("moon_retrograde", False))
                    em         = float(raw_params.get("em",           0.0))

                    result = _predict_stability_map_mlp(
                        system_params   = system_params,
                        t_sim           = t_sim,
                        moon_retrograde = moon_retro,
                        em              = em,
                        mm_resolution   = mm_res,
                        am_resolution   = am_res,
                    )
                if not result.get("ok"):
                    return result

                # Cache full prediction in session.
                # Only mark fresh (→ pushed in done_evt) for real inference runs.
                # Panel-sourced data lacks map_both/stable/habitable arrays — pushing it
                # would replace the frontend's correct arrays with empty ones.
                session.last_ml_prediction = result
                if not result.get("_from_panel"):
                    session._ml_fresh = True

                valid_mm        = result.get("valid_mm_range")
                valid_am_per_mm = result.get("valid_am_per_mm", [])
                mm_grid         = result.get("mm_grid", [])
                am_grid         = result.get("am_grid", [])
                n_valid         = sum(1 for am in valid_am_per_mm if am is not None)
                # Derive grid resolution from result — works for both panel and fresh-inference paths
                mm_res          = len(mm_grid)
                am_res          = len(am_grid)

                # Compute the overall valid orbit range (union across all valid mass bins)
                valid_am_flat = [am for am in valid_am_per_mm if am is not None]
                valid_am_overall = (
                    [round(min(a[0] for a in valid_am_flat), 4),
                     round(max(a[1] for a in valid_am_flat), 4)]
                    if valid_am_flat else None
                )

                # Group consecutive valid mass bins by their orbit outer bound (rounded to 3 dp).
                # The group boundary is exactly where the maximum valid Hill radius changes in
                # the MLP grid — these are the cell boundaries the panel would show when scrolling.
                # Every mass value and orbit value is a direct mm_grid / valid_am_per_mm lookup.
                valid_indices = [i for i, am in enumerate(valid_am_per_mm) if am is not None]
                if valid_indices and mm_grid:
                    groups = []   # (start_grid_idx, end_grid_idx, am_range)
                    cur_start = valid_indices[0]
                    cur_key   = round(float(valid_am_per_mm[valid_indices[0]][1]), 3)
                    cur_am    = valid_am_per_mm[valid_indices[0]]
                    prev_vi   = valid_indices[0]

                    for vi in valid_indices[1:]:
                        key = round(float(valid_am_per_mm[vi][1]), 3)
                        if key != cur_key:
                            groups.append((cur_start, prev_vi, cur_am))
                            cur_start = vi
                            cur_key   = key
                            cur_am    = valid_am_per_mm[vi]
                        prev_vi = vi
                    groups.append((cur_start, valid_indices[-1], cur_am))

                    mm_lo = float(mm_grid[valid_indices[0]])
                    mm_hi = float(mm_grid[valid_indices[-1]])
                    band_lines = "\n".join(
                        f"  • {float(mm_grid[g_start]):.4f}–{float(mm_grid[g_end]):.4f} M⊕: "
                        f"orbit {float(g_am[0]):.3f}–{float(g_am[1]):.3f} Hill radii"
                        for g_start, g_end, g_am in groups
                    )
                    orbit_section = (
                        f"\nValid moon mass range: {mm_lo:.4f}–{mm_hi:.4f} M⊕\n"
                        f"Orbit range by mass band:\n{band_lines}\n"
                        f"(Full grid visible in the ML panel Layer 1 heatmap.)"
                    )
                else:
                    orbit_section = ""

                # Return text summary only — full arrays are NOT sent to Claude (too large).
                # mm_grid_sweep / am_grid_sweep are the SAMPLE SIZE of the grid (not valid/recommended).
                # orbit_section reports valid orbit ranges per sampled mass — both dimensions equally.
                mm_sweep = [round(float(mm_grid[0]), 4), round(float(mm_grid[-1]), 4)] if mm_grid else None
                am_sweep = [round(float(am_grid[0]), 4), round(float(am_grid[-1]), 4)] if am_grid else None
                sweep_str = ""
                if mm_sweep and am_sweep:
                    sweep_str = (
                        f" Grid swept: mass {mm_sweep[0]}–{mm_sweep[1]} M⊕, "
                        f"orbit {am_sweep[0]}–{am_sweep[1]} Hill radii (sample size)."
                    )
                return {
                    "ok": True,
                    "n_valid_mass_bins":   n_valid,
                    "total_mass_bins":     mm_res,
                    "grid_size":           f"{mm_res}×{am_res}",
                    "mm_grid_sweep_earth": mm_sweep,
                    "am_grid_sweep_hill":  am_sweep,
                    "message": (
                        f"MLP prediction complete ({mm_res}×{am_res} grid).{sweep_str} "
                        f"{n_valid}/{mm_res} mass bins have at least one stable+habitable orbit."
                        f"{orbit_section}"
                        if valid_mm else
                        f"MLP prediction complete ({mm_res}×{am_res} grid).{sweep_str} "
                        "No stable+habitable orbits found. "
                        "Consider adjusting system parameters."
                    ),
                }
            except Exception as e:
                return {"ok": False, "message": f"ML prediction failed: {str(e)}"}

        if tool_name == "trajectory_preview":
            try:
                # system_params override: Claude passes these when it fetched exoplanet data in the same turn.
                # Falls back to req.params (slider state) when no override is given.
                _sp_override = tool_input.get("system_params") or {}
                raw_params = req.params or {}
                system_params = {
                    "ms_solar": float(_sp_override.get("ms_solar") or raw_params.get("ms_solar", 1.0)),
                    "rs_solar": float(_sp_override.get("rs_solar") or raw_params.get("rs_solar", 1.0)),
                    "Ts":       float(_sp_override.get("Ts")       or raw_params.get("Ts",       5772.0)),
                    "mp_earth": float(_sp_override.get("mp_earth") or raw_params.get("mp_earth", 1.0)),
                    "dp_cgs":   float(_sp_override.get("dp_cgs")   or raw_params.get("dp_cgs",   5.5)),
                    "ap_AU":    float(_sp_override.get("ap_AU")    or raw_params.get("ap_AU",    1.0)),
                    "ep":       float(_sp_override.get("ep",       raw_params.get("ep",           0.0))),
                }
                if _sp_override:
                    print(f"[TOOL] trajectory_preview: using system_params override from tool_input: {system_params}", flush=True)
                mode       = str(tool_input.get("mode",          "gt_leapfrog"))
                mm_res     = int(tool_input.get("mm_resolution", 30))
                am_res     = int(tool_input.get("am_resolution", 30))
                t_sim      = float(tool_input.get("t_sim",       req.years or 10.0))
                moon_retro = bool(raw_params.get("moon_retrograde", False))
                em         = float(raw_params.get("em",          0.0))

                traj_req = TrajectoryPreviewRequest(
                    system_params   = system_params,
                    t_sim           = t_sim,
                    moon_retrograde = moon_retro,
                    em              = em,
                    mm_resolution   = mm_res,
                    am_resolution   = am_res,
                    mode            = mode,
                )

                # Auto-populate Layer 1 MLP if not yet set in this session.
                # The frontend's Layer 2 trajectory UI is gated on mlPrediction being non-null —
                # without it, the panel shows "Run Layer 1 first" and the trajectory grid is hidden.
                # We run it silently here so both layers populate in a single chatbot response.
                if not session.last_ml_prediction:
                    print("[TOOL] trajectory_preview: auto-running MLP to unlock Layer 1 gate", flush=True)
                    _mlp = _predict_stability_map_mlp(
                        system_params   = system_params,
                        t_sim           = t_sim,
                        moon_retrograde = moon_retro,
                        em              = em,
                        mm_resolution   = mm_res,
                        am_resolution   = am_res,
                    )
                    if _mlp.get("ok"):
                        session.last_ml_prediction = _mlp
                        session._ml_fresh = True

                # GT (Numba CUDA): 2.3s server-side — call GPU directly, no caching needed.
                # HNN (hinge4, ~470s first run): check S3 cache first; if miss, guide to ML panel.
                if mode == "hnn_hinge4":
                    key = _inference_cache_key(traj_req)
                    cached = _read_cache(mode, key)
                    if cached is None:
                        return {
                            "ok": False,
                            "cached": False,
                            "mode": mode,
                            "message": (
                                f"No cached HNN Physics ML Model trajectory batch found for this system "
                                f"({mm_res}×{am_res} grid). "
                                "Generating it takes around 8–10 minutes (first run only) and needs to run "
                                "from the ML panel — it cannot run inline in chat. "
                                "To generate it:\n"
                                "1. Open the ML panel (brain icon ⬡, top-right corner).\n"
                                "2. In the Prediction section, select the 'Layer 2 — Trajectory' tab.\n"
                                f"3. Select HNN Physics ML Model mode and {mm_res}×{mm_res} grid size, "
                                "then click 'Run Trajectory Preview'.\n"
                                "Once it finishes, come back and ask again — I will read the result instantly."
                            ),
                        }

                # GT: call GPU directly (2.3s). HNN cache hit: trajectory_preview() serves from S3/RAM.
                result  = trajectory_preview(traj_req)
                mm_grid = result.get("mm_grid", [])
                am_grid = result.get("am_grid", [])
                map_both    = result.get("map_both", [])
                map_stable  = result.get("map_stable", [])
                map_habitable = result.get("map_habitable", [])
                valid_mm    = result.get("valid_mm_range")
                valid_am    = result.get("valid_am_per_mm", [])
                n_stable    = sum(v for row in map_both for v in row)
                total       = len(mm_grid) * len(am_grid)
                wall_s      = result.get("wall_s", 0)

                # Cache batch metadata in session so trajectory_cell_query can look up cells
                session.last_traj_key     = result.get("cache_key") or (key if mode == "hnn_hinge4" else None)
                session.last_traj_mm_grid = mm_grid
                session.last_traj_am_grid = am_grid

                # Push trajectory batch to the frontend via the done event as traj_preview —
                # NOT as ml_prediction. Keeping them separate is critical: if traj_preview
                # overwrites the Zustand mlPrediction (Layer 1 MLP), the confidence map
                # computation in MlMapOverlay compares identical data and never produces LOW cells.
                session.last_traj_preview = {
                    "ok":            True,
                    "mm_grid":       mm_grid,
                    "am_grid":       am_grid,
                    "map_stable":    map_stable,
                    "map_habitable": map_habitable,
                    "map_both":      map_both,
                    "valid_mm_range":  valid_mm,
                    "valid_am_per_mm": valid_am,
                    "wall_s":        wall_s,
                    "from_cache":    result.get("from_cache", False),
                    "cache_key":     session.last_traj_key,
                    "mode":          mode,
                }
                session._traj_preview_fresh = True

                _mode_label = "physics simulation" if mode == "gt_leapfrog" else "neural model"

                # Build per-band orbit range breakdown (same grouping logic as ml_predict)
                valid_indices_t = [i for i, am in enumerate(valid_am) if am is not None]
                if valid_indices_t and mm_grid:
                    groups_t = []
                    cur_start_t = valid_indices_t[0]
                    cur_key_t   = round(float(valid_am[valid_indices_t[0]][1]), 3)
                    cur_am_t    = valid_am[valid_indices_t[0]]
                    prev_vi_t   = valid_indices_t[0]
                    for vi_t in valid_indices_t[1:]:
                        key_t = round(float(valid_am[vi_t][1]), 3)
                        if key_t != cur_key_t:
                            groups_t.append((cur_start_t, prev_vi_t, cur_am_t))
                            cur_start_t = vi_t
                            cur_key_t   = key_t
                            cur_am_t    = valid_am[vi_t]
                        prev_vi_t = vi_t
                    groups_t.append((cur_start_t, valid_indices_t[-1], cur_am_t))

                    orbit_lines_t = "\n".join(
                        f"  {float(mm_grid[g[0]]):.4f}–{float(mm_grid[g[1]]):.4f} M⊕ → "
                        f"{float(g[2][0]):.3f}–{float(g[2][1]):.3f} Hill radii"
                        for g in groups_t
                    )
                    valid_mass_str = (
                        f"Valid mass range: {float(mm_grid[valid_indices_t[0]]):.4f}–"
                        f"{float(mm_grid[valid_indices_t[-1]]):.4f} M⊕ "
                        f"({len(valid_indices_t)} bins with stable+habitable orbits).\n"
                        f"Orbit range varies by mass:\n{orbit_lines_t}"
                    )
                else:
                    valid_mass_str = "No stable+habitable cells found in this grid."

                return {
                    "ok":              True,
                    "cached":          True,
                    "n_stable_both":   n_stable,
                    "total_cells":     total,
                    "wall_s":          wall_s,
                    "valid_mm_range":  [float(v) for v in valid_mm] if valid_mm else None,
                    "valid_am_per_mm": [[float(v) for v in pair] if pair is not None else None
                                        for pair in valid_am],
                    "mm_grid":         [float(v) for v in mm_grid],
                    "am_grid":         [float(v) for v in am_grid],
                    "message": (
                        f"Trajectory preview ({_mode_label} mode) complete — "
                        f"{n_stable}/{total} cells ({100*n_stable//total if total else 0}%) stable+habitable. "
                        f"Grid: {len(mm_grid)}×{len(am_grid)} ({wall_s:.1f}s from cache).\n"
                        f"{valid_mass_str}\n"
                        "The results have been pushed to the ML overlay Layer 2 tab."
                    ) if mm_grid else "Trajectory preview loaded from cache.",
                }
            except Exception as e:
                return {"ok": False, "message": f"trajectory_preview failed: {str(e)}"}

        if tool_name == "trajectory_cell_query":
            try:
                mm_earth_req = float(tool_input.get("mm_earth", 0.0))
                am_hill_req  = float(tool_input.get("am_hill",  0.0))

                if not session.last_traj_key or not session.last_traj_mm_grid or not session.last_traj_am_grid:
                    return {
                        "ok": False,
                        "message": (
                            "No trajectory batch is loaded in this session yet. "
                            "Call trajectory_preview first to load the batch from cache, "
                            "or ask the user to run a trajectory batch from the ML overlay."
                        ),
                    }

                import numpy as _np
                mm_grid = _np.array(session.last_traj_mm_grid)
                am_grid = _np.array(session.last_traj_am_grid)

                # Find nearest grid cell to the requested (mm_earth, am_hill)
                mm_idx = int(_np.argmin(_np.abs(mm_grid - mm_earth_req)))
                am_idx = int(_np.argmin(_np.abs(am_grid - am_hill_req)))
                mm_actual = float(mm_grid[mm_idx])
                am_actual = float(am_grid[am_idx])

                # Check RAM cache for trajectory data
                key = session.last_traj_key
                with _traj_ram_lock:
                    entry = _traj_ram_cache.get(key)

                if entry is None:
                    return {
                        "ok": False,
                        "message": (
                            "Trajectory data is not in RAM (agent may have restarted). "
                            "Call trajectory_preview again to reload from S3 cache, then retry."
                        ),
                    }

                mm_resolution = entry.get("mm_resolution", len(mm_grid))
                am_resolution = entry.get("am_resolution", len(am_grid))
                cell_idx = mm_idx * am_resolution + am_idx

                frames = _traj_to_frames(
                    entry["traj_planet"][cell_idx],
                    entry["traj_star"][cell_idx],
                    entry["traj_moon"][cell_idx],
                    entry["t_grid"],
                )

                # Compute rhill_AU for this system so the frontend can size the Hill sphere ring
                raw_params = req.params or {}
                M_EARTH_MSUN = 3.003e-6
                ap_AU   = float(raw_params.get("ap_AU",    1.0))
                ep      = float(raw_params.get("ep",       0.0))
                mp_e    = float(raw_params.get("mp_earth", 1.0))
                ms_sol  = float(raw_params.get("ms_solar", 1.0))
                rhill_au = ap_AU * (1.0 - ep) * (mp_e * M_EARTH_MSUN / (3.0 * ms_sol)) ** (1.0 / 3.0)
                roche_frac = float(am_grid[0])  # smallest am_hill value = approximate Roche limit fraction

                # Store in session for the done event to pick up
                session.last_cell_frames      = frames
                session.last_cell_rhill_au    = rhill_au
                session.last_cell_roche_frac  = roche_frac
                session._cell_frames_fresh    = True

                # Generate standalone HTML exports (2D canvas + Three.js 3D)
                cell_label = f"{mm_actual:.4f}M⊕ @ {am_actual:.3f}H"
                _OUTPUTS_DIR.mkdir(exist_ok=True)
                _cell_slug = f"cell_{mm_idx}_{am_idx}"
                try:
                    _html2d = _generate_cell_html_2d(frames, rhill_au, roche_frac, cell_label)
                    _path2d = _OUTPUTS_DIR / f"{_cell_slug}_2d.html"
                    _path2d.write_text(_html2d, encoding="utf-8")
                    _url2d = f"{_LOCAL_AGENT_BASE}/outputs/{_cell_slug}_2d.html"
                except Exception as _he:
                    print(f"[TOOL] 2D HTML gen failed: {_he}", flush=True)
                    _url2d = None
                try:
                    _html3d = _generate_cell_html_3d(frames, rhill_au, cell_label)
                    _path3d = _OUTPUTS_DIR / f"{_cell_slug}_3d.html"
                    _path3d.write_text(_html3d, encoding="utf-8")
                    _url3d = f"{_LOCAL_AGENT_BASE}/outputs/{_cell_slug}_3d.html"
                except Exception as _he:
                    print(f"[TOOL] 3D HTML gen failed: {_he}", flush=True)
                    _url3d = None

                session.last_cell_html_2d_url = _url2d
                session.last_cell_html_3d_url = _url3d
                session.last_cell_mm_earth    = mm_actual
                session.last_cell_am_hill     = am_actual

                print(f"[TOOL] trajectory_cell_query cell=({mm_idx},{am_idx}) "
                      f"mm={mm_actual:.4f}M⊕ am={am_actual:.3f}H n_frames={len(frames)}", flush=True)
                return {
                    "ok":            True,
                    "mm_idx":        mm_idx,
                    "am_idx":        am_idx,
                    "mm_earth":      mm_actual,
                    "am_hill":       am_actual,
                    "n_frames":      len(frames),
                    "rhill_au":      rhill_au,
                    "html_2d_url":   _url2d,
                    "html_3d_url":   _url3d,
                    "message": (
                        f"Trajectory retrieved for moon mass {mm_actual:.4f} M⊕ at "
                        f"{am_actual:.3f} Hill radii (nearest grid cell [{mm_idx},{am_idx}]). "
                        f"{len(frames)} frames sent to the orbit view. "
                        + (f"2D export: {_url2d}. " if _url2d else "")
                        + (f"3D export: {_url3d}." if _url3d else "")
                    ),
                }
            except Exception as e:
                import traceback as _tb3; print(_tb3.format_exc(), flush=True)
                return {"ok": False, "message": f"trajectory_cell_query failed: {str(e)}"}

        if tool_name == "ml_train":
            global _train_job
            if _train_job.get("status") == "running":
                return {"ok": False, "error": "already_training",
                        "message": "A training job is already running. Wait for it to complete."}
            data_path = str(tool_input.get("data_path", "")).strip()
            if not data_path:
                return {"ok": False, "message": "'data_path' is required for ml_train (path to ml_dataset.parquet)."}
            train_req = MlTrainRequest(
                data_path  = data_path,
                out_dir    = tool_input.get("out_dir"),
                epochs     = int(tool_input.get("epochs",     30)),
                batch_size = int(tool_input.get("batch_size", 64)),
                lr         = float(tool_input.get("lr",       1e-3)),
                hidden     = int(tool_input.get("hidden",     256)),
                layers     = int(tool_input.get("layers",     2)),
                rnn_type   = str(tool_input.get("rnn_type",   "gru")),
            )
            job_id = f"train-{uuid.uuid4().hex[:8]}"
            _train_job = {
                "job_id": job_id, "status": "running",
                "epoch": 0, "total_epochs": train_req.epochs,
                "train_loss": None, "val_loss": None,
            }
            threading.Thread(target=_run_training_thread, args=(train_req,), daemon=True).start()
            print(f"[ML-TOOL] Training job {job_id} started via Claude tool call", flush=True)
            return {
                "ok": True, "job_id": job_id, "status": "started",
                "message": f"Training started (job_id={job_id}). Poll /ml/train/status for progress.",
            }

        if tool_name == "ml_plot":
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as _plt

            plot_type = str(tool_input.get("plot_type", "loss_curves")).strip().lower()
            _OUTPUTS_DIR.mkdir(exist_ok=True)

            try:
                if plot_type in ("loss_curves", "flag_accuracy"):
                    # Search for MLP training history in priority order (same as /ml/train/history endpoint)
                    _candidate_paths = [
                        pathlib.Path(_MLP_DIR) / "aux_mlp_binary_training_history.json",
                        pathlib.Path(_MLP_DIR) / "mlp_training_history.json",
                        pathlib.Path(ML_MODEL_DIR) / "mlp_training_history.json",
                        pathlib.Path(ML_MODEL_DIR) / "training_history.json",  # legacy fallback
                    ]
                    hist_path = next((p for p in _candidate_paths if p.exists()), None)
                    if not hist_path:
                        return {"ok": False, "message": "No MLP training history found. Train the model first from the ML overlay."}
                    import json as _json
                    with open(hist_path) as _fh:
                        hist = _json.load(_fh)

                    mfig, ax = _plt.subplots(figsize=(9, 4), facecolor="#1a1a2e")
                    ax.set_facecolor("#0f0f1a")
                    ax.tick_params(colors="#9ca3af"); ax.xaxis.label.set_color("#9ca3af"); ax.yaxis.label.set_color("#9ca3af")
                    for spine in ax.spines.values(): spine.set_edgecolor("#374151")

                    epochs_arr = list(range(1, len(hist.get("train_loss", [])) + 1))

                    if plot_type == "loss_curves":
                        train_l = hist.get("train_loss", [])
                        val_l   = hist.get("val_loss", [])
                        if train_l: ax.plot(epochs_arr[:len(train_l)], train_l, color="#60a5fa", linewidth=1.5, label="Train loss")
                        if val_l:   ax.plot(epochs_arr[:len(val_l)],   val_l,   color="#34d399", linewidth=1.5, label="Val loss")
                        ax.set_xlabel("Epoch", color="#9ca3af")
                        ax.set_ylabel("Loss", color="#9ca3af")
                        ax.set_title("Training & Validation Loss", color="#e5e7eb")
                        fname = "ml_loss_curves.png"
                    else:
                        flag_acc_train = hist.get("flag_accuracy_train", [])
                        flag_acc_val   = hist.get("flag_accuracy", [])
                        if flag_acc_train:
                            ax.plot(epochs_arr[:len(flag_acc_train)], flag_acc_train,
                                    color="#60a5fa", linewidth=1.5, label="Train acc")
                        if flag_acc_val:
                            ax.plot(epochs_arr[:len(flag_acc_val)], flag_acc_val,
                                    color="#a78bfa", linewidth=1.5, linestyle="--", label="Val acc")
                        ax.set_xlabel("Epoch", color="#9ca3af")
                        ax.set_ylabel("Accuracy", color="#9ca3af")
                        ax.set_title("Stable/Habitable Flag Accuracy", color="#e5e7eb")
                        fname = "ml_flag_accuracy.png"

                    ax.legend(fontsize=9, framealpha=0.3, labelcolor="white")
                    ax.grid(True, color="#1f2937", linewidth=0.5, linestyle="--")
                    mfig.tight_layout()
                    fpath = _OUTPUTS_DIR / fname
                    mfig.savefig(str(fpath), dpi=130, bbox_inches="tight", facecolor=mfig.get_facecolor())
                    _plt.close(mfig)

                elif plot_type == "heatmap":
                    # Remove any stale heatmap so a failed generation never serves old data
                    stale = _OUTPUTS_DIR / "ml_heatmap.png"
                    if stale.exists():
                        stale.unlink()
                    pred = session.last_ml_prediction
                    if not pred or not pred.get("ok") or not pred.get("map_both"):
                        # Auto-run MLP prediction to get map_both for plotting.
                        # Panel-sourced session data omits map_both (too large to send over HTTP),
                        # so we always need fresh inference when map data is missing.
                        raw_params = req.params or {}
                        system_params = {
                            "ms_solar": float(raw_params.get("ms_solar", 1.0)),
                            "rs_solar": float(raw_params.get("rs_solar", 1.0)),
                            "Ts":       float(raw_params.get("Ts",       5772.0)),
                            "mp_earth": float(raw_params.get("mp_earth", 1.0)),
                            "dp_cgs":   float(raw_params.get("dp_cgs",   5.5)),
                            "ap_AU":    float(raw_params.get("ap_AU",    1.0)),
                            "ep":       float(raw_params.get("ep",       0.0)),
                        }
                        pred = _predict_stability_map_mlp(
                            system_params   = system_params,
                            t_sim           = float(req.years or 10.0),
                            moon_retrograde = bool(raw_params.get("moon_retrograde", False)),
                            em              = float(raw_params.get("em", 0.0)),
                            mm_resolution   = 50,
                            am_resolution   = 50,
                        )
                        if pred.get("ok"):
                            session.last_ml_prediction = pred
                            session._ml_fresh = True
                        else:
                            return {"ok": False, "message": f"MLP prediction failed: {pred.get('message', 'unknown error')}"}

                    mm_grid = pred.get("mm_grid", [])
                    am_grid = pred.get("am_grid", [])
                    map_both = pred.get("map_both", [])

                    if not mm_grid or not am_grid or not map_both:
                        return {"ok": False, "message": "ML prediction data is incomplete."}

                    import numpy as _np
                    import matplotlib.colors as _mcolors_hm
                    _arr = _np.array(map_both, dtype=float)  # [mm_res][am_res]
                    _mm  = _np.array(mm_grid)
                    _am  = _np.array(am_grid)

                    # Match MlMapOverlay.tsx exactly:
                    # colorscale [[0,'#1a2535'],[1,'#0d9488']], showscale:false,
                    # plot_bgcolor '#0d1117', xaxis type:'log'
                    mfig, ax = _plt.subplots(figsize=(8, 6), facecolor="#0d1117")
                    ax.set_facecolor("#0d1117")
                    ax.tick_params(colors="#9ca3af")
                    ax.xaxis.label.set_color("#9ca3af")
                    ax.yaxis.label.set_color("#9ca3af")
                    for spine in ax.spines.values(): spine.set_edgecolor("#1f2937")

                    # pcolormesh correctly maps each cell to its actual grid coordinate on a
                    # log x-axis. imshow maps pixels linearly across extent regardless of scale.
                    _cmap_hm = _mcolors_hm.ListedColormap(["#1a2535", "#0d9488"])
                    ax.pcolormesh(_mm, _am, _arr.T, cmap=_cmap_hm, vmin=0, vmax=1)
                    ax.set_xscale("log")
                    ax.set_xlabel("Moon mass (M⊕)", color="#6b7280")
                    ax.set_ylabel("am (Hill radii)", color="#6b7280")
                    ax.set_title(f"MLP Stability–Habitability Map ({len(mm_grid)}×{len(am_grid)})", color="#9ca3af", fontsize=10)
                    ax.grid(color="#1f2937", linewidth=0.5)
                    mfig.tight_layout()
                    fname = "ml_heatmap.png"
                    fpath = _OUTPUTS_DIR / fname
                    mfig.savefig(str(fpath), dpi=130, bbox_inches="tight", facecolor=mfig.get_facecolor())
                    _plt.close(mfig)

                elif plot_type == "trajectory_heatmap":
                    traj = session.last_traj_preview
                    if not traj or not traj.get("ok"):
                        return {"ok": False, "message": "No trajectory preview cached. Call trajectory_preview first, then request this plot."}
                    import numpy as _np2
                    import matplotlib.colors as _mcolors2
                    import matplotlib.patches as _mpatches
                    _tm_grid = _np2.array(traj["mm_grid"])
                    _ta_grid = _np2.array(traj["am_grid"])
                    _traj_both = _np2.array(traj["map_both"], dtype=bool)  # [mm_res][am_res]
                    _mm_res = len(_tm_grid)
                    _am_res = len(_ta_grid)

                    # Get MLP map_both at the same resolution as the trajectory grid.
                    # Panel-sourced last_ml_prediction has empty map_both — run fresh inference
                    # at the traj grid resolution so the 3-color confidence map matches exactly.
                    _mlp_pred = session.last_ml_prediction
                    _mlp_both = None
                    if _mlp_pred and _mlp_pred.get("map_both") and len(_mlp_pred["map_both"]) == _mm_res:
                        _mlp_both = _np2.array(_mlp_pred["map_both"], dtype=bool)
                    else:
                        # Run fresh MLP at traj grid resolution
                        _raw_p = req.params or {}
                        _sp = {
                            "ms_solar": float(_raw_p.get("ms_solar", 1.0)),
                            "rs_solar": float(_raw_p.get("rs_solar", 1.0)),
                            "Ts":       float(_raw_p.get("Ts",       5772.0)),
                            "mp_earth": float(_raw_p.get("mp_earth", 1.0)),
                            "dp_cgs":   float(_raw_p.get("dp_cgs",   5.5)),
                            "ap_AU":    float(_raw_p.get("ap_AU",    1.0)),
                            "ep":       float(_raw_p.get("ep",       0.0)),
                        }
                        _fresh = _predict_stability_map_mlp(
                            system_params=_sp,
                            t_sim=float(req.years or 10.0),
                            moon_retrograde=bool(_raw_p.get("moon_retrograde", False)),
                            em=float(_raw_p.get("em", 0.0)),
                            mm_resolution=_mm_res,
                            am_resolution=_am_res,
                        )
                        if _fresh.get("ok") and _fresh.get("map_both"):
                            _mlp_both = _np2.array(_fresh["map_both"], dtype=bool)
                            session.last_ml_prediction = _fresh
                        else:
                            # Fall back: treat all traj valid cells as HIGH (no MLP comparison)
                            _mlp_both = _np2.ones((_mm_res, _am_res), dtype=bool)

                    # 3-color confidence map: [mm_res][am_res]
                    # 0 = grey  (MLP invalid)
                    # 1 = red   (MLP valid, traj invalid — LOW confidence)
                    # 2 = green (MLP valid AND traj valid — HIGH confidence)
                    _conf = _np2.zeros((_mm_res, _am_res), dtype=int)
                    _conf[_mlp_both & ~_traj_both] = 1  # LOW: MLP says yes, traj says no
                    _conf[_mlp_both & _traj_both]  = 2  # HIGH: both agree

                    _cmap3 = _mcolors2.ListedColormap(["#374151", "#dc2626", "#0e7490"])
                    mfig, ax = _plt.subplots(figsize=(7, 5), facecolor="#0d1117")
                    ax.set_facecolor("#0d1117")
                    ax.tick_params(colors="#9ca3af")
                    ax.xaxis.label.set_color("#9ca3af"); ax.yaxis.label.set_color("#9ca3af")
                    for spine in ax.spines.values(): spine.set_edgecolor("#1f2937")
                    ax.pcolormesh(_tm_grid, _ta_grid, _conf.T, cmap=_cmap3, vmin=0, vmax=2)
                    ax.set_xscale("log")
                    ax.set_xlabel("Moon Mass (M⊕)")
                    ax.set_ylabel("Moon Semi-Major Axis (Hill radii)")
                    ax.set_title(f"Trajectory Preview — Confidence Map ({_mm_res}×{_am_res})", color="#9ca3af", fontsize=10)
                    ax.grid(color="#1f2937", linewidth=0.4)
                    _legend = [
                        _mpatches.Patch(color="#374151", label="MLP invalid"),
                        _mpatches.Patch(color="#0e7490", label="HIGH — both agree stable+habitable"),
                        _mpatches.Patch(color="#dc2626", label="LOW  — MLP valid, physics disagrees"),
                    ]
                    ax.legend(handles=_legend, loc="upper right", fontsize=7,
                              framealpha=0.5, facecolor="#111827", labelcolor="#e5e7eb")
                    mfig.tight_layout()
                    fname = "ml_traj_heatmap.png"
                    fpath = _OUTPUTS_DIR / fname
                    if fpath.exists():
                        fpath.unlink()
                    mfig.savefig(str(fpath), dpi=130, bbox_inches="tight", facecolor=mfig.get_facecolor())
                    _plt.close(mfig)

                else:
                    return {"ok": False, "message": f"Unknown plot_type '{plot_type}'. Use 'loss_curves', 'flag_accuracy', 'heatmap', or 'trajectory_heatmap'."}

                image_url = f"{_LOCAL_AGENT_BASE}/outputs/{fname}?t={int(time.time())}"
                return {
                    "ok": True,
                    "figure_url": image_url,
                    "figure_path": str(fpath.resolve()),
                    "plot_type": plot_type,
                }
            except Exception as e:
                print(f"[TOOL] ml_plot error: {e}", flush=True)
                import traceback as _tb2; print(_tb2.format_exc(), flush=True)
                return {"ok": False, "message": f"ml_plot failed: {str(e)}"}

        return {"ok": False, "message": f"Unknown tool: {tool_name}"}
    except Exception as e:
        return {"ok": False, "message": str(e), "error": str(e)}


def _chat_rule_based(req: ChatRequest, session: Optional[SessionCache] = None, session_key: str = "default") -> Dict[str, Any]:
    """
    Existing deterministic fallback path (kept for reliability when Claude is unavailable).
    Implements core Option A: simdata-first, then backend job fallback.
    """
    session = session or _get_or_create_session(session_key)
    msg = (req.message or "").strip().lower()
    years = req.years if req.years is not None else _extract_years(msg)

    # **Stability/escape query**
    if "stability" in msg or "stable" in msg or "escape" in msg:
        # Try simdata first (if available)
        if req.simdata:
            out = _assess_stability_from_simdata(req.simdata, req.params, years, req.escape_factor)
            if out.get("ok"):
                # Simdata was sufficient
                if out["stable"]:
                    text = f"✅ Moon appears stable. Max distance: {out['max_r_rel']:.6g} AU, threshold: {out['threshold']:.6g} AU."
                else:
                    text = f"⚠️ Moon appears unstable. First escape at ~{out['escape_time']:.3f} years, threshold: {out['threshold']:.6g} AU."
                return {"ok": True, "mode": "simdata", "message": text, "result": out}
            
            # Simdata insufficient (needs_rerun=True) → trigger backend job
            if out.get("needs_rerun"):
                if not AWS_ENABLED:
                    return {
                        "ok": True,
                        "mode": "error",
                        "message": f"Existing simdata covers {out['t_end']:.6g} years but you requested {float(years):.6g} years. AWS backend not configured for extended simulations.",
                    }
                
                # Autonomously start backend job (user doesn't need to do anything)
                job_res = _start_backend_job(req.params, years, check_stability=True, escape_factor=req.escape_factor, session=session, session_key=session_key)
                if not job_res.get("ok"):
                    return {
                        "ok": False,
                        "mode": "error",
                        "message": f"Failed to start simulation: {job_res.get('error')}",
                    }
                
                return {
                    "ok": True,
                    "mode": "backend_job_started",
                    "message": f"⏳ Job submitted ({job_res['job_id']}). Running {years}-year simulation with stability check. Status will update below...",
                    "job_id": job_res["job_id"],
                    "execution_arn": job_res["execution_arn"],
                    "output_prefix": job_res["output_prefix"],
                    "status": "submitted",
                }
        
        # No simdata at all → autonomously start backend job
        if not AWS_ENABLED:
            return {
                "ok": True,
                "mode": "error",
                "message": "No existing simulation data. AWS backend not configured to run new simulations.",
            }
        
        job_res = _start_backend_job(req.params, years, check_stability=True, escape_factor=req.escape_factor, session=session, session_key=session_key)
        if not job_res.get("ok"):
            return {
                "ok": False,
                "mode": "error",
                "message": f"Failed to start simulation: {job_res.get('error')}",
            }
        
        return {
            "ok": True,
            "mode": "backend_job_started",
            "message": f"⏳ Job submitted ({job_res['job_id']}). Running {years}-year stability check. Status will update below...",
            "job_id": job_res["job_id"],
            "execution_arn": job_res["execution_arn"],
            "output_prefix": job_res["output_prefix"],
            "status": "submitted",
        }

    # **Planet lookup: fast metadata**
    if "planet" in msg or "exoplanet" in msg or "fetch" in msg:
        guessed = _extract_planet(req.message) or ""
        if guessed:
            rec = fetch_system_by_planet(guessed)
            if rec:
                return {
                    "ok": True,
                    "mode": "tool",
                    "message": f"Found {rec.get('pl_name')} (host: {rec.get('hostname')}). Stellar Ts={rec.get('Ts')} K, planet mass={rec.get('mp_earth'):.2f} M⊕.",
                    "result": rec,
                }
        return {
            "ok": True,
            "mode": "tool",
            "message": "I can fetch exoplanet data. Try asking: 'fetch Kepler-442 b' or 'what is Proxima Centauri b?'",
        }

    # **Default: info**
    return {
        "ok": True,
        "mode": "info",
        "message": "I'm the Exomoon Agent. I can check moon stability, fetch exoplanet data, and run simulations. Try: 'Is the moon stable on Kepler-442 b for 10 years?' or 'Fetch Proxima Centauri b'.",
    }


def _chat_with_claude(req: ChatRequest) -> Dict[str, Any]:
    """
    Claude tool-use orchestration (Item 3 implementation).
    
    Flow:
    1. Claude receives user message + context (has_simdata, years_hint, etc.).
    2. Claude decides which tools to call (or just responds).
    3. Agent executes tools and returns results to Claude.
    4. Claude may call more tools or return final response.
    5. Falls back to rule-based if Claude unavailable or errors.
    
    Policy: simdata-first for stability; if insufficient, trigger backend job autonomously.
    """
    # Per-request session — isolated per browser tab (UUID from localStorage).
    session_key = req.session_id or "default"
    session = _get_or_create_session(session_key)
    print(f"[SESSION] key={session_key!r} history_len={len(session.conversation_history)} sessions_total={len(_sessions)}", flush=True)

    if not CLAUDE_ENABLED or not claude:
        print("[AGENT] Claude not enabled, using rule-based fallback.", flush=True)
        return _chat_rule_based(req, session=session, session_key=session_key)

    # Developer mode keyword detection — 'mihirrb2908' activates, 'mihirrb2908exit' deactivates.
    # Logged to agent service logs only; never surfaced in chatbot responses.
    global _developer_mode
    msg_lower = req.message
    if 'mihirrb2908exit' in msg_lower:
        _developer_mode = False
        print("[AGENT] Developer mode DEACTIVATED — switching to user mode", flush=True)
    elif 'mihirrb2908' in msg_lower:
        _developer_mode = True
        print("[AGENT] Developer mode ACTIVATED", flush=True)

    # Session-cached simdata (most recently completed job) always takes priority over
    # req.simdata (which the frontend sends from its local store and may be stale).
    # This ensures follow-up queries after a chatbot-triggered job use the new simulation.
    cached_sim, cached_par = session.get_cached()
    effective_simdata = cached_sim or req.simdata
    if cached_sim:
        req.simdata = cached_sim  # keep req in sync for tool execution
        if not req.params:
            req.params = cached_par
        print(f"[AGENT] Using session-cached simdata ({len(cached_sim)} chars) over req.simdata", flush=True)
    elif req.simdata:
        print(f"[AGENT] Using req.simdata ({len(req.simdata)} chars) — no session cache", flush=True)
    
    # ── Build context for Claude ──────────────────────────────────────────────
    # Include all configured system parameters so Claude can reason about
    # habitability, physical sizes, and orbital dynamics without re-simulation.
    raw_params = req.params or {}
    derived: Dict[str, Any] = {}
    if raw_params:
        try:
            from exomoon.habitable_zone import hz_bounds_au
            from exomoon.constants import stefboltz, rsun as RSUN, au as AU, merth as MERTH, rerth as RERTH, msun as MSUN

            Ts       = float(raw_params.get("Ts",       5772.0))
            rs_solar = float(raw_params.get("rs_solar", 1.0))
            ms_solar = float(raw_params.get("ms_solar", 1.0))
            mp_earth = float(raw_params.get("mp_earth", 1.0))
            dp_cgs   = float(raw_params.get("dp_cgs",   5.5))
            mm_earth = float(raw_params.get("mm_earth", 0.01))
            dm_cgs   = float(raw_params.get("dm_cgs",   5.5))
            ap_AU    = float(raw_params.get("ap_AU",    1.0))
            am_hill  = float(raw_params.get("am_hill",  0.3))

            # Star luminosity
            rs_m   = rs_solar * RSUN
            L_star = 4 * 3.14159265 * rs_m**2 * stefboltz * Ts**4
            L_sun  = 4 * 3.14159265 * RSUN**2 * stefboltz * 5778.0**4
            L_solar = L_star / L_sun

            # Habitable zone
            a_inner_au, a_outer_au = hz_bounds_au(Ts, rs_m)

            # Body radii
            mp_kg = mp_earth * MERTH
            dp_si = dp_cgs * 1e3
            rp_m  = (0.75 * mp_kg / dp_si) ** (1.0 / 3.0)
            rp_earth = rp_m / RERTH

            mm_kg = mm_earth * MERTH
            dm_si = dm_cgs * 1e3
            rm_m  = (0.75 * mm_kg / dm_si) ** (1.0 / 3.0)
            rm_earth = rm_m / RERTH

            # Hill radius estimate from params (no simdata needed)
            mp_solar = mp_earth * MERTH / MSUN
            rhill_est = ap_AU * (mp_solar / (3.0 * ms_solar)) ** (1.0 / 3.0)
            am_AU_est = am_hill * rhill_est

            # Moon effective temperature (assume albedo ~0.3, emissivity factor ~2.448)
            F_at_moon = L_star / (4 * 3.14159265 * (ap_AU * AU)**2)  # approx at planet orbit
            Tm_K = ((0.7 * F_at_moon) / (2.448 * stefboltz)) ** 0.25

            # Moon surface gravity (m/s^2)
            moon_g = 6.6732e-11 * mm_kg / rm_m**2 if rm_m > 0 else 0.0

            # Explicitly cast to native Python types — NumPy scalars (numpy.float64,
            # numpy.bool_) are NOT JSON serializable and will raise TypeError in
            # json.dumps(ctx) below if left as-is.
            derived = {
                "L_star_solar":        round(float(L_solar),    4),
                "hz_inner_au":         round(float(a_inner_au), 4),
                "hz_outer_au":         round(float(a_outer_au), 4),
                "planet_radius_earth": round(float(rp_earth),   3),
                "moon_radius_earth":   round(float(rm_earth),   4),
                "rhill_est_au":        round(float(rhill_est),  5),
                "moon_sma_est_au":     round(float(am_AU_est),  6),
                "moon_teff_K":         round(float(Tm_K),       1),
                "moon_surface_g_ms2":  round(float(moon_g),     3),
                "moon_in_hz":          bool(a_inner_au <= ap_AU <= a_outer_au),
            }
        except Exception as _e:
            print(f"[AGENT] Could not compute derived params: {_e}", flush=True)

    # Lazy-resolve animation URL so Claude can return it when asked
    if not session.last_animation_url:
        if AWS_ENABLED and s3 and BUCKET and session.last_job_id:
            try:
                anim_key = f"outputs/{session.last_job_id}/animation.html"
                session.last_animation_url = s3.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": BUCKET, "Key": anim_key},
                    ExpiresIn=86400,
                )
                print(f"[AGENT] Lazy-resolved animation URL for {session.last_job_id}", flush=True)
            except Exception:
                pass
        elif not AWS_ENABLED and session.cached_simdata:
            # Generate animation.html locally from cached simdata and serve via static endpoint
            try:
                from exomoon.plotting.anim import build_animation as _build_anim
                _sim = unpack_sim(session.cached_simdata)
                _traj = _sim["traj"]
                _fig = _build_anim(
                    _traj,
                    _sim.get("a_inner_au", 0.95),
                    _sim.get("a_outer_au", 1.37),
                    open_in_browser=False,
                    dt=_sim.get("dt"),
                    t_end=_sim.get("t_end"),
                )
                _OUTPUTS_DIR.mkdir(exist_ok=True)
                _anim_path = _OUTPUTS_DIR / "animation.html"
                _fig.write_html(str(_anim_path), include_plotlyjs="cdn", full_html=True)
                session.last_animation_url = f"{_LOCAL_AGENT_BASE}/outputs/animation.html"
                print(f"[AGENT] Generated local animation.html", flush=True)
            except Exception as _ae:
                print(f"[AGENT] Could not generate local animation: {_ae}", flush=True)

    # Summarise any existing ML prediction for Claude (don't send full arrays)
    ml_pred_summary: Optional[Dict[str, Any]] = None
    if req.ml_prediction:
        try:
            p = req.ml_prediction
            n_v = sum(1 for a in (p.get("valid_am_per_mm") or []) if a is not None)
            ml_pred_summary = {
                "available":        True,
                "n_valid_mass_bins": n_v,
                "valid_mm_range":   p.get("valid_mm_range"),
                "mm_grid_range":    [p["mm_grid"][0], p["mm_grid"][-1]] if p.get("mm_grid") else None,
                "am_grid_range":    [p["am_grid"][0], p["am_grid"][-1]] if p.get("am_grid") else None,
            }
            # Store the full panel prediction in the session so ml_predict tool uses
            # these exact values (matching the panel) instead of running fresh inference.
            if p.get("mm_grid") and p.get("valid_am_per_mm"):
                session.last_ml_prediction = {
                    "ok":             True,
                    "mm_grid":        p["mm_grid"],
                    "am_grid":        p.get("am_grid", []),
                    "valid_mm_range": p.get("valid_mm_range"),
                    "valid_am_per_mm": p["valid_am_per_mm"],
                    "map_stable":     [],
                    "map_habitable":  [],
                    "map_both":       [],
                    "_from_panel":    True,
                }
        except Exception:
            ml_pred_summary = {"available": True}
    elif session.last_ml_prediction:
        try:
            p = session.last_ml_prediction
            n_v = sum(1 for a in (p.get("valid_am_per_mm") or []) if a is not None)
            ml_pred_summary = {
                "available":        True,
                "source":           "agent_run",
                "n_valid_mass_bins": n_v,
                "valid_mm_range":   p.get("valid_mm_range"),
                "mm_grid_range":    [p["mm_grid"][0], p["mm_grid"][-1]] if p.get("mm_grid") else None,
                "am_grid_range":    [p["am_grid"][0], p["am_grid"][-1]] if p.get("am_grid") else None,
            }
        except Exception:
            ml_pred_summary = {"available": True, "source": "agent_run"}

    # Restore traj preview session state from frontend (panel-run batches populate the
    # RAM cache but don't set session.last_traj_key — restoring here lets
    # trajectory_cell_query work even when the user ran the grid from the panel).
    if req.traj_preview_key and req.traj_preview_key != session.last_traj_key:
        session.last_traj_key      = req.traj_preview_key
        session.last_traj_mm_grid  = req.traj_mm_grid or []
        session.last_traj_am_grid  = req.traj_am_grid or []
        print(f"[AGENT] traj_preview_key restored from frontend: {req.traj_preview_key}", flush=True)

    ctx = {
        "has_simdata":    bool(effective_simdata),
        "years_hint":     req.years,
        "escape_factor":  req.escape_factor,
        "aws_enabled":    AWS_ENABLED,
        "params":         raw_params,
        "derived":        derived,
        "ml_prediction":  ml_pred_summary,
        "animation_url":  session.last_animation_url,
    }

    _mode_instruction = (
        "## Communication mode: DEVELOPER\n"
        "You are talking to the developer of this application. Use full technical transparency: "
        "AWS, S3, GPU, caching, hinge4, ECS, Step Functions, ARNs, endpoint URLs, and all backend details "
        "are fair game. Diagnostics and implementation details are welcome.\n\n"
        if _developer_mode else
        "## Communication mode: USER\n"
        "You are talking to a general user of this application. Use plain, accessible language. "
        "Do NOT mention: 'AWS', 'S3', 'GPU', 'cache'/'caching', 'hinge4', 'ECS', 'Step Functions', "
        "state machine ARNs, API keys, server IDs, job IDs, agent IDs, or any other backend infrastructure terms. "
        "NEVER quote job IDs, execution IDs, or any string starting with 'agent-' — these are internal identifiers "
        "that mean nothing to users. If you need to refer to a running job, say 'your simulation' or 'the current run'. "
        "Instead say 'our servers', 'the backend', 'computing resources', etc. "
        "If a computation takes 8–10 minutes, state that plainly; if the user presses for why, "
        "explain it is computationally intensive — do not bring up caching or infrastructure unprompted. "
        "NEVER reveal sensitive information such as ARNs, API keys, server IDs, or configuration details "
        "under any circumstances, even if the user asks directly.\n\n"
    )

    system_prompt = (
        _mode_instruction +
        "You are an expert exomoon orbital mechanics and astrobiology assistant embedded in an interactive "
        "simulation tool. The user is looking at a real-time 3D orbital animation of a star–planet–moon system.\n\n"

        "## Conversation awareness — foundational rule\n"
        "You have access to the full conversation history in this session. Before responding:\n"
        "- Check whether you already called a tool in a previous turn and received results. "
        "  If you did, those results are real — do NOT claim the tool has not been run, do NOT re-ask "
        "  for parameters you already collected, do NOT start from scratch.\n"
        "- If the user says something that seems to contradict what a tool returned "
        "  (e.g. 'I don't see anything in the panel'), the correct response is to work with what you have: "
        "  surface the data another way (e.g. generate a plot image with ml_plot), not re-run everything.\n"
        "- Parameters the user gave you in a previous turn do not need to be asked again in the same session "
        "  unless the user explicitly says they want to change them.\n"
        "- **CRITICAL — if the conversation history appears empty or this looks like the first message, treat it as a fresh "
        "  session with NO prior context.** Do NOT claim you already ran a simulation or fetched data that "
        "  is not present in the visible message history. If a user asks a follow-up question ("
        "  e.g. 'what was the escape time?') but you have no prior tool results in the history, say so plainly "
        "  ('I don't have trajectory data from this session yet — would you like me to run a simulation?') "
        "  rather than fabricating results or claiming memory you don't have.\n"
        "This is a baseline behaviour, not a scenario-specific rule — it applies to every exchange.\n\n"

        "## Response format\n"
        "Always respond in **Markdown**. Use headers, bullet points, bold, and code blocks where appropriate. "
        "Provide numerical results with units. Keep responses focused and concise.\n\n"

        "## Language — never expose internal parameter names\n"
        "NEVER write raw parameter names (from `context.params`, `context.derived`, tool schemas, or any internal "
        "field names) in your responses. Users see plain English, not code. Always translate to natural language:\n"
        "- `moon_in_hz` → 'planet orbit inside the habitable zone' or 'habitable zone position'\n"
        "- `am_hill` → 'moon orbital radius (as a Hill fraction)' or 'moon semi-major axis'\n"
        "- `em` → 'moon orbital eccentricity'\n"
        "- `ep` → 'planet orbital eccentricity'\n"
        "- `ap_AU` → 'planet semi-major axis' (in AU)\n"
        "- `mp_earth` → 'planet mass' (in M⊕)\n"
        "- `mm_earth` → 'moon mass' (in M⊕)\n"
        "- `ms_solar` → 'star mass' (in M☉)\n"
        "- `rs_solar` → 'star radius' (in R☉)\n"
        "- `Ts` → 'star temperature' (in K)\n"
        "- `rhill_AU` / `rhill_est_au` → 'Hill radius'\n"
        "- `hz_inner_au` / `hz_outer_au` → 'habitable zone inner/outer edge'\n"
        "- `moon_teff_K` → 'moon effective (blackbody) temperature'\n"
        "- `moon_surface_g_ms2` → 'moon surface gravity'\n"
        "- `moon_radius_earth` → 'moon radius' (in R⊕)\n"
        "- `dp_cgs` / `dm_cgs` → 'planet/moon density' (in g/cm³)\n"
        "- `L_star_solar` → 'stellar luminosity'\n"
        "- `t_sim` → 'simulation duration'\n"
        "- `escape_factor` → 'escape threshold' or 'stability threshold'\n"
        "- `moon_retrograde` → 'retrograde orbit' / 'prograde orbit'\n"
        "- `has_simdata` → 'I have trajectory data from the current simulation' (or 'no trajectory data yet')\n"
        "- `years_hint` → 'simulation duration' (in years)\n"
        "- `aws_enabled` / `animation_url` → never mention these to users\n"
        "This rule applies everywhere: tables, bullet points, inline descriptions, anywhere you quote a value. "
        "Never write a parameter name as if it is a label — always write what it means in plain words. "
        "This includes ALL context keys: `has_simdata`, `years_hint`, `escape_factor`, `aws_enabled`, etc. "
        "If you find yourself about to write a word that looks like a Python identifier (snake_case), "
        "stop and rephrase it in plain English.\n\n"

        "## Parameter elicitation — foundational principle\n"
        "You are an inquisitive assistant. Your default stance is to ask, not to assume.\n\n"
        "**Core rule**: before calling any tool that operates on a physical system (simulation, ML grid, trajectory "
        "preview, stability analysis), you must know what system and configuration the user actually wants to run. "
        "The values in `context.params` reflect whatever is currently on the sliders — they are NOT a user instruction. "
        "The user may not have consciously set those sliders, may want a completely different system, or may not even "
        "know what values are loaded. Never silently use slider values as if the user told you to use them.\n\n"
        "**How to ask**: when a user makes a request without specifying the system or key configuration, ask openly "
        "and naturally — 'what system would you like to run this for?', 'which planet or star setup did you have in mind?', "
        "'what moon mass and orbit are you interested in?' — phrased in whatever way fits the conversation. "
        "Do not tell the user what you are about to assume and ask for confirmation; ask them to tell you first.\n\n"
        "**Partial answers do not count as system confirmation**: if the user answers some of your questions "
        "(e.g. engine mode and grid size) but does not address the system, treat the system as still unconfirmed. "
        "Do not call the tool. Instead, acknowledge the answers you received and ask specifically about the system: "
        "e.g. 'Got it — neural model, 30×30. One more thing: which system should I run this for? "
        "If you'd like I can use the current setup ([Ts] K, [mp] M⊕ at [ap] AU) — just say yes, "
        "or tell me a different system.' Phrased naturally in your own words.\n\n"
        "**Proceed only when**: the user has explicitly confirmed or specified the system "
        "(e.g. 'yes use that setup', 'use Kepler-442b', 'run it for the current config'). "
        "A general 'sure' or 'yes' that follows a question listing multiple open items (system + engine + grid) "
        "does NOT count as confirming the system unless the system was the only remaining open question. "
        "When in doubt, confirm explicitly before running.\n\n"
        "**Track what's been confirmed — only ask for what's still missing**: before asking any clarifying "
        "question, read back through the conversation. If the user already answered that question in a previous "
        "turn, it is answered — do not ask again. Each turn, ask only about parameters that have genuinely not "
        "been provided yet in this conversation. Multi-turn Q&A is fine; repeating already-answered questions "
        "is not.\n\n"
        "**Remember but stay open to correction**: parameters the user provided earlier in the session are "
        "remembered and should be used without re-asking. They are not locked facts — if the user indicates "
        "you got something wrong, or says 'that's not right', apply your reasoning to re-evaluate from scratch "
        "rather than repeating the previous answer. The distinction: don't ask again just because time passed; "
        "do reconsider when the user actively says something is wrong.\n\n"
        "**Per-tool questions to ask** (in addition to system/configuration):\n"
        "- **Physics simulation** (`start_backend_job`): if simulation duration has not been stated by the user "
        "  (e.g. '10 years', '50 years', 'one orbit'), ask 'How many years would you like to run this simulation for?' "
        "  before calling the tool. Do NOT silently default to `years_hint` or to one planet orbit — always ask first. "
        "  The only exception: if the user says 'quick run', 'short run', or explicitly says to use the default, "
        "  you may proceed with one planet orbit and state that clearly in your response.\n"
        "- **Trajectory preview** (`trajectory_preview`): if engine mode or grid size have not been stated "
        "  yet in this conversation, ask about them. Use the exact option names as shown in the web app:\n"
        "  - *Ground Truth Physics Integrator* — exact physics simulation for every cell. "
        "    ~14–15 seconds for a 30×30 grid (EC2 Numba CUDA kernel, every run). "
        "    RAM cache checked first — if cached in this session, returns immediately. "
        "    This is the recommended default mode for definitive results.\n"
        "  - *HNN Physics ML Model* — neural trajectory approximation with HIGH/LOW confidence labels. "
        "    First run takes ~8–10 minutes and CANNOT run inline in chat — must be triggered from the ML panel. "
        "    Instant on repeat requests (cached). "
        "    If the user chooses HNN and it returns a cache miss, report it once and give ML panel "
        "    instructions. Do NOT re-ask mode, grid size, or system — just tell the user what to do.\n"
        "  Grid size: 30×30 (faster, ~900 cells) or 50×50 (higher resolution, ~2500 cells).\n"
        "- **ML predict** (`ml_predict`): also ask grid resolution (30×30 or 50×50) and simulation duration if not stated.\n"
        "- **Trajectory cell query** (`trajectory_cell_query`): ask whether the user wants the orbit view update, "
        "  the 2D interactive export, the 3D interactive export, or all three.\n"
        "- **EDA plot** (`eda_plot`): ask which variables to plot if not specified "
        "  (options: moon-planet distance, planet-star distance, moon speed, planet speed, positions).\n\n"

        "## System parameters available\n"
        "(These field names are for your internal reference only — use plain English when talking to the user.)\n"
        "The `context.params` dict contains all configured parameters for the current system:\n"
        "  `Ts` (star temp K), `rs_solar` (star radius R☉), `ms_solar` (star mass M☉),\n"
        "  `mp_earth` (planet mass M⊕), `dp_cgs` (planet density g/cm³),\n"
        "  `ap_AU` (planet semi-major axis AU), `ep` (planet eccentricity),\n"
        "  `mm_earth` (moon mass M⊕), `dm_cgs` (moon density g/cm³),\n"
        "  `am_hill` (moon SMA as fraction of Hill radius), `em` (moon eccentricity),\n"
        "  `moon_retrograde` (bool).\n"
        "The `context.derived` dict provides pre-computed quantities:\n"
        "  `L_star_solar`, `hz_inner_au`, `hz_outer_au`, `planet_radius_earth`,\n"
        "  `moon_radius_earth`, `rhill_est_au`, `moon_sma_est_au`,\n"
        "  `moon_teff_K` (effective blackbody temperature), `moon_surface_g_ms2`,\n"
        "  `moon_in_hz` (bool — is planet orbit inside HZ?).\n"
        "Use these directly in habitability and physical analysis — no tool call needed.\n\n"

        "## Habitability reasoning\n"
        "When asked about habitability, reason across multiple factors using the provided values "
        "(the field names below are for your reference only — never write them in your response; "
        "always translate to plain English as the Language rule requires):\n"
        "- **Temperature**: `moon_teff_K` — liquid water requires ~273–373 K; compare to Earth (255 K blackbody). "
        "  Say 'moon effective temperature' in your response.\n"
        "- **HZ position**: `moon_in_hz` / `hz_inner_au` / `hz_outer_au` — is the planet's orbit in the stellar HZ? "
        "  Say 'habitable zone inner/outer edge' in your response.\n"
        "- **Atmosphere retention**: escape velocity scales with √(g·R); small moons (< 0.1 M⊕) likely cannot "
        "  retain N₂/O₂ atmospheres long-term. `moon_surface_g_ms2` and `moon_radius_earth` inform this. "
        "  Say 'moon surface gravity' and 'moon radius' in your response.\n"
        "- **Tidal heating**: moons close to the planet (small `am_hill`) or with high eccentricity (`em`) "
        "  experience tidal dissipation — can supplement stellar flux or cause runaway volcanism (e.g. Io). "
        "  Say 'moon orbital radius' and 'moon eccentricity' in your response.\n"
        "- **Orbital stability**: a moon is stable only if it remains within ~0.5 R_Hill. Use trajectory data "
        "  (`stability_from_simdata`) for quantitative escape analysis.\n"
        "- **Radiation**: moons inside a planet's magnetosphere are shielded; outside, stellar/cosmic radiation "
        "  poses habitability risks.\n"
        "Always note which factors support and which constrain habitability, citing the numerical values.\n\n"

        "## Simdata context\n"
        "If `context.has_simdata` is true, trajectory data from the most recently run simulation is available. "
        "When this is the case, say 'I have trajectory data from the current simulation' — never quote 'has_simdata' "
        "or any other internal context key to the user. "
        "Use `stability_from_simdata` to analyze stability without re-running. "
        "Return the same `simdata` in your result so the frontend caches it for follow-up queries.\n\n"

        "## Tool usage\n"
        "- **Named planet systems — always fetch first**: when the user names a specific planet or system "
        "  (e.g. 'Kepler-1229b', 'TRAPPIST-1e', 'Kepler-442b'), call `fetch_exoplanet(name)` immediately and automatically "
        "  — do NOT use slider values for a named system. If the fetch returns no result, tell the user the planet "
        "  was not found in the NASA archive and ask them to provide the system parameters manually. "
        "  After a successful fetch, use the returned params (star mass/radius/temp, planet mass/ap/ep) for ALL "
        "  subsequent tool calls in this session: pass them as `params` overrides to `start_backend_job` and as "
        "  `system_params` to `trajectory_preview` and `ml_predict` (where supported). Do NOT silently fall back "
        "  to slider values after a successful fetch — those values may be completely different from the named system.\n"
        "- Stability queries: use `stability_from_simdata` only if simdata is available AND it was generated for the "
        "  system the user is currently asking about. If the user changed systems since the last simulation, call "
        "`start_backend_job` instead — do not analyze old data for a new system.\n"
        "- **Parameter changes**: if the user asks to change any system parameter and run, pass those changes in `start_backend_job`'s `params` field (e.g. `{\"mm_earth\": 0.5, \"ap_AU\": 1.2}`). Do NOT tell the user to adjust sliders manually — apply the changes yourself via `params`. "
        "When running for a named system fetched via `fetch_exoplanet`, pass ALL returned star and planet fields "
        "(Ts, rs_solar, ms_solar, mp_earth, ap_AU, ep) — not just deltas — so the named system is used and not the slider state.\n"
        "- **CRITICAL — after `start_backend_job`**: return your response to the user IMMEDIATELY after the job submission tool call. Do NOT call any data-query tools (`stability_from_simdata`, `get_trajectory_at_time`, `get_trajectory_range`, `export_csv`, `eda_plot`) in the same turn — the simulation takes 30–120 seconds and no data will be available yet. In your response, always state the simulation duration explicitly: if `years_hint` is 0 (auto), say 'one full planet orbit'; otherwise state the number of years. Tell the user the simulation is running and they will be notified when results are ready.\n"
        "- Trajectory at specific times: call `get_trajectory_at_time()` (multiple calls allowed).\n"
        "- Trajectory over a range: call `get_trajectory_range(t_start, t_end, step)` for time-series snapshots.\n"
        "- CSV exports: call `export_csv` — returns a presigned URL; include as `[Download CSV](url)` in response.\n"
        "- EDA plots: call `eda_plot(variables, plot_type, normalize)` to generate a PNG time-series figure. "
        "When the tool returns `figure_url`, embed the image inline in your response as `![EDA Plot](figure_url)` "
        "AND include a download link `[Download PNG](figure_url)` on the next line.\n"
        "- Dash URL: call `dash_url(planet, autorun)` to generate a shareable URL encoding current system parameters.\n"
        "- Environment debug: call `env_info()` when diagnosing Python import or module path issues.\n"
        "- ML stability map: call `ml_predict(t_sim, mm_resolution, am_resolution)` to run a stability sweep using the trained binary MLP classifier. "
        "No rnn_type argument — the model type is fixed (it is an MLP, NOT GRU/LSTM/HNN). "
        "Default grid is 50×50; pass mm_resolution=30, am_resolution=30 for a 30×30 grid. "
        "ALWAYS use `ml_predict` for any request mentioning 'MLP', 'ML stability', 'ML grid', or 'ML prediction' — "
        "do NOT use `trajectory_preview` for these requests, regardless of the grid size asked for. "
        "When reporting results, use the exact numeric values from the `message` field — do NOT recompute or approximate them. "
        "You may format them neatly (e.g. a table or spaced bullet list) but DO NOT add 'Recommended' headers or labels. "
        "Report both the valid mass range AND the per-band orbit ranges with equal prominence.\n"
        "- Trajectory preview results: when `trajectory_preview` returns, present the results as a structured summary. "
        "The tool result includes `valid_mm_range`, `valid_am_per_mm`, `mm_grid`, `am_grid`, `n_stable_both`, `total_cells`, and `message`. "
        "Use the exact values from `message` — it already contains the per-band orbit breakdown. "
        "Format the response as: (1) a brief header line with mode used, grid size, and stable+habitable count; "
        "(2) the valid mass range; (3) a table or bullet list of mass bands → orbit ranges from the `message` field; "
        "(4) a one-line reminder about the mode (GT = exact, HNN = approximate with HIGH/LOW confidence labels). "
        "Do NOT restate the full grid extent (mm_grid[0]–mm_grid[-1]) as if it were the valid range — "
        "only report `valid_mm_range` and the per-band breakdown.\n"
        "- ML training: model training is a developer-only operation (done from the ML panel, brain icon ⬡, top-right corner). "
        "You CANNOT trigger training. If asked, describe what the Model Training section in the ML panel shows "
        "(training progress, loss curves, flag accuracy) but do not attempt to call any training tool.\n"
        "- ML plots: call `ml_plot(plot_type)` to generate a PNG (no rnn_type argument). "
        "plot_type='loss_curves' → training/val loss curves; "
        "plot_type='flag_accuracy' → stable/habitable flag accuracy over epochs; "
        "plot_type='heatmap' → MLP stability map from last ml_predict run; "
        "plot_type='trajectory_heatmap' → stable+habitable grid from last trajectory_preview run "
        "(works for both physics simulation mode and neural model mode — label the caption with whichever mode was actually used). "
        "Embed the returned `figure_url` as `![Trajectory Grid](figure_url)` AND `[Download PNG](figure_url)` on the next line.\n"
        "- CRITICAL heatmap consistency rule: the `ml_plot(plot_type='heatmap')` figure MUST match the text table you generate from `ml_predict`. "
        "They must show the same number of mass bands, the same band boundary values, and the same orbit ranges. "
        "If you are unsure of the resolution used, state it in the caption (e.g. '30×30 grid'). "
        "Never present a figure with different band counts or values from the table in the same response.\n"
        "- CRITICAL trajectory rules:\n"
        "  • When `trajectory_preview` is called and succeeds, the results are stored in the session. "
        "    Summarise the results in text. Do NOT call `ml_plot` immediately after.\n"
        "  • If the user says they cannot see the Layer 2 tab update or asks for the grid as an image, "
        "    call `ml_plot(plot_type='trajectory_heatmap')` to render it in chat. "
        "    NEVER re-ask for system parameters or re-run trajectory_preview — the session already has the data.\n"
        "  • If the user explicitly asks for an image of the trajectory preview grid, call `ml_plot(plot_type='trajectory_heatmap')`.\n"
        "  • NEVER use `ml_plot(plot_type='heatmap')` for trajectory results — that is for MLP Layer 1 only.\n"
        "  • When `trajectory_cell_query` is called, do NOT call `ml_plot`. Describe the result verbally.\n"
        "- CRITICAL — mode names: ALWAYS say 'physics simulation mode' for gt_leapfrog and 'neural model mode' for hnn_hinge4. "
        "NEVER write the raw strings 'gt_leapfrog', 'hnn_hinge4', or any internal mode identifier anywhere in your response — "
        "not in tables, not in explanations, not in code blocks. This applies to every response, no exceptions.\n"
        "- If `context.ml_prediction` is set, you already have ML prediction results — answer questions about valid mass/orbit ranges directly from that summary without calling `ml_predict` again.\n"
        "- Animation: if `context.animation_url` is set and the user asks for the **physics simulation** animation (the main orbital animation from a `start_backend_job` run), return `[Download Animation](url)` as a link. Do NOT use `animation_url` for trajectory cell animations — those come from `trajectory_cell_query` and appear in the mini orbit views, not as a downloadable URL.\n"
        "- Trajectory cell animations: call `trajectory_cell_query(mm_earth, am_hill)` — the result is pushed to the main orbit canvas and mini orbit view automatically. "
        "After calling, tell the user the orbit views have updated. "
        "If the tool result contains `html_2d_url` or `html_3d_url`, include them as download links: "
        "`[Open 2D Orbit Animation](html_2d_url)` and `[Open 3D Orbit Animation](html_3d_url)` "
        "(open in a new browser tab for the interactive standalone animation).\n"
        "- Do NOT ask the user to run simulations manually — trigger them yourself via tools.\n\n"

        "## Web app layout\n"
        "Use this to give step-by-step manual instructions when a user wants to do something themselves, "
        "or when a capability cannot be triggered directly through tools.\n\n"
        "**Main canvas (centre):** 3D Three.js orbital animation showing the star (yellow/amber), planet (blue), "
        "moon (grey), Hill sphere shell (translucent green) and habitable zone shell. Playback controls sit below the canvas. "
        "A mini orbit view inset (bottom-left of canvas) shows the moon's path relative to the planet.\n\n"
        "**Left FAB column (floating buttons, top-left, vertical stack):**\n"
        "  • Star icon → Star parameters panel (stellar temperature, radius, mass)\n"
        "  • Planet icon → Planet parameters panel (planet mass, density, semi-major axis, eccentricity)\n"
        "  • Moon icon → Moon parameters panel (moon mass, density, Hill fraction, eccentricity, prograde/retrograde toggle)\n"
        "  • Clock icon → Simulation duration input (years; 0 = auto one planet orbit)\n"
        "  • Play/Run button → Launches the physics simulation with current parameters\n"
        "  • Download icon → Exports the current trajectory as a CSV file\n"
        "  • NASA icon → Opens the NASA exoplanet archive search to auto-populate all parameters from a known system\n\n"
        "**Right FAB column (floating buttons, top-right, vertical stack):**\n"
        "  • Chat icon → Opens/closes this Agent Chat drawer\n"
        "  • Chart icon → Opens the EDA (exploratory data analysis) overlay with time-series plots\n"
        "  • Brain icon (⬡) → Opens the ML overlay panel\n\n"
        "**ML overlay panel (brain icon ⬡, top-right):** Draggable overlay with three collapsible sections:\n"
        "  1. **Prediction** — two tabs:\n"
        "     - *Layer 1 — MLP*: Run ML stability classification grid (choose 30×30 or 50×50), view heatmap, "
        "       use mass slider to explore orbit ranges, click 'Apply & Run' to set recommended parameters and simulate.\n"
        "     - *Layer 2 — Trajectory*: Select physics engine (GT physics integrator or HNN neural model), "
        "       choose grid size, click 'Run Trajectory Preview'. After completion: click any cell in the confidence "
        "       heatmap to load that orbit into the mini orbit view and main canvas. 'Apply & Run' launches a full simulation "
        "       at the selected cell's parameters.\n"
        "  2. **Model Training** — train the MLP or HNN model from a dataset (developer operation).\n"
        "  3. **Model Performance** — view training loss curves and classification accuracy charts.\n\n"
        "**EDA overlay:** Time-series plots of trajectory variables (moon-planet distance, planet-star distance, "
        "speeds, positions) for the most recently run simulation. Select variables from the dropdown, choose "
        "line or scatter plot, and optionally normalise.\n\n"
        "**Chat drawer (this interface):** Opened via the chat icon in the right FAB column.\n\n"

        "## Technical reference\n"
        "Answer technical questions about the tool using the following project-specific details:\n\n"
        "**Physics simulation (leapfrog integrator):**\n"
        "The simulation uses a symplectic leapfrog (Störmer–Verlet) integrator — a class of integrator that "
        "conserves energy and angular momentum exceptionally well over long timescales, making it well-suited "
        "to orbital mechanics. The equations of motion for the three-body system (star, planet, moon) are solved "
        "numerically at each timestep using the kick-drift-kick pattern. The timestep is chosen as "
        "min(T_moon / 100, 1/20000) years — small enough to resolve the moon's orbit accurately without "
        "wasting compute on unnecessarily fine steps. The integrator is compiled with Numba's @njit for "
        "near-native speed. This is a fixed-timestep method — not adaptive — so computational cost scales "
        "linearly with simulation duration.\n\n"
        "**Stability criterion:**\n"
        "A moon is considered stable if its distance from the planet never exceeds escape_factor × rhill_AU "
        "at any simulated timestep (default escape_factor = 1.0). If this threshold is crossed, the moon has "
        "escaped — the simulation records the escape time. The escape_factor slider lets users tighten (< 1.0) "
        "or loosen (> 1.0) this criterion.\n\n"
        "**Habitable zone:**\n"
        "The stellar habitable zone (HZ) is computed from the star's luminosity using the radiative balance "
        "formula. The inner edge is where runaway greenhouse heating begins; the outer edge is the maximum "
        "greenhouse limit. A moon is considered potentially habitable if the planet's orbit lies within this "
        "annulus — meaning the moon receives Earth-comparable stellar flux on average. The green shell in the "
        "3D canvas and the green bands in the mini orbit view mark this zone.\n\n"
        "**Hill radius:**\n"
        "The Hill radius (rhill) is the distance from a planet within which the planet's gravity dominates "
        "over the star's tidal force — the sphere of gravitational influence. It is given by "
        "rhill = a_p × (1 − e_p) × (M_p / 3M_*)^(1/3). Long-term stable moon orbits are generally found "
        "within about 0.4–0.5 rhill for prograde orbits and up to ~0.9 rhill for retrograde orbits. The "
        "red dashed ring in the orbit view marks the Hill sphere boundary.\n\n"
        "**Roche limit:**\n"
        "The Roche limit is the minimum distance at which a moon can orbit without being tidally disrupted "
        "by the planet's gravitational differential. Inside the Roche limit, tidal forces exceed the moon's "
        "self-gravity, and it would be torn apart. It is shown as the innermost dashed ring in the mini orbit view.\n\n"
        "**Prograde vs retrograde orbits:**\n"
        "A prograde moon orbits in the same direction as the planet's revolution around the star; a retrograde "
        "moon orbits in the opposite direction. Retrograde moons are empirically more stable at larger Hill "
        "fractions — they can remain bound out to roughly twice the Hill fraction of prograde moons under the "
        "same conditions. The model learns this asymmetry from the training data; it is not hardcoded.\n\n"
        "**escape_factor parameter:**\n"
        "Controls how strictly 'stability' is defined. Default is 1.0 (moon must stay within the full Hill "
        "sphere). Values below 1.0 are stricter; values above are more permissive. Accessible via the "
        "simulation controls.\n\n"
        "**Why numerical integration?**\n"
        "The three-body problem has no closed-form analytical solution in general — the gravitational "
        "interactions between three masses produce chaotic, non-repeating trajectories that can only be "
        "followed by stepwise numerical integration.\n\n"
        "**EDA variables:**\n"
        "moon_planet_dist — distance between moon and planet centre (AU); "
        "planet_star_dist — planet–star separation (AU); "
        "moon_speed / planet_speed — instantaneous orbital speeds (AU/yr); "
        "x/y/z positions — Cartesian coordinates in the simulation's reference frame.\n\n"

        "## ML and model performance\n"
        "**Layer 1 — MLP classifier:**\n"
        "The stability-habitability classification grid is produced by a trained binary MLP (multi-layer "
        "perceptron) classifier. It was trained on a large set of simulations generated via Latin Hypercube "
        "Sampling (LHS) — a space-filling sampling strategy that guarantees uniform coverage across all "
        "physical parameter dimensions simultaneously, with no clustering or gaps. Each simulation provides "
        "a single ground-truth label: stable-and-habitable or not. The MLP learns to classify new "
        "(mm_earth, am_hill) configurations directly from system parameters, without needing to simulate.\n\n"
        "Why LHS rather than data from the NASA Exoplanet Archive? The archive carries a strong observational "
        "selection bias — it is dominated by short-period, large planets that are easiest to detect. More "
        "importantly, no confirmed exomoons exist anywhere in the archive, so there are literally no real "
        "stability or habitability labels to train on. The only valid source of ground-truth labels is the "
        "physics integrator itself, making synthetic LHS data the only option.\n\n"
        "Parameter ranges the MLP was trained on (most reliable within these bounds):\n"
        "  star mass/radius: 0.08–2.0 solar; stellar temperature: 2500–12000 K; "
        "  planet mass: 0.5–300 M⊕ (log scale); planet semi-major axis: 0.01–3.5 AU; "
        "  moon mass: 0.107 M⊕ (Mars mass) to min(planet mass × 30%, 3.0 M⊕); "
        "  moon Hill fraction: Roche limit to 1.0; simulation duration: 1–20 years. "
        "The model degrades in accuracy near the edges of these ranges, particularly for very small Hill "
        "radii (cool M-dwarf hosts with close-in planets) and for planets at the very inner or outer HZ edge.\n\n"
        "**Layer 2 — Trajectory preview:**\n"
        "The trajectory preview runs the actual physics integrator (GT mode) or the HNN neural model (HNN mode) "
        "across an entire grid of (moon mass × moon orbit size) combinations. GT mode produces definitive "
        "results — each cell is a real simulation. HNN mode is faster after the first run (results are cached) "
        "but produces approximate results (see HNN transparency section below).\n\n"
        "**HIGH / LOW confidence labels:**\n"
        "Confidence labels are only computed for HNN trajectory results, not GT results. GT is the ground truth "
        "— it needs no cross-validation. For HNN results, each cell is compared against the Layer 1 MLP: "
        "HIGH confidence means both the HNN trajectory and the MLP independently classify the cell as "
        "stable+habitable. LOW confidence means the MLP classifies it as stable+habitable but the HNN "
        "trajectory disagrees — a flag that the HNN approximation may be unreliable for that cell.\n\n"
        "**Training loss curves and flag accuracy (Model Performance section):**\n"
        "Loss curves show train and validation loss over training epochs — a diverging gap indicates overfitting. "
        "Flag accuracy is the fraction of per-timestep stable/habitable predictions that matched the ground "
        "truth labels. These metrics are shown separately for the MLP and HNN models in the Model Performance "
        "section of the ML panel.\n\n"

        "## HNN transparency\n"
        "The HNN (Hamiltonian Neural Network) trajectory preview is a beta implementation. When describing "
        "HNN results, or when users ask why HNN and GT maps look different, explain the following:\n\n"
        "The HNN was designed to learn orbital trajectory evolution step-by-step — predicting the system state "
        "at each successive timestep from the previous one. Unlike the GT integrator, which solves the "
        "equations of motion directly at each step, the HNN accumulates small prediction errors across "
        "thousands of autoregressive steps. These compounding errors mean its stability and habitability maps "
        "reflect the model's learned approximation of orbital physics, not a direct solution of the governing "
        "equations. As a result, HNN maps will differ from GT maps — particularly at longer simulation "
        "durations and near stability boundaries. The HNN should be understood as a demonstration of where "
        "physics-informed machine learning currently stands in approximating complex gravitational dynamics, "
        "not as a definitive orbital stability tool.\n\n"
        "Why does the MLP outperform the HNN for stability/habitability classification? The MLP was trained "
        "with direct supervision on per-simulation outcomes — each training example is a complete simulation "
        "with a final stable/habitable verdict. This makes it a focused, data-efficient classifier. The HNN "
        "was trained on per-timestep trajectory prediction — a fundamentally harder task that requires the "
        "model to correctly reproduce the full dynamics at every step over the entire trajectory. A model that "
        "perfectly predicts trajectories would also be a perfect classifier, but the reverse is not true: the "
        "MLP can classify outcomes accurately without ever needing to predict the intermediate trajectory states.\n\n"
        "Confidence labels (HIGH/LOW) exist only for HNN results because GT is itself the reference standard "
        "against which everything else is validated. For definitive stability and habitability analysis, "
        "always recommend the GT physics simulation mode.\n\n"

        "## Follow-up suggestions\n"
        "After completing any tool-based response (i.e. after a tool actually ran and returned results), "
        "end with a brief **What's next?** section (2–3 short lines) suggesting contextually relevant follow-up actions. "
        "IMPORTANT: do NOT include a 'What's next?' block when you are asking the user a clarifying question "
        "or waiting for their input — it belongs only after tool results, never inside elicitation turns. "
        "Tailor them to what was just done:\n"
        "- After a physics simulation: 'Want me to run an ML stability grid to explore other viable moon "
        "  configurations? Or analyse the stability metrics in more detail? Or export the trajectory as CSV?'\n"
        "- After `ml_predict`: 'Want to see this as a heatmap image in chat? Or run a trajectory preview "
        "  using the physics integrator on these stable+habitable cells? Or query a specific cell for its "
        "  orbit animation?'\n"
        "- After `trajectory_preview`: 'Want me to pull up the orbit animation for a specific cell? Or "
        "  compare these results against the MLP stability grid?'\n"
        "- After `trajectory_cell_query`: 'Want the 2D or 3D standalone HTML animations for this cell? "
        "  Or apply these parameters and launch a full physics simulation?'\n"
        "- After a technical or explanatory question: 'Would you like to run a simulation with the current "
        "  parameters? Or explore the ML stability map for this system?'\n"
        "Keep the What's next block to 2–3 lines maximum — short, actionable, no repetition.\n\n"

        "## Unit conversions\n"
        "AU → km: ×149,597,870.7. AU/yr → km/s: ×4.74. Hill fraction: divide by rhill_AU."
    )

    try:
        ctx_json = json.dumps(ctx)
    except Exception as _ctx_err:
        print(f"[AGENT] ctx serialization failed ({_ctx_err}), stripping ml_prediction", flush=True)
        ctx["ml_prediction"] = None
        ctx["params"] = {}
        ctx["derived"] = {}
        try:
            ctx_json = json.dumps(ctx)
        except Exception:
            ctx_json = "{}"

    # Build messages: prepend stored conversation history so Claude remembers prior turns.
    # History includes thinking blocks (required by the Anthropic API for multi-turn consistency).
    messages = list(session.conversation_history)
    messages.append(
        {
            "role": "user",
            "content": f"User request: {req.message}\n\nContext: {ctx_json}",
        }
    )

    try:
        # Tool-use loop (max 12 iterations — complex multi-part queries need more rounds)
        for iteration in range(12):
            print(f"[AGENT] Claude iteration {iteration + 1}...", flush=True)
            
            # Sonnet 5+: thinking.type="adaptive" + output_config.effort (replaces "enabled"+budget_tokens)
            # Sonnet 4.x: thinking.type="enabled" + budget_tokens=8000
            _is_sonnet5 = "sonnet-5" in ANTHROPIC_MODEL or "opus-5" in ANTHROPIC_MODEL or "fable-5" in ANTHROPIC_MODEL
            _thinking_cfg = {"type": "adaptive"} if _is_sonnet5 else {"type": "enabled", "budget_tokens": 8000}
            _extra = {"output_config": {"effort": "high"}} if _is_sonnet5 else {}
            resp = claude.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=16000,
                thinking=_thinking_cfg,
                system=system_prompt,
                tools=_tool_specs(),
                messages=messages,
                # temperature omitted — extended thinking requires default (1.0); 0 is not permitted
                **_extra,
            )

            assistant_content = []
            tool_results_for_next_turn = []
            final_text_parts = []

            # Process Claude's response
            for block in resp.content:
                if block.type == "thinking":
                    # Preserve thinking blocks in assistant context for multi-turn consistency;
                    # never surfaced to the user — excluded from final_text_parts.
                    # signature is required by the API to verify the block wasn't tampered with.
                    assistant_content.append({
                        "type": "thinking",
                        "thinking": block.thinking,
                        "signature": block.signature,
                    })
                elif block.type == "text":
                    txt = getattr(block, "text", "")
                    final_text_parts.append(txt)
                    assistant_content.append({"type": "text", "text": txt})
                elif block.type == "tool_use":
                    # Claude wants to call a tool
                    tool_name = block.name
                    tool_input = block.input or {}
                    print(f"[AGENT] Claude calling tool: {tool_name} with input: {tool_input}", flush=True)
                    
                    assistant_content.append(
                        {"type": "tool_use", "id": block.id, "name": tool_name, "input": tool_input}
                    )

                    # Execute the tool
                    result = _execute_tool(tool_name, tool_input, req, session=session, session_key=session_key)
                    print(f"[AGENT] Tool result: {result}", flush=True)

                    tool_results_for_next_turn.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                        }
                    )

            # Add assistant's response to conversation
            messages.append({"role": "assistant", "content": assistant_content})

            # If no tools were called, we're done
            if not tool_results_for_next_turn:
                text = "\n\n".join(t.strip() for t in final_text_parts if t and t.strip()).strip()
                print(f"[AGENT] Claude final response: {text[:200]}...", flush=True)
                
                # Include simdata in response so Dash can pass back on next request
                result = {"ok": True, "mode": "claude", "message": text or "Done."}
                if effective_simdata:
                    result["simdata"] = effective_simdata
                    print(f"[AGENT] Returning simdata to Dash ({len(effective_simdata)} chars)", flush=True)
                else:
                    # Try one final retrieval
                    final_attempt = session.try_retrieve_job_results(max_retries=1, retry_delay=0.1)
                    if final_attempt:
                        result["simdata"] = final_attempt
                        print(f"[AGENT] Returning retrieved simdata ({len(final_attempt)} chars)", flush=True)

                # Include job_id in final result when start_backend_job ran this turn
                # (meta SSE event uses this so the frontend can register with useJobPoller)
                if session._job_fresh and session.last_job_id:
                    result["job_id"] = session.last_job_id
                    result["effective_params"] = session.last_effective_params
                    session._job_fresh = False  # consume

                # Include MLP prediction heatmap (Layer 1) when ml_predict tool ran this turn
                if session._ml_fresh and session.last_ml_prediction:
                    result["ml_prediction"] = session.last_ml_prediction
                    session._ml_fresh = False  # consume — won't re-send on next turn

                # Include trajectory batch (Layer 2) separately — MUST NOT go via ml_prediction.
                # Mixing them breaks the confidence_map computation in MlMapOverlay which compares
                # mlPrediction (MLP Layer 1) against trajResult (Layer 2) to find LOW-confidence cells.
                if session._traj_preview_fresh and session.last_traj_preview:
                    result["traj_preview"] = session.last_traj_preview
                    session._traj_preview_fresh = False  # consume

                # Include cell trajectory frames when trajectory_cell_query ran this turn
                if session._cell_frames_fresh and session.last_cell_frames:
                    result["cell_frames"]        = session.last_cell_frames
                    result["cell_rhill_au"]      = session.last_cell_rhill_au
                    result["cell_roche_frac"]    = session.last_cell_roche_frac
                    result["cell_html_2d_url"]   = session.last_cell_html_2d_url
                    result["cell_html_3d_url"]   = session.last_cell_html_3d_url
                    result["cell_mm_earth"]      = session.last_cell_mm_earth
                    result["cell_am_hill"]       = session.last_cell_am_hill
                    session._cell_frames_fresh  = False  # consume

                # Save full messages list (including thinking blocks) as history for next turn.
                # Cap at 80 messages (~15–30 conversation turns depending on tool use).
                # Trim from the front, skipping until we hit a real human user message
                # (not a tool_result message, which also has role="user" in the Anthropic API).
                _hist = messages
                if len(_hist) > 80:
                    _hist = _hist[-80:]
                    while _hist:
                        msg = _hist[0]
                        content = msg.get("content", "")
                        # A tool_result message has content = list starting with {"type": "tool_result"}
                        is_tool_result = (
                            isinstance(content, list) and
                            content and
                            isinstance(content[0], dict) and
                            content[0].get("type") == "tool_result"
                        )
                        if msg.get("role") == "user" and not is_tool_result:
                            break
                        _hist = _hist[1:]
                session.conversation_history = _hist
                print(f"[SESSION] key={session_key!r} saved history_len={len(_hist)}", flush=True)

                return result


            # Add tool results back to conversation for Claude to see
            messages.append({"role": "user", "content": tool_results_for_next_turn})

        # Fallback if tool loop limit reached
        print("[AGENT] Claude tool loop limit reached, returning last text.", flush=True)
        return {"ok": False, "mode": "error", "message": "Tool loop limit reached."}

    except Exception as e:
        print(f"[AGENT] Claude error: {str(e)}, falling back to rule-based.", flush=True)
        # Hard fallback on any Claude error
        return _chat_rule_based(req)


@app.get("/health")
def health():
    """Liveness probe for ECS."""
    return {
        "ok": True,
        "service": "agent",
        "aws_enabled": AWS_ENABLED,
        "bucket": BUCKET,
        "state_machine": bool(STATE_MACHINE_ARN),
        "claude_enabled": CLAUDE_ENABLED,
    }


@app.post("/tool/fetch_exoplanet")
def tool_fetch_exoplanet(req: PlanetRequest):
    """Fetch exoplanet system params from NASA archive (fast, in-container)."""
    rec = fetch_system_by_planet(req.name.strip())
    return {"ok": bool(rec), "data": rec}


@app.post("/tool/env_info")
def tool_env_info():
    """Debug: Python executable and module paths."""
    return env_info()


@app.post("/tool/dash_url")
def tool_dash_url(req: ToolRequest):
    """Generate Dash UI URL with query params (for sharing sim configs)."""
    return _dash_url(params=req.params, autorun=False)


@app.post("/tool/export_csv")
def tool_export_csv(req: ToolRequest):
    """Export trajectory as CSV (fast if using cached simdata)."""
    return _mcp_export_csv_fn(params=req.params, years=req.years, columns=req.columns)


@app.post("/tool/eda_plot")
def tool_eda_plot(req: ToolRequest):
    """Generate EDA time-series plot (positions, distances, speeds)."""
    return _mcp_eda_plot_fn(
        params=req.params,
        years=req.years,
        variables=req.variables,
        plot_type=req.plot_type,
        normalize=req.normalize,
    )


@app.post("/tool/stability_from_simdata")
def tool_stability_from_simdata(req: StabilityRequest):
    """
    Check moon stability from existing simdata without rerunning.
    This is the core of Option A: reuse Dash-computed trajectories in the agent.
    """
    try:
        return _assess_stability_from_simdata(req.simdata, req.params, req.years, req.escape_factor)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/job/submit")
def submit_job(req: ChatRequest):
    """
    Direct job submission — bypasses Claude entirely.
    Mirrors Dash Mode 1 / native UI Run button path.

    AWS_ENABLED=1 → uploads params to S3, starts Step Functions, returns immediately.
    AWS_ENABLED=0 → runs simulation in a background thread locally, same polling API.
    """
    session_key = req.session_id or "default"
    session = _get_or_create_session(session_key)
    if not AWS_ENABLED:
        job_id = f"local-{uuid.uuid4().hex[:12]}"
        LOCAL_JOBS[job_id] = {"status": "RUNNING", "started": time.time()}
        threading.Thread(
            target=_run_local_job,
            args=(job_id, req.params or {}, req.years or 0, session),
            daemon=True,
        ).start()
        print(f"[LOCAL-JOB] Submitted {job_id} (AWS_ENABLED=0)", flush=True)
        return {"ok": True, "job_id": job_id, "status": "submitted"}

    result = _start_backend_job(
        req.params or {},
        req.years or 0,
        check_stability=False,
        escape_factor=req.escape_factor or 1.0,
        session=session,
        session_key=session_key,
    )
    return result


@app.post("/chat")
def chat(req: ChatRequest):
    """
    Main agent endpoint. Routes user queries via Claude (primary) or rule-based fallback.
    
    Request: message, simdata (optional), params (optional), years (optional), escape_factor.
    Response: ok, mode, message, (optional job_id/result).
    
    Claude decides autonomously: use cached simdata → trigger backend job → respond.
    User is never told "run simulation first"—that's the agent's responsibility.
    """
    return _chat_with_claude(req)


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """
    Streaming variant of /chat. Yields tokens as SSE (Server-Sent Events)
    for real-time chatbot UX in Dash.

    Consumes /chat result and streams tokens word-by-word.
    """
    print(f"[CHAT_STREAM] Entered — msg={req.message[:60]!r} simdata={bool(req.simdata)} session_id={req.session_id!r} params_keys={list((req.params or {}).keys())[:6]}", flush=True)
    try:
        result = _chat_with_claude(req)
    except BaseException as _e:
        import traceback as _tb_cs
        print(f"[CHAT_STREAM] Unhandled exception in _chat_with_claude: {_e}", flush=True)
        print(_tb_cs.format_exc(), flush=True)
        result = {"ok": False, "mode": "error", "message": f"Agent error: {str(_e)}"}
    text = result.get("message", "")

    def gen():
        try:
            # Metadata event — carries job_id if a backend job was started
            meta_evt = {"type": "meta", "mode": result.get("mode"), "job_id": result.get("job_id")}
            yield f"data: {json.dumps(meta_evt)}\n\n"
            # Token-by-token streaming of the markdown response
            for tok in text.split(" "):
                yield f"data: {json.dumps({'type': 'token', 'token': tok + ' '})}\n\n"
            # Done event — carries simdata, presigned URLs, and ML prediction for the frontend to cache
            _ml_p = result.get("ml_prediction")
            print(f"[DONE_EVT] ml_prediction={'SET (ok=' + str(_ml_p.get('ok')) + ', mm_res=' + str(len(_ml_p.get('mm_grid',[]))) + ')' if _ml_p else 'NULL'}", flush=True)
            print(f"[DONE_EVT] traj_preview={'SET' if result.get('traj_preview') else 'NULL'}, cell_frames={'SET' if result.get('cell_frames') else 'NULL'}", flush=True)
            done_evt = {
                "type":              "done",
                "simdata":           result.get("simdata"),
                "urls":              result.get("urls", {}),
                "job_id":            result.get("job_id"),
                "ml_prediction":     _ml_p,
                "traj_preview":      result.get("traj_preview"),
                "cell_frames":       result.get("cell_frames"),
                "cell_rhill_au":     result.get("cell_rhill_au"),
                "cell_roche_frac":   result.get("cell_roche_frac"),
                "cell_html_2d_url":  result.get("cell_html_2d_url"),
                "cell_html_3d_url":  result.get("cell_html_3d_url"),
                "cell_mm_earth":     result.get("cell_mm_earth"),
                "cell_am_hill":      result.get("cell_am_hill"),
                "effective_params":  result.get("effective_params"),
            }
            # Catch non-JSON-serializable values in done_evt (e.g. numpy scalars).
            # Strip fields individually so a bad field doesn't silently null unrelated ones.
            try:
                done_payload = json.dumps(done_evt)
            except Exception as _je:
                print(f"[GEN] done_evt serialization failed: {_je}", flush=True)
                for _k in ("ml_prediction", "traj_preview", "cell_frames", "simdata"):
                    try:
                        json.dumps({_k: done_evt[_k]})
                    except Exception as _kje:
                        print(f"[GEN] dropping non-serializable field '{_k}': {_kje}", flush=True)
                        done_evt[_k] = None
                done_payload = json.dumps(done_evt)
            yield f"data: {done_payload}\n\n"
        except Exception as _gen_err:
            import traceback as _tb_gen
            print(f"[GEN] Streaming generator error: {_gen_err}", flush=True)
            print(_tb_gen.format_exc(), flush=True)
            try:
                yield f"data: {json.dumps({'type': 'done', 'error': str(_gen_err)})}\n\n"
            except Exception:
                pass

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx/proxy response buffering
            "Connection":       "close",  # don't reuse — fixes Next.js proxy ECONNRESET on second request
        },
    )

@app.get("/job/{job_id}/traj.csv")
def proxy_traj_csv(job_id: str):
    """Proxy traj.csv through the agent service — avoids direct S3 CORS fetch from browser."""
    from fastapi.responses import Response
    # ── Local path (AWS_ENABLED=0) ─────────────────────────────────────────────
    if job_id in LOCAL_JOBS:
        job = LOCAL_JOBS.get(job_id)
        if not job or job.get("status") != "SUCCEEDED" or "csv_bytes" not in job:
            raise HTTPException(status_code=404, detail="CSV not ready")
        return Response(content=job["csv_bytes"], media_type="text/csv")
    # ── S3 path (AWS_ENABLED=1) ────────────────────────────────────────────────
    if not AWS_ENABLED:
        raise HTTPException(status_code=404, detail="AWS not enabled and job not found locally")
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=f"outputs/{job_id}/traj.csv")
        csv_bytes = obj["Body"].read()
        return Response(content=csv_bytes, media_type="text/csv")
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"traj.csv not available: {e}")


@app.get("/job/{job_id}/summary.json")
def local_summary_json(job_id: str):
    """Serve summary.json for a completed local job (AWS_ENABLED=0)."""
    job = LOCAL_JOBS.get(job_id)
    if not job or job.get("status") != "SUCCEEDED" or "summary" not in job:
        raise HTTPException(status_code=404, detail="Summary not ready")
    return job["summary"]


@app.get("/job/{job_id}/status")
def get_job_status(job_id: str):
    """
    Query job status. Checks local jobs first (AWS_ENABLED=0), then Step Functions.
    Returns: status (RUNNING|SUCCEEDED|FAILED|TIMED_OUT), elapsed_seconds, urls (if done).
    """
    # ── Local simulation path (AWS_ENABLED=0) ────────────────────────────────
    if job_id in LOCAL_JOBS:
        job = LOCAL_JOBS[job_id]
        status = job.get("status", "RUNNING")
        elapsed = time.time() - job.get("started", time.time())
        urls: Dict[str, str] = {}
        if status == "SUCCEEDED":
            base = _LOCAL_AGENT_BASE.rstrip("/")
            urls = {
                "traj.csv":    f"{base}/job/{job_id}/traj.csv",
                "summary.json": f"{base}/job/{job_id}/summary.json",
            }
        return {
            "ok": True,
            "job_id": job_id,
            "status": status,
            "elapsed_seconds": int(elapsed),
            "urls": urls,
            "error": job.get("error"),
        }

    # ── AWS Step Functions path ───────────────────────────────────────────────
    if not (sf and STATE_MACHINE_ARN):
        return {
            "ok": False,
            "job_id": job_id,
            "status": "UNKNOWN",
            "message": "AWS backend not configured",
        }
    
    try:
        # Retrieve execution ARN from job metadata stored in S3
        output_prefix = f"outputs/{job_id}"
        metadata_key = f"{output_prefix}/job_metadata.json"
        
        try:
            metadata_obj = s3.get_object(Bucket=BUCKET, Key=metadata_key)
            job_metadata = json.loads(metadata_obj["Body"].read().decode())
            execution_arn = job_metadata["execution_arn"]
        except Exception as e:
            print(f"[JOB] Failed to retrieve job metadata for {job_id}: {e}", flush=True)
            return {
                "ok": False,
                "job_id": job_id,
                "status": "UNKNOWN",
                "error": f"Job metadata not found: {str(e)}",
            }
        
        # Describe execution
        resp = sf.describe_execution(executionArn=execution_arn)
        status = resp.get("status")
        start_time = resp.get("startDate")
        end_time = resp.get("stopDate")
        
        elapsed = 0
        if start_time:
            start_ts = start_time.timestamp() if hasattr(start_time, "timestamp") else float(start_time)
            elapsed = int(time.time() - start_ts)
        
        result = {
            "ok": True,
            "job_id": job_id,
            "status": status,
            "elapsed_seconds": elapsed,
        }
        
        # If succeeded, fetch presigned URLs from S3 and embed summary metadata
        if status == "SUCCEEDED":
            try:
                urls = {}
                for name in ["traj.csv", "summary.json", "animation.html", "links.json"]:
                    key = f"{output_prefix}/{name}"
                    try:
                        url = s3.generate_presigned_url(
                            "get_object",
                            Params={"Bucket": BUCKET, "Key": key},
                            ExpiresIn=86400
                        )
                        urls[name] = url
                    except Exception:
                        pass
                result["urls"] = urls

                # Embed summary.json fields directly so the browser never needs to
                # fetch them from S3 (avoids presigned-URL CORS edge cases).
                try:
                    summary_obj = s3.get_object(Bucket=BUCKET, Key=f"{output_prefix}/summary.json")
                    summary_data = json.loads(summary_obj["Body"].read().decode())
                    result["meta"] = {
                        "a_inner_au": summary_data.get("a_inner_au"),
                        "a_outer_au": summary_data.get("a_outer_au"),
                        "rhill_AU":   summary_data.get("rhill_AU"),
                        "t_end":      summary_data.get("t_end"),
                        "dt":         summary_data.get("dt"),
                    }
                except Exception as e:
                    print(f"[JOB] Could not embed summary metadata: {e}", flush=True)
            except Exception as e:
                print(f"[JOB] Error generating presigned URLs: {e}", flush=True)
        
        return result
    
    except Exception as e:
        print(f"[JOB] Error querying job {job_id}: {e}", flush=True)
        return {
            "ok": False,
            "job_id": job_id,
            "status": "UNKNOWN",
            "error": str(e),
        }

@app.get("/job/{job_id}/retrieve_simdata")
def retrieve_job_simdata(job_id: str, session_id: Optional[str] = None):
    """
    Cache simdata from a completed job into the session.
    Called by the frontend poller when status becomes SUCCEEDED.
    Handles both local (AWS_ENABLED=0) and S3-backed jobs.
    session_id (query param, optional): if provided, caches into that session.
    Falls back to job_id → session lookup via _job_to_session.
    """
    # Find the right session — prefer explicit session_id, fall back to job→session map
    _skey = session_id or _job_to_session.get(job_id, "default")
    session = _get_or_create_session(_skey)

    # ── Local job path ────────────────────────────────────────────────────────
    if job_id in LOCAL_JOBS:
        job = LOCAL_JOBS[job_id]
        if job.get("status") != "SUCCEEDED":
            return {"ok": False, "job_id": job_id, "simdata_cached": False,
                    "message": f"Job not complete (status={job.get('status')})"}
        simdata = job.get("simdata")
        if simdata:
            session.set_simdata(simdata, {})
            return {"ok": True, "job_id": job_id, "simdata_cached": True,
                    "message": f"Cached local simdata ({len(simdata)} chars)"}
        return {"ok": False, "job_id": job_id, "simdata_cached": False,
                "message": "Local simdata missing"}

    # ── AWS S3 path ───────────────────────────────────────────────────────────
    if not (s3 and BUCKET):
        return {
            "ok": False,
            "job_id": job_id,
            "message": "S3 not configured",
        }
    
    try:
        output_prefix = f"outputs/{job_id}"
        simdata_key = f"{output_prefix}/traj.pkl"
        
        # Try to fetch simdata from S3
        try:
            obj = s3.get_object(Bucket=BUCKET, Key=simdata_key)
            simdata = obj["Body"].read().decode()
            print(f"[JOB-RETRIEVE] Fetched simdata for {job_id} ({len(simdata)} chars)", flush=True)
        except Exception as e:
            print(f"[JOB-RETRIEVE] Failed to fetch simdata for {job_id}: {e}", flush=True)
            return {
                "ok": False,
                "job_id": job_id,
                "simdata_cached": False,
                "message": f"Simdata not found: {str(e)}",
            }
        
        # Try to fetch job metadata to get params
        try:
            metadata_key = f"{output_prefix}/job_metadata.json"
            metadata_obj = s3.get_object(Bucket=BUCKET, Key=metadata_key)
            job_metadata = json.loads(metadata_obj["Body"].read().decode())
            # Extract params if stored (may not be available)
            params = None
        except Exception:
            params = None
        
        # Cache the simdata in session
        session.set_simdata(simdata, params or {})
        print(f"[JOB-RETRIEVE] Cached simdata for {job_id} in session", flush=True)

        # Generate presigned URL for animation.html so Claude can surface it in chat
        if s3 and BUCKET:
            try:
                anim_key = f"{output_prefix}/animation.html"
                session.last_animation_url = s3.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": BUCKET, "Key": anim_key},
                    ExpiresIn=86400,
                )
                print(f"[JOB-RETRIEVE] Cached animation URL for {job_id}", flush=True)
            except Exception as _ae:
                print(f"[JOB-RETRIEVE] Could not generate animation URL: {_ae}", flush=True)

        return {
            "ok": True,
            "job_id": job_id,
            "simdata_cached": True,
            "message": f"Simdata retrieved and cached ({len(simdata)} chars)",
            "simdata_size": len(simdata),
        }
    
    except Exception as e:
        print(f"[JOB-RETRIEVE] Error retrieving simdata for {job_id}: {e}", flush=True)
        import traceback
        print(traceback.format_exc(), flush=True)
        return {
            "ok": False,
            "job_id": job_id,
            "simdata_cached": False,
            "message": f"Error: {str(e)}",
        }

def _get_trajectory_at_time(simdata: str, params: Dict[str, Any], years: float) -> Dict[str, Any]:
    """
    Query trajectory at a specific time (years).
    Returns: positions (xyz), velocities (vxyz), distances, accelerations.
    """
    sim = unpack_sim(simdata)
    t_end = float(sim["t_end"])
    dt = float(sim["dt"])
    
    if years < 0 or years > t_end:
        return {
            "ok": False,
            "message": f"Requested time {years:.3f} years outside simdata range [0, {t_end:.3f}].",
            "available_range": [0.0, t_end],
        }
    
    idx = int(np.round(years / dt))
    idx = np.clip(idx, 0, len(sim["traj"]["xyzarr_mp"]) - 1)
    actual_time = idx * dt
    
    traj = sim["traj"]
    xyz_mp = traj["xyzarr_mp"][idx]
    xyz_ms = traj["xyzarr_ms"][idx]
    xyz_mm = traj["xyzarr_mm"][idx]
    
    vel_mp = traj["velarr_mp"][idx] if traj.get("velarr_mp") is not None else None
    vel_ms = traj["velarr_ms"][idx] if traj.get("velarr_ms") is not None else None
    vel_mm = traj["velarr_mm"][idx] if traj.get("velarr_mm") is not None else None
    
    rel_mm_mp = xyz_mm - xyz_mp
    rel_mp_ms = xyz_mp - xyz_ms
    
    moon_planet_dist = float(np.linalg.norm(rel_mm_mp))
    planet_star_dist = float(np.linalg.norm(rel_mp_ms))
    
    speed_mm = float(np.linalg.norm(vel_mm)) if vel_mm is not None else None
    speed_mp = float(np.linalg.norm(vel_mp)) if vel_mp is not None else None
    speed_ms = float(np.linalg.norm(vel_ms)) if vel_ms is not None else None
    
    p = _to_params(params or {})
    rhill = _hill_radius_au(p)
    
    return {
        "ok": True,
        "time_requested": float(years),
        "time_actual": actual_time,
        "time_index": int(idx),
        "positions": {
            "star": {"x": float(xyz_ms[0]), "y": float(xyz_ms[1]), "z": float(xyz_ms[2]), "unit": "AU"},
            "planet": {"x": float(xyz_mp[0]), "y": float(xyz_mp[1]), "z": float(xyz_mp[2]), "unit": "AU"},
            "moon": {"x": float(xyz_mm[0]), "y": float(xyz_mm[1]), "z": float(xyz_mm[2]), "unit": "AU"},
        },
        "velocities": {
            "star": {"vx": float(vel_ms[0]), "vy": float(vel_ms[1]), "vz": float(vel_ms[2]), "unit": "AU/yr"} if vel_ms is not None else None,
            "planet": {"vx": float(vel_mp[0]), "vy": float(vel_mp[1]), "vz": float(vel_mp[2]), "unit": "AU/yr"} if vel_mp is not None else None,
            "moon": {"vx": float(vel_mm[0]), "vy": float(vel_mm[1]), "vz": float(vel_mm[2]), "unit": "AU/yr"} if vel_mm is not None else None,
        },
        "distances": {
            "moon_planet": {"value": moon_planet_dist, "unit": "AU", "fraction_of_hill": moon_planet_dist / rhill if rhill else None},
            "planet_star": {"value": planet_star_dist, "unit": "AU"},
        },
        "speeds": {
            "star": {"value": speed_ms, "unit": "AU/yr"},
            "planet": {"value": speed_mp, "unit": "AU/yr"},
            "moon": {"value": speed_mm, "unit": "AU/yr"},
        },
        "context": {
            "rhill_AU": rhill,
            "simdata_range_years": [0.0, t_end],
            "dt": dt,
        }
    }


def _format_claude_response(text: str) -> str:
    """Strip all markdown formatting aggressively."""
    # Remove ALL markdown: ##, **bold**, __underline__, etc.
    text = re.sub(r'#+\s+', '', text)                    # Remove all heading levels
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text, flags=re.DOTALL)  # **text** → text
    text = re.sub(r'__(.+?)__', r'\1', text, flags=re.DOTALL)      # __text__ → text
    text = re.sub(r'\*(.+?)\*', r'\1', text, flags=re.DOTALL)      # *text* → text
    text = re.sub(r'_(.+?)_', r'\1', text, flags=re.DOTALL)        # _text_ → text
    
    # Convert bullet points on same line into separate lines
    # "• item1, • item2" → "• item1\n• item2"
    text = re.sub(r',\s*•\s+', '\n• ', text)
    
    # Remove redundant section headers that appear right before bullets
    text = re.sub(r'(Key|Main|Additional)\s+(Results|Info|Details|Data):\s*\n', '', text, flags=re.IGNORECASE)
    
    # Clean up excessive whitespace but preserve structure
    lines = text.split('\n')
    lines = [line.strip() for line in lines if line.strip()]
    text = '\n'.join(lines)
    
    # Add blank lines between logical sections (lines starting with •)
    text = re.sub(r'\n([^•])', r'\n\n\1', text)

    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# ML endpoints — /ml/predict, /ml/train, /ml/train/status, /ml/train/history
# All bypass the Claude tool loop and are called directly from the frontend.
# ─────────────────────────────────────────────────────────────────────────────

# Default model directory — override with ML_MODEL_DIR env var
ML_MODEL_DIR = os.getenv("ML_MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "models"))
ML_MODEL_DIR = os.path.abspath(ML_MODEL_DIR)

# AuxMLPBinary binary classifier cache
_mlp_binary_cache: Optional[dict] = None
_mlp_binary_lock  = threading.Lock()

# Absolute path to src/ (one level above exomoon/)
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# AuxMLPBinary model directory
_MLP_DIR = os.path.join(_SRC_DIR, "models_mlp")
# HNN hinge4 model directory
_HNN_DIR = os.path.join(_SRC_DIR, "models_hnn_hill_hinge4")

# Training job state (single training job at a time)
_train_job: Dict[str, Any] = {}


class MlPredictRequest(BaseModel):
    system_params:   Dict[str, Any]       # ms_solar, rs_solar, Ts, mp_earth, ap_AU, ep
    t_sim:           float   = 10.0
    moon_retrograde: bool    = False
    em:              float   = 0.0
    mm_resolution:   int     = 50
    am_resolution:   int     = 50


class MlTrainRequest(BaseModel):
    data_path:  str
    out_dir:    Optional[str]  = None     # defaults to ML_MODEL_DIR
    epochs:     int   = 30
    batch_size: int   = 64
    lr:         float = 1e-3
    hidden:     int   = 256
    layers:     int   = 2
    rnn_type:   str   = "gru"
    input_noise_scale: float = 0.0   # 0.0 disables; 1.0 = noise std matches measured per-column MAE


def _load_mlp_binary() -> Optional[dict]:
    """Lazy-load AuxMLPBinary + scaler from _MLP_DIR. Returns {model, scaler} or None."""
    global _mlp_binary_cache
    pt_path = os.path.join(_MLP_DIR, "aux_mlp_binary.pt")
    sc_path = os.path.join(_MLP_DIR, "aux_mlp_scaler.pkl")
    if not (os.path.exists(pt_path) and os.path.exists(sc_path)):
        return None
    try:
        import pickle, torch, torch.nn as nn

        class _AuxMLPBinary(nn.Module):
            def __init__(self, input_dim: int = 14, hidden: int = 64):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden), nn.ReLU(),
                    nn.Linear(hidden, hidden),    nn.ReLU(),
                    nn.Linear(hidden, hidden),    nn.ReLU(),
                    nn.Linear(hidden, 2),
                )
            def forward(self, x):  # type: ignore[override]
                return self.net(x)

        model = _AuxMLPBinary()
        model.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
        model.eval()
        with open(sc_path, "rb") as fh:
            scaler = pickle.load(fh)
        _mlp_binary_cache = {"model": model, "scaler": scaler}
        print("[ML] Loaded AuxMLPBinary from models_mlp/", flush=True)
        return _mlp_binary_cache
    except Exception as e:
        print(f"[ML] Failed to load AuxMLPBinary: {e}", flush=True)
        return None


def _predict_stability_map_mlp(
    system_params:   dict,
    t_sim:           float,
    moon_retrograde: bool,
    em:              float,
    mm_resolution:   int,
    am_resolution:   int,
) -> dict:
    """AuxMLPBinary grid sweep — replaces GRU predict_stability_map for /ml/predict."""
    import torch
    from exomoon.ml.dataset   import SYS_COLS, LOG_SYS_COLS
    from exomoon.habitable_zone import hz_bounds_au
    from exomoon.constants      import merth, msun, rsun, au

    # Load cached model + scaler
    with _mlp_binary_lock:
        cached = _mlp_binary_cache if _mlp_binary_cache is not None else _load_mlp_binary()
    if cached is None:
        return {"ok": False, "error": "no_model",
                "message": "No AuxMLPBinary model found in models_mlp/. "
                           "Run eval_aux_mlp.py --mode cls first."}

    model  = cached["model"]
    scaler = cached["scaler"]

    mp  = float(system_params.get("mp_earth", 1.0))
    ms  = float(system_params.get("ms_solar",  1.0))
    rs  = float(system_params.get("rs_solar",  1.0))
    Ts  = float(system_params.get("Ts",        5772.0))
    ap  = float(system_params.get("ap_AU",     1.0))
    ep  = float(system_params.get("ep",        0.0))
    dp  = float(system_params.get("dp_cgs",    5.5))

    # Derived physical quantities
    mp_kg    = mp * merth
    ms_kg    = ms * msun
    rs_m     = rs * rsun
    rhill_AU = ap * (1.0 - ep) * (mp_kg / (3.0 * ms_kg)) ** (1.0 / 3.0)
    a_inner_au, a_outer_au = hz_bounds_au(Ts, rs_m)

    # Roche limit (fluid-body; rocky moon assumption)
    _MOON_DENSITY_CGS = 3.0
    rp_m       = (0.75 * mp_kg / (np.pi * (dp * 1e3))) ** (1.0 / 3.0)
    a_roche_AU = (2.456 * rp_m * (dp / _MOON_DENSITY_CGS) ** (1.0 / 3.0)) / au

    # Build grids (identical to GRU inference.py construction)
    _MARS_MASS = 0.107
    mm_min  = _MARS_MASS
    mm_max  = max(min(mp, 3.0), mm_min * 1.01)
    mm_grid = np.exp(np.linspace(np.log(mm_min), np.log(mm_max), mm_resolution))

    am_min  = max(a_roche_AU / rhill_AU, 1e-3) if rhill_AU > 1e-6 else 1e-3
    am_grid = np.linspace(am_min, 1.0, am_resolution)

    # Build feature matrix [mm_res × am_res, 14]
    retro = float(int(moon_retrograde))
    vecs: list = []
    for mm in mm_grid:
        for am_ in am_grid:
            vecs.append([ms, rs, Ts, mp, ap, ep,
                         float(mm), float(am_), em, retro,
                         t_sim, rhill_AU, a_inner_au, a_outer_au])

    X = np.array(vecs, dtype=np.float32)

    # Log-transform LOG_SYS_COLS before scaling (matches training preprocessing)
    log_idx = [SYS_COLS.index(c) for c in LOG_SYS_COLS if c in SYS_COLS]
    X_log   = X.copy()
    X_log[:, log_idx] = np.log(np.clip(X_log[:, log_idx], 1e-10, None))
    X_sc = scaler.transform(X_log).astype(np.float32)

    # Forward pass
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(X_sc))).numpy()

    _THRESH = 0.5
    stable    = (probs[:, 0] >= _THRESH).reshape(mm_resolution, am_resolution)
    habitable = (probs[:, 1] >= _THRESH).reshape(mm_resolution, am_resolution)
    both      = stable & habitable

    # Compute valid ranges
    valid_am_per_mm: list = []
    valid_mm_idx:    list = []
    for i, row in enumerate(both):
        cols = np.where(row)[0]
        if len(cols) == 0:
            valid_am_per_mm.append(None)
        else:
            valid_am_per_mm.append([float(am_grid[cols[0]]), float(am_grid[cols[-1]])])
            valid_mm_idx.append(i)

    valid_mm_range = (
        [float(mm_grid[valid_mm_idx[0]]), float(mm_grid[valid_mm_idx[-1]])]
        if valid_mm_idx else None
    )

    return {
        "ok":             True,
        "map_stable":     stable.tolist(),
        "map_habitable":  habitable.tolist(),
        "map_both":       both.tolist(),
        "mm_grid":        mm_grid.tolist(),
        "am_grid":        am_grid.tolist(),
        "valid_mm_range": valid_mm_range,
        "valid_am_per_mm": valid_am_per_mm,
    }


@app.post("/ml/predict")
def ml_predict(req: MlPredictRequest):
    """
    Run stability-habitability map inference over a mm_earth × am_hill grid.
    Uses AuxMLPBinary (fast, ~ms); lazy-loads on first call.
    Returns {"ok": False, "error": "no_model"} if model weights are missing.
    """
    try:
        result = _predict_stability_map_mlp(
            system_params   = req.system_params,
            t_sim           = req.t_sim,
            moon_retrograde = req.moon_retrograde,
            em              = req.em,
            mm_resolution   = req.mm_resolution,
            am_resolution   = req.am_resolution,
        )
        # Always sync to session so ml_plot(heatmap) works even when called from UI button
        if result.get("ok"):
            session.last_ml_prediction = result
        return result
    except Exception as e:
        print(f"[ML] Predict error: {e}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def _run_training_thread(req: MlTrainRequest) -> None:
    """Background thread: run training and update _train_job dict."""
    global _ml_model, _train_job
    out_dir = req.out_dir or ML_MODEL_DIR

    # Resolve data_path relative to src/ (same anchor as ML_MODEL_DIR) so that
    # a bare filename like "ml_dataset.parquet" always finds the file next to
    # run_ml_dataset.py regardless of where uvicorn was launched from.
    _src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = req.data_path if os.path.isabs(req.data_path) else os.path.join(_src_dir, req.data_path)

    def _status_cb(epoch, total, train_loss, val_loss):
        _train_job.update({
            "status": "running", "epoch": epoch, "total_epochs": total,
            "train_loss": round(train_loss, 6), "val_loss": round(val_loss, 6),
        })

    try:
        from exomoon.ml.train import train
        history = train(
            data_path  = data_path,
            out_dir    = out_dir,
            epochs     = req.epochs,
            batch_size = req.batch_size,
            lr         = req.lr,
            hidden     = req.hidden,
            layers     = req.layers,
            rnn_type   = req.rnn_type,
            verbose    = True,
            status_cb  = _status_cb,
            input_noise_scale = req.input_noise_scale,
        )
        _train_job.update({
            "status": "complete",
            "epoch": req.epochs,
            "total_epochs": req.epochs,
            "train_loss": history["train_loss"][-1] if history["train_loss"] else None,
            "val_loss":   history["val_loss"][-1]   if history["val_loss"]   else None,
        })
        # Invalidate MLP binary cache so next /ml/predict reloads fresh weights
        global _mlp_binary_cache
        with _mlp_binary_lock:
            _mlp_binary_cache = None
        print(f"[ML] Training complete. Model saved to {out_dir}", flush=True)
    except Exception as e:
        _train_job.update({"status": "failed", "error": str(e)})
        print(f"[ML] Training failed: {e}", flush=True)
        traceback.print_exc()


@app.post("/ml/train")
def ml_train(req: MlTrainRequest):
    """
    Start a background training job. Returns immediately with a job_id.
    Only one training job runs at a time (returns error if one is already running).
    """
    global _train_job
    if _train_job.get("status") == "running":
        return {"ok": False, "error": "already_training",
                "message": "A training job is already running. Wait for it to complete."}

    job_id = f"train-{uuid.uuid4().hex[:8]}"
    _train_job = {
        "job_id": job_id, "status": "running",
        "epoch": 0, "total_epochs": req.epochs,
        "train_loss": None, "val_loss": None,
    }
    threading.Thread(
        target=_run_training_thread, args=(req,), daemon=True
    ).start()
    print(f"[ML] Training job {job_id} started (rnn_type={req.rnn_type}, epochs={req.epochs})", flush=True)
    return {"ok": True, "job_id": job_id, "status": "started"}


@app.get("/ml/train/status")
def ml_train_status():
    """
    Return current training progress from train_status.json (written each epoch).
    Also includes training_history.json content if training is complete.
    """
    # Check in-memory state first
    status = dict(_train_job) if _train_job else {"status": "idle"}

    # Also try to read train_status.json written by the training process
    status_file = os.path.join(ML_MODEL_DIR, "train_status.json")
    if os.path.exists(status_file):
        try:
            with open(status_file) as f:
                file_status = json.load(f)
            # Merge: in-memory takes priority for live updates
            status = {**file_status, **status}
        except Exception:
            pass

    return {"ok": True, **status}


@app.get("/ml/train/history")
def ml_train_history(model_type: str = "mlp"):
    """
    Return training history JSON for the requested model type.
    model_type="mlp" → models_mlp/aux_mlp_binary_training_history.json
    model_type="hnn" → models_hnn_hill_hinge4/hnn_hill_training_history.json
    Returns {"ok": False} if no history file exists yet.
    """
    layer = model_type.lower().strip()

    if layer == "hnn":
        hist_file = os.path.join(_HNN_DIR, "hnn_hill_training_history.json")
        if not os.path.exists(hist_file):
            return {"ok": False, "message": "No HNN training history found."}
        try:
            with open(hist_file) as f:
                history = json.load(f)
            return {"ok": True, **history}
        except Exception as e:
            return {"ok": False, "message": f"Error reading HNN history: {e}"}

    if layer == "mlp":
        for candidate in [
            os.path.join(_MLP_DIR, "mlp_training_history.json"),
            os.path.join(_MLP_DIR, "aux_mlp_binary_training_history.json"),
            os.path.join(ML_MODEL_DIR, "mlp_training_history.json"),
        ]:
            if os.path.exists(candidate):
                hist_file = candidate
                break
        else:
            return {"ok": False, "message": "No MLP training history found. Train a model first."}
        try:
            with open(hist_file) as f:
                history = json.load(f)
            return {"ok": True, **history}
        except Exception as e:
            return {"ok": False, "message": f"Error reading MLP history: {e}"}

    return {"ok": False, "message": f"Unknown model_type '{model_type}'. Use 'mlp' or 'hnn'."}


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory preview endpoints — GPU HNN hinge4 + GPU GT batch leapfrog
# Both proxy to hnn_gpu_service.py on EC2, with S3 read-through caching.
# Cache bucket: exomoon-ml-inference-cache (separate from nbody-time-series-storage)
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryPreviewRequest(BaseModel):
    system_params:   Dict[str, Any]
    t_sim:           float = 10.0
    moon_retrograde: bool  = False
    em:              float = 0.0
    mm_resolution:   int   = 30
    am_resolution:   int   = 30
    escape_factor:   float = 1.0
    mode:            str   = "hnn_hinge4"    # "hnn_hinge4" or "gt_leapfrog"
    model_version:   str   = HNN_MODEL_VERSION
    force_refresh:   bool  = False           # bypass S3 cache and force a fresh EC2 call


# Large array fields returned by the GPU inference functions that are never consumed
# by the frontend — strip them before caching and before sending the HTTP response.
# This keeps the response JSON < 100 KB instead of 200+ MB.
_STRIP_KEYS = frozenset({
    "traj_planet", "traj_star", "traj_moon",
    "t_grid", "moon_planet_dist",
    "stop_step", "initially_habitable",
})


def _strip_heavy(result: Dict) -> Dict:
    return {k: v for k, v in result.items() if k not in _STRIP_KEYS}


def _inference_cache_key(req: TrajectoryPreviewRequest) -> str:
    """SHA-256 of canonical JSON over all request fields (floats rounded to 6dp)."""
    key_dict = {
        "params":        {k: round(float(v), 6) for k, v in req.system_params.items()},
        "t_sim":         round(req.t_sim, 4),
        "moon_retrograde": bool(req.moon_retrograde),
        "em":            round(req.em, 6),
        "mm_res":        req.mm_resolution,
        "am_res":        req.am_resolution,
        "escape_factor": round(req.escape_factor, 4),
        "mode":          req.mode,
        "model_version": req.model_version,
    }
    return hashlib.sha256(
        json.dumps(key_dict, sort_keys=True).encode()
    ).hexdigest()[:16]


def _cache_s3_key(mode: str, key: str) -> str:
    return f"ml_inference_cache/{mode}/{key}.json"


def _read_cache(mode: str, key: str) -> Optional[Dict]:
    """Return cached result dict if present in S3, else None."""
    if _s3_cache is None:
        return None
    try:
        obj = _s3_cache.get_object(Bucket=INFERENCE_CACHE_BUCKET, Key=_cache_s3_key(mode, key))
        data = json.loads(obj["Body"].read().decode())
        data["from_cache"] = True
        return data
    except _s3_cache.exceptions.NoSuchKey:
        return None
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        print(f"[CACHE] S3 read error: {e}", flush=True)
        return None
    except Exception as e:
        print(f"[CACHE] Unexpected read error: {e}", flush=True)
        return None


def _write_cache(mode: str, key: str, result: Dict) -> None:
    """Upload result JSON to S3 cache (best-effort, never blocks the response)."""
    if _s3_cache is None:
        return
    try:
        payload = dict(result)
        payload["from_cache"] = False
        _s3_cache.put_object(
            Bucket=INFERENCE_CACHE_BUCKET,
            Key=_cache_s3_key(mode, key),
            Body=json.dumps(payload).encode(),
            ContentType="application/json",
        )
        print(f"[CACHE] Written {mode}/{key} ({len(json.dumps(payload))} bytes)", flush=True)
    except Exception as e:
        print(f"[CACHE] Write failed (non-fatal): {e}", flush=True)


def _store_traj_ram_cache(key: str, result: Dict, mm_resolution: int, am_resolution: int) -> None:
    """Store full trajectory arrays in RAM as numpy float32 arrays for instant per-cell access."""
    import numpy as np
    missing = [k for k in ("traj_planet", "traj_star", "traj_moon", "t_grid") if k not in result]
    if missing:
        print(f"[TRAJ_RAM] Skipping store — missing keys: {missing}", flush=True)
        return
    try:
        tp = result["traj_planet"]
        ts = result["traj_star"]
        tm = result["traj_moon"]
        tg = result["t_grid"]
        tp_shape = tp.shape if hasattr(tp, 'shape') else (len(tp),)
        print(f"[TRAJ_RAM] Converting arrays: traj_planet shape={tp_shape}", flush=True)
        entry = {
            "traj_planet":   np.array(tp, dtype=np.float32),  # (N, n_out, 3)
            "traj_star":     np.array(ts, dtype=np.float32),
            "traj_moon":     np.array(tm, dtype=np.float32),
            "t_grid":        np.array(tg, dtype=np.float32),  # (n_out,)
            "mm_resolution": mm_resolution,
            "am_resolution": am_resolution,
        }
        # Verify shape before storing
        assert entry["traj_planet"].ndim == 3, f"traj_planet ndim={entry['traj_planet'].ndim}, expected 3"
        with _traj_ram_lock:
            _traj_ram_cache[key] = entry
            while len(_traj_ram_cache) > _MAX_TRAJ_RAM:
                _traj_ram_cache.pop(next(iter(_traj_ram_cache)))
        n_out = entry["t_grid"].shape[0]
        mb = (entry["traj_planet"].nbytes + entry["traj_star"].nbytes +
              entry["traj_moon"].nbytes) / 1e6
        print(f"[TRAJ_RAM] Stored key={key} shape=({mm_resolution}×{am_resolution}, {n_out}) "
              f"size={mb:.1f}MB cache_size={len(_traj_ram_cache)}", flush=True)
    except Exception as e:
        print(f"[TRAJ_RAM] Store FAILED: {type(e).__name__}: {e}", flush=True)


def _decompress_gt_traj(result: Dict) -> Dict:
    """Decompress float32 zlib trajectory arrays from /gt/predict_numba response.

    EC2 sends trajectory data as base64(zlib(float32 binary)) to reduce transfer size
    from ~486MB JSON to ~15-30MB. This function restores numpy arrays in-place.
    """
    import numpy as np
    import zlib as _zlib
    import base64 as _b64
    for name in ("traj_planet", "traj_star", "traj_moon", "t_grid"):
        b64_key   = f"{name}_zlib_f32"
        shape_key = f"{name}_shape"
        if b64_key in result and shape_key in result:
            arr_bytes    = _zlib.decompress(_b64.b64decode(result.pop(b64_key)))
            arr_shape    = result.pop(shape_key)
            result[name] = np.frombuffer(arr_bytes, dtype=np.float32).reshape(arr_shape)
    return result


def _forward_to_gpu(mode: str, req: TrajectoryPreviewRequest) -> Dict:
    """Forward batch request to EC2 hnn_gpu_service.py and return parsed JSON result."""
    endpoint = "/hnn/predict" if mode == "hnn_hinge4" else "/gt/predict_numba"
    url = GPU_SERVICE_URL.rstrip("/") + endpoint
    body = {
        "system_params":   req.system_params,
        "t_sim":           req.t_sim,
        "moon_retrograde": req.moon_retrograde,
        "em":              req.em,
        "mm_resolution":   req.mm_resolution,
        "am_resolution":   req.am_resolution,
        "escape_factor":   req.escape_factor,
        "n_steps":         5000,
    }
    resp = _requests.post(url, json=body, timeout=GPU_SERVICE_TIMEOUT_S)
    resp.raise_for_status()
    return resp.json()


@app.post("/trajectory/preview")
def trajectory_preview(req: TrajectoryPreviewRequest):
    """
    GPU trajectory preview with S3 read-through cache.

    mode="hnn_hinge4"  → EC2 /hnn/predict      (HNN hinge4 on T4 GPU, ~470s first run; S3 cached)
    mode="gt_leapfrog" → EC2 /gt/predict_numba (Numba CUDA kernel, ~2.3s; NO S3 cache)

    HNN: S3 read-through cache — cache HIT returns in ~10ms, MISS triggers EC2 call.
    GT:  No S3 caching — always calls EC2 directly (~2.3s). RAM cache populated for instant cell clicks.
    """
    try:
        return _trajectory_preview_inner(req)
    except HTTPException:
        raise
    except Exception as e:
        print(f"[TRAJECTORY] Unhandled exception: {type(e).__name__}: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"Trajectory preview error: {type(e).__name__}: {e}")


def _trajectory_preview_inner(req: TrajectoryPreviewRequest):
    mode = req.mode
    if mode not in ("hnn_hinge4", "gt_leapfrog"):
        raise HTTPException(status_code=400, detail=f"Unknown mode '{mode}'. Use 'hnn_hinge4' or 'gt_leapfrog'.")

    key = _inference_cache_key(req)
    print(f"[TRAJECTORY] mode={mode} key={key} mm={req.mm_resolution}x{req.am_resolution} force_refresh={req.force_refresh}", flush=True)

    # ── S3 cache read — HNN only (GT uses Numba at ~2.3s, no S3 caching) ─────────
    if mode == "hnn_hinge4" and not req.force_refresh:
        cached = _read_cache(mode, key)
        if cached is not None:
            with _traj_ram_lock:
                ram_hit = key in _traj_ram_cache
            if ram_hit:
                print(f"[TRAJECTORY] S3+RAM cache HIT for {mode}/{key}", flush=True)
            else:
                # S3 HIT but RAM empty (agent restarted). Populate RAM from S3 data.
                print(f"[TRAJECTORY] S3 HIT, RAM empty — populating RAM from S3 data", flush=True)
                _store_traj_ram_cache(key, cached, req.mm_resolution, req.am_resolution)
            r = _strip_heavy(cached)
            r["cache_key"] = key
            return r

    # ── GPU inference ──────────────────────────────────────────────────────────
    print(f"[TRAJECTORY] Forwarding to GPU service at {GPU_SERVICE_URL} (mode={mode})", flush=True)
    try:
        result = _forward_to_gpu(mode, req)
    except _requests.exceptions.Timeout:
        raise HTTPException(status_code=504,
                            detail=f"GPU service timed out after {GPU_SERVICE_TIMEOUT_S}s")
    except _requests.exceptions.ConnectionError as e:
        raise HTTPException(status_code=502,
                            detail=f"Cannot reach GPU service at {GPU_SERVICE_URL}: {e}")
    except _requests.exceptions.HTTPError as e:
        raise HTTPException(status_code=502,
                            detail=f"GPU service returned error: {e}")
    except Exception as e:
        raise HTTPException(status_code=502,
                            detail=f"GPU service unexpected error: {type(e).__name__}: {e}")

    # GT: decompress zlib-encoded trajectory arrays before storing in RAM
    if mode == "gt_leapfrog":
        result = _decompress_gt_traj(result)

    # Store trajectory arrays in RAM — cell clicks read from here (both modes)
    _store_traj_ram_cache(key, result, req.mm_resolution, req.am_resolution)

    # Write to S3 for HNN only — GT is fast enough (~2.3s) to re-run on restart
    if mode == "hnn_hinge4":
        full_for_s3 = dict(result)
        full_for_s3["mode"]          = mode
        full_for_s3["model_version"] = req.model_version
        full_for_s3["from_cache"]    = False
        full_for_s3["cache_key"]     = key
        threading.Thread(
            target=_write_cache, args=(mode, key, full_for_s3), daemon=True
        ).start()

    # Strip heavy arrays for the HTTP response — frontend only needs maps/grids
    result = _strip_heavy(result)
    result["mode"]          = mode
    result["model_version"] = req.model_version
    result["from_cache"]    = False
    result["cache_key"]     = key
    return result


def _generate_cell_html_2d(frames: list, rhill_au: float, roche_frac: float, label: str) -> str:
    """Standalone 2D canvas animation with play/pause, scrub slider and speed control."""
    moon_rel = [[f["moon_x"] - f["planet_x"], f["moon_y"] - f["planet_y"]] for f in frames]
    max_r = max((abs(x) for p in moon_rel for x in p), default=rhill_au) or rhill_au
    data = json.dumps({"rel": moon_rel, "rhill": rhill_au, "roche": roche_frac, "maxR": max_r * 1.15})
    safe = label.replace('"', "'")
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Mini Orbit — {safe}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d1117;display:flex;flex-direction:column;align-items:center;
  justify-content:center;min-height:100vh;font:12px/1.4 monospace;color:#9ca3af;gap:8px;padding:12px}}
h2{{color:#e5e7eb;font-size:13px}}
canvas{{border-radius:8px;display:block}}
#lbl{{font-size:11px;text-align:center;min-height:1.4em}}
#controls{{display:flex;align-items:center;gap:10px;flex-wrap:wrap;justify-content:center;width:420px}}
#scrub{{flex:1;min-width:120px;accent-color:#3b82f6;cursor:pointer}}
button{{background:#1f2937;border:1px solid #374151;color:#d1d5db;border-radius:5px;
  padding:3px 10px;cursor:pointer;font:inherit;transition:background .15s}}
button:hover{{background:#374151}}
button.active{{background:#3b82f6;border-color:#3b82f6;color:#fff}}
select{{background:#1f2937;border:1px solid #374151;color:#d1d5db;border-radius:5px;
  padding:3px 6px;font:inherit;cursor:pointer}}
</style></head><body>
<h2>Moon Orbit — {safe}</h2>
<canvas id="c" width="420" height="420"></canvas>
<div id="lbl"></div>
<div id="controls">
  <button id="btn">⏸ Pause</button>
  <input id="scrub" type="range" min="0" value="0">
  <select id="spd">
    <option value="0.25">0.25×</option>
    <option value="0.5">0.5×</option>
    <option value="1" selected>1×</option>
    <option value="2">2×</option>
    <option value="4">4×</option>
  </select>
</div>
<script>
const D={data};
const rel=D.rel,rh=D.rhill,ro=D.roche,mr=D.maxR,N=rel.length;
const cv=document.getElementById('c'),ctx=cv.getContext('2d');
const W=cv.width,H=cv.height,cx=W/2,cy=H/2,sc=(W/2-22)/mr;
const scrub=document.getElementById('scrub'),btn=document.getElementById('btn');
const spdSel=document.getElementById('spd'),lbl=document.getElementById('lbl');
scrub.max=N-1;
const TRAIL=100;
let fi=0,playing=true,speed=1.0,acc=0,trail=[];

btn.onclick=()=>{{playing=!playing;btn.textContent=playing?'⏸ Pause':'▶ Play';}};
scrub.addEventListener('mousedown',()=>{{playing=false;btn.textContent='▶ Play';}});
scrub.addEventListener('input',()=>{{fi=+scrub.value;trail=[];drawFrame();}});
spdSel.onchange=()=>{{speed=+spdSel.value;}};

function drawFrame(){{
  ctx.clearRect(0,0,W,H);
  [0.25,0.5,0.75,1.0].forEach(f=>{{
    ctx.beginPath();ctx.arc(cx,cy,rh*f*sc,0,Math.PI*2);
    ctx.strokeStyle=f===1.0?'rgba(239,68,68,0.5)':'rgba(31,41,55,0.9)';
    ctx.lineWidth=f===1.0?1.5:0.6;ctx.stroke();
  }});
  if(ro>0){{
    ctx.beginPath();ctx.arc(cx,cy,ro*rh*sc,0,Math.PI*2);
    ctx.strokeStyle='rgba(220,38,38,0.65)';ctx.setLineDash([4,3]);
    ctx.lineWidth=1;ctx.stroke();ctx.setLineDash([]);
  }}
  trail.push([...rel[fi]]);
  if(trail.length>TRAIL)trail.shift();
  for(let i=1;i<trail.length;i++){{
    const a=i/trail.length;
    ctx.beginPath();
    ctx.moveTo(cx+trail[i-1][0]*sc,cy-trail[i-1][1]*sc);
    ctx.lineTo(cx+trail[i][0]*sc,  cy-trail[i][1]*sc);
    ctx.strokeStyle=`rgba(99,179,237,${{a*0.78}})`;ctx.lineWidth=1.5;ctx.stroke();
  }}
  ctx.beginPath();ctx.arc(cx,cy,6,0,Math.PI*2);ctx.fillStyle='#3b82f6';ctx.fill();
  const mx=cx+rel[fi][0]*sc,my=cy-rel[fi][1]*sc;
  ctx.beginPath();ctx.arc(mx,my,4,0,Math.PI*2);ctx.fillStyle='#94a3b8';ctx.fill();
  const dist=Math.hypot(rel[fi][0],rel[fi][1]);
  lbl.textContent=`Frame ${{fi+1}}/${{N}} · dist ${{dist.toFixed(4)}} AU · Hill ${{rh.toFixed(4)}} AU`;
  scrub.value=fi;
}}

let last=null;
(function loop(ts){{
  requestAnimationFrame(loop);
  if(!last){{last=ts;drawFrame();return;}}
  const dt=(ts-last)/1000;last=ts;
  if(playing){{
    acc+=speed*60*dt;
    const steps=Math.floor(acc);acc-=steps;
    if(steps>0){{fi=(fi+steps)%N;drawFrame();}}
  }}
}})();
</script></body></html>"""


def _generate_cell_html_3d(frames: list, rhill_au: float, label: str) -> str:
    """Standalone Three.js 3D orbit animation using importmap for reliable CDN loading."""
    sp = [[f["star_x"],   f["star_y"],   f["star_z"]]   for f in frames]
    pp = [[f["planet_x"], f["planet_y"], f["planet_z"]] for f in frames]
    mp = [[f["moon_x"],   f["moon_y"],   f["moon_z"]]   for f in frames]
    max_psd = max(
        ((pp[i][0]-sp[i][0])**2 + (pp[i][1]-sp[i][1])**2 + (pp[i][2]-sp[i][2])**2)**0.5
        for i in range(len(frames))
    ) or 1.0
    scale = 5.0 / max_psd
    data = json.dumps({"s": sp, "p": pp, "m": mp, "rhill": rhill_au, "scale": scale})
    safe = label.replace('"', "'")
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>3D Orbit — {safe}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0d1117;overflow:hidden}}
#hud{{position:absolute;top:12px;left:14px;color:#9ca3af;font:11px/1.6 monospace;pointer-events:none;
  background:rgba(13,17,23,0.7);padding:6px 10px;border-radius:6px}}
#controls{{position:absolute;bottom:14px;left:50%;transform:translateX(-50%);
  display:flex;align-items:center;gap:8px;background:rgba(13,17,23,0.8);
  padding:6px 14px;border-radius:8px;font:11px monospace}}
button{{background:#1f2937;border:1px solid #374151;color:#d1d5db;border-radius:5px;
  padding:3px 10px;cursor:pointer;font:inherit}}
button:hover{{background:#374151}}
#scrub3d{{accent-color:#3b82f6;width:180px;cursor:pointer}}
select{{background:#1f2937;border:1px solid #374151;color:#d1d5db;border-radius:5px;
  padding:2px 5px;font:inherit;cursor:pointer}}
</style>
<script type="importmap">
{{"imports":{{"three":"https://cdn.jsdelivr.net/npm/three@0.163.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.163.0/examples/jsm/"}}}}
</script>
</head><body>
<div id="hud">3D Orbit — {safe}<br>Drag to rotate · Scroll to zoom</div>
<div id="controls">
  <button id="btn3">⏸ Pause</button>
  <input id="scrub3d" type="range" min="0" value="0">
  <select id="spd3">
    <option value="0.25">0.25×</option>
    <option value="0.5">0.5×</option>
    <option value="1" selected>1×</option>
    <option value="2">2×</option>
    <option value="4">4×</option>
  </select>
  <span id="lbl3" style="color:#6b7280"></span>
</div>
<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

const D={data},SC=D.scale,N=D.s.length,RH=D.rhill*SC;
// Swap Y↔Z so the orbital plane is horizontal in Three.js (Y-up)
function v(arr,i){{return new THREE.Vector3(arr[i][0]*SC,arr[i][2]*SC,-arr[i][1]*SC);}}

const renderer=new THREE.WebGLRenderer({{antialias:true}});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
renderer.setSize(innerWidth,innerHeight);
renderer.setClearColor(0x0d1117);
document.body.insertBefore(renderer.domElement,document.body.firstChild);

const scene=new THREE.Scene();
const cam=new THREE.PerspectiveCamera(55,innerWidth/innerHeight,0.01,500);
cam.position.set(0,5,9);
const ctrl=new OrbitControls(cam,renderer.domElement);
ctrl.enableDamping=true;ctrl.dampingFactor=0.08;

scene.add(new THREE.AmbientLight(0xffffff,0.4));
const ptLight=new THREE.PointLight(0xfff8e1,2.5,80);scene.add(ptLight);

function mkMesh(r,c,basic=false){{
  return new THREE.Mesh(
    new THREE.SphereGeometry(r,24,14),
    basic?new THREE.MeshBasicMaterial({{color:c}}):new THREE.MeshPhongMaterial({{color:c,shininess:70}}));
}}
const starM=mkMesh(0.18,0xfde68a,true);
const planM=mkMesh(0.065,0x3b82f6);
const moonM=mkMesh(0.028,0x94a3b8);
// Glow ring around star
const glowM=new THREE.Mesh(new THREE.SphereGeometry(0.22,24,14),
  new THREE.MeshBasicMaterial({{color:0xfde68a,transparent:true,opacity:0.08}}));
starM.add(glowM);
scene.add(starM,planM,moonM);

// Hill sphere wireframe around planet
const hillM=new THREE.Mesh(new THREE.SphereGeometry(RH,32,16),
  new THREE.MeshBasicMaterial({{color:0xef4444,wireframe:true,transparent:true,opacity:0.07}}));
scene.add(hillM);

// Trail lines (circular buffer)
const MAX_T=200;
function mkTrail(c){{
  const g=new THREE.BufferGeometry();
  const pos=new Float32Array(MAX_T*3);
  g.setAttribute('position',new THREE.BufferAttribute(pos,3));
  g.setDrawRange(0,0);
  const line=new THREE.Line(g,new THREE.LineBasicMaterial({{color:c,transparent:true,opacity:0.55}}));
  return {{line,pos,buf:g}};
}}
const tS=mkTrail(0xfde68a),tP=mkTrail(0x3b82f6),tM=mkTrail(0x94a3b8);
scene.add(tS.line,tP.line,tM.line);

// Playback state
let fi=0,playing=true,speed=1.0,acc=0,tc=0;
const scrub=document.getElementById('scrub3d');
const btn=document.getElementById('btn3');
const spdSel=document.getElementById('spd3');
const lbl3=document.getElementById('lbl3');
scrub.max=N-1;
btn.onclick=()=>{{playing=!playing;btn.textContent=playing?'⏸ Pause':'▶ Play';}};
scrub.addEventListener('mousedown',()=>{{playing=false;btn.textContent='▶ Play';}});
scrub.addEventListener('input',()=>{{fi=+scrub.value;tc=0;tS.buf.setDrawRange(0,0);tP.buf.setDrawRange(0,0);tM.buf.setDrawRange(0,0);}});
spdSel.onchange=()=>{{speed=+spdSel.value;}};

function updateTrail(t,pt){{
  const ti=(tc%MAX_T)*3;
  t.pos[ti]=pt.x;t.pos[ti+1]=pt.y;t.pos[ti+2]=pt.z;
  t.buf.setDrawRange(0,Math.min(tc+1,MAX_T));
  t.buf.attributes.position.needsUpdate=true;
}}

let last=null;
function animate(ts){{
  requestAnimationFrame(animate);
  if(!last){{last=ts;}}
  const dt=(ts-last)/1000;last=ts;
  if(playing){{
    acc+=speed*60*dt;
    const steps=Math.floor(acc);acc-=steps;
    if(steps>0){{fi=(fi+steps)%N;tc+=steps;}}
  }}
  ctrl.update();
  const s=v(D.s,fi),p=v(D.p,fi),m=v(D.m,fi);
  starM.position.copy(s);planM.position.copy(p);moonM.position.copy(m);
  ptLight.position.copy(s);hillM.position.copy(p);
  if(playing){{updateTrail(tS,s);updateTrail(tP,p);updateTrail(tM,m);}}
  scrub.value=fi;
  lbl3.textContent=`frame ${{fi+1}}/${{N}}`;
  renderer.render(scene,cam);
}}
animate(0);

window.addEventListener('resize',()=>{{
  cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();
  renderer.setSize(innerWidth,innerHeight);
}});
</script></body></html>"""


def _traj_to_frames(planet_arr, star_arr, moon_arr, t_grid) -> list:
    """Convert (n_out, 3) numpy/list arrays → TrajectoryFrame dicts."""
    frames = []
    for i in range(len(t_grid)):
        px, py, pz = float(planet_arr[i][0]), float(planet_arr[i][1]), float(planet_arr[i][2])
        sx, sy, sz = float(star_arr[i][0]),   float(star_arr[i][1]),   float(star_arr[i][2])
        mx, my, mz = float(moon_arr[i][0]),   float(moon_arr[i][1]),   float(moon_arr[i][2])
        mpd = ((mx - px) ** 2 + (my - py) ** 2 + (mz - pz) ** 2) ** 0.5
        psd = ((px - sx) ** 2 + (py - sy) ** 2 + (pz - sz) ** 2) ** 0.5
        msd = ((mx - sx) ** 2 + (my - sy) ** 2 + (mz - sz) ** 2) ** 0.5
        frames.append({
            "t_years":          float(t_grid[i]),
            "star_x":           sx,  "star_y":   sy,  "star_z":   sz,
            "planet_x":         px,  "planet_y": py,  "planet_z": pz,
            "moon_x":           mx,  "moon_y":   my,  "moon_z":   mz,
            "star_vx":          0.0, "star_vy":  0.0, "star_vz":  0.0,
            "planet_vx":        0.0, "planet_vy":0.0, "planet_vz":0.0,
            "moon_vx":          0.0, "moon_vy":  0.0, "moon_vz":  0.0,
            "moon_planet_dist": mpd,
            "planet_star_dist": psd,
            "moon_star_dist":   msd,
            "moon_speed":       0.0,
            "planet_speed":     0.0,
            "star_speed":       0.0,
        })
    return frames


class CellPreviewRequest(BaseModel):
    system_params:   Dict[str, Any]
    mm_idx:          int             # grid indices — primary lookup key
    am_idx:          int
    mm_earth:        float           # physical values — EC2 fallback only
    am_hill:         float
    t_sim:           float = 10.0
    moon_retrograde: bool  = False
    em:              float = 0.0
    mm_resolution:   int   = 50
    am_resolution:   int   = 50
    escape_factor:   float = 1.0
    mode:            str   = "gt_leapfrog"
    model_version:   str   = HNN_MODEL_VERSION
    cache_key:       Optional[str] = None  # exact key from batch response — skips reconstruction


@app.post("/trajectory/cell_preview")
def trajectory_cell_preview(req: CellPreviewRequest):
    """
    Return trajectory frames for a single grid cell from the RAM cache populated by /trajectory/preview.

    The batch runs at n_steps=5000, so the RAM cache holds smooth ~50-frames/orbit trajectories.
    Cell clicks read from RAM instantly — no per-click EC2 call.
    If the batch hasn't been run yet (RAM empty), returns 503.
    """
    if req.mode not in ("hnn_hinge4", "gt_leapfrog"):
        raise HTTPException(status_code=400, detail=f"Unknown mode '{req.mode}'")

    key = req.cache_key
    if not key:
        raise HTTPException(status_code=400, detail="cache_key is required — send the key returned by /trajectory/preview")

    with _traj_ram_lock:
        entry = _traj_ram_cache.get(key)

    if entry is None:
        print(f"[CELL_PREVIEW] RAM empty for key={key} mode={req.mode} — batch not yet complete or agent restarted", flush=True)
        raise HTTPException(
            status_code=503,
            detail="Trajectory batch not yet loaded. Run 'Run Trajectory Previews' first and wait for it to complete."
        )

    mm_resolution = entry.get("mm_resolution", req.mm_resolution)
    am_resolution = entry.get("am_resolution", req.am_resolution)

    if req.mm_idx < 0 or req.mm_idx >= mm_resolution:
        raise HTTPException(status_code=400, detail=f"mm_idx {req.mm_idx} out of range [0, {mm_resolution})")
    if req.am_idx < 0 or req.am_idx >= am_resolution:
        raise HTTPException(status_code=400, detail=f"am_idx {req.am_idx} out of range [0, {am_resolution})")

    cell_idx = req.mm_idx * am_resolution + req.am_idx
    traj_planet = entry["traj_planet"]
    traj_star   = entry["traj_star"]
    traj_moon   = entry["traj_moon"]
    t_grid      = entry["t_grid"]

    frames = _traj_to_frames(traj_planet[cell_idx], traj_star[cell_idx], traj_moon[cell_idx], t_grid)
    print(f"[CELL_PREVIEW] RAM hit key={key} cell=({req.mm_idx},{req.am_idx}) idx={cell_idx} n_frames={len(frames)}", flush=True)
    return {"ok": True, "frames": frames, "n_frames": len(frames),
            "from_ram_cache": True, "mode": req.mode}


@app.get("/trajectory/preview/cache/invalidate")
@app.get("/trajectory/ram_cache/debug")
def ram_cache_debug():
    """Show current RAM cache state — keys stored, shapes, sizes."""
    with _traj_ram_lock:
        entries = {}
        for k, v in _traj_ram_cache.items():
            try:
                import numpy as np
                tp = v["traj_planet"]
                entries[k] = {
                    "shape": list(tp.shape) if hasattr(tp, "shape") else f"list[{len(tp)}]",
                    "mm_resolution": v.get("mm_resolution"),
                    "am_resolution": v.get("am_resolution"),
                    "n_out": int(v["t_grid"].shape[0]) if hasattr(v["t_grid"], "shape") else len(v["t_grid"]),
                    "mb": round((tp.nbytes + v["traj_star"].nbytes + v["traj_moon"].nbytes) / 1e6, 1) if hasattr(tp, "nbytes") else "unknown",
                }
            except Exception as e:
                entries[k] = {"error": str(e)}
    return {"cache_size": len(_traj_ram_cache), "max": _MAX_TRAJ_RAM, "entries": entries}


def invalidate_cache(mode: str = "hnn_hinge4", key: str = ""):
    """
    Delete one cache entry (for testing / after model weight update).
    Pass key= from the cache key computed at request time, or leave blank to see usage.
    """
    if not key:
        return {"ok": False, "message": "Provide ?key=<16-char-hex> to delete a specific entry."}
    s3_key = _cache_s3_key(mode, key)
    try:
        _s3_cache.delete_object(Bucket=INFERENCE_CACHE_BUCKET, Key=s3_key)
        return {"ok": True, "deleted": s3_key}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))