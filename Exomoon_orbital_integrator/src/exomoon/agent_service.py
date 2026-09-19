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

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_ORIGIN", "*")],
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

ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

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
        self._cell_frames_fresh: bool = False
        self.last_effective_params: Optional[Dict[str, Any]] = None  # params used for last chat-triggered job
        self._job_fresh: bool = False  # True only for the turn in which start_backend_job was called

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

_session = SessionCache()

# ── Local job store (used when AWS_ENABLED=0) ──────────────────────────────────
# Maps job_id → {status, started, elapsed, csv_bytes, summary, simdata, error}
LOCAL_JOBS: Dict[str, Dict] = {}
_LOCAL_AGENT_BASE = os.getenv("AGENT_BASE_URL", "http://localhost:8000")


def _run_local_job(job_id: str, params_dict: Dict, years: float) -> None:
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
        _session.set_simdata(simdata, params_dict)
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


def _start_backend_job(params: Dict[str, Any], years: Optional[float], check_stability: bool = False, escape_factor: float = 1.0) -> Dict[str, Any]:
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
        _session.update_job(job_id, out_prefix)
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
            "description": "Run ML stability-habitability prediction: sweeps a moon mass × semi-major axis grid and returns valid stable+habitable orbit ranges. Requires a trained model. Use when the user asks about optimal moon parameters or ML-predicted stability.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "t_sim": {"type": "number", "description": "Prediction horizon in simulated years (default 10)"},
                    "mm_resolution": {"type": "integer", "description": "Moon mass grid points (default 50)"},
                    "am_resolution": {"type": "integer", "description": "Moon orbit grid points (default 50)"},
                    "rnn_type": {"type": "string", "description": "'gru' or 'lstm' — which trained model to use (default 'gru')"},
                },
                "required": [],
            },
        },
        {
            "name": "trajectory_preview",
            "description": "Run a GPU batch trajectory preview over a moon mass × semi-major axis grid. Returns how many cells are stable+habitable. Use when the user asks for a physics-based trajectory sweep or wants to validate the ML prediction with actual dynamics.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "description": "'gt_leapfrog' (Numba CUDA, ~2.3s) or 'hnn_hinge4' (HNN ML model, ~470s+S3 cached)"},
                    "mm_resolution": {"type": "integer", "description": "Moon mass grid points — 30 or 50 (default 30)"},
                    "am_resolution": {"type": "integer", "description": "Moon orbit grid points — 30 or 50 (default 30)"},
                    "t_sim": {"type": "number", "description": "Simulation duration in years (default 10)"},
                },
                "required": [],
            },
        },
        {
            "name": "ml_train",
            "description": "Start training the ML stability predictor model in the background. Returns immediately with a job_id. Use when the user asks to train or retrain the ML model.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to ml_dataset.parquet training file (required)"},
                    "epochs": {"type": "integer", "description": "Training epochs (default 30)"},
                    "batch_size": {"type": "integer", "description": "Batch size (default 64)"},
                    "lr": {"type": "number", "description": "Learning rate (default 0.001)"},
                    "hidden": {"type": "integer", "description": "GRU hidden size (default 256)"},
                    "layers": {"type": "integer", "description": "Number of GRU layers (default 2)"},
                    "rnn_type": {"type": "string", "description": "'gru' or 'lstm' (default 'gru')"},
                },
                "required": ["data_path"],
            },
        },
        {
            "name": "trajectory_cell_query",
            "description": (
                "Retrieve the trajectory animation for a specific moon mass and orbit radius from the last cached "
                "trajectory batch. Use when the user asks to see the orbit animation for a specific (moon mass, "
                "semi-major axis) combination, e.g. 'show me the trajectory for 0.2 M⊕ at 0.4 Hill radii'. "
                "Requires that a trajectory batch has already been run for the current system (call "
                "trajectory_preview first if unsure). Returns the animation directly to the frontend."
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
                "'heatmap' — 50×50 moon mass × orbit stability map from the last ML prediction. "
                "Use when the user asks to visualise ML model performance or the stability heatmap."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "plot_type": {
                        "type": "string",
                        "description": "'loss_curves', 'flag_accuracy', or 'heatmap'",
                    },
                    "rnn_type": {
                        "type": "string",
                        "description": "'gru' or 'lstm' — which model's history to plot for loss_curves/flag_accuracy (default 'gru')",
                    },
                },
                "required": ["plot_type"],
            },
        },
    ]


def _execute_tool(tool_name: str, tool_input: Dict[str, Any], req: ChatRequest) -> Dict[str, Any]:
    """
    Execute a tool called by Claude.
    Passes request context (simdata, params) to tool handlers.
    """
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
                cached_sim, _ = _session.get_cached()
                if cached_sim:
                    simdata_to_use = cached_sim
                    print(f"[TOOL] Using cached simdata", flush=True)
                else:
                    retrieved = _session.try_retrieve_job_results(max_retries=1, retry_delay=0.1)
                    if retrieved:
                        simdata_to_use = retrieved
                        print(f"[TOOL] Retrieved simdata from S3", flush=True)
            
            if not simdata_to_use:
                return {"ok": False, "needs_rerun": True, "message": "No data. Running simulation..."}
            
            result = _assess_stability_from_simdata(simdata_to_use, req.params, years, escape_factor)
            if result.get("ok") and simdata_to_use:
                _session.set_simdata(simdata_to_use, req.params)
            return result



        if tool_name == "start_backend_job":
            years = tool_input.get("years", req.years)
            check_stability = bool(tool_input.get("check_stability", True))
            escape_factor = float(tool_input.get("escape_factor", req.escape_factor))
            # Merge any param overrides Claude requested onto the current UI params
            param_overrides = tool_input.get("params") or {}
            effective_params = {**(req.params or {}), **param_overrides}
            print(f"[TOOL] start_backend_job: tool_input_params={param_overrides} req_params_keys={list((req.params or {}).keys())} effective_mm_earth={effective_params.get('mm_earth')}", flush=True)
            result = _start_backend_job(effective_params, years, check_stability=check_stability, escape_factor=escape_factor)

            if result.get("ok"):
                print(f"[AGENT] Job started: {result['job_id']}, session will monitor for results", flush=True)
                # Store effective params so chat_stream can push them back to the frontend
                _session.last_effective_params = effective_params
                result["effective_params"] = effective_params
                _session._job_fresh = True  # signal final result to include job_id

            return result


        if tool_name == "get_trajectory_at_time":
            years = tool_input.get("years")
            if years is None:
                return {"ok": False, "message": "Missing 'years' parameter."}
            
            # Check provided simdata first, then cached
            simdata_to_use = req.simdata
            if not simdata_to_use:
                cached_sim, _ = _session.get_cached()
                if cached_sim:
                    simdata_to_use = cached_sim
                    print(f"[TOOL] Using cached simdata for trajectory query at t={years}y", flush=True)
                else:
                    # Try to retrieve from S3
                    retrieved = _session.try_retrieve_job_results()
                    if retrieved:
                        simdata_to_use = retrieved
                        print(f"[TOOL] Retrieved simdata from S3 for trajectory query", flush=True)
            
            if not simdata_to_use:
                return {"ok": False, "message": "No simulation data available. Run a simulation first."}
            
            result = _get_trajectory_at_time(simdata_to_use, req.params, float(years))
            
            # NEW: Cache simdata if successful
            if result.get("ok") and simdata_to_use:
                _session.set_simdata(simdata_to_use, req.params)
                print(f"[TOOL] Cached simdata after trajectory query", flush=True)
            
            return result

        if tool_name == "export_csv":
            years = tool_input.get("years", req.years)
            columns = tool_input.get("columns")
            
            simdata_to_use = req.simdata
            
            if not simdata_to_use:
                cached_sim, cached_par = _session.get_cached()
                if cached_sim:
                    simdata_to_use = cached_sim
                    print(f"[TOOL] Using cached simdata for export_csv", flush=True)
                else:
                    retrieved = _session.try_retrieve_job_results(max_retries=1, retry_delay=0.1)
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
                _session.set_simdata(simdata_to_use, req.params)

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
                cached_sim, _ = _session.get_cached()
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
            _session.set_simdata(simdata_to_use, req.params)
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
                cached_sim, _ = _session.get_cached()
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
                _session.set_simdata(simdata_to_use, req.params)
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

                # Cache full prediction in session so it can be sent to frontend via done event
                _session.last_ml_prediction = result
                _session._ml_fresh = True

                valid_mm        = result.get("valid_mm_range")
                valid_am_per_mm = result.get("valid_am_per_mm", [])
                mm_grid         = result.get("mm_grid", [])
                am_grid         = result.get("am_grid", [])
                n_valid         = sum(1 for am in valid_am_per_mm if am is not None)

                # Return text summary only — full arrays are NOT sent to Claude (too large)
                return {
                    "ok": True,
                    "valid_mm_range_earth":  valid_mm,
                    "n_valid_mass_bins":     n_valid,
                    "total_mass_bins":       mm_res,
                    "mm_grid_range":         [round(mm_grid[0], 4), round(mm_grid[-1], 4)] if mm_grid else None,
                    "am_grid_range":         [round(am_grid[0], 4), round(am_grid[-1], 4)] if am_grid else None,
                    "message": (
                        f"ML prediction complete. {n_valid}/{mm_res} mass bins have stable+habitable orbits. "
                        f"Valid mass range: {valid_mm[0]:.4f}–{valid_mm[1]:.4f} M⊕ "
                        f"(grid: {mm_grid[0]:.4f}–{mm_grid[-1]:.4f} M⊕). "
                        f"Moon orbit grid spans {am_grid[0]:.3f}–{am_grid[-1]:.3f} Hill radii."
                        if valid_mm else
                        f"ML prediction complete. No stable+habitable orbits found in the {mm_res}×{am_res} grid. "
                        "Consider adjusting system parameters or training the model on more data."
                    ),
                }
            except Exception as e:
                return {"ok": False, "message": f"ML prediction failed: {str(e)}"}

        if tool_name == "trajectory_preview":
            try:
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

                # Check S3 cache before calling trajectory_preview.
                # If MISS: the EC2 call would take 8-10 min and block the SSE stream — not viable
                # over chat. Guide the user to run it from the UI instead.
                key = _inference_cache_key(traj_req)
                cached = _read_cache(mode, key)
                if cached is None:
                    return {
                        "ok": False,
                        "cached": False,
                        "message": (
                            f"No cached trajectory batch found for this system in {mode} mode "
                            f"({mm_res}×{am_res} grid). Running the batch takes 8–10 minutes on the GPU "
                            "and cannot be done inline in chat. To generate it:\n"
                            "1. Open the ML overlay (brain icon, top-right).\n"
                            "2. Go to Layer 2 — Trajectory Previews.\n"
                            "3. Select the mode and grid size, then click 'Run Trajectory Previews'.\n"
                            "Once the batch completes (progress shown in the overlay), come back and ask "
                            "again — I will read the cached result instantly."
                        ),
                    }

                # S3 HIT — trajectory_preview returns in seconds from S3/RAM
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
                _session.last_traj_key     = result.get("cache_key", key)
                _session.last_traj_mm_grid = mm_grid
                _session.last_traj_am_grid = am_grid

                # Push the heatmap to the frontend via the done event (same path as ml_predict)
                _session.last_ml_prediction = {
                    "ok":            True,
                    "mm_grid":       mm_grid,
                    "am_grid":       am_grid,
                    "map_stable":    map_stable,
                    "map_habitable": map_habitable,
                    "map_both":      map_both,
                    "valid_mm_range":  valid_mm,
                    "valid_am_per_mm": valid_am,
                }
                _session._ml_fresh = True

                return {
                    "ok":            True,
                    "mode":          mode,
                    "cached":        True,
                    "n_stable_both": n_stable,
                    "total_cells":   total,
                    "wall_s":        wall_s,
                    "message": (
                        f"Trajectory preview ({mode}) loaded from cache in {wall_s:.1f}s. "
                        f"{n_stable}/{total} cells stable+habitable. "
                        f"Grid: {len(mm_grid)}×{len(am_grid)}, "
                        f"{mm_grid[0]:.3f}–{mm_grid[-1]:.3f} M⊕ × "
                        f"{am_grid[0]:.3f}–{am_grid[-1]:.3f} Hill radii. "
                        "The heatmap has been pushed to the ML overlay."
                        if mm_grid else "Trajectory preview loaded from cache."
                    ),
                }
            except Exception as e:
                return {"ok": False, "message": f"trajectory_preview failed: {str(e)}"}

        if tool_name == "trajectory_cell_query":
            try:
                mm_earth_req = float(tool_input.get("mm_earth", 0.0))
                am_hill_req  = float(tool_input.get("am_hill",  0.0))

                if not _session.last_traj_key or not _session.last_traj_mm_grid or not _session.last_traj_am_grid:
                    return {
                        "ok": False,
                        "message": (
                            "No trajectory batch is loaded in this session yet. "
                            "Call trajectory_preview first to load the batch from cache, "
                            "or ask the user to run a trajectory batch from the ML overlay."
                        ),
                    }

                import numpy as _np
                mm_grid = _np.array(_session.last_traj_mm_grid)
                am_grid = _np.array(_session.last_traj_am_grid)

                # Find nearest grid cell to the requested (mm_earth, am_hill)
                mm_idx = int(_np.argmin(_np.abs(mm_grid - mm_earth_req)))
                am_idx = int(_np.argmin(_np.abs(am_grid - am_hill_req)))
                mm_actual = float(mm_grid[mm_idx])
                am_actual = float(am_grid[am_idx])

                # Check RAM cache for trajectory data
                key = _session.last_traj_key
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
                _session.last_cell_frames      = frames
                _session.last_cell_rhill_au    = rhill_au
                _session.last_cell_roche_frac  = roche_frac
                _session._cell_frames_fresh    = True

                print(f"[TOOL] trajectory_cell_query cell=({mm_idx},{am_idx}) "
                      f"mm={mm_actual:.4f}M⊕ am={am_actual:.3f}H n_frames={len(frames)}", flush=True)
                return {
                    "ok":        True,
                    "mm_idx":    mm_idx,
                    "am_idx":    am_idx,
                    "mm_earth":  mm_actual,
                    "am_hill":   am_actual,
                    "n_frames":  len(frames),
                    "rhill_au":  rhill_au,
                    "message": (
                        f"Trajectory retrieved for moon mass {mm_actual:.4f} M⊕ at "
                        f"{am_actual:.3f} Hill radii (nearest grid cell [{mm_idx},{am_idx}]). "
                        f"{len(frames)} trajectory frames sent to the mini orbit view."
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
            rnn_type_plot = str(tool_input.get("rnn_type", "gru")).lower().strip()
            _OUTPUTS_DIR.mkdir(exist_ok=True)

            try:
                if plot_type in ("loss_curves", "flag_accuracy"):
                    hist_path = pathlib.Path(ML_MODEL_DIR) / f"{rnn_type_plot}_training_history.json"
                    if not hist_path.exists():
                        hist_path = pathlib.Path(ML_MODEL_DIR) / "training_history.json"  # backward compat
                    if not hist_path.exists():
                        return {"ok": False, "message": f"No training history found for {rnn_type_plot.upper()} model. Train the model first via ml_train."}
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
                    pred = _session.last_ml_prediction
                    if not pred or not pred.get("ok"):
                        return {"ok": False, "message": "No ML prediction available. Run ml_predict first."}

                    mm_grid = pred.get("mm_grid", [])
                    am_grid = pred.get("am_grid", [])
                    map_both = pred.get("map_both", [])

                    if not mm_grid or not am_grid or not map_both:
                        return {"ok": False, "message": "ML prediction data is incomplete."}

                    import numpy as _np
                    _arr = _np.array(map_both, dtype=float)  # [mm_res][am_res]

                    mfig, ax = _plt.subplots(figsize=(8, 6), facecolor="#1a1a2e")
                    ax.set_facecolor("#0f0f1a")
                    ax.tick_params(colors="#9ca3af"); ax.xaxis.label.set_color("#9ca3af"); ax.yaxis.label.set_color("#9ca3af")
                    for spine in ax.spines.values(): spine.set_edgecolor("#374151")

                    _im = ax.imshow(
                        _arr.T,
                        origin="lower",
                        aspect="auto",
                        extent=[mm_grid[0], mm_grid[-1], am_grid[0], am_grid[-1]],
                        cmap="YlGn",
                        vmin=0, vmax=1,
                    )
                    _cb = mfig.colorbar(_im, ax=ax, fraction=0.03, pad=0.04)
                    _cb.ax.yaxis.label.set_color("#9ca3af"); _cb.ax.tick_params(colors="#9ca3af")
                    _cb.set_label("Stable + Habitable", color="#9ca3af")
                    ax.set_xlabel("Moon Mass (M⊕, log scale)")
                    ax.set_xscale("log")
                    ax.set_ylabel("Moon Semi-Major Axis (Hill radii)")
                    ax.set_title("ML Stability–Habitability Map (50×50)", color="#e5e7eb")
                    mfig.tight_layout()
                    fname = "ml_heatmap.png"
                    fpath = _OUTPUTS_DIR / fname
                    mfig.savefig(str(fpath), dpi=130, bbox_inches="tight", facecolor=mfig.get_facecolor())
                    _plt.close(mfig)

                else:
                    return {"ok": False, "message": f"Unknown plot_type '{plot_type}'. Use 'loss_curves', 'flag_accuracy', or 'heatmap'."}

                image_url = f"{_LOCAL_AGENT_BASE}/outputs/{fname}"
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


def _chat_rule_based(req: ChatRequest) -> Dict[str, Any]:
    """
    Existing deterministic fallback path (kept for reliability when Claude is unavailable).
    Implements core Option A: simdata-first, then backend job fallback.
    """
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
                job_res = _start_backend_job(req.params, years, check_stability=True, escape_factor=req.escape_factor)
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
        
        job_res = _start_backend_job(req.params, years, check_stability=True, escape_factor=req.escape_factor)
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
    if not CLAUDE_ENABLED or not claude:
        print("[AGENT] Claude not enabled, using rule-based fallback.", flush=True)
        return _chat_rule_based(req)

    # Session-cached simdata (most recently completed job) always takes priority over
    # req.simdata (which the frontend sends from its local store and may be stale).
    # This ensures follow-up queries after a chatbot-triggered job use the new simulation.
    cached_sim, cached_par = _session.get_cached()
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
    if not _session.last_animation_url:
        if AWS_ENABLED and s3 and BUCKET and _session.last_job_id:
            try:
                anim_key = f"outputs/{_session.last_job_id}/animation.html"
                _session.last_animation_url = s3.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": BUCKET, "Key": anim_key},
                    ExpiresIn=86400,
                )
                print(f"[AGENT] Lazy-resolved animation URL for {_session.last_job_id}", flush=True)
            except Exception:
                pass
        elif not AWS_ENABLED and _session.cached_simdata:
            # Generate animation.html locally from cached simdata and serve via static endpoint
            try:
                from exomoon.plotting.anim import build_animation as _build_anim
                _sim = unpack_sim(_session.cached_simdata)
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
                _session.last_animation_url = f"{_LOCAL_AGENT_BASE}/outputs/animation.html"
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
        except Exception:
            ml_pred_summary = {"available": True}
    elif _session.last_ml_prediction:
        try:
            p = _session.last_ml_prediction
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

    ctx = {
        "has_simdata":    bool(effective_simdata),
        "years_hint":     req.years,
        "escape_factor":  req.escape_factor,
        "aws_enabled":    AWS_ENABLED,
        "params":         raw_params,
        "derived":        derived,
        "ml_prediction":  ml_pred_summary,
        "animation_url":  _session.last_animation_url,
    }

    system_prompt = (
        "You are an expert exomoon orbital mechanics and astrobiology assistant embedded in an interactive "
        "simulation tool. The user is looking at a real-time 3D orbital animation of a star–planet–moon system.\n\n"

        "## Response format\n"
        "Always respond in **Markdown**. Use headers, bullet points, bold, and code blocks where appropriate. "
        "Provide numerical results with units. Keep responses focused and concise.\n\n"

        "## System parameters available\n"
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
        "When asked about habitability, reason across multiple factors using the provided values:\n"
        "- **Temperature**: `moon_teff_K` — liquid water requires ~273–373 K; compare to Earth (255 K blackbody).\n"
        "- **HZ position**: `moon_in_hz` / `hz_inner_au` / `hz_outer_au` — is the planet's orbit in the stellar HZ?\n"
        "- **Atmosphere retention**: escape velocity scales with √(g·R); small moons (< 0.1 M⊕) likely cannot "
        "  retain N₂/O₂ atmospheres long-term. `moon_surface_g_ms2` and `moon_radius_earth` inform this.\n"
        "- **Tidal heating**: moons close to the planet (small `am_hill`) or with high eccentricity (`em`) "
        "  experience tidal dissipation — can supplement stellar flux or cause runaway volcanism (e.g. Io).\n"
        "- **Orbital stability**: a moon is stable only if it remains within ~0.5 R_Hill. Use trajectory data "
        "  (`stability_from_simdata`) for quantitative escape analysis.\n"
        "- **Radiation**: moons inside a planet's magnetosphere are shielded; outside, stellar/cosmic radiation "
        "  poses habitability risks.\n"
        "Always note which factors support and which constrain habitability, citing the numerical values.\n\n"

        "## Simdata context\n"
        "If `has_simdata` is true, you have trajectory data for the most recently run simulation. "
        "Use `stability_from_simdata` to analyze stability without re-running. "
        "Return the same `simdata` in your result so the frontend caches it for follow-up queries.\n\n"

        "## Tool usage\n"
        "- Stability queries: use `stability_from_simdata` if simdata available; otherwise call `start_backend_job`.\n"
        "- **Parameter changes**: if the user asks to change any system parameter and run, pass those changes in `start_backend_job`'s `params` field (e.g. `{\"mm_earth\": 0.5, \"ap_AU\": 1.2}`). Do NOT tell the user to adjust sliders manually — apply the changes yourself via `params`.\n"
        "- **CRITICAL — after `start_backend_job`**: return your response to the user IMMEDIATELY after the job submission tool call. Do NOT call any data-query tools (`stability_from_simdata`, `get_trajectory_at_time`, `get_trajectory_range`, `export_csv`, `eda_plot`) in the same turn — the AWS simulation takes 30–120 seconds and no data will be available yet. Tell the user the job is running and they will be notified when results are ready.\n"
        "- Trajectory at specific times: call `get_trajectory_at_time()` (multiple calls allowed).\n"
        "- Trajectory over a range: call `get_trajectory_range(t_start, t_end, step)` for time-series snapshots.\n"
        "- CSV exports: call `export_csv` — returns a presigned URL; include as `[Download CSV](url)` in response.\n"
        "- EDA plots: call `eda_plot(variables, plot_type, normalize)` to generate a PNG time-series figure. "
        "When the tool returns `figure_url`, embed the image inline in your response as `![EDA Plot](figure_url)` "
        "AND include a download link `[Download PNG](figure_url)` on the next line.\n"
        "- Dash URL: call `dash_url(planet, autorun)` to generate a shareable URL encoding current system parameters.\n"
        "- Environment debug: call `env_info()` when diagnosing Python import or module path issues.\n"
        "- ML stability map: call `ml_predict(t_sim, mm_resolution, am_resolution, rnn_type)` to run stability sweep (rnn_type='gru' or 'lstm'; default 'gru'). Requires a trained model of that type.\n"
        "- ML training: call `ml_train(data_path, rnn_type, epochs, ...)` to start background model training for the specified model type.\n"
        "- ML plots: call `ml_plot(plot_type, rnn_type)` to generate a PNG. "
        "plot_type='loss_curves' → training/val loss curves; "
        "plot_type='flag_accuracy' → stable/habitable flag accuracy over epochs; "
        "plot_type='heatmap' → 50×50 stability map from last ml_predict run. "
        "Pass rnn_type to select which model's history to plot (default 'gru'). "
        "Embed the returned `figure_url` as `![ML Plot](figure_url)` AND `[Download PNG](figure_url)` on the next line.\n"
        "- If `context.ml_prediction` is set, you already have ML prediction results — answer questions about valid mass/orbit ranges directly from that summary without calling `ml_predict` again.\n"
        "- Animation: if `context.animation_url` is set and the user asks for the animation, return `[Download Animation](url)` as a link. Do NOT call any tool for this — the URL is already in context.\n"
        "- Do NOT ask the user to run simulations manually — trigger them yourself.\n\n"

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

    messages = [
        {
            "role": "user",
            "content": f"User request: {req.message}\n\nContext: {ctx_json}",
        }
    ]

    try:
        # Tool-use loop (max 12 iterations — complex multi-part queries need more rounds)
        for iteration in range(12):
            print(f"[AGENT] Claude iteration {iteration + 1}...", flush=True)
            
            resp = claude.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=16000,   # must exceed budget_tokens; accommodates 8k thinking + 8k response
                thinking={
                    "type": "enabled",
                    "budget_tokens": 8000,
                },
                system=system_prompt,
                tools=_tool_specs(),
                messages=messages,
                # temperature omitted — extended thinking requires default (1.0); 0 is not permitted
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
                    result = _execute_tool(tool_name, tool_input, req)
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
                    final_attempt = _session.try_retrieve_job_results(max_retries=1, retry_delay=0.1)
                    if final_attempt:
                        result["simdata"] = final_attempt
                        print(f"[AGENT] Returning retrieved simdata ({len(final_attempt)} chars)", flush=True)

                # Include job_id in final result when start_backend_job ran this turn
                # (meta SSE event uses this so the frontend can register with useJobPoller)
                if _session._job_fresh and _session.last_job_id:
                    result["job_id"] = _session.last_job_id
                    result["effective_params"] = _session.last_effective_params
                    _session._job_fresh = False  # consume

                # Include ML prediction / heatmap when ml_predict or trajectory_preview ran this turn
                if _session._ml_fresh and _session.last_ml_prediction:
                    result["ml_prediction"] = _session.last_ml_prediction
                    _session._ml_fresh = False  # consume — won't re-send on next turn

                # Include cell trajectory frames when trajectory_cell_query ran this turn
                if _session._cell_frames_fresh and _session.last_cell_frames:
                    result["cell_frames"]       = _session.last_cell_frames
                    result["cell_rhill_au"]     = _session.last_cell_rhill_au
                    result["cell_roche_frac"]   = _session.last_cell_roche_frac
                    _session._cell_frames_fresh = False  # consume

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
    if not AWS_ENABLED:
        job_id = f"local-{uuid.uuid4().hex[:12]}"
        LOCAL_JOBS[job_id] = {"status": "RUNNING", "started": time.time()}
        threading.Thread(
            target=_run_local_job,
            args=(job_id, req.params or {}, req.years or 0),
            daemon=True,
        ).start()
        print(f"[LOCAL-JOB] Submitted {job_id} (AWS_ENABLED=0)", flush=True)
        return {"ok": True, "job_id": job_id, "status": "submitted"}

    result = _start_backend_job(
        req.params or {},
        req.years or 0,
        check_stability=False,
        escape_factor=req.escape_factor or 1.0,
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
    print(f"[CHAT_STREAM] Entered — msg={req.message[:60]!r} simdata={bool(req.simdata)} params_keys={list((req.params or {}).keys())[:6]}", flush=True)
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
            done_evt = {
                "type":             "done",
                "simdata":          result.get("simdata"),
                "urls":             result.get("urls", {}),
                "job_id":           result.get("job_id"),
                "ml_prediction":    result.get("ml_prediction"),
                "cell_frames":      result.get("cell_frames"),
                "cell_rhill_au":    result.get("cell_rhill_au"),
                "cell_roche_frac":  result.get("cell_roche_frac"),
                "effective_params": result.get("effective_params"),
            }
            # Catch non-JSON-serializable values in done_evt (e.g. numpy scalars from ml_prediction)
            try:
                done_payload = json.dumps(done_evt)
            except Exception as _je:
                print(f"[GEN] done_evt serialization failed ({_je}), stripping heavy fields", flush=True)
                done_evt["ml_prediction"] = None
                done_evt["cell_frames"] = None
                done_evt["simdata"] = None
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
def local_traj_csv(job_id: str):
    """Serve traj.csv for a completed local job (AWS_ENABLED=0)."""
    job = LOCAL_JOBS.get(job_id)
    if not job or job.get("status") != "SUCCEEDED" or "csv_bytes" not in job:
        raise HTTPException(status_code=404, detail="CSV not ready")
    from fastapi.responses import Response
    return Response(content=job["csv_bytes"], media_type="text/csv")


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
def retrieve_job_simdata(job_id: str):
    """
    Cache simdata from a completed job into the session.
    Called by the frontend poller when status becomes SUCCEEDED.
    Handles both local (AWS_ENABLED=0) and S3-backed jobs.
    """
    # ── Local job path ────────────────────────────────────────────────────────
    if job_id in LOCAL_JOBS:
        job = LOCAL_JOBS[job_id]
        if job.get("status") != "SUCCEEDED":
            return {"ok": False, "job_id": job_id, "simdata_cached": False,
                    "message": f"Job not complete (status={job.get('status')})"}
        simdata = job.get("simdata")
        if simdata:
            _session.set_simdata(simdata, {})
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
        _session.set_simdata(simdata, params or {})
        print(f"[JOB-RETRIEVE] Cached simdata for {job_id} in session", flush=True)

        # Generate presigned URL for animation.html so Claude can surface it in chat
        if s3 and BUCKET:
            try:
                anim_key = f"{output_prefix}/animation.html"
                _session.last_animation_url = s3.generate_presigned_url(
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

# Lazy-loaded model cache keyed by rnn_type ("gru" / "lstm")
_ml_model_cache: Dict[str, Any] = {}
_ml_model_lock  = threading.Lock()

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
    model_type:      str     = "gru"      # "gru" or "lstm"


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


def _load_ml_model(rnn_type: str = "gru"):
    """Lazy-load the trained MoonRNN from ML_MODEL_DIR. Returns None if not found."""
    global _ml_model_cache
    model_pt = os.path.join(ML_MODEL_DIR, f"{rnn_type}_model.pt")
    cfg_pt   = os.path.join(ML_MODEL_DIR, f"{rnn_type}_model_config.json")
    # backward compat: also accept legacy model_config.json for gru
    if not os.path.exists(cfg_pt) and rnn_type == "gru":
        cfg_pt = os.path.join(ML_MODEL_DIR, "model_config.json")
    if not (os.path.exists(model_pt) and os.path.exists(cfg_pt)):
        return None
    try:
        from exomoon.ml.model import MoonRNN
        model = MoonRNN.load(ML_MODEL_DIR, rnn_type=rnn_type)
        _ml_model_cache[rnn_type] = model
        print(f"[ML] Loaded {rnn_type.upper()} model from {ML_MODEL_DIR}", flush=True)
        return model
    except Exception as e:
        print(f"[ML] Failed to load {rnn_type} model: {e}", flush=True)
        return None


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
        return _predict_stability_map_mlp(
            system_params   = req.system_params,
            t_sim           = req.t_sim,
            moon_retrograde = req.moon_retrograde,
            em              = req.em,
            mm_resolution   = req.mm_resolution,
            am_resolution   = req.am_resolution,
        )
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
        # Invalidate this model type's cache so next /ml/predict reloads fresh weights
        with _ml_model_lock:
            _ml_model_cache.pop(req.rnn_type, None)
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
def ml_train_history(model_type: str = "gru"):
    """
    Return training history JSON for the requested model type.
    model_type="hnn"  → models_hnn_hill_hinge4/hnn_hill_training_history.json
    model_type="gru"  → {ML_MODEL_DIR}/training_history.json (legacy fallback)
    Returns {"ok": False} if no history file exists yet.
    """
    rnn_type = model_type.lower().strip()

    # HNN hinge4 lives in its own directory outside ML_MODEL_DIR
    if rnn_type == "hnn":
        hist_file = os.path.join(_HNN_DIR, "hnn_hill_training_history.json")
        if not os.path.exists(hist_file):
            return {"ok": False, "message": "No HNN training history found."}
        try:
            with open(hist_file) as f:
                history = json.load(f)
            return {"ok": True, **history}
        except Exception as e:
            return {"ok": False, "message": f"Error reading HNN history: {e}"}

    # MLP binary model has its own directory; check there first
    if rnn_type == "mlp":
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
    else:
        # GRU / LSTM path
        hist_file = os.path.join(ML_MODEL_DIR, f"{rnn_type}_training_history.json")
        if not os.path.exists(hist_file):
            # backward compat: legacy filename for gru
            if rnn_type == "gru":
                hist_file = os.path.join(ML_MODEL_DIR, "training_history.json")
            if not os.path.exists(hist_file):
                return {"ok": False, "message": f"No {rnn_type.upper()} training history found. Train a model first."}
    try:
        with open(hist_file) as f:
            history = json.load(f)
        return {"ok": True, **history}
    except Exception as e:
        return {"ok": False, "message": f"Error reading history: {e}"}


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
        print(f"[TRAJ_RAM] Converting arrays: traj_planet N={len(tp) if tp else 0}, "
              f"n_out={len(tp[0]) if tp and tp[0] else 0}", flush=True)
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


def _forward_to_gpu(mode: str, req: TrajectoryPreviewRequest) -> Dict:
    """Forward batch request to EC2 hnn_gpu_service.py and return parsed JSON result."""
    endpoint = "/hnn/predict" if mode == "hnn_hinge4" else "/gt/predict"
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

    mode="hnn_hinge4"  → EC2 /hnn/predict (HNN hinge4 on T4 GPU, ~44–474s)
    mode="gt_leapfrog" → EC2 /gt/predict  (GT batch leapfrog on T4 GPU, ~193–250s)

    On cache HIT:  returns stored result with from_cache=true  (~10ms)
    On cache MISS: runs GPU inference, stores result, returns with from_cache=false
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

    # ── Cache read (skipped when force_refresh=True) ───────────────────────────
    if not req.force_refresh:
        cached = _read_cache(mode, key)
        if cached is not None:
            with _traj_ram_lock:
                ram_hit = key in _traj_ram_cache
            if ram_hit:
                # S3 hit + RAM hit: everything ready, return instantly
                print(f"[TRAJECTORY] S3+RAM cache HIT for {mode}/{key}", flush=True)
            else:
                # S3 HIT but RAM empty (agent restarted). Populate RAM directly from
                # S3 data — full trajectory arrays are stored in S3 so no EC2 call needed.
                print(f"[TRAJECTORY] S3 HIT, RAM empty — populating RAM from S3 data", flush=True)
                _store_traj_ram_cache(key, cached, req.mm_resolution, req.am_resolution)
            r = _strip_heavy(cached)
            r["cache_key"] = key
            return r

    # ── Cache miss → GPU inference ─────────────────────────────────────────────
    print(f"[TRAJECTORY] Cache MISS — forwarding to GPU service at {GPU_SERVICE_URL}", flush=True)
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

    # Store full trajectory arrays in RAM — cell clicks read from here
    _store_traj_ram_cache(key, result, req.mm_resolution, req.am_resolution)

    # Write FULL result (with trajectory arrays) to S3 so subsequent runs after
    # agent restarts can populate RAM directly from S3, with no EC2 call needed.
    full_for_s3 = dict(result)
    full_for_s3["mode"]          = mode
    full_for_s3["model_version"] = req.model_version
    full_for_s3["from_cache"]    = False
    full_for_s3["cache_key"]     = key
    threading.Thread(
        target=_write_cache, args=(mode, key, full_for_s3), daemon=True
    ).start()

    # Strip heavy arrays for the HTTP response — frontend only needs the maps/grids
    result = _strip_heavy(result)
    result["mode"]          = mode
    result["model_version"] = req.model_version
    result["from_cache"]    = False
    result["cache_key"]     = key
    return result


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
        print(f"[CELL_PREVIEW] RAM empty for key={key} — batch not yet complete or agent restarted without a cache hit", flush=True)
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