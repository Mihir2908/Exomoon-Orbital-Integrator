import os
import json
import re
import uuid
import time
import signal
import pathlib
import threading
import traceback
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

ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

CLAUDE_ENABLED = os.getenv("CLAUDE_ENABLED", "0") == "1"

claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if CLAUDE_ENABLED else None

# Startup diagnostics
print(f"[STARTUP] CLAUDE_ENABLED={CLAUDE_ENABLED}, AWS_ENABLED={AWS_ENABLED}", flush=True)
print(f"[STARTUP] anthropic available={anthropic is not None}, client created={claude is not None}", flush=True)

# NumPy version guard — Numba 0.60 requires NumPy ≤ 2.0
_np_ver = tuple(int(x) for x in np.__version__.split(".")[:2])
if _np_ver > (2, 0):
    print(
        f"[STARTUP] WARNING: NumPy {np.__version__} may be incompatible with Numba 0.60 "
        f"(requires ≤ 2.0). Local simulations will likely crash. "
        f"Fix: pip install 'numpy<2.1' then delete __pycache__ dirs and restart.",
        flush=True,
    )

s3 = boto3.client("s3", region_name=AWS_REGION) if AWS_ENABLED and BUCKET else None
sf = boto3.client("stepfunctions", region_name=AWS_REGION) if AWS_ENABLED and STATE_MACHINE_ARN else None


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
            "description": "Start a Step Functions job to run a new simulation (with optional stability check).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "years": {"type": "number", "description": "Simulation duration (years)"},
                    "check_stability": {"type": "boolean", "description": "Include stability check in job (default true)"},
                    "escape_factor": {"type": "number", "description": "Escape threshold multiplier (default 1.0)"},
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
            "name": "ml_plot",
            "description": (
                "Generate a PNG plot for ML model results. "
                "plot_type options: "
                "'loss_curves' — training + validation loss over epochs from training_history.json; "
                "'flag_accuracy' — stable/habitable flag accuracy over epochs from training_history.json; "
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
            result = _start_backend_job(req.params, years, check_stability=check_stability, escape_factor=escape_factor)
            
            # NEW: If job started successfully, attempt to retrieve results later
            # The session cache will try_retrieve_job_results() on next request
            if result.get("ok"):
                print(f"[AGENT] Job started: {result['job_id']}, session will monitor for results", flush=True)
            
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
                from exomoon.ml.inference import predict_stability_map as _predict_map

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
                t_sim        = float(tool_input.get("t_sim",         req.years or 10.0))
                mm_res       = int(tool_input.get("mm_resolution",   50))
                am_res       = int(tool_input.get("am_resolution",   50))
                moon_retro   = bool(raw_params.get("moon_retrograde", False))
                em           = float(raw_params.get("em",            0.0))

                result = _predict_map(
                    system_params   = system_params,
                    t_sim           = t_sim,
                    moon_retrograde = moon_retro,
                    em              = em,
                    mm_resolution   = mm_res,
                    am_resolution   = am_res,
                    model_dir       = ML_MODEL_DIR,
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
                    hist_path = pathlib.Path(ML_MODEL_DIR) / "training_history.json"
                    if not hist_path.exists():
                        return {"ok": False, "message": "No training_history.json found. Train the model first via ml_train."}
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

    # NEW: Check for cached simdata if not provided in request
    effective_simdata = req.simdata
    if not effective_simdata:
        cached_sim, cached_par = _session.get_cached()
        if cached_sim:
            effective_simdata = cached_sim
            # Update req object for tool execution
            req.simdata = cached_sim
            if not req.params:
                req.params = cached_par
            print(f"[AGENT] Using cached simdata from previous job", flush=True)
    
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
        "- Trajectory at specific times: call `get_trajectory_at_time()` (multiple calls allowed).\n"
        "- Trajectory over a range: call `get_trajectory_range(t_start, t_end, step)` for time-series snapshots.\n"
        "- CSV exports: call `export_csv` — returns a presigned URL; include as `[Download CSV](url)` in response.\n"
        "- EDA plots: call `eda_plot(variables, plot_type, normalize)` to generate a PNG time-series figure. "
        "When the tool returns `figure_url`, embed the image inline in your response as `![EDA Plot](figure_url)` "
        "AND include a download link `[Download PNG](figure_url)` on the next line.\n"
        "- Dash URL: call `dash_url(planet, autorun)` to generate a shareable URL encoding current system parameters.\n"
        "- Environment debug: call `env_info()` when diagnosing Python import or module path issues.\n"
        "- ML stability map: call `ml_predict(t_sim, mm_resolution, am_resolution)` to run GRU stability sweep (requires trained model).\n"
        "- ML training: call `ml_train(data_path, epochs, ...)` to start background model training.\n"
        "- ML plots: call `ml_plot(plot_type)` to generate a PNG. "
        "plot_type='loss_curves' → training/val loss curves; "
        "plot_type='flag_accuracy' → stable/habitable flag accuracy over epochs; "
        "plot_type='heatmap' → 50×50 stability map from last ml_predict run. "
        "Embed the returned `figure_url` as `![ML Plot](figure_url)` AND `[Download PNG](figure_url)` on the next line.\n"
        "- If `context.ml_prediction` is set, you already have ML prediction results — answer questions about valid mass/orbit ranges directly from that summary without calling `ml_predict` again.\n"
        "- Animation: if `context.animation_url` is set and the user asks for the animation, return `[Download Animation](url)` as a link. Do NOT call any tool for this — the URL is already in context.\n"
        "- Do NOT ask the user to run simulations manually — trigger them yourself.\n\n"

        "## Unit conversions\n"
        "AU → km: ×149,597,870.7. AU/yr → km/s: ×4.74. Hill fraction: divide by rhill_AU."
    )

    messages = [
        {
            "role": "user",
            "content": f"User request: {req.message}\n\nContext: {json.dumps(ctx)}",
        }
    ]

    try:
        # Tool-use loop (max 12 iterations — complex multi-part queries need more rounds)
        for iteration in range(12):
            print(f"[AGENT] Claude iteration {iteration + 1}...", flush=True)
            
            resp = claude.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=4096,
                temperature=0,
                system=system_prompt,
                tools=_tool_specs(),
                messages=messages,
            )

            assistant_content = []
            tool_results_for_next_turn = []
            final_text_parts = []

            # Process Claude's response
            for block in resp.content:
                if block.type == "text":
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

                # Include ML prediction result only when ml_predict was called this turn
                if _session._ml_fresh and _session.last_ml_prediction:
                    result["ml_prediction"] = _session.last_ml_prediction
                    _session._ml_fresh = False  # consume — won't re-send on next turn

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
    result = _chat_with_claude(req)
    text = result.get("message", "")

    def gen():
        # Metadata event — carries job_id if a backend job was started
        meta_evt = {"type": "meta", "mode": result.get("mode"), "job_id": result.get("job_id")}
        yield f"data: {json.dumps(meta_evt)}\n\n"
        # Token-by-token streaming of the markdown response
        for tok in text.split(" "):
            yield f"data: {json.dumps({'type': 'token', 'token': tok + ' '})}\n\n"
        # Done event — carries simdata, presigned URLs, and ML prediction for the frontend to cache
        done_evt = {
            "type":          "done",
            "simdata":       result.get("simdata"),
            "urls":          result.get("urls", {}),
            "job_id":        result.get("job_id"),
            "ml_prediction": result.get("ml_prediction"),
        }
        yield f"data: {json.dumps(done_evt)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")

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
        
        # If succeeded, fetch presigned URLs from S3
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

# Lazy-loaded model: populated on first /ml/predict call, reloaded after training
_ml_model      = None
_ml_model_lock = threading.Lock()

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


def _load_ml_model():
    """Lazy-load the trained MoonRNN from ML_MODEL_DIR. Returns None if not found."""
    global _ml_model
    model_pt = os.path.join(ML_MODEL_DIR, "gru_model.pt")
    cfg_pt   = os.path.join(ML_MODEL_DIR, "model_config.json")
    if not (os.path.exists(model_pt) and os.path.exists(cfg_pt)):
        return None
    try:
        from exomoon.ml.model import MoonRNN
        _ml_model = MoonRNN.load(ML_MODEL_DIR)
        print(f"[ML] Loaded model from {ML_MODEL_DIR}", flush=True)
        return _ml_model
    except Exception as e:
        print(f"[ML] Failed to load model: {e}", flush=True)
        return None


@app.post("/ml/predict")
def ml_predict(req: MlPredictRequest):
    """
    Run stability-habitability map inference over a mm_earth × am_hill grid.
    Lazy-loads the trained MoonRNN on first call.
    Returns {"ok": False, "error": "no_model"} if no trained model exists yet.
    """
    global _ml_model
    with _ml_model_lock:
        if _ml_model is None:
            _ml_model = _load_ml_model()
        if _ml_model is None:
            return {"ok": False, "error": "no_model",
                    "message": f"No trained model found in {ML_MODEL_DIR}. "
                               "Train one first using the ML window."}

    try:
        from exomoon.ml.inference import predict_stability_map
        result = predict_stability_map(
            system_params   = req.system_params,
            t_sim           = req.t_sim,
            moon_retrograde = req.moon_retrograde,
            em              = req.em,
            mm_resolution   = req.mm_resolution,
            am_resolution   = req.am_resolution,
            model_dir       = ML_MODEL_DIR,
        )
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
        )
        _train_job.update({
            "status": "complete",
            "epoch": req.epochs,
            "total_epochs": req.epochs,
            "train_loss": history["train_loss"][-1] if history["train_loss"] else None,
            "val_loss":   history["val_loss"][-1]   if history["val_loss"]   else None,
        })
        # Invalidate cached model so next /ml/predict reloads the freshly-trained weights
        with _ml_model_lock:
            _ml_model = None
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
def ml_train_history():
    """
    Return training_history.json (loss curves, hyperparams, flag accuracy).
    Returns {"ok": False} if no history file exists yet.
    """
    hist_file = os.path.join(ML_MODEL_DIR, "training_history.json")
    if not os.path.exists(hist_file):
        return {"ok": False, "message": "No training history found. Train a model first."}
    try:
        with open(hist_file) as f:
            history = json.load(f)
        return {"ok": True, **history}
    except Exception as e:
        return {"ok": False, "message": f"Error reading history: {e}"}