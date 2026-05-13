import os
import json
import re
import uuid
import pathlib
import traceback
from typing import Any, Dict, Optional

import numpy as np
import boto3
import botocore
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from exomoon.params import SystemParams
from exomoon.constants import FOUR_PI2, merth, msun
from exomoon.eda import unpack_sim
from exomoon.exoplanet_archive import fetch_system_by_planet
from exomoon.mcp_server import env_info, dash_url, export_csv, eda_plot

# NEW: Claude SDK
try:
    import anthropic
except Exception:
    anthropic = None

app = FastAPI(title="Exomoon Agent Service", version="0.1.0")

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

# DEBUG: Startup diagnostics
print(f"[STARTUP] anthropic module available: {anthropic is not None}", flush=True)
print(f"[STARTUP] ANTHROPIC_API_KEY length: {len(ANTHROPIC_API_KEY)} chars", flush=True)
print(f"[STARTUP] ANTHROPIC_API_KEY is empty: {len(ANTHROPIC_API_KEY) == 0}", flush=True)
if ANTHROPIC_API_KEY:
    print(f"[STARTUP] ANTHROPIC_API_KEY first 10 chars: {ANTHROPIC_API_KEY[:10]}", flush=True)
print(f"[STARTUP] CLAUDE_ENABLED: {CLAUDE_ENABLED}", flush=True)
print(f"[STARTUP] Claude client created: {claude is not None}", flush=True)

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


class ChatRequest(BaseModel):
    """User message + context (simdata, params, duration, escape threshold)."""
    message: str
    simdata: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    years: Optional[float] = None
    escape_factor: float = 1.0


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
                
                # Build frame from unpacked simdata
                from exomoon.eda import traj_to_frame, to_csv_bytes
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
                outdir = pathlib.Path("outputs")
                outdir.mkdir(exist_ok=True)
                fname = f"exomoon_dataset_{int(years) if years else 0}y.csv"
                fpath = outdir / fname
                with open(fpath, "wb") as f:
                    f.write(csv_bytes)
                
                n_rows = len(frame.get("t_years", [])) if isinstance(frame, dict) else (frame.shape[0] if hasattr(frame, 'shape') else 0)
                print(f"[TOOL] export_csv success: {n_rows} rows saved to {fpath}", flush=True)
                
                # Cache simdata after successful export
                _session.set_simdata(simdata_to_use, req.params)
                
                return {
                    "ok": True,
                    "csv_path": str(fpath.resolve()),
                    "rows": n_rows,
                    "columns_exported": len(columns) if columns else None,
                    "message": f"✅ Exported {n_rows} rows to {fname}"
                }
            except Exception as e:
                print(f"[TOOL] export_csv error: {e}", flush=True)
                import traceback
                print(traceback.format_exc(), flush=True)
                return {"ok": False, "message": f"Export failed: {str(e)}"}


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
    
    # Build context for Claude
    ctx = {
        "has_simdata": bool(effective_simdata),
        "years_hint": req.years,
        "escape_factor": req.escape_factor,
        "has_params": bool(req.params),
        "aws_enabled": AWS_ENABLED,
    }

    system_prompt = (
        "You are an exomoon orbital mechanics assistant. "
        
        "Use the provided tools to answer user requests. "
        "For stability queries: first try stability_from_simdata if simdata is available. "
        "If simdata is missing or insufficient (needs_rerun=true), call start_backend_job automatically. "
        "For range queries (e.g., 'show moon distance every 0.5 years from 0 to 10 years'), "
        "call get_trajectory_at_time() multiple times (eg. years=0, 0.5, 1.0, 1.5, ..., 10.0) and aggregate the results. "
        "Do not ask the user to manually run simulations—that's your responsibility. "
        "Be concise and provide numerical results when available."
        "Be concise and accurate"
        "For conversion requests (AU → km) or (AU -> fraction of planetary hill radius), multiply by 149,597,870.7 km/AU or divide by rhill."
    )

    messages = [
        {
            "role": "user",
            "content": f"User request: {req.message}\n\nContext: {json.dumps(ctx)}",
        }
    ]

    try:
        # Tool-use loop (max 5 iterations to avoid infinite loops)
        for iteration in range(5):
            print(f"[AGENT] Claude iteration {iteration + 1}...", flush=True)
            
            resp = claude.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=900,
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
                text = " ".join(t.strip() for t in final_text_parts if t and t.strip()).strip()
                text = _format_claude_response(text)
                print(f"[AGENT] Claude final response: {text}", flush=True)
                
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
    return dash_url(params=req.params, autorun=False)


@app.post("/tool/export_csv")
def tool_export_csv(req: ToolRequest):
    """Export trajectory as CSV (fast if using cached simdata)."""
    return export_csv(params=req.params, years=req.years, columns=req.columns)


@app.post("/tool/eda_plot")
def tool_eda_plot(req: ToolRequest):
    """Generate EDA time-series plot (positions, distances, speeds)."""
    return eda_plot(
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
        # Emit metadata (mode, job_id if applicable)
        yield f"data: {json.dumps({'type': 'meta', 'payload': {'mode': result.get('mode'), 'job_id': result.get('job_id')}})}\n\n"
        # Stream tokens
        for tok in text.split():
            yield f"data: {json.dumps({'type': 'token', 'payload': tok + ' '})}\n\n"
        # Final payload
        yield f"data: {json.dumps({'type': 'done', 'payload': result})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")

@app.get("/job/{job_id}/status")
def get_job_status(job_id: str):
    """
    Query Step Functions job status.
    Returns: status (RUNNING|SUCCEEDED|FAILED|TIMED_OUT), elapsed_seconds, urls (if done).
    """
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
            import time
            from datetime import datetime as _dt
            start_ts = start_time.timestamp() if hasattr(start_time, "timestamp") else start_time
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
    Retrieve and cache simdata from completed job.
    Called by Dash when job status becomes SUCCEEDED.
    Returns: ok, simdata_cached (bool), message
    """
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