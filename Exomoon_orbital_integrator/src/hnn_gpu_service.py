"""
hnn_gpu_service.py — FastAPI service for GPU inference (HNN hinge4 + GT batch leapfrog).

Runs on EC2 g4dn.xlarge (NVIDIA T4, 8.1 TFLOPS F32).
Does NOT touch: existing agent_service, ECS/Fargate, Step Functions, NLB, S3 bucket.

Endpoints:
  GET  /health              — CUDA availability + GPU name
  POST /hnn/predict         — stability map via HNN hinge4 on GPU
  POST /gt/predict          — stability map via GT batch leapfrog on GPU (PyTorch, Python loop)
  POST /gt/predict_numba    — stability map + full trajectories via GT batch leapfrog on GPU
                               (Numba CUDA kernel, zero Python iterations, ~2.3s, n_steps frames)
                               Trajectory arrays returned compressed (float32 binary + zlib +
                               base64) — ~15-30MB vs 486MB raw JSON, no EC2-side caching.

Environment variables:
  ML_DEVICE      default "cuda" (set to "cpu" for local testing)
  HNN_MODEL_DIR  default "/app/src/models_hnn_hill_hinge4" (absolute path inside container)
"""

from __future__ import annotations

import base64
import os
import sys
import time
import zlib
from typing import Optional

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ML_DEVICE     = os.getenv("ML_DEVICE", "cuda")
HNN_MODEL_DIR = os.getenv(
    "HNN_MODEL_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_hnn_hill_hinge4"),
)

app = FastAPI(title="Exomoon HNN GPU Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class HnnPredictRequest(BaseModel):
    system_params:   dict
    t_sim:           float = 10.0
    moon_retrograde: bool  = False
    em:              float = 0.0
    mm_resolution:   int   = 50
    am_resolution:   int   = 50
    escape_factor:   float = 1.0
    n_steps:         int   = 1000
    model_dir:       Optional[str] = None


@app.get("/health")
def health():
    import torch
    cuda_ok = torch.cuda.is_available()
    return {
        "ok":             True,
        "device":         ML_DEVICE,
        "cuda_available": cuda_ok,
        "gpu_name":       torch.cuda.get_device_name(0) if cuda_ok else None,
        "hnn_model_dir":  HNN_MODEL_DIR,
    }


def _to_serializable(obj):
    """Recursively convert numpy arrays / scalars to Python-native types."""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


@app.post("/hnn/predict")
def hnn_predict(req: HnnPredictRequest):
    from exomoon.ml.hnn_inference_hill import batch_hnn_hill_trajectories

    t0 = time.perf_counter()
    result = batch_hnn_hill_trajectories(
        system_params   = req.system_params,
        t_sim           = req.t_sim,
        moon_retrograde = req.moon_retrograde,
        em              = req.em,
        mm_resolution   = req.mm_resolution,
        am_resolution   = req.am_resolution,
        escape_factor   = req.escape_factor,
        n_steps         = req.n_steps,
        model_dir       = req.model_dir or HNN_MODEL_DIR,
        device          = ML_DEVICE,
    )
    result["wall_s"] = round(time.perf_counter() - t0, 1)
    result["device"] = ML_DEVICE
    result["cells_computed"] = req.mm_resolution * req.am_resolution
    result["grid_shape"] = [req.mm_resolution, req.am_resolution]
    return _to_serializable(result)


@app.post("/gt/predict")
def gt_predict(req: HnnPredictRequest):
    from exomoon.ml.batch_leapfrog import batch_leapfrog_trajectories

    t0 = time.perf_counter()
    result = batch_leapfrog_trajectories(
        system_params   = req.system_params,
        t_sim           = req.t_sim,
        moon_retrograde = req.moon_retrograde,
        em              = req.em,
        mm_resolution   = req.mm_resolution,
        am_resolution   = req.am_resolution,
        escape_factor   = req.escape_factor,
        n_steps         = req.n_steps,
        device          = ML_DEVICE,
    )
    result["wall_s"] = round(time.perf_counter() - t0, 1)
    result["device"] = ML_DEVICE
    result["cells_computed"] = req.mm_resolution * req.am_resolution
    result["grid_shape"] = [req.mm_resolution, req.am_resolution]
    return _to_serializable(result)


@app.post("/gt/predict_numba")
def gt_predict_numba(req: HnnPredictRequest):
    """
    GT batch leapfrog via Numba CUDA kernel (~2.3s server-side).

    Trajectory arrays are compressed (float32 binary + zlib level 6 + base64) and included
    in the response. Transfer size: ~15-30MB vs 486MB uncompressed JSON. No EC2-side storage.
    Agent service decompresses and stores in its own RAM for instant cell clicks.
    """
    import numpy as np
    from exomoon.ml.batch_leapfrog_numba_cuda import batch_leapfrog_numba_trajectories

    t0 = time.perf_counter()
    result = batch_leapfrog_numba_trajectories(
        system_params   = req.system_params,
        t_sim           = req.t_sim,
        moon_retrograde = req.moon_retrograde,
        em              = req.em,
        mm_resolution   = req.mm_resolution,
        am_resolution   = req.am_resolution,
        escape_factor   = req.escape_factor,
        n_steps         = req.n_steps,
        device          = ML_DEVICE,
    )

    # Compress trajectory arrays: float32 binary + zlib → base64.
    # Orbital trajectories (smooth periodic motion) compress ~5-15x with zlib.
    # ~162MB float32 raw → ~15-30MB base64 → fast transfer, no EC2-side caching needed.
    for key in ("traj_planet", "traj_star", "traj_moon", "t_grid"):
        arr = result.pop(key, None)
        if arr is not None:
            arr_f32    = np.array(arr, dtype=np.float32)
            compressed = zlib.compress(arr_f32.tobytes(), level=6)
            result[f"{key}_zlib_f32"] = base64.b64encode(compressed).decode("ascii")
            result[f"{key}_shape"]    = list(arr_f32.shape)

    result["wall_s"]         = round(time.perf_counter() - t0, 1)
    result["device"]         = ML_DEVICE
    result["cells_computed"] = req.mm_resolution * req.am_resolution
    result["grid_shape"]     = [req.mm_resolution, req.am_resolution]
    return _to_serializable(result)


class CellPredictRequest(BaseModel):
    system_params:   dict
    mm_earth:        float
    am_hill:         float
    t_sim:           float = 10.0
    moon_retrograde: bool  = False
    em:              float = 0.0
    escape_factor:   float = 1.0
    # Higher n_steps = smaller stride = more output frames per orbit.
    # HNN already runs 200k physics steps; n_steps only controls how many are saved.
    # 5000 gives ~50 frames/orbit (vs 10 at n_steps=1000) → smooth MiniOrbitView arc.
    n_steps:         int   = 5000
    model_dir:       Optional[str] = None


@app.post("/hnn/predict_cell")
def hnn_predict_cell(req: CellPredictRequest):
    """Single-cell HNN trajectory — used by /trajectory/cell_preview for MiniOrbitView."""
    import numpy as np
    from exomoon.ml.hnn_inference_hill import batch_hnn_hill_trajectories

    t0 = time.perf_counter()
    result = batch_hnn_hill_trajectories(
        system_params    = req.system_params,
        t_sim            = req.t_sim,
        moon_retrograde  = req.moon_retrograde,
        em               = req.em,
        mm_resolution    = 1,
        am_resolution    = 1,
        n_steps          = req.n_steps,
        escape_factor    = req.escape_factor,
        model_dir        = req.model_dir or HNN_MODEL_DIR,
        device           = ML_DEVICE,
        mm_grid_override = np.array([req.mm_earth]),
        am_grid_override = np.array([req.am_hill]),
    )
    result["wall_s"] = round(time.perf_counter() - t0, 1)
    result["device"] = ML_DEVICE
    return _to_serializable(result)


@app.post("/gt/predict_cell")
def gt_predict_cell(req: CellPredictRequest):
    """Single-cell GT trajectory — used by /trajectory/cell_preview for MiniOrbitView."""
    import numpy as np
    from exomoon.ml.batch_leapfrog import batch_leapfrog_trajectories

    t0 = time.perf_counter()
    result = batch_leapfrog_trajectories(
        system_params    = req.system_params,
        t_sim            = req.t_sim,
        moon_retrograde  = req.moon_retrograde,
        em               = req.em,
        mm_resolution    = 1,
        am_resolution    = 1,
        n_steps          = req.n_steps,
        escape_factor    = req.escape_factor,
        device           = ML_DEVICE,
        mm_grid_override = np.array([req.mm_earth]),
        am_grid_override = np.array([req.am_hill]),
    )
    result["wall_s"] = round(time.perf_counter() - t0, 1)
    result["device"] = ML_DEVICE
    return _to_serializable(result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
