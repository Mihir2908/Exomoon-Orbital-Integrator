import sys, time
print("step 1: Python running", flush=True)
t0 = time.time()

print("step 2: importing os, sys...", flush=True)
import os
print(f"  -> os ok ({time.time()-t0:.1f}s)", flush=True)

print("step 3: importing numpy...", flush=True)
import numpy as np
print(f"  -> numpy ok ({time.time()-t0:.1f}s)", flush=True)

print("step 4: importing torch... (this is the suspected hang point)", flush=True)
# Disable torch telemetry to avoid any network calls
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
import torch
print(f"  -> torch ok ({time.time()-t0:.1f}s) version={torch.__version__}", flush=True)

print("step 5: all imports done", flush=True)
