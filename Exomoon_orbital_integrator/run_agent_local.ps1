# Run the agent service locally with AWS disabled.
# Simulations run in-process (Numba leapfrog) — no Step Functions or S3 needed.
#
# Usage (from Exomoon_orbital_integrator/):
#   .\run_agent_local.ps1
#
# Prerequisites:
#   pip install "numpy<2.1"   ← Numba 0.60 requires NumPy ≤ 2.0
#   pip install -r requirements_docker_1.txt
#   After changing NumPy version, delete all __pycache__ dirs to clear Numba cache:
#     Get-ChildItem -Recurse -Filter __pycache__ | Remove-Item -Recurse -Force

$env:AWS_ENABLED      = "0"
$env:CLAUDE_ENABLED   = "1"          # set to 0 to disable Claude (chat won't work)
$env:ANTHROPIC_API_KEY = $env:ANTHROPIC_API_KEY  # inherit from shell, or set explicitly
$env:AGENT_BASE_URL   = "http://127.0.0.1:8000"  # used to construct local CSV URLs
$env:PYTHONPATH       = "src"

# Print NumPy version for diagnostics
python -c "import numpy; print('[agent] NumPy', numpy.__version__)"

Write-Host "[agent] Starting agent service on http://127.0.0.1:8000 (AWS_ENABLED=0)" -ForegroundColor Cyan
uvicorn exomoon.agent_service:app --host 127.0.0.1 --port 8000
