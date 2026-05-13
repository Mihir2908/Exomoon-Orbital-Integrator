FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    #PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=8050 \
    DEBUG=0 \
    PYTHONPATH=/app/src

WORKDIR /app

# Install system build deps (optional: only if numba needs them; slim often ok)
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements_docker_1.txt /app/requirements_docker_1.txt
RUN pip install --no-cache-dir -r requirements_docker_1.txt

#Copy project
COPY ./src ./src

EXPOSE 8050
#RESTORE IF AGENT_SERVICE FAILS #EXPOSE 8050
#RESTORE IF AGENT_SERVICE FAILS #CMD ["sh", "-c", "python src/mcp_server.py & python src/run_dash.py"]

CMD ["python", "src/run_dash.py"]
#CMD ["sh", "-c", "cd /app/src && uvicorn exomoon.agent_service:app --host 0.0.0.0 --port 8000 & cd /app/src && python -m exomoon.mcp_server & cd /app/src && python run_dash.py & wait"]
#CMD ["python", "src/run_dash.py"]