FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    #PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=8050 \
    DEBUG=0

WORKDIR /app

# Install system build deps (optional: only if numba needs them; slim often ok)
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements_docker_1.txt /app/requirements_docker_1.txt
RUN pip install --no-cache-dir -r requirements_docker_1.txt

#Copy project
COPY ./src ./src

EXPOSE 8050
CMD ["python", "src/run_dash.py"]