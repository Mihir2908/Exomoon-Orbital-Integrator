FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Minimal deps for batch (reuse your existing requirements file if you prefer)
COPY requirements_docker_1.txt /app/requirements_docker_1.txt
RUN pip install --no-cache-dir -r /app/requirements_docker_1.txt

COPY ./src ./src

# Batch entrypoint (ECS Task overrides CMD to this)
CMD ["python", "src/run_job.py"]