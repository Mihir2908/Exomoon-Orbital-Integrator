FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    ML_DEVICE=cuda \
    HNN_MODEL_DIR=/app/src/models_hnn_hill_hinge4

WORKDIR /app

COPY requirements_gpu.txt .
RUN pip install --no-cache-dir -r requirements_gpu.txt

COPY ./src ./src

EXPOSE 8001
CMD ["uvicorn", "hnn_gpu_service:app", "--host", "0.0.0.0", "--port", "8001"]
