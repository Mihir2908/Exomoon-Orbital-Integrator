FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    DEBUG=0 \
    PYTHONPATH=/app/src
    
WORKDIR /app
COPY requirements_docker_1.txt .
RUN pip install --no-cache-dir -r requirements_docker_1.txt
COPY ./src ./src
EXPOSE 8000
CMD ["uvicorn", "exomoon.agent_service:app", "--host", "0.0.0.0", "--port", "8000"]