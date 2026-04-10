FROM python:3.11-slim

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ALL source files (including inference.py)
COPY env.py server.py inference.py openenv.yaml ./

EXPOSE 7860

# Health check so HF Spaces knows when the container is ready
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
