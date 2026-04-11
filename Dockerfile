# ============================================================
# Brain Tumor AI — Multi-stage Dockerfile
# Base: PyTorch 2.1 + CUDA 11.8
# ============================================================

# Stage 1: Build dependencies
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime AS base

LABEL maintainer="BrainTumorAI"
LABEL version="1.0.0"
LABEL description="Brain Tumor MRI Classification System"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Stage 2: Python dependencies
# ---------------------------------------------------------------------------
FROM base AS dependencies

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 3: Application
# ---------------------------------------------------------------------------
FROM dependencies AS app

WORKDIR /app

# Copy source
COPY . .

# Create output directories
RUN mkdir -p \
    data/raw \
    data/processed \
    checkpoints \
    outputs/heatmaps \
    outputs/shap \
    outputs/ig \
    results \
    runs

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV CONFIG_PATH=/app/config.yaml
ENV CHECKPOINT_DIR=/app/checkpoints

# Expose ports
EXPOSE 8000 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: run FastAPI
CMD ["uvicorn", "src.api.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
