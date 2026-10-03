# ==============================================================================
# Gecko Med Question Paper Extraction Pipeline (v2 Neural Network)
# Unified Full-Stack Production Container (Optimized for 1-Click Railway Deployment)
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Build React Frontend
# ------------------------------------------------------------------------------
FROM node:20-alpine AS frontend-builder

WORKDIR /frontend

COPY frontend/package*.json ./
RUN npm install

COPY frontend/ ./
RUN npm run build

# ------------------------------------------------------------------------------
# Stage 2: Python Backend with Embedded Frontend
# ------------------------------------------------------------------------------
FROM python:3.11-slim

# Prevent Python bytecode caching and enforce real-time terminal log flushing
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    FLAGS_allocator_strategy=naive_best_fit \
    OCR_WORKERS=2 \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

# Install native system dependencies required for image decoding and C++ ONNX extensions
# libgl1, libglib2.0-0, libgomp1, libxcb1, libx11-6, libxext6, libsm6, libxrender1
# resolve missing libxcb.so.1 and libGL.so.1 shared library errors in headless Debian
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libxcb1 \
    libx11-6 \
    libxext6 \
    libsm6 \
    libxrender1 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install backend Python dependencies
# Pre-install CPU-only PyTorch to prevent downloading 2.8 GB CUDA libraries on Railway
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y opencv-python opencv-contrib-python && \
    pip install --no-cache-dir opencv-python-headless

# Pre-download and cache RapidOCR ONNX models into the image layer
# Eliminates cold-start runtime model downloads on container boot
RUN python -c "from rapidocr_onnxruntime import RapidOCR; RapidOCR()"

# Copy backend application source
COPY backend/ /app/

# Copy compiled production React frontend directly into /app/static for FastAPI to serve
COPY --from=frontend-builder /frontend/dist /app/static

# Ensure persistent storage directories exist
RUN mkdir -p /app/storage/uploads /app/storage/jobs

EXPOSE 8000

# Docker healthcheck pinging FastAPI health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://127.0.0.1:${PORT:-8000}/api/health || exit 1

# Dynamically bind to Railway's assigned $PORT (defaults to 8000)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
