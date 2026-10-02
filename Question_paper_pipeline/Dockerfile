# Gecko Med Question Paper Extraction Pipeline
# Unified Full-Stack Production Container (Optimized for 1-Click Railway Deployment)

# ==============================================================================
# Stage 1: Build React Frontend
# ==============================================================================
FROM node:20-alpine AS frontend-builder

WORKDIR /frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# ==============================================================================
# Stage 2: Python Backend with Embedded UI
# ==============================================================================
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enforce unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    FLAGS_allocator_strategy=naive_best_fit \
    PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT=0 \
    FLAGS_use_mkldnn=0 \
    PADDLE_ENABLE_MKLDNN=0

WORKDIR /app

# Install native dependencies required for OpenCV, PyMuPDF, and PaddlePaddle
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install backend Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download and cache PaddleOCR detection and recognition models
# Ensures zero cold-start delay when the container starts on Railway
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='en', use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, enable_mkldnn=False)"

# Copy backend application source
COPY backend/ /app/

# Copy built frontend assets into /app/static for FastAPI to serve
COPY --from=frontend-builder /frontend/dist /app/static

# Ensure persistent storage directories exist
RUN mkdir -p /app/storage/uploads /app/storage/jobs

EXPOSE 8000

# Dynamically bind to Railway's assigned $PORT (defaults to 8000)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
