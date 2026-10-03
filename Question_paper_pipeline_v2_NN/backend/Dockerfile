# Gecko Med Question Paper Extraction Pipeline
# Production Backend Container for Railway & Cloud Hosting

FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
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

# Install system dependencies required for OpenCV, PyMuPDF, and C++ ONNX extensions
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

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y opencv-python opencv-contrib-python && \
    pip install --no-cache-dir opencv-python-headless

# Pre-download and cache RapidOCR ONNX models and SentenceTransformer neural models into the image layer
RUN python -c "\
from rapidocr_onnxruntime import RapidOCR; RapidOCR(); \
try:\
    from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); \
    print('Neural models pre-cached successfully into Docker image layer.'); \
except Exception as e:\
    print(f'SentenceTransformer pre-cache notice: {e}'); \
"

# Copy application code
COPY . .

# Ensure storage directories exist
RUN mkdir -p /app/storage/uploads /app/storage/jobs

EXPOSE 8000

# Dynamically bind to Railway's $PORT (defaults to 8000 if not set)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
