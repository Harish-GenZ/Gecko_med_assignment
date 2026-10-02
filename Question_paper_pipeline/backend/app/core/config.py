"""
Configuration settings for the pipeline.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR    = Path(__file__).parent.parent.parent  # backend/

# Load .env file from backend/ directory if present
ENV_FILE = BASE_DIR / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
else:
    load_dotenv()

# Server & Network Configuration (Railway dynamic PORT support)
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
CORS_ORIGINS_RAW = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173,*")
CORS_ORIGINS = [o.strip() for o in CORS_ORIGINS_RAW.split(",") if o.strip()]

# Storage Configuration (Supports Railway persistent volumes)
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", str(BASE_DIR / "storage")))
UPLOADS_DIR = STORAGE_DIR / "uploads"
JOBS_DIR    = STORAGE_DIR / "jobs"

# Static directory for serving pre-built frontend SPA if present
STATIC_DIR = Path(os.getenv("STATIC_DIR", str(BASE_DIR / "static")))
FRONTEND_DIST_DIR = BASE_DIR.parent / "frontend" / "dist"

# Ensure directories exist at import time
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# OCR
OCR_LANGUAGE         = "en"
OCR_MIN_CONFIDENCE   = 0.60   # Below this → mark page as needs_review
PAGE_SCANNED_THRESHOLD = 50   # chars: pages with fewer chars are treated as scanned
OCR_WORKERS          = int(os.getenv("OCR_WORKERS", "5"))

# Classification confidence
CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.70

# Processing
MAX_FILE_SIZE_MB = 200
DPI_RENDER       = 200   # DPI for rasterising PDF pages for OCR

# Job timeout in seconds (40 mins for large multi-page PDFs)
JOB_TIMEOUT = 2400

# ===========================================================================
# Vision Verification Layer Settings (Gemini Multimodal)
# ===========================================================================
GEMINI_API_KEY          = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_VISION_MODEL     = os.getenv("GEMINI_VISION_MODEL", "gemini-3.1-flash-lite").strip()
VISION_MAX_CONCURRENT   = int(os.getenv("VISION_MAX_CONCURRENT", "4"))
VISION_MAX_RETRIES      = int(os.getenv("VISION_MAX_RETRIES", "2"))
VISION_ENABLED          = os.getenv("VISION_ENABLED", "true").lower() in ("true", "1", "yes")
VISION_TIMEOUT_SECONDS  = int(os.getenv("VISION_TIMEOUT_SECONDS", "30"))

