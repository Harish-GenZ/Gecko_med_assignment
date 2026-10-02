"""
Configuration settings for the pipeline.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR    = Path(__file__).parent.parent.parent  # backend/
STORAGE_DIR = BASE_DIR / "storage"
UPLOADS_DIR = STORAGE_DIR / "uploads"
JOBS_DIR    = STORAGE_DIR / "jobs"

# Load .env file from backend/ directory if present
ENV_FILE = BASE_DIR / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
else:
    load_dotenv()

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

