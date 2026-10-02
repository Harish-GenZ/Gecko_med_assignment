"""
Configuration settings for the pipeline.
"""
import os
from pathlib import Path

BASE_DIR    = Path(__file__).parent.parent.parent  # backend/
STORAGE_DIR = BASE_DIR / "storage"
UPLOADS_DIR = STORAGE_DIR / "uploads"
JOBS_DIR    = STORAGE_DIR / "jobs"

# Ensure directories exist at import time
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# OCR
OCR_LANGUAGE         = "en"
OCR_MIN_CONFIDENCE   = 0.60   # Below this → mark page as needs_review
PAGE_SCANNED_THRESHOLD = 50   # chars: pages with fewer chars are treated as scanned

# Classification confidence
CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.70

# Processing
MAX_FILE_SIZE_MB = 200
DPI_RENDER       = 200   # DPI for rasterising PDF pages for OCR

# Job timeout in seconds
JOB_TIMEOUT = 600
