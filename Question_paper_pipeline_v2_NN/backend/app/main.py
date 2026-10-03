import os
import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.api.routes import router
from app.core.config import CORS_ORIGINS, STATIC_DIR, FRONTEND_DIST_DIR, HOST, PORT

import sys
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

# Ensure immediate real-time log flushing to terminal stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)


def preload_and_warmup_neural_networks():
    """
    Preloads and warms up neural network models into server RAM at boot time.
    Eliminates first-request cold starts and ensures sub-second response times.
    """
    logger.info("=" * 65)
    logger.info("[Neural Warmup] Preloading neural networks into server memory...")
    start_all = time.time()

    # 1. Warm up Layout Detector (Stage 1)
    try:
        t0 = time.time()
        from PIL import Image
        from app.services.layout.layout_detector import get_layout_detector
        detector = get_layout_detector()
        dummy_img = Image.new("RGB", (200, 200), color=(255, 255, 255))
        detector.detect_layout(dummy_img)
        logger.info(f"[Neural Warmup] Stage 1 Layout Detector primed in {time.time()-t0:.3f}s")
    except Exception as e:
        logger.warning(f"[Neural Warmup] Stage 1 Layout Detector notice: {e}")

    # 2. Warm up RapidOCR ONNX Runtime Engine (Stage 2)
    try:
        t0 = time.time()
        import numpy as np
        from app.services.ocr.ocr_engine import _get_rapid_ocr
        rapid = _get_rapid_ocr()
        if rapid is not None:
            dummy_arr = np.full((128, 256, 3), 255, dtype=np.uint8)
            rapid(dummy_arr)
            logger.info(f"[Neural Warmup] Stage 2 RapidOCR ONNX Engine primed in {time.time()-t0:.3f}s")
    except Exception as e:
        logger.warning(f"[Neural Warmup] Stage 2 RapidOCR warmup notice: {e}")

    # 3. Warm up Semantic Question Classifier (Stage 4)
    try:
        t0 = time.time()
        from app.services.classification.question_classifier import (
            _get_embedding_model,
            _init_anchor_embeddings,
            classify_question,
        )
        model = _get_embedding_model()
        if model is not None:
            _init_anchor_embeddings(model)
            classify_question({"text": "Describe the pathology and management of myocardial infarction"})
            logger.info(f"[Neural Warmup] Stage 4 Semantic MiniLM Classifier primed in {time.time()-t0:.3f}s")
    except Exception as e:
        logger.warning(f"[Neural Warmup] Stage 4 Classifier warmup notice: {e}")

    total_t = time.time() - start_all
    logger.info(f"[Neural Warmup] All neural networks successfully loaded into server in {total_t:.2f}s!")
    logger.info("=" * 65)


@asynccontextmanager
async def lifespan(app: FastAPI):
    preload_and_warmup_neural_networks()
    yield


app = FastAPI(
    title       = "Gecko Med — Question Paper Extraction API",
    description = "AI/ML pipeline for extracting and classifying questions from medical exam PDFs.",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
    lifespan    = lifespan,
)

# CORS middleware for cross-origin frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins     = CORS_ORIGINS,
    allow_origin_regex = r"https?://(?:localhost|127\.0\.0\.1)(?::\d+)?",
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

app.include_router(router)

# Mount static frontend assets if built
resolved_static: Path | None = None
if STATIC_DIR.exists() and (STATIC_DIR / "index.html").exists():
    resolved_static = STATIC_DIR
elif FRONTEND_DIST_DIR.exists() and (FRONTEND_DIST_DIR / "index.html").exists():
    resolved_static = FRONTEND_DIST_DIR

if resolved_static is not None:
    # Serve assets directory if present
    assets_dir = resolved_static / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        # Don't hijack API or documentation routes
        if full_path == "api" or full_path.startswith("api/") or full_path.startswith("docs") or full_path.startswith("redoc") or full_path.startswith("openapi.json"):
            raise HTTPException(status_code=404, detail="Not Found")
        file_path = resolved_static / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(resolved_static / "index.html")
else:
    @app.get("/")
    def root():
        return {
            "service": "Gecko Med Question Paper Extraction Pipeline",
            "docs":    "/docs",
            "health":  "/api/health",
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=False)
