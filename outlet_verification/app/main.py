import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import candidates_router, health_router, verification_router
from app.config import get_settings
from app.core.logging import setup_logging
from app.db.init_db import inspect_and_initialize_db
from app.db.session import engine

settings = get_settings()
logger = setup_logging(debug=settings.DEBUG)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    # Startup
    logger.info("==================================================")
    logger.info("Starting %s (%s environment)", settings.APP_NAME, settings.APP_ENV)
    logger.info("Host: %s:%s", settings.HOST, settings.PORT)
    logger.info("Target Database: %s", settings.masked_database_url)
    logger.info("==================================================")

    # Perform DB connectivity check, version check, and pgvector extension verification
    db_status = inspect_and_initialize_db()
    if db_status.get("connected"):
        logger.info("Database verification completed successfully.")
    else:
        logger.warning(
            "Database verification could not establish connection. "
            "Server will remain online for health reporting. Reason: %s",
            db_status.get("error"),
        )

    # Pre-warm AI models in background thread so first HTTP request is instant
    import threading

    def _warmup_models():
        try:
            logger.info("Pre-warming AI models (CLIP ViT, EasyOCR, SentenceTransformers) in background...")
            from app.services.authenticity.service import get_authenticity_service
            from app.services.embeddings import (
                get_image_embedding_service,
                get_text_embedding_service,
            )

            get_text_embedding_service()
            get_image_embedding_service()
            get_authenticity_service()
            logger.info("AI models pre-warmed and ready.")
        except Exception as e:
            logger.warning("Model pre-warming encountered non-fatal error: %s", e)

    threading.Thread(target=_warmup_models, daemon=True).start()

    yield

    # Shutdown
    logger.info("Shutting down %s...", settings.APP_NAME)
    if engine:
        engine.dispose()
        logger.info("Database connection engine disposed.")


app = FastAPI(
    title=settings.APP_NAME,
    description="Backend service for outlet verification, duplicate detection, and genuine outlet analysis.",
    version="0.1.0",
    lifespan=lifespan,
)

# Standard CORS middleware for API clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(health_router)
app.include_router(candidates_router)
app.include_router(verification_router)

# Mount Frontend SPA if built, otherwise provide API directory
from pathlib import Path
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent.parent
dist_dir = BASE_DIR / "frontend" / "dist"

if (dist_dir / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=dist_dir / "assets"), name="assets")

    @app.get("/", summary="Frontend Web Application")
    async def serve_spa():
        return FileResponse(dist_dir / "index.html")

    @app.get("/api", summary="API status index")
    def api_index():
        return {
            "service": settings.APP_NAME,
            "status": "online",
            "documentation": "/docs",
            "health_check": "/health",
            "db_diagnostics": "/health/db",
        }
else:
    @app.get("/", summary="Root index")
    def root_index():
        return {
            "service": settings.APP_NAME,
            "status": "online",
            "documentation": "/docs",
            "health_check": "/health",
            "db_diagnostics": "/health/db",
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
