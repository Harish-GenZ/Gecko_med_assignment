import os
import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.api.routes import router
from app.core.config import CORS_ORIGINS, STATIC_DIR, FRONTEND_DIST_DIR, HOST, PORT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

app = FastAPI(
    title       = "Gecko Med — Question Paper Extraction API",
    description = "AI/ML pipeline for extracting and classifying questions from medical exam PDFs.",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# CORS middleware for cross-origin frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins     = CORS_ORIGINS,
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
        # Don't hijack API routes
        if full_path.startswith("api/") or full_path.startswith("docs") or full_path.startswith("redoc") or full_path.startswith("openapi.json"):
            return None
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
