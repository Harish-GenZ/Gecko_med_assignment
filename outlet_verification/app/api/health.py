from typing import Any
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.db.init_db import check_db_health, inspect_and_initialize_db

router = APIRouter(tags=["Health"])


@router.get("/health", summary="Application & Database Health Check")
def health_check() -> dict[str, Any]:
    """
    Verifies that the backend is running and reports Railway PostgreSQL connectivity.
    """
    settings = get_settings()
    db_health = check_db_health()

    is_db_healthy = db_health.get("status") == "healthy"
    overall_status = "ok" if is_db_healthy else "degraded"

    return {
        "status": overall_status,
        "backend": {
            "name": settings.APP_NAME,
            "status": "running",
            "environment": settings.APP_ENV,
        },
        "database": db_health,
    }


@router.get("/health/db", summary="Detailed Database & pgvector Diagnostics")
def detailed_db_diagnostics() -> JSONResponse:
    """
    Detailed inspection of PostgreSQL version, extension availability, and pgvector state.
    """
    inspection = inspect_and_initialize_db()
    http_status = status.HTTP_200_OK if inspection["connected"] else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(status_code=http_status, content=inspection)
