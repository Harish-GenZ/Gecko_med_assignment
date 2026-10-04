import logging
from typing import Any
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.config import get_settings
from app.db.session import engine

logger = logging.getLogger("outlet_verification.db")


def inspect_and_initialize_db() -> dict[str, Any]:
    """
    Inspects PostgreSQL connection, version, and pgvector extension status.
    If pgvector is available in pg_available_extensions, activates it with:
    CREATE EXTENSION IF NOT EXISTS vector;
    Safe to run repeatedly.
    """
    settings = get_settings()
    result: dict[str, Any] = {
        "connected": False,
        "database_url_configured": bool(settings.effective_database_url),
        "target_url": settings.masked_database_url,
        "postgres_version": None,
        "pgvector_available": False,
        "pgvector_available_version": None,
        "pgvector_enabled": False,
        "pgvector_installed_version": None,
        "error": None,
    }

    if not settings.effective_database_url:
        msg = "DATABASE_URL is not configured in environment or .env file."
        logger.error(msg)
        result["error"] = msg
        return result

    if engine is None:
        msg = "Database engine could not be initialized."
        logger.error(msg)
        result["error"] = msg
        return result

    logger.info("Attempting connection to PostgreSQL at %s ...", settings.masked_database_url)

    try:
        with engine.connect() as conn:
            # 1. Connectivity test
            conn.execute(text("SELECT 1;"))
            result["connected"] = True
            logger.info("Database connection successfully established.")

            # 2. Check PostgreSQL version
            version_row = conn.execute(text("SELECT version();")).fetchone()
            if version_row:
                version_str = str(version_row[0])
                result["postgres_version"] = version_str
                logger.info("PostgreSQL Version: %s", version_str)

            # 3. Check if pgvector is available
            avail_ext = conn.execute(
                text("SELECT default_version, comment FROM pg_available_extensions WHERE name = 'vector';")
            ).fetchone()

            if avail_ext:
                result["pgvector_available"] = True
                result["pgvector_available_version"] = avail_ext[0]
                logger.info("pgvector extension is AVAILABLE in PostgreSQL (default version: %s).", avail_ext[0])

                # 4. Enable pgvector if not already enabled
                logger.info("Ensuring pgvector extension is enabled (CREATE EXTENSION IF NOT EXISTS vector)...")
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                conn.commit()

                # 5. Verify that extension is enabled
                installed_ext = conn.execute(
                    text("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';")
                ).fetchone()

                if installed_ext:
                    result["pgvector_enabled"] = True
                    result["pgvector_installed_version"] = installed_ext[1]
                    logger.info("pgvector extension is ENABLED and verified (version: %s).", installed_ext[1])
                else:
                    logger.warning("pgvector CREATE EXTENSION succeeded but extension was not found in pg_extension.")
            else:
                result["pgvector_available"] = False
                logger.warning(
                    "pgvector extension is NOT AVAILABLE in pg_available_extensions on this PostgreSQL instance."
                )

    except SQLAlchemyError as exc:
        err_msg = str(exc)
        # Avoid logging raw password if present in unexpected exception text
        masked_err = err_msg
        if settings.effective_database_url and "@" in settings.effective_database_url:
            raw_credentials = settings.effective_database_url.split("@")[0].split("//")[-1]
            if ":" in raw_credentials:
                password = raw_credentials.split(":")[-1]
                if password:
                    masked_err = err_msg.replace(password, "***")

        result["error"] = masked_err
        logger.error("Database connection or initialization failed: %s", masked_err)
    except Exception as exc:
        result["error"] = str(exc)
        logger.error("Unexpected error during database inspection: %s", exc)

    return result


import time

_db_health_cache: dict[str, Any] | None = None
_db_health_cache_time: float = 0.0
DB_HEALTH_CACHE_TTL_SECONDS: float = 15.0


def check_db_health(force: bool = False) -> dict[str, Any]:
    """
    Lightweight database health probe for /health endpoint.
    Caches the result for 15 seconds to avoid expensive network round-trips
    on high-frequency status polls.
    """
    global _db_health_cache, _db_health_cache_time
    now = time.monotonic()

    if not force and _db_health_cache is not None and (now - _db_health_cache_time) < DB_HEALTH_CACHE_TTL_SECONDS:
        return _db_health_cache

    settings = get_settings()
    if not settings.effective_database_url or engine is None:
        res = {
            "status": "unconfigured",
            "message": "DATABASE_URL is not set.",
            "target": settings.masked_database_url,
        }
        _db_health_cache = res
        _db_health_cache_time = now
        return res

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1;"))
            res = {
                "status": "healthy",
                "message": "Database connection verified.",
                "target": settings.masked_database_url,
            }
            _db_health_cache = res
            _db_health_cache_time = now
            return res
    except Exception as exc:
        res = {
            "status": "unhealthy",
            "message": f"Connection error: {type(exc).__name__}",
            "target": settings.masked_database_url,
        }
        _db_health_cache = res
        _db_health_cache_time = now
        return res
