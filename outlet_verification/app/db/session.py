import logging
from collections.abc import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import get_settings

logger = logging.getLogger("outlet_verification.db")

settings = get_settings()

# Configure SQLAlchemy engine with psycopg3 driver
if settings.sqlalchemy_database_url:
    engine = create_engine(
        settings.sqlalchemy_database_url,
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
        pool_timeout=settings.DB_POOL_TIMEOUT,
        pool_pre_ping=True,
        connect_args={"connect_timeout": settings.DB_CONNECT_TIMEOUT},
    )
else:
    engine = None

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine) if engine else None

Base = declarative_base()


def get_db() -> Generator:
    """
    FastAPI dependency yielding a database session per request.
    """
    if SessionLocal is None:
        raise RuntimeError("Database engine is not initialized. Check DATABASE_URL configuration.")

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
