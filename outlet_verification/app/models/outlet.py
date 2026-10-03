import uuid
from datetime import datetime
from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    Float,
    Index,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.types import UserDefinedType

from app.db.session import Base

# Gracefully support pgvector.sqlalchemy if installed, with custom UserDefinedType fallback
try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    class Vector(UserDefinedType):  # type: ignore
        """
        Fallback SQLAlchemy type for pgvector when pgvector python package is not yet installed.
        """
        def __init__(self, dim: int | None = None):
            self.dim = dim

        def get_col_spec(self, **kw):
            return f"vector({self.dim})" if self.dim else "vector"

        def bind_processor(self, dialect):
            def process(value):
                if value is None:
                    return None
                if isinstance(value, list | tuple):
                    return f"[{','.join(str(float(x)) for x in value)}]"
                return str(value)
            return process

        def result_processor(self, dialect, coltype):
            def process(value):
                if value is None:
                    return None
                if isinstance(value, str):
                    clean = value.strip("[]()")
                    return [float(x) for x in clean.split(",") if x.strip()]
                return value
            return process


class PointType(UserDefinedType):
    """
    SQLAlchemy type for PostgreSQL native POINT type or PostGIS GEOGRAPHY point.
    """
    def get_col_spec(self, **kw):
        return "POINT"

    def bind_processor(self, dialect):
        def process(value):
            if value is None:
                return None
            if isinstance(value, tuple | list) and len(value) == 2:
                # Store as (longitude, latitude)
                return f"({value[0]},{value[1]})"
            return str(value)
        return process

    def result_processor(self, dialect, coltype):
        def process(value):
            if value is None:
                return None
            if isinstance(value, str):
                clean = value.strip("()")
                parts = clean.split(",")
                if len(parts) == 2:
                    return float(parts[0]), float(parts[1])
            return value
        return process


class Outlet(Base):
    """
    SQLAlchemy model for registered outlets.
    Stores metadata, coordinates, auto-synced location, and multimodal embeddings.
    """
    __tablename__ = "outlets"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
        nullable=False,
    )
    name = Column(Text, nullable=False, index=True)
    latitude = Column(Float(precision=53), nullable=False)
    longitude = Column(Float(precision=53), nullable=False)
    location = Column(PointType(), nullable=True)
    image_url = Column(Text, nullable=False)
    
    # Multimodal vector embeddings (Phase 3B)
    name_embedding = Column(Vector(384), nullable=True)
    image_embedding = Column(Vector(512), nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        server_default=text("CURRENT_TIMESTAMP"),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=text("CURRENT_TIMESTAMP"),
        onupdate=datetime.utcnow,
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint("latitude >= -90.0 AND latitude <= 90.0", name="chk_outlets_latitude"),
        CheckConstraint("longitude >= -180.0 AND longitude <= 180.0", name="chk_outlets_longitude"),
        Index("idx_outlets_lat_lng", "latitude", "longitude"),
        Index("idx_outlets_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Outlet(id={self.id}, name='{self.name}', lat={self.latitude}, lng={self.longitude})>"
