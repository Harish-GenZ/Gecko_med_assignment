import math
from functools import lru_cache
from pathlib import Path
from pydantic import computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.logging import mask_database_url

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"


class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables and .env file.
    """
    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_NAME: str = "Outlet Verification System"
    APP_ENV: str = "development"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # PostgreSQL Connection URL
    DATABASE_URL: str | None = None
    POSTGRESQL_URL_RAILWAYS: str | None = None

    # Railway / Tigris S3 Object Storage Settings
    ENDPOINT_URL: str = "https://t3.storageapi.dev"
    Region: str = "auto"
    Bucket_Name: str = ""
    ACCESS_KEY_ID: str = ""
    SECRET_ACCESS_KEY: str = ""

    # Connection pool configuration
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_TIMEOUT: int = 30
    DB_CONNECT_TIMEOUT: int = 10

    # Candidate Retrieval Configuration (Phase 4)
    RETRIEVAL_NAME_TOP_K: int = 5
    RETRIEVAL_IMAGE_TOP_K: int = 5
    RETRIEVAL_GEO_RADIUS_METERS: float = 500.0
    RETRIEVAL_GEO_TOP_K: int = 10

    # Phase 5: Verification Decision Engine Configuration
    VERIFICATION_NAME_WEIGHT: float = 0.30
    VERIFICATION_IMAGE_WEIGHT: float = 0.45
    VERIFICATION_GEO_WEIGHT: float = 0.25

    VERIFICATION_GEO_SCALE_METERS: float = 100.0

    VERIFICATION_DUPLICATE_THRESHOLD: float = 0.80
    VERIFICATION_GENUINE_THRESHOLD: float = 0.35

    VERIFICATION_REVIEW_MIN_COVERAGE: float = 0.55

    VERIFICATION_STRONG_IMAGE_THRESHOLD: float = 0.90
    VERIFICATION_STRONG_IMAGE_GEO_METERS: float = 100.0

    VERIFICATION_MIN_MARGIN: float = 0.10

    @model_validator(mode="after")
    def validate_verification_settings(self) -> "Settings":
        total_weight = (
            self.VERIFICATION_NAME_WEIGHT
            + self.VERIFICATION_IMAGE_WEIGHT
            + self.VERIFICATION_GEO_WEIGHT
        )
        if not math.isclose(total_weight, 1.0, abs_tol=1e-5):
            raise ValueError(
                f"Verification weights must sum to 1.0. Got {total_weight:.4f} "
                f"(name={self.VERIFICATION_NAME_WEIGHT}, image={self.VERIFICATION_IMAGE_WEIGHT}, geo={self.VERIFICATION_GEO_WEIGHT})"
            )
        if self.VERIFICATION_GEO_SCALE_METERS <= 0:
            raise ValueError("VERIFICATION_GEO_SCALE_METERS must be positive.")
        if self.VERIFICATION_DUPLICATE_THRESHOLD <= self.VERIFICATION_GENUINE_THRESHOLD:
            raise ValueError(
                f"VERIFICATION_DUPLICATE_THRESHOLD ({self.VERIFICATION_DUPLICATE_THRESHOLD}) "
                f"must be strictly greater than VERIFICATION_GENUINE_THRESHOLD ({self.VERIFICATION_GENUINE_THRESHOLD})."
            )
        return self

    @computed_field
    @property
    def effective_database_url(self) -> str:
        """
        Returns the active database URL, preferring DATABASE_URL and falling back to POSTGRESQL_URL_RAILWAYS.
        """
        raw_url = self.DATABASE_URL or self.POSTGRESQL_URL_RAILWAYS or ""
        return raw_url.strip()

    @computed_field
    @property
    def sqlalchemy_database_url(self) -> str:
        """
        Ensures the connection string uses the psycopg (v3) driver with SQLAlchemy.
        Converts 'postgres://' or 'postgresql://' to 'postgresql+psycopg://'.
        """
        url = self.effective_database_url
        if not url:
            return ""

        if url.startswith("postgres://"):
            return "postgresql+psycopg://" + url[len("postgres://"):]
        if url.startswith("postgresql://") and not url.startswith("postgresql+"):
            return "postgresql+psycopg://" + url[len("postgresql://"):]
        return url

    @computed_field
    @property
    def masked_database_url(self) -> str:
        """
        Safe-to-log masked representation of the active database URL.
        """
        return mask_database_url(self.effective_database_url)


@lru_cache()
def get_settings() -> Settings:
    return Settings()
