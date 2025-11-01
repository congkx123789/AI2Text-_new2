"""Dependencies for metadata service."""
import os
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    service_name: str = "metadata"
    log_level: str = "INFO"
    
    # Database
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:postgres@localhost:5432/asrmeta"
    )
    
    # NATS
    nats_url: str = os.getenv("NATS_URL", "nats://localhost:4222")
    
    # Observability
    otlp_endpoint: str = os.getenv("OTLP_ENDPOINT", "http://localhost:4317")
    
    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings."""
    return Settings()

