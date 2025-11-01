"""Dependencies for search service."""
import os
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    service_name: str = "search"
    log_level: str = "INFO"
    
    # Qdrant
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "transcripts")
    
    # Observability
    otlp_endpoint: str = os.getenv("OTLP_ENDPOINT", "http://localhost:4317")
    
    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings."""
    return Settings()

