"""FastAPI dependencies for AI2Text service."""
import os
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Service configuration
    service_name: str = "service-name"
    log_level: str = "INFO"
    
    # NATS configuration
    nats_url: str = os.getenv("NATS_URL", "nats://localhost:4222")
    
    # Observability
    otlp_endpoint: str = os.getenv("OTLP_ENDPOINT", "http://localhost:4317")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

