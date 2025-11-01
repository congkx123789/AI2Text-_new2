"""Dependencies for ASR service."""
import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    service_name: str = "asr"
    log_level: str = "INFO"
    model_path: str = os.getenv("MODEL_PATH", "/models/asr")
    nats_url: str = os.getenv("NATS_URL", "nats://localhost:4222")
    otlp_endpoint: str = os.getenv("OTLP_ENDPOINT", "http://localhost:4317")


@lru_cache()
def get_settings() -> Settings:
    return Settings()

