"""
AI2Text Common - Shared library for microservices
"""

__version__ = "0.1.0"

from ai2text_common.events import CloudEventsHelper
from ai2text_common.observability import setup_logging, setup_tracing, setup_metrics
from ai2text_common.schemas import (
    HealthResponse,
    ErrorResponse,
    RecordingIngested,
    TranscriptionCompleted,
    NLPPostprocessed,
    EmbeddingsCreated,
    ModelPromoted,
)

__all__ = [
    # Events
    "CloudEventsHelper",
    # Observability
    "setup_logging",
    "setup_tracing",
    "setup_metrics",
    # Schemas
    "HealthResponse",
    "ErrorResponse",
    "RecordingIngested",
    "TranscriptionCompleted",
    "NLPPostprocessed",
    "EmbeddingsCreated",
    "ModelPromoted",
]


