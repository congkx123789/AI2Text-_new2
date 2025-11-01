"""
Shared Pydantic schemas
"""

from ai2text_common.schemas.common import (
    HealthResponse,
    ErrorResponse,
)
from ai2text_common.schemas.events import (
    RecordingIngested,
    TranscriptionCompleted,
    NLPPostprocessed,
    EmbeddingsCreated,
    ModelPromoted,
)

__all__ = [
    "HealthResponse",
    "ErrorResponse",
    "RecordingIngested",
    "TranscriptionCompleted",
    "NLPPostprocessed",
    "EmbeddingsCreated",
    "ModelPromoted",
]


