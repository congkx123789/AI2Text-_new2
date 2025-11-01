"""Shared libraries for microservices."""

from .schemas.api import *
from .events.schemas import *
from .observability import wire_otel

__all__ = [
    # API schemas
    "AudioMetadata",
    "SpeakerMetadata",
    "TranscriptionRequest",
    "TranscriptionResponse",
    "NLPNormalizeRequest",
    "NLPNormalizeResponse",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "SearchRequest",
    "SearchResponse",
    # Events
    "CloudEvent",
    "RecordingIngestedEvent",
    "TranscriptionCompletedEvent",
    "NLPPostprocessedEvent",
    "EmbeddingsIndexedEvent",
    "ModelPromotedEvent",
    "EVENT_TYPES",
    "wire_otel",
]

