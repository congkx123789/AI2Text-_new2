"""Shared API schemas for microservices."""

from .api import (
    AudioMetadata,
    SpeakerMetadata,
    TranscriptionRequest,
    TranscriptionResponse,
    NLPNormalizeRequest,
    NLPNormalizeResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    TrainingRequest,
    TrainingJobResponse,
)

__all__ = [
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
    "SearchResult",
    "TrainingRequest",
    "TrainingJobResponse",
]

