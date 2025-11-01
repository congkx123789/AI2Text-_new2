"""
Shared API request/response schemas.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from datetime import datetime


# Metadata Service Schemas

class AudioMetadata(BaseModel):
    """Audio metadata model."""
    id: str
    audio_path: str
    transcript: Optional[str] = None
    speaker_id: Optional[str] = None
    snr_estimate: Optional[float] = None
    device_type: Optional[str] = None
    split_assignment: Optional[str] = None  # TRAIN, VAL, TEST
    created_at: datetime


class SpeakerMetadata(BaseModel):
    """Speaker metadata model."""
    id: str
    pseudonymous_id: str  # Never expose PII
    device_types: List[str]
    total_recordings: int


# ASR Service Schemas

class TranscriptionRequest(BaseModel):
    """ASR transcription request."""
    audio_uri: str
    model_name: Optional[str] = "default"
    use_beam_search: bool = True
    beam_width: int = 5
    streaming: bool = False


class TranscriptionResponse(BaseModel):
    """ASR transcription response."""
    audio_id: str
    transcript: str
    confidence: float
    processing_time: float
    transcript_uri: str


# NLP Post-Processing Schemas

class NLPNormalizeRequest(BaseModel):
    """NLP normalization request."""
    text: str
    restore_diacritics: bool = True
    fix_typos: bool = True
    language: str = "vi"


class NLPNormalizeResponse(BaseModel):
    """NLP normalization response."""
    text_clean: str
    text_with_diacritics: str
    corrections: Dict[str, Any]
    confidence: float


# Embeddings Service Schemas

class EmbeddingRequest(BaseModel):
    """Embedding generation request."""
    audio_id: str
    text: Optional[str] = None
    audio_uri: Optional[str] = None
    embedding_type: str = "text"  # "text" or "dvector"


class EmbeddingResponse(BaseModel):
    """Embedding generation response."""
    audio_id: str
    vector_id: str
    vector_type: str
    dimensions: int


# Search Service Schemas

class SearchRequest(BaseModel):
    """Semantic search request."""
    query: str
    limit: int = 10
    filter: Optional[Dict[str, Any]] = None


class SearchResult(BaseModel):
    """Search result."""
    audio_id: str
    transcript: str
    score: float
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    """Search response."""
    results: List[SearchResult]
    total: int
    query_time_ms: float


# Training Orchestrator Schemas

class TrainingRequest(BaseModel):
    """Training job request."""
    dataset_version: str
    model_config: Dict[str, Any]
    hyperparameters: Dict[str, Any]
    priority: str = "normal"  # low, normal, high


class TrainingJobResponse(BaseModel):
    """Training job response."""
    job_id: str
    status: str
    dataset_version: str
    created_at: datetime
    estimated_completion: Optional[datetime] = None

