"""
Event payload schemas matching AsyncAPI contracts
"""

from datetime import datetime
from typing import Optional, List
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl


class RecordingIngested(BaseModel):
    """Recording ingested event payload"""

    recording_id: UUID = Field(..., description="Unique recording identifier")
    audio_url: HttpUrl = Field(..., description="URL to stored audio file")
    language: str = Field(..., pattern="^(vi|en)$", description="Language code")
    duration_sec: Optional[float] = Field(None, description="Audio duration in seconds")
    file_size_bytes: Optional[int] = Field(None, ge=0, description="File size in bytes")
    mime_type: Optional[str] = Field(None, description="MIME type")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Event timestamp")


class TranscriptionCompleted(BaseModel):
    """Transcription completed event payload"""

    recording_id: UUID = Field(..., description="Recording identifier")
    text: str = Field(..., description="Raw transcription text")
    language: str = Field(..., pattern="^(vi|en)$", description="Language code")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Transcription confidence")
    duration_sec: Optional[float] = Field(None, description="Audio duration processed")
    processing_time_sec: Optional[float] = Field(None, description="Processing time")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Event timestamp")


class NLPChange(BaseModel):
    """NLP processing change record"""

    type: str = Field(..., pattern="^(diacritics|normalization|typo|spacing)$", description="Change type")
    original: str = Field(..., description="Original text segment")
    corrected: str = Field(..., description="Corrected text segment")
    position: Optional[int] = Field(None, ge=0, description="Character position")


class NLPPostprocessed(BaseModel):
    """NLP post-processed event payload"""

    recording_id: UUID = Field(..., description="Recording identifier")
    text_original: str = Field(..., description="Text before NLP processing")
    text_processed: str = Field(..., description="Text after NLP processing")
    language: str = Field(default="vi", pattern="^(vi|en)$", description="Language code")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="NLP confidence")
    changes: Optional[List[NLPChange]] = Field(default_factory=list, description="List of changes")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Event timestamp")


class EmbeddingsCreated(BaseModel):
    """Embeddings created event payload"""

    recording_id: UUID = Field(..., description="Recording identifier")
    vector_id: str = Field(..., description="Qdrant vector ID")
    text: Optional[str] = Field(None, description="Processed text that was embedded")
    embedding_dim: Optional[int] = Field(None, description="Embedding dimension")
    collection: str = Field(default="transcripts", description="Qdrant collection name")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Event timestamp")


class ModelMetrics(BaseModel):
    """Model performance metrics"""

    wer: Optional[float] = Field(None, ge=0.0, description="Word Error Rate")
    cer: Optional[float] = Field(None, ge=0.0, description="Character Error Rate")


class ModelPromoted(BaseModel):
    """Model promoted event payload"""

    model_id: str = Field(..., description="Unique model identifier")
    model_version: str = Field(..., description="Semantic version")
    model_url: HttpUrl = Field(..., description="URL to model artifact")
    metrics: Optional[ModelMetrics] = Field(None, description="Performance metrics")
    training_config: Optional[dict] = Field(None, description="Training configuration")
    promoted_by: Optional[str] = Field(None, description="User/system that promoted")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Event timestamp")


