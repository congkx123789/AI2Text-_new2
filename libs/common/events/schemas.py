"""
Event schemas for microservices communication.

Follows CloudEvents JSON format for interoperability.
"""

from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel
import uuid


class CloudEvent(BaseModel):
    """CloudEvents-compliant event structure."""
    id: str
    source: str
    type: str
    specversion: str = "1.0"
    time: str
    datacontenttype: str = "application/json"
    data: Dict[str, Any]
    
    @classmethod
    def create(cls, source: str, event_type: str, data: Dict[str, Any]) -> "CloudEvent":
        """Create a CloudEvent."""
        return cls(
            id=str(uuid.uuid4()),
            source=source,
            type=event_type,
            time=datetime.utcnow().isoformat() + "Z",
            data=data
        )


# Event Definitions

class RecordingIngestedEvent(BaseModel):
    """Event: Audio file ingested and stored."""
    audio_id: str
    object_uri: str  # s3://bucket/path
    speaker_id: Optional[str] = None
    device_type: Optional[str] = None
    snr_estimate: Optional[float] = None
    file_size_bytes: int
    duration_seconds: Optional[float] = None


class TranscriptionCompletedEvent(BaseModel):
    """Event: ASR transcription completed."""
    audio_id: str
    transcript_uri: str  # s3://bucket/transcripts/{id}.json
    wer_baseline: Optional[float] = None
    model_version: Optional[str] = None
    processing_time_seconds: float


class NLPPostprocessedEvent(BaseModel):
    """Event: NLP post-processing completed."""
    audio_id: str
    text_clean: str
    text_with_diacritics: str
    corrections: Dict[str, Any]
    output_uri: str


class EmbeddingsIndexedEvent(BaseModel):
    """Event: Embeddings indexed in vector DB."""
    audio_id: str
    vector_id: str
    vector_type: str  # "text" or "dvector"
    embedding_model: str


class ModelPromotedEvent(BaseModel):
    """Event: Model version promoted."""
    model_name: str
    version: str
    artifact_uri: str
    metrics: Dict[str, float]


# Event type constants
EVENT_TYPES = {
    "RecordingIngested": "recording.ingested",
    "TranscriptionCompleted": "transcription.completed",
    "NLPPostprocessed": "nlp.postprocessed",
    "EmbeddingsIndexed": "embeddings.indexed",
    "ModelPromoted": "model.promoted",
}

