"""
ASR Service - Batch transcription API endpoint.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os

app = FastAPI(title="asr", version="0.1.0")
try:
    from libs.common.observability import wire_otel

    wire_otel(app, "asr")
except ImportError:
    pass


class TranscriptionRequest(BaseModel):
    """Transcription request."""
    audio_uri: str
    model_name: Optional[str] = "default"
    use_beam_search: Optional[bool] = True
    beam_width: Optional[int] = 5


class TranscriptionResponse(BaseModel):
    """Transcription response."""
    audio_id: str
    transcript: str
    transcript_uri: str
    status: str = "completed"


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy", "service": "asr"}


@app.post("/v1/asr/transcribe", response_model=TranscriptionResponse)
async def transcribe(request: TranscriptionRequest):
    """
    Batch transcription endpoint.
    
    TODO: Download audio from object store, run ASR, upload transcript.
    For now, returns placeholder.
    """
    # Extract audio_id from URI
    audio_id = request.audio_uri.split("/")[-1].split(".")[0]
    
    # TODO: Download from MinIO and transcribe
    # TODO: Upload transcript to object store
    
    transcript_uri = f"s3://{os.getenv('BUCKET', 'audio')}/transcripts/{audio_id}.json"
    
    return TranscriptionResponse(
        audio_id=audio_id,
        transcript="xin chào việt nam",  # Placeholder
        transcript_uri=transcript_uri,
        status="completed"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
