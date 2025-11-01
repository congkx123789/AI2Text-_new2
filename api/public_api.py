"""
Public API interface - sanitized for frontend use.

This module provides a clean public API interface without exposing
internal implementation details.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel
from typing import Optional
import time
from pathlib import Path
import sys

# Import internal API (but hide implementation)
sys.path.append(str(Path(__file__).parent.parent))

from api.app import (
    transcribe_audio_internal,
    get_models_internal,
    check_health_internal
)

app = FastAPI(
    title="Vietnamese ASR Public API",
    description="Public REST API for Vietnamese Speech-to-Text system",
    version="1.0.0",
    docs_url="/api/docs",  # Custom docs path
    redoc_url="/api/redoc"
)

# Security headers
@app.middleware("http")
async def security_headers(request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    # Don't expose server details
    response.headers["Server"] = "ASR-API"
    return response

# CORS middleware - configure for your frontend origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        # Add production origins here
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
    expose_headers=["X-Request-ID"],
)

# Public request/response models (no internal details)
class PublicTranscriptionResponse(BaseModel):
    """Public transcription response - sanitized."""
    text: str
    confidence: float
    processing_time: float
    # Note: model_name removed for security


class PublicModelInfo(BaseModel):
    """Public model information - sanitized."""
    name: str
    available: bool
    # Note: path, size removed for security


@app.get("/")
async def root():
    """Root endpoint - public information only."""
    return {
        "service": "Vietnamese ASR API",
        "version": "1.0.0",
        "endpoints": {
            "transcribe": "POST /transcribe",
            "models": "GET /models",
            "health": "GET /health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint - no internal details."""
    try:
        result = await check_health_internal()
        # Sanitize response
        return {
            "status": "healthy" if result.get("status") == "healthy" else "unhealthy",
            "timestamp": int(time.time())
        }
    except Exception:
        return {
            "status": "unhealthy",
            "timestamp": int(time.time())
        }


@app.get("/models", response_model=List[PublicModelInfo])
async def list_models_public():
    """List available models - sanitized information only."""
    try:
        models_data = await get_models_internal()
        
        # Sanitize model information
        public_models = []
        for model in models_data.get("models", []):
            public_models.append({
                "name": model.get("name", "unknown"),
                "available": True
            })
        
        return public_models
    except Exception as e:
        # Don't expose error details
        return []


@app.post("/transcribe", response_model=PublicTranscriptionResponse)
async def transcribe_audio_public(
    audio: UploadFile = File(...),
    model_name: Optional[str] = None,
    use_beam_search: bool = True,
    beam_width: int = 5,
    use_lm: bool = False,
    min_confidence: Optional[float] = None
):
    """
    Transcribe audio file - public endpoint.
    
    Only returns sanitized public information.
    """
    try:
        # Call internal function
        result = await transcribe_audio_internal(
            audio=audio,
            model_name=model_name or "default",
            use_beam_search=use_beam_search,
            beam_width=beam_width,
            use_lm=use_lm,
            min_confidence=min_confidence
        )
        
        # Sanitize response - remove internal details
        return PublicTranscriptionResponse(
            text=result.get("text", ""),
            confidence=result.get("confidence", 0.0),
            processing_time=result.get("processing_time", 0.0)
        )
    except Exception as e:
        # Don't expose internal error details
        raise HTTPException(
            status_code=500,
            detail="Transcription failed. Please try again."
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

