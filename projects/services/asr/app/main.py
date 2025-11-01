"""AI2Text ASR Service - Speech recognition."""
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel

from ai2text_common.observability import setup_tracing, setup_logging

setup_logging("asr")
logger = logging.getLogger(__name__)

# Metrics
transcription_requests_total = Counter("transcription_requests_total", "Total transcription requests")
transcription_duration_seconds = Histogram("transcription_duration_seconds", "Transcription duration")


class TranscribeRequest(BaseModel):
    recording_id: str
    audio_url: str
    language: str
    model_version: str = "latest"


class TranscribeResponse(BaseModel):
    recording_id: str
    status: str
    message: str


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    checks: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting ASR service...")
    setup_tracing("asr")
    # TODO: Load ASR model
    logger.info("ASR service ready")
    yield
    logger.info("Shutting down ASR service...")


app = FastAPI(
    title="AI2Text ASR API",
    version="1.0.0",
    description="Automatic Speech Recognition service",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        timestamp=datetime.utcnow(),
        checks={"model": "ok"}
    )


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest().decode())


@app.post("/transcribe", response_model=TranscribeResponse, status_code=202)
async def transcribe_batch(request: TranscribeRequest) -> TranscribeResponse:
    """Trigger batch transcription."""
    transcription_requests_total.inc()
    
    logger.info(f"Transcription request for {request.recording_id}")
    
    # TODO: Emit to NATS for async processing
    
    return TranscribeResponse(
        recording_id=request.recording_id,
        status="accepted",
        message="Transcription job queued"
    )


@app.websocket("/stream")
async def stream_asr(websocket: WebSocket):
    """Real-time streaming ASR with optimizations for p95 < 500ms E2E."""
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    # Initialize optimizer
    from app.streaming_optimization import StreamingOptimizer, process_streaming_audio_optimized
    
    optimizer = StreamingOptimizer(target_e2e_p95_ms=500)
    config = optimizer.get_optimized_streaming_config()
    
    last_partial_time = time.time()
    
    try:
        while True:
            # Receive audio chunk
            chunk_start = time.time()
            data = await websocket.receive_bytes()
            
            # Model inference function (placeholder)
            async def model_inference(audio_bytes: bytes):
                # TODO: Replace with actual ASR model
                return {
                    "text": "Processing audio...",
                    "is_final": False,
                }
            
            # Process with optimizations
            result = await process_streaming_audio_optimized(
                audio_chunk=data,
                optimizer=optimizer,
                model_inference_fn=model_inference,
            )
            
            # Check if we should send partial (rate limiting)
            if not result["is_final"]:
                if not optimizer.should_send_partial(last_partial_time):
                    continue  # Skip this partial
            
            # Send result
            await websocket.send_json({
                "text": result["text"],
                "is_final": result["is_final"],
                "latency_ms": result["e2e_latency_ms"],
                "meets_slo": result["meets_slo"],
            })
            
            last_partial_time = time.time()
            
            # Log if SLO violated
            if not result["meets_slo"]:
                logger.warning(
                    f"Streaming latency {result['e2e_latency_ms']:.2f}ms "
                    f"exceeds target 500ms"
                )
    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

