"""
Ingestion Service - Handles audio file uploads and storage
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
import uuid
from datetime import datetime
from pathlib import Path

from minio import Minio
from minio.error import S3Error

from ai2text_common import setup_logging, setup_metrics, HealthResponse
from ai2text_common.events import CloudEventsHelper, RecordingIngested
from ai2text_common.observability.metrics import generate_metrics_response
from fastapi.responses import Response
import nats

logger = setup_logging("ingestion")
metrics = setup_metrics("ingestion")

app = FastAPI(
    title="AI2Text Ingestion Service",
    version="1.0.0",
    description="Audio file upload and storage service",
)

# MinIO client
minio_client = Minio(
    os.getenv("MINIO_ENDPOINT", "localhost:9000"),
    access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
    secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
    secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
)
BUCKET_NAME = os.getenv("MINIO_BUCKET", "ai2text-audio")

# NATS client
nats_url = os.getenv("NATS_URL", "nats://localhost:4222")
nats_client = None


@app.on_event("startup")
async def startup():
    """Initialize services on startup"""
    global nats_client
    
    # Ensure bucket exists
    try:
        if not minio_client.bucket_exists(BUCKET_NAME):
            minio_client.make_bucket(BUCKET_NAME)
            logger.info(f"Created bucket: {BUCKET_NAME}")
    except S3Error as e:
        logger.error(f"Error creating bucket: {e}")
    
    # Connect to NATS
    try:
        nats_client = await nats.connect(nats_url)
        logger.info(f"Connected to NATS at {nats_url}")
    except Exception as e:
        logger.error(f"Failed to connect to NATS: {e}")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    if nats_client:
        await nats_client.close()


@app.get("/health")
async def health():
    """Health check endpoint"""
    return HealthResponse(status="healthy")


@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint"""
    from fastapi.responses import Response
    return Response(content=generate_metrics_response(), media_type="text/plain")


@app.post("/ingest")
async def ingest_audio(
    file: UploadFile = File(...),
    language: str = "vi",
):
    """
    Upload audio file for processing
    
    Args:
        file: Audio file (WAV, MP3, FLAC)
        language: Language code (vi, en)
    
    Returns:
        Recording ID and status
    """
    if language not in ["vi", "en"]:
        raise HTTPException(status_code=400, detail="Invalid language code")
    
    recording_id = uuid.uuid4()
    
    try:
        # Save file to MinIO
        file_extension = Path(file.filename).suffix
        object_name = f"raw/{recording_id}{file_extension}"
        
        # Read file content
        content = await file.read()
        file_size = len(content)
        
        # Upload to MinIO
        from io import BytesIO
        minio_client.put_object(
            BUCKET_NAME,
            object_name,
            BytesIO(content),
            length=file_size,
            content_type=file.content_type,
        )
        
        # Build audio URL
        audio_url = f"s3://{BUCKET_NAME}/{object_name}"
        # Or: f"http://{os.getenv('MINIO_ENDPOINT')}/{BUCKET_NAME}/{object_name}"
        
        # Create event
        event_data = RecordingIngested(
            recording_id=recording_id,
            audio_url=audio_url,
            language=language,
            file_size_bytes=file_size,
            mime_type=file.content_type,
        )
        
        event = CloudEventsHelper.create(
            type="recording.ingested.v1",
            source="ingestion-service",
            data=event_data,
        )
        
        # Publish event
        if nats_client:
            await nats_client.publish(
                "recording.ingested.v1",
                CloudEventsHelper.to_json(event),
            )
            logger.info(f"Published recording.ingested.v1 for {recording_id}")
        
        return {
            "recording_id": str(recording_id),
            "status": "accepted",
            "message": "File uploaded successfully",
        }
        
    except Exception as e:
        logger.error(f"Error ingesting file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to ingest file")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

