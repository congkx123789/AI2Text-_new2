"""
Ingestion Service - Upload audio, store in object storage, emit events.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
import nats
import os
import json
import asyncio
import uuid
import subprocess
import glob
from datetime import datetime
from pathlib import Path

from minio import Minio
from minio.error import S3Error

app = FastAPI(title="ingestion", version="0.1.0")
try:
    from libs.common.observability import wire_otel

    wire_otel(app, "ingestion")
except ImportError:
    pass

NATS_URL = os.getenv("NATS_URL", "nats://nats:4222")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000").replace("http://", "").replace("https://", "")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio123")
BUCKET = os.getenv("BUCKET", "audio")


@app.on_event("startup")
async def startup():
    """Initialize connections."""
    app.nc = await nats.connect(NATS_URL)
    app.minio = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )
    if not app.minio.bucket_exists(BUCKET):
        app.minio.make_bucket(BUCKET)
    print(f"[OK] Ingestion service started, bucket: {BUCKET}")
    
    # Start watcher if enabled (Phase A)
    if os.getenv("ENABLE_WATCHER", "false").lower() == "true":
        asyncio.create_task(watch_sources())
        print("[OK] Started background watcher")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup."""
    await app.nc.drain()
    await app.nc.close()


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy", "service": "ingestion"}


@app.post("/v1/ingest")
async def ingest(file: UploadFile = File(...)):
    """
    Ingest audio file.
    
    Uploads to object storage and publishes RecordingIngested event.
    """
    audio_id = str(uuid.uuid4())
    obj_name = f"raw/{audio_id}.wav"
    
    try:
        # Upload to MinIO
        app.minio.put_object(
            BUCKET,
            obj_name,
            file.file,
            length=-1,
            part_size=10*1024*1024,
            content_type=file.content_type or "audio/wav"
        )
    except S3Error as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    
    # Create CloudEvent
    event = {
        "specversion": "1.0",
        "id": str(uuid.uuid4()),
        "source": "services/ingestion",
        "type": "RecordingIngested",
        "time": datetime.utcnow().isoformat() + "Z",
        "datacontenttype": "application/json",
        "data": {
            "audio_id": audio_id,
            "path": f"s3://{BUCKET}/{obj_name}"
        }
    }
    
    # Publish event
    await app.nc.publish("recording.ingested", json.dumps(event).encode())
    
    return {
        "audio_id": audio_id,
        "object_uri": event["data"]["path"]
    }


# Phase A: Auto-Ingestion Watcher
# ================================

HOT_FOLDER = os.getenv("HOT_FOLDER", "/mnt/inbox")
PROCESSED_FOLDER = os.getenv("PROCESSED_FOLDER", "/mnt/processed")
S3_INBOX_PREFIX = os.getenv("S3_INBOX_PREFIX", "inbox/")
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL_SEC", "30"))


async def watch_sources():
    """
    Background task to watch hot folder and S3 inbox for new audio files.
    
    Monitors:
    1. Local hot folder (e.g., /mnt/inbox/**/*.{wav,mp3})
    2. S3/MinIO inbox prefix (e.g., s3://audio/inbox/*.{wav,mp3})
    
    On detect:
    - Transcode .mp3 → .wav (16kHz mono PCM)
    - Upload to s3://audio/raw/{audio_id}.wav
    - Publish recording.ingested event
    - Move/delete source file
    """
    print(f"[Watcher] Starting source monitoring")
    print(f"  Hot folder: {HOT_FOLDER}")
    print(f"  S3 inbox: s3://{BUCKET}/{S3_INBOX_PREFIX}")
    print(f"  Scan interval: {SCAN_INTERVAL}s")
    
    # Ensure folders exist
    Path(HOT_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(PROCESSED_FOLDER).mkdir(parents=True, exist_ok=True)
    
    while True:
        try:
            # 1. Scan local hot folder
            await scan_hot_folder()
            
            # 2. Scan S3 inbox
            await scan_s3_inbox()
            
        except Exception as e:
            print(f"[ERROR] Watcher error: {e}")
            import traceback
            traceback.print_exc()
        
        # Sleep until next scan
        await asyncio.sleep(SCAN_INTERVAL)


async def scan_hot_folder():
    """Scan local hot folder for audio files."""
    # Find all audio files
    for ext in ["wav", "mp3", "m4a", "flac"]:
        pattern = os.path.join(HOT_FOLDER, f"**/*.{ext}")
        files = glob.glob(pattern, recursive=True)
        
        for file_path in files:
            try:
                await handle_local_file(file_path)
            except Exception as e:
                print(f"[ERROR] Failed to process {file_path}: {e}")


async def scan_s3_inbox():
    """Scan S3/MinIO inbox prefix for audio files."""
    try:
        objects = app.minio.list_objects(BUCKET, prefix=S3_INBOX_PREFIX, recursive=True)
        
        for obj in objects:
            # Check if audio file
            if any(obj.object_name.endswith(ext) for ext in [".wav", ".mp3", ".m4a", ".flac"]):
                try:
                    await handle_s3_file(obj.object_name)
                except Exception as e:
                    print(f"[ERROR] Failed to process s3://{BUCKET}/{obj.object_name}: {e}")
                    
    except Exception as e:
        print(f"[ERROR] Failed to scan S3 inbox: {e}")


async def handle_local_file(file_path: str):
    """
    Process a local audio file.
    
    Steps:
    1. Generate audio_id
    2. Normalize to 16kHz mono WAV
    3. Upload to S3
    4. Publish event
    5. Move to processed folder
    """
    print(f"[Watcher] Processing local file: {file_path}")
    
    # Generate ID
    audio_id = str(uuid.uuid4())
    orig_filename = os.path.basename(file_path)
    
    # Normalize audio
    wav_path = await transcode_to_wav_16k_mono(file_path, audio_id)
    
    # Upload to S3
    obj_key = f"raw/{audio_id}.wav"
    with open(wav_path, 'rb') as f:
        app.minio.put_object(
            BUCKET,
            obj_key,
            f,
            length=os.path.getsize(wav_path),
            content_type="audio/wav"
        )
    
    print(f"[OK] Uploaded to s3://{BUCKET}/{obj_key}")
    
    # Publish event
    event = {
        "specversion": "1.0",
        "id": str(uuid.uuid4()),
        "source": "services/ingestion",
        "type": "RecordingIngested",
        "time": datetime.utcnow().isoformat() + "Z",
        "datacontenttype": "application/json",
        "data": {
            "audio_id": audio_id,
            "path": f"s3://{BUCKET}/{obj_key}",
            "original_filename": orig_filename,
            "source": "hot_folder"
        }
    }
    
    await app.nc.publish("recording.ingested", json.dumps(event).encode())
    print(f"[OK] Published recording.ingested for {audio_id}")
    
    # Move to processed folder
    processed_path = os.path.join(PROCESSED_FOLDER, orig_filename)
    os.rename(file_path, processed_path)
    print(f"[OK] Moved to {processed_path}")
    
    # Cleanup temp file
    if os.path.exists(wav_path) and wav_path != file_path:
        os.remove(wav_path)


async def handle_s3_file(object_name: str):
    """
    Process an S3 inbox file.
    
    Steps:
    1. Download from inbox
    2. Process same as local file
    3. Delete from inbox
    """
    print(f"[Watcher] Processing S3 file: s3://{BUCKET}/{object_name}")
    
    # Download to temp file
    temp_path = f"/tmp/{os.path.basename(object_name)}"
    app.minio.fget_object(BUCKET, object_name, temp_path)
    
    try:
        # Process as local file
        await handle_local_file(temp_path)
        
        # Delete from inbox
        app.minio.remove_object(BUCKET, object_name)
        print(f"[OK] Removed from s3://{BUCKET}/{object_name}")
        
    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


async def transcode_to_wav_16k_mono(input_path: str, audio_id: str) -> str:
    """
    Transcode audio to 16kHz mono WAV using ffmpeg.
    
    Args:
        input_path: Path to input audio file (.mp3, .wav, etc.)
        audio_id: Unique identifier for output file
    
    Returns:
        Path to output WAV file
    """
    output_path = f"/tmp/{audio_id}.wav"
    
    # ffmpeg command for 16kHz mono PCM16
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-ar", "16000",        # 16kHz sample rate
        "-ac", "1",            # Mono
        "-sample_fmt", "s16",  # PCM 16-bit
        "-y",                  # Overwrite output
        output_path
    ]
    
    # Run ffmpeg
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")
    
    print(f"[OK] Transcoded to 16kHz mono WAV: {output_path}")
    return output_path


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
