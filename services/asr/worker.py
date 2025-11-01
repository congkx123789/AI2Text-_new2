"""
ASR Worker - Subscribes to recording.ingested events and transcribes.

This worker:
1. Listens for recording.ingested events from NATS
2. Downloads audio from MinIO/S3
3. Runs ASR (currently stub, replace with real model)
4. Uploads transcript to MinIO/S3
5. Publishes transcription.completed event
"""

import nats
import asyncio
import json
import os
import uuid
import io
from datetime import datetime
from minio import Minio
from minio.error import S3Error

# Configuration
NATS_URL = os.getenv("NATS_URL", "nats://nats:4222")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000").replace("http://", "").replace("https://", "")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio123")
RAW_BUCKET = os.getenv("BUCKET", "audio")
TRANSCRIPT_BUCKET = os.getenv("TRANSCRIPT_BUCKET", "transcripts")

# Global clients
nc = None
s3_client = None

# TODO: Import and load your ASR model here
# from models.lstm_asr import LSTMASRModel
# asr_model = LSTMASRModel.load_checkpoint("checkpoints/best_model.pt")
# 
# Or use Whisper:
# import whisper
# asr_model = whisper.load_model("base")
#
# Or use FastConformer/Conformer for streaming


async def transcribe_audio(audio_data: bytes, audio_id: str) -> dict:
    """
    Transcribe audio data.
    
    TODO: Replace this stub with real ASR inference.
    For now, returns a placeholder transcript so the pipeline is end-to-end.
    
    Integration points:
    - Whisper: result = asr_model.transcribe(audio_data)
    - Your LSTM: result = asr_model.transcribe(audio_data)
    - FastConformer: Use NeMo for streaming
    """
    # Placeholder Vietnamese transcript
    text = "xin chào thế giới"
    
    # Return format matches typical ASR output
    return {
        "audio_id": audio_id,
        "text": text,
        "segments": [
            {
                "start_ms": 0,
                "end_ms": 800,
                "text": text,
                "confidence": 0.95
            }
        ],
        "language": "vi",
        "model_version": "stub-1.0"
    }


async def handle_recording_ingested(msg):
    """Handle recording.ingested events."""
    try:
        evt = json.loads(msg.data.decode())
        audio_id = evt["data"]["audio_id"]
        path = evt["data"]["path"]  # s3://audio/raw/{audio_id}.wav
        
        print(f"[ASR] Processing {audio_id} from {path}")
        
        # Extract object key from S3 path
        # path format: s3://bucket/key
        parts = path.replace("s3://", "").split("/", 1)
        if len(parts) == 2:
            bucket, key = parts
        else:
            bucket, key = RAW_BUCKET, f"raw/{audio_id}.wav"
        
        # Download audio from object storage
        try:
            response = s3_client.get_object(bucket, key)
            audio_data = response.read()
            response.close()
            response.release_conn()
            print(f"[ASR] Downloaded {len(audio_data)} bytes from {bucket}/{key}")
        except S3Error as e:
            print(f"[ERROR] Failed to download audio: {e}")
            return
        
        # Run ASR
        result = await transcribe_audio(audio_data, audio_id)
        print(f"[ASR] Transcribed: {result['text']}")
        
        # Upload transcript to object storage
        transcript_json = json.dumps(result, ensure_ascii=False, indent=2)
        transcript_bytes = transcript_json.encode("utf-8")
        buf = io.BytesIO(transcript_bytes)
        
        transcript_key = f"transcripts/{audio_id}.json"
        s3_client.put_object(
            TRANSCRIPT_BUCKET,
            transcript_key,
            buf,
            length=len(transcript_bytes),
            content_type="application/json"
        )
        
        transcript_uri = f"s3://{TRANSCRIPT_BUCKET}/{transcript_key}"
        print(f"[ASR] Uploaded transcript to {transcript_uri}")
        
        # Publish TranscriptionCompleted event
        out_event = {
            "specversion": "1.0",
            "id": str(uuid.uuid4()),
            "source": "services/asr",
            "type": "TranscriptionCompleted",
            "time": datetime.utcnow().isoformat() + "Z",
            "datacontenttype": "application/json",
            "data": {
                "audio_id": audio_id,
                "transcript_uri": transcript_uri,
                "text": result["text"],
                "wer_baseline": None,
                "model_version": result["model_version"],
                "processing_time_seconds": 0.1
            }
        }
        
        await nc.publish("transcription.completed", json.dumps(out_event, ensure_ascii=False).encode())
        print(f"[OK] ASR completed for {audio_id}")
        
    except Exception as e:
        print(f"[ERROR] ASR processing failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main worker loop."""
    global nc, s3_client
    
    # Connect to NATS
    nc = await nats.connect(NATS_URL)
    print(f"[OK] Connected to NATS at {NATS_URL}")
    
    # Connect to MinIO/S3
    s3_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False  # Set to True for production S3
    )
    
    # Ensure buckets exist
    for bucket in (RAW_BUCKET, TRANSCRIPT_BUCKET):
        try:
            if not s3_client.bucket_exists(bucket):
                s3_client.make_bucket(bucket)
                print(f"[OK] Created bucket: {bucket}")
            else:
                print(f"[OK] Bucket exists: {bucket}")
        except S3Error as e:
            print(f"[WARNING] Bucket check failed for {bucket}: {e}")
    
    # Subscribe to events
    await nc.subscribe("recording.ingested", cb=handle_recording_ingested)
    print("[OK] ASR worker listening for recording.ingested events...")
    print("[INFO] Waiting for audio to transcribe...")
    
    try:
        # Keep worker alive
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("[INFO] Shutting down...")
    finally:
        await nc.drain()
        await nc.close()


if __name__ == "__main__":
    asyncio.run(main())

