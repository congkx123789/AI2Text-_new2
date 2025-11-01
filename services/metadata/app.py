"""
Metadata Service - ACID metadata store for audio/transcripts.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import os
import psycopg2
import psycopg2.extras
from datetime import datetime
from uuid import UUID
import httpx
import nats
import asyncio
import json

DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/asrmeta")
NLP_POST_URL = os.getenv("NLP_POST_URL", "http://nlp-post:8000")
NATS_URL = os.getenv("NATS_URL", "nats://nats:4222")

app = FastAPI(title="metadata", version="0.1.0")
try:
    from libs.common.observability import wire_otel

    wire_otel(app, "metadata")
except ImportError:
    pass

# Global NATS connection
nc = None


def get_db_connection():
    """Get database connection."""
    return psycopg2.connect(DB_URL)


# Request/Response Models

class AudioMetadata(BaseModel):
    """Audio metadata model."""
    audio_id: str
    speaker_id: Optional[str] = None
    audio_path: str
    snr_estimate: Optional[float] = None
    device_type: Optional[str] = None
    environment: Optional[str] = None
    split_assignment: str  # TRAIN, VAL, TEST
    duration_seconds: Optional[float] = None
    sample_rate: Optional[int] = None


class TranscriptResponse(BaseModel):
    """Transcript response."""
    audio_id: str
    text: Optional[str] = None
    text_clean: Optional[str] = None
    raw_json: Optional[dict] = None


async def handle_transcription_completed(msg):
    """Handle transcription.completed events and store in database."""
    try:
        evt = json.loads(msg.data.decode())
        audio_id = evt["data"]["audio_id"]
        
        print(f"[Metadata] Received transcription for {audio_id}")
        
        # TODO: Download transcript from object store if needed
        # For now, we'll wait for the NLP service to process it
        
    except Exception as e:
        print(f"[ERROR] Failed to handle transcription: {e}")


async def handle_nlp_postprocessed(msg):
    """Handle nlp.postprocessed events and update database."""
    try:
        evt = json.loads(msg.data.decode())
        audio_id = evt["data"]["audio_id"]
        text = evt["data"].get("text_clean", "")
        text_clean = evt["data"].get("text_with_diacritics", text)
        
        print(f"[Metadata] Updating {audio_id} with NLP-processed text")
        
        # Update database with cleaned text
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO transcripts (audio_id, text, text_clean)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (audio_id) 
                    DO UPDATE SET text = EXCLUDED.text,
                                 text_clean = EXCLUDED.text_clean,
                                 updated_at = now()
                    """,
                    (audio_id, text, text_clean)
                )
                conn.commit()
        
        print(f"[OK] Metadata updated for {audio_id}")
        
    except Exception as e:
        print(f"[ERROR] Failed to update metadata: {e}")


@app.on_event("startup")
async def startup():
    """Initialize database and event subscriptions."""
    global nc
    
    # Connect to NATS
    try:
        nc = await nats.connect(NATS_URL)
        await nc.subscribe("transcription.completed", cb=handle_transcription_completed)
        await nc.subscribe("nlp.postprocessed", cb=handle_nlp_postprocessed)
        print(f"[OK] Subscribed to transcription and NLP events")
    except Exception as e:
        print(f"[WARNING] Could not connect to NATS: {e}")
    
    print("[OK] Metadata service started")


@app.get("/health")
async def health():
    """Health check."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        return {"status": "healthy", "service": "metadata", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.get("/v1/audio/{audio_id}")
async def get_audio(audio_id: str):
    """Get audio metadata."""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT audio_id, speaker_id, audio_path, snr_estimate, device_type, "
                "environment, split_assignment, duration_seconds, sample_rate "
                "FROM audio WHERE audio_id = %s",
                (audio_id,)
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Audio not found")
            return dict(row)


@app.get("/v1/transcripts/{audio_id}", response_model=TranscriptResponse)
async def get_transcript(audio_id: str):
    """Get transcript for audio."""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT audio_id, text, text_clean, raw_json FROM transcripts WHERE audio_id = %s",
                (audio_id,)
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Transcript not found")
            return TranscriptResponse(
                audio_id=str(row['audio_id']),
                text=row.get('text'),
                text_clean=row.get('text_clean'),
                raw_json=row.get('raw_json')
            )


@app.get("/v1/speakers/{speaker_id}")
async def get_speaker(speaker_id: str):
    """Get speaker metadata."""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT speaker_id, pseudonymous_id, region, device_types, total_recordings, created_at "
                "FROM speakers WHERE speaker_id = %s",
                (speaker_id,)
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Speaker not found")
            return dict(row)


@app.post("/v1/audio")
async def create_audio(metadata: AudioMetadata):
    """Create audio metadata record."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    """
                    INSERT INTO audio (audio_id, speaker_id, audio_path, snr_estimate, 
                                    device_type, environment, split_assignment, duration_seconds, sample_rate)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::split_enum, %s, %s)
                    """,
                    (
                        metadata.audio_id,
                        metadata.speaker_id,
                        metadata.audio_path,
                        metadata.snr_estimate,
                        metadata.device_type,
                        metadata.environment,
                        metadata.split_assignment,
                        metadata.duration_seconds,
                        metadata.sample_rate
                    )
                )
                conn.commit()
                return {"status": "created", "audio_id": metadata.audio_id}
            except psycopg2.IntegrityError as e:
                conn.rollback()
                raise HTTPException(status_code=400, detail=f"Database error: {str(e)}")


@app.put("/v1/transcripts/{audio_id}")
async def update_transcript(audio_id: str, text: str, text_clean: Optional[str] = None, raw_json: Optional[dict] = None):
    """Update transcript for audio and trigger NLP post-processing."""
    import json as json_lib
    
    # If text_clean is not provided, call NLP service
    if not text_clean and text:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{NLP_POST_URL}/v1/nlp/normalize",
                    json={
                        "text": text,
                        "restore_diacritics": True,
                        "fix_typos": True
                    }
                )
                if response.status_code == 200:
                    nlp_result = response.json()
                    text_clean = nlp_result.get("text_with_diacritics", text)
                    print(f"[OK] NLP processed text for {audio_id}")
                else:
                    print(f"[WARNING] NLP service returned {response.status_code}, using original text")
                    text_clean = text
        except Exception as e:
            print(f"[WARNING] Could not call NLP service: {e}, using original text")
            text_clean = text
    
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO transcripts (audio_id, text, text_clean, raw_json)
                VALUES (%s, %s, %s, %s::jsonb)
                ON CONFLICT (audio_id) 
                DO UPDATE SET text = EXCLUDED.text, 
                             text_clean = EXCLUDED.text_clean,
                             raw_json = EXCLUDED.raw_json,
                             updated_at = now()
                """,
                (audio_id, text, text_clean, json_lib.dumps(raw_json) if raw_json else None)
            )
            conn.commit()
            return {"status": "updated", "audio_id": audio_id, "text_clean": text_clean}


@app.on_event("shutdown")
async def shutdown():
    """Cleanup."""
    global nc
    if nc:
        await nc.drain()
        await nc.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
