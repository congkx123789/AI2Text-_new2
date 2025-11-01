"""
NLP Post-Processing Service - Vietnamese text normalization.
Diacritics restoration and typo correction.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import nats
import asyncio
import json
import os
import uuid
from datetime import datetime

app = FastAPI(title="nlp-post", version="0.1.0")
try:
    from libs.common.observability import wire_otel

    wire_otel(app, "nlp-post")
except ImportError:
    pass

NATS_URL = os.getenv("NATS_URL", "nats://nats:4222")

# Global NATS connection
nc = None


class NormalizeRequest(BaseModel):
    """NLP normalization request."""
    text: str
    restore_diacritics: bool = True
    fix_typos: bool = True


class NormalizeResponse(BaseModel):
    """NLP normalization response."""
    text_clean: str
    text_with_diacritics: str
    corrections: List[Dict[str, Any]]


async def normalize_vietnamese_text(text: str, restore_diacritics: bool = True, fix_typos: bool = True) -> NormalizeResponse:
    """
    Normalize Vietnamese text with diacritics restoration and typo correction.
    
    This is a placeholder implementation. In production:
    - Use ByT5-style seq2seq model for joint diacritics restoration + typo correction
    - Or use underthesea library for Vietnamese NLP tasks
    - Or fine-tune mBERT/PhoBERT for Vietnamese text normalization
    
    Common Vietnamese diacritics issues:
    - "chao" → "chào" (greeting)
    - "viet nam" → "việt nam" (country name)
    - "cam on" → "cảm ơn" (thank you)
    """
    # Basic normalization
    text_clean = text.strip()
    
    # Simple rule-based corrections for common Vietnamese words
    # TODO: Replace with proper ML model
    diacritics_map = {
        "xin chao": "xin chào",
        "chao": "chào",
        "viet nam": "việt nam",
        "cam on": "cảm ơn",
        "hen gap lai": "hẹn gặp lại",
        "tam biet": "tạm biệt",
        "khong": "không",
        "duoc": "được",
        "nhan": "nhận",
        "gui": "gửi",
        "chua": "chưa",
        "roi": "rồi",
        "den": "đến",
        "tren": "trên",
        "duoi": "dưới",
        "vua": "vừa",
        "lam": "làm",
        "hay": "hãy",
        "ngay": "ngày",
        "gio": "giờ",
        "nha": "nhà",
        "co": "có",
        "la": "là",
        "ma": "mà",
        "thi": "thì",
        "nhu": "như",
        "ban": "bạn",
        "toi": "tôi",
        "minh": "mình",
        "ho": "họ",
        "chung ta": "chúng ta",
        "chung toi": "chúng tôi",
    }
    
    corrections = []
    text_with_diacritics = text_clean.lower()
    
    if restore_diacritics:
        # Apply simple diacritics restoration
        for without_diacritics, with_diacritics in diacritics_map.items():
            if without_diacritics in text_with_diacritics:
                text_with_diacritics = text_with_diacritics.replace(
                    without_diacritics, with_diacritics
                )
                corrections.append({
                    "type": "diacritics",
                    "original": without_diacritics,
                    "corrected": with_diacritics,
                    "position": text_with_diacritics.find(with_diacritics)
                })
    
    if fix_typos:
        # Common typo corrections for Vietnamese
        # These are typical ASR errors for Vietnamese consonants
        typo_corrections = {
            "d ": "đ ",  # Common ASR confusion
            " tr ": " ch ",  # Sometimes confused
        }
        
        for typo, correction in typo_corrections.items():
            if typo in text_with_diacritics:
                # Only apply if it makes sense contextually
                # TODO: Add context-aware correction
                pass
    
    return NormalizeResponse(
        text_clean=text_clean,
        text_with_diacritics=text_with_diacritics,
        corrections=corrections
    )


async def handle_transcription_completed(msg):
    """Handle transcription.completed events."""
    try:
        evt = json.loads(msg.data.decode())
        audio_id = evt["data"]["audio_id"]
        transcript_uri = evt["data"].get("transcript_uri", "")
        text = evt["data"].get("text", "")
        
        print(f"[NLP] Processing {audio_id}")
        print(f"[NLP] Original text: {text}")
        
        # If no text in event, we would download from transcript_uri
        # TODO: Download transcript JSON from MinIO if text not in event
        # from minio import Minio
        # s3_client = Minio(...)
        # transcript_data = s3_client.get_object(...)
        
        # Run normalization on the transcribed text
        if not text:
            print(f"[WARNING] No text in event for {audio_id}, skipping NLP")
            return
            
        normalized = await normalize_vietnamese_text(text)
        
        print(f"[NLP] Normalized text: {normalized.text_with_diacritics}")
        print(f"[NLP] Applied {len(normalized.corrections)} corrections")
        
        # Publish nlp.postprocessed event
        nlp_event = {
            "specversion": "1.0",
            "id": str(uuid.uuid4()),
            "source": "services/nlp-post",
            "type": "NLPPostprocessed",
            "time": datetime.utcnow().isoformat() + "Z",
            "datacontenttype": "application/json",
            "data": {
                "audio_id": audio_id,
                "text_clean": normalized.text_clean,
                "text_with_diacritics": normalized.text_with_diacritics,
                "corrections": normalized.corrections,
                "transcript_uri": transcript_uri
            }
        }
        
        await nc.publish("nlp.postprocessed", json.dumps(nlp_event, ensure_ascii=False).encode())
        print(f"[OK] NLP post-processing completed for {audio_id}")
        
    except Exception as e:
        print(f"[ERROR] NLP processing failed: {e}")
        import traceback
        traceback.print_exc()


@app.on_event("startup")
async def startup():
    """Initialize service."""
    global nc
    nc = await nats.connect(NATS_URL)
    await nc.subscribe("transcription.completed", cb=handle_transcription_completed)
    print(f"[OK] NLP-Post service started, subscribed to transcription.completed")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup."""
    global nc
    if nc:
        await nc.drain()
        await nc.close()


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy", "service": "nlp-post", "nats": nc is not None}


@app.post("/v1/nlp/normalize", response_model=NormalizeResponse)
async def normalize(r: NormalizeRequest):
    """
    Normalize Vietnamese text.
    
    TODO: Plug a ByT5-style seq2seq to jointly restore diacritics + typos.
    """
    return await normalize_vietnamese_text(
        r.text,
        r.restore_diacritics,
        r.fix_typos
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
