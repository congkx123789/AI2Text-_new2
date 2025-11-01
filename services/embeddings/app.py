"""
Embeddings Service - Generate and index embeddings.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import requests
import os
import nats
import asyncio
import json
import uuid
from datetime import datetime

app = FastAPI(title="embeddings", version="0.1.0")
try:
    from libs.common.observability import wire_otel

    wire_otel(app, "embeddings")
except ImportError:
    pass

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLL = "texts"
NATS_URL = os.getenv("NATS_URL", "nats://nats:4222")

# Global clients
nc = None


class EmbeddingRequest(BaseModel):
    """Embedding generation request."""
    text: str
    audio_id: str


class EmbeddingResponse(BaseModel):
    """Embedding generation response."""
    indexed: bool
    vector_id: str


def generate_embedding(text: str) -> List[float]:
    """
    Generate embedding for text.
    
    TODO: Use your Word2Vec/Phon2Vec model or transformer encoder.
    For now, placeholder vector.
    """
    # Placeholder: Return random vector of size 768
    # TODO: Load Word2Vec/Phon2Vec model
    import random
    return [random.random() for _ in range(768)]


async def handle_nlp_postprocessed(msg):
    """Handle nlp.postprocessed events."""
    try:
        evt = json.loads(msg.data.decode())
        audio_id = evt["data"]["audio_id"]
        text = evt["data"]["text_with_diacritics"]
        
        print(f"[Embeddings] Processing {audio_id}")
        
        # Generate embedding
        vec = generate_embedding(text)
        
        # Index in Qdrant
        payload = {
            "points": [{
                "id": audio_id,
                "vector": vec,
                "payload": {
                    "audio_id": audio_id,
                    "text": text
                }
            }]
        }
        
        requests.put(f"{QDRANT_URL}/collections/{COLL}/points", json=payload)
        
        # Publish embeddings.indexed event
        embedding_event = {
            "specversion": "1.0",
            "id": str(uuid.uuid4()),
            "source": "services/embeddings",
            "type": "EmbeddingsIndexed",
            "time": datetime.utcnow().isoformat() + "Z",
            "datacontenttype": "application/json",
            "data": {
                "audio_id": audio_id,
                "vector_id": audio_id,
                "vector_type": "text",
                "embedding_model": "word2vec-v1"
            }
        }
        
        await nc.publish("embeddings.indexed", json.dumps(embedding_event).encode())
        print(f"[OK] Embeddings indexed for {audio_id}")
        
    except Exception as e:
        print(f"[ERROR] Embedding processing failed: {e}")


@app.on_event("startup")
async def startup():
    """Initialize service."""
    global nc
    
    # Connect to NATS
    nc = await nats.connect(NATS_URL)
    await nc.subscribe("nlp.postprocessed", cb=handle_nlp_postprocessed)
    print(f"[OK] Subscribed to nlp.postprocessed")
    
    # Ensure Qdrant collection exists
    try:
        # Check if collection exists
        r = requests.get(f"{QDRANT_URL}/collections/{COLL}")
        if r.status_code == 404:
            # Create collection
            requests.put(
                f"{QDRANT_URL}/collections/{COLL}",
                json={
                    "vectors": {
                        "size": 768,
                        "distance": "Cosine"
                    }
                }
            )
            print(f"[OK] Created Qdrant collection: {COLL}")
        else:
            print(f"[OK] Qdrant collection exists: {COLL}")
    except Exception as e:
        print(f"[ERROR] Qdrant setup failed: {e}")


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
    try:
        r = requests.get(f"{QDRANT_URL}/collections/{COLL}")
        qdrant_ok = r.status_code == 200
    except:
        qdrant_ok = False
    
    return {
        "status": "healthy",
        "service": "embeddings",
        "nats": nc is not None,
        "qdrant": qdrant_ok
    }


@app.post("/v1/embed", response_model=EmbeddingResponse)
async def embed(r: EmbeddingRequest):
    """
    Generate and index embedding.
    """
    # Generate embedding
    vec = generate_embedding(r.text)
    
    # Index in Qdrant
    payload = {
        "points": [{
            "id": r.audio_id,
            "vector": vec,
            "payload": {
                "audio_id": r.audio_id,
                "text": r.text
            }
        }]
    }
    
    try:
        requests.put(f"{QDRANT_URL}/collections/{COLL}/points", json=payload)
        return EmbeddingResponse(indexed=True, vector_id=r.audio_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
