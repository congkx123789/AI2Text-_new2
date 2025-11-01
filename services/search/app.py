"""
Search Service - Semantic search over transcripts.
"""

from fastapi import FastAPI
from fastapi.params import Query
from pydantic import BaseModel
from typing import List, Dict, Any
import requests
import os
import time

app = FastAPI(title="search", version="0.1.0")
try:
    from libs.common.observability import wire_otel

    wire_otel(app, "search")
except ImportError:
    pass

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLL = "texts"
METADATA_URL = os.getenv("METADATA_URL", "http://localhost:8002")


class SearchResult(BaseModel):
    """Search result."""
    audio_id: str
    text: str
    score: float
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    """Search response."""
    results: List[SearchResult]
    total: int
    query_time_ms: float


def generate_query_embedding(query: str) -> List[float]:
    """
    Generate embedding for search query.
    
    TODO: Use same embedding model as /embed endpoint.
    """
    # Placeholder: Same as embeddings service
    import random
    return [random.random() for _ in range(768)]


@app.get("/health")
async def health():
    """Health check."""
    try:
        r = requests.get(f"{QDRANT_URL}/collections/{COLL}")
        return {"status": "healthy", "service": "search", "qdrant": r.status_code == 200}
    except:
        return {"status": "unhealthy", "service": "search"}


@app.get("/v1/search", response_model=SearchResponse)
async def search(q: str = Query(..., description="Search query")):
    """
    Semantic search over transcripts.
    """
    start_time = time.time()
    
    # Generate query embedding
    qvec = generate_query_embedding(q)
    
    # Search Qdrant
    body = {
        "vector": qvec,
        "limit": 10,
        "with_payload": True
    }
    
    try:
        r = requests.post(f"{QDRANT_URL}/collections/{COLL}/points/search", json=body)
        r.raise_for_status()
        qdrant_results = r.json()["result"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    
    # Format results
    results = []
    for point in qdrant_results:
        payload = point.get("payload", {})
        results.append(SearchResult(
            audio_id=payload.get("audio_id", ""),
            text=payload.get("text", ""),
            score=point.get("score", 0.0),
            metadata=payload
        ))
    
    query_time = (time.time() - start_time) * 1000
    
    return SearchResponse(
        results=results,
        total=len(results),
        query_time_ms=query_time
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
