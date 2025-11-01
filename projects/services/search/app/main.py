"""AI2Text Search Service - Semantic search over transcripts."""
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator, List
from uuid import UUID

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from ai2text_common.observability import setup_tracing, setup_logging
from app.deps import get_settings

# Setup
setup_logging("search")
logger = logging.getLogger(__name__)

# Metrics
search_requests_total = Counter("search_requests_total", "Total search requests")
search_duration_seconds = Histogram("search_duration_seconds", "Search query duration")
search_results_total = Counter("search_results_total", "Total search results returned")

# Global client
qdrant_client: QdrantClient = None


class SearchHit(BaseModel):
    """Search result hit."""
    id: str
    score: float
    text: str
    recording_id: str | None = None
    segment_index: int | None = None
    metadata: dict = {}


class SearchResponse(BaseModel):
    """Search response."""
    hits: List[SearchHit]
    total: int
    query_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    checks: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan."""
    global qdrant_client
    
    logger.info("Starting search service...")
    setup_tracing("search")
    
    settings = get_settings()
    qdrant_client = QdrantClient(url=settings.qdrant_url)
    
    # Test connection
    try:
        collections = qdrant_client.get_collections()
        logger.info(f"Connected to Qdrant: {len(collections.collections)} collections")
        
        # Optimize HNSW for p95 < 50ms
        from app.optimization import optimize_hnsw_collection, configure_search_ef
        
        optimize_hnsw_collection(
            client=qdrant_client,
            collection_name=settings.qdrant_collection,
            vector_size=384,  # TODO: Get from config
        )
        configure_search_ef(
            client=qdrant_client,
            collection_name=settings.qdrant_collection,
            ef=128,  # Optimized for p95 < 50ms
        )
        
        logger.info("HNSW optimization complete")
    except Exception as e:
        logger.error(f"Failed to optimize Qdrant: {e}")
    
    logger.info("Search service ready")
    yield
    
    logger.info("Shutting down search service...")


app = FastAPI(
    title="AI2Text Search API",
    version="1.0.0",
    description="Semantic search service for AI2Text transcripts",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    checks = {}
    
    # Check Qdrant
    try:
        qdrant_client.get_collections()
        checks["qdrant"] = "ok"
    except Exception as e:
        logger.error(f"Qdrant health check failed: {e}")
        checks["qdrant"] = "error"
    
    status = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
    
    return HealthResponse(
        status=status,
        timestamp=datetime.utcnow(),
        checks=checks
    )


@app.get("/metrics", response_class=PlainTextResponse, tags=["observability"])
async def metrics() -> PlainTextResponse:
    """Prometheus metrics."""
    return PlainTextResponse(generate_latest().decode())


@app.get("/search", response_model=SearchResponse, tags=["search"])
async def search(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(10, ge=1, le=100, description="Max results"),
    threshold: float = Query(0.7, ge=0, le=1, description="Min similarity score"),
    collection: str = Query("transcripts", description="Qdrant collection")
) -> SearchResponse:
    """Semantic search over transcripts."""
    search_requests_total.inc()
    start_time = time.time()
    
    try:
        # TODO: Generate query embedding (placeholder for now)
        # In real implementation, call embedding service or use local model
        query_vector = [0.1] * 384  # Placeholder
        
        # Use optimized search parameters for p95 < 50ms
        from app.optimization import get_optimized_search_params
        
        search_params = get_optimized_search_params()
        search_params.update({
            "limit": limit,
            "score_threshold": threshold,
        })
        
        # Search Qdrant with HNSW optimization
        with search_duration_seconds.time():
            results = qdrant_client.search(
                collection_name=collection,
                query_vector=query_vector,
                query_filter=None,
                search_params=models.SearchParams(
                    hnsw_ef=search_params["hnsw_ef"],  # Optimized for speed (128)
                ),
                limit=search_params["limit"],
                score_threshold=search_params["score_threshold"],
                with_payload=search_params["with_payload"],
                with_vectors=search_params["with_vectors"],
            )
        
        # Convert results
        hits = []
        for result in results:
            hits.append(SearchHit(
                id=str(result.id),
                score=result.score,
                text=result.payload.get("text", ""),
                recording_id=result.payload.get("recording_id"),
                segment_index=result.payload.get("segment_index"),
                metadata=result.payload.get("metadata", {})
            ))
        
        search_results_total.inc(len(hits))
        query_time_ms = (time.time() - start_time) * 1000
        
        return SearchResponse(
            hits=hits,
            total=len(hits),
            query_time_ms=query_time_ms
        )
    
    except UnexpectedResponse as e:
        logger.error(f"Qdrant error: {e}")
        raise HTTPException(status_code=500, detail="Search backend error")
    except Exception as e:
        logger.exception(f"Search error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

