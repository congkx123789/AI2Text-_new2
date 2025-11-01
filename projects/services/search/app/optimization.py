"""Qdrant HNSW optimization for Search service."""
import logging
from typing import Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models

logger = logging.getLogger(__name__)


def optimize_hnsw_collection(
    client: QdrantClient,
    collection_name: str = "transcripts",
    vector_size: int = 384,
) -> Dict[str, Any]:
    """
    Optimize Qdrant collection with HNSW parameters for p95 < 50ms.
    
    HNSW (Hierarchical Navigable Small World) tuning:
    - M: Number of connections (higher = better recall, slower)
    - ef_construction: Size of candidate list during construction
    - ef: Size of candidate list during search
    
    Targets:
    - p95 latency < 50ms
    - p99 latency < 120ms
    - Recall@10 >= baseline
    """
    logger.info(f"Optimizing HNSW for collection: {collection_name}")
    
    # Check if collection exists
    try:
        collections = client.get_collections()
        existing = [c.name for c in collections.collections]
        
        if collection_name not in existing:
            logger.info(f"Creating optimized collection: {collection_name}")
            _create_optimized_collection(client, collection_name, vector_size)
        else:
            logger.info(f"Updating existing collection: {collection_name}")
            _update_collection_params(client, collection_name)
        
        return {"status": "optimized", "collection": collection_name}
    
    except Exception as e:
        logger.error(f"Failed to optimize collection: {e}")
        raise


def _create_optimized_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
) -> None:
    """Create collection with optimized HNSW parameters."""
    
    # Optimized parameters for p95 < 50ms
    # Based on Qdrant best practices for search latency
    hnsw_config = models.HnswConfigDiff(
        m=16,              # Moderate connections (balance speed/recall)
        ef_construction=200,  # High construction quality
        full_scan_threshold=10000,  # Use index for >10k points
    )
    
    # Optimize for search speed
    optimizers_config = models.OptimizersConfigDiff(
        indexing_threshold=10000,  # Start indexing after 10k points
        flush_interval_sec=5,
        max_optimization_threads=4,
    )
    
    # Distance metric (cosine for semantic search)
    distance = models.Distance.COSINE
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=distance,
            hnsw_config=hnsw_config,
        ),
        optimizers_config=optimizers_config,
        replication_factor=3,  # High availability
    )
    
    logger.info(f"Created optimized collection: {collection_name}")


def _update_collection_params(
    client: QdrantClient,
    collection_name: str,
) -> None:
    """Update existing collection with optimized parameters."""
    
    # Update HNSW config
    hnsw_config = models.HnswConfigDiff(
        m=16,              # Increase from default 16
        ef_construction=200,  # Increase from default 100
    )
    
    # Update collection params
    client.update_collection(
        collection_name=collection_name,
        hnsw_config=hnsw_config,
    )
    
    logger.info(f"Updated collection params: {collection_name}")


def configure_search_ef(
    client: QdrantClient,
    collection_name: str,
    ef: int = 128,
) -> None:
    """
    Configure search ef parameter for query time.
    
    Higher ef = better recall but slower
    ef=128 is good balance for p95 < 50ms target
    """
    # Note: ef is set per-query, not collection config
    # This is a helper to document the recommended value
    logger.info(f"Recommended ef parameter for {collection_name}: {ef}")
    
    # Store as metadata for reference
    client.set_payload(
        collection_name=collection_name,
        payload={
            "_search_config": {
                "recommended_ef": ef,
                "target_p95_ms": 50,
            }
        },
        points=[0],  # Dummy point to store metadata
    )


def get_optimized_search_params() -> Dict[str, Any]:
    """
    Get optimized search parameters for p95 < 50ms.
    
    Returns dict with recommended query parameters.
    """
    return {
        "hnsw_ef": 128,           # Search candidate list size
        "limit": 10,               # Number of results
        "score_threshold": 0.7,    # Minimum similarity
        "with_payload": True,      # Include metadata
        "with_vectors": False,     # Don't return vectors (faster)
    }

