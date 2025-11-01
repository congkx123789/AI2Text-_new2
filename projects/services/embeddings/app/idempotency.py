"""Idempotent embeddings pipeline with retry handling."""
import hashlib
import logging
import asyncio
from typing import Dict, Any, Optional
from uuid import UUID
from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

# Metrics
embedding_jobs_total = Counter(
    "embedding_jobs_total",
    "Total embedding jobs",
    ["status"]
)

embedding_retries_total = Counter(
    "embedding_retries_total",
    "Total embedding retries",
    ["reason"]
)

embedding_duration_seconds = Histogram(
    "embedding_duration_seconds",
    "Embedding job duration",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

duplicate_embeddings_total = Counter(
    "duplicate_embeddings_total",
    "Total duplicate embeddings detected"
)


class IdempotencyKey:
    """Generate idempotency keys for embeddings."""
    
    @staticmethod
    def from_content(content: str) -> str:
        """Generate idempotency key from content hash."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        return f"emb_{content_hash}"
    
    @staticmethod
    def from_recording(recording_id: str, segment_index: int) -> str:
        """Generate idempotency key from recording + segment."""
        key_string = f"{recording_id}:{segment_index}"
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()
        return f"emb_{key_hash}"


class EmbeddingStore:
    """Store for tracking processed embeddings (idempotency)."""
    
    def __init__(self):
        # In production, use Redis or database
        self.processed_keys: set[str] = set()
        self.embedding_cache: dict[str, list[float]] = {}
    
    async def is_processed(self, idempotency_key: str) -> bool:
        """Check if embedding was already processed."""
        return idempotency_key in self.processed_keys
    
    async def get_existing(self, idempotency_key: str) -> Optional[list[float]]:
        """Get existing embedding if available."""
        return self.embedding_cache.get(idempotency_key)
    
    async def mark_processed(
        self,
        idempotency_key: str,
        embedding: list[float],
    ) -> None:
        """Mark embedding as processed."""
        self.processed_keys.add(idempotency_key)
        self.embedding_cache[idempotency_key] = embedding
    
    async def check_duplicate(
        self,
        idempotency_key: str,
    ) -> tuple[bool, Optional[list[float]]]:
        """
        Check for duplicate and return existing if found.
        
        Returns:
            (is_duplicate, existing_embedding)
        """
        is_processed = await self.is_processed(idempotency_key)
        existing = None
        
        if is_processed:
            existing = await self.get_existing(idempotency_key)
            duplicate_embeddings_total.inc()
            logger.info(f"Duplicate embedding detected: {idempotency_key}")
        
        return (is_processed, existing)


class RetryHandler:
    """Handle retries with exponential backoff."""
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 30.0,
        backoff_factor: float = 2.0,
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
    
    async def execute_with_retry(
        self,
        func,
        *args,
        **kwargs,
    ) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                result = await func(*args, **kwargs)
                
                if attempt > 0:
                    embedding_retries_total.labels(reason="success_after_retry").inc()
                    logger.info(f"Operation succeeded after {attempt} retries")
                
                return result
            
            except Exception as e:
                last_exception = e
                embedding_retries_total.labels(reason=type(e).__name__).inc()
                
                if attempt < self.max_retries - 1:
                    delay = min(
                        self.initial_delay * (self.backoff_factor ** attempt),
                        self.max_delay
                    )
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries} attempts failed")
        
        raise last_exception


async def process_embedding_idempotent(
    content: str,
    recording_id: str,
    segment_index: int,
    embedding_store: EmbeddingStore,
    generate_embedding_fn,  # Function to generate embedding
    retry_handler: RetryHandler,
) -> Dict[str, Any]:
    """
    Process embedding with idempotency and retry handling.
    
    Args:
        content: Text content to embed
        recording_id: Recording identifier
        segment_index: Segment index
        embedding_store: Store for idempotency checks
        generate_embedding_fn: Function to generate embedding
        retry_handler: Retry handler
    
    Returns:
        Dict with embedding and metadata
    """
    start_time = asyncio.get_event_loop().time()
    
    # Generate idempotency key
    idempotency_key = IdempotencyKey.from_recording(recording_id, segment_index)
    
    # Check for duplicate
    is_duplicate, existing_embedding = await embedding_store.check_duplicate(
        idempotency_key
    )
    
    if is_duplicate and existing_embedding:
        logger.info(f"Returning existing embedding: {idempotency_key}")
        embedding_jobs_total.labels(status="duplicate").inc()
        
        return {
            "embedding": existing_embedding,
            "idempotency_key": idempotency_key,
            "is_duplicate": True,
            "duration_seconds": 0.0,
        }
    
    # Generate new embedding with retry
    async def _generate():
        embedding = await generate_embedding_fn(content)
        
        # Store for future idempotency
        await embedding_store.mark_processed(idempotency_key, embedding)
        
        return embedding
    
    try:
        embedding = await retry_handler.execute_with_retry(_generate)
        
        duration = asyncio.get_event_loop().time() - start_time
        embedding_duration_seconds.observe(duration)
        embedding_jobs_total.labels(status="success").inc()
        
        return {
            "embedding": embedding,
            "idempotency_key": idempotency_key,
            "is_duplicate": False,
            "duration_seconds": duration,
        }
    
    except Exception as e:
        embedding_jobs_total.labels(status="failed").inc()
        logger.error(f"Failed to generate embedding: {e}")
        raise


def get_retry_config() -> Dict[str, Any]:
    """Get retry configuration."""
    return {
        "max_retries": 3,
        "initial_delay_seconds": 1.0,
        "max_delay_seconds": 30.0,
        "backoff_factor": 2.0,
        "retry_on_errors": ["ConnectionError", "TimeoutError", "QdrantError"],
    }

