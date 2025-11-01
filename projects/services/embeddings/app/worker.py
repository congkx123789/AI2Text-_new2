"""Embeddings worker."""
import asyncio
import logging
from ai2text_common.observability import setup_logging

setup_logging("embeddings")
logger = logging.getLogger(__name__)


async def main():
    """Main worker loop."""
    logger.info("Starting embeddings worker...")
    
    # TODO: Connect to NATS, subscribe to nlp.postprocessed.v1
    # TODO: Generate embeddings from transcript
    # TODO: Write to Qdrant
    # TODO: Publish embeddings.created.v1
    
    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())

