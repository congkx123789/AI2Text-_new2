"""NLP post-processing worker."""
import asyncio
import logging
from ai2text_common.observability import setup_logging

setup_logging("nlp-post")
logger = logging.getLogger(__name__)


async def main():
    """Main worker loop."""
    logger.info("Starting NLP post-processing worker...")
    
    # TODO: Connect to NATS, subscribe to transcription.completed.v1
    # TODO: Process transcript, restore diacritics
    # TODO: Publish nlp.postprocessed.v1
    
    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())

