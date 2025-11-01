# AI2Text Embeddings Service

Vector embedding generation and Qdrant indexing service.

## Features

- Generate embeddings from transcripts
- Write to Qdrant vector database
- NATS event-driven processing
- Batch processing support

## Configuration

- `QDRANT_URL` - Qdrant server URL
- `NATS_URL` - NATS server URL
- `MODEL_PATH` - Embedding model path

## Development

```bash
uv pip install -e .[dev]
python -m app.worker
pytest
```

