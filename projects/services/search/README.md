# AI2Text Search Service

Semantic search service for AI2Text transcripts using Qdrant vector database.

## Features

- Vector-based similarity search
- Semantic query understanding
- Fast retrieval with Qdrant
- Prometheus metrics
- Health checks

## API Endpoints

- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics
- `GET /search?q=query&limit=10&threshold=0.7` - Semantic search

## Configuration

Environment variables:
- `QDRANT_URL` - Qdrant server URL (default: `http://localhost:6333`)
- `QDRANT_COLLECTION` - Collection name (default: `transcripts`)
- `NATS_URL` - NATS server URL
- `LOG_LEVEL` - Logging level

## Development

```bash
# Install dependencies
uv pip install -e .[dev]

# Run locally
python -m app.main

# Run tests
pytest

# Build Docker image
make docker-build
```

## Deployment

```bash
# Deploy to Kubernetes
helm upgrade --install search ../../ai2text-platform/helm/charts/search \
  -f ../../ai2text-platform/helm/values/dev/search.yaml
```

## Performance

- Target p95 latency: < 50ms
- Qdrant read-only (writes handled by embeddings service)

