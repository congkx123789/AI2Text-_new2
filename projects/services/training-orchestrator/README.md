# AI2Text Training Orchestrator

ML training orchestration service for model management.

## Features

- Dataset preparation
- Training job orchestration
- Model promotion workflow
- Version management

## Configuration

- `NATS_URL` - NATS server URL
- `MINIO_URL` - MinIO URL for model storage
- `LOG_LEVEL` - Logging level

## Development

```bash
uv pip install -e .[dev]
python -m app.orchestrator
pytest
```

