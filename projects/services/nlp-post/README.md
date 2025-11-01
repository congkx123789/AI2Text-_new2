# AI2Text NLP Post-Processing Service

Vietnamese NLP post-processing service for transcript normalization.

## Features

- Vietnamese diacritics restoration
- Text normalization and cleaning
- NATS event-driven processing
- Metrics and health checks

## Configuration

- `NATS_URL` - NATS server URL
- `LOG_LEVEL` - Logging level

## Development

```bash
uv pip install -e .[dev]
python -m app.worker
pytest
```

