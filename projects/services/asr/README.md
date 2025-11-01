# AI2Text ASR Service

Automatic Speech Recognition service with batch and streaming support.

## Features

- Batch transcription via NATS workers
- Real-time streaming via WebSocket
- Multi-language support (Vietnamese, English)
- Model version management
- Health checks and metrics

## API Endpoints

- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics
- `POST /transcribe` - Trigger batch transcription
- `GET /stream` - WebSocket streaming ASR

## Configuration

Environment variables:
- `MODEL_PATH` - Path to ASR model
- `NATS_URL` - NATS server URL
- `LOG_LEVEL` - Logging level

## Development

```bash
uv pip install -e .[dev]
python -m app.main
pytest
```

## SLOs

- Batch transcription: < 0.5x realtime
- Streaming partial latency: < 500ms p95
- Model availability: 99.9%

