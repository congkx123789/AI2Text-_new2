# AI2Text Common

**Shared library** for all AI2Text services.

This package contains:
- Event schemas and CloudEvents helpers
- Observability middleware (logging, tracing, metrics)
- Shared data types and DTOs
- NATS message helpers

## Installation

```bash
pip install ai2text-common
# Or from source:
pip install -e .
```

## Usage

### Events

```python
from ai2text_common.events import CloudEventsHelper, RecordingIngested

# Create a CloudEvent
event = CloudEventsHelper.create(
    type="recording.ingested.v1",
    source="ingestion-service",
    data=RecordingIngested(
        recording_id="...",
        audio_url="s3://...",
        language="vi"
    )
)

# Publish via NATS
await nats_client.publish("recording.ingested.v1", event.to_json())
```

### Observability

```python
from ai2text_common.observability import setup_logging, setup_tracing

# Setup logging
logger = setup_logging("my-service", level="INFO")

# Setup tracing
tracer = setup_tracing("my-service", endpoint="http://jaeger:14268/api/traces")
```

### Schemas

```python
from ai2text_common.schemas import HealthResponse, ErrorResponse

response = HealthResponse(status="healthy", timestamp=datetime.now())
```

## Versioning

- **SemVer** for releases
- Keep changes **additive** when possible
- Breaking changes → new major version

## Guidelines

1. Keep this package **small** (<2% of total code)
2. Only shared code that's used by 3+ services
3. Avoid service-specific logic
4. Maintain backwards compatibility


