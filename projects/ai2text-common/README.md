# ai2text-common

Shared library for AI2Text microservices.

**Version**: 0.1.0  
**License**: MIT  
**Python**: >= 3.11

## Installation

### From GitHub Releases (Recommended)
```bash
pip install ai2text-common==0.1.0
```

### From PyPI (After publication)
```bash
pip install ai2text-common
```

### Development Installation
```bash
git clone https://github.com/congkx123789/AI2Text-_new2.git
cd projects/ai2text-common
pip install -e .[dev]
```

## Features

### Observability
- **Tracing**: OpenTelemetry distributed tracing
- **Logging**: Structured JSON logging
- **Metrics**: Prometheus metrics helpers

### Events
- **CloudEvents**: Standard event format helpers
- **NATS**: Event publishing/subscribing utilities

### Schemas
- **Common Types**: Shared Pydantic models
- **Event Schemas**: Typed event payloads

## Usage

### Logging
```python
from ai2text_common.observability import setup_logging

setup_logging("my-service")
logger = logging.getLogger(__name__)
logger.info("Structured log message")
```

### Tracing
```python
from ai2text_common.observability import setup_tracing

setup_tracing("my-service")
# Traces automatically instrument FastAPI, HTTP, NATS
```

### Events
```python
from ai2text_common.events import publish_event

await publish_event(
    "transcription.completed.v1",
    {"recording_id": "123", "text": "Hello"}
)
```

### Metrics
```python
from ai2text_common.observability.metrics import record_latency

with record_latency("operation_name"):
    # Your code here
    pass
```

## Versioning

Follows [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

## Publishing

This package is published to PyPI via GitHub Actions on release.

```bash
# Create a release tag
git tag v0.1.0
git push origin v0.1.0
```

The publish workflow will automatically:
1. Build the package
2. Run checks
3. Publish to PyPI

## Development

```bash
# Install with dev dependencies
pip install -e .[dev]

# Run tests
pytest

# Lint
ruff check .
black .

# Type check
mypy ai2text_common
```

## Changelog

### 0.1.0 (2025-11-01)
- Initial release
- Observability modules (tracing, logging, metrics)
- Event helpers (CloudEvents, NATS)
- Common schemas

## License

MIT
