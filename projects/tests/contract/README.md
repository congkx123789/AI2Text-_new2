# Contract Tests for AI2Text Services

Contract tests ensure services conform to their OpenAPI/AsyncAPI specifications.

## Overview

Contract tests validate:
- **Request/Response schemas** match OpenAPI specs
- **Event schemas** match AsyncAPI specs
- **API compatibility** across versions
- **Breaking changes** are detected

## Structure

```
contract/
├── gateway/        # Gateway service contract tests
├── asr/            # ASR service contract tests
├── search/         # Search service contract tests
└── shared/         # Shared test utilities
```

## Running Tests

### All Contract Tests
```bash
cd projects/tests/contract
pytest -v
```

### Specific Service
```bash
pytest gateway/ -v
pytest asr/ -v
pytest search/ -v
```

### With Coverage
```bash
pytest --cov=. --cov-report=html
```

## CI Integration

Contract tests run automatically in CI:
- **On PR**: Block merge if tests fail
- **On Push**: Run all contract tests
- **Nightly**: Extended contract validation

## Writing Contract Tests

### REST API Contract Test
```python
from contract.shared.client import get_gateway_client

def test_gateway_search_endpoint():
    client = get_gateway_client()
    response = client.search(q="test", limit=10)
    
    # Validate response matches OpenAPI spec
    assert response.status_code == 200
    assert "hits" in response.json()
    assert "total" in response.json()
```

### Event Contract Test
```python
from contract.shared.events import subscribe_to_event

async def test_transcription_event():
    async for event in subscribe_to_event("transcription.completed.v1"):
        # Validate event matches AsyncAPI spec
        assert "recording_id" in event.payload
        assert "text" in event.payload
        break
```

## Test Data

Test fixtures are in `shared/fixtures/`:
- `sample_recordings.json` - Sample recording data
- `sample_transcripts.json` - Sample transcript data
- `sample_events.json` - Sample event payloads

## Version Compatibility

Tests validate compatibility with:
- Current version (v1.1.0)
- Previous major version (v1.0.0)
- Next version (v1.2.0-dev) - when available

## CI Gates

PRs are **blocked** if:
- Contract tests fail
- OpenAPI spec validation fails
- AsyncAPI spec validation fails
- Breaking changes detected without version bump

