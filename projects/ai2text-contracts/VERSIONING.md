# AI2Text Contracts Versioning Strategy

## Version: 1.1.0

## Semantic Versioning

Contracts follow **SemVer (Semantic Versioning)**:

```
MAJOR.MINOR.PATCH

1.1.0
^ ^ ^
| | +-- Patch: Backward-compatible bug fixes (typos, descriptions)
| +---- Minor: Backward-compatible additions (new optional fields)
+------ Major: Breaking changes (removed fields, changed types)
```

## What Triggers Version Bumps

### Major (Breaking Changes)
- Removing a field from request/response
- Changing field type (string → integer)
- Renaming a field
- Removing an endpoint
- Changing required/optional status (optional → required)
- Changing event schema structure

**Example:**
```yaml
# v1.0.0
properties:
  user_id: { type: string }

# v2.0.0 (BREAKING)
properties:
  userId: { type: string }  # Renamed field
```

### Minor (Additions)
- Adding a new optional field
- Adding a new endpoint
- Adding a new event type
- Expanding enum values

**Example:**
```yaml
# v1.1.0
properties:
  user_id: { type: string }
  email: { type: string }  # NEW optional field
```

### Patch (Fixes)
- Fixing typos in descriptions
- Clarifying documentation
- Adding examples
- Fixing constraint values that were incorrect

## Version Lifecycle

### Active Versions
- Current: **1.1.0** (production)
- Previous: **1.0.0** (deprecated, supported until Feb 2026)

### Support Policy
- **N (current)**: Fully supported
- **N-1 (previous major)**: Supported for 90 days after N release
- **N-2 and older**: Unsupported

## Migration Path for Breaking Changes

### 1. Pre-announcement (T-30 days)
- Announce upcoming breaking change
- Publish migration guide
- Provide sample code

### 2. Parallel Support (T-0 to T+90 days)
- Both versions available
- Gradual client migration
- Monitor usage metrics

### 3. Deprecation (T+90 days)
- Old version marked deprecated
- Warning messages in API responses
- Final migration push

### 4. Removal (T+180 days)
- Old version removed
- 410 Gone response for old endpoints

## Contract Evolution Examples

### REST API (OpenAPI)

#### Adding Optional Field (Minor)
```yaml
# v1.1.0 - MINOR bump
components:
  schemas:
    RecordingResponse:
      properties:
        recording_id: { type: string }
        status: { type: string }
        metadata: { type: object }  # NEW optional field
```

#### Breaking Change (Major)
```yaml
# v2.0.0 - MAJOR bump
paths:
  /recordings/{id}:
    parameters:
      - name: id
        schema:
          type: integer  # BREAKING: was string in v1.x
```

### Events (AsyncAPI)

#### Adding Event Field (Minor)
```yaml
# v1.1.0
channels:
  transcription.completed.v1:
    payload:
      properties:
        recording_id: { type: string }
        text: { type: string }
        confidence: { type: number }  # NEW field
```

#### Breaking Event Schema (Major + New Subject)
```yaml
# v2.0.0 - Publish to NEW subject
channels:
  transcription.completed.v2:  # New subject for v2
    payload:
      properties:
        recordingId: { type: string }  # Renamed (breaking)
        transcript: { type: string }   # Renamed (breaking)
```

## Client Usage

### REST API Versioning
```python
# Clients specify version via URL or header
GET /v1/recordings/{id}
GET /v2/recordings/{id}

# Or via Accept header
Accept: application/vnd.ai2text.v1+json
```

### Event Versioning
```python
# Subscribe to specific event version
nats.subscribe("transcription.completed.v1")
nats.subscribe("transcription.completed.v2")
```

## Validation & CI

### Pre-merge Checks
```bash
# Detect breaking changes
make validate-breaking

# Check version bump matches changes
make validate-version
```

### Contract Tests
```python
# Test v1 client against v1 spec
pytest tests/contract/v1/

# Test v2 client against v2 spec
pytest tests/contract/v2/
```

## Changelog

### v1.1.0 (November 2025)
**Added:**
- Optional `metadata` field to RecordingResponse
- Optional `confidence` field to transcription events
- New `/search` endpoint with semantic search

**Changed:**
- Improved error response format (backward-compatible)

**Deprecated:**
- None

### v1.0.0 (October 2025)
**Initial release:**
- Gateway API
- Ingestion API
- Search API
- Metadata API
- ASR API
- All event schemas

## References

- [OpenAPI Specification](https://spec.openapis.org/oas/v3.0.3)
- [AsyncAPI Specification](https://www.asyncapi.com/docs/reference/specification/v3.0.0)
- [Semantic Versioning](https://semver.org/)

---

**Current Version**: 1.1.0
**Last Updated**: November 1, 2025

