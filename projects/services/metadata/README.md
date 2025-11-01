# AI2Text Metadata Service

Recording metadata management service with PostgreSQL backend.

## Features

- CRUD operations for recording metadata
- ACID transactions
- Status tracking (uploaded → transcribing → completed)
- PostgreSQL-backed storage
- Health checks and metrics

## API Endpoints

- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics
- `POST /recordings` - Create recording
- `GET /recordings` - List recordings
- `GET /recordings/{id}` - Get recording
- `PATCH /recordings/{id}` - Update recording
- `DELETE /recordings/{id}` - Delete recording

## Configuration

Environment variables:
- `DATABASE_URL` - PostgreSQL connection string
- `NATS_URL` - NATS server URL
- `LOG_LEVEL` - Logging level

## Development

```bash
# Install dependencies
uv pip install -e .[dev]

# Run migrations
psql $DATABASE_URL < ../../ai2text-platform/migrations/metadata-db/V1__init.sql

# Run locally
python -m app.main

# Run tests
pytest
```

## Database Schema

```sql
CREATE TABLE recordings (
    recording_id UUID PRIMARY KEY,
    status VARCHAR(50) NOT NULL,
    audio_url TEXT NOT NULL,
    transcript TEXT,
    language VARCHAR(10) NOT NULL,
    duration_sec FLOAT,
    error TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

## SLOs

- Write p95 latency: < 40ms
- Error rate: < 0.5%
- Read availability: 99.9%

