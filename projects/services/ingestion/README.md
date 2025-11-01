# AI2Text Ingestion Service

**Handles audio file uploads** and stores them in S3/MinIO.

## Features

- Multipart file upload
- Audio file validation
- S3/MinIO storage integration
- Event publishing (recording.ingested.v1)

## Endpoints

- `POST /ingest` - Upload audio file
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics

## Configuration

Environment variables:
- `MINIO_ENDPOINT` - MinIO endpoint
- `MINIO_ACCESS_KEY` - Access key
- `MINIO_SECRET_KEY` - Secret key
- `MINIO_BUCKET` - Bucket name
- `NATS_URL` - NATS connection URL

## Running

```bash
# Development
uv run python -m app.main

# Production
docker run -p 8080:8080 \
  -e MINIO_ENDPOINT="minio:9000" \
  -e NATS_URL="nats://nats:4222" \
  ghcr.io/yourorg/ai2text-ingestion:latest
```


