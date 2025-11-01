# AI2Text API Gateway

**Edge service** providing authentication, rate limiting, and request routing.

## Features

- JWT authentication (RS256)
- Rate limiting (60 requests/minute)
- Request routing to backend services
- Health checks and metrics
- CORS support

## Endpoints

- `GET /health` - Health check
- `POST /v1/ingest` - Route to ingestion service
- `GET /v1/search` - Route to search service
- `GET /v1/metadata/{id}` - Route to metadata service

## Configuration

Environment variables:
- `JWT_PUBLIC_KEY` - RSA public key for JWT verification
- `RATE_LIMIT_PER_MINUTE` - Rate limit (default: 60)
- `INGESTION_SERVICE_URL` - Backend service URLs
- `SEARCH_SERVICE_URL`
- `METADATA_SERVICE_URL`

## Running

```bash
# Development
uv run python -m app.main

# Production
docker run -p 8080:8080 \
  -e JWT_PUBLIC_KEY="$(cat jwt-public.pem)" \
  ghcr.io/yourorg/ai2text-gateway:latest
```


