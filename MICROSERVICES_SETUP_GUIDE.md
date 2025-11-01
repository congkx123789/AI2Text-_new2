# AI2Text ASR Microservices - Complete Setup Guide

🚀 **Production-ready Vietnamese ASR system with end-to-end microservices architecture**

## Overview

This is a complete microservices implementation for Vietnamese Speech-to-Text (ASR) with:

- **Real-time WebSocket streaming** for low-latency transcription
- **Batch processing** for high-throughput workloads
- **Vietnamese NLP post-processing** (diacritics restoration + typo correction)
- **Semantic search** over transcripts using vector embeddings
- **Three-tier data architecture**: Object storage (audio) + ACID database (metadata) + Vector DB (embeddings)

## Architecture

```
┌─────────────┐
│ API Gateway │ ← JWT Auth + Rate Limiting
└──────┬──────┘
       │
   ┌───┴────────────────────────────────┐
   │                                    │
┌──▼─────────┐                  ┌──────▼──────┐
│ Ingestion  │─────NATS────────▶│     ASR     │
│  Service   │   (Events)       │   Worker    │
└─────┬──────┘                  └──────┬──────┘
      │                                │
   ┌──▼──────┐                    ┌────▼─────┐
   │  MinIO  │                    │ NLP-Post │
   │ (Audio) │                    │  Service │
   └─────────┘                    └────┬─────┘
                                       │
                              ┌────────▼────────┐
                              │    Metadata     │
                              │   (PostgreSQL)  │
                              └────────┬────────┘
                                       │
                              ┌────────▼────────┐
                              │   Embeddings    │
                              │     Service     │
                              └────────┬────────┘
                                       │
                              ┌────────▼────────┐
                              │     Qdrant      │
                              │  (Vector DB)    │
                              └────────┬────────┘
                                       │
                              ┌────────▼────────┐
                              │     Search      │
                              │    Service      │
                              └─────────────────┘
```

## Services

| Service | Port | Purpose |
|---------|------|---------|
| **API Gateway** | 8080 | JWT auth, rate limiting, routing |
| **Ingestion** | 8001 | Upload audio to object storage |
| **ASR Streaming** | 8003 | WebSocket real-time transcription |
| **Metadata** | 8002 | ACID metadata store (PostgreSQL) |
| **NLP-Post** | 8004 | Vietnamese text normalization |
| **Embeddings** | 8005 | Vector generation & indexing |
| **Search** | 8006 | Semantic search over transcripts |
| **PostgreSQL** | 5432 | Structured metadata |
| **MinIO** | 9000/9001 | S3-compatible object storage |
| **Qdrant** | 6333 | Vector database |
| **NATS** | 4222 | Event bus / message queue |

## Quick Start (Development)

### Prerequisites

- Docker & Docker Compose
- Python 3.11+ (for local testing)
- Make (optional, for convenience commands)

### 1. Initialize Environment

```bash
# Copy environment template
make init
# or manually:
cp env.example .env
```

### 2. Start All Services

```bash
# Start the full stack
make up

# View logs
make logs

# Check service status
make ps
```

### 3. Run Database Migrations

```bash
make migrate
```

### 4. Verify Setup

```bash
# Check all services are healthy
make health

# Run end-to-end tests
make test-e2e
```

### 5. Access Services

- **API Gateway**: http://localhost:8080
- **MinIO Console**: http://localhost:9001 (username: `minio`, password: `minio123`)
- **Qdrant Dashboard**: http://localhost:6333/dashboard
- **PostgreSQL**: `localhost:5432` (username: `postgres`, password: `postgres`)

## Usage Examples

### 1. WebSocket Streaming (Real-time ASR)

```python
import asyncio
import websockets
import json
import base64

async def stream_audio():
    async with websockets.connect("ws://localhost:8003/v1/asr/stream") as ws:
        # Start session
        await ws.send(json.dumps({
            "type": "start",
            "audio_format": {
                "sample_rate": 16000,
                "channels": 1,
                "encoding": "pcm16"
            }
        }))
        
        # Get audio_id
        ack = json.loads(await ws.recv())
        audio_id = ack["audio_id"]
        print(f"Session started: {audio_id}")
        
        # Send audio frames
        # (In production, read from microphone or file)
        audio_data = b'\x00' * 32000  # 1 second of silence
        await ws.send(json.dumps({
            "type": "frame",
            "base64": base64.b64encode(audio_data).decode()
        }))
        
        # Signal end
        await ws.send(json.dumps({"type": "end"}))
        
        # Receive results
        async for msg in ws:
            result = json.loads(msg)
            if result["type"] == "partial":
                print(f"Partial: {result['text']}")
            elif result["type"] == "final":
                print(f"Final: {result['text']}")
                break

asyncio.run(stream_audio())
```

### 2. Batch Upload via API Gateway

```bash
# Upload audio file (requires JWT token)
curl -X POST http://localhost:8080/v1/ingest \
  -H "Authorization: Bearer dev" \
  -F "file=@audio.wav"

# Response:
# {
#   "audio_id": "550e8400-e29b-41d4-a716-446655440000",
#   "object_uri": "s3://audio/raw/550e8400-e29b-41d4-a716-446655440000.wav"
# }
```

### 3. Retrieve Transcript

```bash
# Get transcript from metadata service
curl -X GET http://localhost:8080/v1/transcripts/{audio_id} \
  -H "Authorization: Bearer dev"

# Response:
# {
#   "audio_id": "550e8400-e29b-41d4-a716-446655440000",
#   "text": "xin chao viet nam",
#   "text_clean": "xin chào việt nam"
# }
```

### 4. Semantic Search

```bash
# Search transcripts
curl -X GET "http://localhost:8080/v1/search?q=xin+chào" \
  -H "Authorization: Bearer dev"

# Response:
# {
#   "results": [
#     {
#       "audio_id": "...",
#       "text": "xin chào việt nam",
#       "score": 0.95
#     }
#   ],
#   "total": 1,
#   "query_time_ms": 12.5
# }
```

### 5. Vietnamese NLP Post-Processing

```bash
# Normalize Vietnamese text
curl -X POST http://localhost:8004/v1/nlp/normalize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "xin chao viet nam",
    "restore_diacritics": true,
    "fix_typos": true
  }'

# Response:
# {
#   "text_clean": "xin chao viet nam",
#   "text_with_diacritics": "xin chào việt nam",
#   "corrections": [
#     {
#       "type": "diacritics",
#       "original": "chao",
#       "corrected": "chào"
#     }
#   ]
# }
```

## Event Flow

The system uses event-driven architecture with NATS:

1. **RecordingIngested** → Published by Ingestion service
2. **TranscriptionCompleted** → Published by ASR worker
3. **NLPPostprocessed** → Published by NLP-Post service
4. **EmbeddingsIndexed** → Published by Embeddings service

```
Upload Audio
     ↓
RecordingIngested event
     ↓
ASR Worker processes
     ↓
TranscriptionCompleted event
     ↓
NLP-Post normalizes text
     ↓
NLPPostprocessed event
     ↓
Metadata service stores in PostgreSQL
     ↓
Embeddings service generates vectors
     ↓
EmbeddingsIndexed event
     ↓
Qdrant indexes for search
```

## Data Planes

### 1. Object Storage (MinIO/S3)

- **Raw audio**: `s3://audio/raw/{audio_id}.wav`
- **Transcripts**: `s3://audio/transcripts/{audio_id}.json`
- **Purpose**: Blob storage for audio files and large JSON outputs

### 2. Metadata Store (PostgreSQL)

- **Tables**: `audio`, `transcripts`, `speakers`
- **Purpose**: ACID transactions, structured queries, speaker-level split enforcement
- **Key features**:
  - Speaker-level data split (prevents leakage)
  - SNR estimation (for drift detection)
  - Device type tracking (for error analysis)

### 3. Vector Database (Qdrant)

- **Collection**: `texts`
- **Purpose**: Semantic search, ANN queries
- **Vector size**: 768 (configurable)

## Configuration

### Environment Variables

See `env.example` for all configuration options:

```bash
# Messaging
NATS_URL=nats://nats:4222

# Object Storage
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minio
MINIO_SECRET_KEY=minio123

# Database
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/asrmeta

# Vector DB
QDRANT_URL=http://qdrant:6333

# Auth
JWT_PUBLIC_KEY=dev
JWT_ALGO=HS256
```

## Makefile Commands

```bash
make help              # Show all available commands
make init              # Initialize .env file
make up                # Start all services
make down              # Stop all services
make logs              # View logs
make ps                # Show service status
make migrate           # Run database migrations
make migrate-fresh     # Reset database
make test-e2e          # Run end-to-end tests
make health            # Check service health
make clean             # Clean up Docker resources
make restart           # Restart all services
```

## Database Schema

### Speaker-Level Split Enforcement

```sql
-- Speakers table
CREATE TABLE speakers (
  speaker_id UUID PRIMARY KEY,
  pseudonymous_id TEXT UNIQUE NOT NULL,
  region TEXT,
  device_types TEXT[],
  total_recordings INTEGER DEFAULT 0
);

-- Audio table with split assignment
CREATE TABLE audio (
  audio_id UUID PRIMARY KEY,
  speaker_id UUID REFERENCES speakers(speaker_id),
  audio_path TEXT NOT NULL,
  snr_estimate REAL,
  device_type TEXT,
  environment TEXT,
  split_assignment split_enum NOT NULL,  -- TRAIN, VAL, TEST
  duration_seconds REAL,
  sample_rate INTEGER
);

-- Trigger prevents speaker appearing in multiple splits
CREATE TRIGGER trg_speaker_split
BEFORE INSERT OR UPDATE ON audio
FOR EACH ROW EXECUTE FUNCTION enforce_speaker_split();

-- Transcripts table
CREATE TABLE transcripts (
  audio_id UUID PRIMARY KEY REFERENCES audio(audio_id),
  raw_json JSONB,
  text TEXT,
  text_clean TEXT  -- After NLP post-processing
);
```

## Vietnamese NLP Processing

The NLP-Post service provides:

### 1. Diacritics Restoration

Common Vietnamese words without diacritics are automatically corrected:

- `xin chao` → `xin chào`
- `viet nam` → `việt nam`
- `cam on` → `cảm ơn`
- `khong` → `không`

### 2. Typo Correction

ASR-specific errors are corrected:

- Consonant confusion (d/đ, ch/tr)
- Tone mark errors
- Word boundary errors

### 3. Integration Points

**For production**, replace the rule-based system with:

- **ByT5** for seq2seq diacritics restoration
- **underthesea** library for Vietnamese NLP
- **PhoBERT** fine-tuned for text normalization

## Production Deployment

### Kubernetes (Helm)

```bash
# Deploy to Kubernetes cluster
helm upgrade --install ai2text infra/helm/ai-stt \
  -f infra/helm/ai-stt/values.prod.yaml
```

### Environment Changes for Production

1. **Replace JWT dev key** with RS256 and proper key management
2. **Use AWS S3** instead of MinIO
3. **Use managed PostgreSQL** (RDS, Cloud SQL)
4. **Use managed Qdrant** or hosted solution
5. **Add monitoring** (Prometheus, Grafana, Jaeger)
6. **Enable TLS** for all service communication

### Production Checklist

- [ ] Replace dev JWT with RS256 keys
- [ ] Configure proper CORS origins
- [ ] Set up SSL/TLS certificates
- [ ] Configure database backups
- [ ] Set up log aggregation (ELK, Loki)
- [ ] Configure resource limits (CPU, memory)
- [ ] Enable rate limiting in API Gateway
- [ ] Set up monitoring and alerting
- [ ] Configure horizontal pod autoscaling
- [ ] Implement circuit breakers
- [ ] Add distributed tracing

## Testing

### Unit Tests

```bash
make test-unit
```

### End-to-End Tests

```bash
make test-e2e
```

### Manual Testing

```bash
# Test health endpoints
make health

# Test streaming
python tests/e2e/test_flow.py::test_asr_websocket_streaming -v

# Test full pipeline
python tests/e2e/test_flow.py::test_full_pipeline -v
```

## Troubleshooting

### Services won't start

```bash
# Check logs
make logs

# Restart specific service
docker compose -f infra/docker-compose.yml restart <service-name>

# Rebuild from scratch
make down-volumes
make up
make migrate
```

### Database connection issues

```bash
# Check PostgreSQL is running
docker compose -f infra/docker-compose.yml ps postgres

# Manually connect
make shell-postgres

# Re-run migrations
make migrate-fresh
```

### NATS connection issues

```bash
# Check NATS health
curl http://localhost:8222/

# View NATS logs
docker compose -f infra/docker-compose.yml logs nats
```

## Performance Tuning

### 1. ASR Worker Scaling

```bash
# Scale ASR workers for higher throughput
docker compose -f infra/docker-compose.yml up -d --scale asr-worker=3
```

### 2. Database Connection Pooling

For production, use PgBouncer or configure connection pooling in the application.

### 3. Caching

Add Redis for frequently accessed metadata:

```yaml
redis:
  image: redis:7-alpine
  ports:
    - "6379:6379"
```

## Next Steps

### 1. Integrate Real ASR Models

Replace `DummyTranscriber` in `services/asr/streaming_server.py`:

```python
# Option 1: Whisper
import whisper
model = whisper.load_model("base")

# Option 2: Your custom model
from models.lstm_asr import LSTMASRModel
model = LSTMASRModel.load_checkpoint("checkpoints/best_model.pt")

# Option 3: FastConformer
# Use NeMo or similar for low-latency streaming
```

### 2. Add Real Vietnamese NLP Models

Update `services/nlp-post/app.py`:

```python
# Option 1: underthesea
from underthesea import word_tokenize

# Option 2: ByT5 for diacritics
from transformers import T5ForConditionalGeneration

# Option 3: PhoBERT
from transformers import AutoModel
```

### 3. Implement Training Orchestration

Add the training-orchestrator service to manage:

- DVC for dataset versioning
- Gradient accumulation for limited GPU memory
- Mixed precision training
- Tarred shards for large datasets

### 4. Add Monitoring

```bash
# Add Prometheus + Grafana
docker compose -f infra/docker-compose.monitoring.yml up -d

# Add Jaeger for distributed tracing
docker compose -f infra/docker-compose.tracing.yml up -d
```

## License

[Your License Here]

## Support

For issues, questions, or contributions:

- GitHub Issues: [Your Repo URL]
- Documentation: [Your Docs URL]
- Email: [Your Email]

