# Microservices Architecture

## Overview

The ASR system has been refactored into a microservices architecture following best practices for scalability, maintainability, and performance.

---

## Architecture Diagram

```
Frontend (React)
    ↓
API Gateway (port 8080)
    ↓
    ├─→ Ingestion → MinIO → NATS: recording.ingested
    │                                 ↓
    │                              ASR Worker
    │                                 ↓
    │                          NATS: transcription.completed
    │                                 ↓
    │                              NLP-Post
    │                                 ↓
    │                          NATS: nlp.postprocessed
    │                                 ↓
    │                              Embeddings → Qdrant
    │                                 ↓
    ├─→ Metadata ← PostgreSQL         │
    │                                  │
    └─→ Search ← Qdrant ←─────────────┘
```

---

## Services

### 1. API Gateway
- **Port**: 8080
- **Role**: Single entry point, routing, authentication (JWT ready)
- **Endpoints**: `/v1/audio`, `/v1/transcripts/{id}`, `/v1/search`

### 2. Ingestion Service
- **Port**: 8001
- **Role**: Upload audio, store in MinIO, emit `recording.ingested`
- **Storage**: MinIO (S3-compatible)

### 3. Metadata Service
- **Port**: 8002
- **Role**: ACID metadata store (PostgreSQL)
- **Schema**: `audio`, `transcripts`, `speakers` tables
- **Features**: Speaker-level split enforcement, SNR/device tracking

### 4. ASR Service
- **Port**: 8003 (API), Worker (background)
- **Role**: Batch transcription, subscribes to `recording.ingested`
- **Models**: Whisper (batch) or Conformer (streaming-ready)
- **Publishes**: `transcription.completed`

### 5. ASR Streaming Service
- **Port**: 8007 (WebSocket)
- **Role**: Low-latency streaming transcription
- **Protocol**: `start` → PCM frames → `end`
- **Publishes**: `transcription.completed`

### 6. NLP-Post Service
- **Port**: 8004
- **Role**: Vietnamese text normalization
- **Features**: Diacritics restoration, typo correction (ByT5-ready)
- **Subscribes**: `transcription.completed`
- **Publishes**: `nlp.postprocessed`

### 7. Embeddings Service
- **Port**: 8005
- **Role**: Generate embeddings, index in Qdrant
- **Subscribes**: `nlp.postprocessed`
- **Publishes**: `embeddings.indexed`

### 8. Search Service
- **Port**: 8006
- **Role**: Semantic search over transcripts
- **Storage**: Qdrant (vector DB)

---

## Data Planes

### Unstructured Plane: MinIO (Object Storage)
- **Raw audio**: `s3://bucket/raw/{audio_id}.wav`
- **Transcripts**: `s3://bucket/transcripts/{audio_id}.json`
- **Reason**: Durable, cheap, scalable

### Structured Plane: PostgreSQL
- **Tables**: `audio`, `transcripts`, `speakers`
- **Features**: ACID transactions, speaker-level splits, SNR/device tracking
- **Reason**: Reproducibility, analytics, split enforcement

### Vector Plane: Qdrant
- **Collections**: `texts` (768-dim embeddings)
- **Features**: Cosine similarity search, payload filtering
- **Reason**: Fast nearest-neighbor, semantic search

---

## Event Flow

1. **Frontend** → API Gateway → Ingestion
2. **Ingestion** → MinIO → NATS: `recording.ingested`
3. **ASR Worker** (subscribes) → Transcribes → NATS: `transcription.completed`
4. **NLP-Post** (subscribes) → Normalizes → NATS: `nlp.postprocessed`
5. **Embeddings** (subscribes) → Generates vectors → Qdrant → NATS: `embeddings.indexed`
6. **Metadata** (subscribes to all) → Updates PostgreSQL

---

## Event Schema (CloudEvents)

```json
{
  "specversion": "1.0",
  "id": "evt-uuid",
  "source": "services/ingestion",
  "type": "RecordingIngested",
  "time": "2025-10-31T00:00:00Z",
  "datacontenttype": "application/json",
  "data": {
    "audio_id": "aud_ab12cd34",
    "path": "s3://bucket/raw/aud_ab12cd34.wav"
  }
}
```

**Event Types**:
- `recording.ingested`
- `transcription.completed`
- `nlp.postprocessed`
- `embeddings.indexed`
- `model.promoted`

**DLQ Pattern**: Consumers run with durable JetStream subscriptions and publish failures to `*.dlq` subjects for replay.

---

## Database Schema

### Speakers Table
- `speaker_id` (UUID, PK)
- `pseudonymous_id` (TEXT, UNIQUE) - Never expose PII
- `region` (TEXT) - Optional: north/central/south
- `device_types` (TEXT[]) - Array of device types
- `total_recordings` (INTEGER)
- `created_at` (TIMESTAMP)

### Audio Table
- `audio_id` (UUID, PK)
- `speaker_id` (UUID, FK) - References speakers
- `audio_path` (TEXT) - s3:// path
- `snr_estimate` (REAL) - For drift analytics
- `device_type` (TEXT) - mic/headset/mobile
- `environment` (TEXT) - quiet/office/street
- `split_assignment` (split_enum) - TRAIN/VAL/TEST (enforced)
- `duration_seconds` (REAL)
- `sample_rate` (INTEGER)
- `created_at` (TIMESTAMP)

### Transcripts Table
- `audio_id` (UUID, PK, FK)
- `raw_json` (JSONB) - ASR lattice/word timings
- `text` (TEXT) - Post-ASR normalized
- `text_clean` (TEXT) - Post-NLP normalized
- `created_at` (TIMESTAMP)
- `updated_at` (TIMESTAMP)

### Constraints
- **Speaker-level split**: Trigger enforces one split per speaker
- **Indexes**: On `speaker_id`, `snr_estimate`, `device_type`, `split_assignment`

---

## Infrastructure

### Docker Compose Services
- **NATS** (port 4222, monitoring 8222)
- **PostgreSQL** (port 5432)
- **MinIO** (ports 9000, 9001)
- **Qdrant** (ports 6333, 6334)
- **API Gateway, Ingestion, Metadata, ASR API, ASR Streaming, NLP-Post, Embeddings, Search**

### Helm Chart
- Umbrella chart at `infra/helm/ai-stt`
- Deploys infrastructure (NATS, PostgreSQL, MinIO, Qdrant)
- Deploys all services with configurable images/env vars
- Ingress routes REST via gateway and WebSocket traffic directly to streaming ASR

### Observability
- Optional OpenTelemetry wiring (`libs/common/observability.py`)
- Export to console by default; plug OTLP exporter for production
- Prometheus-friendly metrics ready once OTEL collector is introduced

### Networking
- Docker: services share the `asr-network`
- Kubernetes: ingress host configurable via Helm values
- Streaming ASR exposed separately for WebSocket stability

---

## Migration Path

### Phase 1: Infrastructure (Complete)
- ✅ Docker Compose setup
- ✅ Helm chart (NATS/Postgres/MinIO/Qdrant + services)
- ✅ OpenTelemetry hooks

### Phase 2: Extract Services
- [ ] Move ingestion logic from monolith
- [ ] Migrate metadata DB tables
- [ ] Extract ASR to service
- [ ] Extract NLP to service

### Phase 3: Event-Driven
- [ ] Wire up all event handlers
- [ ] Add retry logic
- [ ] Implement dead-letter queues

### Phase 4: Production
- [ ] Plug identity provider and rotate JWT keys
- [ ] Deploy OTEL collector + Prometheus/Grafana
- [ ] Harden NATS JetStream DLQs and retries
- [ ] Automate CI/CD to build & push images

---

## Benefits

1. **Decoupled Services** - Each can scale independently
2. **Event-Driven** - Loose coupling via events
3. **Proper Data Planes** - Structured (PostgreSQL), Unstructured (MinIO), Vector (Qdrant)
4. **Vietnamese NLP** - First-class service for text quality
5. **Scalable** - Ready for production deployment

---

## Next Steps

1. Integrate existing ASR models into ASR service
2. Integrate NLP models (ByT5) into NLP-Post service
3. Migrate existing database logic to Metadata service
4. Connect all event handlers
5. Add authentication and monitoring

---

**Your microservices architecture is ready!** 🚀

