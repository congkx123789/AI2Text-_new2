# Microservices Migration Guide

## Overview

This document describes the migration from monolithic architecture to microservices.

---

## Architecture

### Services

1. **API Gateway** - Single entry point, routing, authentication
2. **Ingestion** - Upload audio, store in object storage
3. **Metadata** - ACID metadata store (PostgreSQL)
4. **ASR** - Batch and streaming transcription
5. **NLP-Post** - Vietnamese text normalization
6. **Embeddings** - Embedding generation and indexing
7. **Search** - Semantic search over transcripts
8. **Training-Orchestrator** - Dataset packaging and training jobs

### Infrastructure

- **NATS** - Event bus (lightweight, fast)
- **PostgreSQL** - Structured metadata
- **MinIO** - Object storage (S3-compatible)
- **Qdrant** - Vector database

---

## Setup

### 1. Start Infrastructure

```bash
cd infra
docker compose up -d
```

This starts:
- NATS (port 4222)
- PostgreSQL (port 5432)
- MinIO (ports 9000, 9001)
- Qdrant (ports 6333/6334)
- All microservices (including streaming ASR)

### 2. Initialize Services

Each service has its own requirements. Install dependencies:

```bash
# API Gateway
cd services/api-gateway
pip install -r requirements.txt

# Ingestion
cd services/ingestion
pip install -r requirements.txt

# Metadata
cd services/metadata
pip install -r requirements.txt

# ASR
cd services/asr
pip install -r requirements.txt

# NLP-Post
cd services/nlp-post
pip install -r requirements.txt

# Embeddings
cd services/embeddings
pip install -r requirements.txt

# Search
cd services/search
pip install -r requirements.txt
```

### 3. Run Services

Start services in order:

```bash
# Terminal 1: Ingestion
cd services/ingestion
python app.py

# Terminal 2: Metadata
cd services/metadata
python app.py

# Terminal 3: ASR
cd services/asr
python app.py

# Terminal 4: ASR streaming (WebSocket)
cd services/asr
python streaming_server.py

# Terminal 5: NLP-Post
cd services/nlp-post
python app.py

# Terminal 6: Embeddings
cd services/embeddings
python app.py

# Terminal 7: API Gateway
cd services/api-gateway
python app.py
```

---

## Event Flow

1. **Ingestion** receives audio → stores in MinIO → publishes `recording.ingested`
2. **ASR API/worker** subscribes to `recording.ingested` → transcribes → publishes `transcription.completed`
3. **ASR Streaming** accepts WS sessions → emits `transcription.completed`
4. **NLP-Post** subscribes to `transcription.completed` → normalizes → publishes `nlp.postprocessed`
5. **Embeddings** subscribes to `nlp.postprocessed` → generates embeddings → indexes in Qdrant

---

## Migration Steps

### Phase 1: Infrastructure (Complete)
- ✅ Docker Compose + Helm chart
- ✅ Service implementations & event schemas
- ✅ OpenTelemetry hooks

### Phase 2: Extract Services
- [ ] Move ingestion logic from monolith
- [ ] Move metadata DB to Metadata service
- [ ] Move ASR to ASR service
- [ ] Move NLP to NLP-Post service

### Phase 3: Event-Driven
- [ ] Wire up event handlers
- [ ] Implement retry logic
- [ ] Add dead-letter queues

### Phase 4: Production
- [ ] Integrate Conformer/FastConformer + ByT5 models
- [ ] Wire MinIO read/write paths
- [ ] Harden JWT/ratelimit policies
- [ ] Deploy OTEL collector, Prometheus/Grafana

---

## Next Steps

See individual service READMEs for implementation details.

