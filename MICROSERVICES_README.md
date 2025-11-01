# Microservices Architecture

## ✅ **Microservices Structure Created!**

Your project has been restructured into a microservices architecture following the blueprint.

---

## 📁 **New Structure**

```
project-root/
├── services/
│   ├── api-gateway/          # Single entry point
│   ├── ingestion/            # Audio upload & storage
│   ├── metadata/              # PostgreSQL metadata
│   ├── asr/                   # ASR transcription
│   ├── nlp-post/              # Vietnamese NLP
│   ├── embeddings/            # Embedding generation
│   ├── search/                # Semantic search
│   └── training-orchestrator/ # Training jobs
│
├── libs/common/               # Shared schemas & events
│   ├── schemas/               # API schemas
│   └── events/                # Event definitions
│
├── infra/                     # Infrastructure
│   └── docker-compose.yml     # All services
│
└── [existing folders...]       # Keep your existing code
```

---

## 🚀 **Quick Start**

### 1. Start Infrastructure

```bash
cd infra
docker-compose up -d
```

This starts:
- ✅ NATS (event bus) - port 4222
- ✅ PostgreSQL (metadata) - port 5432
- ✅ MinIO (object storage) - ports 9000, 9001
- ✅ Qdrant (vector DB) - port 6333

### 2. Install Dependencies

```bash
pip install -r requirements/microservices.txt
```

### 3. Run Services

Each service can run independently:

```bash
# Ingestion Service
cd services/ingestion
python app.py

# Metadata Service
cd services/metadata
python app.py

# ASR Service
cd services/asr
python app.py

# API Gateway (runs on port 8000)
cd services/api-gateway
python app.py
```

---

## 🔄 **Event Flow**

1. **Frontend** → API Gateway → Ingestion
2. **Ingestion** → Stores audio → Publishes `recording.ingested`
3. **ASR** → Subscribes to events → Transcribes → Publishes `transcription.completed`
4. **NLP-Post** → Subscribes → Normalizes → Publishes `nlp.postprocessed`
5. **Embeddings** → Subscribes → Generates embeddings → Indexes in Qdrant

---

## 📋 **Next Steps**

### **Phase 1: Complete Service Implementation**
- [ ] Integrate existing ASR model into ASR service
- [ ] Integrate NLP models into NLP-Post service
- [ ] Implement MinIO upload/download in Ingestion
- [ ] Complete Metadata service database operations

### **Phase 2: Event-Driven Integration**
- [ ] Wire up all event handlers
- [ ] Add retry logic
- [ ] Implement dead-letter queues
- [ ] Add event replay capability

### **Phase 3: Production Readiness**
- [ ] Add authentication to API Gateway
- [ ] Add monitoring (Prometheus/Grafana)
- [ ] Add tracing (OpenTelemetry)
- [ ] Set up CI/CD pipelines

---

## 📚 **Documentation**

- **Migration Guide**: `docs/MICROSERVICES_MIGRATION.md`
- **Event Schemas**: `libs/common/events/schemas.py`
- **API Schemas**: `libs/common/schemas/api.py`

---

## 🎯 **Benefits**

1. **Decoupled Services** - Each service can scale independently
2. **Event-Driven** - Loose coupling via events
3. **Proper Data Planes** - Structured (PostgreSQL), Unstructured (MinIO), Vector (Qdrant)
4. **Vietnamese NLP** - First-class service for text quality
5. **Scalable** - Ready for production deployment

---

**Your microservices architecture is ready!** 🚀

Start with `infra/docker-compose.yml` to get infrastructure running, then implement each service using your existing code.

