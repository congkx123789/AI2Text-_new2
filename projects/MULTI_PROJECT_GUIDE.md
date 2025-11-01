# AI2Text Multi-Project Implementation Guide

This document describes the multi-project architecture that splits your monolithic microservices codebase into independently owned, versioned projects.

## 📋 Overview

The AI2Text system is now organized into **10 independent projects**:

1. **ai2text-contracts** - API & event contracts (OpenAPI/AsyncAPI)
2. **ai2text-common** - Shared library package
3. **ai2text-platform** - Infrastructure as code (Helm, migrations, NATS)
4. **ai2text-gateway** - API Gateway service
5. **ai2text-ingestion** - File upload service
6. **ai2text-asr** - ASR transcription service
7. **ai2text-nlp-post** - Vietnamese NLP processing
8. **ai2text-embeddings** - Vector embedding generation
9. **ai2text-search** - Semantic search service
10. **ai2text-metadata** - Metadata API service
11. **ai2text-train-orchestrator** - Training orchestration service

## 🗂️ Project Structure

```
projects/
├── ai2text-contracts/          # Contract definitions
│   ├── openapi/
│   ├── asyncapi/
│   └── codegen/
│
├── ai2text-common/             # Shared library
│   ├── ai2text_common/
│   │   ├── events.py
│   │   ├── schemas/
│   │   └── observability/
│   └── pyproject.toml
│
├── ai2text-platform/            # Infrastructure
│   ├── helm/
│   ├── migrations/
│   ├── nats/
│   └── observability/
│
└── services/                    # Individual services
    ├── gateway/
    ├── ingestion/
    ├── asr/
    ├── nlp-post/
    ├── embeddings/
    ├── search/
    ├── metadata/
    └── train-orchestrator/
```

## 🚀 Quick Start

### 1. Set Up Contracts

```bash
cd projects/ai2text-contracts/codegen
make validate  # Validate all contracts
make clients   # Generate typed clients
```

### 2. Install Common Library

```bash
cd projects/ai2text-common
pip install -e .
```

### 3. Run a Service Locally

```bash
cd projects/services/gateway
uv run python -m app.main
```

### 4. Deploy with Platform

```bash
cd projects/ai2text-platform
helm upgrade --install ai2text helm/charts/ai2text \
  -f helm/values/dev/values.yaml \
  --namespace ai2text-dev
```

## 📦 Service Template

Each service follows this structure:

```
service-name/
├── app/
│   ├── main.py          # FastAPI or worker entry
│   ├── handlers/        # Business logic
│   └── deps.py          # Dependencies
├── tests/
├── Dockerfile
├── Makefile
├── pyproject.toml
└── .github/workflows/
    └── ci.yml
```

## 🔄 Event Flow

1. **Frontend** → `gateway` → `ingestion`
2. **ingestion** → Stores file → Publishes `recording.ingested.v1`
3. **asr** → Subscribes → Transcribes → Publishes `transcription.completed.v1`
4. **nlp-post** → Subscribes → Normalizes → Publishes `nlp.postprocessed.v1`
5. **embeddings** → Subscribes → Generates → Indexes in Qdrant → Publishes `embeddings.created.v1`

## 🛠️ Development Workflow

### Adding a New Service

1. Copy `projects/services/.template` to `projects/services/new-service`
2. Update service name in all files
3. Implement handlers in `app/handlers/`
4. Add tests
5. Update contracts if needed
6. Add Helm chart to platform

### Making Changes

1. **API Changes**: Update `ai2text-contracts/openapi/*.yaml`, regenerate clients
2. **Shared Code**: Update `ai2text-common`, bump version
3. **Service Code**: Update service repo, test, deploy
4. **Infrastructure**: Update `ai2text-platform/helm/`

## 🔍 Contract Testing

All services must:
1. Pull contract specs from `ai2text-contracts`
2. Run contract tests in CI
3. Fail CI if contracts incompatible
4. Use generated clients/servers

```bash
# In service CI
git clone https://github.com/yourorg/ai2text-contracts ../contracts
cd ../contracts/codegen
make validate
```

## 📊 Data Plane Ownership

- **MinIO/S3**: Owned by `ingestion` (writes), `asr` (reads)
- **PostgreSQL**: Owned by `metadata` only
- **Qdrant**: Owned by `embeddings` (writes), `search` (reads)

## 🎯 Versioning Strategy

- **Contracts**: SemVer (breaking changes → major version)
- **Common Library**: SemVer (keep changes additive)
- **Services**: Docker tags `service:x.y.z`
- **Events**: Subject suffix `.vN` for versioning

## 🔐 Security

- JWT authentication (RS256) in gateway
- Secrets via Kubernetes Secrets
- Network policies in production
- Service mesh (optional)

## 📈 Observability

Each service exposes:
- `/health` - Health check
- `/metrics` - Prometheus metrics
- Structured logging (JSON)
- OpenTelemetry tracing

## 🚢 Deployment

### Development
```bash
helm upgrade --install ai2text helm/charts/ai2text \
  -f helm/values/dev/values.yaml \
  --namespace ai2text-dev
```

### Production
```bash
helm upgrade --install ai2text helm/charts/ai2text \
  -f helm/values/prod/values.yaml \
  --namespace ai2text-prod
```

## 📝 Migration Plan

**Sprint 1**: Contracts & Common
- Extract contracts
- Create common library
- Setup CI

**Sprint 2**: Platform
- Move Helm charts
- Setup migrations
- Configure NATS streams

**Sprint 3**: Core Services
- Split gateway, asr, search
- Wire contract tests
- Deploy to dev

**Sprint 4**: Remaining Services
- Split remaining services
- Load tests
- Promote to stage

## 🔗 Next Steps

1. Review the contract specifications
2. Implement remaining services (asr, nlp-post, embeddings, search, metadata, train-orchestrator)
3. Set up CI/CD pipelines
4. Configure production infrastructure
5. Run load tests and validate SLOs

---

For detailed implementation of each service, see the README in each service directory.


