# AI2Text Multi-Project Deployment Guide

Complete guide for deploying the AI2Text microservices platform.

## 📋 Prerequisites

- Kubernetes cluster (1.25+)
- Helm 3.x
- kubectl configured
- Docker registry access (GitHub Container Registry)
- Domain name (for production)

## 🚀 Quick Start (Development)

### 1. Install Common Library

```bash
cd ai2text-common
pip install -e .
```

### 2. Validate Contracts

```bash
cd ai2text-contracts/codegen
make validate
```

### 3. Deploy to Kubernetes (Dev)

```bash
cd ai2text-platform

# Add required Helm repos
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add nats https://nats-io.github.io/k8s/helm/charts/
helm repo add minio https://charts.min.io/
helm repo update

# Install AI2Text platform
helm upgrade --install ai2text \
  ./helm/charts/ai2text \
  -f ./helm/values/dev/values.yaml \
  --namespace ai2text-dev \
  --create-namespace
```

### 4. Verify Deployment

```bash
kubectl get pods -n ai2text-dev
kubectl get services -n ai2text-dev

# Check logs
kubectl logs -n ai2text-dev -l app.kubernetes.io/component=gateway
```

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        API Gateway                           │
│                    (Authentication, Routing)                 │
└────────────┬─────────────────────────────────────┬──────────┘
             │                                     │
   ┌─────────▼────────┐                 ┌─────────▼─────────┐
   │   Ingestion      │                 │     Search        │
   │   (Upload S3)    │                 │   (Qdrant Read)   │
   └────────┬─────────┘                 └───────────────────┘
            │
            │ NATS: recording.ingested.v1
            │
   ┌────────▼─────────┐
   │       ASR        │
   │  (Transcription) │
   └────────┬─────────┘
            │
            │ NATS: transcription.completed.v1
            │
   ┌────────▼─────────┐
   │    NLP-Post      │
   │  (Vietnamese NLP)│
   └────────┬─────────┘
            │
            │ NATS: nlp.postprocessed.v1
            │
   ┌────────▼─────────┐
   │   Embeddings     │
   │ (Qdrant Write)   │
   └──────────────────┘

Metadata Service (PostgreSQL) handles all recording metadata
Training Orchestrator manages ML model lifecycle
```

## 📦 Service Inventory

| Service | Purpose | Tech Stack | SLO |
|---------|---------|------------|-----|
| **Gateway** | API routing, auth, rate limiting | FastAPI, JWT | p95 < 150ms |
| **Ingestion** | File upload, S3 storage | FastAPI, MinIO | Upload < 2s |
| **ASR** | Speech recognition | FastAPI, WS | Batch < 0.5x RT |
| **NLP-Post** | Vietnamese text processing | Python, NATS | CPU-bound |
| **Embeddings** | Vector generation | Python, Qdrant | Write < 1s |
| **Search** | Semantic search | FastAPI, Qdrant | p95 < 50ms |
| **Metadata** | Recording metadata | FastAPI, Postgres | Write p95 < 40ms |
| **Training** | Model orchestration | Python, NATS | Async |

## 🔧 Configuration

### Environment Variables (Per Service)

#### Gateway
- `NATS_URL` - NATS connection string
- `JWT_SECRET` - JWT signing secret
- `LOG_LEVEL` - Logging level

#### ASR
- `MODEL_PATH` - Path to ASR model weights
- `NATS_URL` - NATS connection
- `LOG_LEVEL` - Logging level

#### Metadata
- `DATABASE_URL` - PostgreSQL connection string
- `NATS_URL` - NATS connection

#### Search
- `QDRANT_URL` - Qdrant server URL
- `QDRANT_COLLECTION` - Collection name (default: `transcripts`)

#### Embeddings
- `QDRANT_URL` - Qdrant server URL
- `MODEL_PATH` - Embedding model path
- `NATS_URL` - NATS connection

### Kubernetes Secrets

```bash
# Create secrets
kubectl create secret generic ai2text-secrets \
  --from-literal=jwt-secret=your-secret-here \
  --from-literal=db-password=your-db-password \
  --namespace ai2text-dev
```

## 🌍 Environment Deployments

### Development

```bash
helm upgrade --install ai2text ./helm/charts/ai2text \
  -f ./helm/values/dev/values.yaml \
  --namespace ai2text-dev \
  --create-namespace
```

**Features:**
- Single replica per service
- Minimal resources
- Local ingress (optional)
- No autoscaling

### Staging

```bash
helm upgrade --install ai2text ./helm/charts/ai2text \
  -f ./helm/values/stage/values.yaml \
  --namespace ai2text-stage \
  --create-namespace
```

**Features:**
- 1-2 replicas per service
- Medium resources
- TLS enabled
- Basic autoscaling

### Production

```bash
helm upgrade --install ai2text ./helm/charts/ai2text \
  -f ./helm/values/prod/values.yaml \
  --namespace ai2text-prod \
  --create-namespace
```

**Features:**
- 3+ replicas per service
- Horizontal Pod Autoscaling (HPA)
- High availability PostgreSQL
- Distributed MinIO
- Multi-zone Qdrant
- Ingress with TLS
- Prometheus + Grafana monitoring

## 🗄️ Database Migrations

### Run Migrations

```bash
# Port-forward to PostgreSQL
kubectl port-forward -n ai2text-dev svc/ai2text-postgresql 5432:5432

# Run migrations
psql postgresql://postgres:postgres@localhost:5432/asrmeta \
  -f ./migrations/metadata-db/V1__init.sql
```

### Automated Migrations (Flyway)

```bash
# Using Flyway
flyway -url=jdbc:postgresql://localhost:5432/asrmeta \
  -user=postgres \
  -password=postgres \
  -locations=filesystem:./migrations/metadata-db \
  migrate
```

## 📊 Monitoring

### Prometheus Metrics

All services expose metrics at `/metrics`:

```bash
# Port-forward gateway
kubectl port-forward -n ai2text-dev svc/ai2text-gateway 8080:8080

# Fetch metrics
curl http://localhost:8080/metrics
```

**Key Metrics:**
- `http_requests_total` - Total HTTP requests
- `http_request_duration_seconds` - Request latency
- `search_requests_total` - Search requests
- `db_operations_total` - Database operations

### Grafana Dashboards

Access Grafana:

```bash
kubectl port-forward -n ai2text-dev svc/ai2text-grafana 3000:3000
```

Default credentials: `admin` / `admin`

### Health Checks

```bash
# Check service health
curl http://gateway:8080/health
```

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

Each service has a `.github/workflows/ci.yml`:

1. **Test** - Run pytest with coverage
2. **Lint** - Run ruff, black, mypy
3. **Contract Tests** - Validate OpenAPI/AsyncAPI specs
4. **Build** - Build and push Docker image
5. **Deploy Dev** - Auto-deploy to dev on main push
6. **Deploy Prod** - Deploy on git tag (e.g., `v1.0.0`)

### Manual Deployment

```bash
# Build service
cd services/gateway
make docker-build

# Push to registry
make docker-push

# Update Helm deployment
helm upgrade ai2text ./helm/charts/ai2text \
  --set gateway.image.tag=new-version \
  --namespace ai2text-dev
```

## 🧪 Testing

### Unit Tests

```bash
cd services/gateway
pytest --cov=app --cov-report=html
```

### Integration Tests

```bash
cd ai2text-contracts
make validate

# Run contract tests
pytest tests/e2e/
```

### Load Testing

```bash
# Using k6
k6 run tests/load/search-load.js
```

## 🛡️ Security

### JWT Authentication

Generate dev token:

```bash
python scripts/jwt_dev_token.py
```

### Network Policies

Apply network policies:

```bash
kubectl apply -f helm/charts/ai2text/templates/network-policies.yaml
```

### Secrets Management

Use Kubernetes secrets or external secret managers:

```bash
# Sealed Secrets
kubeseal < secret.yaml > sealed-secret.yaml
kubectl apply -f sealed-secret.yaml
```

## 📈 Scaling

### Manual Scaling

```bash
# Scale gateway
kubectl scale deployment ai2text-gateway --replicas=5 -n ai2text-dev
```

### Autoscaling (HPA)

```yaml
# HPA is enabled by default in prod
gateway:
  autoscaling:
    enabled: true
    minReplicas: 3
    maxReplicas: 20
    targetCPUUtilizationPercentage: 70
```

## 🐛 Troubleshooting

### Pod not starting

```bash
kubectl describe pod <pod-name> -n ai2text-dev
kubectl logs <pod-name> -n ai2text-dev
```

### Service unreachable

```bash
kubectl get svc -n ai2text-dev
kubectl get endpoints -n ai2text-dev
```

### NATS connection issues

```bash
# Check NATS pod
kubectl logs -n ai2text-dev -l app.kubernetes.io/name=nats

# Test NATS connection
nats stream ls --server=nats://localhost:4222
```

### Database connection issues

```bash
# Check PostgreSQL
kubectl logs -n ai2text-dev -l app.kubernetes.io/name=postgresql

# Test connection
kubectl run -it --rm debug --image=postgres:15 --restart=Never -- \
  psql postgresql://postgres@ai2text-postgresql:5432/asrmeta
```

## 📚 Additional Resources

- [Multi-Project Guide](MULTI_PROJECT_GUIDE.md)
- [API Contracts](../ai2text-contracts/README.md)
- [Common Library](../ai2text-common/README.md)
- [Service Templates](../services/.template/README.md)

## 🎯 SLO Summary

| Service | Metric | Target | Measurement |
|---------|--------|--------|-------------|
| Gateway | p95 latency | < 150ms | Prometheus |
| Gateway | Availability | 99.9% | Uptime |
| Search | p95 latency | < 50ms | Prometheus |
| Metadata | Write p95 | < 40ms | Prometheus |
| ASR Batch | Speed | < 0.5x RT | Custom |
| ASR Stream | Partial latency | < 500ms | Prometheus |

## 🔄 Rollback

```bash
# Rollback to previous release
helm rollback ai2text -n ai2text-dev

# Rollback to specific revision
helm rollback ai2text 2 -n ai2text-dev
```

## 🎉 Success Checklist

- [ ] All pods running (`kubectl get pods`)
- [ ] Health checks passing (`/health` endpoints)
- [ ] Metrics exposed (`/metrics` endpoints)
- [ ] NATS streams created
- [ ] Database migrations applied
- [ ] Ingress configured (prod)
- [ ] TLS certificates valid (prod)
- [ ] Monitoring dashboards accessible
- [ ] Load tests passing SLOs

---

**Questions?** Check the [Multi-Project Guide](MULTI_PROJECT_GUIDE.md) or service READMEs.

