# AI2Text Multi-Project Quick Start

Get AI2Text running in 5 minutes.

## 🚀 Prerequisites

- Docker Desktop or Kubernetes cluster
- Helm 3.x
- kubectl
- Python 3.11+

## ⚡ Fast Track (Local Development)

### Option 1: Docker Compose (Simplest)

```bash
cd projects/ai2text-platform

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f gateway

# Test API
curl http://localhost:8080/health
```

### Option 2: Kubernetes (Recommended)

```bash
# 1. Add Helm repos
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add nats https://nats-io.github.io/k8s/helm/charts/
helm repo update

# 2. Install AI2Text
cd projects/ai2text-platform
helm upgrade --install ai2text ./helm/charts/ai2text \
  -f ./helm/values/dev/values.yaml \
  --namespace ai2text \
  --create-namespace

# 3. Wait for pods
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=ai2text -n ai2text --timeout=300s

# 4. Port-forward gateway
kubectl port-forward -n ai2text svc/ai2text-gateway 8080:8080

# 5. Test
curl http://localhost:8080/health
```

## 📦 What You Get

- ✅ API Gateway (port 8080)
- ✅ Ingestion service
- ✅ ASR service (batch + streaming)
- ✅ NLP post-processing
- ✅ Embeddings service
- ✅ Search API
- ✅ Metadata API
- ✅ PostgreSQL database
- ✅ NATS messaging
- ✅ MinIO object storage
- ✅ Qdrant vector database

## 🧪 Quick Test

### 1. Upload Audio

```bash
curl -X POST http://localhost:8080/ingest \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@sample.wav" \
  -F "language=vi"
```

### 2. Check Status

```bash
curl http://localhost:8080/metadata/{recording_id}
```

### 3. Search Transcripts

```bash
curl "http://localhost:8080/search?q=hello&limit=10"
```

## 🛠️ Development

### Build a Service

```bash
cd projects/services/gateway
make docker-build
```

### Run Tests

```bash
cd projects/services/gateway
pytest
```

### Validate Contracts

```bash
cd projects/ai2text-contracts/codegen
make validate
```

## 📊 Access Dashboards

```bash
# Grafana
kubectl port-forward -n ai2text svc/ai2text-grafana 3000:3000
# http://localhost:3000 (admin/admin)

# MinIO Console
kubectl port-forward -n ai2text svc/ai2text-minio 9001:9001
# http://localhost:9001 (minioadmin/minioadmin)

# Qdrant
kubectl port-forward -n ai2text svc/ai2text-qdrant 6333:6333
# http://localhost:6333/dashboard
```

## 🐛 Troubleshooting

### Pods not starting?

```bash
kubectl get pods -n ai2text
kubectl describe pod <pod-name> -n ai2text
kubectl logs <pod-name> -n ai2text
```

### Service unreachable?

```bash
kubectl get svc -n ai2text
kubectl port-forward -n ai2text svc/<service-name> <port>:<port>
```

### Clean restart

```bash
helm uninstall ai2text -n ai2text
kubectl delete namespace ai2text
# Then reinstall
```

## 🎯 Next Steps

1. **Review Architecture**: Read [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
2. **Explore Services**: Check individual service READMEs in `services/`
3. **Review Contracts**: See `ai2text-contracts/`
4. **Customize Config**: Edit `helm/values/dev/values.yaml`
5. **Deploy to Prod**: Follow [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

## 📚 Documentation

- [Full Deployment Guide](DEPLOYMENT_GUIDE.md)
- [Multi-Project Architecture](MULTI_PROJECT_GUIDE.md)
- [API Contracts](ai2text-contracts/README.md)
- [Common Library](ai2text-common/README.md)

---

**Need help?** Check the main [README.md](README.md) or service-specific documentation.

