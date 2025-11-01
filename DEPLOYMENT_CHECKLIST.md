# AI2Text ASR - Deployment Checklist

## 🎯 Pre-Deployment Checklist

### Development Environment ✅
- [ ] Code tested locally
- [ ] All tests passing (`make test-e2e`)
- [ ] Services healthy (`make health`)
- [ ] Documentation up to date

### Infrastructure Requirements
- [ ] Kubernetes cluster provisioned (or Docker host)
- [ ] PostgreSQL database instance
- [ ] S3 bucket or object storage
- [ ] Vector database (Qdrant) instance
- [ ] Message queue (NATS) cluster
- [ ] SSL/TLS certificates obtained
- [ ] Domain name configured
- [ ] Monitoring stack ready

## 🔐 Security Configuration

### 1. JWT Authentication
**Priority**: 🔴 Critical

```bash
# Generate RSA key pair
openssl genrsa -out jwt-private.pem 2048
openssl rsa -in jwt-private.pem -pubout -out jwt-public.pem

# Update .env
JWT_PUBLIC_KEY=$(cat jwt-public.pem)
JWT_ALGO=RS256
```

- [ ] Generated RSA key pair
- [ ] Private key stored securely (Kubernetes Secret / AWS Secrets Manager)
- [ ] Public key configured in API Gateway
- [ ] Dev token removed from code

### 2. Service Secrets
**Priority**: 🔴 Critical

```bash
# Generate secure passwords
POSTGRES_PASSWORD=$(openssl rand -base64 32)
MINIO_SECRET_KEY=$(openssl rand -base64 32)
```

- [ ] Database password changed from default
- [ ] Object storage credentials changed
- [ ] Secrets stored in vault (not in .env)
- [ ] Environment variables injected securely

### 3. Network Security
**Priority**: 🟠 High

- [ ] Internal services not exposed publicly
- [ ] Only API Gateway has public endpoint
- [ ] Firewall rules configured
- [ ] VPC/network isolation enabled
- [ ] Service mesh configured (optional)

## 🗄️ Database Setup

### 1. PostgreSQL Configuration
**Priority**: 🔴 Critical

```bash
# Create production database
createdb -h your-db-host -U postgres asrmeta_prod

# Run migrations
DATABASE_URL=postgresql://user:pass@your-db-host:5432/asrmeta_prod \
  psql -f services/metadata/migrations/001_init.sql

# Create read replica (optional)
# Configure connection pooling (PgBouncer)
```

- [ ] Production database created
- [ ] Migrations applied
- [ ] Connection pooling configured
- [ ] Backup strategy implemented
- [ ] Read replicas configured (if needed)
- [ ] Monitoring enabled

### 2. Database Backup Strategy
**Priority**: 🟠 High

- [ ] Automated daily backups
- [ ] Point-in-time recovery enabled
- [ ] Backup retention policy (30 days recommended)
- [ ] Backup restoration tested
- [ ] Disaster recovery plan documented

## 📦 Object Storage Setup

### 1. S3 Configuration
**Priority**: 🔴 Critical

```bash
# Create S3 bucket
aws s3 mb s3://ai2text-asr-prod

# Configure lifecycle rules
aws s3api put-bucket-lifecycle-configuration \
  --bucket ai2text-asr-prod \
  --lifecycle-configuration file://s3-lifecycle.json

# Enable versioning
aws s3api put-bucket-versioning \
  --bucket ai2text-asr-prod \
  --versioning-configuration Status=Enabled
```

- [ ] Production bucket created
- [ ] IAM roles configured
- [ ] Lifecycle rules set (auto-delete old files)
- [ ] Versioning enabled
- [ ] Cross-region replication (optional)
- [ ] CloudFront CDN configured (optional)

### 2. Object Storage Structure

```
s3://ai2text-asr-prod/
├── raw/                  # Raw audio uploads
├── transcripts/          # Transcription results
├── models/              # Model artifacts
└── backups/             # Database backups
```

## 🔍 Vector Database Setup

### 1. Qdrant Configuration
**Priority**: 🟠 High

```bash
# Create collection with proper settings
curl -X PUT "https://qdrant.your-domain.com/collections/texts" \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": {
      "size": 768,
      "distance": "Cosine"
    },
    "optimizers_config": {
      "indexing_threshold": 10000
    }
  }'
```

- [ ] Qdrant cluster deployed
- [ ] Collection created with correct vector size
- [ ] Indexing threshold configured
- [ ] Persistent storage configured
- [ ] Backup strategy implemented

## 📡 Message Queue Setup

### 1. NATS Configuration
**Priority**: 🟠 High

```bash
# Deploy NATS cluster with JetStream
nats-server --jetstream --store_dir=/data/nats

# Create streams
nats stream add RECORDINGS --subjects "recording.*"
nats stream add TRANSCRIPTIONS --subjects "transcription.*"
nats stream add NLP --subjects "nlp.*"
```

- [ ] NATS cluster deployed (3+ nodes recommended)
- [ ] JetStream enabled
- [ ] Streams created
- [ ] Persistent storage configured
- [ ] Monitoring enabled

## 🚀 Service Deployment

### 1. Container Registry
**Priority**: 🔴 Critical

```bash
# Tag images for production
docker tag ai2text/api-gateway:latest your-registry.com/api-gateway:v1.0.0
docker tag ai2text/ingestion:latest your-registry.com/ingestion:v1.0.0
# ... repeat for all services

# Push to registry
docker push your-registry.com/api-gateway:v1.0.0
```

- [ ] Container registry configured (ECR, GCR, Docker Hub)
- [ ] All images tagged with semantic versions
- [ ] Images pushed to registry
- [ ] Image scanning enabled
- [ ] Image signing configured (optional)

### 2. Kubernetes Deployment
**Priority**: 🔴 Critical

```bash
# Create namespace
kubectl create namespace ai2text-prod

# Create secrets
kubectl create secret generic ai2text-secrets \
  --from-literal=jwt-public-key="$(cat jwt-public.pem)" \
  --from-literal=db-password="$POSTGRES_PASSWORD" \
  --namespace ai2text-prod

# Deploy services
helm upgrade --install ai2text infra/helm/ai-stt \
  -f infra/helm/ai-stt/values.prod.yaml \
  --namespace ai2text-prod
```

- [ ] Kubernetes namespace created
- [ ] Secrets created
- [ ] ConfigMaps created
- [ ] Helm chart deployed
- [ ] Services verified running
- [ ] Health checks passing

### 3. Resource Limits
**Priority**: 🟠 High

Update `values.prod.yaml`:

```yaml
resources:
  api-gateway:
    requests:
      cpu: 100m
      memory: 128Mi
    limits:
      cpu: 500m
      memory: 512Mi
  
  asr:
    requests:
      cpu: 2000m
      memory: 4Gi
      nvidia.com/gpu: 1
    limits:
      cpu: 4000m
      memory: 8Gi
      nvidia.com/gpu: 1
```

- [ ] Resource requests configured
- [ ] Resource limits set
- [ ] GPU resources allocated (for ASR)
- [ ] Horizontal Pod Autoscaling configured
- [ ] Pod disruption budgets set

## 🌐 Networking & DNS

### 1. Ingress Configuration
**Priority**: 🔴 Critical

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ai2text-ingress
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.ai2text.com
    secretName: ai2text-tls
  rules:
  - host: api.ai2text.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: api-gateway
            port:
              number: 8080
```

- [ ] Ingress controller deployed
- [ ] SSL/TLS certificates configured
- [ ] DNS A record pointing to load balancer
- [ ] HTTPS redirect enabled
- [ ] CORS headers configured

### 2. Load Balancer
**Priority**: 🟠 High

- [ ] External load balancer configured
- [ ] Health checks enabled
- [ ] Connection draining configured
- [ ] WebSocket support enabled (for ASR streaming)
- [ ] Rate limiting at LB level (optional)

## 📊 Monitoring & Observability

### 1. Metrics Collection
**Priority**: 🟠 High

```yaml
# Prometheus ServiceMonitor
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: ai2text-services
spec:
  selector:
    matchLabels:
      app: ai2text
  endpoints:
  - port: metrics
    interval: 30s
```

- [ ] Prometheus deployed
- [ ] Service monitors configured
- [ ] Metrics exported from all services
- [ ] Grafana dashboards created
- [ ] Alert rules configured

### 2. Logging
**Priority**: 🟠 High

- [ ] Centralized logging (ELK, Loki, CloudWatch)
- [ ] Log levels configured (INFO for prod)
- [ ] Log retention policy set
- [ ] Log aggregation working
- [ ] Search and analysis tools configured

### 3. Tracing
**Priority**: 🟡 Medium

- [ ] Jaeger or similar deployed
- [ ] OpenTelemetry configured
- [ ] Services instrumented
- [ ] Trace sampling configured (1% recommended)
- [ ] Trace retention policy set

## 🔔 Alerting

### 1. Alert Rules
**Priority**: 🟠 High

```yaml
# Example Prometheus alert
- alert: HighErrorRate
  expr: |
    rate(http_requests_total{status=~"5.."}[5m]) > 0.05
  for: 5m
  annotations:
    summary: "High error rate detected"
```

- [ ] Alert rules defined
- [ ] Alert channels configured (PagerDuty, Slack, email)
- [ ] On-call rotation set up
- [ ] Runbooks created for common alerts
- [ ] Alert fatigue minimized

### 2. Key Alerts

- [ ] Service down
- [ ] High error rate (>5%)
- [ ] High latency (p95 >1s)
- [ ] Database connection issues
- [ ] Disk space low (<10%)
- [ ] Memory usage high (>90%)
- [ ] Queue backlog growing

## 🧪 Testing in Production

### 1. Smoke Tests
**Priority**: 🔴 Critical

```bash
# Run post-deployment tests
kubectl exec -it test-pod -- python3 - <<'EOF'
import requests
import sys

# Test API Gateway
r = requests.get("https://api.ai2text.com/health")
assert r.status_code == 200, "API Gateway health check failed"

# Test upload
files = {"file": ("test.wav", b"test", "audio/wav")}
r = requests.post(
    "https://api.ai2text.com/v1/ingest",
    files=files,
    headers={"Authorization": "Bearer <prod-token>"}
)
assert r.status_code == 200, "Upload failed"

print("✅ All smoke tests passed")
EOF
```

- [ ] Health endpoints responding
- [ ] Authentication working
- [ ] File upload working
- [ ] Database queries working
- [ ] Search working

### 2. Load Testing
**Priority**: 🟡 Medium

```bash
# Use k6 or similar
k6 run --vus 100 --duration 5m load-test.js
```

- [ ] Load tests executed
- [ ] Performance metrics collected
- [ ] Bottlenecks identified and fixed
- [ ] Auto-scaling tested
- [ ] Results documented

## 📈 Performance Tuning

### 1. Database Optimization
**Priority**: 🟠 High

- [ ] Indexes verified
- [ ] Query performance analyzed (EXPLAIN)
- [ ] Connection pooling tuned
- [ ] Vacuum and analyze scheduled
- [ ] Cache warming implemented

### 2. Application Optimization
**Priority**: 🟡 Medium

- [ ] Connection pools configured
- [ ] HTTP keep-alive enabled
- [ ] Compression enabled (gzip)
- [ ] Static assets cached
- [ ] Database query caching (Redis)

## 🔄 CI/CD Pipeline

### 1. Continuous Integration
**Priority**: 🟠 High

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: make test-all
      - name: Build images
        run: make build
```

- [ ] CI pipeline configured
- [ ] Tests run on every commit
- [ ] Docker images built automatically
- [ ] Security scanning integrated
- [ ] Code quality checks enabled

### 2. Continuous Deployment
**Priority**: 🟡 Medium

```yaml
# .github/workflows/cd.yml
name: CD
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to staging
        run: kubectl apply -f k8s/staging/
      - name: Run smoke tests
        run: make test-smoke-staging
      - name: Deploy to production
        if: success()
        run: kubectl apply -f k8s/production/
```

- [ ] Staging environment configured
- [ ] Automatic deployment to staging
- [ ] Manual approval for production
- [ ] Rollback procedure documented
- [ ] Blue-green or canary deployment strategy

## 🛡️ Disaster Recovery

### 1. Backup & Restore
**Priority**: 🔴 Critical

- [ ] Database backups tested
- [ ] Object storage backups tested
- [ ] Recovery time objective (RTO) defined
- [ ] Recovery point objective (RPO) defined
- [ ] Disaster recovery plan documented
- [ ] DR drill conducted

### 2. Failover Strategy
**Priority**: 🟠 High

- [ ] Multi-region deployment (optional)
- [ ] Database replication configured
- [ ] Object storage replication configured
- [ ] DNS failover configured
- [ ] Failover procedure documented

## 📝 Documentation

### 1. Operational Documentation
**Priority**: 🟠 High

- [ ] Runbooks created
- [ ] Architecture diagrams updated
- [ ] API documentation published
- [ ] Troubleshooting guide available
- [ ] Change log maintained

### 2. User Documentation
**Priority**: 🟡 Medium

- [ ] API usage examples
- [ ] Authentication guide
- [ ] Rate limiting documentation
- [ ] Error codes documented
- [ ] SLA documented

## ✅ Pre-Launch Verification

### Final Checks (24 hours before launch)

- [ ] All tests passing
- [ ] Monitoring dashboards working
- [ ] Alerts firing correctly
- [ ] Backups working
- [ ] DNS propagated
- [ ] SSL certificates valid
- [ ] Team trained on operations
- [ ] On-call schedule published
- [ ] Incident response plan ready
- [ ] Communication plan ready

### Launch Day Checklist

- [ ] Announce maintenance window
- [ ] Deploy to production
- [ ] Run smoke tests
- [ ] Monitor metrics for 2 hours
- [ ] Check error logs
- [ ] Verify all services healthy
- [ ] Announce launch complete
- [ ] Continue monitoring

## 🎉 Post-Launch

### First Week

- [ ] Monitor error rates daily
- [ ] Review performance metrics
- [ ] Optimize based on real traffic
- [ ] Fix any issues discovered
- [ ] Gather user feedback
- [ ] Update documentation

### First Month

- [ ] Review and optimize costs
- [ ] Analyze usage patterns
- [ ] Plan capacity scaling
- [ ] Review security posture
- [ ] Conduct post-launch retrospective

---

## 📞 Emergency Contacts

| Role | Name | Contact |
|------|------|---------|
| Tech Lead | [Name] | [Phone/Email] |
| DevOps Lead | [Name] | [Phone/Email] |
| Database Admin | [Name] | [Phone/Email] |
| Security Lead | [Name] | [Phone/Email] |

## 🚨 Rollback Procedure

If deployment fails:

```bash
# 1. Rollback Kubernetes deployment
helm rollback ai2text -n ai2text-prod

# 2. Verify services
kubectl get pods -n ai2text-prod

# 3. Check health
make health

# 4. Investigate logs
make logs

# 5. Document incident
# Create post-mortem document
```

---

**Use this checklist systematically. Check off items as you complete them. Don't skip steps!**

