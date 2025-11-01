#!/usr/bin/env bash
# Bootstrap script - bring up infrastructure, run migrations, and prepare dev environment

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"

echo "[bootstrap] Starting AI2Text ASR infrastructure..."

# 1) Ensure .env exists
if [ ! -f "${ROOT}/.env" ]; then
  if [ -f "${ROOT}/env.example" ]; then
    cp "${ROOT}/env.example" "${ROOT}/.env"
    echo "[bootstrap] ✓ Created .env from env.example"
  else
    echo "[bootstrap] ⚠ Warning: No env.example found, using defaults"
  fi
else
  echo "[bootstrap] ✓ .env already exists"
fi

# 2) Start base infrastructure (no application services yet)
echo "[bootstrap] Starting infrastructure services (NATS, PostgreSQL, MinIO, Qdrant)..."
docker compose -f "${ROOT}/infra/docker-compose.yml" up -d nats postgres minio qdrant

# 3) Wait for services to be ready
echo "[bootstrap] Waiting for services to be ready..."
max_attempts=60
attempt=0

check_service() {
  local port=$1
  nc -z localhost "$port" >/dev/null 2>&1
}

while [ $attempt -lt $max_attempts ]; do
  if check_service 5432 && check_service 9000 && check_service 6333 && check_service 4222; then
    echo "[bootstrap] ✓ All infrastructure services are ready"
    break
  fi
  attempt=$((attempt + 1))
  sleep 1
  echo -n "."
done

if [ $attempt -eq $max_attempts ]; then
  echo "[bootstrap] ✗ Timeout waiting for services"
  exit 1
fi

# 4) Run database migrations
echo "[bootstrap] Running database migrations..."
if command -v make >/dev/null 2>&1 && [ -f "${ROOT}/Makefile" ]; then
  make -C "${ROOT}" migrate
else
  echo "[bootstrap] Running migration directly..."
  docker run --rm -i --network asr-network \
    -e PGPASSWORD=postgres postgres:16 \
    psql -h postgres -U postgres -d asrmeta < "${ROOT}/services/metadata/migrations/001_init.sql" 2>&1 | grep -v "already exists" || true
fi
echo "[bootstrap] ✓ Database migrations completed"

# 5) Initialize Qdrant collection
echo "[bootstrap] Initializing Qdrant collection..."
curl -fsS -X PUT http://localhost:6333/collections/texts \
  -H 'Content-Type: application/json' \
  -d '{"vectors":{"size":768,"distance":"Cosine"}}' >/dev/null 2>&1 || {
    # Collection might already exist
    echo "[bootstrap] ✓ Qdrant collection ready"
  }
echo "[bootstrap] ✓ Qdrant collection initialized"

# 6) Create MinIO buckets if needed
echo "[bootstrap] Initializing MinIO buckets..."
docker run --rm --network asr-network \
  -e MC_HOST_myminio=http://minio:minio123@minio:9000 \
  minio/mc:latest \
  mb myminio/audio myminio/transcripts --ignore-existing 2>/dev/null || true
echo "[bootstrap] ✓ MinIO buckets ready"

# 7) Configure NATS JetStream (if streams.json exists)
if [ -f "${ROOT}/infra/nats/streams.json" ]; then
  echo "[bootstrap] Configuring NATS JetStream..."
  if command -v nats >/dev/null 2>&1; then
    nats context add dev --server nats://127.0.0.1:4222 --select >/dev/null 2>&1 || true
    nats stream add --config "${ROOT}/infra/nats/streams.json" >/dev/null 2>&1 || {
      echo "[bootstrap] ⚠ NATS streams config skipped (may already exist)"
    }
  else
    echo "[bootstrap] ⚠ NATS CLI not installed, skipping JetStream setup"
    echo "           Install from: https://docs.nats.io/nats-concepts/jetstream"
  fi
fi

# 8) Generate a dev JWT token
echo ""
echo "=========================================="
echo "🎉 Infrastructure bootstrap complete!"
echo "=========================================="
echo ""
echo "📋 Services Status:"
echo "  - PostgreSQL:  ✓ Running on :5432"
echo "  - MinIO:       ✓ Running on :9000 (console :9001)"
echo "  - Qdrant:      ✓ Running on :6333"
echo "  - NATS:        ✓ Running on :4222"
echo ""
echo "🔑 Development JWT Token:"
echo "=========================================="

# Generate JWT token
python3 - <<'PY' 2>/dev/null || echo "Install PyJWT: pip install PyJWT"
import time
try:
    import jwt
    token = jwt.encode({
        "sub": "dev-user",
        "iat": int(time.time()),
        "exp": int(time.time()) + 86400  # 24 hours
    }, "dev", algorithm="HS256")
    print(f"Bearer {token}")
    print("")
    print("Use this token for API requests:")
    print(f'Authorization: Bearer {token}')
except ImportError:
    print("PyJWT not installed. Run: pip install PyJWT")
PY

echo ""
echo "=========================================="
echo "🚀 Next Steps:"
echo "=========================================="
echo "1. Start application services:"
echo "   docker compose -f infra/docker-compose.yml up -d"
echo ""
echo "2. View logs:"
echo "   make logs"
echo "   # or: docker compose -f infra/docker-compose.yml logs -f"
echo ""
echo "3. Check health:"
echo "   make health"
echo ""
echo "4. Run tests:"
echo "   make test-e2e"
echo ""
echo "5. Access UIs:"
echo "   - MinIO Console: http://localhost:9001 (minio/minio123)"
echo "   - Qdrant Dashboard: http://localhost:6333/dashboard"
echo ""

