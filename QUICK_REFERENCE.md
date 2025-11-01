# AI2Text ASR - Quick Reference

## 🚀 Getting Started (30 seconds)

```bash
# 1. Initialize
make init

# 2. Start everything
make up

# 3. Run migrations
make migrate

# 4. Test it
make health
```

## 📋 Common Commands

| Command | What it does |
|---------|--------------|
| `make up` | Start all services |
| `make down` | Stop all services |
| `make logs` | View all logs |
| `make ps` | Show service status |
| `make health` | Check all services |
| `make test-e2e` | Run end-to-end tests |
| `make restart` | Restart everything |
| `make clean` | Clean up Docker |

## 🌐 Service URLs

| Service | URL | Port |
|---------|-----|------|
| API Gateway | http://localhost:8080 | 8080 |
| Ingestion | http://localhost:8001 | 8001 |
| ASR Streaming | ws://localhost:8003 | 8003 |
| Metadata | http://localhost:8002 | 8002 |
| NLP-Post | http://localhost:8004 | 8004 |
| Embeddings | http://localhost:8005 | 8005 |
| Search | http://localhost:8006 | 8006 |
| MinIO Console | http://localhost:9001 | 9001 |
| Qdrant Dashboard | http://localhost:6333 | 6333 |

**MinIO credentials**: `minio` / `minio123`

## 🔑 API Quick Examples

### Upload Audio

```bash
curl -X POST http://localhost:8080/v1/ingest \
  -H "Authorization: Bearer dev" \
  -F "file=@audio.wav"
```

### Get Transcript

```bash
curl http://localhost:8080/v1/transcripts/{audio_id} \
  -H "Authorization: Bearer dev"
```

### Search

```bash
curl "http://localhost:8080/v1/search?q=xin+chào" \
  -H "Authorization: Bearer dev"
```

### Normalize Vietnamese Text

```bash
curl -X POST http://localhost:8004/v1/nlp/normalize \
  -H "Content-Type: application/json" \
  -d '{"text": "xin chao", "restore_diacritics": true}'
```

## 📊 Data Flow

```
1. Upload → Ingestion → MinIO (object storage)
                     ↓
2. RecordingIngested event → NATS
                     ↓
3. ASR Worker → Transcribes audio
                     ↓
4. TranscriptionCompleted event → NATS
                     ↓
5. NLP-Post → Normalizes Vietnamese text
                     ↓
6. NLPPostprocessed event → NATS
                     ↓
7. Metadata → Stores in PostgreSQL
   Embeddings → Indexes in Qdrant
                     ↓
8. Search → Semantic search available
```

## 🐛 Troubleshooting

### Services won't start
```bash
make down
make up
```

### Check specific service logs
```bash
docker compose -f infra/docker-compose.yml logs -f <service-name>
# Examples: api-gateway, asr, metadata, nlp-post
```

### Database issues
```bash
make migrate-fresh  # Reset and re-create DB
```

### Port already in use
```bash
# Find and kill process using port
lsof -ti:8080 | xargs kill -9  # Replace 8080 with your port
```

## 🧪 Testing

### Full test suite
```bash
make test-e2e
```

### Individual tests
```bash
pytest tests/e2e/test_flow.py::test_asr_websocket_streaming -v
pytest tests/e2e/test_flow.py::test_nlp_normalization -v
```

### Manual WebSocket test
```bash
# Install dependencies
pip install websockets

# Run test
python - <<'EOF'
import asyncio, websockets, json, base64

async def test():
    async with websockets.connect("ws://localhost:8003/v1/asr/stream") as ws:
        await ws.send(json.dumps({"type":"start","audio_format":{"sample_rate":16000,"channels":1,"encoding":"pcm16"}}))
        msg = await ws.recv()
        print(f"Received: {msg}")
        await ws.send(json.dumps({"type":"frame","base64":base64.b64encode(b'\x00'*32000).decode()}))
        await ws.send(json.dumps({"type":"end"}))
        async for msg in ws:
            print(msg)
            if "final" in msg:
                break

asyncio.run(test())
EOF
```

## 📁 Project Structure

```
AI2Text frist/
├── services/           # Microservices
│   ├── api-gateway/
│   ├── ingestion/
│   ├── asr/
│   ├── metadata/
│   ├── nlp-post/
│   ├── embeddings/
│   └── search/
├── infra/
│   ├── docker-compose.yml
│   └── helm/          # Kubernetes deployment
├── tests/
│   └── e2e/           # End-to-end tests
├── Makefile           # Common commands
└── env.example        # Environment template
```

## 🔧 Configuration Files

| File | Purpose |
|------|---------|
| `env.example` | Environment variables template |
| `infra/docker-compose.yml` | Service definitions |
| `services/*/Dockerfile` | Service container definitions |
| `services/metadata/migrations/` | Database schema |
| `Makefile` | Convenience commands |

## 🎯 Next Actions

### For Development
1. ✅ Run `make dev-setup` for complete environment
2. ✅ Run `make test-e2e` to verify everything works
3. ✅ Check `make health` periodically

### For Production
1. Replace JWT dev key in `.env`
2. Use managed PostgreSQL (not Docker)
3. Use S3 instead of MinIO
4. Enable TLS/SSL
5. Set up monitoring (Prometheus/Grafana)
6. Configure proper CORS origins
7. Set resource limits in Kubernetes

### To Integrate Real Models
1. Update `services/asr/streaming_server.py` with your ASR model
2. Update `services/nlp-post/app.py` with Vietnamese NLP model
3. Update `services/embeddings/app.py` with your embedding model

## 📚 Documentation

- **Full Setup**: See `MICROSERVICES_SETUP_GUIDE.md`
- **Architecture**: See `docs/architecture/MICROSERVICES_ARCHITECTURE.md`
- **API Docs**: Visit http://localhost:8080/docs when running

## 🆘 Need Help?

1. Check logs: `make logs`
2. Check health: `make health`
3. Restart services: `make restart`
4. Clean and rebuild: `make clean && make up && make migrate`
5. Read full documentation: `MICROSERVICES_SETUP_GUIDE.md`

## 💡 Tips

- Use `make logs-<service>` for specific service logs (e.g., `make logs-asr`)
- All services auto-restart on failure (Docker `restart: unless-stopped`)
- JWT token for dev: `Bearer dev`
- MinIO console is useful for viewing uploaded audio files
- Qdrant dashboard shows indexed vectors in real-time

