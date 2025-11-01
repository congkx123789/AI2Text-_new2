# AI2Text ASR - Complete Run Guide

## 🚀 End-to-End Setup (5 Minutes)

Follow these steps to get the complete microservices stack running.

### Prerequisites Check

```bash
# Check Docker
docker --version
# Should show: Docker version 20.x or higher

# Check Docker Compose
docker compose version
# Should show: Docker Compose version v2.x or higher

# Check Python (for token generation)
python3 --version
# Should show: Python 3.8 or higher

# Install PyJWT for token generation
pip install PyJWT
```

### Step 1: Bootstrap Infrastructure (1 minute)

```bash
# Make bootstrap script executable
chmod +x scripts/bootstrap.sh

# Run bootstrap (starts infra, runs migrations, creates buckets)
bash scripts/bootstrap.sh
```

**What this does:**
- ✅ Creates `.env` file from template
- ✅ Starts PostgreSQL, MinIO, Qdrant, NATS
- ✅ Runs database migrations
- ✅ Creates MinIO buckets (audio, transcripts)
- ✅ Initializes Qdrant collection
- ✅ Generates a dev JWT token

**Expected output:**
```
[bootstrap] Starting AI2Text ASR infrastructure...
[bootstrap] ✓ Created .env from env.example
[bootstrap] Starting infrastructure services...
[bootstrap] ✓ All infrastructure services are ready
[bootstrap] ✓ Database migrations completed
[bootstrap] ✓ Qdrant collection initialized
[bootstrap] ✓ MinIO buckets ready

🎉 Infrastructure bootstrap complete!

🔑 Development JWT Token:
==========================================
Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

Use this token for API requests:
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Save the JWT token** - you'll need it for API requests!

### Step 2: Start Application Services (2 minutes)

```bash
# Start all application services
docker compose -f infra/docker-compose.yml up -d

# Wait for services to be ready (takes 30-60 seconds)
sleep 60

# Check all services are running
docker compose -f infra/docker-compose.yml ps
```

**Expected services:**
```
NAME                  STATUS
nats                  running
postgres              running
minio                 running
qdrant                running
api-gateway           running
ingestion             running
asr                   running
metadata              running
nlp-post              running
embeddings            running
search                running
```

### Step 3: Verify Health (30 seconds)

```bash
# Check all services are healthy
make health

# Or manually:
curl http://localhost:8080/health
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8004/health
curl http://localhost:8005/health
curl http://localhost:8006/health
```

**All should return:** `{"status": "healthy", ...}`

### Step 4: Test the Pipeline (2 minutes)

#### A) Test WebSocket Streaming

```bash
# Test real-time streaming
python3 - <<'EOF'
import asyncio, websockets, json, base64

async def test_streaming():
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
        
        # Get acknowledgment
        ack = json.loads(await ws.recv())
        audio_id = ack["audio_id"]
        print(f"✓ Session started: {audio_id}")
        
        # Send 1 second of silence
        silence = b'\x00' * 32000
        await ws.send(json.dumps({
            "type": "frame",
            "base64": base64.b64encode(silence).decode()
        }))
        print("✓ Audio frame sent")
        
        # End session
        await ws.send(json.dumps({"type": "end"}))
        
        # Receive results
        async for msg in ws:
            result = json.loads(msg)
            print(f"✓ Received: {result['type']}")
            if result["type"] == "final":
                print(f"✓ Final transcript: {result['text']}")
                break

asyncio.run(test_streaming())
EOF
```

**Expected output:**
```
✓ Session started: aud_a1b2c3d4
✓ Audio frame sent
✓ Received: partial
✓ Received: final
✓ Final transcript: xin chào thế giới
```

#### B) Test Batch Upload

First, create a test audio file:

```bash
# Create a simple WAV file (1 second of silence)
python3 - <<'EOF'
import wave
import struct

# Create a minimal WAV file
with wave.open('test_audio.wav', 'w') as wav:
    wav.setnchannels(1)  # Mono
    wav.setsampwidth(2)  # 16-bit
    wav.setframerate(16000)  # 16kHz
    
    # 1 second of silence
    silence = struct.pack('<h', 0) * 16000
    wav.writeframes(silence)

print("✓ Created test_audio.wav")
EOF
```

Now upload it:

```bash
# Get your JWT token
TOKEN=$(python3 scripts/jwt_dev_token.py)

# Upload audio file
curl -X POST http://localhost:8080/v1/ingest \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@test_audio.wav"
```

**Expected response:**
```json
{
  "audio_id": "550e8400-e29b-41d4-a716-446655440000",
  "object_uri": "s3://audio/raw/550e8400-e29b-41d4-a716-446655440000.wav"
}
```

**Save the audio_id!**

#### C) Check Processing Pipeline

```bash
# Wait for async processing (3-5 seconds)
sleep 5

# Check logs to see the event flow
docker compose -f infra/docker-compose.yml logs --tail=50 asr nlp-post metadata
```

**You should see:**
```
asr         | [ASR] Processing 550e8400... from s3://audio/raw/...
asr         | [ASR] Transcribed: xin chào thế giới
asr         | [OK] ASR completed for 550e8400...

nlp-post    | [NLP] Processing 550e8400...
nlp-post    | [NLP] Original text: xin chào thế giới
nlp-post    | [NLP] Normalized text: xin chào thế giới
nlp-post    | [OK] NLP post-processing completed for 550e8400...

metadata    | [Metadata] Received transcription for 550e8400...
metadata    | [Metadata] Updating 550e8400... with NLP-processed text
metadata    | [OK] Metadata updated for 550e8400...
```

#### D) Retrieve Transcript

```bash
# Get transcript using the audio_id from step B
AUDIO_ID="your-audio-id-here"
TOKEN=$(python3 scripts/jwt_dev_token.py)

curl http://localhost:8080/v1/transcripts/$AUDIO_ID \
  -H "Authorization: Bearer $TOKEN" | jq
```

**Expected response:**
```json
{
  "audio_id": "550e8400-e29b-41d4-a716-446655440000",
  "text": "xin chào thế giới",
  "text_clean": "xin chào thế giới",
  "raw_json": null
}
```

### Step 5: Run Full Test Suite (1 minute)

```bash
# Install test dependencies
pip install pytest websockets httpx

# Run end-to-end tests
make test-e2e

# Or directly:
pytest tests/e2e/test_flow.py -v
```

---

## 🎯 Quick Command Reference

### Service Management

```bash
# Start everything
make up

# Stop everything
make down

# Restart
make restart

# View logs (all services)
make logs

# View specific service logs
make logs-asr
make logs-metadata
make logs-nlp
```

### Database

```bash
# Run migrations
make migrate

# Reset database
make migrate-fresh

# Connect to PostgreSQL
make shell-postgres

# Backup database
docker exec postgres pg_dump -U postgres asrmeta > backup.sql
```

### Testing

```bash
# Health checks
make health

# End-to-end tests
make test-e2e

# Smoke test
make smoke-test
```

### Utilities

```bash
# Generate JWT token
python3 scripts/jwt_dev_token.py

# View MinIO console
# Open: http://localhost:9001
# Login: minio / minio123

# View Qdrant dashboard
# Open: http://localhost:6333/dashboard
```

---

## 📊 Event Flow Visualization

When you upload an audio file, here's what happens:

```
1. Client uploads audio → API Gateway (JWT auth + rate limit)
                             ↓
2. API Gateway → Ingestion Service
                     ↓
3. Ingestion → MinIO (stores raw audio)
                     ↓
4. Ingestion → NATS (publishes "recording.ingested")
                     ↓
5. ASR Worker receives event
                     ↓
6. ASR Worker → Downloads audio from MinIO
                → Transcribes (currently stub)
                → Uploads transcript to MinIO
                → Publishes "transcription.completed"
                     ↓
7. NLP-Post receives event
                     ↓
8. NLP-Post → Normalizes Vietnamese text
              (xin chao → xin chào)
              → Publishes "nlp.postprocessed"
                     ↓
9. Metadata Service receives event
                     ↓
10. Metadata → Stores in PostgreSQL
                  ↓
11. Embeddings Service receives event
                  ↓
12. Embeddings → Generates vector
                → Indexes in Qdrant
                → Publishes "embeddings.indexed"
                  ↓
13. Done! Audio is searchable
```

**Timeline:** ~2-5 seconds from upload to searchable

---

## 🔧 Troubleshooting

### Services won't start

```bash
# Check Docker resources
docker system df

# Clean up if needed
make clean

# Restart from scratch
make down-volumes
make up
make migrate
```

### Port conflicts

```bash
# Check what's using a port
lsof -i :8080  # Replace with your port

# Kill the process
kill -9 <PID>
```

### Database connection issues

```bash
# Check PostgreSQL is running
docker compose -f infra/docker-compose.yml ps postgres

# View PostgreSQL logs
docker compose -f infra/docker-compose.yml logs postgres

# Reset database
make migrate-fresh
```

### NATS connection issues

```bash
# Check NATS is running
docker compose -f infra/docker-compose.yml ps nats

# Test NATS
curl http://localhost:8222/

# View NATS logs
docker compose -f infra/docker-compose.yml logs nats
```

### Services show "unhealthy"

```bash
# Wait a bit longer (services take 30-60s to start)
sleep 30
make health

# Check specific service logs
docker compose -f infra/docker-compose.yml logs <service-name>

# Restart specific service
docker compose -f infra/docker-compose.yml restart <service-name>
```

### No events flowing

```bash
# Check NATS is receiving events
docker compose -f infra/docker-compose.yml logs nats | grep recording

# Check ASR worker is subscribed
docker compose -f infra/docker-compose.yml logs asr | grep "listening"

# Check NLP is subscribed
docker compose -f infra/docker-compose.yml logs nlp-post | grep "Subscribed"
```

---

## 🎨 Web UIs

### MinIO Console
**URL:** http://localhost:9001  
**Credentials:** `minio` / `minio123`

**Use for:**
- Browse uploaded audio files
- View transcript JSON files
- Monitor storage usage
- Manage buckets

### Qdrant Dashboard
**URL:** http://localhost:6333/dashboard

**Use for:**
- View indexed vectors
- Monitor collection size
- Test vector search
- Inspect embeddings

### PostgreSQL (via CLI)
```bash
make shell-postgres
# Then: \dt to list tables, SELECT * FROM transcripts; etc.
```

---

## 🚀 Next Steps

### 1. Integrate Real ASR Model

Edit `services/asr/worker.py`:

```python
# Option A: Your custom LSTM model
from models.lstm_asr import LSTMASRModel
asr_model = LSTMASRModel.load_checkpoint("checkpoints/best_model.pt")

# In transcribe_audio():
result = asr_model.transcribe(audio_data)

# Option B: Whisper
import whisper
asr_model = whisper.load_model("base")

# In transcribe_audio():
result = asr_model.transcribe(audio_data)
```

### 2. Improve Vietnamese NLP

Edit `services/nlp-post/app.py`:

```python
# Option A: underthesea
from underthesea import word_tokenize

# Option B: ByT5
from transformers import T5ForConditionalGeneration
model = T5ForConditionalGeneration.from_pretrained("byt5-vietnamese")
```

### 3. Add Real Embeddings

Edit `services/embeddings/app.py`:

```python
# Load your Word2Vec model
from gensim.models import Word2Vec
model = Word2Vec.load("models/word2vec.model")

# Or use sentence transformers
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
```

---

## 📚 Documentation

- **Full Setup Guide:** `MICROSERVICES_SETUP_GUIDE.md`
- **Quick Reference:** `QUICK_REFERENCE.md`
- **Deployment:** `DEPLOYMENT_CHECKLIST.md`
- **Architecture:** `docs/architecture/MICROSERVICES_ARCHITECTURE.md`

---

## 💡 Tips

1. **Always check logs first:** `make logs`
2. **Use health checks:** `make health`
3. **JWT tokens expire in 24 hours** - regenerate with `python3 scripts/jwt_dev_token.py`
4. **Services auto-restart** on failure
5. **Database changes require migrations** - run `make migrate`
6. **For production:** Update `.env` with real credentials and JWT keys

---

**You're all set! 🎉**

Your AI2Text ASR microservices stack is now running end-to-end.

