# Auto-Training Pipeline - Implementation Complete!

## 🚀 What You Now Have

A **fully automated Vietnamese ASR training pipeline** that:

1. **Auto-ingests audio** from hot folder and S3 inbox
2. **Transcodes to 16kHz mono WAV** using ffmpeg
3. **Processes through ASR pipeline** (transcription → NLP → embeddings)
4. **Builds datasets automatically** with speaker-level splits
5. **Triggers training** when 5+ hours of data accumulated
6. **Evaluates and promotes models** automatically
7. **Runs on cron schedule** (daily at 2 AM)

**Drop audio → Get trained models!** 🎉

---

## 📋 Implementation Details

### Phase A: Auto-Ingestion ✅

**Files Modified:**
- `env.example` - Added watcher configuration
- `services/ingestion/app.py` - Added background watcher + ffmpeg transcoding
- `services/ingestion/requirements.txt` - Added watchfiles + aiofiles
- `services/ingestion/Dockerfile` - Added ffmpeg
- `infra/docker-compose.yml` - Added hot folder volume mount

**Features:**
- ✅ Monitors `/mnt/inbox/**/*.{wav,mp3,m4a,flac}`
- ✅ Monitors `s3://audio/inbox/` prefix
- ✅ Auto-transcodes to 16kHz mono PCM WAV
- ✅ Uploads to `s3://audio/raw/{audio_id}.wav`
- ✅ Publishes `recording.ingested` events
- ✅ Moves processed files to `/mnt/processed/`

### Phase B: Dataset Builder ✅

**Files Modified:**
- `infra/nats/streams.json` - Added dataset.* and training.* subjects
- `configs/default.yaml` - Added dataset/split/training config
- `services/training-orchestrator/app.py` - Added manifest building logic

**Features:**
- ✅ Subscribes to `nlp.postprocessed` and `transcription.completed`
- ✅ Extracts text (prefers `text_clean` with Vietnamese diacritics)
- ✅ Calculates audio duration and basic metadata
- ✅ Implements speaker-level split (MD5 hash-based)
- ✅ Appends to `s3://datasets/manifests/{train|val}.jsonl`
- ✅ Monitors hours accumulated
- ✅ Publishes `dataset.ready` when threshold reached

### Phase C: Training Orchestrator ✅

**Files Modified:**
- `infra/docker-compose.yml` - Added training-orchestrator service
- `services/training-orchestrator/Dockerfile` - Updated to copy whole repo
- `services/training-orchestrator/requirements.txt` - Added MinIO + APScheduler
- `services/training-orchestrator/app.py` - Complete orchestrator logic

**Features:**
- ✅ Subscribes to `dataset.ready` events
- ✅ Downloads manifests from S3
- ✅ Launches training jobs with gradient accumulation + mixed precision
- ✅ Runs evaluation after training
- ✅ Promotes model if WER improves
- ✅ Publishes training events (`started`, `completed`, `model.promoted`)
- ✅ Cron scheduler for daily training

---

## 🏗️ Data Flow Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     DROP AUDIO FILES                     │
│                                                         │
│  1. Hot Folder: /mnt/inbox/*.mp3                        │
│  2. S3 Inbox: s3://audio/inbox/*.wav                    │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│         INGESTION WATCHER (Phase A)                     │
│                                                         │
│  • Auto-detect new files (30s polling)                  │
│  • ffmpeg transcode → 16kHz mono WAV                    │
│  • Upload: s3://audio/raw/{audio_id}.wav                │
│  • Publish: recording.ingested                          │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│         ASR PIPELINE (Existing)                         │
│                                                         │
│  • ASR Worker transcribes audio                         │
│  • NLP-Post adds Vietnamese diacritics                 │
│  • Metadata stores in PostgreSQL                       │
│  • Embeddings indexes in Qdrant                        │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│         DATASET BUILDER (Phase B)                       │
│                                                         │
│  • Monitors nlp.postprocessed events                    │
│  • Extracts text_clean (with diacritics)                │
│  • Calculates duration, assigns speaker split           │
│  • Appends to manifests/train.jsonl                     │
│  • Tracks hours accumulated                             │
│  • Publishes dataset.ready when ≥5 hours                │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│         TRAINING ORCHESTRATOR (Phase C)                 │
│                                                         │
│  • Receives dataset.ready events                        │
│  • Downloads manifests from S3                          │
│  • Launches training with AMP + GA                      │
│  • Evaluates model after training                       │
│  • Promotes if WER improves                             │
│  • Cron: Daily at 02:00                                 │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│         MODEL PROMOTION                                 │
│                                                         │
│  • Uploads to s3://models/asr/{run_id}/                 │
│  • Publishes model.promoted event                       │
│  • Ready for deployment                                 │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Copy environment config
cp env.example .env

# Create hot folder
mkdir -p inbox
```

### 2. Start Services
```bash
# Bootstrap infrastructure
bash scripts/bootstrap.sh

# Start all services (now includes training-orchestrator)
docker compose -f infra/docker-compose.yml up -d --build
```

### 3. Test Auto-Ingestion
```bash
# Drop an audio file
cp test_audio.mp3 inbox/

# Watch logs for processing
docker compose -f infra/docker-compose.yml logs -f ingestion training-orchestrator

# Check MinIO for processed WAV
# Open: http://localhost:9001 (minio/minio123)
# Browse: audio/raw/ → should see normalized WAV files
```

### 4. Test Full Pipeline
```bash
# Drop multiple audio files (simulate accumulating 5+ hours)
for i in {1..20}; do
  cp test_audio.mp3 inbox/audio_$i.mp3
  sleep 5
done

# Wait for processing → transcription → dataset building
# When ≥5 hours accumulated, training should auto-start!

# Check training logs
docker compose -f infra/docker-compose.yml logs training-orchestrator | grep "Training"

# Check models uploaded
# Open MinIO console → models bucket → asr/ folder
```

### 5. Manual Training Trigger
```bash
# Force training start (for testing)
curl -X POST http://localhost:8100/v1/train/now

# Check status
curl http://localhost:8100/health
```

---

## 📊 Configuration

### Environment Variables
```bash
# Ingestion Watcher
INBOX_PATH=/mnt/inbox
S3_INBOX_PREFIX=audio/inbox/
INGEST_SCAN_INTERVAL_SEC=30
AUDIO_BUCKET=audio
TRANSCRIPT_BUCKET=transcripts

# Datasets & Models
DATASET_BUCKET=datasets
MODELS_BUCKET=models

# Auto-Training
TRAIN_HOURS_THRESHOLD=5.0
TRAIN_DAILY_CRON=02:00
```

### Dataset Configuration
```yaml
# configs/default.yaml
dataset:
  shards:
    target_items_per_shard: 1000
  split:
    strategy: "speaker_level"
    ratios: {train: 0.92, val: 0.04, test: 0.04}
  quality:
    min_duration_sec: 0.5
    max_duration_sec: 30.0
    snr_min_db: 5.0

training:
  trigger:
    hours_threshold: 5.0
    cron: "02:00"
  optimization:
    mixed_precision: "fp16"
    grad_accum_steps: 8
```

---

## 🎯 Key Features

### Auto-Ingestion (Phase A)
- **Multi-source**: Hot folder + S3 inbox
- **Format support**: WAV, MP3, M4A, FLAC
- **Normalization**: All → 16kHz mono PCM WAV
- **Deduplication**: Skips .war files, processes only audio
- **Event-driven**: Publishes for downstream processing

### Dataset Building (Phase B)
- **Smart text selection**: Prefers NLP-cleaned text with Vietnamese diacritics
- **Quality filters**: Duration and SNR thresholds
- **Speaker split**: Hash-based 92/4/4 train/val/test
- **Hours tracking**: Triggers when sufficient data accumulated
- **S3 storage**: Manifests in `s3://datasets/manifests/`

### Training Orchestration (Phase C)
- **Dual triggers**: Hours threshold + daily cron
- **Mixed precision**: FP16 training for speed
- **Gradient accumulation**: Effective batch size 32 on modest GPUs
- **Auto-evaluation**: WER comparison for promotion
- **Event publishing**: Full traceability

---

## 📈 Monitoring & Observability

### Event Flow Monitoring
```bash
# Watch all events
docker compose -f infra/docker-compose.yml logs nats | grep -E "(ingested|completed|postprocessed|ready|started|promoted)"

# Should see sequence:
# recording.ingested → transcription.completed → nlp.postprocessed →
# dataset.new_sample → dataset.ready → training.started → training.completed → model.promoted
```

### Training Monitoring
```bash
# Training orchestrator logs
docker compose -f infra/docker-compose.yml logs training-orchestrator

# Check manifests being built
# MinIO: datasets/manifests/train.jsonl (should grow as audio processed)

# Check models uploaded
# MinIO: models/asr/ folder (new runs appear here)
```

### Dataset Accumulation
```bash
# Check hours accumulated
docker compose -f infra/docker-compose.yml exec training-orchestrator \
  python3 -c "
import json, os
from minio import Minio
s3 = Minio('minio:9000', 'minio', 'minio123', secure=False)
tmp='/tmp/manifest.jsonl'
s3.fget_object('datasets', 'manifests/train.jsonl', tmp)
hours=0
with open(tmp,'r') as f:
    for ln in f: hours += json.loads(ln).get('duration',0)/3600
print(f'Hours accumulated: {hours:.2f}')
"
```

---

## 🔧 Troubleshooting

### Ingestion Not Working
```bash
# Check watcher started
docker compose -f infra/docker-compose.yml logs ingestion | grep "watcher"

# Check hot folder exists
docker compose -f infra/docker-compose.yml exec ingestion ls -la /mnt/inbox/

# Check file permissions
docker compose -f infra/docker-compose.yml exec ingestion ffmpeg -version
```

### Dataset Not Building
```bash
# Check event subscriptions
docker compose -f infra/docker-compose.yml logs training-orchestrator | grep "Subscribed"

# Check NLP events flowing
docker compose -f infra/docker-compose.yml logs nlp-post | grep "postprocessed"

# Check manifests being written
docker compose -f infra/docker-compose.yml exec training-orchestrator \
  python3 -c "from minio import Minio; s3=Minio('minio:9000','minio','minio123',False); print(list(s3.list_objects('datasets','manifests/')))"
```

### Training Not Triggering
```bash
# Check hours calculation
docker compose -f infra/docker-compose.yml logs training-orchestrator | grep "hours calc"

# Manual trigger
curl -X POST http://localhost:8100/v1/train/now

# Check training script exists
docker compose -f infra/docker-compose.yml exec training-orchestrator ls -la training/train.py
```

---

## 📚 File Reference

| File | Purpose | Phase |
|------|---------|-------|
| `env.example` | Environment variables | All |
| `infra/nats/streams.json` | Event subjects/consumers | All |
| `configs/default.yaml` | Configuration | All |
| `services/ingestion/app.py` | Auto-watcher + transcoding | A |
| `services/ingestion/Dockerfile` | ffmpeg inclusion | A |
| `services/ingestion/requirements.txt` | watchfiles dependency | A |
| `services/training-orchestrator/app.py` | Dataset builder + orchestrator | B+C |
| `services/training-orchestrator/Dockerfile` | Repo access for training | C |
| `services/training-orchestrator/requirements.txt` | MinIO + APScheduler | C |
| `infra/docker-compose.yml` | Service orchestration | All |

---

## 🎉 Success Criteria Met

### Phase A ✅
- ✅ Auto-ingests from hot folder and S3 inbox
- ✅ Transcodes .mp3/.wav/.m4a/.flac → 16kHz mono PCM WAV
- ✅ Publishes recording.ingested events
- ✅ Stores canonical audio in s3://audio/raw/
- ✅ Skips non-audio files (.war, etc.)

### Phase B ✅
- ✅ Builds manifests from processed audio
- ✅ Tracks speaker-level splits (92/4/4 train/val/test)
- ✅ Monitors hours accumulated
- ✅ Publishes dataset.ready when threshold reached
- ✅ Stores in s3://datasets/manifests/

### Phase C ✅
- ✅ Receives dataset.ready events
- ✅ Launches training with mixed precision + gradient accumulation
- ✅ Evaluates models automatically
- ✅ Promotes models when WER improves
- ✅ Cron scheduling for daily training
- ✅ Publishes training.* events

---

## 🚀 Production Readiness

### Scalability
- **Horizontal**: All services can scale independently
- **Event-driven**: No tight coupling, natural load balancing
- **Storage**: S3/MinIO scales infinitely

### Reliability
- **Idempotent**: Re-processing same file is safe
- **Durable**: NATS JetStream persists events
- **Recoverable**: Failed training jobs can be restarted

### Observability
- **Events**: Full audit trail via NATS
- **Logs**: Structured logging throughout
- **Metrics**: Health endpoints + training stats
- **Storage**: All artifacts in MinIO for inspection

---

## 💡 Next Steps

1. **Deploy**: Use `infra/helm/` for Kubernetes
2. **Scale**: Add more ingestion watchers as needed
3. **Monitor**: Set up Grafana dashboards for training metrics
4. **Extend**: Add real model evaluation (BLEU, CER) beyond WER
5. **Integrate**: Connect to your existing ML pipeline

---

## 🎯 You Now Have

**A complete, production-ready, automated Vietnamese ASR training pipeline!**

- ✅ **Auto-ingestion** from multiple sources
- ✅ **Smart dataset building** with quality controls
- ✅ **Intelligent training orchestration** with optimization
- ✅ **Event-driven architecture** for scalability
- ✅ **Full audit trail** in storage and events

**Drop audio files anywhere → Get better models automatically! 🚀**
