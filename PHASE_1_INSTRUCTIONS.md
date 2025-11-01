# Phase 1 - Wire the Async Pipeline (Week 1)

**Goal:** Complete end-to-end async pipeline: `Ingestion → NATS → ASR Worker → MinIO → Metadata`

**Timeline:** Week 1 (Days 2-7)

---

## 🎯 Phase 1 Overview

Wire the full async event-driven pipeline so that:

1. Upload triggers `recording.ingested` event
2. ASR worker processes audio and creates transcript JSON
3. Transcript is stored in MinIO
4. `transcription.completed` event triggers metadata storage
5. PostgreSQL has complete audit trail with speaker-level split enforcement

---

## ✅ Phase 1 Exit Criteria

- [ ] Upload audio file → see transcript JSON in MinIO under `transcripts/`
- [ ] `GET /v1/transcripts/{id}` returns text from database
- [ ] PostgreSQL has rows in `audio` table (with SNR/device/split populated)
- [ ] PostgreSQL has rows in `transcripts` table (with text and text_clean)
- [ ] Event flow verified: `recording.ingested` → `transcription.completed` → `nlp.postprocessed`
- [ ] MinIO console shows both raw audio and transcript JSON files
- [ ] All services handle errors gracefully (retry logic works)

---

## 📋 Tasks Breakdown

### Task 1.1: Verify Ingestion Service ✅ (Already Done)

**Status:** ✅ Complete

**Current Implementation:**
```python
# services/ingestion/app.py already publishes:
event = {
    "specversion": "1.0",
    "id": str(uuid.uuid4()),
    "source": "services/ingestion",
    "type": "RecordingIngested",
    "data": {
        "audio_id": audio_id,
        "path": f"s3://{BUCKET}/{obj_name}"  # ✓ Includes object URI
    }
}
await app.nc.publish("recording.ingested", json.dumps(event).encode())
```

**Verification:**
```bash
# Upload a file and check logs
TOKEN=$(python3 scripts/jwt_dev_token.py)
curl -X POST http://localhost:8080/v1/ingest \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@test_audio.wav"

# Check ingestion logs
docker compose -f infra/docker-compose.yml logs ingestion | grep "RecordingIngested"
```

**Expected:** You should see the event published with `audio_id` and `path`.

---

### Task 1.2: Complete ASR Worker ✅ (Already Done)

**Status:** ✅ Complete

**Current Implementation:**
The ASR worker (`services/asr/worker.py`) already:
- ✅ Subscribes to `recording.ingested` events
- ✅ Downloads audio from MinIO using S3 path
- ✅ Transcribes audio (currently stub: "xin chào thế giới")
- ✅ Creates transcript JSON with proper structure
- ✅ Uploads transcript to MinIO `transcripts/` bucket
- ✅ Publishes `transcription.completed` event with text included

**Verification:**
```bash
# Check ASR worker logs
docker compose -f infra/docker-compose.yml logs asr | tail -50

# You should see:
# [ASR] Processing {audio_id} from s3://audio/raw/...
# [ASR] Downloaded X bytes from audio/raw/{audio_id}.wav
# [ASR] Transcribed: xin chào thế giới
# [ASR] Uploaded transcript to s3://transcripts/transcripts/{audio_id}.json
# [OK] ASR completed for {audio_id}
```

**Check MinIO:**
1. Open http://localhost:9001 (login: minio/minio123)
2. Browse `transcripts` bucket
3. You should see `transcripts/{audio_id}.json` files

**Transcript JSON Format:**
```json
{
  "audio_id": "550e8400-e29b-41d4-a716-446655440000",
  "text": "xin chào thế giới",
  "segments": [
    {
      "start_ms": 0,
      "end_ms": 800,
      "text": "xin chào thế giới",
      "confidence": 0.95
    }
  ],
  "language": "vi",
  "model_version": "stub-1.0"
}
```

---

### Task 1.3: Wire Metadata Service ✅ (Already Done)

**Status:** ✅ Complete

**Current Implementation:**
The metadata service (`services/metadata/app.py`) already:
- ✅ Subscribes to `nlp.postprocessed` events
- ✅ Stores transcript in PostgreSQL with text and text_clean
- ✅ Uses ACID transactions
- ✅ Handles conflicts with ON CONFLICT clause

**Event Flow:**
```
transcription.completed (from ASR)
         ↓
    NLP-Post Service (processes text)
         ↓
nlp.postprocessed (includes text_clean)
         ↓
    Metadata Service (stores in PostgreSQL)
```

**Verification:**
```bash
# Check metadata logs
docker compose -f infra/docker-compose.yml logs metadata | tail -50

# You should see:
# [Metadata] Updating {audio_id} with NLP-processed text
# [OK] Metadata updated for {audio_id}

# Check database directly
make shell-postgres
# Then in psql:
SELECT audio_id, text, text_clean FROM transcripts ORDER BY created_at DESC LIMIT 5;
```

---

### Task 1.4: Add Audio Metadata Tracking (NEW - To Implement)

**Status:** 🔴 TODO

**Goal:** Store audio metadata in the `audio` table with SNR, device, and split assignment.

**Current Gap:** The ingestion service uploads to MinIO but doesn't create an `audio` row in PostgreSQL.

**Implementation Plan:**

#### Step 1: Update Ingestion Service to Call Metadata

Edit `services/ingestion/app.py`:

```python
# After uploading to MinIO, before publishing event
# Add this section:

# Create audio metadata record
import httpx
METADATA_URL = os.getenv("METADATA_URL", "http://metadata:8000")

try:
    # Calculate basic audio properties
    # TODO: Add real SNR calculation, device detection
    metadata = {
        "audio_id": audio_id,
        "speaker_id": None,  # TODO: Extract from filename or metadata
        "audio_path": f"s3://{BUCKET}/{obj_name}",
        "snr_estimate": None,  # TODO: Calculate SNR
        "device_type": "unknown",  # TODO: Detect device
        "environment": "unknown",  # TODO: Detect environment
        "split_assignment": "TRAIN",  # Default to TRAIN for now
        "duration_seconds": None,  # TODO: Calculate from audio
        "sample_rate": 16000  # Assume 16kHz for now
    }
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            f"{METADATA_URL}/v1/audio",
            json=metadata
        )
        if response.status_code == 200:
            print(f"[OK] Audio metadata created for {audio_id}")
        else:
            print(f"[WARNING] Failed to create audio metadata: {response.status_code}")
except Exception as e:
    print(f"[WARNING] Could not create audio metadata: {e}")
    # Don't fail the upload if metadata fails
```

#### Step 2: Test Audio Metadata Creation

```bash
# Restart ingestion service
docker compose -f infra/docker-compose.yml restart ingestion

# Upload a file
TOKEN=$(python3 scripts/jwt_dev_token.py)
curl -X POST http://localhost:8080/v1/ingest \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@test_audio.wav"

# Check database
make shell-postgres
# In psql:
SELECT audio_id, audio_path, snr_estimate, device_type, split_assignment 
FROM audio 
ORDER BY created_at DESC 
LIMIT 5;
```

**Expected Result:**
```
audio_id                             | audio_path                | snr_estimate | device_type | split_assignment
-------------------------------------|---------------------------|--------------|-------------|------------------
550e8400-e29b-41d4-a716-446655440000 | s3://audio/raw/...        | null         | unknown     | TRAIN
```

---

### Task 1.5: Add SNR and Audio Analysis (OPTIONAL for Week 1)

**Status:** 🟡 Optional Enhancement

**Goal:** Calculate SNR, duration, and other audio properties during ingestion.

**Implementation:**

Create `services/ingestion/audio_analyzer.py`:

```python
"""
Audio analysis utilities for SNR calculation and metadata extraction.
"""

import numpy as np
import wave
import io
from typing import Dict, Any


def analyze_audio(audio_bytes: bytes) -> Dict[str, Any]:
    """
    Analyze audio file and extract metadata.
    
    Returns:
        dict with: duration_seconds, sample_rate, channels, snr_estimate
    """
    try:
        # Parse WAV file
        with wave.open(io.BytesIO(audio_bytes), 'rb') as wav:
            sample_rate = wav.getframerate()
            channels = wav.getnchannels()
            n_frames = wav.getnframes()
            duration_seconds = n_frames / sample_rate
            
            # Read audio data
            audio_data = wav.readframes(n_frames)
            
            # Convert to numpy array
            if wav.getsampwidth() == 2:  # 16-bit
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
            else:
                audio_array = np.frombuffer(audio_data, dtype=np.int8)
            
            # Calculate SNR estimate (simple method)
            # SNR = 10 * log10(signal_power / noise_power)
            signal_power = np.mean(audio_array.astype(float) ** 2)
            
            # Estimate noise from quietest 10% of frames
            frame_size = int(sample_rate * 0.1)  # 100ms frames
            frames = [audio_array[i:i+frame_size] for i in range(0, len(audio_array), frame_size)]
            frame_powers = [np.mean(f.astype(float) ** 2) for f in frames if len(f) == frame_size]
            
            if frame_powers:
                noise_power = np.percentile(frame_powers, 10)
                snr_estimate = 10 * np.log10(signal_power / max(noise_power, 1e-10))
            else:
                snr_estimate = None
            
            return {
                "duration_seconds": float(duration_seconds),
                "sample_rate": sample_rate,
                "channels": channels,
                "snr_estimate": float(snr_estimate) if snr_estimate else None
            }
    except Exception as e:
        print(f"[WARNING] Audio analysis failed: {e}")
        return {
            "duration_seconds": None,
            "sample_rate": None,
            "channels": None,
            "snr_estimate": None
        }
```

Then use it in `services/ingestion/app.py`:

```python
from audio_analyzer import analyze_audio

# After reading file
audio_bytes = await file.read()
audio_analysis = analyze_audio(audio_bytes)

# Reset file pointer for upload
await file.seek(0)

# Use analysis results in metadata
metadata = {
    "audio_id": audio_id,
    "duration_seconds": audio_analysis["duration_seconds"],
    "sample_rate": audio_analysis["sample_rate"],
    "snr_estimate": audio_analysis["snr_estimate"],
    # ... rest of metadata
}
```

**Note:** This is optional for Week 1. You can implement this later and update existing records.

---

## 🧪 Phase 1 Verification Checklist

Run this checklist to verify Phase 1 is complete:

### 1. End-to-End Upload Test

```bash
# Generate JWT token
TOKEN=$(python3 scripts/jwt_dev_token.py)

# Create test audio file (if you don't have one)
python3 - <<'EOF'
import wave, struct
with wave.open('test_audio.wav', 'w') as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(16000)
    silence = struct.pack('<h', 0) * 16000
    wav.writeframes(silence)
print("✓ Created test_audio.wav")
EOF

# Upload audio
RESPONSE=$(curl -s -X POST http://localhost:8080/v1/ingest \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@test_audio.wav")

echo "$RESPONSE" | jq .

# Extract audio_id
AUDIO_ID=$(echo "$RESPONSE" | jq -r '.audio_id')
echo "Audio ID: $AUDIO_ID"
```

### 2. Verify MinIO Storage

```bash
# Wait for processing
sleep 5

# Check MinIO via console
# Open: http://localhost:9001
# Browse:
#   - 'audio' bucket -> raw/ -> should see {audio_id}.wav
#   - 'transcripts' bucket -> transcripts/ -> should see {audio_id}.json
```

### 3. Verify Event Flow

```bash
# Check logs for event flow
docker compose -f infra/docker-compose.yml logs --tail=100 | grep -E "(RecordingIngested|TranscriptionCompleted|NLPPostprocessed)"

# Should see sequence:
# ingestion  | RecordingIngested ...
# asr        | [ASR] Processing ...
# asr        | TranscriptionCompleted ...
# nlp-post   | [NLP] Processing ...
# nlp-post   | NLPPostprocessed ...
# metadata   | [Metadata] Updating ...
```

### 4. Verify Database Storage

```bash
# Check PostgreSQL
make shell-postgres

# In psql, run:
\dt  # List tables (should see: audio, transcripts, speakers)

SELECT COUNT(*) FROM audio;
SELECT COUNT(*) FROM transcripts;

# Get latest records
SELECT audio_id, audio_path, split_assignment, snr_estimate, device_type 
FROM audio 
ORDER BY created_at DESC 
LIMIT 1;

SELECT audio_id, text, text_clean 
FROM transcripts 
ORDER BY created_at DESC 
LIMIT 1;

# Exit
\q
```

### 5. Verify API Retrieval

```bash
# Get transcript via API (use audio_id from step 1)
curl -s http://localhost:8080/v1/transcripts/$AUDIO_ID \
  -H "Authorization: Bearer $TOKEN" | jq .

# Expected response:
# {
#   "audio_id": "...",
#   "text": "xin chào thế giới",
#   "text_clean": "xin chào thế giới",
#   "raw_json": null
# }
```

### 6. Test Speaker-Level Split Enforcement

```bash
# Create a test speaker
make shell-postgres

# In psql:
INSERT INTO speakers (speaker_id, pseudonymous_id, region) 
VALUES ('550e8400-e29b-41d4-a716-446655440001', 'speaker_001', 'north');

# Try to add audio with same speaker but different split (should fail)
INSERT INTO audio (audio_id, speaker_id, audio_path, split_assignment) 
VALUES ('audio_1', '550e8400-e29b-41d4-a716-446655440001', 's3://test', 'TRAIN');

INSERT INTO audio (audio_id, speaker_id, audio_path, split_assignment) 
VALUES ('audio_2', '550e8400-e29b-41d4-a716-446655440001', 's3://test2', 'VAL');
-- This should FAIL with: "Speaker ... already assigned to a different split"

\q
```

**Expected:** Second insert fails with trigger error (speaker split enforcement working).

---

## 📊 Phase 1 Success Metrics

At the end of Week 1, you should have:

- [ ] ✅ **100% async pipeline completion**: Upload → MinIO → Event → ASR → Transcript JSON → Event → Metadata → PostgreSQL
- [ ] ✅ **MinIO audit trail**: Both raw audio and transcript JSON visible in console
- [ ] ✅ **PostgreSQL audit trail**: `audio` and `transcripts` tables populated
- [ ] ✅ **API works**: `/v1/transcripts/{id}` returns data
- [ ] ✅ **Events flow**: All 3 events (ingested, completed, postprocessed) in logs
- [ ] ✅ **Error handling**: Services recover from failures gracefully
- [ ] ✅ **Speaker split works**: Trigger prevents cross-split speaker contamination

---

## 🐛 Troubleshooting Phase 1

### Issue: ASR worker not processing events

```bash
# Check ASR worker is subscribed
docker compose -f infra/docker-compose.yml logs asr | grep "listening"

# Should see: "[OK] ASR worker listening for recording.ingested events..."

# If not, restart ASR service
docker compose -f infra/docker-compose.yml restart asr

# Check NATS connectivity
docker compose -f infra/docker-compose.yml logs nats
```

### Issue: Transcript not appearing in database

```bash
# Check metadata service logs
docker compose -f infra/docker-compose.yml logs metadata | tail -50

# Check if nlp.postprocessed event is published
docker compose -f infra/docker-compose.yml logs nlp-post | grep "NLPPostprocessed"

# Manually trigger metadata update (for debugging)
TOKEN=$(python3 scripts/jwt_dev_token.py)
curl -X PUT "http://localhost:8002/v1/transcripts/$AUDIO_ID?text=test+text"
```

### Issue: MinIO not showing transcript JSON

```bash
# Check ASR worker logs
docker compose -f infra/docker-compose.yml logs asr | grep "Uploaded transcript"

# Check MinIO buckets exist
docker exec minio ls /data/

# Should show: audio/ transcripts/

# If missing, restart bootstrap
bash scripts/bootstrap.sh
```

### Issue: PostgreSQL trigger not enforcing split

```bash
# Check trigger exists
make shell-postgres

# In psql:
\dt  # List tables
\df  # List functions

# You should see:
# - enforce_speaker_split()
# - trg_speaker_split trigger

# Re-run migration if missing
\q
make migrate-fresh
```

---

## 📝 Next Steps After Phase 1

Once Phase 1 is complete (all exit criteria met), you can proceed to:

**Phase 2 - Real Model Integration (Week 2)**
- Replace ASR stub with your LSTM model or Whisper
- Add real Vietnamese NLP model (ByT5, underthesea)
- Benchmark performance

**Phase 3 - Production Hardening (Week 3)**
- Add proper error recovery and retries
- Implement circuit breakers
- Add comprehensive logging and metrics
- Load testing

**Phase 4 - Deployment (Week 4)**
- Deploy to staging Kubernetes cluster
- Configure monitoring and alerting
- Security hardening (RS256 JWT, etc.)
- Production launch

---

## 🎯 Current Status Summary

| Component | Status | Notes |
|-----------|--------|-------|
| Ingestion → NATS | ✅ Complete | Publishing recording.ingested |
| ASR Worker | ✅ Complete | Downloads audio, creates transcript JSON |
| Transcript Upload | ✅ Complete | Stored in MinIO transcripts/ bucket |
| Event Publishing | ✅ Complete | transcription.completed with text |
| NLP Processing | ✅ Complete | Vietnamese diacritics restoration |
| Metadata Storage | ✅ Complete | PostgreSQL with text_clean |
| Audio Metadata | 🔴 TODO | Need to add audio table population |
| SNR Calculation | 🟡 Optional | Can add later |

**Phase 1 Completion: 90% (1 task remaining: audio metadata)**

---

## 📞 Questions or Issues?

1. Check logs: `make logs`
2. Check health: `make health`
3. Verify Phase 0 first: `python3 scripts/verify_setup.py`
4. Review documentation: `RUN_GUIDE.md`, `MICROSERVICES_SETUP_GUIDE.md`

**You're almost there! Just need to add audio metadata tracking and you'll have Phase 1 complete! 🚀**

