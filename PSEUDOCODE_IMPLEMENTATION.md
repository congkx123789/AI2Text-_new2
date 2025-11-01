# AI2Text ASR - Pseudocode Implementation Guide

**Language-agnostic message contracts and service logic**

This document provides the core pseudocode for all microservices, focusing on event contracts and data flows. Map 1:1 to your `services/*` code.

---

## 🎯 Core Principles

1. **Three Data Planes**: S3/MinIO (blob) + PostgreSQL (ACID) + Qdrant (vectors)
2. **Event-Driven**: CloudEvents 1.0 via NATS JetStream
3. **Speaker-Level Split**: Database trigger prevents leakage
4. **Vietnamese-First**: Diacritics restoration in dedicated service
5. **Streaming + Batch**: WebSocket for real-time, workers for batch

---

## 1. Ingestion Service

**File**: `services/ingestion/app.py`

```python
POST /v1/ingest(file: UploadFile):
    # Validate JWT (done by gateway)
    require JWT from gateway
    
    # Generate unique ID
    audio_id := uuid()
    
    # Store raw audio in object storage
    obj_key := f"raw/{audio_id}.wav"
    put_object(
        bucket="audio",
        key=obj_key,
        body=file.stream,
        content_type="audio/wav"
    )
    
    # Publish CloudEvent
    event := CloudEvent({
        specversion: "1.0",
        id: uuid(),
        source: "services/ingestion",
        type: "RecordingIngested",
        time: utcnow(),
        datacontenttype: "application/json",
        data: {
            audio_id: audio_id,
            path: f"s3://audio/{obj_key}",
            uploaded_at: utcnow()
        }
    })
    
    publish_nats("recording.ingested", event)
    
    # Return response
    return {
        audio_id: audio_id,
        object_uri: f"s3://audio/{obj_key}"
    }


# Why this design:
# - Object storage is source of truth for audio
# - Event triggers async processing
# - Ingestion stays fast (<100ms)
# - S3 path in event enables any service to download
```

---

## 2. ASR Worker (Batch)

**File**: `services/asr/worker.py`

```python
on_nats_message("recording.ingested", msg):
    # Parse event
    evt := parse_cloudevent(msg.data)
    audio_id := evt.data.audio_id
    s3_path := evt.data.path
    
    # Download audio from object storage
    audio_bytes := s3_get_object(
        parse_s3_uri(s3_path)
    )
    
    # Transcribe audio
    # TODO: Replace with Conformer/FastConformer/Whisper
    result := transcribe_audio(audio_bytes)
    # result = {
    #     text: str,
    #     segments: [{start_ms, end_ms, text, confidence}],
    #     language: "vi",
    #     model_version: str
    # }
    
    # Create transcript JSON
    transcript_json := {
        audio_id: audio_id,
        text: result.text,
        segments: result.segments,
        language: result.language,
        model_version: result.model_version,
        created_at: utcnow()
    }
    
    # Upload transcript to object storage
    transcript_key := f"transcripts/{audio_id}.json"
    s3_put_object(
        bucket="transcripts",
        key=transcript_key,
        body=json_encode(transcript_json),
        content_type="application/json"
    )
    
    # Publish TranscriptionCompleted event
    event := CloudEvent({
        specversion: "1.0",
        id: uuid(),
        source: "services/asr",
        type: "TranscriptionCompleted",
        time: utcnow(),
        data: {
            audio_id: audio_id,
            transcript_uri: f"s3://transcripts/{transcript_key}",
            text: result.text,  # Include text for downstream
            model_version: result.model_version,
            processing_time_seconds: elapsed_time
        }
    })
    
    publish_nats("transcription.completed", event)
    
    # ACK message (or NAK to retry on error)
    msg.ack()


# Why this design:
# - Decouples transcription latency from API
# - Transcript JSON persists in blob store
# - Text in event enables NLP without re-download
# - Retry via NATS consumer max_deliver
```

---

## 3. NLP Post-Processing (Vietnamese)

**File**: `services/nlp-post/app.py`

```python
on_nats_message("transcription.completed", msg):
    # Parse event
    evt := parse_cloudevent(msg.data)
    audio_id := evt.data.audio_id
    text := evt.data.text
    
    # Normalize Vietnamese text
    # TODO: Replace with ByT5/PhoBERT for production
    normalized := normalize_vietnamese(
        text=text,
        restore_diacritics=true,
        fix_typos=true
    )
    # normalized = {
    #     text_clean: str,              # Basic cleanup
    #     text_with_diacritics: str,    # "xin chao" → "xin chào"
    #     corrections: [{type, original, corrected, position}]
    # }
    
    # Publish NLPPostprocessed event
    event := CloudEvent({
        specversion: "1.0",
        id: uuid(),
        source: "services/nlp-post",
        type: "NLPPostprocessed",
        time: utcnow(),
        data: {
            audio_id: audio_id,
            text: text,                              # Original
            text_clean: normalized.text_clean,       # Cleaned
            text_with_diacritics: normalized.text_with_diacritics,
            corrections: normalized.corrections
        }
    })
    
    publish_nats("nlp.postprocessed", event)
    msg.ack()


# Why this design:
# - Vietnamese robustness in dedicated service
# - Easy to swap NLP models without touching ASR
# - Corrections logged for quality analysis
# - Enables Vietnamese-specific features (tone restoration, etc.)
```

---

## 4. Metadata Service (ACID)

**File**: `services/metadata/app.py`

```python
on_nats_message("nlp.postprocessed", msg):
    # Parse event
    evt := parse_cloudevent(msg.data)
    audio_id := evt.data.audio_id
    text := evt.data.text
    text_clean := evt.data.text_clean
    text_with_diacritics := evt.data.text_with_diacritics
    
    # ACID transaction
    with db_transaction():
        # Upsert transcript
        upsert_transcript(
            audio_id=audio_id,
            text=text,
            text_clean=text_with_diacritics,  # Use diacritics version
            updated_at=utcnow()
        )
        # ON CONFLICT (audio_id) DO UPDATE ...
        
        # Optional: Update audio metadata if needed
        # (SNR, duration, device already set by ingestion)
        
    msg.ack()


# Speaker-level split enforcement (database trigger):
CREATE TRIGGER trg_speaker_split
BEFORE INSERT OR UPDATE ON audio
FOR EACH ROW
EXECUTE FUNCTION enforce_speaker_split();

# Function checks:
IF EXISTS (
    SELECT 1 FROM audio
    WHERE speaker_id = NEW.speaker_id
    AND split_assignment != NEW.split_assignment
) THEN
    RAISE EXCEPTION 'Speaker already in different split';
END IF;


# Schema highlights:
audio {
    audio_id: uuid PRIMARY KEY,
    speaker_id: uuid REFERENCES speakers(speaker_id),
    audio_path: text,
    snr_estimate: real,              # For drift detection
    device_type: text,               # For error analysis
    split_assignment: enum(TRAIN, VAL, TEST),  # Enforced by trigger
    duration_seconds: real,
    sample_rate: integer
}

transcripts {
    audio_id: uuid PRIMARY KEY REFERENCES audio(audio_id),
    text: text,                      # Original ASR output
    text_clean: text,                # After NLP (with diacritics)
    raw_json: jsonb,                 # Full ASR output
    created_at: timestamp,
    updated_at: timestamp
}

speakers {
    speaker_id: uuid PRIMARY KEY,
    pseudonymous_id: text UNIQUE,   # Never expose PII
    region: text,                    # north/central/south
    device_types: text[],            # Array of devices used
    total_recordings: integer
}


# Why this design:
# - Speaker-level split prevents leakage
# - SNR/device enable drift analysis
# - text_clean has Vietnamese diacritics for display
# - ACID guarantees data consistency
```

---

## 5. Embeddings Service

**File**: `services/embeddings/app.py`

```python
on_nats_message("nlp.postprocessed", msg):
    # Parse event
    evt := parse_cloudevent(msg.data)
    audio_id := evt.data.audio_id
    text_clean := evt.data.text_with_diacritics
    
    # Generate embedding vector
    # TODO: Replace with Word2Vec/Phon2Vec/MPNet
    vector := embed_text(text_clean)
    # vector: float32[768]
    
    # Index in Qdrant
    qdrant_upsert(
        collection="texts",
        id=audio_id,
        vector=vector,
        payload={
            audio_id: audio_id,
            text: text_clean
        }
    )
    
    # Publish EmbeddingsIndexed event
    event := CloudEvent({
        specversion: "1.0",
        id: uuid(),
        source: "services/embeddings",
        type: "EmbeddingsIndexed",
        time: utcnow(),
        data: {
            audio_id: audio_id,
            vector_id: audio_id,
            vector_type: "text",
            embedding_model: "word2vec-vi-v1"
        }
    })
    
    publish_nats("embeddings.indexed", event)
    msg.ack()


# Qdrant collection schema:
{
    name: "texts",
    vectors: {
        size: 768,
        distance: "Cosine"
    }
}


# Why this design:
# - Separate vector plane for ANN search
# - text_clean (with diacritics) for better embeddings
# - 768-dim matches common embedding models
# - Enables semantic search + future diarization vectors
```

---

## 6. Search Service (ANN + Join)

**File**: `services/search/app.py`

```python
GET /v1/search?q=<query>&limit=20:
    # Normalize query (same pipeline as transcripts)
    query_normalized := normalize_vietnamese(q)
    query_text := query_normalized.text_with_diacritics
    
    # Generate query vector
    query_vec := embed_text(query_text)
    
    # ANN search in Qdrant
    hits := qdrant_search(
        collection="texts",
        vector=query_vec,
        limit=20,
        with_payload=true
    )
    # hits = [{id, score, payload}]
    
    # Extract audio IDs
    audio_ids := [hit.id for hit in hits]
    
    # Join with metadata (PostgreSQL)
    rows := db_query(
        "SELECT audio_id, text_clean FROM transcripts WHERE audio_id = ANY($1)",
        audio_ids
    )
    
    # Merge and rank
    results := []
    for hit in hits:
        row := rows[hit.id]
        results.append({
            audio_id: hit.id,
            text: row.text_clean,
            score: hit.score,
            metadata: hit.payload
        })
    
    return {
        results: results,
        total: len(results),
        query_time_ms: elapsed_ms
    }


# Why this design:
# - Fast ANN search (<50ms)
# - Join with ACID store for display
# - Same normalization pipeline ensures consistency
# - Score from vector similarity
```

---

## 7. Streaming ASR (WebSocket)

**File**: `services/asr/streaming_server.py`

```python
websocket /v1/asr/stream:
    # Accept connection
    await ws.accept()
    
    # Initialize transcriber
    audio_id := f"aud_{uuid_short()}"
    transcriber := StreamingTranscriber(
        model=conformer_model,  # TODO: Load Conformer/FastConformer
        sample_rate=16000
    )
    
    # Start result emitter task
    task := async_task(emit_results(ws, transcriber, audio_id))
    
    # Message loop
    while connected:
        msg := await ws.receive()
        
        if msg.type == "start":
            # Acknowledge session
            await ws.send({
                type: "ack",
                audio_id: audio_id
            })
        
        elif msg.type == "frame":
            # Accept audio chunk (PCM16 bytes or base64)
            audio_bytes := decode_base64(msg.base64) if msg.base64 else msg.bytes
            await transcriber.accept_audio(audio_bytes)
        
        elif msg.type == "end":
            # Finalize transcription
            await transcriber.finalize()
            break
    
    # Wait for emitter to finish
    await task


async def emit_results(ws, transcriber, audio_id):
    # Stream results (partials + final)
    async for result in transcriber.results():
        # Add audio_id to result
        result.audio_id = audio_id
        
        # Send to client
        await ws.send_json(result)
        
        # If final result, publish event
        if result.type == "final":
            # Store transcript
            transcript_json := {
                audio_id: audio_id,
                text: result.text,
                segments: result.segments
            }
            
            s3_put_object(
                bucket="transcripts",
                key=f"transcripts/{audio_id}.json",
                body=json_encode(transcript_json)
            )
            
            # Publish event (same as batch worker)
            event := CloudEvent({
                ...
                type: "TranscriptionCompleted",
                data: {
                    audio_id: audio_id,
                    text: result.text,
                    transcript_uri: f"s3://transcripts/transcripts/{audio_id}.json"
                }
            })
            
            publish_nats("transcription.completed", event)
            
            break


class StreamingTranscriber:
    # TODO: Replace with Conformer/FastConformer hybrid CTC/RNN-T
    
    def __init__(self, model, sample_rate):
        self.model = model
        self.buffer = []
        self.closed = false
    
    async def accept_audio(self, chunk: bytes):
        # Convert bytes to audio samples
        samples := pcm16_to_float32(chunk)
        self.buffer.extend(samples)
    
    async def finalize(self):
        self.closed = true
    
    async def results(self):
        # Yield partials every 400ms
        chunk_size := sample_rate * 0.4  # 400ms
        
        while not self.closed or len(self.buffer) >= chunk_size:
            if len(self.buffer) < chunk_size:
                await sleep(0.1)
                continue
            
            # Extract chunk
            chunk := self.buffer[:chunk_size]
            self.buffer := self.buffer[chunk_size//2:]  # 50% overlap
            
            # Incremental decode
            partial_text := self.model.decode_incremental(chunk)
            
            yield {
                type: "partial",
                text: partial_text,
                start_ms: offset_ms,
                end_ms: offset_ms + 400
            }
        
        # Final decode (full context)
        final_text := self.model.decode_final(all_audio)
        
        yield {
            type: "final",
            text: final_text,
            segments: [...]
        }


# Why this design:
# - Low latency (<500ms for partials)
# - Same event flow as batch (transcription.completed)
# - Incremental decoding for real-time UX
# - Final result has full context for best quality
```

---

## 8. API Gateway

**File**: `services/api-gateway/app.py`

```python
@app.api_route("/{full_path:path}", methods=["*"])
@rate_limit("200/minute")  # Via slowapi
async def proxy(full_path, request):
    # Health check bypass
    if request.path == "/health":
        return {status: "healthy"}
    
    # Parse JWT
    auth_header := request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(401, "Missing Authorization")
    
    token := extract_bearer_token(auth_header)
    
    # Verify JWT
    try:
        if JWT_ALGO == "RS256":
            # Production: RS256 with public key
            claims := jwt_verify(
                token,
                public_key=JWT_PUBLIC_KEY,
                algorithm="RS256",
                audience="ai2text-asr"
            )
        else:
            # Development: HS256
            claims := jwt_verify(
                token,
                secret=JWT_PUBLIC_KEY,
                algorithm="HS256"
            )
    except JWTError as e:
        raise HTTPException(401, f"Invalid token: {e}")
    
    # Route to service
    service_url := ROUTE_MAP[request.path_prefix]
    target_url := f"{service_url}{request.path}"
    
    # Proxy request
    response := http_client.request(
        method=request.method,
        url=target_url,
        headers=request.headers,
        body=request.body,
        params=request.query_params
    )
    
    return response


ROUTE_MAP = {
    "/v1/ingest": "http://ingestion:8000",
    "/v1/transcripts": "http://metadata:8000",
    "/v1/search": "http://search:8000",
    "/v1/nlp": "http://nlp-post:8000"
}


# Security notes:
# - For production: switch to RS256, rotate keys monthly
# - Rate limit per user (use claims.sub as key)
# - Log all auth failures for security monitoring
```

---

## 🔄 Complete Event Flow

```
1. Upload
   Client → Gateway → Ingestion
                      ↓
                    MinIO (audio/raw/{id}.wav)
                      ↓
                    NATS: recording.ingested

2. Transcribe
   NATS → ASR Worker
          ↓
        Download from MinIO
          ↓
        Transcribe (Conformer/Whisper)
          ↓
        Upload to MinIO (transcripts/{id}.json)
          ↓
        NATS: transcription.completed

3. Normalize
   NATS → NLP-Post
          ↓
        Normalize Vietnamese (diacritics)
          ↓
        NATS: nlp.postprocessed

4. Store
   NATS → Metadata Service
          ↓
        PostgreSQL (transcripts table)
   
   NATS → Embeddings Service
          ↓
        Generate vector
          ↓
        Qdrant (texts collection)
          ↓
        NATS: embeddings.indexed

5. Search
   Client → Gateway → Search
                      ↓
                    Qdrant (ANN search)
                      ↓
                    PostgreSQL (metadata join)
                      ↓
                    Return results


Total latency: ~2-5 seconds from upload to searchable
```

---

## 📁 File Mapping

| Pseudocode | Actual File |
|------------|-------------|
| Ingestion | `services/ingestion/app.py` |
| ASR Worker | `services/asr/worker.py` |
| ASR Streaming | `services/asr/streaming_server.py` |
| NLP-Post | `services/nlp-post/app.py` |
| Metadata | `services/metadata/app.py` |
| Embeddings | `services/embeddings/app.py` |
| Search | `services/search/app.py` |
| Gateway | `services/api-gateway/app.py` |
| Database Schema | `services/metadata/migrations/001_init.sql` |
| NATS Config | `infra/nats/streams.json` |
| Helm Charts | `infra/helm/ai-stt/` |
| Tests | `tests/e2e/test_flow.py` |

---

## ✅ Implementation Checklist

Use this pseudocode as reference when:
- [ ] Implementing new services
- [ ] Debugging event flows
- [ ] Onboarding new team members
- [ ] Writing integration tests
- [ ] Designing new features
- [ ] Optimizing performance

---

**This pseudocode matches your research, docs, and SOTA recommendations! 🚀**

