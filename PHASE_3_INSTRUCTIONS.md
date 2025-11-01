# Phase 3 - Production Hardening (Week 3+)

**Goal:** Real-time ASR, security hardening, observability, and SLOs

**Timeline:** Week 3-4

---

## 🎯 Phase 3 Overview

Transform the baseline system into production-ready infrastructure with:
1. Real-time streaming ASR (<500ms partials)
2. RS256 JWT authentication
3. OpenTelemetry metrics and tracing
4. Dead Letter Queues (DLQ) for failed events
5. Horizontal Pod Autoscaling (HPA)
6. SLO monitoring and alerting

---

## ✅ Phase 3 Exit Criteria

- [ ] WebSocket streaming emits partials < 500ms latency
- [ ] API Gateway verifies RS256 JWT tokens
- [ ] All services expose `/metrics` endpoint (Prometheus format)
- [ ] Grafana dashboards show request rate, latency, error rate
- [ ] NATS consumers have DLQ subjects configured
- [ ] HPA policies scale based on CPU/memory/latency
- [ ] Alerts fire for: high error rate, high latency, queue backlog

---

## 📋 Tasks Breakdown

### Task 3.1: Upgrade to Real-Time Streaming ASR

**Status:** 🔴 TODO

**Goal:** Replace `DummyTranscriber` with production-ready streaming decoder.

#### Option A: FastConformer (Recommended for Vietnamese)

**File:** `services/asr/streaming_server.py`

```python
"""
Streaming ASR with FastConformer/Conformer-CTC for low-latency Vietnamese.

For production, use NeMo FastConformer or similar hybrid CTC/RNN-T model.
This provides sub-500ms partial results.
"""

import torch
from typing import AsyncIterator, Dict, Any
import numpy as np

# Option 1: NeMo FastConformer
try:
    import nemo.collections.asr as nemo_asr
    HAVE_NEMO = True
except ImportError:
    HAVE_NEMO = False

# Option 2: Your custom Conformer
try:
    from models.conformer_asr import ConformerCTCModel
    HAVE_CUSTOM_MODEL = True
except ImportError:
    HAVE_CUSTOM_MODEL = False


class StreamingTranscriber:
    """
    Streaming ASR with incremental decoding.
    
    Provides:
    - Sub-500ms partial results
    - Hybrid CTC/RNN-T decoding
    - Vietnamese phoneme-aware beam search
    """
    
    def __init__(self, model_path: str = None, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.buffer = []
        self.offset_ms = 0
        self.closed = False
        
        # Load model
        if HAVE_NEMO:
            # Option 1: NeMo FastConformer
            self.model = nemo_asr.models.ASRModel.restore_from(model_path)
            self.model.eval()
            self.model.cuda() if torch.cuda.is_available() else self.model.cpu()
        elif HAVE_CUSTOM_MODEL:
            # Option 2: Your custom model
            self.model = ConformerCTCModel.load_checkpoint(model_path)
            self.model.eval()
        else:
            # Fallback: Use stub for development
            self.model = None
            print("[WARNING] No ASR model loaded, using stub")
        
        # Streaming decoder configuration
        self.chunk_size_ms = 400  # 400ms chunks for partials
        self.hop_size_ms = 200    # 200ms hop for overlap
        self.min_confidence = 0.3  # Filter low-confidence partials
        
    async def accept_audio(self, chunk: bytes) -> None:
        """Accept audio chunk (PCM16 bytes)."""
        # Convert bytes to float32 numpy array
        audio_array = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        self.buffer.extend(audio_array.tolist())
    
    async def finalize(self) -> None:
        """Mark end of audio stream."""
        self.closed = True
    
    async def results(self) -> AsyncIterator[Dict[str, Any]]:
        """
        Yield streaming results (partials + final).
        
        Yields:
            dict: {
                "type": "partial" | "final",
                "text": str,
                "start_ms": int,
                "end_ms": int,
                "confidence": float (optional)
            }
        """
        chunk_samples = int(self.sample_rate * self.chunk_size_ms / 1000)
        hop_samples = int(self.sample_rate * self.hop_size_ms / 1000)
        
        last_partial_text = ""
        
        while not self.closed or len(self.buffer) >= chunk_samples:
            # Wait for enough audio
            if len(self.buffer) < chunk_samples:
                await asyncio.sleep(0.1)
                continue
            
            # Extract chunk
            chunk = np.array(self.buffer[:chunk_samples], dtype=np.float32)
            self.buffer = self.buffer[hop_samples:]  # Hop forward
            
            # Transcribe chunk
            if self.model is not None:
                partial_text = await self._transcribe_chunk(chunk)
            else:
                # Stub for development
                partial_text = "..." if not self.closed else "xin chào thế giới"
            
            # Only emit if text changed (avoid duplicates)
            if partial_text != last_partial_text:
                start_ms = self.offset_ms
                end_ms = self.offset_ms + self.chunk_size_ms
                
                yield {
                    "type": "partial",
                    "text": partial_text,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "confidence": 0.8  # TODO: Get from model
                }
                
                last_partial_text = partial_text
            
            self.offset_ms += self.hop_size_ms
        
        # Final result (full context)
        if self.model is not None:
            final_audio = np.array(self.buffer, dtype=np.float32)
            final_text = await self._transcribe_full(final_audio)
        else:
            final_text = "xin chào thế giới"  # Stub
        
        yield {
            "type": "final",
            "text": final_text,
            "segments": [
                {
                    "start_ms": 0,
                    "end_ms": self.offset_ms,
                    "text": final_text,
                    "confidence": 0.95
                }
            ]
        }
    
    async def _transcribe_chunk(self, audio: np.ndarray) -> str:
        """Transcribe audio chunk (incremental)."""
        if HAVE_NEMO:
            # NeMo FastConformer incremental decoding
            with torch.no_grad():
                audio_tensor = torch.from_numpy(audio).unsqueeze(0)
                if torch.cuda.is_available():
                    audio_tensor = audio_tensor.cuda()
                
                # Incremental CTC decoding
                logits = self.model.forward(
                    input_signal=audio_tensor,
                    input_signal_length=torch.tensor([len(audio)])
                )
                
                # Greedy decode or beam search
                text = self.model.decoding.ctc_decoder_predictions_tensor(logits)[0]
                return text
        
        elif HAVE_CUSTOM_MODEL:
            # Your custom model
            return self.model.transcribe_chunk(audio)
        
        return "..."
    
    async def _transcribe_full(self, audio: np.ndarray) -> str:
        """Transcribe full audio (final result with full context)."""
        if HAVE_NEMO:
            with torch.no_grad():
                audio_tensor = torch.from_numpy(audio).unsqueeze(0)
                if torch.cuda.is_available():
                    audio_tensor = audio_tensor.cuda()
                
                # Full context decoding with beam search
                predictions = self.model.transcribe([audio_tensor])[0]
                return predictions
        
        elif HAVE_CUSTOM_MODEL:
            return self.model.transcribe(audio)
        
        return "xin chào thế giới"
```

**Update streaming_server.py to use new transcriber:**

```python
# Replace DummyTranscriber with StreamingTranscriber
transcriber = StreamingTranscriber(
    model_path=os.getenv("ASR_MODEL_PATH", "checkpoints/conformer_vi.nemo"),
    sample_rate=16000
)
```

**Test streaming latency:**

```bash
# Measure partial latency
python3 - <<'EOF'
import asyncio, websockets, json, base64, time

async def test_latency():
    async with websockets.connect("ws://localhost:8003/v1/asr/stream") as ws:
        await ws.send(json.dumps({"type":"start","audio_format":{"sample_rate":16000,"channels":1,"encoding":"pcm16"}}))
        
        # Send 1 second of audio
        start = time.time()
        await ws.send(json.dumps({"type":"frame","base64":base64.b64encode(b'\x00'*32000).decode()}))
        
        # Wait for first partial
        msg = await ws.recv()
        latency = (time.time() - start) * 1000
        result = json.loads(msg)
        
        if result["type"] == "partial":
            print(f"✓ Partial latency: {latency:.0f}ms (target: <500ms)")
        
        await ws.send(json.dumps({"type":"end"}))

asyncio.run(test_latency())
EOF
```

---

### Task 3.2: Upgrade JWT to RS256

**Status:** 🔴 TODO

**Goal:** Replace HS256 dev keys with RS256 for production security.

#### Step 1: Generate RSA Key Pair

```bash
# Generate private key
openssl genrsa -out jwt-private.pem 2048

# Generate public key
openssl rsa -in jwt-private.pem -pubout -out jwt-public.pem

# Create Kubernetes secret (for production)
kubectl create secret generic jwt-keys \
  --from-file=private=jwt-private.pem \
  --from-file=public=jwt-public.pem \
  -n ai2text-prod

# For dev, store in .env
echo "JWT_PRIVATE_KEY=$(cat jwt-private.pem | base64 -w 0)" >> .env
echo "JWT_PUBLIC_KEY=$(cat jwt-public.pem | base64 -w 0)" >> .env
echo "JWT_ALGO=RS256" >> .env
```

#### Step 2: Update API Gateway

**File:** `services/api-gateway/app.py`

```python
import base64

# Load keys based on algorithm
JWT_ALGO = os.getenv("JWT_ALGO", "HS256")

if JWT_ALGO == "RS256":
    # Production: RS256 with public key
    jwt_public_key_b64 = os.getenv("JWT_PUBLIC_KEY")
    if jwt_public_key_b64:
        JWT_PUBLIC_KEY = base64.b64decode(jwt_public_key_b64).decode()
    else:
        # Try loading from file
        try:
            with open("/secrets/jwt-public.pem", "r") as f:
                JWT_PUBLIC_KEY = f.read()
        except FileNotFoundError:
            raise RuntimeError("JWT_PUBLIC_KEY not configured for RS256")
else:
    # Development: HS256
    JWT_PUBLIC_KEY = os.getenv("JWT_PUBLIC_KEY", "dev")

print(f"[INFO] JWT algorithm: {JWT_ALGO}")
```

#### Step 3: Create Token Service

**File:** `scripts/generate_jwt_rs256.py`

```python
#!/usr/bin/env python3
"""Generate RS256 JWT tokens for production."""

import time
import jwt
import sys
from pathlib import Path

def generate_token(private_key_path: str, expiry_hours: int = 24, subject: str = "user"):
    """Generate RS256 JWT token."""
    
    # Load private key
    with open(private_key_path, "r") as f:
        private_key = f.read()
    
    now = int(time.time())
    payload = {
        "sub": subject,
        "iat": now,
        "exp": now + (expiry_hours * 3600),
        "aud": "ai2text-asr",
        "iss": "ai2text-auth"
    }
    
    token = jwt.encode(payload, private_key, algorithm="RS256")
    return token

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 generate_jwt_rs256.py <private_key.pem> [subject] [hours]")
        sys.exit(1)
    
    private_key_path = sys.argv[1]
    subject = sys.argv[2] if len(sys.argv) > 2 else "user"
    hours = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    
    token = generate_token(private_key_path, hours, subject)
    print(token)
```

**Usage:**

```bash
# Generate production token
python3 scripts/generate_jwt_rs256.py jwt-private.pem prod-user 168  # 7 days

# Test with API
TOKEN=$(python3 scripts/generate_jwt_rs256.py jwt-private.pem test-user)
curl http://localhost:8080/health -H "Authorization: Bearer $TOKEN"
```

---

### Task 3.3: Add OpenTelemetry Metrics

**Status:** 🔴 TODO

**Goal:** Expose Prometheus metrics for monitoring.

#### Step 1: Add Metrics Dependencies

**File:** `requirements/observability.txt`

```txt
opentelemetry-api==1.21.0
opentelemetry-sdk==1.21.0
opentelemetry-instrumentation-fastapi==0.42b0
opentelemetry-exporter-prometheus==0.42b0
prometheus-client==0.19.0
```

#### Step 2: Create Metrics Module

**File:** `libs/common/metrics.py`

```python
"""
OpenTelemetry metrics for all services.

Exposes Prometheus-compatible /metrics endpoint.
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST
from fastapi import Response
from fastapi.routing import APIRoute
from typing import Callable
import time

# Metrics
request_count = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['service', 'method', 'endpoint', 'status']
)

request_latency = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['service', 'method', 'endpoint']
)

asr_jobs_active = Gauge(
    'asr_jobs_active',
    'Number of active ASR transcription jobs'
)

asr_jobs_total = Counter(
    'asr_jobs_total',
    'Total ASR jobs processed',
    ['status']  # success, error
)

nats_events_received = Counter(
    'nats_events_received_total',
    'NATS events received',
    ['service', 'subject']
)

nats_events_processed = Counter(
    'nats_events_processed_total',
    'NATS events processed successfully',
    ['service', 'subject']
)

nats_events_failed = Counter(
    'nats_events_failed_total',
    'NATS events that failed processing',
    ['service', 'subject']
)


class MetricsMiddleware:
    """Middleware to collect request metrics."""
    
    def __init__(self, app, service_name: str):
        self.app = app
        self.service_name = service_name
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        method = scope["method"]
        path = scope["path"]
        
        start_time = time.time()
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status = message["status"]
                latency = time.time() - start_time
                
                # Record metrics
                request_count.labels(
                    service=self.service_name,
                    method=method,
                    endpoint=path,
                    status=status
                ).inc()
                
                request_latency.labels(
                    service=self.service_name,
                    method=method,
                    endpoint=path
                ).observe(latency)
            
            await send(message)
        
        await self.app(scope, receive, send_wrapper)


def setup_metrics(app, service_name: str):
    """Setup metrics collection for a FastAPI app."""
    
    # Add middleware
    app.add_middleware(MetricsMiddleware, service_name=service_name)
    
    # Add /metrics endpoint
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )
    
    print(f"[OK] Metrics enabled for {service_name}")
    return app
```

#### Step 3: Wire Metrics into Services

**Example for API Gateway:**

```python
# services/api-gateway/app.py

from libs.common.metrics import setup_metrics

app = FastAPI(title="api-gateway", version="0.1.0")

# Setup metrics
setup_metrics(app, "api-gateway")
```

**Apply to all services:**
- services/api-gateway/app.py
- services/ingestion/app.py
- services/metadata/app.py
- services/nlp-post/app.py
- services/embeddings/app.py
- services/search/app.py

#### Step 4: Test Metrics

```bash
# Check metrics endpoint
curl http://localhost:8080/metrics

# Should show:
# http_requests_total{service="api-gateway",method="GET",endpoint="/health",status="200"} 42
# http_request_duration_seconds_bucket{service="api-gateway",method="GET",endpoint="/health",le="0.005"} 40
# ...
```

---

### Task 3.4: Configure NATS Dead Letter Queues

**Status:** 🔴 TODO

**Goal:** Add DLQ for failed event processing with retry logic.

#### Step 1: Update NATS Streams Configuration

**File:** `infra/nats/streams.json`

```json
{
  "streams": [
    {
      "name": "EVENTS",
      "subjects": [
        "recording.ingested",
        "transcription.completed",
        "nlp.postprocessed",
        "embeddings.indexed",
        "model.promoted"
      ],
      "retention": "limits",
      "storage": "file",
      "max_age": 604800000000000,
      "max_msgs": 1000000,
      "replicas": 1,
      "discard": "old"
    },
    {
      "name": "DLQ",
      "subjects": [
        "dlq.>",
        "dlq.asr",
        "dlq.nlp",
        "dlq.embeddings"
      ],
      "retention": "limits",
      "storage": "file",
      "max_age": 2592000000000000,
      "max_msgs": 100000,
      "replicas": 1
    }
  ],
  "consumers": [
    {
      "stream": "EVENTS",
      "name": "asr",
      "durable_name": "asr",
      "filter_subject": "recording.ingested",
      "ack_policy": "explicit",
      "ack_wait": 30000000000,
      "max_deliver": 3,
      "replay_policy": "instant",
      "max_ack_pending": 100
    },
    {
      "stream": "EVENTS",
      "name": "nlp",
      "durable_name": "nlp",
      "filter_subject": "transcription.completed",
      "ack_policy": "explicit",
      "ack_wait": 30000000000,
      "max_deliver": 3,
      "replay_policy": "instant",
      "max_ack_pending": 100
    },
    {
      "stream": "EVENTS",
      "name": "embeddings",
      "durable_name": "emb",
      "filter_subject": "nlp.postprocessed",
      "ack_policy": "explicit",
      "ack_wait": 30000000000,
      "max_deliver": 3,
      "replay_policy": "instant",
      "max_ack_pending": 100
    }
  ]
}
```

#### Step 2: Add DLQ Handler to Services

**File:** `libs/common/nats_dlq.py`

```python
"""NATS DLQ (Dead Letter Queue) handler."""

import nats
import json
from typing import Callable, Any
import asyncio

class DLQHandler:
    """Handle failed events and route to DLQ."""
    
    def __init__(self, nc: nats.NATS, service_name: str, max_retries: int = 3):
        self.nc = nc
        self.service_name = service_name
        self.max_retries = max_retries
    
    async def process_with_dlq(
        self,
        msg: nats.aio.client.Msg,
        handler: Callable,
        dlq_subject: str = None
    ):
        """
        Process message with automatic DLQ routing on failure.
        
        Args:
            msg: NATS message
            handler: Async function to process the message
            dlq_subject: DLQ subject (defaults to dlq.{service_name})
        """
        dlq_subject = dlq_subject or f"dlq.{self.service_name}"
        
        # Get delivery count from message metadata
        metadata = msg.metadata()
        delivery_count = metadata.num_delivered if metadata else 1
        
        try:
            # Process message
            await handler(msg)
            
            # ACK on success
            await msg.ack()
            
        except Exception as e:
            print(f"[ERROR] Processing failed (attempt {delivery_count}/{self.max_retries}): {e}")
            
            if delivery_count >= self.max_retries:
                # Max retries exceeded - send to DLQ
                dlq_payload = {
                    "original_subject": msg.subject,
                    "original_data": msg.data.decode(),
                    "error": str(e),
                    "delivery_count": delivery_count,
                    "service": self.service_name
                }
                
                await self.nc.publish(
                    dlq_subject,
                    json.dumps(dlq_payload).encode()
                )
                
                print(f"[DLQ] Message sent to {dlq_subject}")
                
                # ACK to remove from original queue
                await msg.ack()
            else:
                # NAK to retry
                await msg.nak(delay=5)  # 5 second delay before retry
```

**Usage in services:**

```python
# services/asr/worker.py

from libs.common.nats_dlq import DLQHandler

dlq_handler = DLQHandler(nc, "asr", max_retries=3)

async def handle_recording_ingested(msg):
    await dlq_handler.process_with_dlq(
        msg,
        _process_recording,  # Your actual handler
        dlq_subject="dlq.asr"
    )

async def _process_recording(msg):
    # Your existing processing code
    evt = json.loads(msg.data.decode())
    # ... process ...
```

---

### Task 3.5: Add Horizontal Pod Autoscaling

**Status:** 🔴 TODO (Kubernetes only)

**Goal:** Auto-scale services based on CPU, memory, and custom metrics.

**File:** `infra/helm/ai-stt/templates/hpa.yaml`

```yaml
{{- range $service := list "api-gateway" "ingestion" "asr" "metadata" "nlp-post" "embeddings" "search" }}
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {{ $service }}-hpa
  namespace: {{ $.Release.Namespace }}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {{ $service }}
  minReplicas: {{ $.Values.autoscaling.minReplicas | default 2 }}
  maxReplicas: {{ $.Values.autoscaling.maxReplicas | default 10 }}
  metrics:
  # CPU-based scaling
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  
  # Memory-based scaling
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  
  {{- if eq $service "api-gateway" }}
  # Custom metric: request latency (requires metrics-server)
  - type: Pods
    pods:
      metric:
        name: http_request_duration_seconds_p95
      target:
        type: AverageValue
        averageValue: "150m"  # 150ms
  {{- end }}
  
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
      - type: Pods
        value: 4
        periodSeconds: 15
      selectPolicy: Max
{{- end }}
```

**Values configuration:**

```yaml
# infra/helm/ai-stt/values.prod.yaml

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 20
  targetCPUUtilization: 70
  targetMemoryUtilization: 80
```

---

## 📊 Monitoring Dashboard

**File:** `infra/monitoring/grafana-dashboard.json`

<details>
<summary>Click to expand Grafana dashboard JSON</summary>

```json
{
  "dashboard": {
    "title": "AI2Text ASR - Production Metrics",
    "panels": [
      {
        "title": "Request Rate (req/s)",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{service}} - {{endpoint}}"
          }
        ]
      },
      {
        "title": "Request Latency (p95)",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "{{service}} - {{endpoint}}"
          }
        ]
      },
      {
        "title": "Error Rate (%)",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m]) / rate(http_requests_total[5m]) * 100",
            "legendFormat": "{{service}}"
          }
        ]
      },
      {
        "title": "ASR Jobs (Active)",
        "targets": [
          {
            "expr": "asr_jobs_active",
            "legendFormat": "Active Jobs"
          }
        ]
      },
      {
        "title": "NATS Consumer Lag",
        "targets": [
          {
            "expr": "nats_consumer_num_pending",
            "legendFormat": "{{consumer}}"
          }
        ]
      }
    ]
  }
}
```

</details>

---

## 🚨 Alerting Rules

**File:** `infra/monitoring/prometheus-rules.yaml`

```yaml
groups:
- name: ai2text_slos
  interval: 30s
  rules:
  
  # SLO: API latency < 150ms (p95)
  - alert: HighAPILatency
    expr: |
      histogram_quantile(0.95,
        rate(http_request_duration_seconds_bucket{service="api-gateway"}[5m])
      ) > 0.15
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "API Gateway latency exceeded SLO"
      description: "p95 latency is {{ $value }}s (target: <150ms)"
  
  # SLO: Error rate < 1%
  - alert: HighErrorRate
    expr: |
      rate(http_requests_total{status=~"5.."}[5m]) /
      rate(http_requests_total[5m]) > 0.01
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value | humanizePercentage }}"
  
  # ASR: Streaming latency < 500ms
  - alert: StreamingLatencyHigh
    expr: |
      histogram_quantile(0.95,
        rate(asr_streaming_partial_latency_seconds_bucket[5m])
      ) > 0.5
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "ASR streaming latency exceeded SLO"
      description: "Partial latency is {{ $value }}s (target: <500ms)"
  
  # NATS: Consumer lag
  - alert: NATSConsumerLag
    expr: nats_consumer_num_pending > 1000
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "NATS consumer {{ $labels.consumer }} has high lag"
      description: "Pending messages: {{ $value }}"
  
  # Service down
  - alert: ServiceDown
    expr: up{job=~"ai2text-.*"} == 0
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "Service {{ $labels.job }} is down"
      description: "Service has been down for 2+ minutes"
```

---

## ✅ Phase 3 Verification

### 1. Test Streaming Latency

```bash
# Measure partial latency (should be <500ms)
python3 scripts/test_streaming_latency.py
```

### 2. Verify RS256 JWT

```bash
# Generate RS256 token
TOKEN=$(python3 scripts/generate_jwt_rs256.py jwt-private.pem)

# Test with API
curl http://localhost:8080/health -H "Authorization: Bearer $TOKEN"
```

### 3. Check Metrics

```bash
# Check all services expose /metrics
for port in 8080 8001 8002 8003 8004 8005 8006; do
  echo "Port $port:"
  curl -s http://localhost:$port/metrics | grep http_requests_total | head -3
done
```

### 4. Test DLQ

```bash
# Inject a message that will fail
# (Manually publish malformed event to NATS)

# Check DLQ
docker compose -f infra/docker-compose.yml exec nats \
  nats stream view DLQ --last
```

### 5. Verify Autoscaling (K8s)

```bash
kubectl get hpa -n ai2text-prod

# Load test to trigger scaling
hey -z 60s -c 100 http://api-gateway/health
```

---

## 📈 SLO Summary

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| API Latency (p95) | <150ms | >150ms for 5min |
| Streaming Partials | <500ms | >500ms for 5min |
| Error Rate | <1% | >1% for 5min |
| Availability | 99.9% | <99% in 1 hour |
| NATS Consumer Lag | <100 | >1000 for 10min |

---

## 🎯 Phase 3 Complete Checklist

- [ ] Streaming ASR emits partials <500ms
- [ ] RS256 JWT configured and tested
- [ ] All services expose `/metrics`
- [ ] Grafana dashboard imported
- [ ] Prometheus alerts configured
- [ ] DLQ configured for all consumers
- [ ] HPA policies deployed (K8s)
- [ ] Load tests passed
- [ ] Monitoring verified for 24 hours
- [ ] Runbook created for common issues

---

**Next: Phase 4 (Training & TTQ) for production model training →**

