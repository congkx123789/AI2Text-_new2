# AI2Text Performance Testing Framework

Performance testing and SLO validation for AI2Text services.

## Tools

- **k6** - Load testing tool
- **Grafana** - Metrics visualization
- **Prometheus** - Metrics collection

## Test Scenarios

### Gateway Performance
- **Target**: p95 < 150ms at 1000 rps
- **Script**: `gateway-load-test.js`

### ASR Streaming
- **Target**: p95 < 500ms E2E with 50 concurrent streams
- **Script**: `asr-streaming-test.js`

### Search Performance
- **Target**: p95 < 50ms on 10M vectors
- **Script**: `search-load-test.js`

### Metadata Write
- **Target**: p95 < 40ms
- **Script**: `metadata-write-test.js`

## Running Tests

### Prerequisites
```bash
# Install k6
brew install k6  # macOS
# or
choco install k6  # Windows
# or
apt-get install k6  # Linux
```

### Run Individual Test
```bash
cd tests/performance

# Gateway test
k6 run gateway-load-test.js

# ASR streaming test
k6 run asr-streaming-test.js

# Search test
k6 run search-load-test.js
```

### Run Full Suite
```bash
./run-all-tests.sh
```

## Test Configuration

### Stages
Tests use ramping stages:
1. **Ramp-up**: 0 → target VUs over 2 minutes
2. **Sustain**: Hold at target for 5 minutes
3. **Spike**: 2x target for 1 minute
4. **Ramp-down**: target → 0 over 1 minute

### Thresholds
All tests enforce SLO thresholds:
```javascript
thresholds: {
  'http_req_duration{percentile:95}': ['p(95)<150'],  // p95 < 150ms
  'http_req_failed': ['rate<0.01'],  // < 1% errors
}
```

## Interpreting Results

### Success Criteria
- ✅ All thresholds pass
- ✅ No HTTP 5xx errors
- ✅ p95 latency within target
- ✅ Error rate < 1%

### Failure Analysis
If tests fail:
1. Check Grafana dashboards for bottlenecks
2. Review service logs
3. Check resource utilization (CPU, memory)
4. Verify database/cache performance

## CI Integration

Tests run automatically:
- **On PR**: Smoke test (low load)
- **Nightly**: Full load test suite
- **Pre-release**: Extended soak test (1 hour)

## Metrics

Results are exported to:
- **Grafana**: Real-time dashboards
- **JSON**: `results/*.json`
- **HTML**: `results/*.html` reports

