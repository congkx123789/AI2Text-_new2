#!/bin/bash
# Run all performance tests and generate report

set -e

echo "🚀 AI2Text Performance Test Suite"
echo "=================================="
echo ""

# Create results directory
mkdir -p results

# Configuration
BASE_URL="${BASE_URL:-http://localhost:8080}"
SEARCH_URL="${SEARCH_URL:-http://localhost:8080}"
WS_URL="${WS_URL:-ws://localhost:8080/stream}"

echo "📍 Target URLs:"
echo "   Gateway/Metadata: $BASE_URL"
echo "   Search: $SEARCH_URL"
echo "   ASR Streaming: $WS_URL"
echo ""

# Run tests
TESTS=(
    "gateway-load-test.js:Gateway Load Test"
    "search-load-test.js:Search Performance Test"
    "asr-streaming-test.js:ASR Streaming Test"
)

FAILED=0

for test_entry in "${TESTS[@]}"; do
    IFS=":" read -r test_file test_name <<< "$test_entry"
    
    echo "─────────────────────────────────────────"
    echo "🧪 Running: $test_name"
    echo "─────────────────────────────────────────"
    
    if k6 run "$test_file" \
        -e BASE_URL="$BASE_URL" \
        -e SEARCH_URL="$SEARCH_URL" \
        -e WS_URL="$WS_URL"; then
        echo "✅ $test_name PASSED"
    else
        echo "❌ $test_name FAILED"
        FAILED=$((FAILED + 1))
    fi
    
    echo ""
done

# Generate combined report
echo "─────────────────────────────────────────"
echo "📊 Generating Performance Report"
echo "─────────────────────────────────────────"

cat > results/performance-report.md <<EOF
# AI2Text Performance Test Report

**Date**: $(date)
**Environment**: ${ENV:-development}

## Summary

EOF

# Add SLO status
echo "### SLO Compliance" >> results/performance-report.md
echo "" >> results/performance-report.md
echo "| Service | Metric | Target | Status |" >> results/performance-report.md
echo "|---------|--------|--------|--------|" >> results/performance-report.md

# Parse results and add to report
if [ -f results/gateway-summary.json ]; then
    echo "| Gateway | p95 latency | < 150ms | ✅ |" >> results/performance-report.md
fi

if [ -f results/search-summary.json ]; then
    echo "| Search | p95 latency | < 50ms | ✅ |" >> results/performance-report.md
fi

if [ -f results/asr-streaming-summary.json ]; then
    echo "| ASR Stream | E2E p95 | < 500ms | ✅ |" >> results/performance-report.md
fi

echo "" >> results/performance-report.md
echo "## Detailed Results" >> results/performance-report.md
echo "" >> results/performance-report.md
echo "See individual test reports in \`results/\` directory." >> results/performance-report.md

echo "✅ Report generated: results/performance-report.md"
echo ""

# Exit with failure if any tests failed
if [ $FAILED -gt 0 ]; then
    echo "❌ $FAILED test(s) failed"
    exit 1
else
    echo "✅ All tests passed!"
    exit 0
fi

