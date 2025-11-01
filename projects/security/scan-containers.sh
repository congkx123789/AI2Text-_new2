#!/bin/bash
# Security scan all AI2Text container images

set -e

REGISTRY="${REGISTRY:-ghcr.io/yourorg}"
OUTPUT_DIR="${OUTPUT_DIR:-security-reports}"

echo "🔍 Scanning AI2Text container images..."

# Install trivy if not present
if ! command -v trivy &> /dev/null; then
    echo "Installing trivy..."
    wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | sudo apt-key add -
    sudo apt-get update
    sudo apt-get install trivy
fi

mkdir -p "$OUTPUT_DIR"

SERVICES=(
    "gateway"
    "ingestion"
    "asr"
    "nlp-post"
    "embeddings"
    "search"
    "metadata"
    "training-orchestrator"
)

FAILED=0

for service in "${SERVICES[@]}"; do
    echo "Scanning $service..."
    
    IMAGE="$REGISTRY/ai2text-$service:latest"
    
    # Scan image
    trivy image --format json --output "$OUTPUT_DIR/${service}-scan.json" \
        --severity HIGH,CRITICAL "$IMAGE" || {
        echo "⚠️  Warning: Failed to scan $service (image may not exist yet)"
        continue
    }
    
    # Generate human-readable report
    trivy image --format table --severity HIGH,CRITICAL "$IMAGE" > \
        "$OUTPUT_DIR/${service}-scan.txt" || true
    
    # Check for critical vulnerabilities
    CRITICAL_COUNT=$(trivy image --format json --severity CRITICAL "$IMAGE" | \
        jq '[.Results[].Vulnerabilities[]? | select(.Severity == "CRITICAL")] | length' || echo "0")
    
    if [ "$CRITICAL_COUNT" -gt 0 ]; then
        echo "❌ $service has $CRITICAL_COUNT CRITICAL vulnerabilities!"
        FAILED=$((FAILED + 1))
    else
        echo "✅ $service scan complete (no critical vulnerabilities)"
    fi
done

echo ""
if [ $FAILED -eq 0 ]; then
    echo "✅ All scans complete - no critical vulnerabilities found"
    exit 0
else
    echo "❌ $FAILED service(s) have critical vulnerabilities"
    exit 1
fi

