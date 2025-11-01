#!/bin/bash
# Generate SBOM for all AI2Text services

set -e

REGISTRY="${REGISTRY:-ghcr.io/yourorg}"
OUTPUT_DIR="${OUTPUT_DIR:-sbom}"

echo "🔐 Generating SBOM for AI2Text services..."

# Install syft if not present
if ! command -v syft &> /dev/null; then
    echo "Installing syft..."
    curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin
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

for service in "${SERVICES[@]}"; do
    echo "Generating SBOM for $service..."
    
    IMAGE="$REGISTRY/ai2text-$service:latest"
    
    # Generate SBOM in SPDX format
    syft packages "docker:$IMAGE" -o spdx-json > "$OUTPUT_DIR/${service}-sbom.json" || {
        echo "⚠️  Warning: Failed to generate SBOM for $service (image may not exist yet)"
    }
    
    echo "✅ SBOM generated: $OUTPUT_DIR/${service}-sbom.json"
done

echo ""
echo "✅ SBOM generation complete!"
echo "Files saved in: $OUTPUT_DIR/"

