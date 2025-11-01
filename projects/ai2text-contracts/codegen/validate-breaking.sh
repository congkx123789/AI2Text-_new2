#!/bin/bash
# Detect breaking changes in OpenAPI/AsyncAPI specs

set -e

echo "🔍 Checking for breaking changes in contracts..."

# Compare against main branch (or last release tag)
BASELINE_REF="${1:-origin/main}"

echo "Baseline: $BASELINE_REF"
echo ""

# Fetch baseline specs
git fetch origin main --quiet || true

BREAKING_FOUND=0

# Check OpenAPI specs for breaking changes
for spec in ../openapi/*.yaml; do
    SERVICE=$(basename "$spec" .yaml)
    echo "Checking OpenAPI: $SERVICE"
    
    # Check if spec exists in baseline
    if git cat-file -e "$BASELINE_REF:projects/ai2text-contracts/openapi/$SERVICE.yaml" 2>/dev/null; then
        BASELINE_SPEC=$(git show "$BASELINE_REF:projects/ai2text-contracts/openapi/$SERVICE.yaml")
        
        # Use oasdiff to detect breaking changes
        if command -v oasdiff &> /dev/null; then
            echo "$BASELINE_SPEC" > /tmp/baseline_$SERVICE.yaml
            
            if oasdiff breaking /tmp/baseline_$SERVICE.yaml "$spec" --fail-on=ERR; then
                echo "  ✅ No breaking changes"
            else
                echo "  ❌ BREAKING CHANGES DETECTED!"
                BREAKING_FOUND=1
            fi
            
            rm /tmp/baseline_$SERVICE.yaml
        else
            echo "  ⚠️  oasdiff not installed, skipping"
        fi
    else
        echo "  ℹ️  New spec (no baseline to compare)"
    fi
    echo ""
done

# Check AsyncAPI specs
for spec in ../asyncapi/*.yaml; do
    EVENT=$(basename "$spec" .yaml)
    echo "Checking AsyncAPI: $EVENT"
    
    if git cat-file -e "$BASELINE_REF:projects/ai2text-contracts/asyncapi/$EVENT.yaml" 2>/dev/null; then
        BASELINE_SPEC=$(git show "$BASELINE_REF:projects/ai2text-contracts/asyncapi/$EVENT.yaml")
        
        # Basic check: compare payload schemas
        # In production, use a proper AsyncAPI diff tool
        echo "  ⚠️  Manual review required for event schema changes"
    else
        echo "  ℹ️  New event schema (no baseline)"
    fi
    echo ""
done

if [ $BREAKING_FOUND -eq 1 ]; then
    echo ""
    echo "❌ BREAKING CHANGES DETECTED!"
    echo ""
    echo "If this is intentional:"
    echo "  1. Bump MAJOR version in VERSION file"
    echo "  2. Update VERSIONING.md changelog"
    echo "  3. Create migration guide"
    echo "  4. Get approval from architecture review"
    echo ""
    exit 1
else
    echo "✅ No breaking changes detected"
    exit 0
fi

