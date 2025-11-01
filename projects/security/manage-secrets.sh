#!/bin/bash
# Secrets management utilities for AI2Text

set -e

NAMESPACE="${NAMESPACE:-ai2text-dev}"
SECRET_NAME="${SECRET_NAME:-ai2text-secrets}"

create_secrets() {
    echo "🔐 Creating Kubernetes secrets..."
    
    kubectl create secret generic "$SECRET_NAME" \
        --from-literal=jwt-secret="${JWT_SECRET:-$(openssl rand -hex 32)}" \
        --from-literal=db-password="${DB_PASSWORD:-$(openssl rand -hex 16)}" \
        --from-literal=minio-access-key="${MINIO_ACCESS_KEY:-minioadmin}" \
        --from-literal=minio-secret-key="${MINIO_SECRET_KEY:-$(openssl rand -hex 16)}" \
        --namespace="$NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    echo "✅ Secrets created in namespace: $NAMESPACE"
}

rotate_secrets() {
    echo "🔄 Rotating secrets..."
    
    # Generate new secrets
    NEW_JWT_SECRET=$(openssl rand -hex 32)
    NEW_DB_PASSWORD=$(openssl rand -hex 16)
    
    # Update secret
    kubectl create secret generic "$SECRET_NAME" \
        --from-literal=jwt-secret="$NEW_JWT_SECRET" \
        --from-literal=db-password="$NEW_DB_PASSWORD" \
        --namespace="$NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    echo "✅ Secrets rotated - restart services to pick up new secrets"
}

list_secrets() {
    echo "📋 Listing secrets in namespace: $NAMESPACE"
    kubectl get secrets -n "$NAMESPACE" | grep "$SECRET_NAME"
}

case "${1:-help}" in
    create)
        create_secrets
        ;;
    rotate)
        rotate_secrets
        ;;
    list)
        list_secrets
        ;;
    *)
        echo "Usage: $0 {create|rotate|list}"
        echo ""
        echo "  create  - Create secrets in Kubernetes"
        echo "  rotate  - Rotate secrets (generate new values)"
        echo "  list    - List existing secrets"
        exit 1
        ;;
esac

