# AI2Text Security Hardening

Security hardening implementation for AI2Text platform.

## Security Features

### 1. SBOM (Software Bill of Materials)

Generate SBOM for all container images to track dependencies and vulnerabilities.

### 2. Container Security Scanning

Automated vulnerability scanning for all Docker images.

### 3. Secrets Management

Secure storage and rotation of secrets using Kubernetes Secrets and external secret managers.

## Implementation

### SBOM Generation

```bash
# Generate SBOM for service
cd projects/services/gateway
syft packages docker:ghcr.io/yourorg/ai2text-gateway:latest -o spdx-json > sbom.json
```

### Container Scanning

```bash
# Scan container image
trivy image ghcr.io/yourorg/ai2text-gateway:latest
```

### Secrets Management

```bash
# Create Kubernetes secret
kubectl create secret generic ai2text-secrets \
  --from-literal=jwt-secret=your-secret \
  --from-literal=db-password=your-password

# Or use Sealed Secrets
kubeseal < secret.yaml > sealed-secret.yaml
kubectl apply -f sealed-secret.yaml
```

## CI/CD Integration

All security checks run automatically in CI:
- SBOM generation on image build
- Container scanning before push
- Secrets validation
- Dependency vulnerability checks

## Compliance

- **OWASP Top 10**: Addressed
- **CWE**: Common weaknesses covered
- **CVE**: Tracked and remediated

