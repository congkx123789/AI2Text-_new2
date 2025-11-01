# AI2Text Service Template

This is a template for creating new AI2Text microservices.

## Quick Start

1. Copy this template:
   ```bash
   cp -r .template new-service-name
   cd new-service-name
   ```

2. Update service name in:
   - `pyproject.toml`
   - `Dockerfile`
   - `Makefile`
   - `.github/workflows/ci.yml`
   - `README.md`

3. Implement your handlers in `app/main.py`

4. Add tests in `tests/`

5. Deploy:
   ```bash
   make docker-build
   make docker-push
   ```

## Structure

```
service-name/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app
│   ├── deps.py          # Dependencies
│   └── handlers/        # Request handlers
├── tests/
│   ├── __init__.py
│   └── test_main.py
├── Dockerfile
├── Makefile
├── pyproject.toml
├── README.md
└── .github/
    └── workflows/
        └── ci.yml
```

## Development

```bash
# Install dependencies
uv pip install -e .[dev]

# Run locally
uv run python -m app.main

# Run tests
pytest

# Lint
ruff check app/ tests/
black app/ tests/

# Type check
mypy app/
```

## Deployment

```bash
# Build image
make docker-build

# Push to registry
make docker-push

# Deploy to k8s
helm upgrade --install service-name ../../ai2text-platform/helm/charts/service-name
```
