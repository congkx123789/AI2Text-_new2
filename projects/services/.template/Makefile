.PHONY: help install dev test lint format typecheck docker-build docker-push clean

SERVICE_NAME := service-name
VERSION := $(shell cat pyproject.toml | grep "^version = " | cut -d '"' -f 2)
REGISTRY := ghcr.io/yourorg
IMAGE := $(REGISTRY)/$(SERVICE_NAME):$(VERSION)
IMAGE_LATEST := $(REGISTRY)/$(SERVICE_NAME):latest

help:
	@echo "AI2Text $(SERVICE_NAME) - Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  install       - Install production dependencies"
	@echo "  dev           - Install development dependencies"
	@echo "  test          - Run tests with coverage"
	@echo "  lint          - Run linters (ruff, black check)"
	@echo "  format        - Format code (black, ruff fix)"
	@echo "  typecheck     - Run mypy type checking"
	@echo "  docker-build  - Build Docker image"
	@echo "  docker-push   - Push Docker image to registry"
	@echo "  clean         - Remove cache and build artifacts"

install:
	uv pip install -e .

dev:
	uv pip install -e .[dev]

test:
	pytest --cov=app --cov-report=html --cov-report=term

lint:
	ruff check app/ tests/
	black --check app/ tests/

format:
	black app/ tests/
	ruff check --fix app/ tests/

typecheck:
	mypy app/

docker-build:
	docker build -t $(IMAGE) -t $(IMAGE_LATEST) .
	@echo "Built: $(IMAGE)"

docker-push: docker-build
	docker push $(IMAGE)
	docker push $(IMAGE_LATEST)
	@echo "Pushed: $(IMAGE)"

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
