.PHONY: help install dev test lint format docker-build docker-push

SERVICE_NAME := search
VERSION := 1.0.0
REGISTRY := ghcr.io/yourorg
IMAGE := $(REGISTRY)/ai2text-$(SERVICE_NAME):$(VERSION)

help:
	@echo "AI2Text Search Service"
	@echo ""
	@echo "Targets:"
	@echo "  install       - Install dependencies"
	@echo "  dev           - Install dev dependencies"
	@echo "  test          - Run tests"
	@echo "  lint          - Run linters"
	@echo "  format        - Format code"
	@echo "  docker-build  - Build Docker image"
	@echo "  docker-push   - Push to registry"

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

docker-build:
	docker build -t $(IMAGE) .

docker-push: docker-build
	docker push $(IMAGE)

