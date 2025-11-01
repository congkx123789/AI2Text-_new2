.PHONY: help install dev test lint format docker-build docker-push

SERVICE_NAME := metadata
VERSION := 1.0.0
REGISTRY := ghcr.io/yourorg
IMAGE := $(REGISTRY)/ai2text-$(SERVICE_NAME):$(VERSION)

help:
	@echo "AI2Text Metadata Service"

install:
	uv pip install -e .

dev:
	uv pip install -e .[dev]

test:
	pytest --cov=app --cov-report=html

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

