.PHONY: help install test docker-build

SERVICE_NAME := asr
VERSION := 1.0.0
REGISTRY := ghcr.io/yourorg
IMAGE := $(REGISTRY)/ai2text-$(SERVICE_NAME):$(VERSION)

help:
	@echo "AI2Text ASR Service"

install:
	uv pip install -e .[dev]

test:
	pytest

docker-build:
	docker build -t $(IMAGE) .

docker-push: docker-build
	docker push $(IMAGE)

