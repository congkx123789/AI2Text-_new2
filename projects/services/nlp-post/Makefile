.PHONY: help install test docker-build

SERVICE_NAME := nlp-post
VERSION := 1.0.0

install:
	uv pip install -e .[dev]

test:
	pytest

docker-build:
	docker build -t ghcr.io/yourorg/ai2text-$(SERVICE_NAME):$(VERSION) .

