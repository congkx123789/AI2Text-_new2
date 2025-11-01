.PHONY: install test docker-build

install:
	uv pip install -e .[dev]

test:
	pytest

docker-build:
	docker build -t ghcr.io/yourorg/ai2text-embeddings:1.0.0 .

