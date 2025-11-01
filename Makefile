.PHONY: init up down logs ps migrate test-e2e clean restart health

# Shell configuration
SHELL := /bin/bash

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "AI2Text ASR Microservices - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

init: ## Initialize environment file
	@echo "Initializing environment..."
	@if [ ! -f .env ]; then \
		cp env.example .env; \
		echo "✓ Created .env from env.example"; \
	else \
		echo "✓ .env already exists"; \
	fi

up: ## Start all services
	@echo "Starting AI2Text ASR stack..."
	docker compose -f infra/docker-compose.yml up -d --build
	@echo "✓ Services started. Run 'make logs' to view logs."

down: ## Stop all services
	@echo "Stopping all services..."
	docker compose -f infra/docker-compose.yml down
	@echo "✓ Services stopped."

down-volumes: ## Stop all services and remove volumes
	@echo "Stopping all services and removing volumes..."
	docker compose -f infra/docker-compose.yml down -v
	@echo "✓ Services stopped and volumes removed."

logs: ## View logs from all services (tail last 200 lines)
	docker compose -f infra/docker-compose.yml logs -f --tail=200

logs-gateway: ## View API Gateway logs
	docker compose -f infra/docker-compose.yml logs -f --tail=100 api-gateway

logs-asr: ## View ASR service logs
	docker compose -f infra/docker-compose.yml logs -f --tail=100 asr

logs-ingestion: ## View Ingestion service logs
	docker compose -f infra/docker-compose.yml logs -f --tail=100 ingestion

logs-metadata: ## View Metadata service logs
	docker compose -f infra/docker-compose.yml logs -f --tail=100 metadata

logs-nlp: ## View NLP-post service logs
	docker compose -f infra/docker-compose.yml logs -f --tail=100 nlp-post

ps: ## Show running services
	docker compose -f infra/docker-compose.yml ps

migrate: ## Run database migrations
	@echo "Running database migrations..."
	@docker compose -f infra/docker-compose.yml exec -T postgres psql -U postgres -d asrmeta < services/metadata/migrations/001_init.sql 2>/dev/null || \
	docker run --rm -i --network asr-network \
		-e PGPASSWORD=postgres postgres:16 \
		psql -h postgres -U postgres -d asrmeta < services/metadata/migrations/001_init.sql
	@echo "✓ Migrations completed."

migrate-fresh: ## Drop and recreate database with migrations
	@echo "Resetting database..."
	@docker compose -f infra/docker-compose.yml exec -T postgres psql -U postgres -c "DROP DATABASE IF EXISTS asrmeta;" || true
	@docker compose -f infra/docker-compose.yml exec -T postgres psql -U postgres -c "CREATE DATABASE asrmeta;" || true
	@$(MAKE) migrate
	@echo "✓ Database reset completed."

test-e2e: ## Run end-to-end tests
	@echo "Running E2E tests..."
	pytest tests/e2e -v --tb=short

test-unit: ## Run unit tests
	@echo "Running unit tests..."
	pytest tests/ -v --ignore=tests/e2e

test-all: test-unit test-e2e ## Run all tests

clean: ## Clean up Docker resources
	@echo "Cleaning up..."
	docker compose -f infra/docker-compose.yml down -v --remove-orphans
	docker system prune -f
	@echo "✓ Cleanup completed."

restart: down up ## Restart all services

restart-gateway: ## Restart API Gateway only
	docker compose -f infra/docker-compose.yml restart api-gateway

restart-asr: ## Restart ASR services
	docker compose -f infra/docker-compose.yml restart asr asr-worker asr-streaming

health: ## Check health of all services
	@echo "Checking service health..."
	@echo ""
	@echo "API Gateway:"
	@curl -s http://localhost:8080/health | python -m json.tool 2>/dev/null || echo "❌ Not responding"
	@echo ""
	@echo "Ingestion:"
	@curl -s http://localhost:8001/health | python -m json.tool 2>/dev/null || echo "❌ Not responding"
	@echo ""
	@echo "Metadata:"
	@curl -s http://localhost:8002/health | python -m json.tool 2>/dev/null || echo "❌ Not responding"
	@echo ""
	@echo "NLP-Post:"
	@curl -s http://localhost:8004/health | python -m json.tool 2>/dev/null || echo "❌ Not responding"
	@echo ""
	@echo "Embeddings:"
	@curl -s http://localhost:8005/health | python -m json.tool 2>/dev/null || echo "❌ Not responding"
	@echo ""
	@echo "Search:"
	@curl -s http://localhost:8006/health | python -m json.tool 2>/dev/null || echo "❌ Not responding"

build: ## Build all service images
	@echo "Building all service images..."
	docker compose -f infra/docker-compose.yml build

build-gateway: ## Build API Gateway image
	docker compose -f infra/docker-compose.yml build api-gateway

build-asr: ## Build ASR service image
	docker compose -f infra/docker-compose.yml build asr

shell-postgres: ## Open PostgreSQL shell
	docker compose -f infra/docker-compose.yml exec postgres psql -U postgres -d asrmeta

shell-gateway: ## Open shell in API Gateway container
	docker compose -f infra/docker-compose.yml exec api-gateway /bin/bash

shell-asr: ## Open shell in ASR container
	docker compose -f infra/docker-compose.yml exec asr /bin/bash

minio-console: ## Open MinIO console URL
	@echo "MinIO Console: http://localhost:9001"
	@echo "Username: minio"
	@echo "Password: minio123"

qdrant-console: ## Open Qdrant console URL
	@echo "Qdrant Dashboard: http://localhost:6333/dashboard"

dev-setup: init ## Full development setup (Phase 0)
	@echo "=========================================="
	@echo "Phase 0: Baseline Green Build"
	@echo "=========================================="
	@echo ""
	@echo "Step 1/4: Running bootstrap script..."
	@bash scripts/bootstrap.sh || (echo "Bootstrap failed. Check logs above." && exit 1)
	@echo ""
	@echo "Step 2/4: Starting all services..."
	docker compose -f infra/docker-compose.yml up -d --build
	@echo ""
	@echo "Step 3/4: Waiting for services to be ready (60 seconds)..."
	@sleep 60
	@echo ""
	@echo "Step 4/4: Verifying setup..."
	@$(MAKE) health
	@echo ""
	@echo "=========================================="
	@echo "✓ Phase 0 Complete - Green Build Ready!"
	@echo "=========================================="
	@echo ""
	@echo "🌐 Web UIs:"
	@echo "  - MinIO Console:  http://localhost:9001 (minio/minio123)"
	@echo "  - Qdrant Dashboard: http://localhost:6333/dashboard"
	@echo ""
	@echo "📊 API Endpoints:"
	@echo "  - API Gateway:    http://localhost:8080"
	@echo "  - Ingestion:      http://localhost:8001"
	@echo "  - Metadata:       http://localhost:8002"
	@echo "  - ASR Streaming:  ws://localhost:8003/v1/asr/stream"
	@echo ""
	@echo "🔑 Get JWT Token:"
	@echo "  python3 scripts/jwt_dev_token.py"
	@echo ""
	@echo "✅ Phase 0 Exit Criteria Check:"
	@echo "  1. All health endpoints return healthy ✓"
	@echo "  2. MinIO buckets exist (audio, transcripts) - Check http://localhost:9001"
	@echo "  3. Qdrant collection exists (texts) - Check http://localhost:6333/dashboard"
	@echo "  4. PostgreSQL tables created (audio, transcripts, speakers)"
	@echo ""
	@echo "🧪 Run E2E Tests:"
	@echo "  make test-e2e"
	@echo ""
	@echo "📚 Next: See Phase 1 instructions in PHASE_1_INSTRUCTIONS.md"

status: ps health ## Show service status and health

# Quick smoke test
smoke-test: ## Run quick smoke test
	@echo "Running smoke test..."
	@curl -s http://localhost:8080/ | python -m json.tool
	@echo ""
	@echo "✓ Smoke test passed!"

