.PHONY: help validate clients servers clean all

help:
	@echo "AI2Text Contracts - Code Generation"
	@echo ""
	@echo "Targets:"
	@echo "  validate    - Validate all OpenAPI and AsyncAPI specs"
	@echo "  clients     - Generate typed clients (Python, TypeScript)"
	@echo "  servers     - Generate server stubs (FastAPI)"
	@echo "  clean       - Remove generated files"
	@echo "  all         - Validate and generate everything"

validate:
	@echo "Validating OpenAPI specifications..."
	@for file in ../openapi/*.yaml; do \
		echo "  - $$(basename $$file)"; \
		npx --yes @apidevtools/swagger-cli validate $$file || exit 1; \
	done
	@echo "✅ All OpenAPI specs valid"
	@echo ""
	@echo "Validating AsyncAPI specifications..."
	@for file in ../asyncapi/*.yaml; do \
		echo "  - $$(basename $$file)"; \
		npx --yes @asyncapi/cli validate $$file || exit 1; \
	done
	@echo "✅ All AsyncAPI specs valid"

clients:
	@echo "Generating Python clients..."
	@mkdir -p out/python
	@for file in ../openapi/*.yaml; do \
		service=$$(basename $$file .yaml); \
		echo "  - $$service"; \
		npx --yes @openapitools/openapi-generator-cli generate \
			-i $$file \
			-g python \
			-o out/python/$$service-client \
			--additional-properties=packageName=ai2text_$${service}_client; \
	done
	@echo "✅ Python clients generated"

servers:
	@echo "Generating FastAPI server stubs..."
	@mkdir -p out/python-server
	@for file in ../openapi/*.yaml; do \
		service=$$(basename $$file .yaml); \
		echo "  - $$service"; \
		npx --yes @openapitools/openapi-generator-cli generate \
			-i $$file \
			-g python-fastapi \
			-o out/python-server/$$service; \
	done
	@echo "✅ FastAPI stubs generated"

clean:
	@echo "Cleaning generated files..."
	@rm -rf out/
	@echo "✅ Cleaned"

all: validate clients servers
	@echo "✅ All tasks complete"
