# AI2Text Contracts

**Single source of truth** for all API contracts and event schemas.

This repository contains:
- OpenAPI 3.0 specifications for REST APIs
- AsyncAPI 3.0 specifications for event-driven communication
- JSON Schemas for shared data types
- Code generation tooling for typed clients and servers

## Structure

```
contracts/
├── openapi/           # REST API specifications
│   ├── gateway.yaml
│   ├── search.yaml
│   ├── metadata.yaml
│   └── ingestion.yaml
├── asyncapi/          # Event specifications
│   ├── recording.ingested.yaml
│   ├── transcription.completed.yaml
│   ├── nlp.postprocessed.yaml
│   ├── embeddings.created.yaml
│   └── model.promoted.yaml
├── schemas/           # Shared JSON schemas
│   └── common.json
└── codegen/           # Code generation tooling
    ├── Makefile
    └── templates/
```

## Usage

### Generate Clients

```bash
cd codegen
make clients  # Generates Python, TypeScript, etc.
```

### Generate Server Stubs

```bash
make servers  # Generates FastAPI stubs, etc.
```

### Validate Contracts

```bash
make validate  # Validates all OpenAPI/AsyncAPI specs
```

## Versioning

- **SemVer** for contract versions
- Breaking changes → new major version
- Additive changes → minor version bump
- Bug fixes → patch version bump

## Integration

All services must:
1. Pull contract specs from this repo
2. Run contract tests in CI
3. Fail CI if contracts are incompatible
4. Use generated clients/servers for type safety


