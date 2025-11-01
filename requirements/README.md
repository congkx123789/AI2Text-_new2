# Requirements Files

This directory contains all Python dependency files.

## Files

- **`base.txt`** - Core dependencies (models, training, preprocessing)
- **`api.txt`** - Additional dependencies for API and advanced features

## Installation

### Install Base Requirements
```bash
pip install -r requirements/base.txt
```

### Install API Requirements (for REST API)
```bash
pip install -r requirements/api.txt
```

### Install All Requirements
```bash
pip install -r requirements/base.txt -r requirements/api.txt
```

## Notes

- Base requirements are required for core functionality
- API requirements are optional (needed for REST API, embeddings, etc.)

