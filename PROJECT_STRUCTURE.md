# Project Structure - Complete Overview

## 📁 Complete Directory Structure

```
AI2text/
├── data/                      # Data storage
│   ├── raw/                   # Raw audio files
│   │   ├── train/             # Training audio files
│   │   ├── val/               # Validation audio files
│   │   └── test/              # Test audio files
│   ├── processed/             # Preprocessed data
│   ├── external/              # External datasets
│   └── README.md              # Data directory documentation
│
├── database/                  # Database management
│   ├── init_db.sql           # Database schema
│   ├── db_utils.py           # Database utilities (fully documented)
│   ├── __init__.py           # Package init
│   └── asr_training.db       # SQLite database (created on first run)
│
├── models/                    # Model architectures
│   ├── asr_base.py           # Base transformer ASR model
│   └── __init__.py           # Package init
│
├── training/                  # Training pipeline
│   ├── dataset.py            # Dataset and data loaders
│   ├── train.py              # Main training script
│   ├── evaluate.py           # Evaluation and inference
│   └── __init__.py           # Package init
│
├── preprocessing/             # Audio and text preprocessing
│   ├── audio_processing.py   # Audio feature extraction and augmentation
│   ├── text_cleaning.py      # Vietnamese text normalization and tokenization
│   └── __init__.py           # Package init
│
├── configs/                   # Configuration files
│   ├── default.yaml          # Default training config
│   └── db.yaml               # Database config
│
├── utils/                     # Utilities
│   ├── metrics.py            # WER, CER calculation
│   ├── logger.py             # Logging setup
│   └── __init__.py           # Package init
│
├── scripts/                   # Helper scripts
│   ├── prepare_data.py       # Data preparation and import (enhanced)
│   ├── validate_data.py      # Database validation
│   └── download_sample_data.py # Sample data download info
│
├── tests/                     # Unit tests (comprehensive)
│   ├── __init__.py           # Test package init
│   ├── conftest.py           # Pytest fixtures
│   ├── pytest.ini            # Pytest configuration
│   ├── test_database.py      # Database tests
│   ├── test_preprocessing.py # Preprocessing tests
│   ├── test_models.py        # Model tests
│   ├── test_metrics.py       # Metrics tests
│   ├── test_training.py      # Training tests
│   ├── run_tests_simple.py   # Simple test runner
│   └── README_TESTS.md       # Testing guide
│
├── notebooks/                 # Jupyter notebooks for experimentation
│   ├── README.md             # Notebooks guide
│   └── .gitkeep              # Track empty directory
│
├── checkpoints/               # Model checkpoints
│   └── README.md             # Checkpoints documentation
│
├── logs/                      # Training logs
│   └── README.md             # Logs documentation
│
├── results/                   # Evaluation results
│   └── README.md             # Results documentation
│
├── File kiến thức/           # Knowledge base documents
│   ├── AI2text_embeddings_patch/ # Embeddings patch
│   └── *.docx                # Research documents
│
├── requirements.txt           # Python dependencies
├── README.md                 # Main project README
├── .gitignore                # Git ignore patterns
└── example_data_template.csv  # CSV template for data import
```

## ✅ Files & Directories Status

### Core Modules ✅
- ✅ `database/` - Complete with documented utilities
- ✅ `models/` - ASR model architecture
- ✅ `training/` - Complete training pipeline
- ✅ `preprocessing/` - Audio and text processing
- ✅ `utils/` - Metrics and logging
- ✅ `configs/` - Configuration files

### Scripts ✅
- ✅ `scripts/prepare_data.py` - Enhanced data import with validation
- ✅ `scripts/validate_data.py` - Database validation
- ✅ `scripts/download_sample_data.py` - Sample data download info

### Tests ✅
- ✅ Complete test suite with 50+ test cases
- ✅ Test fixtures and utilities
- ✅ Simple test runner

### Documentation ✅
- ✅ `README.md` - Main project documentation
- ✅ `data/README.md` - Data directory guide
- ✅ `notebooks/README.md` - Notebooks guide
- ✅ `checkpoints/README.md` - Checkpoints guide
- ✅ `logs/README.md` - Logs guide
- ✅ `results/README.md` - Results guide
- ✅ `tests/README_TESTS.md` - Testing guide

### Data Directories ✅
- ✅ `data/raw/` - For raw audio files
- ✅ `data/processed/` - For preprocessed data
- ✅ `data/external/` - For external datasets

### Output Directories ✅
- ✅ `checkpoints/` - Model checkpoints
- ✅ `logs/` - Training logs
- ✅ `results/` - Evaluation results
- ✅ `notebooks/` - Jupyter notebooks

## 🎯 Quick Reference

### Import Data
```bash
python scripts/prepare_data.py --csv data.csv --audio_base data/raw --auto_split
```

### Validate Database
```bash
python scripts/validate_data.py
```

### Download Sample Data Info
```bash
python scripts/download_sample_data.py
```

### Run Tests
```bash
pytest  # or: python tests/run_tests_simple.py
```

### Start Training
```bash
python training/train.py --config configs/default.yaml
```

### Evaluate Model
```bash
python training/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best_model.pt --split test
```

---

**Your project structure is now complete and well-organized!** 🎉

