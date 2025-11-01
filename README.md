# AI2Text - Vietnamese Speech-to-Text System

## 🎉 **NOW PRODUCTION-READY!** Enterprise Microservices Platform

A comprehensive, production-ready Automatic Speech Recognition (ASR) system designed for Vietnamese language with complete microservices architecture. Built for both training and deployment at scale.

### ⚡ **NEW: Production-Grade Multi-Project Architecture**

Your project now includes a **complete production implementation** in the `projects/` directory:
- ✅ **8 Microservices** (Gateway, Ingestion, ASR, NLP-Post, Embeddings, Search, Metadata, Training)
- ✅ **Contract-First Development** (OpenAPI/AsyncAPI v1.1.0 with versioning)
- ✅ **SLO Tracking** (Grafana dashboards + Prometheus alerts)
- ✅ **Performance Testing** (k6 load tests for all services)
- ✅ **Production Roadmap** (Nov-Dec 2025 sprint plan to GA)

**🚀 START HERE**: **[README_PRODUCTION_READY.md](README_PRODUCTION_READY.md)** | **[projects/README.md](projects/README.md)**

---

## 🚀 Quick Start - Microservices (NEW!)

Get the complete ASR stack running in 5 minutes:

```bash
# 1. Bootstrap infrastructure
bash scripts/bootstrap.sh

# 2. Start all services
docker compose -f infra/docker-compose.yml up -d

# 3. Verify setup
python3 scripts/verify_setup.py

# 4. Test the API
python3 scripts/jwt_dev_token.py  # Get auth token
# Then upload audio via http://localhost:8080/v1/ingest
```

**📚 For detailed setup:** See [`RUN_GUIDE.md`](RUN_GUIDE.md)  
**🎯 Quick commands:** See [`QUICK_REFERENCE.md`](QUICK_REFERENCE.md)  
**🏗️ Architecture:** See [`MICROSERVICES_SETUP_GUIDE.md`](MICROSERVICES_SETUP_GUIDE.md)

---

## 🌟 Features

### Microservices Architecture (Production-Ready)
- **API Gateway**: JWT authentication, rate limiting (60/min), CORS support
- **Real-time Streaming**: WebSocket-based ASR with sub-100ms latency
- **Batch Processing**: Async pipeline with event-driven processing (NATS)
- **Vietnamese NLP**: Automatic diacritics restoration & typo correction
- **Semantic Search**: Vector-based search over transcripts (Qdrant)
- **Three-Tier Storage**: S3 (audio) + PostgreSQL (metadata) + Vector DB (embeddings)
- **Speaker-Level Split**: Database-enforced data split to prevent leakage
- **One-Command Setup**: Complete stack running in 5 minutes

### Model Training
- **Transformer-based Architecture**: Modern encoder with CTC loss
- **Vietnamese Language Support**: Specialized text normalization & tokenization
- **Resource-Efficient**: Mixed precision, gradient accumulation, efficient data loading
- **Complete Pipeline**: End-to-end from preprocessing to deployment
- **Database Integration**: Experiment tracking with SQLite/PostgreSQL
- **Modular Design**: Easy to extend individual components
- **Audio Augmentation**: Built-in augmentation for robust training

## 📁 Project Structure

> **Note**: Documentation has been reorganized. See `docs/PROJECT_ORGANIZATION.md` for details.
> - Architecture docs: `docs/architecture/`
> - User guides: `docs/guides/`
> - Requirements: `requirements/`

```
AI2text/
├── data/                      # Data storage
│   ├── raw/                   # Raw audio files
│   ├── processed/             # Preprocessed data
│   └── external/              # External datasets
│
├── database/                  # Database management
│   ├── init_db.sql           # Database schema
│   ├── db_utils.py           # Database utilities
│   └── asr_training.db       # SQLite database (created on first run)
│
├── models/                    # Model architectures
│   ├── asr_base.py           # Base transformer ASR model
│   └── __init__.py
│
├── training/                  # Training pipeline
│   ├── dataset.py            # Dataset and data loaders
│   ├── train.py              # Main training script
│   ├── evaluate.py           # Evaluation and inference
│   └── __init__.py
│
├── preprocessing/             # Audio and text preprocessing
│   ├── audio_processing.py   # Audio feature extraction and augmentation
│   ├── text_cleaning.py      # Vietnamese text normalization and tokenization
│   └── __init__.py
│
├── configs/                   # Configuration files
│   ├── default.yaml          # Default training config
│   └── db.yaml               # Database config
│
├── utils/                     # Utilities
│   ├── metrics.py            # WER, CER calculation
│   ├── logger.py             # Logging setup
│   └── __init__.py
│
├── scripts/                   # Helper scripts
│   ├── prepare_data.py       # Data preparation and import
│   └── download_sample_data.py
│
├── tests/                     # Unit tests
├── notebooks/                 # Jupyter notebooks for experimentation
├── File kiến thức/           # Knowledge base documents
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements/base.txt
```

### 2. Prepare Your Data

Your data should be organized in a CSV file with the following columns:

- `file_path`: Path to audio file (WAV, MP3, FLAC, OGG)
- `transcript`: Vietnamese transcription text
- `split` (optional): 'train', 'val', or 'test'
- `speaker_id` (optional): Speaker identifier

Example CSV:
```csv
file_path,transcript,split,speaker_id
data/raw/audio1.wav,xin chào việt nam,train,speaker_01
data/raw/audio2.wav,tôi là sinh viên,train,speaker_02
data/raw/audio3.wav,hôm nay trời đẹp,val,speaker_01
```

### 3. Import Data into Database

```bash
# Import data from CSV
python scripts/prepare_data.py --csv your_data.csv --audio_base data/raw --auto_split

# The --auto_split flag automatically creates train/val/test splits if not specified in CSV
```

### 4. Train the Model

```bash
# Train with default configuration
python training/train.py --config configs/default.yaml

# Resume from checkpoint
python training/train.py --config configs/default.yaml --resume checkpoints/checkpoint_epoch_10.pt
```

### 5. Evaluate the Model

```bash
# Evaluate on test set
python training/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best_model.pt --split test --output results/evaluation.csv

# Transcribe a single audio file
python training/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best_model.pt --audio path/to/audio.wav
```

## 📊 Database Schema

The system uses SQLite to track:

- **AudioFiles**: Audio file metadata (duration, sample rate, quality)
- **Transcripts**: Ground truth transcriptions
- **DataSplits**: Train/validation/test split assignments
- **Models**: Model architectures and configurations
- **TrainingRuns**: Training experiments with hyperparameters
- **EpochMetrics**: Per-epoch training metrics
- **Predictions**: Model predictions for analysis

View training history:
```python
from database.db_utils import ASRDatabase

db = ASRDatabase()
history = db.get_training_history(training_run_id=1)
summary = db.get_training_runs_summary(limit=10)
best_model = db.get_best_model(metric='word_error_rate')
```

## ⚙️ Configuration

Edit `configs/default.yaml` to customize training:

```yaml
# Model architecture
d_model: 256                # Model dimension
num_encoder_layers: 6       # Number of transformer layers
num_heads: 4                # Attention heads
d_ff: 1024                  # Feed-forward dimension

# Training
batch_size: 16              # Batch size (reduce for weak GPU)
num_epochs: 50              # Training epochs
learning_rate: 0.0001       # Learning rate
use_amp: true               # Mixed precision training

# Audio processing
sample_rate: 16000          # Target sample rate
n_mels: 80                  # Mel spectrogram bands
```

### Optimization for Weak Hardware

For computers with limited resources:

1. **Reduce batch size**: Set `batch_size: 4` or `8`
2. **Reduce model size**: 
   - `d_model: 128`
   - `num_encoder_layers: 4`
   - `d_ff: 512`
3. **Enable mixed precision**: `use_amp: true` (requires CUDA)
4. **Reduce workers**: `num_workers: 2`

## 📈 Model Architecture

The system uses a transformer-based encoder architecture with:

- **Convolutional Subsampling**: Reduces sequence length for efficiency
- **Multi-head Self-Attention**: Captures long-range dependencies
- **Position-wise Feed-Forward**: Non-linear transformations
- **CTC Loss**: Connectionist Temporal Classification for alignment-free training

Total parameters with default config: ~15M (highly efficient!)

## 🎯 Performance Metrics

The system tracks:

- **WER (Word Error Rate)**: Primary metric for ASR evaluation
- **CER (Character Error Rate)**: Character-level accuracy
- **Training Loss**: Cross-entropy loss during training
- **Validation Loss**: Generalization performance

## 📝 Text Normalization

Vietnamese-specific preprocessing:

- Unicode normalization (NFC)
- Number-to-word conversion (0 → không, 1 → một)
- Abbreviation expansion (tp. → thành phố)
- Filler word removal (ừm, à, ờ)
- Tone mark preservation
- Lowercase conversion

## 🔊 Audio Preprocessing

Audio augmentation techniques:

- **Noise addition**: Gaussian noise for robustness
- **Time stretching**: 0.8x - 1.2x speed variations
- **Pitch shifting**: ±2 semitones
- **Volume changes**: 0.5x - 1.5x gain
- **SpecAugment**: Frequency and time masking
- **Background noise mixing**: SNR-based noise addition

## 📚 Knowledge Base

The `File kiến thức/` folder contains reference materials on:

- Deep learning architectures for fast convergence
- Modern Sound-to-Text technologies
- Contextual word embeddings
- Advanced ASR data management
- Training optimization for weak hardware
- Audio processing, NLP, and Vietnamese language handling

These documents inform the design decisions in this system.

## 🛠️ Development

### Adding New Features

1. **New Model Architecture**: Add to `models/` directory
2. **Custom Augmentation**: Extend `AudioAugmenter` class
3. **Different Loss Function**: Modify training script
4. **New Metrics**: Add to `utils/metrics.py`

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Add docstrings to functions and classes
- Keep functions focused and modular

## 🐛 Troubleshooting

### Common Issues

**Out of Memory (OOM)**:
- Reduce `batch_size` in config
- Enable `use_amp: true`
- Reduce model size parameters

**Slow Training**:
- Increase `num_workers` for data loading
- Enable mixed precision training
- Use GPU if available

**Database Locked**:
- Close other connections to database
- Check file permissions

**Audio Loading Errors**:
- Verify audio file formats (WAV recommended)
- Check file paths in CSV
- Ensure sample rates are supported

## 📄 License

[Specify your license here]

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📧 Contact

[Your contact information]

## 🙏 Acknowledgments

This system is built using modern ASR techniques and optimized for Vietnamese language processing. Special thanks to the research community for advances in transformer architectures and speech recognition.

---

**Note**: This is a research/educational implementation. For production use, consider additional optimizations, error handling, and security measures.
