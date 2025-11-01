# AI2Text - How to Run the Project

Complete guide for running the Vietnamese Speech-to-Text system.

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Running the Project](#running-the-project)
6. [Testing](#testing)
7. [Usage Examples](#usage-examples)
8. [Project Structure](#project-structure)
9. [Troubleshooting](#troubleshooting)
10. [API Documentation](#api-documentation)

---

## 🚀 Quick Start

Get the project running in 5 minutes:

```bash
# 1. Install dependencies
pip install -r requirements/base.txt

# 2. Start the API server
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

# 3. Test the API (in another terminal)
curl http://localhost:8000/health
```

**That's it!** Your API server is now running at `http://localhost:8000`

---

## 📦 Prerequisites

### Required Software

- **Python**: 3.8 - 3.11 (3.10 recommended)
- **pip**: Latest version
- **Git**: For cloning the repository

### System Requirements

- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: 8GB minimum (16GB recommended)
- **Disk**: 10GB free space
- **OS**: Windows 10+, Linux, or macOS

### Optional (for GPU training)

- **CUDA**: 11.8+ (if using GPU)
- **cuDNN**: Compatible version
- **NVIDIA GPU**: With 6GB+ VRAM

---

## 🔧 Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd "AI2Text frist"
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install base requirements
pip install -r requirements/base.txt

# Install API dependencies (if using API)
pip install -r requirements/api.txt

# Install testing dependencies (optional)
pip install pytest pytest-cov pytest-xdist pytest-benchmark
```

### Step 4: Verify Installation

```bash
# Check project health
python scripts/check_project.py

# Test imports
python -c "from preprocessing.audio_processing import AudioProcessor; print('OK')"
python -c "from models.asr_base import ASRModel; print('OK')"
python -c "from database.db_utils import ASRDatabase; print('OK')"
```

### Step 5: Initialize Database

The database will be created automatically on first use, but you can initialize it manually:

```bash
python -c "from database.db_utils import ASRDatabase; db = ASRDatabase('database/asr_training.db'); print('Database initialized')"
```

---

## ⚙️ Configuration

### Configuration Files

Configuration files are located in `configs/`:

- `default.yaml` - Main configuration
- `db.yaml` - Database configuration
- `embeddings.yaml` - Embedding configuration

### Environment Variables

Create a `.env` file (optional):

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true

# Database Configuration
DB_PATH=database/asr_training.db

# Model Configuration
MODEL_PATH=checkpoints/best_model.pt
MODEL_TYPE=transformer

# Audio Processing
SAMPLE_RATE=16000
N_MELS=80
```

---

## 🏃 Running the Project

### Option 1: Run API Server (Recommended for Development)

**Using uvicorn:**
```bash
# Development mode (auto-reload)
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

# Production mode
uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

**Using Python:**
```bash
python api/app.py
```

**Using gunicorn (Production):**
```bash
gunicorn api.app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Option 2: Run Training Script

```bash
# Train a new model
python training/train.py --config configs/default.yaml

# Train with custom parameters
python training/train.py \
    --config configs/default.yaml \
    --num_epochs 10 \
    --batch_size 16 \
    --learning_rate 0.0001
```

### Option 3: Run Preprocessing

```bash
# Preprocess audio files
python scripts/prepare_data.py \
    --input_dir data/raw \
    --output_dir data/processed \
    --csv_path data/metadata.csv
```

### Option 4: Run Evaluation

```bash
# Evaluate a trained model
python training/evaluate.py \
    --model_path checkpoints/best_model.pt \
    --test_data data/test.csv
```

### Option 5: Run Microservices (Advanced)

For the full microservices architecture:

```bash
# Start all services with Docker Compose
docker compose -f infra/docker-compose.yml up -d

# Or start individual services
cd projects/services/asr
python app/main.py
```

---

## 🧪 Testing

### Run All Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=preprocessing --cov=models --cov-report=html

# Run only fast tests
python -m pytest tests/ -m "not slow" -v
```

### Run Specific Test Categories

```bash
# Unit tests
python -m pytest tests/unit/ -v

# Integration tests
python -m pytest tests/integration/ -v

# Advanced tests
python -m pytest tests/unit/*/test_*_advanced.py -v

# Performance tests
python -m pytest tests/performance/ --benchmark-only

# Security tests
python -m pytest tests/security/ -v
```

### Check Project Health

```bash
# Run comprehensive health check
python scripts/check_project.py

# List all tests
python scripts/list_tests.py
```

### Known Testing Issue

**Note**: On Windows, pytest may encounter a PyTorch DLL loading issue when running through `conftest.py`. This doesn't affect the actual code functionality. Solutions:

1. **Workaround**: Test modules directly
   ```bash
   python -c "from database.db_utils import ASRDatabase; import tempfile; f, p = tempfile.mkstemp(suffix='.db'); import os; os.close(f); db = ASRDatabase(p); print('OK'); os.unlink(p)"
   ```

2. **Alternative**: Use lazy-loading conftest
   ```bash
   # Copy conftest_lazy.py to conftest.py
   cp tests/conftest_lazy.py tests/conftest.py
   ```

3. **Best Solution**: Run tests on Linux/Mac (no DLL issues)

---

## 💡 Usage Examples

### Example 1: Transcribe Audio via API

**Using curl:**
```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "audio=@path/to/audio.wav" \
  -F "model_name=default" \
  -F "use_beam_search=true" \
  -F "beam_width=5"
```

**Using Python:**
```python
import requests

with open('audio.wav', 'rb') as f:
    files = {'audio': f}
    data = {
        'model_name': 'default',
        'use_beam_search': 'true',
        'beam_width': '5'
    }
    response = requests.post('http://localhost:8000/transcribe', files=files, data=data)
    print(response.json())
```

**Response:**
```json
{
  "text": "xin chào việt nam",
  "confidence": 0.95,
  "processing_time": 1.23
}
```

### Example 2: Train a Model

```python
from training.train import train_model

# Train with default config
train_model(config_path='configs/default.yaml')

# Train with custom parameters
train_model(
    config_path='configs/default.yaml',
    num_epochs=10,
    batch_size=16,
    learning_rate=0.0001
)
```

### Example 3: Preprocess Audio

```python
from preprocessing.audio_processing import AudioProcessor, preprocess_audio_file

# Initialize processor
processor = AudioProcessor(sample_rate=16000, n_mels=80)

# Process audio file
result = preprocess_audio_file(
    'audio.wav',
    processor=processor,
    extract_features=True
)

print(f"Duration: {result['duration']}s")
print(f"Features shape: {result['feature_shape']}")
```

### Example 4: Use Database

```python
from database.db_utils import ASRDatabase

# Initialize database
db = ASRDatabase('database/asr_training.db')

# Add audio file
audio_id = db.add_audio_file(
    file_path='audio.wav',
    filename='audio.wav',
    duration=5.0,
    sample_rate=16000
)

# Add transcript
transcript_id = db.add_transcript(
    audio_file_id=audio_id,
    transcript='xin chào việt nam',
    normalized_transcript='xin chao viet nam'
)

# Get statistics
stats = db.get_dataset_statistics('v1')
print(stats)
```

### Example 5: Use Models

```python
import torch
from models.asr_base import ASRModel

# Initialize model
model = ASRModel(
    input_dim=80,
    vocab_size=100,
    d_model=256,
    num_encoder_layers=6,
    num_heads=4
)

# Create dummy input
batch_size = 2
seq_len = 100
input_features = torch.randn(batch_size, seq_len, 80)
lengths = torch.tensor([seq_len, seq_len - 10])

# Forward pass
logits, output_lengths = model(input_features, lengths)
print(f"Logits shape: {logits.shape}")
```

---

## 📁 Project Structure

```
AI2Text frist/
├── api/                    # REST API server
│   ├── app.py             # Main FastAPI application
│   └── public_api.py      # Public API endpoints
│
├── preprocessing/          # Audio and text preprocessing
│   ├── audio_processing.py
│   ├── text_cleaning.py
│   └── ...
│
├── models/                # Model architectures
│   ├── asr_base.py       # Base transformer model
│   ├── enhanced_asr.py   # Enhanced model
│   └── lstm_asr.py       # LSTM model
│
├── decoding/              # Decoding algorithms
│   ├── beam_search.py
│   ├── lm_decoder.py
│   └── ...
│
├── training/              # Training pipeline
│   ├── train.py          # Training script
│   ├── evaluate.py       # Evaluation script
│   └── dataset.py        # Dataset handling
│
├── database/              # Database management
│   ├── db_utils.py       # Database utilities
│   └── init_db.sql       # Database schema
│
├── database/              # NLP components
│   ├── word2vec_trainer.py
│   ├── faiss_index.py
│   └── ...
│
├── tests/                 # Test suite
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   ├── performance/      # Performance tests
│   └── security/         # Security tests
│
├── configs/               # Configuration files
│   ├── default.yaml
│   └── db.yaml
│
├── scripts/               # Utility scripts
│   ├── check_project.py  # Health check
│   ├── list_tests.py     # List tests
│   └── ...
│
├── requirements/          # Dependencies
│   ├── base.txt
│   ├── api.txt
│   └── microservices.txt
│
└── data/                  # Data storage
    ├── raw/              # Raw audio files
    ├── processed/       # Processed data
    └── external/         # External datasets
```

---

## 🔍 API Documentation

### Endpoints

Once the server is running, visit:
- **API Docs**: http://localhost:8000/docs (Swagger UI)
- **Alternative Docs**: http://localhost:8000/redoc (ReDoc)
- **Health Check**: http://localhost:8000/health

### Main Endpoints

#### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": 1,
  "tokenizer_ready": true,
  "processor_ready": true
}
```

#### 2. Transcribe Audio
```http
POST /transcribe
Content-Type: multipart/form-data
```

**Parameters:**
- `audio` (file, required): Audio file (WAV, MP3, FLAC)
- `model_name` (string, optional): Model name (default: "default")
- `use_beam_search` (boolean, optional): Use beam search (default: true)
- `beam_width` (integer, optional): Beam width (default: 5)
- `use_lm` (boolean, optional): Use language model (default: false)
- `min_confidence` (float, optional): Minimum confidence threshold

**Response:**
```json
{
  "text": "transcribed text",
  "confidence": 0.95,
  "processing_time": 1.23
}
```

#### 3. List Models
```http
GET /models
```

**Response:**
```json
{
  "models": [
    {"name": "default"},
    {"name": "best_model"}
  ]
}
```

#### 4. Load Model
```http
POST /models/load
Content-Type: application/json
```

**Body:**
```json
{
  "model_path": "checkpoints/model.pt",
  "model_name": "my_model"
}
```

---

## 🐛 Troubleshooting

### Common Issues

#### Issue 1: PyTorch DLL Loading Error (Windows)

**Error**: `OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed`

**Solution**:
```bash
# Reinstall PyTorch
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Issue 2: Module Import Errors

**Error**: `ModuleNotFoundError: No module named 'xxx'`

**Solution**:
```bash
# Install missing dependencies
pip install -r requirements/base.txt

# Or install specific module
pip install gensim  # Example
```

#### Issue 3: Database Connection Error

**Error**: `sqlite3.OperationalError: unable to open database file`

**Solution**:
```bash
# Create database directory
mkdir -p database

# Or run database initialization
python -c "from database.db_utils import ASRDatabase; db = ASRDatabase('database/asr_training.db')"
```

#### Issue 4: Port Already in Use

**Error**: `Address already in use`

**Solution**:
```bash
# Use a different port
uvicorn api.app:app --port 8001

# Or kill the process using port 8000
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/macOS:
lsof -ti:8000 | xargs kill -9
```

#### Issue 5: Out of Memory During Training

**Error**: `RuntimeError: CUDA out of memory`

**Solution**:
- Reduce batch size in config
- Use gradient accumulation
- Use mixed precision training
- Use smaller model

### Getting Help

1. **Check Project Health**:
   ```bash
   python scripts/check_project.py
   ```

2. **Check Test Status**:
   ```bash
   python scripts/list_tests.py
   ```

3. **View Logs**:
   - API logs: Check console output when running uvicorn
   - Training logs: Check `logs/` directory

4. **Common Commands**:
   ```bash
   # Verify installation
   python -c "import torch; print(torch.__version__)"
   python -c "import fastapi; print(fastapi.__version__)"

   # Check database
   python -c "from database.db_utils import ASRDatabase; print('OK')"
   ```

---

## 📚 Additional Resources

### Documentation Files

- **Testing Guide**: `tests/README_TESTS.md`
- **Testing Roadmap**: `TESTING_ROADMAP.md`
- **Advanced Tests**: `tests/README_ADVANCED_TESTS.md`
- **Project Health**: `PROJECT_HEALTH_REPORT.md`
- **Run Guide**: `RUN_TESTS.md`

### Quick Reference

- **Health Check**: `python scripts/check_project.py`
- **List Tests**: `python scripts/list_tests.py`
- **Run Tests**: `python -m pytest tests/ -v`
- **Start API**: `uvicorn api.app:app --reload`

---

## 🎯 Next Steps

1. **Train Your First Model**:
   ```bash
   python training/train.py --config configs/default.yaml
   ```

2. **Test the API**:
   ```bash
   # Start server
   uvicorn api.app:app --reload

   # In another terminal, test health
   curl http://localhost:8000/health
   ```

3. **Process Your Data**:
   ```bash
   python scripts/prepare_data.py --input_dir data/raw --output_dir data/processed
   ```

4. **Run Tests**:
   ```bash
   python -m pytest tests/ -v
   ```

---

## 📝 License

[Your License Here]

---

## 🤝 Contributing

[Your Contributing Guidelines Here]

---

## 📧 Support

For issues and questions:
1. Check `PROJECT_HEALTH_REPORT.md` for project status
2. Run `python scripts/check_project.py` for diagnostics
3. Check `TROUBLESHOOTING.md` for common issues
4. Open an issue on GitHub

---

**Last Updated**: 2024  
**Version**: 1.0.0  
**Status**: ✅ Production Ready

