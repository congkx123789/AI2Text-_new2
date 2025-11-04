# Training Guide - AI2Text ASR Model

Complete guide for training your Vietnamese Speech-to-Text model.

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Data Preparation](#data-preparation)
3. [Configuration](#configuration)
4. [Training Process](#training-process)
5. [Monitoring Training](#monitoring-training)
6. [Evaluation](#evaluation)
7. [Using Trained Models](#using-trained-models)
8. [Advanced Training](#advanced-training)
9. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

Train a model in 3 steps:

```bash
# 1. Prepare your data
python scripts/prepare_data.py --csv_path data/metadata.csv

# 2. Start training
python training/train.py --config configs/default.yaml

# 3. Evaluate
python training/evaluate.py --model_path checkpoints/best_model.pt
```

---

## 📊 Data Preparation

### Step 1: Organize Your Audio Files

Create the following structure:

```
data/
├── raw/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
└── metadata.csv
```

### Step 2: Create Metadata CSV

Your `metadata.csv` should have these columns:

```csv
file_path,transcript,duration,split,speaker_id
data/raw/audio1.wav,xin chào việt nam,5.2,train,speaker_01
data/raw/audio2.wav,tôi là sinh viên,4.8,train,speaker_02
data/raw/audio3.wav,hôm nay trời đẹp,3.5,val,speaker_01
```

**Required columns**:
- `file_path`: Path to audio file (WAV format recommended)
- `transcript`: Text transcription in Vietnamese
- `duration`: Audio duration in seconds (optional, will be calculated if missing)
- `split`: Data split (`train`, `val`, or `test`)
- `speaker_id`: Speaker identifier (optional but recommended)

### Step 3: Prepare Data with Script

```bash
# Validate and prepare data
python scripts/prepare_data.py \
    --csv_path data/metadata.csv \
    --output_dir data/processed \
    --validate

# This will:
# - Validate audio files exist
# - Check audio format and quality
# - Normalize text
# - Split data properly
# - Store in database
```

### Step 4: Verify Data in Database

```bash
# Check database statistics
python -c "
from database.db_utils import ASRDatabase
db = ASRDatabase('database/asr_training.db')
stats = db.get_dataset_statistics('v1')
print('Dataset Statistics:')
print(f\"Train: {stats.get('train_files', 0)} files\")
print(f\"Val: {stats.get('val_files', 0)} files\")
print(f\"Test: {stats.get('test_files', 0)} files\")
"
```

### Data Requirements

**Minimum**:
- 100 hours of audio for basic model
- 1000+ hours for production quality

**Recommended**:
- Balanced across speakers
- Various recording conditions
- Clean transcriptions
- Audio: 16kHz, mono, WAV format

---

## ⚙️ Configuration

### Basic Configuration

Edit `configs/default.yaml`:

```yaml
# Model architecture
d_model: 256              # Model dimension (128-512)
num_encoder_layers: 6     # Transformer layers (4-12)
num_heads: 4              # Attention heads (4-8)
d_ff: 1024               # Feedforward dimension
dropout: 0.1             # Dropout rate (0.1-0.3)

# Training hyperparameters
batch_size: 16           # Batch size (adjust for your GPU/CPU)
num_epochs: 50           # Number of epochs
learning_rate: 0.0001    # Learning rate (0.00001-0.001)
weight_decay: 0.01       # Weight decay for regularization
grad_clip: 1.0           # Gradient clipping

# Audio processing
sample_rate: 16000       # Audio sample rate
n_mels: 80              # Number of mel filterbanks
n_fft: 400              # FFT window size
hop_length: 160         # Hop length between frames

# Optimization (for weak hardware)
use_amp: true           # Use mixed precision (FP16)
num_workers: 4          # Data loading workers
pin_memory: true        # Pin memory for faster GPU transfer

# Paths
database_path: "database/asr_training.db"
checkpoint_dir: "checkpoints"
log_file: "logs/training.log"
```

### Configuration for Different Hardware

**Weak Hardware (4GB RAM, No GPU)**:
```yaml
batch_size: 4
num_workers: 2
use_amp: false
grad_accum_steps: 4  # Accumulate gradients
d_model: 128
num_encoder_layers: 4
```

**Medium Hardware (8GB RAM, GTX 1060)**:
```yaml
batch_size: 16
num_workers: 4
use_amp: true
d_model: 256
num_encoder_layers: 6
```

**Strong Hardware (16GB+ RAM, RTX 3090)**:
```yaml
batch_size: 32
num_workers: 8
use_amp: true
d_model: 512
num_encoder_layers: 12
```

---

## 🏃 Training Process

### Option 1: Basic Training

```bash
# Train with default config
python training/train.py --config configs/default.yaml
```

### Option 2: Custom Parameters

```bash
# Override config parameters
python training/train.py \
    --config configs/default.yaml \
    --num_epochs 20 \
    --batch_size 32 \
    --learning_rate 0.0005 \
    --checkpoint_dir checkpoints/experiment1
```

### Option 3: Resume Training

```bash
# Resume from checkpoint
python training/train.py \
    --config configs/default.yaml \
    --resume checkpoints/last_checkpoint.pt
```

### Option 4: Fine-tuning

```bash
# Fine-tune a pre-trained model
python training/train.py \
    --config configs/default.yaml \
    --pretrained_model checkpoints/base_model.pt \
    --learning_rate 0.00001  # Lower learning rate for fine-tuning
```

### Training Script Explained

The training script performs these steps:

1. **Load Configuration**: Read YAML config
2. **Initialize Database**: Load training data from database
3. **Setup Data Loaders**: Create train/val data loaders
4. **Initialize Model**: Create ASR model architecture
5. **Setup Optimizer**: AdamW optimizer with learning rate scheduling
6. **Setup Callbacks**: Checkpointing, early stopping, logging
7. **Training Loop**: 
   - Forward pass
   - Compute CTC loss
   - Backward pass
   - Update weights
   - Log metrics
8. **Validation**: Evaluate on validation set after each epoch
9. **Save Checkpoint**: Save best model based on validation loss

---

## 📈 Monitoring Training

### Real-time Monitoring

Training progress is displayed in the terminal:

```
Epoch 1/50
Train: 100%|██████████| 500/500 [05:23<00:00, 1.55it/s, loss=2.45]
Val:   100%|██████████| 50/50 [00:32<00:00, 1.56it/s, loss=2.12]

Epoch 1 Summary:
├── Train Loss: 2.45
├── Val Loss: 2.12
├── Val WER: 45.3%
├── Val CER: 23.1%
├── Learning Rate: 0.0001
└── Time: 5m 55s

Saved best model to: checkpoints/best_model.pt
```

### TensorBoard (Optional)

If you have TensorBoard installed:

```bash
# Start TensorBoard
tensorboard --logdir runs/

# View in browser
# http://localhost:6006
```

### WandB (Optional)

For cloud-based monitoring:

```bash
# Install wandb
pip install wandb

# Login
wandb login

# Add to config
wandb:
  enabled: true
  project: "vietnamese-asr"
  entity: "your-username"
```

### Check Training Logs

```bash
# View training logs
tail -f logs/training.log

# Or check specific run
cat logs/training.log | grep "Epoch"
```

### Monitor Checkpoints

```bash
# List saved checkpoints
ls -lh checkpoints/

# Expected files:
# - best_model.pt         (best validation loss)
# - last_checkpoint.pt    (most recent)
# - epoch_5.pt            (periodic saves)
```

---

## 🎯 Evaluation

### Evaluate Trained Model

```bash
# Evaluate on test set
python training/evaluate.py \
    --model_path checkpoints/best_model.pt \
    --test_data database/asr_training.db \
    --split test
```

### Evaluation Output

```
Evaluation Results:
==================
Test Set: 500 samples
Duration: 25.3 hours

Metrics:
├── Word Error Rate (WER): 12.5%
├── Character Error Rate (CER): 6.2%
├── Accuracy: 87.5%
└── Processing Speed: 0.15x realtime

Error Analysis:
├── Substitutions: 45%
├── Deletions: 30%
├── Insertions: 25%

Top Errors:
1. "việt nam" → "viêt nam" (15 occurrences)
2. "xin chào" → "sin chào" (12 occurrences)
```

### Transcribe Single File

```bash
# Transcribe a single audio file
python training/evaluate.py \
    --model_path checkpoints/best_model.pt \
    --audio_file test_audio.wav \
    --use_beam_search \
    --beam_width 10
```

### Batch Transcription

```python
# Python script for batch transcription
from training.evaluate import ASREvaluator

evaluator = ASREvaluator('checkpoints/best_model.pt')

audio_files = ['audio1.wav', 'audio2.wav', 'audio3.wav']
results = evaluator.transcribe_batch(audio_files)

for file, result in zip(audio_files, results):
    print(f"{file}: {result['text']}")
    print(f"  Confidence: {result['confidence']:.2f}")
```

---

## 🎓 Using Trained Models

### Load Model in Python

```python
import torch
from models.asr_base import ASRModel
from preprocessing.audio_processing import AudioProcessor
from preprocessing.text_cleaning import Tokenizer

# Load model
checkpoint = torch.load('checkpoints/best_model.pt')
model = ASRModel(
    input_dim=80,
    vocab_size=checkpoint['vocab_size'],
    d_model=checkpoint['config']['d_model']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Process audio
processor = AudioProcessor(sample_rate=16000)
audio, sr = processor.load_audio('test.wav')
features = processor.extract_mel_spectrogram(audio)

# Get prediction
with torch.no_grad():
    logits, lengths = model(features.unsqueeze(0))
    # Decode logits to text...
```

### Use with API

```bash
# Start API server with trained model
uvicorn api.app:app --reload

# The API will automatically load models from checkpoints/
```

### Export Model

```python
# Export model for production
from training.evaluate import export_model

export_model(
    model_path='checkpoints/best_model.pt',
    output_path='models/production_model.pt',
    optimize=True,
    quantize=False  # Set True for smaller model
)
```

---

## 🔬 Advanced Training

### Mixed Precision Training

Enable mixed precision for faster training:

```yaml
# In config
use_amp: true
```

Or in code:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop
with autocast():
    logits, lengths = model(features)
    loss = criterion(logits, targets, lengths, target_lengths)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Gradient Accumulation

For larger effective batch size:

```yaml
# In config
batch_size: 8
grad_accum_steps: 4  # Effective batch size: 32
```

### Learning Rate Scheduling

```python
from torch.optim.lr_scheduler import OneCycleLR

scheduler = OneCycleLR(
    optimizer,
    max_lr=0.001,
    epochs=num_epochs,
    steps_per_epoch=len(train_loader)
)

# In training loop
for batch in train_loader:
    # ... training code ...
    scheduler.step()
```

### Data Augmentation

Enable during training:

```yaml
# In config
augmentation:
  enabled: true
  types: ['noise', 'volume', 'shift', 'stretch']
  prob: 0.5  # Apply to 50% of samples
```

### Multi-GPU Training

```bash
# Use DataParallel
python training/train.py \
    --config configs/default.yaml \
    --multi_gpu

# Or use DistributedDataParallel
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    training/train.py --config configs/default.yaml
```

### Hyperparameter Tuning

```python
# Use Ray Tune or Optuna
import optuna

def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    
    # Train model with these hyperparameters
    metrics = train_model(lr=lr, batch_size=batch_size)
    return metrics['val_loss']

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print(f"Best hyperparameters: {study.best_params}")
```

---

## 🐛 Troubleshooting

### Issue 1: Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
```yaml
# Reduce batch size
batch_size: 4

# Enable gradient accumulation
grad_accum_steps: 4

# Use mixed precision
use_amp: true

# Reduce model size
d_model: 128
num_encoder_layers: 4
```

### Issue 2: Slow Training

**Problem**: Training is very slow

**Solutions**:
```yaml
# Increase workers
num_workers: 8

# Enable pinned memory
pin_memory: true

# Use GPU if available
device: "cuda"

# Reduce validation frequency
validate_every: 5  # Validate every 5 epochs
```

### Issue 3: Loss Not Decreasing

**Problem**: Loss stays high or doesn't improve

**Solutions**:
1. **Check data**: Verify audio and transcripts match
2. **Reduce learning rate**:
   ```yaml
   learning_rate: 0.00005
   ```
3. **Check for NaN**: Add gradient clipping
   ```yaml
   grad_clip: 1.0
   ```
4. **Increase model capacity**:
   ```yaml
   d_model: 512
   num_encoder_layers: 8
   ```

### Issue 4: Overfitting

**Problem**: Train loss decreases but validation loss increases

**Solutions**:
```yaml
# Increase dropout
dropout: 0.3

# Add weight decay
weight_decay: 0.01

# Enable augmentation
augmentation:
  enabled: true

# Early stopping
early_stopping:
  enabled: true
  patience: 10
```

### Issue 5: Convergence Issues

**Problem**: Training is unstable

**Solutions**:
1. **Lower learning rate**:
   ```yaml
   learning_rate: 0.00005
   ```
2. **Warm up**:
   ```yaml
   warmup_epochs: 3
   ```
3. **Gradient clipping**:
   ```yaml
   grad_clip: 0.5
   ```

---

## 📊 Training Checklist

Before starting training:

- [ ] Data prepared and validated
- [ ] Metadata CSV created with correct format
- [ ] Database initialized and populated
- [ ] Configuration file reviewed
- [ ] Output directories created (checkpoints/, logs/)
- [ ] GPU/CUDA available (optional but recommended)
- [ ] Enough disk space for checkpoints (~1GB per checkpoint)
- [ ] Monitoring set up (TensorBoard/WandB)

During training:

- [ ] Monitor loss curves (should decrease)
- [ ] Check validation metrics (WER/CER)
- [ ] Watch for overfitting
- [ ] Save best checkpoints
- [ ] Log hyperparameters

After training:

- [ ] Evaluate on test set
- [ ] Analyze errors
- [ ] Compare with baseline
- [ ] Export production model
- [ ] Document results

---

## 📚 Additional Resources

### Example Training Commands

```bash
# Quick experiment (small model, few epochs)
python training/train.py \
    --config configs/default.yaml \
    --num_epochs 5 \
    --batch_size 8 \
    --d_model 128

# Full training run
python training/train.py \
    --config configs/default.yaml \
    --num_epochs 50 \
    --batch_size 32

# Fine-tune existing model
python training/train.py \
    --config configs/default.yaml \
    --pretrained_model checkpoints/base_model.pt \
    --learning_rate 0.00001 \
    --num_epochs 10
```

### Monitoring Scripts

```bash
# Watch training progress
watch -n 1 'tail -20 logs/training.log'

# Check GPU usage
watch -n 1 nvidia-smi

# Monitor disk space
df -h
```

### Related Documentation

- `TESTING_ROADMAP.md` - Testing strategy
- `README_RUN_GUIDE.md` - Project setup
- `PROJECT_HEALTH_REPORT.md` - Project status
- `configs/default.yaml` - Configuration reference

---

**Last Updated**: 2024  
**Version**: 1.0.0  
**Status**: Ready for Training 🚀

