# System Architecture

## 📐 Layered Architecture Overview

The ASR training system follows a clean layered architecture pattern, with each layer having a specific responsibility:

```
┌─────────────────────────────────────────────────────────────┐
│                       Data Layer                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────────┐  │
│  │   Raw    │  │Processed │  │      SQLite Database     │  │
│  │  Audio   │─▶│  Audio   │  │  (Metadata & Metrics)    │  │
│  └──────────┘  └──────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Preprocessing Layer                        │
│  ┌─────────────────────┐  ┌───────────────────────────┐    │
│  │  Audio Processing   │  │  Text Normalization       │    │
│  │  - Feature Extract  │  │  - Vietnamese Specific    │    │
│  │  - Augmentation     │  │  - Tokenization           │    │
│  └─────────────────────┘  └───────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Model Layer                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           Transformer Encoder + CTC Decoder          │   │
│  │                                                       │   │
│  │  Input → Conv Subsample → Transformer Layers → CTC  │   │
│  │   (80)      (d_model)      (d_model)         (vocab)│   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Training Layer                           │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │   Trainer    │  │  Optimizer   │  │   Callbacks     │   │
│  │   - Loop     │  │  - AdamW     │  │   - Checkpt     │   │
│  │   - AMP      │  │  - Scheduler │  │   - Logging     │   │
│  └──────────────┘  └──────────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Evaluation Layer                           │
│  ┌─────────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   Inference     │  │   Metrics    │  │   Analysis   │   │
│  │   - Greedy CTC  │  │   - WER/CER  │  │   - Results  │   │
│  └─────────────────┘  └──────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 🔷 Layer Details

### 1. Data Layer

**Purpose**: Manage raw audio data, processed features, and metadata storage.

**Components**:
- **Raw Audio** (`data/raw/`): Original audio files (WAV, MP3, FLAC, OGG)
- **Processed Audio** (`data/processed/`): Preprocessed features (mel spectrograms)
- **SQLite Database** (`database/`): Stores metadata, transcripts, splits, training runs

**Key Modules**:
- `database/db_utils.py`: Database operations
- `database/init_db.sql`: Database schema

**Data Flow**:
```
Raw Audio → Import to Database → Assign Splits → Ready for Training
```

### 2. Preprocessing Layer

**Purpose**: Transform raw audio and text into model-ready features.

**Components**:

#### Audio Processing (`preprocessing/audio_processing.py`):
- **Feature Extraction**: 
  - Mel spectrograms (80 frequency bins)
  - MFCC features (optional)
  - Energy normalization
- **Augmentation**:
  - Noise addition
  - Time shifting
  - Time stretching
  - Pitch shifting
  - Volume changes
  - SpecAugment

#### Text Normalization (`preprocessing/text_cleaning.py`):
- **Vietnamese Text Normalization**:
  - Unicode normalization
  - Lowercase conversion
  - Number-to-word conversion
  - Abbreviation expansion
  - Filler word removal
- **Tokenization**:
  - Character-level tokenization
  - Vietnamese character support
  - Special tokens (PAD, BLANK)

**Data Flow**:
```
Audio → Mel Spectrogram (80, time) → Normalized Features
Text → Normalized Text → Token IDs → Ready for Model
```

### 3. Model Layer

**Purpose**: Deep learning architecture for speech recognition.

**Architecture** (`models/asr_base.py`):

```
Input: (batch, time, 80)  [Mel spectrogram features]
    ↓
ConvSubsampling: 2D convolutions with stride 2
    ↓ (reduces time dimension by 4x)
Linear Projection: (subsampled_dim → d_model)
    ↓
Positional Encoding: Add position information
    ↓
Transformer Encoder Layers (N layers):
    ├─ Multi-Head Self-Attention
    ├─ Feed-Forward Network
    ├─ Layer Normalization
    └─ Residual Connections
    ↓
Output: (batch, time/4, d_model)
    ↓
CTC Decoder: Linear projection
    ↓
Output: (batch, time/4, vocab_size)  [Logits]
```

**Components**:
- `ConvSubsampling`: Reduces sequence length (4x reduction)
- `PositionalEncoding`: Adds position information
- `MultiHeadAttention`: Self-attention mechanism
- `FeedForward`: Position-wise feed-forward network
- `EncoderLayer`: Single transformer encoder layer
- `ASREncoder`: Complete encoder stack
- `ASRDecoder`: CTC decoder (linear projection)
- `ASRModel`: Complete model (encoder + decoder)

**Key Features**:
- Configurable depth (num_encoder_layers)
- Configurable width (d_model, d_ff)
- Dropout for regularization
- CTC loss for alignment-free training

### 4. Training Layer

**Purpose**: Orchestrate the training process with optimization and callbacks.

**Components** (`training/train.py`, `training/callbacks.py`):

#### Trainer (`ASRTrainer`):
- **Training Loop**: Iterates through epochs and batches
- **Mixed Precision (AMP)**: Automatic mixed precision for efficiency
- **Gradient Clipping**: Prevents exploding gradients
- **State Management**: Tracks training progress

#### Optimizer:
- **AdamW**: Adaptive learning rate optimizer
- **Weight Decay**: L2 regularization
- **OneCycleLR**: Learning rate scheduler (cosine annealing)

#### Callbacks (`training/callbacks.py`):
- **CheckpointCallback**: Saves model checkpoints
  - Best model (based on validation loss)
  - Periodic checkpoints
- **EarlyStoppingCallback**: Stops training if no improvement
- **LoggingCallback**: Logs training progress
- **MetricsCallback**: Tracks and logs metrics
- **CallbackManager**: Coordinates all callbacks

**Training Flow**:
```
For each epoch:
    For each batch:
        1. Load audio features and transcripts
        2. Forward pass (model)
        3. Compute CTC loss
        4. Backward pass (with AMP if enabled)
        5. Gradient clipping
        6. Optimizer step
        7. Scheduler step
    Validate on validation set
    Calculate WER/CER
    Call callbacks (checkpoint, logging, metrics)
    Check early stopping
```

### 5. Evaluation Layer

**Purpose**: Evaluate model performance and generate transcriptions.

**Components** (`training/evaluate.py`, `utils/metrics.py`):

#### Inference (`ASREvaluator`):
- **CTC Decoding**: Greedy decoding (argmax)
  - Remove consecutive duplicates
  - Remove blank tokens
  - Decode to text
- **Batch Inference**: Process multiple files efficiently

#### Metrics:
- **WER (Word Error Rate)**: `(Substitutions + Insertions + Deletions) / Total Words`
- **CER (Character Error Rate)**: `(Substitutions + Insertions + Deletions) / Total Characters`
- **Accuracy**: Sentence-level exact match rate
- **Levenshtein Distance**: Edit distance calculation

#### Analysis:
- Detailed comparison (prediction vs reference)
- Error analysis
- Metrics export (CSV, text)

## 🔄 Data Flow Through System

### Training Flow:
```
1. Data Layer:
   Raw Audio → Database Import → Split Assignment

2. Preprocessing Layer:
   Database Query → Audio Loading → Mel Spectrogram → Text Normalization

3. Model Layer:
   Features (80, time) → Encoder → Decoder → Logits (vocab_size)

4. Training Layer:
   Logits → CTC Loss → Backward Pass → Optimizer Update → Metrics

5. Evaluation Layer:
   Model → Inference → Decode → Calculate WER/CER → Results
```

### Inference Flow:
```
1. Audio File → Preprocessing (Mel Spectrogram)
2. Model Forward Pass (Encoder + Decoder)
3. CTC Decoding (Greedy)
4. Text Output
```

## 📦 Module Organization

### By Layer:
```
data/              → Data Layer
database/          → Data Layer (metadata)
preprocessing/     → Preprocessing Layer
models/            → Model Layer
training/          → Training Layer & Evaluation Layer
utils/             → Utilities (metrics, logging)
configs/           → Configuration (all layers)
scripts/           → Scripts (data preparation, validation)
```

### Key Files:
- `database/db_utils.py`: Data layer operations
- `preprocessing/audio_processing.py`: Audio preprocessing
- `preprocessing/text_cleaning.py`: Text preprocessing
- `models/asr_base.py`: Model architecture
- `training/train.py`: Training logic
- `training/callbacks.py`: Training callbacks
- `training/evaluate.py`: Evaluation logic
- `utils/metrics.py`: Metrics calculation

## 🎯 Design Principles

1. **Separation of Concerns**: Each layer handles one responsibility
2. **Modularity**: Components can be swapped or extended independently
3. **Extensibility**: Easy to add new features (augmentations, metrics, callbacks)
4. **Resource Efficiency**: Optimized for weak hardware (AMP, gradient clipping)
5. **Vietnamese-First**: Specialized for Vietnamese language processing

## 🔧 Configuration

All layers are configurable via `configs/default.yaml`:
- Model architecture (dimensions, depth)
- Training hyperparameters (learning rate, batch size)
- Preprocessing settings (sample rate, mel bins)
- Optimization flags (AMP, gradient clipping)

---

**This architecture ensures clean separation, easy maintenance, and scalability!** 🎉

