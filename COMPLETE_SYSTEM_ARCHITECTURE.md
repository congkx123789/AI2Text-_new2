# Complete System Architecture

## 🚀 Complete Feature Set Implementation

Your ASR system now includes all features from the specification:

## 📊 Complete System Architecture

```
Vietnamese ASR System (Complete)

├── 🎯 Core ASR Models
│   ├── Basic ASR (LSTM + CTC)          ✅ models/lstm_asr.py
│   └── Enhanced ASR (Transformer + Contextual Embeddings)  ✅ models/enhanced_asr.py
│
├── 🧠 Embedding Systems
│   ├── Our Enhanced Embeddings:
│   │   ├── Subword tokenization (BPE) ✅ preprocessing/bpe_tokenizer.py
│   │   ├── Contextual embeddings (Transformer)  ✅ models/enhanced_asr.py
│   │   ├── Cross-modal attention       ✅ models/enhanced_asr.py
│   │   └── Word2Vec auxiliary training ✅ models/enhanced_asr.py
│   │
│   └── AI2text Embeddings Patch:
│       ├── Semantic Word2Vec (transcript-based)  ✅ nlp/word2vec_trainer.py
│       ├── Phon2Vec (phonetic/sound embeddings) ✅ nlp/phon2vec_trainer.py
│       ├── FAISS indexing              ✅ (from embeddings patch)
│       └── N-best rescoring            ✅ decoding/rescoring.py
│
├── 🔄 Training Pipeline
│   ├── Multi-task learning (CTC + Word2Vec)  ✅ models/enhanced_asr.py
│   ├── Mixed precision training              ✅ training/train.py
│   ├── Callbacks (Checkpoint, EarlyStop, Logging)  ✅ training/callbacks.py
│   └── Database experiment tracking          ✅ database/db_utils.py
│
└── 🎯 Evaluation & Inference
    ├── Beam search decoding           ✅ decoding/beam_search.py
    ├── N-best rescoring with embeddings ✅ decoding/rescoring.py
    ├── Contextual biasing              ✅ decoding/rescoring.py
    └── WER/CER metrics                 ✅ utils/metrics.py
```

## ✅ Feature Implementation Status

### Core ASR System ✅

1. **LSTM + CTC Model** (`models/lstm_asr.py`)
   - Bidirectional LSTM layers
   - Convolutional feature extraction
   - CTC decoder
   - Suitable for weak hardware

2. **Transformer + CTC Model** (`models/asr_base.py`)
   - Transformer encoder with attention
   - Convolutional subsampling
   - CTC decoder
   - Production-ready architecture

3. **Enhanced Transformer** (`models/enhanced_asr.py`)
   - Contextual word embeddings
   - Cross-modal attention (audio ↔ text)
   - Multi-task learning (CTC + Word2Vec)

### Embedding Systems ✅

#### Our Enhanced Embeddings:
1. **BPE Tokenization** (`preprocessing/bpe_tokenizer.py`)
   - Subword tokenization for Vietnamese
   - Better OOV handling
   - Trained on transcripts

2. **Contextual Embeddings** (`models/enhanced_asr.py`)
   - Transformer-based word embeddings
   - Context-aware representations
   - Used for cross-modal attention

3. **Cross-Modal Attention** (`models/enhanced_asr.py`)
   - Audio attends to text
   - Text attends to audio
   - Enables contextual biasing

4. **Word2Vec Auxiliary** (`models/enhanced_asr.py`)
   - Multi-task learning
   - Auxiliary Word2Vec task during training
   - Improves embeddings quality

#### AI2text Embeddings Patch:
1. **Semantic Word2Vec** (`nlp/word2vec_trainer.py`)
   - Trained on transcripts
   - Skip-gram architecture
   - Semantic similarity matching

2. **Phon2Vec** (`nlp/phon2vec_trainer.py`)
   - Trained on phonetic tokens (Telex+tone)
   - Sound-based similarity
   - Handles rare/OOV words

3. **Phonetic Processing** (`preprocessing/phonetic.py`)
   - Telex encoding
   - VnSoundex
   - Tone detection

4. **N-best Rescoring** (`decoding/rescoring.py`)
   - Semantic similarity scoring
   - Phonetic similarity scoring
   - Contextual biasing

### Training Pipeline ✅

1. **Multi-task Learning**
   - CTC loss (main task)
   - Word2Vec auxiliary loss (optional)
   - Shared encoder

2. **Training Optimizations**
   - Mixed precision (AMP)
   - Gradient clipping
   - OneCycleLR scheduler

3. **Callbacks** (`training/callbacks.py`)
   - CheckpointCallback
   - EarlyStoppingCallback
   - LoggingCallback
   - MetricsCallback

### Evaluation & Inference ✅

1. **Beam Search Decoding** (`decoding/beam_search.py`)
   - Multiple hypotheses exploration
   - Length penalty
   - Better than greedy decoding

2. **N-best Rescoring** (`decoding/rescoring.py`)
   - Combines acoustic + semantic + phonetic scores
   - Contextual biasing support
   - Handles rare words

3. **Contextual Biasing** (`decoding/rescoring.py`)
   - Bias towards specific words (names, domain terms)
   - Uses semantic and phonetic similarity
   - Dynamic OOV handling

## 🔄 Complete Data Flow

### Training Flow:
```
1. Data Layer:
   Raw Audio → Database → Split Assignment

2. Preprocessing Layer:
   Audio → Mel Spectrogram (80 dim)
   Text → BPE/Character Tokens → Token IDs

3. Model Layer:
   Features (80, time) → Encoder → Cross-Modal Attention ← Context Embeddings
                         ↓
                      CTC Decoder → Logits
                      Word2Vec Projection (optional)

4. Training Layer:
   Logits → CTC Loss
   Word2Vec → Auxiliary Loss (optional)
   Optimizer → AdamW
   Callbacks → Checkpoint, Logging, Metrics

5. Evaluation Layer:
   Model → Beam Search → N-best → Rescoring → Best Hypothesis
```

### Inference Flow:
```
1. Audio File → Preprocessing → Mel Spectrogram

2. Model → Beam Search Decoding → N-best Hypotheses

3. Rescoring:
   - Acoustic scores (from model)
   - Semantic similarity (Word2Vec)
   - Phonetic similarity (Phon2Vec)
   - Contextual biasing (if context provided)

4. Best Hypothesis → Text Output
```

## 🎯 Usage Examples

### 1. Train Basic LSTM Model:
```python
from models.lstm_asr import LSTMASRModel

model = LSTMASRModel(
    input_dim=80,
    vocab_size=1000,
    hidden_size=256,
    num_lstm_layers=3
)
```

### 2. Train Enhanced Transformer:
```python
from models.enhanced_asr import EnhancedASRModel

model = EnhancedASRModel(
    input_dim=80,
    vocab_size=1000,
    use_contextual_embeddings=True,
    use_cross_modal_attention=True,
    use_word2vec_auxiliary=True
)
```

### 3. Train Embeddings:
```bash
# Train Word2Vec and Phon2Vec
python scripts/build_embeddings.py --db database/asr_training.db --config configs/embeddings.yaml
```

### 4. Use BPE Tokenization:
```python
from preprocessing.bpe_tokenizer import BPETokenizer

tokenizer = BPETokenizer()
tokenizer.train(texts, vocab_size=1000)
token_ids = tokenizer.encode("xin chào")
```

### 5. Beam Search Decoding:
```python
from decoding.beam_search import BeamSearchDecoder

decoder = BeamSearchDecoder(
    vocab_size=1000,
    blank_token_id=2,
    beam_width=5
)
nbest = decoder.decode(logits, lengths)
```

### 6. N-best Rescoring:
```python
from decoding.rescoring import rescore_nbest
from gensim.models import KeyedVectors

# Load embeddings
semantic_kv = KeyedVectors.load("models/embeddings/word2vec.kv")
phon_kv = KeyedVectors.load("models/embeddings/phon2vec.kv")

# Rescore N-best
context = "đặt bánh gato sinh nhật"
rescored = rescore_nbest(
    nbest,
    semantic_kv=semantic_kv,
    phon_kv=phon_kv,
    context_text=context,
    gamma=0.5,
    delta=0.3
)
```

### 7. Contextual Biasing:
```python
from decoding.rescoring import contextual_biasing

bias_words = ["nguyễn", "hồ chí minh", "hà nội"]
biased = contextual_biasing(
    nbest,
    bias_list=bias_words,
    semantic_kv=semantic_kv,
    phon_kv=phon_kv,
    bias_weight=0.3
)
```

## 📁 Complete File Structure

```
AI2text/
├── models/
│   ├── asr_base.py           ✅ Base Transformer + CTC
│   ├── lstm_asr.py            ✅ LSTM + CTC (NEW)
│   ├── enhanced_asr.py        ✅ Enhanced with embeddings (NEW)
│   └── __init__.py
│
├── preprocessing/
│   ├── audio_processing.py    ✅ Audio features
│   ├── text_cleaning.py       ✅ Text normalization
│   ├── bpe_tokenizer.py       ✅ BPE tokenization (NEW)
│   ├── phonetic.py            ✅ Phonetic processing (NEW)
│   └── __init__.py
│
├── nlp/
│   ├── word2vec_trainer.py    ✅ Word2Vec training (NEW)
│   ├── phon2vec_trainer.py    ✅ Phon2Vec training (NEW)
│   └── __init__.py
│
├── decoding/
│   ├── beam_search.py         ✅ Beam search (NEW)
│   ├── rescoring.py           ✅ N-best rescoring (NEW)
│   └── __init__.py
│
├── training/
│   ├── dataset.py             ✅ Dataset
│   ├── train.py               ✅ Training with callbacks
│   ├── evaluate.py            ✅ Evaluation
│   ├── callbacks.py           ✅ Callbacks
│   └── __init__.py
│
├── scripts/
│   ├── prepare_data.py        ✅ Data preparation
│   ├── build_embeddings.py    ✅ Embeddings training (NEW)
│   └── validate_data.py       ✅ Validation
│
└── configs/
    ├── default.yaml           ✅ Training config
    ├── db.yaml                ✅ Database config
    └── embeddings.yaml        ✅ Embeddings config (NEW)
```

## ✨ Key Features

### 1. Multiple Model Architectures
- **LSTM**: Fast, lightweight, good for weak hardware
- **Transformer**: State-of-the-art, best accuracy
- **Enhanced**: With contextual embeddings and cross-modal attention

### 2. Multiple Tokenization Methods
- **Character-level**: Simple, handles all words
- **BPE**: Subword units, better for OOV words

### 3. Embeddings for Contextual Biasing
- **Semantic (Word2Vec)**: Meaning-based similarity
- **Phonetic (Phon2Vec)**: Sound-based similarity
- **Contextual**: Transformer-based, context-aware

### 4. Advanced Decoding
- **Greedy**: Fast, single best hypothesis
- **Beam Search**: Multiple hypotheses, better accuracy
- **N-best Rescoring**: Combines multiple signals

### 5. Contextual Biasing
- **Dynamic**: Based on context text
- **Static**: Based on bias word lists
- **Domain-specific**: Custom vocabularies

---

**Your system now has ALL features from the specification!** 🎉

