# Important Functions to Test

This document lists all critical functions that require comprehensive testing in the AI2Text project.

## Preprocessing Module (`preprocessing/`)

### `audio_processing.py`
**Priority: CRITICAL**

1. **AudioProcessor.load_audio()**
   - Audio loading and resampling
   - Format conversion (stereo to mono)
   - Normalization

2. **AudioProcessor.extract_mel_spectrogram()**
   - Mel spectrogram feature extraction
   - Shape validation
   - Numerical stability

3. **AudioProcessor.extract_mfcc()**
   - MFCC feature extraction
   - Parameter validation

4. **AudioProcessor.trim_silence()**
   - Silence detection and removal
   - Edge cases (all silence, no silence)

5. **AudioProcessor.pad_or_truncate()**
   - Padding logic
   - Truncation logic
   - Edge cases (exact length)

6. **AudioAugmenter.add_noise()**
   - Gaussian noise addition
   - Signal-to-noise ratio

7. **AudioAugmenter.time_shift()**
   - Time shifting
   - Energy preservation

8. **AudioAugmenter.spec_augment()**
   - Frequency masking
   - Time masking
   - Mask parameter validation

9. **preprocess_audio_file()**
   - Complete preprocessing pipeline
   - Feature extraction
   - File I/O

### `text_cleaning.py`
**Priority: CRITICAL**

1. **VietnameseTextNormalizer.normalize()**
   - Complete normalization pipeline
   - Unicode normalization
   - Number conversion
   - Abbreviation expansion

2. **VietnameseTextNormalizer.normalize_unicode_text()**
   - Unicode NFC normalization
   - Special character handling

3. **VietnameseTextNormalizer.convert_numbers_to_words()**
   - Digit to Vietnamese word conversion
   - Multi-digit numbers

4. **VietnameseTextNormalizer.expand_abbreviations()**
   - Vietnamese abbreviation expansion
   - Case sensitivity

5. **Tokenizer.encode() / decode()**
   - Text to token ID conversion
   - Token ID to text conversion
   - Roundtrip consistency
   - Special token handling

### `bpe_tokenizer.py`
**Priority: HIGH**

1. **BPETokenizer.train()**
   - BPE training algorithm
   - Vocabulary building

2. **BPETokenizer.encode() / decode()**
   - BPE encoding/decoding
   - Subword tokenization

### `phonetic.py`
**Priority: MEDIUM**

1. **PhoneticFeatureExtractor.extract()**
   - Phonetic feature extraction
   - Vietnamese phoneme handling

---

## Models Module (`models/`)

### `asr_base.py`
**Priority: CRITICAL**

1. **ASREncoder.forward()**
   - Encoder forward pass
   - Attention mechanism
   - Positional encoding
   - Output shape validation

2. **ASRDecoder.forward()**
   - Decoder forward pass
   - Logit computation
   - Vocabulary projection

3. **ASRModel.forward()**
   - Complete model forward pass
   - Sequence length handling
   - Batch processing
   - Gradient flow

4. **ASRModel.get_num_params()**
   - Parameter counting
   - Trainable vs. total parameters

### `enhanced_asr.py`
**Priority: CRITICAL**

1. **ContextualEmbedding.forward()**
   - Contextual embedding generation
   - Transformer encoding

2. **CrossModalAttention.forward()**
   - Cross-modal attention mechanism
   - Audio-text alignment

3. **EnhancedASRModel.forward()**
   - Enhanced forward pass
   - Contextual biasing
   - Optional features (context, Word2Vec)

### `lstm_asr.py`
**Priority: HIGH**

1. **LSTMASRModel.forward()**
   - LSTM encoder forward pass
   - Bidirectional LSTM
   - CTC decoding

---

## Decoding Module (`decoding/`)

### `beam_search.py`
**Priority: HIGH**

1. **BeamSearchDecoder.decode()**
   - Beam search algorithm
   - CTC path merging
   - Hypothesis generation
   - Beam width effects

2. **BeamSearchDecoder.decode_batch()**
   - Batch processing
   - Variable length sequences

### `lm_decoder.py`
**Priority: HIGH**

1. **LMBeamSearchDecoder.decode()**
   - Language model integration
   - KenLM integration
   - Score combination
   - Fallback to basic decoder

2. **LMBeamSearchDecoder.decode_batch()**
   - Batch LM decoding
   - Performance with LM

### `confidence.py`
**Priority: MEDIUM**

1. **ConfidenceScorer.compute()**
   - Confidence score calculation
   - Probability computation
   - Uncertainty estimation

### `rescoring.py`
**Priority: MEDIUM**

1. **Rescorer.rescore()**
   - N-best list rescoring
   - Embedding-based rescoring
   - Rank reordering

---

## Database Module (`database/`)

### `db_utils.py`
**Priority: HIGH**

1. **ASRDatabase.add_audio_file()**
   - Audio file insertion
   - Metadata handling
   - File path validation

2. **ASRDatabase.add_transcript()**
   - Transcript insertion
   - Foreign key relationships
   - Text normalization storage

3. **ASRDatabase.assign_split()**
   - Data split assignment
   - Split versioning
   - Update operations

4. **ASRDatabase.get_dataset_statistics()**
   - Statistics computation
   - Aggregate queries
   - Performance with large datasets

5. **ASRDatabase.get_audio_files_by_split()**
   - Split-based queries
   - Filtering and sorting
   - Pagination support

6. **ASRDatabase.add_training_run()**
   - Training run tracking
   - Model association

7. **ASRDatabase.add_epoch_metrics()**
   - Metrics storage
   - Time series data

8. **ASRDatabase.get_connection()**
   - Connection management
   - Transaction handling
   - Error recovery

---

## API Module (`api/`)

### `app.py`
**Priority: CRITICAL**

1. **POST /transcribe**
   - Audio file upload
   - File validation
   - Model loading
   - Feature extraction
   - Decoding
   - Response formatting
   - Error handling

2. **GET /health**
   - Health check
   - Component status
   - Model availability

3. **GET /models**
   - Model listing
   - Checkpoint discovery

4. **POST /models/load**
   - Model loading
   - Cache management
   - Error handling

5. **DELETE /models/{model_name}**
   - Model unloading
   - Cache cleanup

6. **security_headers()**
   - Security header injection
   - Header validation

### `public_api.py`
**Priority: HIGH**

1. All public endpoints
   - Authentication
   - Rate limiting
   - Input validation

---

## NLP Module (`nlp/`)

### `word2vec_trainer.py`
**Priority: MEDIUM**

1. **Word2VecTrainer.train()**
   - Word2Vec training
   - Vocabulary building
   - Embedding generation

2. **Word2VecTrainer.get_embeddings()**
   - Embedding retrieval
   - Vocabulary mapping

### `phon2vec_trainer.py`
**Priority: MEDIUM**

1. **Phon2VecTrainer.train()**
   - Phonetic embedding training
   - Phoneme sequence handling

### `faiss_index.py`
**Priority: MEDIUM**

1. **EmbeddingIndex.add_embeddings()**
   - Index building
   - Normalization
   - Large dataset handling

2. **EmbeddingIndex.search()**
   - Nearest neighbor search
   - K-NN queries
   - Distance computation

---

## Training Module (`training/`)

### `train.py`
**Priority: HIGH**

1. **Training loop**
   - Epoch iteration
   - Batch processing
   - Loss computation
   - Gradient updates
   - Checkpoint saving

2. **Optimizer configuration**
   - Learning rate scheduling
   - Weight decay
   - Gradient clipping

### `dataset.py`
**Priority: HIGH**

1. **Dataset loading**
   - Data loading
   - Batching
   - Shuffling
   - Data augmentation

2. **Collate function**
   - Variable length sequences
   - Padding
   - Mask generation

### `callbacks.py`
**Priority: MEDIUM**

1. **Callback execution**
   - Early stopping
   - Checkpoint saving
   - Metrics logging

### `evaluate.py`
**Priority: HIGH**

1. **Model evaluation**
   - WER computation
   - CER computation
   - Accuracy calculation
   - Batch evaluation

---

## Utils Module (`utils/`)

### `metrics.py`
**Priority: MEDIUM**

1. **calculate_wer()**
   - Word Error Rate calculation
   - Edit distance
   - Normalization

2. **calculate_cer()**
   - Character Error Rate calculation
   - Vietnamese character handling

3. **calculate_accuracy()**
   - Accuracy computation
   - Exact match detection

### `logger.py`
**Priority: LOW**

1. **Logging functions**
   - Log level handling
   - File logging
   - Formatting

---

## Microservices (`projects/services/`)

### ASR Service
**Priority: CRITICAL**

1. **Transcription endpoints**
   - Audio processing
   - Model inference
   - Streaming support

2. **Streaming optimization**
   - Chunk processing
   - Latency optimization

### Embeddings Service
**Priority: HIGH**

1. **Embedding generation**
   - Batch processing
   - Idempotency
   - Caching

### Search Service
**Priority: HIGH**

1. **Search endpoints**
   - Query processing
   - Index lookup
   - Result ranking

### Gateway Service
**Priority: CRITICAL**

1. **Routing**
   - Request routing
   - Load balancing
   - Authentication

### Training Orchestrator
**Priority: HIGH**

1. **Orchestration logic**
   - Workflow management
   - Service coordination
   - Error recovery

---

## Frontend (`frontend/src/`)

### `services/api.js`
**Priority: HIGH**

1. **transcribeAudio()**
   - File upload
   - Progress tracking
   - Error handling
   - Response parsing

2. **getModels()**
   - API calls
   - Error handling

3. **checkHealth()**
   - Health check
   - Connection status

4. **Error interceptors**
   - Global error handling
   - Retry logic
   - User feedback

### `utils/validation.js`
**Priority: MEDIUM**

1. **File validation**
   - File type checking
   - Size validation
   - Format validation

---

## Test Coverage Priorities

### Phase 1 (Immediate - Week 1-2)
1. ✅ Audio preprocessing core functions
2. ✅ Text normalization
3. ✅ Model forward passes
4. ✅ Database CRUD operations
5. ✅ API endpoints (transcribe, health)

### Phase 2 (High Priority - Week 3-4)
1. ✅ Decoding algorithms (beam search, LM)
2. ✅ Training pipeline
3. ✅ Integration tests
4. ✅ Error handling

### Phase 3 (Medium Priority - Week 5-6)
1. NLP embeddings
2. Microservices core logic
3. Frontend API client
4. Performance optimizations

### Phase 4 (Lower Priority - Week 7+)
1. Advanced features
2. Edge cases
3. Performance benchmarks
4. Security tests

---

## Testing Checklist

### For Each Function:
- [ ] Happy path (normal operation)
- [ ] Edge cases (empty input, None values, boundaries)
- [ ] Error handling (invalid input, exceptions)
- [ ] Type validation (input/output types)
- [ ] Performance (execution time, memory)
- [ ] Integration (with other components)

### For Each Module:
- [ ] Unit tests (>80% coverage)
- [ ] Integration tests
- [ ] Documentation
- [ ] Error messages
- [ ] Logging

---

**Last Updated**: 2024
**Maintained By**: Development Team

