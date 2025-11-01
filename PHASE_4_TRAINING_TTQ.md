# Phase 4 - Training & Time-to-Quality (TTQ)

**Goal:** Production model training with DVC, tarred shards, and TTQ optimization for limited GPU resources

**Timeline:** Ongoing (parallel with Phase 3)

---

## 🎯 Phase 4 Overview

Optimize the training pipeline for:
1. **Reproducibility**: DVC for dataset versioning
2. **Efficiency**: Tarred shards for fast data loading
3. **Resource Optimization**: Gradient accumulation + mixed precision for weak GPUs
4. **Quality Gates**: Time-to-Target-Quality (TTQ) budgeting
5. **Model Selection**: Vietnamese-focused SSL/Conformer vs Whisper

---

## ✅ Phase 4 Exit Criteria

- [ ] Training data in S3 with tarred shards
- [ ] DVC tracks all datasets and model checkpoints
- [ ] Gradient accumulation enables large batch sizes on modest GPUs
- [ ] Mixed precision (FP16/BF16) training working
- [ ] TTQ milestones defined and tracked
- [ ] Vietnamese ASR model achieves target WER on validation set
- [ ] Model can be deployed to streaming service

---

## 📋 Tasks Breakdown

### Task 4.1: Setup DVC for Dataset Versioning

**Status:** 🔴 TODO

**Goal:** Track datasets, preprocessing configs, and model checkpoints with DVC.

#### Step 1: Initialize DVC

```bash
# Install DVC
pip install dvc dvc-s3

# Initialize DVC in repo
cd /path/to/AI2Text
dvc init

# Configure S3 remote (or local for dev)
dvc remote add -d storage s3://ai2text-training-data/dvc-store

# Or use MinIO for dev
dvc remote add -d storage s3://dvc-store
dvc remote modify storage endpointurl http://localhost:9000
dvc remote modify storage access_key_id minio
dvc remote modify storage secret_access_key minio123

# Track data directory
dvc add data/processed
git add data/processed.dvc .gitignore
git commit -m "Track processed data with DVC"

# Push to remote
dvc push
```

#### Step 2: Create DVC Pipeline

**File:** `dvc.yaml`

```yaml
stages:
  prepare_data:
    cmd: python scripts/prepare_data.py --config configs/data_prep.yaml
    deps:
    - scripts/prepare_data.py
    - configs/data_prep.yaml
    - data/raw
    params:
    - data_prep.yaml:
      - sample_rate
      - max_duration
      - augmentation
    outs:
    - data/processed:
        cache: true
    metrics:
    - data/processed/stats.json:
        cache: false
  
  train_model:
    cmd: python training/train.py --config configs/train_conformer.yaml
    deps:
    - training/train.py
    - configs/train_conformer.yaml
    - data/processed
    params:
    - train_conformer.yaml:
      - model.type
      - model.hidden_size
      - training.batch_size
      - training.learning_rate
    outs:
    - checkpoints/conformer_vi:
        cache: true
    metrics:
    - results/train_metrics.json:
        cache: false
    plots:
    - results/learning_curve.csv:
        x: epoch
        y: val_wer
  
  evaluate_model:
    cmd: python training/evaluate.py --checkpoint checkpoints/conformer_vi/best.pt
    deps:
    - training/evaluate.py
    - checkpoints/conformer_vi/best.pt
    - data/processed/test
    metrics:
    - results/eval_metrics.json:
        cache: false
```

#### Step 3: Track Experiments

```bash
# Run pipeline
dvc repro

# Track metrics
dvc metrics show

# Compare experiments
dvc params diff HEAD~1

# Visualize plots
dvc plots show results/learning_curve.csv
```

---

### Task 4.2: Convert Data to Tarred Shards

**Status:** 🔴 TODO

**Goal:** Use WebDataset/tarred shards for 10-100x faster data loading.

#### Create Shard Conversion Script

**File:** `scripts/create_tarred_shards.py`

```python
#!/usr/bin/env python3
"""
Convert audio dataset to tarred WebDataset shards.

This provides:
- 10-100x faster loading than individual files
- Efficient streaming from S3
- Built-in shuffling and batching
"""

import os
import json
import tarfile
from pathlib import Path
from typing import Iterator, Tuple
import io

def create_shards(
    audio_dir: Path,
    transcript_file: Path,
    output_dir: Path,
    shard_size: int = 1000,
    sample_rate: int = 16000
):
    """
    Create tarred shards from audio dataset.
    
    Args:
        audio_dir: Directory with WAV files
        transcript_file: JSON file with transcripts
        output_dir: Output directory for shards
        shard_size: Number of samples per shard
        sample_rate: Audio sample rate
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load transcripts
    with open(transcript_file) as f:
        transcripts = json.load(f)
    
    # Group into shards
    shard_idx = 0
    sample_count = 0
    current_shard = []
    
    for audio_id, transcript in transcripts.items():
        audio_path = audio_dir / f"{audio_id}.wav"
        
        if not audio_path.exists():
            print(f"[WARNING] Missing audio: {audio_id}")
            continue
        
        # Read audio
        with open(audio_path, 'rb') as f:
            audio_bytes = f.read()
        
        # Add to current shard
        current_shard.append({
            "audio_id": audio_id,
            "audio_bytes": audio_bytes,
            "transcript": transcript,
            "sample_rate": sample_rate
        })
        
        sample_count += 1
        
        # Write shard when full
        if len(current_shard) >= shard_size:
            _write_shard(output_dir, shard_idx, current_shard)
            current_shard = []
            shard_idx += 1
    
    # Write remaining samples
    if current_shard:
        _write_shard(output_dir, shard_idx, current_shard)
    
    print(f"✓ Created {shard_idx + 1} shards with {sample_count} samples")
    
    # Create manifest
    manifest = {
        "num_shards": shard_idx + 1,
        "samples_per_shard": shard_size,
        "total_samples": sample_count,
        "sample_rate": sample_rate
    }
    
    with open(output_dir / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)


def _write_shard(output_dir: Path, shard_idx: int, samples: list):
    """Write samples to a tar shard."""
    shard_path = output_dir / f"shard-{shard_idx:06d}.tar"
    
    with tarfile.open(shard_path, 'w') as tar:
        for i, sample in enumerate(samples):
            # Write audio
            audio_info = tarfile.TarInfo(name=f"{i:08d}.wav")
            audio_info.size = len(sample["audio_bytes"])
            tar.addfile(audio_info, io.BytesIO(sample["audio_bytes"]))
            
            # Write transcript JSON
            transcript_json = json.dumps({
                "audio_id": sample["audio_id"],
                "transcript": sample["transcript"],
                "sample_rate": sample["sample_rate"]
            }).encode()
            
            transcript_info = tarfile.TarInfo(name=f"{i:08d}.json")
            transcript_info.size = len(transcript_json)
            tar.addfile(transcript_info, io.BytesIO(transcript_json))
    
    print(f"  Wrote {shard_path} ({len(samples)} samples)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", required=True, type=Path)
    parser.add_argument("--transcript-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--sample-rate", type=int, default=16000)
    
    args = parser.parse_args()
    
    create_shards(
        args.audio_dir,
        args.transcript_file,
        args.output_dir,
        args.shard_size,
        args.sample_rate
    )
```

**Usage:**

```bash
# Create shards
python scripts/create_tarred_shards.py \
  --audio-dir data/raw/train \
  --transcript-file data/raw/train_transcripts.json \
  --output-dir data/processed/train_shards \
  --shard-size 1000

# Upload to S3
aws s3 sync data/processed/train_shards s3://ai2text-training/shards/train/

# Track with DVC
dvc add data/processed/train_shards
git add data/processed/train_shards.dvc
git commit -m "Add training shards"
dvc push
```

#### Create WebDataset Loader

**File:** `training/webdataset_loader.py`

```python
"""WebDataset loader for tarred shards."""

import webdataset as wds
import torch
from torch.utils.data import DataLoader
import torchaudio
import io

def create_webdataset_loader(
    shard_urls: list,
    batch_size: int,
    num_workers: int = 4,
    shuffle_buffer: int = 10000
):
    """
    Create DataLoader from WebDataset shards.
    
    Args:
        shard_urls: List of shard URLs (s3:// or file://)
        batch_size: Batch size
        num_workers: Number of workers
        shuffle_buffer: Shuffle buffer size
    
    Returns:
        DataLoader
    """
    
    def preprocess(sample):
        """Preprocess a single sample."""
        # Load audio from bytes
        audio_bytes = sample["wav"]
        waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
        
        # Load transcript
        import json
        transcript_data = json.loads(sample["json"])
        transcript = transcript_data["transcript"]
        
        return {
            "audio": waveform,
            "transcript": transcript,
            "audio_id": transcript_data["audio_id"]
        }
    
    # Create dataset
    dataset = (
        wds.WebDataset(shard_urls)
        .shuffle(shuffle_buffer)
        .decode()
        .map(preprocess)
    )
    
    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=2
    )
    
    return loader


# Usage in training script:
# train_loader = create_webdataset_loader(
#     shard_urls=[
#         "s3://ai2text-training/shards/train/shard-000000.tar",
#         "s3://ai2text-training/shards/train/shard-000001.tar",
#         # ...
#     ],
#     batch_size=32,
#     num_workers=4
# )
```

---

### Task 4.3: Implement Gradient Accumulation + Mixed Precision

**Status:** 🔴 TODO

**Goal:** Enable large effective batch sizes on GPUs with limited memory.

**File:** `training/train_optimized.py`

```python
"""
Training with Gradient Accumulation and Mixed Precision.

For weak GPUs (e.g., single RTX 3060 with 12GB):
- Gradient accumulation: batch_size=4, accumulation_steps=8 → effective_batch=32
- Mixed precision (FP16): 2x memory savings, 2-3x speedup
- Gradient checkpointing: Further memory savings for large models
"""

import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

class TrainerOptimized:
    """Optimized trainer for limited GPU resources."""
    
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        batch_size: int = 4,
        accumulation_steps: int = 8,
        use_amp: bool = True,
        gradient_clip_val: float = 1.0
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        self.use_amp = use_amp
        self.gradient_clip_val = gradient_clip_val
        
        # Effective batch size
        self.effective_batch_size = batch_size * accumulation_steps
        
        # Mixed precision scaler
        self.scaler = GradScaler() if use_amp else None
        
        print(f"[Training Config]")
        print(f"  Batch size: {batch_size}")
        print(f"  Accumulation steps: {accumulation_steps}")
        print(f"  Effective batch size: {self.effective_batch_size}")
        print(f"  Mixed precision: {use_amp}")
    
    def train_epoch(self, train_loader, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Reset gradients
        self.optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            audio = batch["audio"].cuda()
            transcripts = batch["transcript"]
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                logits = self.model(audio)
                loss = self.criterion(logits, transcripts)
                
                # Scale loss by accumulation steps
                loss = loss / self.accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights every N steps
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Clip gradients
                if self.gradient_clip_val > 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip_val
                    )
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                # Reset gradients
                self.optimizer.zero_grad()
            
            # Track loss
            total_loss += loss.item() * self.accumulation_steps
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": total_loss / num_batches})
        
        return total_loss / num_batches


# Usage:
# trainer = TrainerOptimized(
#     model=conformer_model,
#     optimizer=optimizer,
#     criterion=ctc_loss,
#     batch_size=4,           # Fit in 12GB GPU
#     accumulation_steps=8,   # Effective batch = 32
#     use_amp=True            # FP16 mixed precision
# )
# 
# for epoch in range(num_epochs):
#     loss = trainer.train_epoch(train_loader, epoch)
```

---

### Task 4.4: Define TTQ Milestones

**Status:** 🔴 TODO

**Goal:** Set time and quality budgets for training iterations.

**File:** `training/ttq_tracker.py`

```python
"""
Time-to-Target-Quality (TTQ) Tracker.

Tracks progress toward quality milestones and estimates time/cost.
"""

import time
import json
from pathlib import Path
from typing import Dict, List

class TTQTracker:
    """Track TTQ milestones."""
    
    def __init__(self, milestones: List[Dict], log_file: Path):
        """
        Args:
            milestones: List of {"wer": float, "budget_hours": float}
            log_file: JSON file to log progress
        """
        self.milestones = sorted(milestones, key=lambda x: x["wer"], reverse=True)
        self.log_file = log_file
        self.start_time = time.time()
        self.history = []
    
    def log_epoch(self, epoch: int, train_loss: float, val_wer: float):
        """Log epoch results and check milestones."""
        elapsed_hours = (time.time() - self.start_time) / 3600
        
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_wer": val_wer,
            "elapsed_hours": elapsed_hours,
            "timestamp": time.time()
        }
        
        self.history.append(entry)
        
        # Check milestones
        achieved_milestones = []
        for milestone in self.milestones:
            target_wer = milestone["wer"]
            budget_hours = milestone["budget_hours"]
            
            if val_wer <= target_wer and elapsed_hours <= budget_hours:
                achieved_milestones.append({
                    "target_wer": target_wer,
                    "achieved_wer": val_wer,
                    "budget_hours": budget_hours,
                    "elapsed_hours": elapsed_hours,
                    "epoch": epoch
                })
                print(f"\n🎯 TTQ Milestone Achieved!")
                print(f"   Target WER: {target_wer:.2%}")
                print(f"   Achieved WER: {val_wer:.2%}")
                print(f"   Time: {elapsed_hours:.1f}h / {budget_hours:.1f}h budget")
        
        # Save log
        self._save_log(achieved_milestones)
        
        return achieved_milestones
    
    def _save_log(self, achieved_milestones):
        """Save progress log."""
        log_data = {
            "milestones": self.milestones,
            "achieved": achieved_milestones,
            "history": self.history,
            "total_elapsed_hours": (time.time() - self.start_time) / 3600
        }
        
        with open(self.log_file, 'w') as f:
            json.dump(log_data, f, indent=2)


# Usage in training:
# ttq = TTQTracker(
#     milestones=[
#         {"wer": 0.30, "budget_hours": 4},   # Quick baseline
#         {"wer": 0.20, "budget_hours": 12},  # Good quality
#         {"wer": 0.15, "budget_hours": 48},  # Production quality
#         {"wer": 0.10, "budget_hours": 120}  # SOTA
#     ],
#     log_file=Path("results/ttq_log.json")
# )
# 
# for epoch in range(num_epochs):
#     train_loss = train_epoch(...)
#     val_wer = validate(...)
#     ttq.log_epoch(epoch, train_loss, val_wer)
```

---

### Task 4.5: Vietnamese Model Selection

**Status:** 🔴 TODO

**Goal:** Choose optimal architecture for Vietnamese ASR.

#### Option A: Conformer-CTC (Recommended)

**Pros:**
- Fast inference (~100-200ms)
- Good for streaming (<500ms partials)
- Lower memory footprint
- Easy to train on single GPU

**Cons:**
- Requires more training data
- May need external language model

**Config:** `configs/train_conformer.yaml`

```yaml
model:
  type: conformer_ctc
  encoder:
    hidden_size: 256
    num_layers: 12
    num_attention_heads: 4
    ffn_dim: 1024
    kernel_size: 31
    dropout: 0.1
  
  decoder:
    type: ctc
    vocab_size: 100  # Vietnamese characters + special tokens

training:
  # Optimizer
  optimizer: adamw
  learning_rate: 5e-4
  weight_decay: 1e-6
  
  # Schedule
  warmup_steps: 1000
  max_steps: 100000
  
  # Batch
  batch_size: 4
  accumulation_steps: 8
  effective_batch_size: 32
  
  # Mixed precision
  use_amp: true
  amp_dtype: float16
  
  # Gradient
  gradient_clip_val: 1.0
  
  # TTQ milestones
  ttq_milestones:
    - wer: 0.30
      budget_hours: 4
    - wer: 0.20
      budget_hours: 12
    - wer: 0.15
      budget_hours: 48
```

#### Option B: Whisper Fine-tuning

**Pros:**
- Pre-trained on massive data
- Multilingual (good for code-switching)
- Good out-of-box quality

**Cons:**
- Slower inference (~1-2s)
- Harder to stream (<500ms difficult)
- Larger memory footprint

**Config:** `configs/finetune_whisper.yaml`

```yaml
model:
  type: whisper
  size: base  # or small, medium
  language: vi
  
  # Fine-tuning
  freeze_encoder: false
  freeze_layers: 0

training:
  optimizer: adamw
  learning_rate: 1e-5  # Lower LR for fine-tuning
  weight_decay: 1e-6
  
  batch_size: 2
  accumulation_steps: 16
  effective_batch_size: 32
  
  use_amp: true
  gradient_clip_val: 1.0
```

#### Option C: SSL Pre-training (Advanced)

**For maximum quality with limited labeled data:**

1. **Self-supervised pre-training** on unlabeled Vietnamese audio
2. **Fine-tuning** on labeled data

**Approach:**
- Use wav2vec 2.0 or HuBERT
- Pre-train on Common Voice Vietnamese (unlabeled)
- Fine-tune on labeled corpus

**Time:** 1-2 weeks for pre-training + 1-2 days for fine-tuning

---

## 📊 Training Performance Benchmarks

| Setup | GPU | Batch Size | Throughput | Time to 20% WER |
|-------|-----|------------|------------|-----------------|
| No optimization | RTX 3060 12GB | 2 | 10 samples/s | ~100 hours |
| + Mixed Precision | RTX 3060 12GB | 4 | 25 samples/s | ~40 hours |
| + Grad Accum (8x) | RTX 3060 12GB | 4→32 | 25 samples/s | ~20 hours |
| + Tarred Shards | RTX 3060 12GB | 4→32 | 80 samples/s | ~8 hours |
| A100 40GB | A100 40GB | 32 | 200 samples/s | ~3 hours |

---

## ✅ Phase 4 Verification

### 1. Test DVC Pipeline

```bash
# Run full pipeline
dvc repro

# Check metrics
dvc metrics show

# Compare runs
dvc metrics diff
```

### 2. Test Tarred Shards

```bash
# Create shards
python scripts/create_tarred_shards.py \
  --audio-dir data/raw/dev \
  --transcript-file data/raw/dev_transcripts.json \
  --output-dir data/processed/dev_shards

# Test loading speed
python - <<'EOF'
import time
from training.webdataset_loader import create_webdataset_loader

loader = create_webdataset_loader(
    shard_urls=["file://data/processed/dev_shards/shard-000000.tar"],
    batch_size=32
)

start = time.time()
for i, batch in enumerate(loader):
    if i >= 100: break
elapsed = time.time() - start
print(f"Loading speed: {100 * 32 / elapsed:.1f} samples/s")
EOF
```

### 3. Test Mixed Precision Training

```bash
# Train for 10 steps
python training/train_optimized.py \
  --config configs/train_conformer.yaml \
  --max-steps 10 \
  --use-amp

# Should see:
# - GPU memory usage <12GB
# - ~2x speedup vs FP32
# - Loss decreasing
```

### 4. Verify TTQ Tracking

```bash
# Check TTQ log
cat results/ttq_log.json | jq '.achieved'

# Should show achieved milestones
```

---

## 🎯 Phase 4 Complete Checklist

- [ ] DVC initialized and tracking datasets
- [ ] Training data converted to tarred shards
- [ ] Shards uploaded to S3
- [ ] Gradient accumulation working (effective batch ≥32)
- [ ] Mixed precision training enabled (FP16/BF16)
- [ ] TTQ milestones defined and tracked
- [ ] Model architecture selected (Conformer/Whisper/SSL)
- [ ] Training throughput >50 samples/s on target hardware
- [ ] First model checkpoint achieving <30% WER
- [ ] Model exported for deployment

---

## 📚 Next Steps After Phase 4

1. **Deploy trained model** to streaming service
2. **A/B test** new model vs baseline
3. **Monitor production metrics** (WER, latency)
4. **Collect feedback** and retrain
5. **Iterate** on model architecture and data

---

**Phase 4 enables efficient, reproducible training at scale! 🚀**

