# How to Train Your AI2Text Model

**Quick Summary**: Complete guide to training your Vietnamese Speech-to-Text model.

## 📚 Available Guides

1. **QUICK_TRAIN.md** - Start training in 5 minutes (recommended for beginners)
2. **TRAINING_GUIDE.md** - Comprehensive training guide with all details
3. **examples/train_example.py** - Simple training script example

---

## 🚀 Quick Start (5 Minutes)

### Step 1: Prepare Your Data

Create `data/metadata.csv`:
```csv
file_path,transcript,split
data/raw/audio1.wav,xin chào việt nam,train
data/raw/audio2.wav,tôi là sinh viên,train
data/raw/audio3.wav,hôm nay trời đẹp,val
```

### Step 2: Load Data to Database

```bash
python scripts/prepare_data.py --csv_path data/metadata.csv
```

### Step 3: Start Training

```bash
python training/train.py --config configs/default.yaml
```

**That's it!** Your model is now training.

---

## 📖 Detailed Steps

### 1. Data Preparation

You need:
- **Audio files**: WAV format, 16kHz, mono
- **Transcripts**: Vietnamese text
- **Metadata**: CSV linking audio to transcripts

**Minimum data**: 100 hours (1000+ hours recommended for production)

**Example CSV format**:
```csv
file_path,transcript,duration,split,speaker_id
data/raw/audio1.wav,xin chào việt nam,5.2,train,speaker_01
data/raw/audio2.wav,tôi là sinh viên,4.8,train,speaker_02
```

**Load to database**:
```python
from database.db_utils import ASRDatabase
import pandas as pd

db = ASRDatabase('database/asr_training.db')
df = pd.read_csv('data/metadata.csv')

for _, row in df.iterrows():
    audio_id = db.add_audio_file(
        file_path=row['file_path'],
        filename=row['file_path'].split('/')[-1],
        duration=row.get('duration', 5.0),
        sample_rate=16000
    )
    db.add_transcript(
        audio_file_id=audio_id,
        transcript=row['transcript']
    )
    db.assign_split(audio_id, row['split'], 'v1')
```

### 2. Configuration

Edit `configs/default.yaml`:

```yaml
# Model size (adjust for your hardware)
d_model: 256              # 128=small, 256=medium, 512=large
num_encoder_layers: 6     # 4=small, 6=medium, 12=large

# Training
batch_size: 16            # Reduce if out of memory
num_epochs: 50            # More epochs = better model
learning_rate: 0.0001     # Standard value

# Hardware optimization
use_amp: true             # Mixed precision (faster)
num_workers: 4            # Data loading threads
```

**For weak hardware**:
```yaml
batch_size: 4
d_model: 128
num_encoder_layers: 4
use_amp: false
```

**For strong hardware**:
```yaml
batch_size: 32
d_model: 512
num_encoder_layers: 12
use_amp: true
num_workers: 8
```

### 3. Training

**Basic training**:
```bash
python training/train.py --config configs/default.yaml
```

**With custom parameters**:
```bash
python training/train.py \
    --config configs/default.yaml \
    --num_epochs 20 \
    --batch_size 32 \
    --learning_rate 0.0005
```

**Resume from checkpoint**:
```bash
python training/train.py \
    --config configs/default.yaml \
    --resume checkpoints/last_checkpoint.pt
```

### 4. Monitor Training

Watch progress in terminal:
```
Epoch 1/50
Train: 100%|██████████| 500/500 [05:23<00:00]
Val:   100%|██████████| 50/50 [00:32<00:00]

Epoch 1 Summary:
├── Train Loss: 2.45
├── Val Loss: 2.12
├── Val WER: 45.3%
└── Saved best model!
```

**Check logs**:
```bash
tail -f logs/training.log
```

**View checkpoints**:
```bash
ls -lh checkpoints/
```

### 5. Evaluate

```bash
python training/evaluate.py \
    --model_path checkpoints/best_model.pt \
    --test_data database/asr_training.db \
    --split test
```

Output:
```
Test Results:
├── WER: 12.5%
├── CER: 6.2%
└── Accuracy: 87.5%
```

### 6. Use Trained Model

**Via API**:
```bash
# Start server
uvicorn api.app:app --reload

# Test
curl -X POST "http://localhost:8000/transcribe" \
  -F "audio=@test.wav"
```

**Via Python**:
```python
from training.evaluate import ASREvaluator

evaluator = ASREvaluator('checkpoints/best_model.pt')
result = evaluator.transcribe('test.wav')
print(result['text'])
```

---

## 🎯 Training Workflow

```
1. Data Preparation
   ├── Collect audio files (WAV, 16kHz)
   ├── Create transcripts
   └── Create metadata CSV

2. Load to Database
   └── python scripts/prepare_data.py

3. Configure Training
   └── Edit configs/default.yaml

4. Start Training
   └── python training/train.py

5. Monitor Progress
   ├── Watch terminal output
   ├── Check logs/training.log
   └── View checkpoints/

6. Evaluate Model
   └── python training/evaluate.py

7. Deploy Model
   ├── Use with API
   └── Or load in Python
```

---

## 💡 Training Tips

### For Best Results

1. **More data is better**
   - Minimum: 100 hours
   - Good: 500 hours
   - Excellent: 1000+ hours

2. **Balanced data**
   - Multiple speakers
   - Various accents
   - Different recording conditions

3. **Clean transcripts**
   - Accurate spelling
   - Proper Vietnamese diacritics
   - Remove fillers and noise labels

4. **Train longer**
   - Start with 10-20 epochs (quick test)
   - Train 50-100 epochs (production)
   - Monitor validation loss

5. **Use validation set**
   - 80% train, 10% val, 10% test
   - Validate after each epoch
   - Save best model

### Common Issues

**Out of memory?**
```yaml
batch_size: 4          # Reduce batch size
use_amp: true          # Use mixed precision
grad_accum_steps: 4    # Accumulate gradients
```

**Training too slow?**
```yaml
num_workers: 8         # More data loading threads
pin_memory: true       # Faster GPU transfer
use_amp: true          # Mixed precision
```

**Loss not decreasing?**
- Check data quality
- Reduce learning rate
- Add gradient clipping
- Increase model size

**Overfitting?**
```yaml
dropout: 0.3           # More dropout
weight_decay: 0.01     # Add regularization
augmentation: true     # Data augmentation
```

---

## 📊 Expected Training Time

**Small model** (d_model=128, 4 layers):
- CPU: ~30 minutes/epoch (100 hours data)
- GPU: ~5 minutes/epoch

**Medium model** (d_model=256, 6 layers):
- CPU: ~60 minutes/epoch
- GPU: ~10 minutes/epoch

**Large model** (d_model=512, 12 layers):
- CPU: ~120 minutes/epoch
- GPU: ~20 minutes/epoch

*Times vary based on hardware and data size*

---

## 🎓 Example Training Session

```bash
# 1. Check project health
python scripts/check_project.py

# 2. Prepare data
python scripts/prepare_data.py --csv_path data/metadata.csv

# 3. Verify data
python -c "
from database.db_utils import ASRDatabase
db = ASRDatabase('database/asr_training.db')
stats = db.get_dataset_statistics('v1')
print(f'Train: {stats.get(\"train_files\", 0)} files')
print(f'Val: {stats.get(\"val_files\", 0)} files')
"

# 4. Start training
python training/train.py --config configs/default.yaml

# 5. Monitor (in another terminal)
tail -f logs/training.log

# 6. After training, evaluate
python training/evaluate.py --model_path checkpoints/best_model.pt

# 7. Test with API
uvicorn api.app:app --reload
```

---

## 📁 Files You'll Need

**Input**:
- `data/raw/*.wav` - Your audio files
- `data/metadata.csv` - Audio-transcript mapping
- `configs/default.yaml` - Training configuration

**Output**:
- `database/asr_training.db` - Training database
- `checkpoints/best_model.pt` - Best model
- `checkpoints/last_checkpoint.pt` - Latest checkpoint
- `logs/training.log` - Training logs

---

## 🔗 Related Documentation

- **QUICK_TRAIN.md** - 5-minute quick start
- **TRAINING_GUIDE.md** - Comprehensive guide
- **README_RUN_GUIDE.md** - Project setup
- **PROJECT_HEALTH_REPORT.md** - Project status
- **examples/train_example.py** - Training example

---

## 🆘 Need Help?

1. **Check project health**:
   ```bash
   python scripts/check_project.py
   ```

2. **Read comprehensive guide**:
   - Open `TRAINING_GUIDE.md`

3. **Run example**:
   ```bash
   python examples/train_example.py
   ```

4. **Check logs**:
   ```bash
   cat logs/training.log
   ```

---

## ✅ Training Checklist

Before training:
- [ ] Audio files prepared (WAV, 16kHz, mono)
- [ ] Transcripts ready (Vietnamese text)
- [ ] metadata.csv created
- [ ] Data loaded to database
- [ ] Configuration reviewed
- [ ] Checkpoints directory exists
- [ ] Enough disk space

During training:
- [ ] Monitor loss (should decrease)
- [ ] Check validation metrics
- [ ] Watch for errors
- [ ] Save checkpoints

After training:
- [ ] Evaluate on test set
- [ ] Check WER/CER metrics
- [ ] Test with real audio
- [ ] Deploy model

---

**Ready to train?** Start with `QUICK_TRAIN.md` for the fastest path! 🚀

**Last Updated**: 2024  
**Version**: 1.0.0

