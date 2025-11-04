# Quick Training Guide - 5 Minutes

Get your model training in 5 minutes!

## Prerequisites

```bash
# Ensure dependencies are installed
pip install -r requirements/base.txt
```

## Step 1: Prepare Data (2 minutes)

### Create metadata.csv

```csv
file_path,transcript,split
data/raw/audio1.wav,xin chào việt nam,train
data/raw/audio2.wav,tôi là sinh viên,train
data/raw/audio3.wav,hôm nay trời đẹp,val
```

### Load into database

```bash
python -c "
from database.db_utils import ASRDatabase
import pandas as pd

# Load CSV
df = pd.read_csv('data/metadata.csv')

# Initialize database
db = ASRDatabase('database/asr_training.db')

# Add each file
for _, row in df.iterrows():
    audio_id = db.add_audio_file(
        file_path=row['file_path'],
        filename=row['file_path'].split('/')[-1],
        duration=5.0,  # Will be calculated
        sample_rate=16000
    )
    db.add_transcript(
        audio_file_id=audio_id,
        transcript=row['transcript'],
        normalized_transcript=row['transcript'].lower()
    )
    db.assign_split(audio_id, row['split'], 'v1')

print('Data loaded!')
"
```

## Step 2: Configure Training (1 minute)

Edit `configs/default.yaml` (optional, defaults are good):

```yaml
# Adjust for your hardware
batch_size: 16          # Reduce if out of memory
num_epochs: 10          # Start small for testing
learning_rate: 0.0001
```

## Step 3: Start Training (30 seconds)

```bash
# Start training
python training/train.py --config configs/default.yaml
```

## Step 4: Monitor Training (1 minute)

Watch the progress:

```bash
# In another terminal
tail -f logs/training.log
```

You should see:

```
Epoch 1/10
Train: 100%|██████████| 50/50 [00:45<00:00]
Val: 100%|██████████| 5/5 [00:05<00:00]
Val Loss: 2.34 | Val WER: 45.2%
Saved best model!
```

## Step 5: Use Trained Model (1 minute)

```bash
# Transcribe audio
python -c "
from training.evaluate import ASREvaluator

evaluator = ASREvaluator('checkpoints/best_model.pt')
result = evaluator.transcribe('test_audio.wav')
print(f'Transcription: {result[\"text\"]}')
print(f'Confidence: {result[\"confidence\"]:.2f}')
"
```

## That's It! 🎉

Your model is now training. It will:
- ✅ Train for specified epochs
- ✅ Save checkpoints automatically
- ✅ Validate after each epoch
- ✅ Save best model based on validation loss
- ✅ Log all metrics

## Next Steps

1. **Monitor training**: `tail -f logs/training.log`
2. **Check checkpoints**: `ls -lh checkpoints/`
3. **Evaluate**: `python training/evaluate.py --model_path checkpoints/best_model.pt`

## Troubleshooting

**Out of memory?**
```yaml
# In config
batch_size: 4
```

**Training too slow?**
```yaml
num_workers: 8
use_amp: true  # Use mixed precision
```

**Need more details?**
- See `TRAINING_GUIDE.md` for comprehensive guide
- Run `python scripts/check_project.py` for health check

---

**Time to train**: ~5-10 minutes per epoch (depends on dataset size and hardware)

