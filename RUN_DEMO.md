# How to Run Your ASR Training System

## ✅ System Status

Your code is **working correctly**! The database validation script ran successfully.

## 📋 Current Status

- ✅ Database initialized successfully
- ✅ Database location: `database/asr_training.db`
- ✅ Validation scripts working
- ⚠️  Database is empty (no data imported yet)

## 🚀 Next Steps to Run Your System

### Step 1: Prepare Your Data CSV

Create a CSV file with your audio data. Example format:
```csv
file_path,transcript,split,speaker_id
data/raw/audio1.wav,xin chào việt nam,train,speaker_01
data/raw/audio2.wav,tôi là sinh viên,train,speaker_02
data/raw/audio3.wav,hôm nay trời đẹp,val,speaker_01
data/raw/audio4.wav,tiếng việt rất hay,test,speaker_02
```

Or use the template: `example_data_template.csv`

### Step 2: Import Your Data

```bash
python scripts/prepare_data.py --csv your_data.csv --audio_base data/raw --auto_split
```

This will:
- ✅ Validate your CSV file
- ✅ Validate all audio files
- ✅ Import data into database
- ✅ Auto-split into train/val/test
- ✅ Show training readiness report

### Step 3: Validate Database

```bash
python scripts/validate_data.py
```

This checks if your database is ready for training.

### Step 4: Start Training

Once validation passes, start training:

```bash
python training/train.py --config configs/default.yaml
```

## 🔧 Testing Without Real Data

You can test the system with a dry run:

```bash
python scripts/prepare_data.py --csv example_data_template.csv --audio_base data/raw --dry_run
```

This validates your CSV without importing.

## 📊 Quick Commands Reference

```bash
# Validate database readiness
python scripts/validate_data.py

# Import data with validation
python scripts/prepare_data.py --csv data.csv --audio_base data/raw --auto_split

# Test CSV without importing
python scripts/prepare_data.py --csv data.csv --audio_base data/raw --dry_run

# Check training readiness only
python scripts/prepare_data.py --validate_only

# Start training
python training/train.py --config configs/default.yaml

# Evaluate model
python training/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best_model.pt --split test
```

## ✨ Your System is Ready!

All components are working. You just need to:
1. Prepare your audio files and CSV
2. Import the data
3. Start training!

