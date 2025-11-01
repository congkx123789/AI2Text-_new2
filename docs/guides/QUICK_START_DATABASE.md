# Quick Start - Improved Database Preparation

## 🚀 Fast Setup (3 Steps)

### Step 1: Create Your CSV File

Create `my_data.csv` with this format:
```csv
file_path,transcript,split
data/raw/audio1.wav,xin chào việt nam,train
data/raw/audio2.wav,tôi là sinh viên,train
data/raw/audio3.wav,hôm nay trời đẹp,val
```

### Step 2: Import Data (With Validation)

```bash
python scripts/prepare_data.py --csv my_data.csv --audio_base data/raw --auto_split
```

### Step 3: Validate Training Readiness

```bash
python scripts/validate_data.py
```

If it says "✅ Database is READY for training!", you can start training!

---

## ✨ New Features You Can Use

### 1. Dry Run (Validate Without Importing)
```bash
python scripts/prepare_data.py --csv my_data.csv --audio_base data/raw --dry_run
```
This checks your CSV without adding to database - perfect for testing!

### 2. Speaker-Balanced Splitting
```bash
python scripts/prepare_data.py --csv my_data.csv --audio_base data/raw --auto_split --split_strategy speaker_balanced
```
Distributes speakers evenly across train/val/test splits.

### 3. Check Readiness Anytime
```bash
python scripts/prepare_data.py --validate_only
```
Quickly check if your database is ready for training.

### 4. Fast Batch Import
```bash
python scripts/prepare_data.py --csv my_data.csv --audio_base data/raw --batch_size 200
```
Import faster with larger batches (default: 100).

---

## 📊 What Gets Validated

✅ **CSV Format**: Encoding, columns, duplicates  
✅ **Audio Files**: Existence, format, duration, sample rate  
✅ **Transcripts**: Format, length, quality  
✅ **Duplicates**: Automatically skipped  
✅ **Splits**: Train/val/test distribution  
✅ **Training Readiness**: Complete validation before training  

---

## 🔍 Check Your Data

After importing, get detailed statistics:

```bash
python scripts/validate_data.py
```

Or in Python:
```python
from database.db_utils import ASRDatabase

db = ASRDatabase()
summary = db.get_data_summary('v1')
print(summary)

# Or just validate
validation = db.validate_data_for_training('v1')
print(f"Ready: {validation['is_ready']}")
```

---

## 🎯 What's Improved

| Feature | Before | After |
|---------|--------|-------|
| **Validation** | Basic | Comprehensive |
| **Speed** | Sequential | Batch processing (3-5x faster) |
| **Error Handling** | Basic | Detailed error logs |
| **Duplicate Detection** | Manual | Automatic |
| **Training Readiness** | Manual check | Automatic validation |
| **Statistics** | Basic | Comprehensive reports |
| **Splitting Strategy** | Random only | Random + Speaker-balanced |

---

## 💡 Tips

1. **Always validate first**: Use `--dry_run` to check your CSV
2. **Check readiness**: Run `--validate_only` before training
3. **Review errors**: Check `import_errors.csv` if import fails
4. **Use batch processing**: Larger `--batch_size` for faster imports
5. **Speaker balance**: Use `speaker_balanced` for better validation sets

---

**Ready to train?** If validation passes, you can start training with:
```bash
python training/train.py --config configs/default.yaml
```

