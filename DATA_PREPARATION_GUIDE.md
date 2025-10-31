# Data Preparation Guide - How to Input Data for Training

## 📋 Step-by-Step Guide

### Step 1: Prepare Your Audio Files

Organize your audio files in a folder structure. Example:
```
data/
├── raw/
│   ├── audio1.wav
│   ├── audio2.wav
│   ├── audio3.wav
│   └── ...
```

**Supported formats**: WAV, MP3, FLAC, OGG

### Step 2: Create CSV File with Audio Paths and Transcripts

Create a CSV file (e.g., `data/train_data.csv`) with the following columns:

#### Required Columns:
- **`file_path`**: Path to audio file (relative or absolute)
- **`transcript`**: Vietnamese transcription text

#### Optional Columns:
- **`split`**: 'train', 'val', or 'test' (if not provided, use --auto_split)
- **`speaker_id`**: Speaker identifier (optional)

#### Example CSV Content:

**Option A: Relative Paths** (Recommended)
```csv
file_path,transcript,split,speaker_id
data/raw/audio1.wav,xin chào việt nam,train,speaker_01
data/raw/audio2.wav,tôi là sinh viên,train,speaker_02
data/raw/audio3.wav,hôm nay trời đẹp,val,speaker_01
data/raw/audio4.wav,tiếng việt rất hay,test,speaker_02
```

**Option B: Absolute Paths**
```csv
file_path,transcript,split
D:/AT2Text/AI2Text frist/data/raw/audio1.wav,xin chào việt nam,train
D:/AT2Text/AI2Text frist/data/raw/audio2.wav,tôi là sinh viên,train
```

**Option C: Without Split Column** (will use --auto_split)
```csv
file_path,transcript
data/raw/audio1.wav,xin chào việt nam
data/raw/audio2.wav,tôi là sinh viên
data/raw/audio3.wav,hôm nay trời đẹp
```

### Step 3: Import Data into Database

Run the data preparation script:

#### If CSV has relative paths:
```bash
python scripts/prepare_data.py --csv data/train_data.csv --audio_base data/raw --auto_split
```

#### If CSV has absolute paths:
```bash
python scripts/prepare_data.py --csv data/train_data.csv --auto_split
```

#### If splits are already in CSV:
```bash
python scripts/prepare_data.py --csv data/train_data.csv --audio_base data/raw
```

### Step 4: Verify Data Import

Check that data was imported successfully:

```python
from database.db_utils import ASRDatabase

db = ASRDatabase()
stats = db.get_dataset_statistics('v1')
print(stats)
```

You should see:
```
   split_type split_version  num_files  total_duration_seconds  avg_duration_seconds  num_speakers
0       train           v1          X                      X                   X             X
1         val           v1          X                      X                   X             X
2        test           v1          X                      X                   X             X
```

### Step 5: Start Training

Once data is imported, you can start training:

```bash
python training/train.py --config configs/default.yaml
```

## 📝 CSV File Template

You can create a template CSV file like this:

```python
import pandas as pd

# Create template
data = {
    'file_path': [
        'data/raw/audio1.wav',
        'data/raw/audio2.wav',
        'data/raw/audio3.wav'
    ],
    'transcript': [
        'xin chào việt nam',
        'tôi là sinh viên',
        'hôm nay trời đẹp'
    ],
    'split': ['train', 'train', 'val'],
    'speaker_id': ['speaker_01', 'speaker_02', 'speaker_01']
}

df = pd.DataFrame(data)
df.to_csv('data/my_data.csv', index=False, encoding='utf-8')
print("Template CSV created!")
```

## ⚠️ Important Notes

### 1. **CSV Encoding**
- Always save CSV files with **UTF-8 encoding** to support Vietnamese characters
- In Excel: Save As → CSV UTF-8 (Comma delimited) (*.csv)

### 2. **Audio File Paths**
- Use forward slashes `/` or double backslashes `\\` in paths
- Check that all audio files exist before importing
- Paths can be relative (recommended) or absolute

### 3. **Transcript Format**
- Write transcripts in Vietnamese with proper diacritics
- The system will automatically normalize them
- Example: "Xin chào" → automatically normalized to "xin chào"

### 4. **Split Distribution**
If using `--auto_split`:
- **Train**: 80% (default)
- **Val**: 10% (default)
- **Test**: 10% (default)

You can change these in `configs/db.yaml`:
```yaml
splits:
  train: 0.8
  val: 0.1
  test: 0.1
```

### 5. **Minimum Data Requirements**
- **Minimum recommended**: 100-200 audio files for basic training
- **Better**: 1000+ audio files
- **Best**: 10,000+ audio files (10+ hours of speech)

## 🔍 Troubleshooting

### "File not found" Warning
- Check that `--audio_base` path is correct
- Verify audio file paths in CSV match actual file locations
- Use absolute paths if relative paths don't work

### "CSV must contain column: file_path"
- Make sure CSV has exactly `file_path` and `transcript` columns
- Check column names match exactly (case-sensitive)

### "Database locked" Error
- Close any other programs accessing the database
- Make sure previous import finished

### Vietnamese Characters Not Showing
- Save CSV as UTF-8 encoding
- Use a text editor that supports UTF-8 (Notepad++, VS Code)

## 📊 Example: Complete Workflow

```bash
# 1. Create your CSV file (using Excel or text editor)
# Save as: data/my_vietnamese_data.csv

# 2. Place audio files in data/raw/ folder
# Example: data/raw/audio1.wav, audio2.wav, etc.

# 3. Import data
python scripts/prepare_data.py --csv data/my_vietnamese_data.csv --audio_base data/raw --auto_split

# 4. Verify import (optional)
python -c "from database.db_utils import ASRDatabase; db = ASRDatabase(); print(db.get_dataset_statistics('v1'))"

# 5. Start training
python training/train.py --config configs/default.yaml
```

## 📝 CSV Example with Multiple Files

Here's a complete example CSV for reference:

```csv
file_path,transcript,split,speaker_id
data/raw/spk01_001.wav,xin chào bạn có khỏe không,train,speaker_01
data/raw/spk01_002.wav,tôi đi học ở trường đại học,train,speaker_01
data/raw/spk02_001.wav,hôm nay trời rất đẹp,train,speaker_02
data/raw/spk02_002.wav,bạn có muốn đi chơi không,train,speaker_02
data/raw/spk01_003.wav,tiếng việt là ngôn ngữ mẹ đẻ,val,speaker_01
data/raw/spk02_003.wav,ăn uống đúng giờ rất quan trọng,val,speaker_02
data/raw/spk01_004.wav,chúc bạn một ngày tốt lành,test,speaker_01
```

---

**Need help?** Check the README.md for more details or inspect `scripts/prepare_data.py` for the exact CSV format expected.

