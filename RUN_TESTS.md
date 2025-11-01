# Running Tests - Quick Guide

## Current Issue

There's a PyTorch DLL loading issue on Windows. The tests are set up correctly, but PyTorch needs to be properly installed for Windows.

## Quick Fix Options

### Option 1: Reinstall PyTorch (Recommended)
```bash
# Uninstall current torch
pip uninstall torch torchvision torchaudio -y

# Reinstall PyTorch for Windows CPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Or if you have CUDA:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Option 2: Run Tests Without Torch Dependencies First
Some tests might work if we skip torch-dependent imports. However, most tests require torch.

### Option 3: Use Test Discovery Only
```bash
# List all tests (may fail on import, but shows structure)
python -m pytest tests/ --collect-only --ignore=tests/conftest.py
```

## Test Files Available

### Basic Unit Tests
- `tests/unit/preprocessing/test_audio_processing.py` - Audio processing tests
- `tests/unit/preprocessing/test_text_cleaning.py` - Text cleaning tests
- `tests/unit/models/test_asr_base.py` - Base model tests
- `tests/unit/models/test_enhanced_asr.py` - Enhanced model tests
- `tests/unit/decoding/test_beam_search.py` - Beam search tests
- `tests/unit/decoding/test_lm_decoder.py` - LM decoder tests
- `tests/unit/database/test_db_utils.py` - Database tests
- `tests/unit/api/test_app.py` - API tests

### Advanced Tests
- `tests/unit/preprocessing/test_audio_processing_advanced.py` - Advanced audio tests
- `tests/unit/models/test_models_advanced.py` - Advanced model tests
- `tests/unit/decoding/test_decoding_advanced.py` - Advanced decoding tests
- `tests/unit/api/test_api_advanced.py` - Advanced API tests

### Performance Tests
- `tests/performance/test_performance_benchmarks.py` - Performance benchmarks

### Security Tests
- `tests/security/test_security.py` - Security tests

### Integration Tests
- `tests/integration/test_api_integration.py` - Integration tests

## Running Tests (After Fixing PyTorch)

### Run All Tests
```bash
python -m pytest tests/ -v
```

### Run Specific Test File
```bash
python -m pytest tests/unit/preprocessing/test_audio_processing.py -v
```

### Run with Coverage
```bash
python -m pytest tests/ --cov=preprocessing --cov=models --cov-report=html
```

### Run Only Fast Tests
```bash
python -m pytest tests/ -m "not slow" -v
```

### Run Performance Tests
```bash
python -m pytest tests/performance/ --benchmark-only
```

### Run Security Tests
```bash
python -m pytest tests/security/ -v
```

### Run Advanced Tests Only
```bash
python -m pytest tests/unit/*/test_*_advanced.py -v
```

## Expected Test Count

After fixing PyTorch, you should see:
- **Basic Tests**: ~150 test cases
- **Advanced Tests**: ~350 test cases
- **Performance Tests**: ~15 benchmarks
- **Security Tests**: ~50 test cases
- **Total**: ~500+ test cases

## Test Markers

Tests are marked with:
- `@pytest.mark.slow` - Slow tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.performance` - Performance tests
- `@pytest.mark.requires_gpu` - GPU tests

## Troubleshooting

### If Tests Still Fail After Reinstalling PyTorch

1. **Check Python Version**: Should be Python 3.8-3.11
2. **Check Virtual Environment**: Make sure you're in the right environment
3. **Check Dependencies**: Install all from `requirements/base.txt`
4. **Check System**: May need Visual C++ Redistributables on Windows

### Install All Dependencies
```bash
pip install -r requirements/base.txt
pip install pytest pytest-cov pytest-xdist pytest-benchmark
```

### Verify Installation
```bash
python -c "import torch; print(torch.__version__)"
python -c "import pytest; print(pytest.__version__)"
```

## Test Structure

```
tests/
├── unit/              # Unit tests
│   ├── preprocessing/
│   ├── models/
│   ├── decoding/
│   ├── database/
│   └── api/
├── integration/       # Integration tests
├── performance/       # Performance tests
├── security/          # Security tests
├── e2e/              # End-to-end tests
└── fixtures/         # Test fixtures
```

## Next Steps

1. Fix PyTorch installation (Option 1 above)
2. Run: `python -m pytest tests/ -v` to see all tests
3. Run: `python -m pytest tests/ --co -q` to see test count
4. Run specific test suites as needed

Once PyTorch is fixed, all ~500+ tests should run successfully!

