# Testing Guide for AI2Text Project

## Quick Start

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/preprocessing/test_audio_processing.py

# Run with coverage
pytest --cov=preprocessing --cov=models --cov=decoding --cov-report=html

# Run only fast tests (exclude slow/integration)
pytest -m "not slow"

# Run in parallel
pytest -n auto
```

### Test Structure

```
tests/
├── unit/              # Unit tests (60% of tests)
│   ├── preprocessing/ # Audio and text preprocessing
│   ├── models/        # ASR models
│   ├── decoding/      # Decoding algorithms
│   ├── database/      # Database operations
│   ├── nlp/           # NLP components
│   ├── training/      # Training pipeline
│   ├── api/           # API endpoints
│   └── utils/         # Utility functions
├── integration/       # Integration tests (30%)
├── e2e/               # End-to-end tests (10%)
├── performance/      # Performance benchmarks
├── security/          # Security tests
└── fixtures/          # Test fixtures and utilities
```

## Test Categories

### Unit Tests
- **Location**: `tests/unit/`
- **Purpose**: Test individual functions and classes in isolation
- **Speed**: Fast (<1 second each)
- **Coverage Target**: 80%+

### Integration Tests
- **Location**: `tests/integration/`
- **Purpose**: Test component interactions
- **Speed**: Medium (1-10 seconds each)
- **Marked with**: `@pytest.mark.integration`

### End-to-End Tests
- **Location**: `tests/e2e/`
- **Purpose**: Test complete user workflows
- **Speed**: Slow (>10 seconds each)
- **Marked with**: `@pytest.mark.e2e`

## Important Test Files

### Critical Tests (Must Pass)
1. `tests/unit/preprocessing/test_audio_processing.py` - Audio processing core
2. `tests/unit/models/test_asr_base.py` - Base model architecture
3. `tests/unit/api/test_app.py` - API endpoints
4. `tests/unit/database/test_db_utils.py` - Database operations

### High Priority Tests
1. `tests/unit/decoding/test_beam_search.py` - Decoding algorithms
2. `tests/unit/decoding/test_lm_decoder.py` - Language model integration
3. `tests/integration/test_api_integration.py` - API workflows

## Running Specific Test Suites

### By Module
```bash
# Preprocessing tests
pytest tests/unit/preprocessing/

# Model tests
pytest tests/unit/models/

# API tests
pytest tests/unit/api/
```

### By Markers
```bash
# Only fast tests
pytest -m "not slow"

# Only integration tests
pytest -m integration

# Only GPU tests (if available)
pytest -m requires_gpu
```

## Coverage Goals

| Module | Target Coverage |
|--------|----------------|
| Preprocessing | 85% |
| Models | 80% |
| Decoding | 80% |
| Database | 85% |
| API | 75% |

## Test Fixtures

Common fixtures are defined in `tests/conftest.py`:

- `temp_db`: Temporary database for testing
- `temp_dir`: Temporary directory
- `sample_audio_file`: Sample audio file
- `audio_processor`: AudioProcessor instance
- `audio_augmenter`: AudioAugmenter instance
- `text_normalizer`: VietnameseTextNormalizer instance
- `tokenizer`: Tokenizer instance
- `asr_model`: ASR model instance

## Writing New Tests

### Test Naming
- Files: `test_*.py`
- Functions: `test_*`
- Classes: `Test*`

### Test Structure
```python
def test_function_name_scenario():
    """Test description."""
    # Arrange
    setup_data = create_test_data()
    
    # Act
    result = function_to_test(setup_data)
    
    # Assert
    assert result == expected_value
```

### Using Fixtures
```python
def test_with_fixture(audio_processor, sample_audio_file):
    """Test using fixtures."""
    audio, sr = audio_processor.load_audio(sample_audio_file[0])
    assert sr == 16000
```

### Parametrized Tests
```python
@pytest.mark.parametrize("input_value,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_multiply_by_two(input_value, expected):
    assert input_value * 2 == expected
```

## Debugging Tests

### Run with Verbose Output
```bash
pytest -v tests/unit/preprocessing/test_audio_processing.py
```

### Run Single Test
```bash
pytest tests/unit/preprocessing/test_audio_processing.py::TestAudioProcessor::test_load_audio
```

### Print Output
```bash
pytest -s  # Show print statements
```

### Debug on Failure
```bash
pytest --pdb  # Drop into debugger on failure
```

## Continuous Integration

Tests run automatically on:
- Every commit (fast tests only)
- Pull requests (full test suite)
- Main branch merges (full suite + integration)

## Best Practices

1. **Isolation**: Each test should be independent
2. **Speed**: Keep unit tests fast (<1s)
3. **Clarity**: Use descriptive test names
4. **Coverage**: Aim for 80%+ coverage on critical paths
5. **Maintainability**: Keep tests simple and readable

## Common Issues

### Import Errors
- Ensure you're running from project root
- Check that modules are in Python path

### Missing Dependencies
- Install test requirements: `pip install -r requirements/base.txt`
- Ensure pytest is installed: `pip install pytest pytest-cov`

### Database Errors
- Tests use temporary databases (automatically cleaned up)
- If issues persist, check database fixtures in `conftest.py`

### GPU Tests Failing
- GPU tests are marked with `@pytest.mark.requires_gpu`
- Skip if GPU not available: `pytest -m "not requires_gpu"`

## Resources

- [Testing Roadmap](./TESTING_ROADMAP.md) - Comprehensive testing strategy
- [Pytest Documentation](https://docs.pytest.org/)
- [Python Testing Guide](https://realpython.com/python-testing/)
