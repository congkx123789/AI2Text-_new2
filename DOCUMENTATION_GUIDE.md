# Code Documentation Guide

## 📚 Documentation Added to All Code Files

Comprehensive documentation has been added to all code files including:

### ✅ Database Module (`database/db_utils.py`)
- **Module-level docstring**: Overview, examples, usage
- **Class docstring**: Purpose, attributes, examples
- **Function docstrings**: Detailed descriptions, parameters, returns, examples, notes
- **Inline comments**: Explanations for complex logic

### ✅ Preprocessing Modules
- **Audio processing** (`preprocessing/audio_processing.py`): All methods documented
- **Text cleaning** (`preprocessing/text_cleaning.py`): Vietnamese-specific features explained

### ✅ Model Architecture (`models/asr_base.py`)
- **Class documentation**: Architecture details, purpose of each component
- **Method documentation**: Forward passes, parameter explanations

### ✅ Training Pipeline
- **Dataset** (`training/dataset.py`): Data loading and processing explained
- **Training** (`training/train.py`): Training loop, optimization details
- **Evaluation** (`training/evaluate.py`): Evaluation metrics and inference

### ✅ Utilities
- **Metrics** (`utils/metrics.py`): Algorithm explanations, examples
- **Logger** (`utils/logger.py`): Setup and configuration

## 📝 Documentation Standards

All functions now include:

1. **Purpose**: What the function does
2. **Parameters**: All arguments with types and descriptions
3. **Returns**: Return value type and description
4. **Examples**: Usage examples where helpful
5. **Notes**: Important warnings or implementation details
6. **Performance**: Performance characteristics if relevant

## 🎯 Example Documentation Format

```python
def function_name(param1: type, param2: type = default) -> return_type:
    """
    Brief description of what the function does.
    
    More detailed explanation of the function's purpose, behavior,
    and any important implementation details.
    
    Args:
        param1 (type): Description of parameter 1
        param2 (type, optional): Description of parameter 2. 
                                 Defaults to default_value.
    
    Returns:
        return_type: Description of return value
    
    Example:
        >>> result = function_name("value1", param2="value2")
        >>> print(result)
        output
    
    Note:
        Important implementation detail or warning.
    
    Raises:
        ExceptionType: When this exception is raised and why
    """
    # Implementation with inline comments for complex logic
    pass
```

## 📖 How to Read Documentation

1. **Module-level**: Start here for overview and examples
2. **Class-level**: Understand the class purpose and structure
3. **Function-level**: Detailed usage for each method
4. **Inline comments**: Explanation of complex code sections

## 🔍 Finding Documentation

All documentation is available via:
- Python `help()` function: `help(ASRDatabase.add_audio_file)`
- IDE tooltips: Hover over function names
- Docstrings: View in source code
- Interactive help: Use `?` in IPython/Jupyter

---

**All code is now fully documented!** 🎉

