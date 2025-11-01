"""
Comprehensive project health check script.
Checks for syntax errors, import errors, and other issues.
"""

import os
import sys
import ast
import importlib
from pathlib import Path
import subprocess

class Colors:
    """ANSI color codes for terminal output."""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")

def print_success(text):
    """Print success message."""
    print(f"{Colors.GREEN}[OK] {text}{Colors.RESET}")

def print_error(text):
    """Print error message."""
    print(f"{Colors.RED}[ERROR] {text}{Colors.RESET}")

def print_warning(text):
    """Print warning message."""
    print(f"{Colors.YELLOW}! {text}{Colors.RESET}")

def check_syntax_errors():
    """Check all Python files for syntax errors."""
    print_header("1. Checking for Syntax Errors")
    
    errors = []
    checked = 0
    
    # Find all Python files
    for root, dirs, files in os.walk('.'):
        # Skip common directories
        if any(skip in root for skip in ['__pycache__', '.git', 'venv', 'node_modules', '.pytest_cache']):
            continue
            
        for file in files:
            if file.endswith('.py'):
                filepath = Path(root) / file
                checked += 1
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        ast.parse(content, filename=str(filepath))
                except SyntaxError as e:
                    errors.append({
                        'file': str(filepath),
                        'line': e.lineno,
                        'error': str(e.msg)
                    })
                except Exception as e:
                    errors.append({
                        'file': str(filepath),
                        'line': 'N/A',
                        'error': str(e)
                    })
    
    if errors:
        print_error(f"Found {len(errors)} syntax errors:")
        for err in errors:
            print(f"  File: {err['file']}")
            print(f"  Line: {err['line']}")
            print(f"  Error: {err['error']}\n")
    else:
        print_success(f"No syntax errors found in {checked} Python files")
    
    return len(errors) == 0

def check_imports():
    """Check if key modules can be imported."""
    print_header("2. Checking Module Imports")
    
    modules_to_check = [
        ('preprocessing.audio_processing', 'AudioProcessor'),
        ('preprocessing.text_cleaning', 'VietnameseTextNormalizer'),
        ('models.asr_base', 'ASRModel'),
        ('models.enhanced_asr', 'EnhancedASRModel'),
        ('database.db_utils', 'ASRDatabase'),
        ('decoding.beam_search', 'BeamSearchDecoder'),
        ('decoding.lm_decoder', 'LMBeamSearchDecoder'),
        ('utils.metrics', 'calculate_wer'),
    ]
    
    import_errors = []
    success_count = 0
    
    # Add current directory to path
    if '.' not in sys.path:
        sys.path.insert(0, '.')
    
    for module_name, class_name in modules_to_check:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, class_name):
                print_success(f"{module_name}.{class_name}")
                success_count += 1
            else:
                print_warning(f"{module_name} imported but {class_name} not found")
        except Exception as e:
            print_error(f"{module_name}: {str(e)[:80]}")
            import_errors.append({
                'module': module_name,
                'error': str(e)
            })
    
    print(f"\nSuccessfully imported: {success_count}/{len(modules_to_check)}")
    
    if import_errors:
        print(f"\nImport errors details:")
        for err in import_errors:
            print(f"\n  Module: {err['module']}")
            print(f"  Error: {err['error']}")
    
    return len(import_errors) == 0

def check_dependencies():
    """Check if required dependencies are installed."""
    print_header("3. Checking Dependencies")
    
    dependencies = [
        'torch',
        'torchaudio',
        'numpy',
        'librosa',
        'soundfile',
        'fastapi',
        'pytest',
        'pandas',
        'sqlite3',
    ]
    
    missing = []
    installed = []
    
    for dep in dependencies:
        try:
            if dep == 'sqlite3':
                import sqlite3
            else:
                __import__(dep)
            print_success(f"{dep}")
            installed.append(dep)
        except ImportError:
            print_error(f"{dep} - NOT INSTALLED")
            missing.append(dep)
    
    print(f"\nInstalled: {len(installed)}/{len(dependencies)}")
    
    if missing:
        print_warning(f"\nMissing dependencies: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
    
    return len(missing) == 0

def check_file_structure():
    """Check if expected files and directories exist."""
    print_header("4. Checking File Structure")
    
    expected_dirs = [
        'preprocessing',
        'models',
        'decoding',
        'database',
        'api',
        'tests',
        'configs',
        'utils',
    ]
    
    expected_files = [
        'requirements/base.txt',
        'tests/conftest.py',
        'tests/pytest.ini',
    ]
    
    missing_dirs = []
    missing_files = []
    
    for directory in expected_dirs:
        if Path(directory).exists():
            print_success(f"Directory: {directory}/")
        else:
            print_error(f"Directory: {directory}/ - MISSING")
            missing_dirs.append(directory)
    
    print()
    
    for filepath in expected_files:
        if Path(filepath).exists():
            print_success(f"File: {filepath}")
        else:
            print_warning(f"File: {filepath} - MISSING")
            missing_files.append(filepath)
    
    return len(missing_dirs) == 0

def check_pytorch_dll():
    """Check if PyTorch can load DLLs properly."""
    print_header("5. Checking PyTorch DLL Loading")
    
    try:
        import torch
        print_success(f"PyTorch version: {torch.__version__}")
        
        # Try to create a tensor
        try:
            tensor = torch.randn(2, 3)
            print_success("PyTorch tensor creation works")
            
            # Check CUDA availability
            if torch.cuda.is_available():
                print_success(f"CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                print_warning("CUDA not available (CPU mode)")
            
            return True
        except Exception as e:
            print_error(f"PyTorch tensor creation failed: {str(e)}")
            print_warning("This is the DLL loading issue affecting tests")
            return False
    except ImportError as e:
        print_error(f"PyTorch import failed: {str(e)}")
        return False

def check_database():
    """Check if database can be initialized."""
    print_header("6. Checking Database")
    
    try:
        from database.db_utils import ASRDatabase
        import tempfile
        
        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        db = ASRDatabase(db_path)
        print_success("Database initialization works")
        
        # Try basic operation
        audio_id = db.add_audio_file(
            file_path="test.wav",
            filename="test.wav",
            duration=1.0,
            sample_rate=16000
        )
        
        if audio_id:
            print_success("Database operations work")
        
        # Cleanup
        Path(db_path).unlink()
        
        return True
    except Exception as e:
        print_error(f"Database check failed: {str(e)}")
        return False

def check_api():
    """Check if API can be initialized."""
    print_header("7. Checking API")
    
    try:
        from api.app import app
        print_success("API application imported successfully")
        
        # Check if FastAPI app has routes
        if hasattr(app, 'routes'):
            route_count = len(app.routes)
            print_success(f"API has {route_count} routes")
        
        return True
    except Exception as e:
        print_error(f"API check failed: {str(e)}")
        return False

def run_linting():
    """Run flake8 linting if available."""
    print_header("8. Running Code Quality Checks")
    
    try:
        result = subprocess.run(
            ['python', '-m', 'flake8', '.', '--count', '--select=E9,F63,F7,F82', '--show-source', '--statistics'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print_success("No critical linting errors found")
            return True
        else:
            print_error("Linting errors found:")
            print(result.stdout)
            return False
    except FileNotFoundError:
        print_warning("flake8 not installed - skipping linting")
        print("Install with: pip install flake8")
        return True
    except Exception as e:
        print_warning(f"Linting check skipped: {str(e)}")
        return True

def generate_summary(results):
    """Generate summary report."""
    print_header("Summary Report")
    
    total = len(results)
    passed = sum(1 for r in results.values() if r)
    failed = total - passed
    
    print(f"Total checks: {total}")
    print(f"Passed: {Colors.GREEN}{passed}{Colors.RESET}")
    print(f"Failed: {Colors.RED}{failed}{Colors.RESET}")
    print()
    
    for check_name, status in results.items():
        status_icon = "[OK]" if status else "[FAIL]"
        color = Colors.GREEN if status else Colors.RED
        print(f"  {color}{status_icon}{Colors.RESET} {check_name}")
    
    print()
    
    if failed == 0:
        print(f"{Colors.GREEN}{Colors.BOLD}All checks passed!{Colors.RESET}")
    else:
        print(f"{Colors.YELLOW}{Colors.BOLD}Some checks failed. See details above.{Colors.RESET}")
    
    return failed == 0

def main():
    """Main function."""
    print(f"\n{Colors.BOLD}AI2Text Project Health Check{Colors.RESET}")
    print(f"Checking project for errors and issues...\n")
    
    results = {}
    
    # Run all checks
    results["Syntax Check"] = check_syntax_errors()
    results["Module Imports"] = check_imports()
    results["Dependencies"] = check_dependencies()
    results["File Structure"] = check_file_structure()
    results["PyTorch DLL"] = check_pytorch_dll()
    results["Database"] = check_database()
    results["API"] = check_api()
    results["Code Quality"] = run_linting()
    
    # Generate summary
    all_passed = generate_summary(results)
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()

