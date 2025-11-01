"""
List all available tests without running them.
This script helps identify test files even if imports fail.
"""

import os
from pathlib import Path

def find_test_files(root_dir="tests"):
    """Find all test files in the project."""
    test_files = []
    root = Path(root_dir)
    
    if not root.exists():
        print(f"Directory {root_dir} does not exist")
        return test_files
    
    for test_file in root.rglob("test_*.py"):
        # Skip __pycache__
        if "__pycache__" in str(test_file):
            continue
        
        relative_path = test_file.relative_to(root)
        test_files.append(str(relative_path))
    
    return sorted(test_files)

def count_test_functions(file_path):
    """Count test functions in a file (simple line-based counting)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Count lines with "def test_" 
            count = content.count("def test_")
            return count
    except Exception:
        return 0

def main():
    """Main function to list all tests."""
    print("=" * 70)
    print("AI2Text Project - Test Files Summary")
    print("=" * 70)
    print()
    
    root = Path("tests")
    if not root.exists():
        print("ERROR: tests/ directory not found!")
        return
    
    test_files = find_test_files("tests")
    
    if not test_files:
        print("No test files found!")
        return
    
    print(f"Found {len(test_files)} test files:\n")
    
    categories = {}
    total_tests = 0
    
    for test_file in test_files:
        file_path = root / test_file
        
        # Categorize
        parts = Path(test_file).parts
        if len(parts) > 1:
            category = parts[0]
        else:
            category = "root"
        
        if category not in categories:
            categories[category] = []
        
        test_count = count_test_functions(file_path)
        total_tests += test_count
        
        categories[category].append({
            'file': test_file,
            'count': test_count
        })
    
    # Print by category
    for category in sorted(categories.keys()):
        print(f"\n{category.upper().replace('_', ' ')}:")
        print("-" * 70)
        
        for item in categories[category]:
            file_name = item['file']
            count = item['count']
            status = "[OK]" if count > 0 else "[  ]"
            print(f"  {status} {file_name:<50} ({count} tests)")
    
    print("\n" + "=" * 70)
    print(f"Total: {len(test_files)} test files, ~{total_tests} test functions")
    print("=" * 70)
    print()
    
    # Print categories summary
    print("Test Categories:")
    for category in sorted(categories.keys()):
        files_in_cat = len(categories[category])
        tests_in_cat = sum(item['count'] for item in categories[category])
        print(f"  - {category}: {files_in_cat} files, ~{tests_in_cat} tests")
    
    print()
    print("To run tests:")
    print("  1. Fix PyTorch DLL issue (see RUN_TESTS.md)")
    print("  2. Run: python -m pytest tests/ -v")
    print("  3. Run specific file: python -m pytest tests/<path> -v")

if __name__ == "__main__":
    main()

