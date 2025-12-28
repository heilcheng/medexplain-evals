# MedExplain-Evals Scripts

This directory contains utility scripts for MedExplain-Evals development, maintenance, and release management.

## Available Scripts

### 🚀 Release Management

#### `prepare_release.py`
Comprehensive release preparation script that automates the release process.

**Features:**
- Version validation and updating across all files
- Automated test execution
- Code quality checks (linting, type checking)
- CHANGELOG.md updates
- Package building
- Release notes generation

**Usage:**
```bash
# Dry run to see what would be changed
python scripts/prepare_release.py --version 1.1.0 --type minor --dry-run

# Prepare a minor release
python scripts/prepare_release.py --version 1.1.0 --type minor

# Prepare a patch release (skipping tests for speed)
python scripts/prepare_release.py --version 1.0.1 --type patch --skip-tests

# Verbose output
python scripts/prepare_release.py --version 1.1.0 --type minor --verbose
```

**What it does:**
1. ✅ Validates version format and increment
2. ✅ Runs comprehensive test suite
3. ✅ Performs code quality checks
4. ✅ Updates version numbers in:
   - `setup.py`
   - `src/__init__.py`
   - `docs/conf.py`
5. ✅ Updates `CHANGELOG.md`
6. ✅ Builds distribution packages
7. ✅ Generates release notes
8. ✅ Provides next steps guidance

#### `validate_release.py`
Validates that a release is ready for publication.

**Features:**
- Module import validation
- Basic functionality testing
- Configuration system checks
- Version consistency validation
- Example script validation
- Package installability testing

**Usage:**
```bash
# Basic validation
python scripts/validate_release.py

# Validate specific package
python scripts/validate_release.py --package-path dist/medexplain-1.1.0-py3-none-any.whl

# Verbose output
python scripts/validate_release.py --verbose
```

**Validation checks:**
1. ✅ All core modules can be imported
2. ✅ Basic functionality works (MedExplain, MedExplainEvaluator)
3. ✅ Configuration system loads correctly
4. ✅ Version numbers are consistent across files
5. ✅ Example scripts have valid syntax
6. ✅ Package can be installed in clean environment

### 📊 Data Processing

#### `process_datasets.py`
Processes and validates external medical datasets for use with MedExplain-Evals.

**Usage:**
```bash
# Process a dataset
python scripts/process_datasets.py --input data/raw_dataset.json --output data/processed_dataset.json

# Validate dataset format
python scripts/process_datasets.py --validate data/dataset.json
```

## Release Workflow

The complete release workflow using these scripts:

### 1. Prepare Release
```bash
# Run preparation with dry-run first
python scripts/prepare_release.py --version 1.1.0 --type minor --dry-run

# If everything looks good, run actual preparation
python scripts/prepare_release.py --version 1.1.0 --type minor
```

### 2. Validate Release
```bash
# Validate the prepared release
python scripts/validate_release.py

# Test specific package if built
python scripts/validate_release.py --package-path dist/medexplain-1.1.0-py3-none-any.whl
```

### 3. Complete Release
Follow the instructions provided by `prepare_release.py`:
```bash
# Commit changes
git add setup.py src/__init__.py docs/conf.py CHANGELOG.md
git commit -m "Prepare release v1.1.0"

# Create and push tag
git tag v1.1.0
git push origin v1.1.0
git push origin main

# Create GitHub release using generated release notes
```

## Script Dependencies

### Common Requirements
- Python 3.8+
- All MedExplain-Evals dependencies installed

### Additional Tools for Release Scripts
```bash
# For comprehensive release preparation
pip install pytest flake8 mypy bandit twine

# For package building
pip install wheel setuptools
```

## Development Guidelines

### Adding New Scripts

When adding new scripts to this directory:

1. **Make scripts executable:**
   ```bash
   chmod +x scripts/new_script.py
   ```

2. **Include shebang:**
   ```python
   #!/usr/bin/env python3
   ```

3. **Add comprehensive docstring:**
   ```python
   """
   Brief description of the script.
   
   Detailed description with usage examples.
   """
   ```

4. **Use argparse for CLI:**
   ```python
   import argparse
   
   def main():
       parser = argparse.ArgumentParser(description="Script description")
       # Add arguments
       args = parser.parse_args()
   ```

5. **Include logging:**
   ```python
   import logging
   
   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger(__name__)
   ```

6. **Update this README** with script documentation

### Script Structure Template

```python
#!/usr/bin/env python3
"""
Script purpose and description.

Usage examples and documentation.
"""

import argparse
import logging
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScriptClass:
    """Main script functionality"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
    
    def main_functionality(self) -> bool:
        """Main script logic"""
        return True


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Script description")
    parser.add_argument("--option", help="Option description")
    args = parser.parse_args()
    
    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Run script
    script = ScriptClass(project_root)
    success = script.main_functionality()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
```

## Troubleshooting

### Common Issues

#### Permission Errors
```bash
# Make scripts executable
chmod +x scripts/*.py
```

#### Import Errors
```bash
# Ensure you're in the project root
cd /path/to/MedExplain-Evals
python scripts/script_name.py
```

#### Missing Dependencies
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest flake8 mypy bandit twine wheel
```

#### Test Failures During Release
```bash
# Run tests manually to see detailed output
pytest tests/ -v

# Run specific test
pytest tests/test_specific.py -v
```

#### Version Update Issues
```bash
# Check current versions
grep -r "version.*=" setup.py src/__init__.py docs/conf.py

# Manually update if needed
```

### Getting Help

- Check script help: `python scripts/script_name.py --help`
- Review logs for detailed error information
- Ensure all dependencies are installed
- Verify you're running from the project root directory

For more information, see:
- [RELEASE_PROCESS.md](../RELEASE_PROCESS.md) - Complete release documentation
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Development guidelines
- [README.md](../README.md) - Project overview