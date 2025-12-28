# MedExplain-Evals Release Process

This document outlines the complete process for preparing and publishing MedExplain-Evals releases.

## Overview

MedExplain-Evals follows [Semantic Versioning](https://semver.org/) (MAJOR.MINOR.PATCH):
- **MAJOR**: Breaking changes or significant API modifications
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

## Release Types

### Patch Release (1.0.X)
- Bug fixes
- Documentation updates
- Security patches
- Performance improvements without API changes

### Minor Release (1.X.0)
- New features
- New model backend support
- Enhanced evaluation metrics
- New data loaders
- Backward compatible changes

### Major Release (X.0.0)
- Breaking API changes
- Significant architectural changes
- Removal of deprecated features
- Major framework updates

## Pre-Release Checklist

### 1. Code Quality ✅
- [ ] All tests pass (`pytest tests/`)
- [ ] Linting checks pass (`flake8`, `mypy`, `bandit`)
- [ ] Code coverage is adequate
- [ ] No security vulnerabilities detected

### 2. Documentation ✅
- [ ] README.md is up to date
- [ ] API documentation is complete
- [ ] CHANGELOG.md reflects all changes
- [ ] Installation instructions are accurate
- [ ] Examples work with current code

### 3. Dependencies ✅
- [ ] requirements.txt is up to date
- [ ] Dependencies are pinned appropriately
- [ ] No unused dependencies
- [ ] Compatibility tested with supported Python versions

### 4. Testing ✅
- [ ] Unit tests cover new functionality
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Performance regression tests (if applicable)
- [ ] Cross-platform testing (Windows, macOS, Linux)

### 5. Version Management ✅
- [ ] Version numbers updated in all files
- [ ] CHANGELOG.md has [Unreleased] section with changes
- [ ] Git working directory is clean
- [ ] On main/master branch

## Automated Release Preparation

Use the automated release preparation script:

```bash
# Dry run to see what would be changed
python scripts/prepare_release.py --version 1.1.0 --type minor --dry-run

# Actual release preparation
python scripts/prepare_release.py --version 1.1.0 --type minor

# Skip tests for faster preparation (not recommended)
python scripts/prepare_release.py --version 1.1.0 --type minor --skip-tests
```

The script automatically:
- ✅ Validates version format and increment
- ✅ Runs test suite
- ✅ Performs linting checks
- ✅ Updates version in all relevant files
- ✅ Updates CHANGELOG.md
- ✅ Builds package
- ✅ Generates release notes
- ✅ Provides next steps instructions

## Manual Release Steps

### 1. Prepare Release Branch (Optional for Major Releases)
```bash
git checkout -b release/v1.1.0
```

### 2. Update Version Numbers
Update version in:
- `setup.py`: `version="1.1.0"`
- `src/__init__.py`: `__version__ = "1.1.0"`
- `docs/conf.py`: `version = "1.1.0"` and `release = "1.1.0"`

### 3. Update CHANGELOG.md
- Move items from `[Unreleased]` to new version section
- Add release date
- Create new empty `[Unreleased]` section

### 4. Run Release Preparation Script
```bash
python scripts/prepare_release.py --version 1.1.0 --type minor
```

### 5. Review and Commit Changes
```bash
git add .
git commit -m "Prepare release v1.1.0"
```

### 6. Create and Push Tag
```bash
git tag v1.1.0
git push origin v1.1.0
git push origin main
```

### 7. Build Package
```bash
python setup.py sdist bdist_wheel
```

### 8. Test Package Installation
```bash
# Create virtual environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install from built package
pip install dist/medexplain-1.1.0-py3-none-any.whl

# Test basic functionality
python -c "import src; print('Package works!')"
```

## Publishing Release

### 1. GitHub Release
1. Go to [GitHub Releases](https://github.com/heilcheng/MedExplain-Evals/releases)
2. Click "Draft a new release"
3. Choose the tag: `v1.1.0`
4. Release title: `MedExplain-Evals v1.1.0`
5. Use generated release notes from `release_notes_1.1.0.md`
6. Upload built packages from `dist/` directory
7. Publish release

### 2. PyPI Publication (Optional)
```bash
# Install publishing tools
pip install twine

# Check package
twine check dist/*

# Upload to Test PyPI first
twine upload --repository testpypi dist/*

# Test installation from Test PyPI
pip install --index-url https://test.pypi.org/simple/ medexplain-evals

# Upload to production PyPI
twine upload dist/*
```

### 3. Documentation Update
- Ensure Read the Docs builds successfully
- Update any version-specific documentation
- Announce release in documentation

## Post-Release Tasks

### 1. Verify Release
- [ ] GitHub release is published
- [ ] Package installs correctly
- [ ] Documentation builds successfully
- [ ] All CI/CD pipelines pass

### 2. Communication
- [ ] Update project README badges if needed
- [ ] Announce on relevant channels
- [ ] Update any external documentation

### 3. Cleanup
- [ ] Delete release branch (if used)
- [ ] Archive old releases if needed
- [ ] Update project roadmap

## Emergency Patch Process

For critical bug fixes or security issues:

1. **Immediate Fix**:
   ```bash
   git checkout main
   git checkout -b hotfix/v1.0.1
   # Make minimal fix
   git commit -m "Fix critical issue"
   ```

2. **Fast-Track Release**:
   ```bash
   python scripts/prepare_release.py --version 1.0.1 --type patch --skip-tests
   git push origin hotfix/v1.0.1
   # Create PR and merge immediately
   ```

3. **Emergency Publication**:
   - Skip extensive testing for critical security fixes
   - Document the urgency in release notes
   - Follow up with comprehensive testing

## Release Schedule

### Regular Schedule
- **Patch releases**: As needed (bug fixes)
- **Minor releases**: Monthly or quarterly
- **Major releases**: Annually or when breaking changes accumulate

### Release Windows
- Avoid releases during holidays
- Prefer Tuesday-Thursday releases
- Allow time for post-release monitoring

## Rollback Procedure

If a release has critical issues:

### 1. Immediate Response
```bash
# Revert the problematic tag
git tag -d v1.1.0
git push origin :refs/tags/v1.1.0

# Create hotfix
git checkout v1.0.0  # Last known good version
git checkout -b hotfix/v1.1.1
```

### 2. Communication
- Announce the issue immediately
- Provide workarounds if available
- Give timeline for fix

### 3. Fix and Re-release
- Address the issue
- Increment version (e.g., 1.1.0 → 1.1.1)
- Follow expedited release process

## Tools and Scripts

### Release Preparation Script
`scripts/prepare_release.py` - Automates most release tasks

### Validation Scripts
- `scripts/validate_package.py` - Package validation
- `scripts/check_dependencies.py` - Dependency validation

### CI/CD Integration
GitHub Actions workflows automatically:
- Run tests on pull requests
- Build packages on tags
- Deploy documentation on releases

## Troubleshooting

### Common Issues

#### Version Conflicts
```bash
# Check current versions
grep -r "version.*=" setup.py src/__init__.py docs/conf.py
```

#### Test Failures
```bash
# Run specific test
pytest tests/test_specific.py -v

# Run with coverage
pytest --cov=src tests/
```

#### Package Build Issues
```bash
# Clean previous builds
rm -rf build/ dist/ *.egg-info/

# Rebuild
python setup.py clean --all
python setup.py sdist bdist_wheel
```

#### Documentation Build Issues
```bash
cd docs/
make clean
make html
```

## Security Considerations

### Pre-Release Security Checks
- [ ] Run security scanning (`bandit`)
- [ ] Check for exposed secrets
- [ ] Validate dependencies for vulnerabilities
- [ ] Review code for security best practices

### Secure Release Process
- Use signed commits for releases
- Verify package integrity
- Use secure channels for communication
- Document security fixes prominently

## Version History Template

Use this template for CHANGELOG.md entries:

```markdown
## [1.1.0] - 2025-01-XX

### Added
- New HealthSearchQA data loader for expanded dataset support
- Apple MLX framework support for optimized inference on Apple Silicon
- Google Gemini API integration for LLM-as-a-judge evaluation
- Enhanced data loading capabilities with custom field mapping
- Comprehensive LLM-as-a-Judge validation framework with three-part strategy

### Changed
- Updated LLMJudge class to support multiple API providers
- Enhanced run_benchmark.py with MLX backend support and retry mechanisms
- Improved error handling and logging throughout the codebase

### Fixed
- Added retry mechanisms with exponential backoff for model generation
- Improved input validation and error messages
- Enhanced robustness for edge cases and empty datasets

### Security
- Updated dependency versions to address security vulnerabilities
- Improved input validation to prevent injection attacks
```

---

For questions about the release process, please contact the maintainers or open an issue on GitHub.