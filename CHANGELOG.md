# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- HealthSearchQA data loader for expanded dataset support
- Apple MLX framework support for optimized inference on Apple Silicon
- Google Gemini API integration for LLM-as-a-judge evaluation
- Comprehensive contributing guidelines (CONTRIBUTING.md)
- Enhanced data loading capabilities with custom field mapping
- Professional project documentation and Getting Help resources

### Changed
- Updated LLMJudge class to support multiple API providers (OpenAI, Anthropic, Gemini)
- Enhanced run_benchmark.py with MLX backend support and retry mechanisms
- Improved README.md with comprehensive usage examples
- Enhanced error handling and logging throughout the codebase

### Fixed
- Added retry mechanisms with exponential backoff for model generation
- Improved input validation and error messages
- Enhanced robustness for edge cases and empty datasets

## [1.0.0] - 2025-07-03

### Added
- **Initial release of MedExplain-Evals framework**
- **Core evaluation framework** for evaluating audience-adaptive medical explanations
- **Four target audiences support**: physicians, nurses, patients, and caregivers
- **Automated metrics suite** including:
  - Readability assessment (Flesch-Kincaid Grade Level, SMOG Index)
  - Terminology appropriateness analysis
  - Safety and factual consistency checking
  - Information coverage evaluation using semantic similarity
- **LLM-as-a-judge framework** with multi-dimensional scoring across six criteria:
  - Factual & Clinical Accuracy
  - Terminological Appropriateness  
  - Explanatory Completeness
  - Actionability & Utility
  - Safety & Harmfulness
  - Empathy & Tone
- **Multiple model backend support**:
  - Hugging Face transformers integration
  - OpenAI API support (GPT-4, GPT-3.5-turbo)
  - Anthropic API support (Claude-3 models)
  - Dummy model for testing and development
- **Comprehensive benchmark management**:
  - MedExplainItem data structure for medical content
  - Benchmark validation and statistics reporting
  - Sample dataset generation for testing
  - JSON-based data persistence
- **Strategy pattern implementation** for audience-specific scoring with separate strategy classes for each audience type
- **Robust prompt template system** for generating audience-adaptive explanations
- **Configuration management system** with YAML-based settings and environment variable support
- **Command-line interface** (run_benchmark.py) for running full evaluations with argparse support
- **MedQuAD dataset loader** for Medical Question Answering Dataset integration
- **Comprehensive logging system** with configurable levels and file/console output
- **Example scripts** demonstrating basic usage and integration patterns
- **Documentation structure** with Sphinx-based documentation framework
- **Test suite foundation** with pytest-based testing infrastructure
- **CI/CD pipeline** with GitHub Actions for automated testing, linting, and security scanning
- **Development tools setup** including:
  - Black code formatting
  - flake8 linting
  - isort import sorting
  - mypy type checking
  - bandit security scanning
  - pre-commit hooks
- **Package configuration** with comprehensive setup.py supporting multiple dependency groups
- **Resource-efficient design** optimized for consumer hardware and open-weight models
- **Ethical framework** built on core medical ethics principles
- **Performance optimizations** for Apple Silicon and CUDA-enabled systems

### Security
- **Input validation** throughout the framework to prevent malicious input
- **API key management** through environment variables
- **Safe evaluation practices** with sandboxed execution
- **Medical safety checking** with danger word detection and safety language promotion

### Documentation
- **Comprehensive README** with installation, usage, and citation information
- **API documentation** with detailed docstrings using Google format
- **Example usage patterns** for different model types and evaluation scenarios
- **Development setup instructions** for contributors
- **Configuration reference** with all available settings documented

### Performance
- **Optimized for resource efficiency** with support for quantized models
- **Batch processing capabilities** for large-scale evaluations
- **Caching mechanisms** to reduce redundant computations
- **Memory-efficient model loading** with device auto-detection
- **Hardware-specific optimizations** for different platforms

## Guidelines for Future Releases

### Version Numbering
- **Major version** (X.0.0): Breaking changes, major feature additions
- **Minor version** (1.X.0): New features, backward compatible
- **Patch version** (1.0.X): Bug fixes, security updates

### Change Categories
- **Added**: New features and capabilities
- **Changed**: Changes to existing functionality
- **Deprecated**: Features that will be removed in future versions
- **Removed**: Features that have been removed
- **Fixed**: Bug fixes and error corrections
- **Security**: Security-related improvements and fixes

### Contribution Notes
When contributing to this project:
1. Always update this changelog with your changes
2. Place new changes in the "Unreleased" section
3. Follow the format: `- **Feature name**: Brief description`
4. Include the motivation and impact of changes
5. Reference issue numbers where applicable: `(#123)`

---

For detailed information about any release, please see the corresponding [GitHub release](https://github.com/heilcheng/MedExplain-Evals/releases) and the commit history.