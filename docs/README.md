# MedExplain-Evals Documentation

This directory contains comprehensive documentation for the MedExplain-Evals framework, built with Sphinx.

## Building the Documentation

### Prerequisites

Install documentation dependencies:

```bash
pip install sphinx sphinx-rtd-theme myst-parser sphinx-autodoc-typehints
```

### Building HTML Documentation

To build the HTML documentation locally:

```bash
cd docs
make html
```

The built documentation will be available in `_build/html/index.html`.

### Building for Development

For live-reloading during development:

```bash
cd docs
pip install sphinx-autobuild
make livehtml
```

This will start a local server at `http://localhost:8000` that automatically rebuilds when files change.

### Other Build Targets

- `make linkcheck` - Check for broken links
- `make coverage` - Generate documentation coverage report
- `make clean` - Clean build directory
- `make clean-all` - Clean all generated files

## Documentation Structure

```
docs/
├── conf.py              # Sphinx configuration
├── index.rst            # Main documentation index
├── installation.rst     # Installation guide
├── quickstart.rst       # Quick start guide
├── api/                 # API documentation
│   └── index.rst        # API reference index
├── _build/              # Built documentation (generated)
├── _static/             # Static files for documentation
└── _templates/          # Custom templates
```

## Key Concepts

### Audience-Adaptive Explanations

MedExplain-Evals evaluates how well models can tailor medical explanations for:

1. **Physicians** - Technical, evidence-based explanations
2. **Nurses** - Practical care implications and monitoring
3. **Patients** - Simple, empathetic, jargon-free language
4. **Caregivers** - Concrete tasks and warning signs

### Evaluation Dimensions

1. **Factual & Clinical Accuracy**
2. **Terminological Appropriateness**
3. **Explanatory Completeness**
4. **Actionability & Utility**
5. **Safety & Harmfulness**
6. **Empathy & Tone**

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Run basic example: `python examples/basic_usage.py`
3. See the API documentation for detailed usage

## Contributing to Documentation

When adding new modules or functions:

1. Add proper docstrings following Google/NumPy style
2. Update the API documentation if needed
3. Rebuild documentation to check for issues
4. Test that all links work with `make linkcheck`

## Citation

```bibtex
@article{medexplain-evals-2025,
  title={MedExplain-Evals: A Resource-Efficient Benchmark for Evaluating Audience-Adaptive Explanation Quality in Medical Large Language Models},
  author={[Author Names]},
  journal={[Journal Name]},
  year={2025}
}
```