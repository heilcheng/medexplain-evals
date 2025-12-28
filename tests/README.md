# MedExplain-Evals Tests

This directory contains comprehensive test suites for the MedExplain-Evals framework.

## Test Structure

```
tests/
├── unit/                 # Unit tests for individual components
├── integration/         # Integration tests for full workflows
├── evaluation/          # Tests for evaluation metrics
├── data/               # Test data and fixtures
└── benchmarks/         # Performance benchmarks
```

## Running Tests

### All Tests
```bash
pytest tests/
```

### Specific Test Categories
```bash
# Unit tests only
pytest tests/unit/

# Integration tests only  
pytest tests/integration/

# Evaluation tests only
pytest tests/evaluation/
```

### With Coverage
```bash
pytest --cov=src tests/
```

## Test Categories

### Unit Tests (`tests/unit/`)
- `test_benchmark.py` - MedExplain class functionality
- `test_evaluator.py` - Evaluation metrics and scoring
- `test_prompt_templates.py` - Prompt formatting and parsing
- `test_utils.py` - Utility functions

### Integration Tests (`tests/integration/`)
- `test_full_pipeline.py` - End-to-end benchmark execution
- `test_data_loading.py` - Data loading and processing
- `test_model_integration.py` - Model integration workflows

### Evaluation Tests (`tests/evaluation/`)
- `test_metrics.py` - Automated metrics validation
- `test_llm_judge.py` - LLM-as-a-judge functionality
- `test_scoring.py` - Scoring algorithm accuracy

## Test Data

Test fixtures are located in `tests/data/` and include:
- Sample medical content
- Expected evaluation outputs
- Reference explanations for validation

## Continuous Integration

Tests are automatically run on:
- Pull requests
- Main branch commits
- Release candidates

## Adding Tests

When adding new functionality:

1. Write unit tests for individual functions
2. Add integration tests for workflows
3. Include edge cases and error conditions
4. Update test documentation

## Test Configuration

Test settings are configured in:
- `pytest.ini` - Pytest configuration
- `tests/conftest.py` - Test fixtures and setup
- `.github/workflows/test.yml` - CI/CD pipeline