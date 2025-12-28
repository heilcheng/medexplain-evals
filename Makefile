.PHONY: help install dev test lint format type-check security docs clean build docker run \
        curate evaluate analyze report smoke-test estimate validate-env

# Default target
help:
	@echo "MedExplain-Evals Development Commands"
	@echo "==================================="
	@echo ""
	@echo "Setup:"
	@echo "  install      Install production dependencies"
	@echo "  dev          Install development dependencies"
	@echo "  pre-commit   Install pre-commit hooks"
	@echo "  validate-env Validate environment setup"
	@echo ""
	@echo "Quality:"
	@echo "  lint         Run Ruff linter"
	@echo "  format       Format code with Ruff"
	@echo "  type-check   Run mypy type checking"
	@echo "  security     Run security scans (bandit, pip-audit)"
	@echo "  check        Run all quality checks"
	@echo ""
	@echo "Testing:"
	@echo "  test         Run all tests"
	@echo "  test-fast    Run tests without coverage"
	@echo "  test-cov     Run tests with HTML coverage report"
	@echo "  test-watch   Run tests in watch mode"
	@echo "  smoke-test   Run quick benchmark smoke test"
	@echo ""
	@echo "Benchmark:"
	@echo "  curate       Curate benchmark dataset"
	@echo "  evaluate     Run full evaluation"
	@echo "  analyze      Analyze evaluation results"
	@echo "  report       Generate HTML/Markdown reports"
	@echo "  estimate     Estimate API costs"
	@echo ""
	@echo "Documentation:"
	@echo "  docs         Build documentation"
	@echo "  docs-serve   Serve documentation locally"
	@echo ""
	@echo "Build:"
	@echo "  build        Build Python package"
	@echo "  docker       Build Docker image"
	@echo "  docker-up    Start Docker services"
	@echo "  clean        Clean build artifacts"
	@echo ""
	@echo "Run:"
	@echo "  run          Run MedExplain-Evals CLI"
	@echo "  info         Show configuration info"

# Python/UV detection
PYTHON := python3
UV := uv

# Default models for evaluation (override with: make evaluate MODELS="gpt-5.1 claude-opus-4.5")
MODELS ?= gpt-4o

# Installation
install:
	$(UV) sync

dev:
	$(UV) sync --dev --all-extras

pre-commit:
	$(UV) run pre-commit install
	$(UV) run pre-commit install --hook-type commit-msg

# Quality checks
lint:
	$(UV) run ruff check src/ tests/

lint-fix:
	$(UV) run ruff check --fix src/ tests/

format:
	$(UV) run ruff format src/ tests/

format-check:
	$(UV) run ruff format --check src/ tests/

type-check:
	$(UV) run mypy src/ --install-types --non-interactive

security:
	$(UV) run bandit -r src/ -c pyproject.toml
	$(UV) run pip-audit

check: lint format-check type-check security

# Testing
test:
	$(UV) run pytest tests/ -v --cov=src --cov-report=term-missing

test-fast:
	$(UV) run pytest tests/ -v -x --tb=short

test-cov:
	$(UV) run pytest tests/ -v --cov=src --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

test-watch:
	$(UV) run ptw --runner "pytest -v -x --tb=short"

test-unit:
	$(UV) run pytest tests/ -v -m unit

test-integration:
	$(UV) run pytest tests/ -v -m integration

# Documentation
docs:
	$(UV) run mkdocs build --strict

docs-serve:
	$(UV) run mkdocs serve --dev-addr localhost:8000

# Build
build: clean
	$(UV) build

build-check:
	$(UV) run twine check dist/*

clean:
	rm -rf dist/ build/ *.egg-info/
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf htmlcov/ .coverage coverage.xml
	rm -rf site/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Docker
docker:
	docker build -t medexplain-evals:2.0 .

docker-run:
	docker run --rm -it \
		-e OPENAI_API_KEY=$(OPENAI_API_KEY) \
		-e ANTHROPIC_API_KEY=$(ANTHROPIC_API_KEY) \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/results:/app/results \
		medexplain-evals:2.0

docker-up:
	docker-compose up -d medexplain

docker-up-gpu:
	docker-compose --profile gpu up -d

docker-down:
	docker-compose down

# ==============================================================================
# Benchmark Pipeline
# ==============================================================================

# Validate environment
validate-env:
	$(PYTHON) scripts/validate_environment.py --verbose

# Curate dataset
curate:
	$(PYTHON) scripts/curate_dataset.py \
		--output data/benchmark_v2/full_dataset.json \
		--target-items 1500

curate-sample:
	$(PYTHON) scripts/curate_dataset.py \
		--output data/benchmark_v2/full_dataset.json \
		--sample-only

# Estimate costs
estimate:
	$(PYTHON) scripts/estimate_cost.py --full-benchmark --items 1500

estimate-quick:
	$(PYTHON) scripts/estimate_cost.py --models gpt-4o --items 100

# Run evaluation
evaluate:
	$(PYTHON) scripts/run_evaluation.py \
		--benchmark data/benchmark_v2/test.json \
		--output results/ \
		--models $(MODELS)

evaluate-quick:
	$(PYTHON) scripts/run_evaluation.py \
		--benchmark data/benchmark_v2/test.json \
		--output results/ \
		--models gpt-4o \
		--max-items 10

# Smoke test
smoke-test:
	bash scripts/run_full_benchmark.sh --smoke-test

# Full benchmark
full-benchmark:
	bash scripts/run_full_benchmark.sh --models gpt-5.1,claude-opus-4.5,gemini-3-pro

# Analyze results
analyze:
	$(PYTHON) -c "\
	from analysis import ScoreAnalyzer; \
	analyzer = ScoreAnalyzer('results'); \
	analyzer.load_scores(); \
	results = analyzer.analyze(); \
	print('Analysis complete:', len(results.models), 'models')"

# Generate reports
report:
	$(PYTHON) -c "\
	import json; \
	from analysis import ScoreAnalyzer, MedExplainVisualizer, ReportGenerator; \
	analyzer = ScoreAnalyzer('results'); \
	analyzer.load_scores(); \
	results = analyzer.analyze(); \
	viz = MedExplainVisualizer('reports/figures'); \
	viz.generate_all_figures(results.to_dict()); \
	reporter = ReportGenerator('reports'); \
	reporter.generate_all_reports(results.to_dict(), figures_dir='reports/figures'); \
	print('Reports saved to reports/')"

# View HTML report
view-report:
	open reports/summary_report.html || xdg-open reports/summary_report.html

# Run
run:
	$(UV) run medexplain-evals $(ARGS)

info:
	$(UV) run medexplain-evals info

evaluate:
	$(UV) run medexplain-evals evaluate $(ARGS)

leaderboard:
	$(UV) run medexplain-evals leaderboard $(ARGS)

# Development utilities
update:
	$(UV) lock --upgrade
	$(UV) sync --dev

outdated:
	$(UV) pip list --outdated

# Release
release-patch:
	$(UV) run bump-my-version bump patch

release-minor:
	$(UV) run bump-my-version bump minor

release-major:
	$(UV) run bump-my-version bump major
