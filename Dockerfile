# MEQ-Bench Dockerfile
# Multi-stage build for minimal production image

# =============================================================================
# Stage 1: Build stage with uv
# =============================================================================
FROM python:3.12-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set environment variables
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Install dependencies first for better caching
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Copy source code and install project
COPY src/ ./src/
COPY config.yaml ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# =============================================================================
# Stage 2: Production image
# =============================================================================
FROM python:3.12-slim AS runtime

# Create non-root user for security
RUN groupadd --gid 1000 meqbench && \
    useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home meqbench

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=meqbench:meqbench /app/.venv /app/.venv

# Copy application code
COPY --chown=meqbench:meqbench src/ ./src/
COPY --chown=meqbench:meqbench scripts/ ./scripts/
COPY --chown=meqbench:meqbench analysis/ ./analysis/
COPY --chown=meqbench:meqbench config.yaml ./
COPY --chown=meqbench:meqbench configs/ ./configs/

# Create necessary directories
RUN mkdir -p /app/data /app/results /app/logs /app/reports && \
    chown -R meqbench:meqbench /app && \
    chmod +x /app/scripts/*.sh 2>/dev/null || true

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app"

# Switch to non-root user
USER meqbench

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from src.settings import get_settings; get_settings()" || exit 1

# Default command (flexible for different entry points)
CMD ["python", "-c", "print('MEQ-Bench 2.0 Ready. Use: python scripts/run_evaluation.py --help')"]

# Labels
LABEL org.opencontainers.image.title="MEQ-Bench" \
      org.opencontainers.image.description="Resource-Efficient Benchmark for Medical LLM Evaluation" \
      org.opencontainers.image.version="2.0.0" \
      org.opencontainers.image.source="https://github.com/heilcheng/MEQ-Bench" \
      org.opencontainers.image.licenses="MIT"

# =============================================================================
# Stage 3: Development image (optional)
# =============================================================================
FROM builder AS development

# Install dev dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# Copy test files
COPY tests/ ./tests/

# Set environment
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# Default command for development
CMD ["pytest", "tests/", "-v"]
