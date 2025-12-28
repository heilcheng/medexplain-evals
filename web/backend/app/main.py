"""
MedExplain-Evals API

FastAPI backend for the MedExplain-Evals evaluation framework.
Provides REST API and WebSocket endpoints for running evaluations,
managing models, and viewing results.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from .models import init_db
from .api.v1 import api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("medexplain.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Initializing database...")
    init_db()
    logger.info("Database initialized")
    yield
    logger.info("Shutting down...")


def custom_openapi():
    """Custom OpenAPI schema for better documentation."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="MedExplain-Evals API",
        version="2.0.0",
        description="""
## MedExplain-Evals - Medical Explanation Quality Benchmark

RESTful API for evaluating audience-adaptive medical explanations from LLMs.

### Features

- **Evaluation Management**: Create, run, and monitor evaluation runs
- **Model Configuration**: Register and configure LLMs for evaluation
- **Audience Personas**: Evaluate for physicians, nurses, patients, caregivers
- **Real-time Updates**: WebSocket support for live progress
- **Results Analysis**: Query and compare evaluation results
- **Leaderboard**: Aggregated model rankings by audience

### Evaluation Dimensions

| Dimension | Description |
|-----------|-------------|
| Factual Accuracy | Clinical correctness and evidence alignment |
| Terminological Appropriateness | Language complexity for audience |
| Explanatory Completeness | Comprehensive yet accessible coverage |
| Actionability | Clear, practical guidance |
| Safety | Warnings and harm avoidance |
| Empathy & Tone | Audience-appropriate communication |

### Supported Models

| Provider | Models |
|----------|--------|
| OpenAI | GPT-5.1, GPT-4o |
| Anthropic | Claude Opus 4.5, Sonnet 4.5 |
| Google | Gemini 3 Pro, Flash |
| Meta | Llama 4 Scout, Maverick |
| DeepSeek | DeepSeek-V3 |
| Alibaba | Qwen3-Max |

### Target Audiences

- **Physicians**: Clinical precision, medical terminology
- **Nurses**: Practical, actionable nursing interventions
- **Patients**: Plain language, empathetic, grade 6-10 reading level
- **Caregivers**: Practical care guidance, warning signs
        """,
        routes=app.routes,
        tags=[
            {
                "name": "evaluations",
                "description": "Create and manage evaluation runs",
            },
            {
                "name": "models",
                "description": "Configure LLMs for evaluation",
            },
            {
                "name": "audiences",
                "description": "Manage audience personas",
            },
            {
                "name": "websocket",
                "description": "Real-time evaluation progress updates",
            },
        ],
    )

    openapi_schema["info"]["contact"] = {
        "name": "MedExplain-Evals",
        "url": "https://github.com/heilcheng/medexplain-evals",
    }

    openapi_schema["info"]["license"] = {
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


# Create FastAPI app
app = FastAPI(
    title="MedExplain-Evals API",
    description="API for evaluating audience-adaptive medical explanations",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

app.openapi = custom_openapi

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")


@app.get("/", tags=["root"])
async def root():
    """API root endpoint."""
    return {
        "name": "MedExplain-Evals API",
        "version": "2.0.0",
        "description": "Benchmark for Audience-Adaptive Medical Explanation Quality in LLMs",
        "docs": "/api/docs",
        "health": "/health",
    }


@app.get("/health", tags=["root"])
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "service": "medexplain-evals-api"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
