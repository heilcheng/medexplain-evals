# MedExplain-Evals Web Platform

Interactive web interface for the MedExplain-Evals benchmark.

## Architecture

```
web/
├── backend/          # FastAPI REST API
│   ├── app/
│   │   ├── api/      # API endpoints
│   │   ├── models/   # Database models
│   │   ├── schemas/  # Pydantic schemas
│   │   └── services/ # Business logic
│   └── requirements.txt
│
└── frontend/         # Next.js dashboard
    └── src/
        ├── app/      # Pages (App Router)
        ├── components/
        └── lib/      # API client, utilities
```

## Quick Start

### Backend

```bash
cd web/backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

API documentation: http://localhost:8000/api/docs

### Frontend

```bash
cd web/frontend
npm install
npm run dev
```

Dashboard: http://localhost:3000

## Features

- **Dashboard**: Overview of evaluations, stats, quick actions
- **Evaluations**: Create and monitor evaluation runs
- **Models**: Configure LLMs for evaluation
- **Audiences**: View target personas (physicians, nurses, patients, caregivers)
- **Results**: Analyze performance by dimension and audience
- **Leaderboard**: Model rankings with filtering

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/evaluations` | GET | List evaluations |
| `/api/v1/evaluations` | POST | Create evaluation |
| `/api/v1/evaluations/{id}` | GET | Get evaluation details |
| `/api/v1/models` | GET | List available models |
| `/api/v1/audiences` | GET | List audience personas |
| `/api/v1/ws/{evaluation_id}` | WS | Real-time progress |

## Environment Variables

### Backend

```bash
DATABASE_URL=sqlite:///./medexplain.db
OPENAI_API_KEY=your-key
ANTHROPIC_API_KEY=your-key
```

### Frontend

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Docker

```bash
# From project root
docker-compose up -d medexplain
```

Access at http://localhost:3000
