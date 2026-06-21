# Auto Insurance Premium ML

End-to-end demo project for predicting car insurance premium values from tabular driver/vehicle features.

> 📖 **Read the [Architecture Design Document](docs/architecture-design-document.md)** for a detailed overview of the system design, ML pipeline, and technical decisions.

This repository includes:

- A Python ML pipeline (data loading, validation, EDA, preprocessing, model training, champion selection, prediction CSV generation)
- A FastAPI inference backend
- A React frontend UI layer (source files are present under `frontend/src`)

## Project Structure

- `backend/app/data/` - loaders, schema validation, outlier filtering
- `backend/app/features/` - feature engineering and preprocessing
- `backend/app/models/` - model factories, training, champion registry, batch prediction generation
- `backend/app/evaluation/` - metrics and reporting utilities
- `backend/app/schemas/` - Pydantic request/response schemas
- `backend/app/services/` - prediction service
- `backend/app/api/` - FastAPI routes
- `backend/app/main.py` - FastAPI app entrypoint
- `configs/config.yaml` - central runtime/training config
- `artifacts/` - generated models, metrics, plots, submissions
- `data/raw/` - input CSV files (`train.csv`, `test.csv`)
- `frontend/src/` - frontend app source

## Prerequisites

- Python 3.13+
- [`uv`](https://docs.astral.sh/uv/)
- Node.js 18+ and npm (for frontend runtime)

Optional but recommended:

- macOS: `brew install libomp` (needed if local `xgboost` import fails at runtime)

## Backend Setup

From repo root:

```bash
cd backend
uv sync
cd ..
```

This creates/updates `backend/.venv` and installs backend dependencies.

## Run the ML Pipeline End-to-End

Run from repo root so imports and relative paths resolve consistently:

```bash
# 1) Train all models and write model + metrics artifacts
backend/.venv/bin/python -m backend.app.models.train

# 2) Select and persist champion artifacts
backend/.venv/bin/python -c "from backend.app.models.registry import select_champion, save_champion; c=select_champion(); save_champion(c['model_name'], c)"

# 3) Generate model comparison CSV / evaluation table
backend/.venv/bin/python -c "from backend.app.evaluation.reports import generate_comparison_table; generate_comparison_table()"

# 4) Generate submission predictions CSV from champion model
backend/.venv/bin/python -m backend.app.models.generate_predictions
```

Expected outputs include:

- `artifacts/models/*.pkl`
- `artifacts/models/champion_model.pkl`
- `artifacts/models/champion_preprocessor.pkl`
- `artifacts/models/champion_metadata.json`
- `artifacts/metrics/*_metrics.json`
- `artifacts/metrics/model_comparison.csv`
- `artifacts/submissions/predictions.csv`

## Run the FastAPI Backend

From repo root:

```bash
backend/.venv/bin/python -m uvicorn backend.app.main:app --reload --port 8000
```

Key endpoints:

- `GET /health`
- `GET /feature-schema`
- `GET /models`
- `POST /predict`

## Frontend Setup and Run

The repo currently contains frontend source files under `frontend/src`, but may not yet include a full Vite scaffold (`package.json`, etc.).

If `frontend/package.json` is missing, initialize the frontend first:

```bash
cd frontend
npm create vite@latest . -- --template react-ts
npm install
cd ..
```

Then run:

```bash
cd frontend
npm run dev
```

By default, frontend API calls target `http://localhost:8000`. Override with:

```bash
VITE_API_BASE_URL=http://localhost:8000
```

## Tests

Run all current tests:

```bash
backend/.venv/bin/python -m pytest tests
```

Or targeted suites:

```bash
backend/.venv/bin/python -m pytest tests/test_data.py
backend/.venv/bin/python -m pytest tests/test_features.py
backend/.venv/bin/python -m pytest tests/test_models.py tests/test_evaluation.py
backend/.venv/bin/python -m pytest tests/test_api.py
```

## Notes

- Config is centralized in `configs/config.yaml`.
- Raw data files are ignored by git via `data/raw/*.csv`.
- If API tests fail due to missing `httpx`, install it in backend env:

```bash
cd backend
uv add httpx
```
