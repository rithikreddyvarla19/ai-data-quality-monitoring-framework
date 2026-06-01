# AI Data Quality Monitoring Framework

Enterprise-grade platform for monitoring data quality, dataset integrity, drift, model health, and AI system performance across traditional ML and generative AI workloads.

## What This Platform Covers

- Data quality validation with schema, null, duplicate, range, enum, and regex checks
- Missing value, outlier, and dataset integrity monitoring
- Distribution drift detection with KS test, chi-square test, Jensen-Shannon divergence, and PSI
- Feature drift and dataset trust scoring
- Label leakage detection for tabular ML datasets
- Model health monitoring for accuracy, precision, recall, F1, calibration, and Brier score
- GPT-style system monitoring for answer quality, latency, empty response rate, refusal rate, and retrieval overlap
- GAN and VAE monitoring with fidelity, reconstruction, and mode-collapse indicators
- Alert routing for quality, drift, and model-health breaches
- FastAPI service for automated monitoring workflows
- Streamlit dashboard for MLOps teams
- PostgreSQL persistence schema
- Docker Compose deployment
- CI/CD workflow with linting and tests

## Repository Structure

```text
ai-data-quality-monitoring-framework/
  alerts/              Alert rules, dispatch, and notification channels
  api/                 FastAPI application and request/response schemas
  dashboards/          Streamlit enterprise monitoring dashboard
  data/sample/         Reference/current/prediction sample datasets
  data_quality/        Validation engine, schema checks, missingness, outliers
  drift_detection/     PSI, KS, chi-square, JSD, feature drift reports
  docs/                Architecture, deployment, operations, and API docs
  monitoring/          Model, GPT, GAN, VAE, leakage, scoring, storage, scheduler
  scripts/             CLI monitoring runner
  sql/                 PostgreSQL schema
  tests/               Unit tests
```

## Quick Start

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
```

Run the API:

```bash
uvicorn api.main:app --reload --port 8000
```

Run the dashboard:

```bash
streamlit run dashboards/streamlit_app.py
```

Run the sample monitoring job:

```bash
python scripts/run_monitoring.py \
  --reference data/sample/reference_dataset.csv \
  --current data/sample/current_dataset.csv \
  --predictions data/sample/model_predictions.csv \
  --output outputs/sample_monitoring_report.json
```

## Docker Deployment

```bash
docker compose up --build
```

Services:

- API: `http://localhost:8000`
- Dashboard: `http://localhost:8501`
- PostgreSQL: `localhost:5432`

## API Highlights

- `GET /health`
- `POST /validate`
- `POST /drift`
- `POST /model-health`
- `POST /monitoring-run`

See [docs/API.md](docs/API.md) for request examples.

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for system diagrams and design notes.

## Production Notes

This repository is designed as a reference implementation for enterprise MLOps teams. It uses optional adapters for Great Expectations, Evidently AI, PySpark, and PostgreSQL so local development stays lightweight while production deployments can enable the full stack.

