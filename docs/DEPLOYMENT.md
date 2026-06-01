# Deployment

## Local Docker Compose

```bash
cp .env.example .env
docker compose up --build
```

The stack starts:

- PostgreSQL on `localhost:5432`
- FastAPI on `http://localhost:8000`
- Streamlit dashboard on `http://localhost:8501`

## Production Container

```bash
docker build -t ai-data-quality-monitoring-framework:latest .
docker run --env-file .env -p 8000:8000 ai-data-quality-monitoring-framework:latest
```

## Required Environment Variables

| Variable | Purpose |
| --- | --- |
| `DATABASE_URL` | PostgreSQL SQLAlchemy URL for report persistence |
| `PERSIST_REPORTS` | Enables report writes when set to `true` |
| `ALERT_WEBHOOK_URL` | Optional webhook endpoint for incident routing |
| `QUALITY_SCORE_THRESHOLD` | Minimum acceptable quality score |
| `DRIFT_SCORE_THRESHOLD` | Maximum acceptable drift threshold |
| `MODEL_HEALTH_THRESHOLD` | Minimum acceptable model-health score |

## CI/CD

The GitHub Actions pipeline runs:

1. Python setup for 3.10 and 3.11
2. Package installation
3. Ruff lint checks
4. Pytest with coverage
5. Docker image build

## Scaling Pattern

For enterprise deployments:

- Run API replicas behind a load balancer.
- Schedule monitoring through Airflow, Dagster, Prefect, or Kubernetes CronJobs.
- Persist report payloads in PostgreSQL and large rendered artifacts in object storage.
- Use Spark or warehouse SQL to precompute large-dataset profiles before calling this service.
- Route high-severity alerts to the incident-management system.
