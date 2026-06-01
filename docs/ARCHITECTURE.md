# Architecture

## System Overview

```mermaid
flowchart LR
  A["Batch and streaming datasets"] --> B["Data quality validation engine"]
  A --> C["Drift detection engine"]
  D["Model predictions and labels"] --> E["Model health monitor"]
  F["GPT, GAN, and VAE telemetry"] --> G["AI system monitor"]
  B --> H["Monitoring reports"]
  C --> H
  E --> H
  G --> H
  H --> I["Alert manager"]
  H --> J["PostgreSQL report store"]
  H --> K["FastAPI service"]
  H --> L["Streamlit dashboard"]
  I --> M["Console, webhook, incident tools"]
```

## Runtime Components

```mermaid
flowchart TB
  subgraph "Core Python Packages"
    DQ["data_quality"]
    DR["drift_detection"]
    MON["monitoring"]
    AL["alerts"]
  end

  subgraph "Serving Layer"
    API["FastAPI"]
    DASH["Streamlit"]
  end

  subgraph "Persistence and Orchestration"
    PG["PostgreSQL"]
    SCH["APScheduler jobs"]
    CI["GitHub Actions CI"]
  end

  DQ --> MON
  DR --> MON
  MON --> AL
  API --> DQ
  API --> DR
  API --> MON
  DASH --> DQ
  DASH --> DR
  DASH --> MON
  MON --> PG
  SCH --> API
  CI --> API
```

## Data Flow

```mermaid
sequenceDiagram
  participant Job as Scheduled Job
  participant API as FastAPI
  participant DQ as Data Quality Engine
  participant Drift as Drift Detector
  participant Model as Model Monitor
  participant Alert as Alert Manager
  participant Store as PostgreSQL
  participant UI as Streamlit Dashboard

  Job->>API: POST /monitoring-run
  API->>DQ: validate current dataset
  API->>Drift: compare reference vs current
  API->>Model: evaluate predictions
  API->>Alert: evaluate checks
  API->>Store: persist reports
  UI->>Store: read latest monitoring state
```

## Design Principles

- Core checks run locally with deterministic Pandas/NumPy/SciPy logic.
- Great Expectations and Evidently AI are optional adapters for teams that already maintain expectation suites or HTML drift reports.
- PySpark frames are accepted through `toPandas()` for validation paths where data volumes have already been sampled or profiled upstream.
- Every engine returns a common `MonitoringReport` object so API, dashboard, storage, and alerting integrations can compose reports consistently.
- Thresholds are explicit, versioned, and configurable through code or `config/settings.yaml`.

## Enterprise Extension Points

- Replace `ConsoleAlertChannel` with Slack, PagerDuty, ServiceNow, or Teams channels.
- Persist reports through `monitoring.storage.ReportStore` using `DATABASE_URL`.
- Add model registry metadata to `MonitoringReport.metadata`.
- Run high-volume drift checks in Spark and pass aggregated feature profiles into the same report contract.
- Export reports to object storage for audit and regulatory retention.

