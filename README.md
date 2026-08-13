# Tabular Data Quality and Drift Monitor

This repository checks tabular datasets before they enter analytics or model pipelines. Its scope is deliberately limited to schema validation, row-level data quality, and univariate distribution drift.

The example data represents credit applications. It is generated deterministically; it contains no customer data.

## What is implemented

- Required-column, type, nullability, range, allowed-value, and regex checks
- Explicit detection of `NaN`, positive infinity, and negative infinity
- Dataset size and duplicate-rate checks
- Numeric drift using reference quantile bins, PSI, Jensen-Shannon divergence, and KS tests
- Categorical drift with rare-value grouping, cardinality bounds, PSI, Jensen-Shannon divergence, and chi-square tests
- Structured result objects and logging events suitable for a job runner
- Deterministic credit-risk data generation and edge-case tests

## Install and verify

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
ruff check .
```

Generate reference and shifted datasets:

```bash
python scripts/generate_credit_data.py --output data/sample/credit_reference.csv --rows 1000 --seed 42
python scripts/generate_credit_data.py --output data/sample/credit_current.csv --rows 1000 --seed 43 --drifted
```

## Design choices and trade-offs

Reference-derived numeric bins make PSI comparisons repeatable and keep out-of-range current values in the edge bins. Duplicate quantile edges are removed, so discrete or highly skewed columns may use fewer bins than configured. Constant columns use one catch-all bin and consequently cannot reveal meaningful shape drift.

Statistical p-values are reported but PSI thresholds decide severity. With large datasets, significance tests can flag negligible changes; with small datasets, they have little power. Features below `min_samples` are marked `skipped` instead of being presented as healthy.

High-cardinality categorical values are grouped into `__OTHER__` using reference frequency. This bounds memory and avoids unstable sparse tests, but can hide movement among rare values. Missing values are treated as quality failures and excluded from drift calculations; monitor missingness separately.

## Known limitations

- Checks are in-memory and intended for batch-sized pandas DataFrames.
- Drift is univariate; it does not detect changes in relationships between columns.
- Thresholds are defaults, not calibrated business limits.
- No built-in authentication, scheduler, dashboard, or managed alert delivery is claimed.
- Persistence and network delivery are intentionally outside the core package; callers should retry and dead-letter failures at the orchestration boundary.

## Roadmap

1. Add versioned YAML schema loading with compatibility checks.
2. Add chunked readers and approximate sketches for large datasets.
3. Calibrate thresholds from historical clean windows and control false discovery across features.
4. Add multivariate drift and segment-level monitoring.
5. Provide optional, separately tested adapters for object storage and databases.

## Scope change

The previous generative-AI, GAN/VAE, explainability, leakage, and model-performance modules were removed from the focused package. They mixed unrelated monitoring problems with shallow implementations and expanded the dependency surface. Git history remains the record if that experimental code is needed later; no history was rewritten.
