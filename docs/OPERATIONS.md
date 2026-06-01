# Operations Guide

## Monitoring Cadence

Recommended baseline schedules:

- Critical customer-facing datasets: hourly
- Model prediction quality: hourly or daily depending on label availability
- Full drift analysis: daily
- GPT telemetry and safety checks: near real time for production assistants
- Feature importance trend checks: per model release and weekly

## Threshold Governance

Treat thresholds as production configuration:

- Review PSI and JSD thresholds with data owners.
- Maintain separate thresholds for high-cardinality categorical features.
- Lower model-health thresholds only through an explicit exception process.
- Version schemas with each upstream data contract.

## Incident Workflow

1. Inspect open alerts in the dashboard.
2. Review failing checks and changed feature distributions.
3. Compare current data with the reference window.
4. Check pipeline freshness and source-system release notes.
5. Decide whether to quarantine the dataset, retrain, rollback, or suppress a known benign alert.

## Dataset Trust Score

The default trust score weights:

- Data quality: 40%
- Drift: 30%
- Model health: 20%
- Explainability: 10%

Tune these weights for regulated domains where data contract failures should dominate the score.

## Runbook Checks

- `data_quality.schema.*` failures usually indicate upstream contract changes.
- `data_quality.missing_values` failures often indicate ingestion gaps or source-field deprecations.
- `drift.feature.*` warnings should be reviewed against expected business seasonality.
- `model.expected_calibration_error` failures may require recalibration even when accuracy is stable.
- `data_quality.label_leakage` failures should block training until feature lineage is reviewed.

## Audit Readiness

Persist each report payload and retain:

- Dataset name and run ID
- Code version or container image digest
- Reference dataset window
- Schema version
- Threshold configuration
- Alert disposition and owner

