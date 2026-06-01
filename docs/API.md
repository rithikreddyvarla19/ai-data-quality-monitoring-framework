# API Reference

Start the service:

```bash
uvicorn api.main:app --reload --port 8000
```

## Health

```bash
curl http://localhost:8000/health
```

## Data Quality Validation

```bash
curl -X POST http://localhost:8000/validate \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "customer_churn_current",
    "target_column": "churn",
    "schema_fields": [
      {"name": "customer_id", "dtype": "integer", "nullable": false},
      {"name": "age", "dtype": "integer", "nullable": false, "min_value": 18, "max_value": 100},
      {"name": "churn", "dtype": "integer", "nullable": false, "allowed_values": [0, 1]}
    ],
    "records": [
      {"customer_id": 1, "age": 34, "churn": 0},
      {"customer_id": 2, "age": 26, "churn": 1}
    ]
  }'
```

## Drift Detection

```bash
curl -X POST http://localhost:8000/drift \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "customer_churn",
    "reference_records": [{"age": 34, "region": "North"}, {"age": 45, "region": "West"}],
    "current_records": [{"age": 28, "region": "South"}, {"age": 29, "region": "South"}]
  }'
```

## Model Health

```bash
curl -X POST http://localhost:8000/model-health \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "churn_classifier",
    "y_true": [0, 1, 1, 0],
    "y_pred": [0, 1, 0, 0],
    "y_probability": [0.10, 0.87, 0.41, 0.20]
  }'
```

## GPT Monitoring

```bash
curl -X POST http://localhost:8000/gpt-monitoring \
  -H "Content-Type: application/json" \
  -d '{
    "system_name": "support_copilot",
    "records": [
      {
        "prompt": "What plan includes SSO?",
        "context": "The enterprise plan includes SSO and audit logs.",
        "response": "The enterprise plan includes SSO.",
        "latency_ms": 892
      }
    ]
  }'
```

## Full Monitoring Run

`POST /monitoring-run` combines data quality, drift, optional label leakage, optional model health, alerts, and trust scoring in one response.

