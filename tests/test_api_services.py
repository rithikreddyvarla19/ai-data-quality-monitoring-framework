from __future__ import annotations

from api.services import run_full_monitoring


def test_full_monitoring_service_returns_trust_score() -> None:
    reference = [
        {"age": 25, "region": "North", "churn": 0},
        {"age": 30, "region": "South", "churn": 1},
        {"age": 35, "region": "West", "churn": 0},
    ]
    current = [
        {"age": 26, "region": "North", "churn": 0},
        {"age": 31, "region": "South", "churn": 1},
        {"age": 36, "region": "West", "churn": 0},
    ]
    predictions = [
        {"actual": 0, "prediction": 0, "probability": 0.2},
        {"actual": 1, "prediction": 1, "probability": 0.8},
        {"actual": 0, "prediction": 0, "probability": 0.3},
    ]

    result = run_full_monitoring(
        {
            "dataset_name": "unit",
            "reference_records": reference,
            "current_records": current,
            "prediction_records": predictions,
            "target_column": "churn",
        }
    )

    assert "trust_score" in result
    assert result["data_quality"]["dataset_name"] == "unit"

