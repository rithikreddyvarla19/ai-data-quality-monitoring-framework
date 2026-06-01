from __future__ import annotations

import pandas as pd

from data_quality.engine import DataQualityConfig, DataQualityEngine
from monitoring.leakage import LabelLeakageDetector
from monitoring.types import CheckStatus


def test_data_quality_engine_scores_valid_dataset() -> None:
    df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "age": [34, 45, 29],
            "plan_type": ["basic", "plus", "enterprise"],
            "churn": [0, 0, 1],
        }
    )
    schema = [
        {"name": "customer_id", "dtype": "integer", "nullable": False},
        {"name": "age", "dtype": "integer", "nullable": False, "min_value": 18},
        {"name": "plan_type", "dtype": "string", "allowed_values": ["basic", "plus", "enterprise"]},
        {"name": "churn", "dtype": "integer", "allowed_values": [0, 1]},
    ]

    report = DataQualityEngine(DataQualityConfig(dataset_name="unit", schema=schema)).validate(df)

    assert report.score > 0.9
    assert report.status == CheckStatus.PASS
    assert "data_quality_score" in report.metrics


def test_data_quality_engine_fails_missing_required_column() -> None:
    df = pd.DataFrame({"customer_id": [1, 2, 3]})
    schema = [
        {"name": "customer_id", "dtype": "integer", "nullable": False},
        {"name": "age", "dtype": "integer", "nullable": False},
    ]

    report = DataQualityEngine(DataQualityConfig(dataset_name="unit", schema=schema)).validate(df)

    assert report.status == CheckStatus.FAIL
    assert any(check.name == "schema.columns" for check in report.checks)


def test_label_leakage_detector_finds_target_copy() -> None:
    df = pd.DataFrame({"feature": [0, 1, 0, 1], "target": [0, 1, 0, 1]})

    result = LabelLeakageDetector().detect(df, target_column="target")

    assert result.status == CheckStatus.FAIL
    assert result.details["findings"][0]["type"] == "exact_target_copy"

