import numpy as np
import pandas as pd
import pytest

from tabular_monitor.validation import FieldRule, ValidationConfig, Validator


def test_corrupt_schema_and_non_finite_values_fail():
    frame = pd.DataFrame(
        {"income": [40_000.0, np.inf, -1.0], "segment": ["prime", "unknown", None]}
    )
    cfg = ValidationConfig(
        min_rows=1,
        fields=[
            FieldRule("income", "number", minimum=0),
            FieldRule("segment", "string", nullable=False, allowed={"prime", "subprime"}),
        ],
    )
    report = Validator(cfg).validate(frame)
    assert not report.passed
    income = next(c for c in report.checks if c.name == "schema.income")
    assert income.details["non_finite_count"] == 1 and income.details["below_minimum_count"] == 1


def test_missing_required_column_is_reported_without_crashing():
    report = Validator(
        ValidationConfig(min_rows=1, fields=[FieldRule("credit_score", "int")])
    ).validate(pd.DataFrame({"age": [30]}))
    assert next(c for c in report.checks if c.name == "schema.columns").details["missing"] == [
        "credit_score"
    ]


def test_invalid_threshold_rejected():
    with pytest.raises(ValueError):
        ValidationConfig(max_missing_rate=2)
