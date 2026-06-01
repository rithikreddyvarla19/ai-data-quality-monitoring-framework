from __future__ import annotations

import pandas as pd

from drift_detection.detector import DriftDetector
from drift_detection.metrics import (
    chi_square_test,
    jensen_shannon_divergence,
    ks_test,
    population_stability_index,
)
from monitoring.types import CheckStatus


def test_population_stability_index_is_low_for_same_distribution() -> None:
    reference = pd.Series([1, 2, 3, 4, 5, 6])
    current = pd.Series([1, 2, 3, 4, 5, 6])

    assert population_stability_index(reference, current) < 0.01
    assert jensen_shannon_divergence(reference, current) < 0.01


def test_statistical_tests_return_expected_keys() -> None:
    reference = pd.Series([1, 2, 3, 4, 5, 6])
    current = pd.Series([8, 9, 10, 11, 12, 13])
    categories_a = pd.Series(["a", "a", "b", "b"])
    categories_b = pd.Series(["a", "b", "b", "b"])

    assert {"statistic", "p_value"} == set(ks_test(reference, current))
    assert {"statistic", "p_value"} == set(chi_square_test(categories_a, categories_b))


def test_drift_detector_reports_feature_checks() -> None:
    reference = pd.DataFrame({"age": [20, 21, 22, 23, 24, 25], "region": ["North", "North", "West", "West", "South", "South"]})
    current = pd.DataFrame({"age": [40, 41, 42, 43, 44, 45], "region": ["East", "East", "East", "West", "West", "West"]})

    report = DriftDetector(reference, dataset_name="unit").detect(current)

    assert report.status in {CheckStatus.WARN, CheckStatus.FAIL}
    assert "feature_drift" in report.metrics
    assert any(check.name == "drift.feature.age" for check in report.checks)

