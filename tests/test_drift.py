import numpy as np
import pandas as pd

from tabular_monitor.drift import DriftConfig, DriftDetector


def test_duplicate_quantile_bins_and_out_of_range_values():
    ref = pd.DataFrame({"score": [600.0] * 50 + [700.0] * 50})
    cur = pd.DataFrame({"score": [-999.0] * 10 + [600.0] * 40 + [900.0] * 50})
    check = DriftDetector(ref, DriftConfig(min_samples=10)).detect(cur).checks[1]
    assert check.details["bin_count"] >= 1 and check.status in {"warn", "fail"}


def test_non_finite_are_removed_and_small_samples_skipped():
    ref = pd.DataFrame({"x": [1.0, np.inf, 2.0]})
    cur = pd.DataFrame({"x": [1.0, np.nan, 2.0]})
    check = DriftDetector(ref, DriftConfig(min_samples=3)).detect(cur).checks[1]
    assert check.status == "skipped" and check.details["reference_samples"] == 2


def test_high_cardinality_is_bounded():
    ref = pd.DataFrame({"merchant": [f"m{i}" for i in range(100)]})
    cur = pd.DataFrame({"merchant": [f"n{i}" for i in range(100)]})
    check = DriftDetector(ref, DriftConfig(min_samples=10, max_categories=10)).detect(cur).checks[1]
    assert check.details["category_count"] <= 10 and check.details["rare_values_collapsed"]


def test_missing_current_column_fails():
    report = DriftDetector(pd.DataFrame({"x": range(40)})).detect(pd.DataFrame({"y": range(40)}))
    assert not report.passed and report.metadata["missing_columns"] == ["x"]
