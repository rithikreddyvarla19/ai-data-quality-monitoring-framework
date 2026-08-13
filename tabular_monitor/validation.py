from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from tabular_monitor.types import Check, Report

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class FieldRule:
    name: str
    dtype: str
    required: bool = True
    nullable: bool = True
    finite: bool = True
    minimum: float | None = None
    maximum: float | None = None
    allowed: set[Any] | None = None
    pattern: str | None = None


@dataclass(slots=True)
class ValidationConfig:
    dataset: str = "dataset"
    fields: list[FieldRule] = field(default_factory=list)
    allow_extra_columns: bool = True
    max_missing_rate: float = 0.05
    max_duplicate_rate: float = 0.01
    min_rows: int = 30

    def __post_init__(self) -> None:
        if not 0 <= self.max_missing_rate <= 1 or not 0 <= self.max_duplicate_rate <= 1:
            raise ValueError("rate thresholds must be between 0 and 1")
        if self.min_rows < 1:
            raise ValueError("min_rows must be positive")


class Validator:
    def __init__(self, config: ValidationConfig) -> None:
        self.config = config

    def validate(self, frame: pd.DataFrame) -> Report:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")
        df = frame.copy()
        LOGGER.info("validation_started", extra={"dataset": self.config.dataset, "rows": len(df)})
        report = Report(
            self.config.dataset, metadata={"rows": len(df), "columns": list(df.columns)}
        )
        report.checks.append(
            Check(
                "row_count",
                "pass" if len(df) >= self.config.min_rows else "fail",
                {"observed": len(df), "minimum": self.config.min_rows},
            )
        )
        expected = {r.name for r in self.config.fields}
        required = {r.name for r in self.config.fields if r.required}
        missing = sorted(required - set(df))
        extra = sorted(set(df) - expected) if expected else []
        report.checks.append(
            Check(
                "schema.columns",
                "fail" if missing or (extra and not self.config.allow_extra_columns) else "pass",
                {"missing": missing, "extra": extra},
            )
        )
        rate = float(df.duplicated().mean()) if len(df) else 0.0
        report.checks.append(
            Check(
                "duplicates",
                "fail" if rate > self.config.max_duplicate_rate else "pass",
                {
                    "count": int(df.duplicated().sum()),
                    "rate": rate,
                    "threshold": self.config.max_duplicate_rate,
                },
            )
        )
        for rule in self.config.fields:
            if rule.name in df:
                report.checks.append(self._field(df[rule.name], rule))
        LOGGER.info(
            "validation_finished", extra={"dataset": self.config.dataset, "passed": report.passed}
        )
        return report

    def _field(self, series: pd.Series, rule: FieldRule) -> Check:
        issues: dict[str, Any] = {}
        missing_rate = float(series.isna().mean()) if len(series) else 0.0
        if not rule.nullable and series.isna().any():
            issues["null_count"] = int(series.isna().sum())
        if missing_rate > self.config.max_missing_rate:
            issues["missing_rate"] = missing_rate
        if not _dtype_matches(series, rule.dtype):
            issues["dtype"] = {"expected": rule.dtype, "observed": str(series.dtype)}
        if rule.dtype in {"number", "float", "int"}:
            numeric = pd.to_numeric(series, errors="coerce")
            failures = int((series.notna() & numeric.isna()).sum())
            if failures:
                issues["non_numeric_count"] = failures
            non_finite = int((numeric.notna() & ~np.isfinite(numeric)).sum())
            if rule.finite and non_finite:
                issues["non_finite_count"] = non_finite
            finite = numeric[numeric.notna() & np.isfinite(numeric)]
            if rule.minimum is not None and (finite < rule.minimum).any():
                issues["below_minimum_count"] = int((finite < rule.minimum).sum())
            if rule.maximum is not None and (finite > rule.maximum).any():
                issues["above_maximum_count"] = int((finite > rule.maximum).sum())
        if rule.allowed is not None:
            bad = series.dropna()[~series.dropna().isin(rule.allowed)].astype(str).unique()
            if len(bad):
                issues["unexpected_values"] = sorted(bad.tolist())[:20]
        if rule.pattern is not None:
            try:
                bad_pattern = ~series.dropna().astype(str).str.fullmatch(rule.pattern)
            except re.error as exc:
                raise ValueError(f"invalid regex for {rule.name}: {exc}") from exc
            if bad_pattern.any():
                issues["pattern_mismatch_count"] = int(bad_pattern.sum())
        return Check(f"schema.{rule.name}", "fail" if issues else "pass", issues)


def _dtype_matches(series: pd.Series, expected: str) -> bool:
    expected = expected.lower()
    if expected == "number":
        return pdt.is_numeric_dtype(series)
    if expected == "float":
        return pdt.is_float_dtype(series) or pdt.is_integer_dtype(series)
    if expected == "int":
        return pdt.is_integer_dtype(series)
    if expected in {"str", "string", "category"}:
        return (
            pdt.is_string_dtype(series)
            or pdt.is_object_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        )
    if expected in {"datetime", "date"}:
        return pdt.is_datetime64_any_dtype(series)
    if expected == "bool":
        return pdt.is_bool_dtype(series)
    return str(series.dtype).lower() == expected
