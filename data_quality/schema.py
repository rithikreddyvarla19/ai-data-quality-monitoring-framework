from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from pandas.api import types as pdt

from monitoring.types import CheckResult, CheckSeverity, CheckStatus


@dataclass(slots=True)
class SchemaField:
    name: str
    dtype: str
    nullable: bool = True
    required: bool = True
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: list[Any] | None = None
    regex: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SchemaField:
        return cls(**payload)


class SchemaValidator:
    def __init__(self, schema: list[SchemaField | dict[str, Any]]) -> None:
        self.schema = [
            field if isinstance(field, SchemaField) else SchemaField.from_dict(field)
            for field in schema
        ]

    def validate(self, df: pd.DataFrame) -> list[CheckResult]:
        checks: list[CheckResult] = []
        checks.extend(self._validate_columns(df))
        checks.extend(self._validate_field_constraints(df))
        return checks

    def _validate_columns(self, df: pd.DataFrame) -> list[CheckResult]:
        expected = {schema_field.name for schema_field in self.schema if schema_field.required}
        observed = set(df.columns)
        missing = sorted(expected - observed)
        unexpected = sorted(observed - {schema_field.name for schema_field in self.schema})
        penalty = min(1.0, (len(missing) * 0.25) + (len(unexpected) * 0.05))
        status = CheckStatus.PASS if not missing else CheckStatus.FAIL
        severity = CheckSeverity.INFO if status == CheckStatus.PASS else CheckSeverity.CRITICAL
        return [
            CheckResult(
                name="schema.columns",
                status=status,
                score=1.0 - penalty,
                severity=severity,
                details={
                    "expected_columns": sorted(expected),
                    "observed_columns": sorted(observed),
                    "missing_columns": missing,
                    "unexpected_columns": unexpected,
                },
            )
        ]

    def _validate_field_constraints(self, df: pd.DataFrame) -> list[CheckResult]:
        checks: list[CheckResult] = []
        for schema_field in self.schema:
            if schema_field.name not in df.columns:
                if schema_field.required:
                    checks.append(
                        CheckResult(
                            name=f"schema.{schema_field.name}",
                            status=CheckStatus.FAIL,
                            score=0.0,
                            severity=CheckSeverity.CRITICAL,
                            details={"reason": "required column missing"},
                        )
                    )
                continue

            series = df[schema_field.name]
            violations: dict[str, Any] = {}

            if not self._dtype_matches(series, schema_field.dtype):
                violations["dtype"] = {
                    "expected": schema_field.dtype,
                    "observed": str(series.dtype),
                }

            if not schema_field.nullable:
                missing_count = int(series.isna().sum())
                if missing_count:
                    violations["nullability"] = missing_count

            if schema_field.min_value is not None:
                invalid = int((pd.to_numeric(series, errors="coerce") < schema_field.min_value).sum())
                if invalid:
                    violations["below_min"] = invalid

            if schema_field.max_value is not None:
                invalid = int((pd.to_numeric(series, errors="coerce") > schema_field.max_value).sum())
                if invalid:
                    violations["above_max"] = invalid

            if schema_field.allowed_values is not None:
                invalid_values = sorted(
                    set(series.dropna().unique()) - set(schema_field.allowed_values),
                    key=lambda value: str(value),
                )
                if invalid_values:
                    violations["unexpected_values"] = invalid_values[:25]

            if schema_field.regex is not None:
                regex_matches = series.dropna().astype(str).str.match(schema_field.regex)
                invalid = int((~regex_matches).sum())
                if invalid:
                    violations["regex_mismatch_count"] = invalid

            checks.append(self._field_check(schema_field.name, violations))
        return checks

    @staticmethod
    def _field_check(name: str, violations: dict[str, Any]) -> CheckResult:
        if not violations:
            return CheckResult(
                name=f"schema.{name}",
                status=CheckStatus.PASS,
                score=1.0,
                details={"violations": {}},
            )
        penalty = min(1.0, len(violations) * 0.25)
        severity = CheckSeverity.HIGH if "dtype" in violations else CheckSeverity.MEDIUM
        return CheckResult(
            name=f"schema.{name}",
            status=CheckStatus.FAIL,
            score=1.0 - penalty,
            severity=severity,
            details={"violations": violations},
        )

    @staticmethod
    def _dtype_matches(series: pd.Series, expected: str) -> bool:
        expected_normalized = expected.lower()
        if expected_normalized in {"number", "numeric"}:
            return pdt.is_numeric_dtype(series)
        if expected_normalized in {"int", "integer"}:
            return pdt.is_integer_dtype(series)
        if expected_normalized in {"float", "double"}:
            return pdt.is_float_dtype(series) or pdt.is_integer_dtype(series)
        if expected_normalized in {"str", "string", "object", "category"}:
            return pdt.is_string_dtype(series) or pdt.is_object_dtype(series) or pdt.is_categorical_dtype(series)
        if expected_normalized in {"bool", "boolean"}:
            return pdt.is_bool_dtype(series)
        if expected_normalized in {"datetime", "timestamp", "date"}:
            return pdt.is_datetime64_any_dtype(series)
        return str(series.dtype).lower() == expected_normalized
