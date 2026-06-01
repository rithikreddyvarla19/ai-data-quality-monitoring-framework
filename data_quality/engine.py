from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from data_quality.great_expectations_adapter import run_great_expectations_checks
from data_quality.missing import missing_value_check
from data_quality.outliers import OutlierConfig, outlier_check
from data_quality.schema import SchemaField, SchemaValidator
from data_quality.utils import to_pandas
from monitoring.types import CheckResult, CheckSeverity, CheckStatus, MonitoringReport


@dataclass(slots=True)
class DataQualityConfig:
    dataset_name: str = "dataset"
    schema: list[SchemaField | dict[str, Any]] = field(default_factory=list)
    missing_threshold: float = 0.05
    duplicate_threshold: float = 0.01
    min_rows: int = 1
    outlier_config: OutlierConfig = field(default_factory=OutlierConfig)
    great_expectations: list[dict[str, Any]] = field(default_factory=list)


class DataQualityEngine:
    def __init__(self, config: DataQualityConfig | None = None) -> None:
        self.config = config or DataQualityConfig()

    def validate(self, frame: Any) -> MonitoringReport:
        df = to_pandas(frame)
        report = MonitoringReport(dataset_name=self.config.dataset_name)
        report.metadata.update(
            {
                "row_count": int(len(df)),
                "column_count": int(len(df.columns)),
                "columns": df.columns.tolist(),
            }
        )

        if self.config.schema:
            report.checks.extend(SchemaValidator(self.config.schema).validate(df))

        report.add_check(self._row_count_check(df))
        report.add_check(self._duplicate_check(df))
        report.add_check(missing_value_check(df, threshold=self.config.missing_threshold))
        report.add_check(outlier_check(df, self.config.outlier_config))
        report.add_check(run_great_expectations_checks(df, self.config.great_expectations))
        report.metrics["data_quality_score"] = report.score
        report.metrics["dataset_integrity"] = self._dataset_integrity_metrics(df)
        return report

    def _row_count_check(self, df: pd.DataFrame) -> CheckResult:
        if len(df) >= self.config.min_rows:
            return CheckResult(
                name="data_quality.row_count",
                status=CheckStatus.PASS,
                score=1.0,
                details={"row_count": int(len(df)), "min_rows": self.config.min_rows},
            )
        return CheckResult(
            name="data_quality.row_count",
            status=CheckStatus.FAIL,
            score=0.0,
            severity=CheckSeverity.CRITICAL,
            details={"row_count": int(len(df)), "min_rows": self.config.min_rows},
        )

    def _duplicate_check(self, df: pd.DataFrame) -> CheckResult:
        duplicate_rate = float(df.duplicated().mean()) if len(df) else 0.0
        score = max(0.0, 1.0 - duplicate_rate)
        if duplicate_rate <= self.config.duplicate_threshold:
            status = CheckStatus.PASS
            severity = CheckSeverity.INFO
        elif duplicate_rate <= self.config.duplicate_threshold * 2:
            status = CheckStatus.WARN
            severity = CheckSeverity.MEDIUM
        else:
            status = CheckStatus.FAIL
            severity = CheckSeverity.HIGH
        return CheckResult(
            name="data_quality.duplicates",
            status=status,
            score=score,
            severity=severity,
            details={
                "duplicate_count": int(df.duplicated().sum()),
                "duplicate_rate": round(duplicate_rate, 6),
                "threshold": self.config.duplicate_threshold,
            },
        )

    @staticmethod
    def _dataset_integrity_metrics(df: pd.DataFrame) -> dict[str, Any]:
        return {
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "memory_usage_mb": round(float(df.memory_usage(deep=True).sum()) / 1_000_000, 4),
            "numeric_columns": df.select_dtypes(include=["number"]).columns.tolist(),
            "categorical_columns": df.select_dtypes(exclude=["number"]).columns.tolist(),
        }

