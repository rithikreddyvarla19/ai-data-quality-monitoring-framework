from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from data_quality.utils import infer_column_groups, to_pandas
from drift_detection.evidently_adapter import run_evidently_data_drift
from drift_detection.metrics import (
    chi_square_test,
    jensen_shannon_divergence,
    ks_test,
    population_stability_index,
)
from monitoring.types import CheckResult, CheckSeverity, CheckStatus, MonitoringReport


@dataclass(slots=True)
class DriftThresholds:
    psi_warning: float = 0.10
    psi_critical: float = 0.25
    ks_p_value: float = 0.05
    chi_square_p_value: float = 0.05
    js_warning: float = 0.05
    js_critical: float = 0.15
    min_common_columns: int = 1
    use_evidently: bool = False


class DriftDetector:
    def __init__(
        self,
        reference_frame: Any,
        thresholds: DriftThresholds | None = None,
        dataset_name: str = "dataset",
    ) -> None:
        self.reference = to_pandas(reference_frame)
        self.thresholds = thresholds or DriftThresholds()
        self.dataset_name = dataset_name

    def detect(self, current_frame: Any) -> MonitoringReport:
        current = to_pandas(current_frame)
        report = MonitoringReport(dataset_name=self.dataset_name)
        common_columns = [column for column in self.reference.columns if column in current.columns]
        report.metadata.update(
            {
                "reference_rows": int(len(self.reference)),
                "current_rows": int(len(current)),
                "common_columns": common_columns,
            }
        )

        if len(common_columns) < self.thresholds.min_common_columns:
            report.add_check(
                CheckResult(
                    name="drift.common_columns",
                    status=CheckStatus.FAIL,
                    score=0.0,
                    severity=CheckSeverity.CRITICAL,
                    details={"common_columns": common_columns},
                )
            )
            return report

        feature_results = [self._feature_drift_check(column, current) for column in common_columns]
        report.checks.extend(feature_results)
        report.add_check(self._overall_drift_check(feature_results))
        if self.thresholds.use_evidently:
            report.add_check(run_evidently_data_drift(self.reference[common_columns], current[common_columns]))
        report.metrics["drift_score"] = report.score
        report.metrics["feature_drift"] = {
            check.name.replace("drift.feature.", ""): check.details for check in feature_results
        }
        return report

    def _feature_drift_check(self, column: str, current: pd.DataFrame) -> CheckResult:
        reference_series = self.reference[column]
        current_series = current[column]
        groups = infer_column_groups(self.reference[[column]])
        is_numeric = column in groups["numeric"] and pd.api.types.is_numeric_dtype(current_series)

        psi = population_stability_index(reference_series, current_series)
        jsd = jensen_shannon_divergence(reference_series, current_series)
        details: dict[str, Any] = {"psi": psi, "jensen_shannon_divergence": jsd}

        if is_numeric:
            ks = ks_test(reference_series, current_series)
            details["ks_test"] = ks
            status, severity, score = self._numeric_status(psi, jsd, ks["p_value"])
        else:
            chi = chi_square_test(reference_series, current_series)
            details["chi_square_test"] = chi
            status, severity, score = self._categorical_status(psi, jsd, chi["p_value"])

        return CheckResult(
            name=f"drift.feature.{column}",
            status=status,
            score=score,
            severity=severity,
            details=details,
        )

    def _numeric_status(self, psi: float, jsd: float, p_value: float) -> tuple[CheckStatus, CheckSeverity, float]:
        if psi >= self.thresholds.psi_critical or jsd >= self.thresholds.js_critical:
            return CheckStatus.FAIL, CheckSeverity.HIGH, 0.35
        if p_value < self.thresholds.ks_p_value or psi >= self.thresholds.psi_warning or jsd >= self.thresholds.js_warning:
            return CheckStatus.WARN, CheckSeverity.MEDIUM, 0.7
        return CheckStatus.PASS, CheckSeverity.INFO, 1.0

    def _categorical_status(self, psi: float, jsd: float, p_value: float) -> tuple[CheckStatus, CheckSeverity, float]:
        if psi >= self.thresholds.psi_critical or jsd >= self.thresholds.js_critical:
            return CheckStatus.FAIL, CheckSeverity.HIGH, 0.35
        if (
            p_value < self.thresholds.chi_square_p_value
            or psi >= self.thresholds.psi_warning
            or jsd >= self.thresholds.js_warning
        ):
            return CheckStatus.WARN, CheckSeverity.MEDIUM, 0.7
        return CheckStatus.PASS, CheckSeverity.INFO, 1.0

    @staticmethod
    def _overall_drift_check(feature_results: list[CheckResult]) -> CheckResult:
        failing = [check.name for check in feature_results if check.status == CheckStatus.FAIL]
        warning = [check.name for check in feature_results if check.status == CheckStatus.WARN]
        score = sum(check.score for check in feature_results) / max(len(feature_results), 1)
        if failing:
            status = CheckStatus.FAIL
            severity = CheckSeverity.HIGH
        elif warning:
            status = CheckStatus.WARN
            severity = CheckSeverity.MEDIUM
        else:
            status = CheckStatus.PASS
            severity = CheckSeverity.INFO
        return CheckResult(
            name="drift.overall",
            status=status,
            score=score,
            severity=severity,
            details={"failing_features": failing, "warning_features": warning},
        )

