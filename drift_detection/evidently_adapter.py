from __future__ import annotations

from typing import Any

import pandas as pd

from monitoring.types import CheckResult, CheckSeverity, CheckStatus


def run_evidently_data_drift(reference: pd.DataFrame, current: pd.DataFrame) -> CheckResult:
    """Optional Evidently AI adapter.

    Evidently APIs have changed across releases, so the core platform computes
    deterministic drift metrics locally and treats Evidently as a report plugin.
    """
    try:
        from evidently.metric_preset import DataDriftPreset  # type: ignore
        from evidently.report import Report  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency path
        return CheckResult(
            name="drift.evidently",
            status=CheckStatus.SKIP,
            score=1.0,
            severity=CheckSeverity.INFO,
            details={"reason": "evidently unavailable", "error": str(exc)},
        )

    try:  # pragma: no cover - optional dependency path
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)
        payload: dict[str, Any] = report.as_dict()
        metrics = payload.get("metrics", [])
        drifted = _extract_drift_flag(metrics)
        return CheckResult(
            name="drift.evidently",
            status=CheckStatus.FAIL if drifted else CheckStatus.PASS,
            score=0.7 if drifted else 1.0,
            severity=CheckSeverity.HIGH if drifted else CheckSeverity.INFO,
            details={"evidently_summary": payload},
        )
    except Exception as exc:
        return CheckResult(
            name="drift.evidently",
            status=CheckStatus.WARN,
            score=0.8,
            severity=CheckSeverity.LOW,
            details={"reason": "evidently execution failed", "error": str(exc)},
        )


def _extract_drift_flag(metrics: list[dict[str, Any]]) -> bool:
    for metric in metrics:
        result = metric.get("result", {})
        if result.get("dataset_drift") is True:
            return True
    return False

