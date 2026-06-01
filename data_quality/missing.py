from __future__ import annotations

import pandas as pd

from monitoring.types import CheckResult, CheckSeverity, CheckStatus


def missing_value_profile(df: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    total = max(len(df), 1)
    return {
        column: {
            "missing_count": int(df[column].isna().sum()),
            "missing_rate": round(float(df[column].isna().mean()), 6),
            "non_null_count": int(total - df[column].isna().sum()),
        }
        for column in df.columns
    }


def missing_value_check(df: pd.DataFrame, threshold: float = 0.05) -> CheckResult:
    profile = missing_value_profile(df)
    failing_columns = {
        column: metrics
        for column, metrics in profile.items()
        if float(metrics["missing_rate"]) > threshold
    }
    worst_missing_rate = max((float(metrics["missing_rate"]) for metrics in profile.values()), default=0.0)
    score = max(0.0, 1.0 - worst_missing_rate)

    if not failing_columns:
        status = CheckStatus.PASS
        severity = CheckSeverity.INFO
    elif worst_missing_rate <= threshold * 2:
        status = CheckStatus.WARN
        severity = CheckSeverity.MEDIUM
    else:
        status = CheckStatus.FAIL
        severity = CheckSeverity.HIGH

    return CheckResult(
        name="data_quality.missing_values",
        status=status,
        score=score,
        severity=severity,
        details={
            "threshold": threshold,
            "worst_missing_rate": round(worst_missing_rate, 6),
            "failing_columns": failing_columns,
            "profile": profile,
        },
    )

