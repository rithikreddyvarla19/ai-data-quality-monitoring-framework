from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from monitoring.types import CheckResult, CheckSeverity, CheckStatus


@dataclass(slots=True)
class OutlierConfig:
    method: str = "iqr"
    iqr_multiplier: float = 1.5
    zscore_threshold: float = 3.0
    max_outlier_rate: float = 0.05


def detect_outliers(df: pd.DataFrame, config: OutlierConfig | None = None) -> dict[str, dict[str, float | int]]:
    config = config or OutlierConfig()
    numeric_df = df.select_dtypes(include=["number"])
    profile: dict[str, dict[str, float | int]] = {}

    for column in numeric_df.columns:
        series = numeric_df[column].dropna()
        if series.empty:
            profile[column] = {"outlier_count": 0, "outlier_rate": 0.0}
            continue

        if config.method == "zscore":
            std = float(series.std(ddof=0))
            if std == 0.0:
                mask = pd.Series(False, index=series.index)
            else:
                zscores = np.abs((series - float(series.mean())) / std)
                mask = zscores > config.zscore_threshold
        else:
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            lower = q1 - (config.iqr_multiplier * iqr)
            upper = q3 + (config.iqr_multiplier * iqr)
            mask = (series < lower) | (series > upper)

        outlier_count = int(mask.sum())
        profile[column] = {
            "outlier_count": outlier_count,
            "outlier_rate": round(outlier_count / max(len(df), 1), 6),
        }

    return profile


def outlier_check(df: pd.DataFrame, config: OutlierConfig | None = None) -> CheckResult:
    config = config or OutlierConfig()
    profile = detect_outliers(df, config)
    failing_columns = {
        column: metrics
        for column, metrics in profile.items()
        if float(metrics["outlier_rate"]) > config.max_outlier_rate
    }
    worst_rate = max((float(metrics["outlier_rate"]) for metrics in profile.values()), default=0.0)
    score = max(0.0, 1.0 - worst_rate)

    if not failing_columns:
        status = CheckStatus.PASS
        severity = CheckSeverity.INFO
    elif worst_rate <= config.max_outlier_rate * 2:
        status = CheckStatus.WARN
        severity = CheckSeverity.MEDIUM
    else:
        status = CheckStatus.FAIL
        severity = CheckSeverity.HIGH

    return CheckResult(
        name="data_quality.outliers",
        status=status,
        score=score,
        severity=severity,
        details={
            "method": config.method,
            "max_outlier_rate": config.max_outlier_rate,
            "worst_outlier_rate": round(worst_rate, 6),
            "failing_columns": failing_columns,
            "profile": profile,
        },
    )

