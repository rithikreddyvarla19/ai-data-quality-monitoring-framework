from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from monitoring.types import CheckResult, CheckSeverity, CheckStatus


@dataclass(slots=True)
class LeakageThresholds:
    max_numeric_correlation: float = 0.98
    max_mutual_information: float = 0.90
    suspicious_name_tokens: tuple[str, ...] = ("target", "label", "outcome", "future", "post_", "leak")


class LabelLeakageDetector:
    def __init__(self, thresholds: LeakageThresholds | None = None) -> None:
        self.thresholds = thresholds or LeakageThresholds()

    def detect(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: list[str] | None = None,
    ) -> CheckResult:
        if target_column not in df.columns:
            return CheckResult(
                name="data_quality.label_leakage",
                status=CheckStatus.FAIL,
                score=0.0,
                severity=CheckSeverity.CRITICAL,
                details={"reason": "target column missing", "target_column": target_column},
            )

        features = feature_columns or [column for column in df.columns if column != target_column]
        findings: list[dict[str, Any]] = []
        target = df[target_column]

        for feature in features:
            if feature not in df.columns:
                continue
            series = df[feature]
            if _is_exact_copy(series, target):
                findings.append({"feature": feature, "type": "exact_target_copy", "score": 1.0})
            if _has_suspicious_name(feature, target_column, self.thresholds.suspicious_name_tokens):
                findings.append({"feature": feature, "type": "suspicious_name", "score": 0.75})
            if pd.api.types.is_numeric_dtype(series) and pd.api.types.is_numeric_dtype(target):
                corr = abs(float(pd.to_numeric(series, errors="coerce").corr(pd.to_numeric(target, errors="coerce"))))
                if np.isfinite(corr) and corr >= self.thresholds.max_numeric_correlation:
                    findings.append({"feature": feature, "type": "high_target_correlation", "score": round(corr, 6)})
            else:
                mutual_information = normalized_mutual_information(series, target)
                if mutual_information >= self.thresholds.max_mutual_information:
                    findings.append(
                        {
                            "feature": feature,
                            "type": "high_mutual_information",
                            "score": round(mutual_information, 6),
                        }
                    )

        if not findings:
            return CheckResult(
                name="data_quality.label_leakage",
                status=CheckStatus.PASS,
                score=1.0,
                details={"findings": []},
            )

        score = max(0.0, 1.0 - min(1.0, len(findings) * 0.25))
        severity = CheckSeverity.CRITICAL if any(item["type"] == "exact_target_copy" for item in findings) else CheckSeverity.HIGH
        return CheckResult(
            name="data_quality.label_leakage",
            status=CheckStatus.FAIL,
            score=score,
            severity=severity,
            details={"findings": findings},
        )


def normalized_mutual_information(feature: pd.Series, target: pd.Series) -> float:
    frame = pd.DataFrame({"feature": feature.astype(str), "target": target.astype(str)}).dropna()
    if frame.empty:
        return 0.0
    try:
        from sklearn.metrics import normalized_mutual_info_score  # type: ignore

        return float(normalized_mutual_info_score(frame["target"], frame["feature"]))
    except Exception:
        joint = pd.crosstab(frame["feature"], frame["target"], normalize=True)
        feature_prob = joint.sum(axis=1).to_numpy()
        target_prob = joint.sum(axis=0).to_numpy()
        joint_prob = joint.to_numpy()
        mi = 0.0
        for i in range(joint_prob.shape[0]):
            for j in range(joint_prob.shape[1]):
                pxy = joint_prob[i, j]
                if pxy > 0:
                    mi += pxy * np.log(pxy / (feature_prob[i] * target_prob[j]))
        entropy_feature = -float(np.sum(feature_prob * np.log(feature_prob + 1e-12)))
        entropy_target = -float(np.sum(target_prob * np.log(target_prob + 1e-12)))
        return float(mi / max((entropy_feature + entropy_target) / 2, 1e-12))


def _is_exact_copy(feature: pd.Series, target: pd.Series) -> bool:
    aligned = pd.DataFrame({"feature": feature, "target": target}).dropna()
    return not aligned.empty and bool((aligned["feature"].astype(str) == aligned["target"].astype(str)).all())


def _has_suspicious_name(feature: str, target: str, tokens: tuple[str, ...]) -> bool:
    feature_lower = feature.lower()
    target_lower = target.lower()
    if feature_lower == target_lower:
        return True
    if target_lower in feature_lower and feature_lower != target_lower:
        return True
    return any(token in feature_lower for token in tokens)

