from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from monitoring.types import CheckResult, CheckSeverity, CheckStatus, MonitoringReport


@dataclass(slots=True)
class ModelMonitoringThresholds:
    min_accuracy: float = 0.80
    min_precision: float = 0.75
    min_recall: float = 0.75
    min_f1: float = 0.75
    max_expected_calibration_error: float = 0.08
    max_brier_score: float = 0.20


class ModelMonitor:
    def __init__(
        self,
        model_name: str = "model",
        thresholds: ModelMonitoringThresholds | None = None,
    ) -> None:
        self.model_name = model_name
        self.thresholds = thresholds or ModelMonitoringThresholds()

    def evaluate_classification(
        self,
        y_true: Iterable[Any],
        y_pred: Iterable[Any],
        y_probability: Iterable[float] | Iterable[Iterable[float]] | None = None,
    ) -> MonitoringReport:
        y_true_array = np.asarray(list(y_true))
        y_pred_array = np.asarray(list(y_pred))
        report = MonitoringReport(dataset_name=self.model_name)
        metrics = classification_metrics(y_true_array, y_pred_array)
        report.metrics.update(metrics)

        report.add_check(
            threshold_check("model.accuracy", metrics["accuracy"], self.thresholds.min_accuracy)
        )
        report.add_check(
            threshold_check("model.precision", metrics["precision"], self.thresholds.min_precision)
        )
        report.add_check(threshold_check("model.recall", metrics["recall"], self.thresholds.min_recall))
        report.add_check(threshold_check("model.f1", metrics["f1"], self.thresholds.min_f1))

        if y_probability is not None:
            probability_array = np.asarray(list(y_probability), dtype=float)
            calibration = calibration_metrics(y_true_array, probability_array)
            report.metrics.update(calibration)
            report.add_check(
                inverse_threshold_check(
                    "model.expected_calibration_error",
                    calibration["expected_calibration_error"],
                    self.thresholds.max_expected_calibration_error,
                )
            )
            if "brier_score" in calibration:
                report.add_check(
                    inverse_threshold_check(
                        "model.brier_score",
                        calibration["brier_score"],
                        self.thresholds.max_brier_score,
                    )
                )

        report.metrics["model_health_score"] = report.score
        return report


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    try:
        from sklearn.metrics import (  # type: ignore
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

        average = "binary" if len(np.unique(y_true)) <= 2 else "weighted"
        return {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
            "precision": round(float(precision_score(y_true, y_pred, average=average, zero_division=0)), 6),
            "recall": round(float(recall_score(y_true, y_pred, average=average, zero_division=0)), 6),
            "f1": round(float(f1_score(y_true, y_pred, average=average, zero_division=0)), 6),
        }
    except Exception:
        accuracy = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0
        labels = sorted(set(y_true) | set(y_pred), key=lambda value: str(value))
        per_label = []
        weights = []
        for label in labels:
            true_positive = int(((y_true == label) & (y_pred == label)).sum())
            false_positive = int(((y_true != label) & (y_pred == label)).sum())
            false_negative = int(((y_true == label) & (y_pred != label)).sum())
            precision = true_positive / max(true_positive + false_positive, 1)
            recall = true_positive / max(true_positive + false_negative, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            per_label.append((precision, recall, f1))
            weights.append(int((y_true == label).sum()))
        weights_array = np.asarray(weights, dtype=float)
        weights_array = weights_array / max(weights_array.sum(), 1.0)
        metric_array = np.asarray(per_label, dtype=float)
        weighted = (metric_array.T * weights_array).T.sum(axis=0)
        return {
            "accuracy": round(accuracy, 6),
            "precision": round(float(weighted[0]), 6),
            "recall": round(float(weighted[1]), 6),
            "f1": round(float(weighted[2]), 6),
        }


def calibration_metrics(y_true: np.ndarray, y_probability: np.ndarray, bins: int = 10) -> dict[str, float]:
    binary_true = _binary_labels(y_true)
    if y_probability.ndim == 2:
        confidence = y_probability.max(axis=1)
        predicted_class = y_probability.argmax(axis=1)
        correctness = (predicted_class == binary_true).astype(float)
    else:
        confidence = y_probability
        correctness = binary_true

    ece = expected_calibration_error(correctness, confidence, bins=bins)
    metrics = {"expected_calibration_error": round(ece, 6)}

    if y_probability.ndim == 1 and set(np.unique(binary_true)).issubset({0, 1}):
        try:
            from sklearn.metrics import brier_score_loss  # type: ignore

            metrics["brier_score"] = round(float(brier_score_loss(binary_true, y_probability)), 6)
        except Exception:
            metrics["brier_score"] = round(float(np.mean((binary_true - y_probability) ** 2)), 6)
    return metrics


def expected_calibration_error(y_true_binary: np.ndarray, confidence: np.ndarray, bins: int = 10) -> float:
    confidence = np.clip(confidence.astype(float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for start, end in zip(edges[:-1], edges[1:], strict=False):
        mask = (confidence > start) & (confidence <= end)
        if not mask.any():
            continue
        accuracy = float(np.mean(y_true_binary[mask]))
        avg_confidence = float(np.mean(confidence[mask]))
        ece += float(mask.mean()) * abs(accuracy - avg_confidence)
    return ece


def threshold_check(name: str, value: float, threshold: float) -> CheckResult:
    if value >= threshold:
        return CheckResult(name=name, status=CheckStatus.PASS, score=value, details={"value": value, "threshold": threshold})
    severity = CheckSeverity.HIGH if value < threshold * 0.8 else CheckSeverity.MEDIUM
    return CheckResult(
        name=name,
        status=CheckStatus.FAIL,
        score=value,
        severity=severity,
        details={"value": value, "threshold": threshold},
    )


def inverse_threshold_check(name: str, value: float, threshold: float) -> CheckResult:
    score = max(0.0, 1.0 - (value / max(threshold, 1e-12)))
    if value <= threshold:
        return CheckResult(name=name, status=CheckStatus.PASS, score=1.0 - value, details={"value": value, "threshold": threshold})
    severity = CheckSeverity.HIGH if value > threshold * 2 else CheckSeverity.MEDIUM
    return CheckResult(
        name=name,
        status=CheckStatus.FAIL,
        score=score,
        severity=severity,
        details={"value": value, "threshold": threshold},
    )


def _binary_labels(y_true: np.ndarray) -> np.ndarray:
    unique = sorted(np.unique(y_true), key=lambda value: str(value))
    if set(unique).issubset({0, 1}):
        return y_true.astype(int)
    mapping = {label: index for index, label in enumerate(unique)}
    return np.asarray([mapping[value] for value in y_true], dtype=int)


def evaluate_predictions_frame(
    predictions: pd.DataFrame,
    target_column: str = "actual",
    prediction_column: str = "prediction",
    probability_column: str | None = "probability",
    model_name: str = "model",
) -> MonitoringReport:
    probability = predictions[probability_column] if probability_column and probability_column in predictions else None
    return ModelMonitor(model_name=model_name).evaluate_classification(
        predictions[target_column],
        predictions[prediction_column],
        probability,
    )

