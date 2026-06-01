from __future__ import annotations

from dataclasses import dataclass

from monitoring.types import MonitoringReport


@dataclass(slots=True)
class DatasetTrustWeights:
    data_quality: float = 0.40
    drift: float = 0.30
    model_health: float = 0.20
    explainability: float = 0.10


class DatasetTrustScorer:
    def __init__(self, weights: DatasetTrustWeights | None = None) -> None:
        self.weights = weights or DatasetTrustWeights()

    def score(
        self,
        data_quality_report: MonitoringReport,
        drift_report: MonitoringReport | None = None,
        model_report: MonitoringReport | None = None,
        explainability_score: float | None = None,
    ) -> dict[str, float]:
        drift_score = drift_report.score if drift_report else 1.0
        model_score = model_report.score if model_report else 1.0
        explainability = 1.0 if explainability_score is None else explainability_score
        score = (
            data_quality_report.score * self.weights.data_quality
            + drift_score * self.weights.drift
            + model_score * self.weights.model_health
            + explainability * self.weights.explainability
        )
        return {
            "dataset_trust_score": round(score, 4),
            "data_quality_score": data_quality_report.score,
            "drift_score": drift_score,
            "model_health_score": model_score,
            "explainability_score": round(explainability, 4),
        }

