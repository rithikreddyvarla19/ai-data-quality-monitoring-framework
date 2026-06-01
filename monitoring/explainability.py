from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class FeatureImportanceSnapshot:
    model_name: str
    importances: dict[str, float]

    def top_features(self, n: int = 10) -> dict[str, float]:
        return dict(sorted(self.importances.items(), key=lambda item: item[1], reverse=True)[:n])


class FeatureImportanceTracker:
    def extract(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        model_name: str = "model",
    ) -> FeatureImportanceSnapshot:
        if hasattr(model, "feature_importances_"):
            values = np.asarray(model.feature_importances_, dtype=float)
        elif hasattr(model, "coef_"):
            values = np.abs(np.asarray(model.coef_, dtype=float)).reshape(-1)[: X.shape[1]]
        elif y is not None:
            values = self._permutation_importance(model, X, y)
        else:
            values = np.zeros(X.shape[1], dtype=float)

        values = values / max(values.sum(), 1e-12)
        importances = {column: round(float(value), 6) for column, value in zip(X.columns, values, strict=False)}
        return FeatureImportanceSnapshot(model_name=model_name, importances=importances)

    @staticmethod
    def compare(
        baseline: FeatureImportanceSnapshot,
        current: FeatureImportanceSnapshot,
    ) -> dict[str, dict[str, float]]:
        features = sorted(set(baseline.importances) | set(current.importances))
        return {
            feature: {
                "baseline": baseline.importances.get(feature, 0.0),
                "current": current.importances.get(feature, 0.0),
                "delta": round(current.importances.get(feature, 0.0) - baseline.importances.get(feature, 0.0), 6),
            }
            for feature in features
        }

    @staticmethod
    def _permutation_importance(model: Any, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        try:
            from sklearn.inspection import permutation_importance  # type: ignore

            result = permutation_importance(model, X, y, n_repeats=5, random_state=42)
            return np.maximum(result.importances_mean, 0.0)
        except Exception:
            return np.zeros(X.shape[1], dtype=float)

