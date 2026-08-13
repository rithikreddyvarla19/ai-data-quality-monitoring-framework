from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, ks_2samp

from tabular_monitor.types import Check, Report

LOGGER = logging.getLogger(__name__)
EPS = 1e-6


@dataclass(slots=True)
class DriftConfig:
    dataset: str = "dataset"
    bins: int = 10
    min_samples: int = 30
    max_categories: int = 50
    rare_category_rate: float = 0.01
    psi_warning: float = 0.1
    psi_failure: float = 0.25

    def __post_init__(self) -> None:
        if self.bins < 2 or self.min_samples < 2 or self.max_categories < 2:
            raise ValueError("invalid drift configuration")


class DriftDetector:
    def __init__(self, reference: pd.DataFrame, config: DriftConfig | None = None) -> None:
        if not isinstance(reference, pd.DataFrame):
            raise TypeError("reference must be a pandas DataFrame")
        self.reference = reference.copy()
        self.config = config or DriftConfig()

    def detect(self, current: pd.DataFrame) -> Report:
        if not isinstance(current, pd.DataFrame):
            raise TypeError("current must be a pandas DataFrame")
        missing = sorted(set(self.reference) - set(current))
        extra = sorted(set(current) - set(self.reference))
        report = Report(
            self.config.dataset,
            metadata={
                "reference_rows": len(self.reference),
                "current_rows": len(current),
                "missing_columns": missing,
                "extra_columns": extra,
            },
        )
        report.checks.append(
            Check(
                "drift.columns", "fail" if missing else "pass", {"missing": missing, "extra": extra}
            )
        )
        for column in [c for c in self.reference if c in current]:
            report.checks.append(self._feature(column, current[column]))
        LOGGER.info(
            "drift_finished", extra={"dataset": self.config.dataset, "passed": report.passed}
        )
        return report

    def _feature(self, column: str, current: pd.Series) -> Check:
        ref = self.reference[column]
        if pd.api.types.is_numeric_dtype(ref) and pd.api.types.is_numeric_dtype(current):
            a = _finite(ref)
            b = _finite(current)
            if min(len(a), len(b)) < self.config.min_samples:
                return Check(
                    f"drift.{column}",
                    "skipped",
                    {
                        "reason": "insufficient_samples",
                        "reference_samples": len(a),
                        "current_samples": len(b),
                    },
                )
            p, q, edges = _numeric_distributions(a, b, self.config.bins)
            psi = _psi(p, q)
            ks = ks_2samp(a, b)
            details = {
                "psi": psi,
                "js_divergence": float(jensenshannon(p, q) ** 2),
                "ks_statistic": float(ks.statistic),
                "ks_p_value": float(ks.pvalue),
                "bin_count": len(edges) - 1,
            }
        else:
            a = ref.dropna().astype(str)
            b = current.dropna().astype(str)
            if min(len(a), len(b)) < self.config.min_samples:
                return Check(
                    f"drift.{column}",
                    "skipped",
                    {
                        "reason": "insufficient_samples",
                        "reference_samples": len(a),
                        "current_samples": len(b),
                    },
                )
            p, q, categories, collapsed = _categorical_distributions(
                a, b, self.config.max_categories, self.config.rare_category_rate
            )
            table = np.vstack((p, q))
            chi = chi2_contingency(table, correction=False)
            psi = _psi(p, q)
            details = {
                "psi": psi,
                "js_divergence": float(jensenshannon(p, q) ** 2),
                "chi_square_p_value": float(chi.pvalue),
                "category_count": len(categories),
                "rare_values_collapsed": collapsed,
            }
        p_value = details.get("ks_p_value", details.get("chi_square_p_value", 1.0))
        status = (
            "fail"
            if psi >= self.config.psi_failure
            else "warn"
            if psi >= self.config.psi_warning or p_value < 0.05
            else "pass"
        )
        return Check(f"drift.{column}", status, details)


def _finite(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    return values[np.isfinite(values)]


def _numeric_distributions(
    ref: np.ndarray, cur: np.ndarray, bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        edges = np.array([ref[0] - 0.5, ref[0] + 0.5])
    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return _norm(np.histogram(ref, edges)[0]), _norm(np.histogram(cur, edges)[0]), edges


def _categorical_distributions(
    ref: pd.Series, cur: pd.Series, maximum: int, rare_rate: float
) -> tuple[np.ndarray, np.ndarray, list[str], bool]:
    freq = ref.value_counts(normalize=True)
    keep = list(freq[freq >= rare_rate].index[: maximum - 1])
    all_values = set(ref) | set(cur)
    collapsed = bool(all_values - set(keep))

    def map_values(s: pd.Series) -> pd.Series:
        return s.where(s.isin(keep), "__OTHER__")

    categories = sorted(set(map_values(ref)) | set(map_values(cur)))
    return (
        _norm(map_values(ref).value_counts().reindex(categories, fill_value=0).to_numpy()),
        _norm(map_values(cur).value_counts().reindex(categories, fill_value=0).to_numpy()),
        categories,
        collapsed,
    )


def _norm(values: np.ndarray) -> np.ndarray:
    values = values.astype(float) + EPS
    return values / values.sum()


def _psi(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum((q - p) * np.log(q / p)))
