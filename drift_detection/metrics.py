from __future__ import annotations

from collections.abc import Iterable
from math import erf, sqrt
from typing import Any

import numpy as np
import pandas as pd

EPSILON = 1e-8


def _as_series(values: Iterable[Any] | pd.Series) -> pd.Series:
    return values if isinstance(values, pd.Series) else pd.Series(list(values))


def _numeric_histogram(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float)
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy(dtype=float)
    if len(ref) == 0 or len(cur) == 0:
        return np.array([1.0]), np.array([1.0])

    quantiles = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 3:
        lower = min(float(ref.min()), float(cur.min()))
        upper = max(float(ref.max()), float(cur.max()))
        if lower == upper:
            return np.array([1.0]), np.array([1.0])
        quantiles = np.linspace(lower, upper, bins + 1)

    ref_counts, _ = np.histogram(ref, bins=quantiles)
    cur_counts, _ = np.histogram(cur, bins=quantiles)
    return _normalize(ref_counts), _normalize(cur_counts)


def _categorical_distribution(
    reference: pd.Series,
    current: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    ref = reference.dropna().astype(str)
    cur = current.dropna().astype(str)
    categories = sorted(set(ref.unique()) | set(cur.unique()))
    if not categories:
        return np.array([1.0]), np.array([1.0])
    ref_counts = np.array([(ref == category).sum() for category in categories], dtype=float)
    cur_counts = np.array([(cur == category).sum() for category in categories], dtype=float)
    return _normalize(ref_counts), _normalize(cur_counts)


def _normalize(counts: np.ndarray) -> np.ndarray:
    counts = counts.astype(float) + EPSILON
    return counts / counts.sum()


def _distribution(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    if pd.api.types.is_numeric_dtype(reference) and pd.api.types.is_numeric_dtype(current):
        return _numeric_histogram(reference, current, bins=bins)
    return _categorical_distribution(reference, current)


def population_stability_index(
    expected: Iterable[Any] | pd.Series,
    actual: Iterable[Any] | pd.Series,
    bins: int = 10,
) -> float:
    expected_series = _as_series(expected)
    actual_series = _as_series(actual)
    expected_percents, actual_percents = _distribution(expected_series, actual_series, bins=bins)
    psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
    return round(float(np.sum(psi_values)), 6)


def jensen_shannon_divergence(
    reference: Iterable[Any] | pd.Series,
    current: Iterable[Any] | pd.Series,
    bins: int = 10,
) -> float:
    reference_series = _as_series(reference)
    current_series = _as_series(current)
    p, q = _distribution(reference_series, current_series, bins=bins)
    midpoint = 0.5 * (p + q)
    divergence = 0.5 * _kl_divergence(p, midpoint) + 0.5 * _kl_divergence(q, midpoint)
    return round(float(divergence), 6)


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(p * np.log((p + EPSILON) / (q + EPSILON))))


def ks_test(reference: Iterable[float] | pd.Series, current: Iterable[float] | pd.Series) -> dict[str, float]:
    ref = pd.to_numeric(_as_series(reference), errors="coerce").dropna().to_numpy(dtype=float)
    cur = pd.to_numeric(_as_series(current), errors="coerce").dropna().to_numpy(dtype=float)
    if len(ref) == 0 or len(cur) == 0:
        return {"statistic": 0.0, "p_value": 1.0}

    try:
        from scipy.stats import ks_2samp  # type: ignore

        result = ks_2samp(ref, cur)
        return {"statistic": round(float(result.statistic), 6), "p_value": round(float(result.pvalue), 6)}
    except Exception:
        ref_sorted = np.sort(ref)
        cur_sorted = np.sort(cur)
        values = np.sort(np.concatenate([ref_sorted, cur_sorted]))
        ref_cdf = np.searchsorted(ref_sorted, values, side="right") / len(ref_sorted)
        cur_cdf = np.searchsorted(cur_sorted, values, side="right") / len(cur_sorted)
        statistic = float(np.max(np.abs(ref_cdf - cur_cdf)))
        effective_n = len(ref) * len(cur) / (len(ref) + len(cur))
        p_value = min(1.0, 2.0 * np.exp(-2.0 * effective_n * statistic**2))
        return {"statistic": round(statistic, 6), "p_value": round(float(p_value), 6)}


def chi_square_test(reference: Iterable[Any] | pd.Series, current: Iterable[Any] | pd.Series) -> dict[str, float]:
    ref = _as_series(reference).dropna().astype(str)
    cur = _as_series(current).dropna().astype(str)
    categories = sorted(set(ref.unique()) | set(cur.unique()))
    if not categories:
        return {"statistic": 0.0, "p_value": 1.0}

    observed = np.array([(cur == category).sum() for category in categories], dtype=float)
    expected = np.array([(ref == category).sum() for category in categories], dtype=float)
    expected = expected * (observed.sum() / max(expected.sum(), EPSILON))
    expected = np.where(expected == 0, EPSILON, expected)

    try:
        from scipy.stats import chisquare  # type: ignore

        result = chisquare(f_obs=observed, f_exp=expected)
        return {"statistic": round(float(result.statistic), 6), "p_value": round(float(result.pvalue), 6)}
    except Exception:
        statistic = float(np.sum((observed - expected) ** 2 / expected))
        # Wilson-Hilferty normal approximation for a chi-square right-tail p-value.
        df = max(len(categories) - 1, 1)
        z_score = ((statistic / df) ** (1 / 3) - (1 - 2 / (9 * df))) / sqrt(2 / (9 * df))
        p_value = 0.5 * (1 - erf(z_score / sqrt(2)))
        return {"statistic": round(statistic, 6), "p_value": round(float(p_value), 6)}

