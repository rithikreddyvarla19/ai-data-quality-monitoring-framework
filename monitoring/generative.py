from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any

import numpy as np
import pandas as pd

from drift_detection.metrics import jensen_shannon_divergence, population_stability_index
from monitoring.types import CheckResult, CheckSeverity, CheckStatus, MonitoringReport


@dataclass(slots=True)
class GPTMonitoringThresholds:
    max_empty_response_rate: float = 0.02
    max_refusal_rate: float = 0.10
    min_grounding_overlap: float = 0.35
    max_p95_latency_ms: float = 5_000.0


class GPTSystemMonitor:
    def __init__(self, system_name: str = "gpt_system", thresholds: GPTMonitoringThresholds | None = None) -> None:
        self.system_name = system_name
        self.thresholds = thresholds or GPTMonitoringThresholds()

    def evaluate(self, records: list[dict[str, Any]]) -> MonitoringReport:
        report = MonitoringReport(dataset_name=self.system_name)
        responses = [str(record.get("response", "")) for record in records]
        contexts = [str(record.get("context", "")) for record in records]
        expected = [record.get("expected_response") for record in records]
        latencies = [float(record.get("latency_ms", 0.0)) for record in records if record.get("latency_ms") is not None]

        empty_rate = sum(not response.strip() for response in responses) / max(len(responses), 1)
        refusal_rate = sum(_looks_like_refusal(response) for response in responses) / max(len(responses), 1)
        grounding_scores = [
            token_overlap(response, context)
            for response, context in zip(responses, contexts, strict=False)
            if context.strip()
        ]
        exact_match_rate = _exact_match_rate(responses, expected)
        p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0

        report.metrics.update(
            {
                "request_count": len(records),
                "empty_response_rate": round(empty_rate, 6),
                "refusal_rate": round(refusal_rate, 6),
                "mean_grounding_overlap": round(mean(grounding_scores), 6) if grounding_scores else None,
                "exact_match_rate": exact_match_rate,
                "p95_latency_ms": round(p95_latency, 3),
            }
        )
        report.add_check(_max_rate_check("gpt.empty_response_rate", empty_rate, self.thresholds.max_empty_response_rate))
        report.add_check(_max_rate_check("gpt.refusal_rate", refusal_rate, self.thresholds.max_refusal_rate))
        if grounding_scores:
            report.add_check(
                _min_score_check(
                    "gpt.grounding_overlap",
                    mean(grounding_scores),
                    self.thresholds.min_grounding_overlap,
                )
            )
        if latencies:
            report.add_check(_max_rate_check("gpt.p95_latency_ms", p95_latency, self.thresholds.max_p95_latency_ms))
        report.metrics["gpt_health_score"] = report.score
        return report


class GenerativeModelMonitor:
    def compare_synthetic_data(
        self,
        reference: pd.DataFrame,
        synthetic: pd.DataFrame,
        dataset_name: str = "synthetic_model",
    ) -> MonitoringReport:
        report = MonitoringReport(dataset_name=dataset_name)
        common_columns = [column for column in reference.columns if column in synthetic.columns]
        feature_metrics: dict[str, dict[str, float]] = {}
        checks: list[CheckResult] = []

        for column in common_columns:
            psi = population_stability_index(reference[column], synthetic[column])
            jsd = jensen_shannon_divergence(reference[column], synthetic[column])
            feature_metrics[column] = {"psi": psi, "jensen_shannon_divergence": jsd}
            if psi >= 0.25 or jsd >= 0.15:
                checks.append(CheckResult(f"gan_vae.fidelity.{column}", CheckStatus.FAIL, 0.35, CheckSeverity.HIGH, feature_metrics[column]))
            elif psi >= 0.10 or jsd >= 0.05:
                checks.append(CheckResult(f"gan_vae.fidelity.{column}", CheckStatus.WARN, 0.7, CheckSeverity.MEDIUM, feature_metrics[column]))
            else:
                checks.append(CheckResult(f"gan_vae.fidelity.{column}", CheckStatus.PASS, 1.0, CheckSeverity.INFO, feature_metrics[column]))

        uniqueness = synthetic.drop_duplicates().shape[0] / max(len(synthetic), 1)
        checks.append(
            CheckResult(
                "gan_vae.mode_collapse",
                CheckStatus.FAIL if uniqueness < 0.5 else CheckStatus.PASS,
                uniqueness,
                CheckSeverity.HIGH if uniqueness < 0.5 else CheckSeverity.INFO,
                {"synthetic_uniqueness_rate": round(uniqueness, 6)},
            )
        )
        report.checks.extend(checks)
        report.metrics["synthetic_fidelity"] = feature_metrics
        report.metrics["mode_collapse_score"] = round(uniqueness, 6)
        report.metrics["generative_model_health_score"] = report.score
        return report

    def evaluate_reconstruction_errors(
        self,
        reconstruction_errors: list[float],
        threshold: float,
        dataset_name: str = "vae",
    ) -> MonitoringReport:
        errors = np.asarray(reconstruction_errors, dtype=float)
        report = MonitoringReport(dataset_name=dataset_name)
        p95 = float(np.percentile(errors, 95)) if len(errors) else 0.0
        mean_error = float(errors.mean()) if len(errors) else 0.0
        status = CheckStatus.PASS if p95 <= threshold else CheckStatus.FAIL
        report.add_check(
            CheckResult(
                "vae.reconstruction_error",
                status,
                max(0.0, 1.0 - (p95 / max(threshold, 1e-12))),
                CheckSeverity.HIGH if status == CheckStatus.FAIL else CheckSeverity.INFO,
                {"p95_reconstruction_error": round(p95, 6), "mean_reconstruction_error": round(mean_error, 6), "threshold": threshold},
            )
        )
        report.metrics["vae_health_score"] = report.score
        return report


def token_overlap(response: str, context: str) -> float:
    response_tokens = set(_normalize_text(response).split())
    context_tokens = set(_normalize_text(context).split())
    if not response_tokens or not context_tokens:
        return 0.0
    return len(response_tokens & context_tokens) / max(len(response_tokens), 1)


def _normalize_text(text: str) -> str:
    return "".join(char.lower() if char.isalnum() or char.isspace() else " " for char in text)


def _looks_like_refusal(response: str) -> bool:
    normalized = response.lower()
    refusal_markers = (
        "i can't",
        "i cannot",
        "unable to",
        "cannot assist",
        "i'm sorry",
        "sorry, but",
    )
    return any(marker in normalized for marker in refusal_markers)


def _exact_match_rate(responses: list[str], expected: list[Any]) -> float | None:
    pairs = [
        (_normalize_text(response).strip(), _normalize_text(str(answer)).strip())
        for response, answer in zip(responses, expected, strict=False)
        if answer is not None
    ]
    if not pairs:
        return None
    return round(sum(response == answer for response, answer in pairs) / len(pairs), 6)


def _max_rate_check(name: str, value: float, threshold: float) -> CheckResult:
    status = CheckStatus.PASS if value <= threshold else CheckStatus.FAIL
    return CheckResult(
        name,
        status,
        max(0.0, 1.0 - value / max(threshold, 1e-12)) if status == CheckStatus.FAIL else 1.0,
        CheckSeverity.HIGH if status == CheckStatus.FAIL else CheckSeverity.INFO,
        {"value": round(value, 6), "threshold": threshold},
    )


def _min_score_check(name: str, value: float, threshold: float) -> CheckResult:
    status = CheckStatus.PASS if value >= threshold else CheckStatus.FAIL
    return CheckResult(
        name,
        status,
        max(0.0, value / max(threshold, 1e-12)),
        CheckSeverity.HIGH if status == CheckStatus.FAIL else CheckSeverity.INFO,
        {"value": round(value, 6), "threshold": threshold},
    )

