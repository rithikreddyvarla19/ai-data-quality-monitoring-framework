from __future__ import annotations

import pandas as pd

from monitoring.generative import GenerativeModelMonitor, GPTSystemMonitor, token_overlap
from monitoring.model_monitor import ModelMonitor
from monitoring.types import CheckStatus


def test_model_monitor_classification_metrics() -> None:
    report = ModelMonitor(model_name="unit").evaluate_classification(
        y_true=[0, 1, 1, 0],
        y_pred=[0, 1, 0, 0],
        y_probability=[0.10, 0.89, 0.42, 0.21],
    )

    assert report.metrics["accuracy"] == 0.75
    assert "expected_calibration_error" in report.metrics
    assert any(check.name == "model.accuracy" for check in report.checks)


def test_gpt_monitoring_scores_grounding_overlap() -> None:
    records = [
        {
            "prompt": "Which plan includes SSO?",
            "context": "The enterprise plan includes SSO and audit logs.",
            "response": "The enterprise plan includes SSO.",
            "latency_ms": 700,
        }
    ]

    report = GPTSystemMonitor(system_name="unit").evaluate(records)

    assert report.status == CheckStatus.PASS
    assert report.metrics["mean_grounding_overlap"] is not None
    assert token_overlap("enterprise plan includes sso", "enterprise plan includes sso and logs") > 0.5


def test_generative_model_monitor_flags_mode_collapse() -> None:
    reference = pd.DataFrame({"x": [1, 2, 3, 4, 5], "segment": ["a", "b", "c", "d", "e"]})
    synthetic = pd.DataFrame({"x": [1, 1, 1, 1, 1], "segment": ["a", "a", "a", "a", "a"]})

    report = GenerativeModelMonitor().compare_synthetic_data(reference, synthetic)

    assert any(check.name == "gan_vae.mode_collapse" for check in report.checks)
    assert report.status == CheckStatus.FAIL

