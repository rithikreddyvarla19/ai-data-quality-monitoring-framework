from __future__ import annotations

import logging
import os
from io import StringIO
from typing import Any

import pandas as pd

from alerts.dispatcher import AlertManager
from data_quality.engine import DataQualityConfig, DataQualityEngine
from drift_detection.detector import DriftDetector, DriftThresholds
from monitoring.generative import GPTSystemMonitor
from monitoring.leakage import LabelLeakageDetector
from monitoring.model_monitor import ModelMonitor, evaluate_predictions_frame
from monitoring.quality_scoring import DatasetTrustScorer
from monitoring.storage import ReportStore
from monitoring.types import MonitoringReport

LOGGER = logging.getLogger(__name__)


def records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records)


def csv_bytes_to_frame(payload: bytes) -> pd.DataFrame:
    return pd.read_csv(StringIO(payload.decode("utf-8")))


def run_data_quality(
    dataset_name: str,
    records: list[dict[str, Any]],
    schema_fields: list[dict[str, Any]] | None = None,
    target_column: str | None = None,
) -> MonitoringReport:
    df = records_to_frame(records)
    engine = DataQualityEngine(
        DataQualityConfig(
            dataset_name=dataset_name,
            schema=schema_fields or [],
        )
    )
    report = engine.validate(df)
    if target_column:
        report.add_check(LabelLeakageDetector().detect(df, target_column=target_column))
        report.metrics["data_quality_score"] = report.score
    AlertManager().evaluate(report)
    persist_report_if_configured(report, "data_quality")
    return report


def run_drift(
    dataset_name: str,
    reference_records: list[dict[str, Any]],
    current_records: list[dict[str, Any]],
    use_evidently: bool = False,
) -> MonitoringReport:
    reference = records_to_frame(reference_records)
    current = records_to_frame(current_records)
    report = DriftDetector(
        reference,
        thresholds=DriftThresholds(use_evidently=use_evidently),
        dataset_name=dataset_name,
    ).detect(current)
    AlertManager().evaluate(report)
    persist_report_if_configured(report, "drift")
    return report


def run_model_health(
    model_name: str,
    y_true: list[Any],
    y_pred: list[Any],
    y_probability: list[float] | list[list[float]] | None = None,
) -> MonitoringReport:
    report = ModelMonitor(model_name=model_name).evaluate_classification(y_true, y_pred, y_probability)
    AlertManager().evaluate(report)
    persist_report_if_configured(report, "model_health")
    return report


def run_gpt_monitoring(system_name: str, records: list[dict[str, Any]]) -> MonitoringReport:
    report = GPTSystemMonitor(system_name=system_name).evaluate(records)
    AlertManager().evaluate(report)
    persist_report_if_configured(report, "gpt_monitoring")
    return report


def run_full_monitoring(payload: dict[str, Any]) -> dict[str, Any]:
    data_quality = run_data_quality(
        dataset_name=payload["dataset_name"],
        records=payload["current_records"],
        schema_fields=payload.get("schema_fields", []),
        target_column=payload.get("target_column"),
    )
    drift = run_drift(
        dataset_name=payload["dataset_name"],
        reference_records=payload["reference_records"],
        current_records=payload["current_records"],
    )
    model_report = None
    prediction_records = payload.get("prediction_records") or []
    if prediction_records:
        prediction_frame = records_to_frame(prediction_records)
        model_report = evaluate_predictions_frame(
            prediction_frame,
            target_column=payload.get("actual_column", "actual"),
            prediction_column=payload.get("prediction_column", "prediction"),
            probability_column=payload.get("probability_column", "probability"),
            model_name=f"{payload['dataset_name']}_model",
        )
        AlertManager().evaluate(model_report)
        persist_report_if_configured(model_report, "model_health")

    trust_score = DatasetTrustScorer().score(data_quality, drift, model_report)
    return {
        "trust_score": trust_score,
        "data_quality": data_quality.to_dict(),
        "drift": drift.to_dict(),
        "model_health": model_report.to_dict() if model_report else None,
    }


def persist_report_if_configured(report: MonitoringReport, report_type: str) -> None:
    should_persist = os.getenv("PERSIST_REPORTS", "false").lower() == "true"
    database_url = os.getenv("DATABASE_URL")
    if not should_persist or not database_url:
        return
    try:
        ReportStore(database_url).save_report(report, report_type=report_type)
    except Exception as exc:
        LOGGER.warning("Failed to persist %s report %s: %s", report_type, report.run_id, exc)
