from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_quality.engine import DataQualityConfig, DataQualityEngine  # noqa: E402
from drift_detection.detector import DriftDetector  # noqa: E402
from monitoring.leakage import LabelLeakageDetector  # noqa: E402
from monitoring.model_monitor import evaluate_predictions_frame  # noqa: E402
from monitoring.quality_scoring import DatasetTrustScorer  # noqa: E402

st.set_page_config(
    page_title="AI Data Quality Monitoring",
    page_icon="",
    layout="wide",
)

SAMPLE_DIR = ROOT / "data" / "sample"


@st.cache_data
def load_sample_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(SAMPLE_DIR / "reference_dataset.csv"),
        pd.read_csv(SAMPLE_DIR / "current_dataset.csv"),
        pd.read_csv(SAMPLE_DIR / "model_predictions.csv"),
    )


def check_table(report_dict: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "check": check["name"],
                "status": check["status"],
                "severity": check["severity"],
                "score": check["score"],
            }
            for check in report_dict["checks"]
        ]
    )


reference_df, current_df, predictions_df = load_sample_data()

schema = [
    {"name": "customer_id", "dtype": "integer", "nullable": False},
    {"name": "age", "dtype": "integer", "nullable": False, "min_value": 18, "max_value": 100},
    {"name": "income", "dtype": "float", "nullable": False, "min_value": 0},
    {"name": "tenure_months", "dtype": "integer", "nullable": False, "min_value": 0},
    {"name": "region", "dtype": "string", "nullable": False, "allowed_values": ["North", "South", "East", "West"]},
    {"name": "plan_type", "dtype": "string", "nullable": False, "allowed_values": ["basic", "plus", "enterprise"]},
    {"name": "monthly_usage", "dtype": "float", "nullable": False, "min_value": 0},
    {"name": "support_tickets", "dtype": "integer", "nullable": False, "min_value": 0},
    {"name": "churn", "dtype": "integer", "nullable": False, "allowed_values": [0, 1]},
]

quality_report = DataQualityEngine(
    DataQualityConfig(dataset_name="customer_churn_current", schema=schema)
).validate(current_df)
quality_report.add_check(LabelLeakageDetector().detect(current_df, target_column="churn"))
drift_report = DriftDetector(reference_df, dataset_name="customer_churn").detect(current_df)
model_report = evaluate_predictions_frame(predictions_df, model_name="churn_classifier")
trust_score = DatasetTrustScorer().score(quality_report, drift_report, model_report)

st.title("AI Data Quality Monitoring")

metric_cols = st.columns(5)
metric_cols[0].metric("Data Quality Score", f"{quality_report.score:.2%}")
metric_cols[1].metric("Dataset Trust Score", f"{trust_score['dataset_trust_score']:.2%}")
metric_cols[2].metric("Drift Score", f"{drift_report.score:.2%}")
metric_cols[3].metric("Model Health", f"{model_report.score:.2%}")
metric_cols[4].metric("Open Alerts", len(quality_report.alerts + drift_report.alerts + model_report.alerts))

tabs = st.tabs(["Quality", "Drift", "Model Health", "Feature Trends", "Datasets"])

with tabs[0]:
    left, right = st.columns([1.2, 1])
    quality_checks = check_table(quality_report.to_dict())
    left.subheader("Validation Results")
    left.dataframe(quality_checks, use_container_width=True, hide_index=True)
    right.subheader("Column Missingness")
    missing_profile = quality_report.to_dict()["checks"]
    missing_check = next((check for check in missing_profile if check["name"] == "data_quality.missing_values"), None)
    if missing_check:
        missing_df = pd.DataFrame.from_dict(missing_check["details"]["profile"], orient="index").reset_index(names="column")
        right.bar_chart(missing_df, x="column", y="missing_rate", use_container_width=True)

with tabs[1]:
    drift_checks = check_table(drift_report.to_dict())
    st.subheader("Feature Drift Summary")
    st.dataframe(drift_checks, use_container_width=True, hide_index=True)
    feature_metrics = drift_report.metrics["feature_drift"]
    drift_rows = [
        {
            "feature": feature,
            "psi": metrics.get("psi", 0.0),
            "jensen_shannon_divergence": metrics.get("jensen_shannon_divergence", 0.0),
        }
        for feature, metrics in feature_metrics.items()
        if feature != "overall"
    ]
    if drift_rows:
        drift_df = pd.DataFrame(drift_rows)
        st.plotly_chart(
            px.bar(drift_df, x="feature", y=["psi", "jensen_shannon_divergence"], barmode="group"),
            use_container_width=True,
        )

with tabs[2]:
    model_checks = check_table(model_report.to_dict())
    st.subheader("Model Health Metrics")
    st.dataframe(model_checks, use_container_width=True, hide_index=True)
    model_metric_df = pd.DataFrame(
        [
            {"metric": key, "value": value}
            for key, value in model_report.metrics.items()
            if isinstance(value, int | float)
        ]
    )
    st.plotly_chart(px.bar(model_metric_df, x="metric", y="value"), use_container_width=True)

with tabs[3]:
    st.subheader("Feature Importance Trends")
    trend_df = pd.DataFrame(
        {
            "feature": ["monthly_usage", "tenure_months", "support_tickets", "income", "plan_type"],
            "baseline": [0.31, 0.23, 0.19, 0.14, 0.13],
            "current": [0.28, 0.19, 0.25, 0.13, 0.15],
        }
    )
    st.plotly_chart(px.line(trend_df, x="feature", y=["baseline", "current"], markers=True), use_container_width=True)

with tabs[4]:
    col_a, col_b = st.columns(2)
    col_a.subheader("Reference Dataset")
    col_a.dataframe(reference_df, use_container_width=True, hide_index=True)
    col_b.subheader("Current Dataset")
    col_b.dataframe(current_df, use_container_width=True, hide_index=True)
