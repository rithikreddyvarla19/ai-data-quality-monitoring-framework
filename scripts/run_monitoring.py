from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an end-to-end AI data quality monitoring job.")
    parser.add_argument("--reference", required=True, help="Path to the reference dataset CSV.")
    parser.add_argument("--current", required=True, help="Path to the current dataset CSV.")
    parser.add_argument("--predictions", required=False, help="Path to model predictions CSV.")
    parser.add_argument("--target-column", default="churn", help="Ground-truth label column for leakage checks.")
    parser.add_argument("--output", default="outputs/monitoring_report.json", help="Output report path.")
    return parser.parse_args()


def main() -> None:
    from data_quality.engine import DataQualityConfig, DataQualityEngine
    from drift_detection.detector import DriftDetector
    from monitoring.leakage import LabelLeakageDetector
    from monitoring.model_monitor import evaluate_predictions_frame
    from monitoring.quality_scoring import DatasetTrustScorer

    args = parse_args()
    reference = pd.read_csv(args.reference)
    current = pd.read_csv(args.current)

    data_quality = DataQualityEngine(
        DataQualityConfig(dataset_name=Path(args.current).stem)
    ).validate(current)
    if args.target_column in current.columns:
        data_quality.add_check(LabelLeakageDetector().detect(current, args.target_column))

    drift = DriftDetector(reference, dataset_name=Path(args.current).stem).detect(current)

    model_report = None
    if args.predictions:
        predictions = pd.read_csv(args.predictions)
        model_report = evaluate_predictions_frame(predictions, model_name="monitored_model")

    trust = DatasetTrustScorer().score(data_quality, drift, model_report)
    payload = {
        "trust_score": trust,
        "data_quality": data_quality.to_dict(),
        "drift": drift.to_dict(),
        "model_health": model_report.to_dict() if model_report else None,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Monitoring report written to {output}")


if __name__ == "__main__":
    main()
