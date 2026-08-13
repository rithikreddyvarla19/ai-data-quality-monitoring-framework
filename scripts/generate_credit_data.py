from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(rows: int, seed: int, drifted: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    income = rng.lognormal(10.8 + (0.12 if drifted else 0), 0.55, rows).round(2)
    utilization = np.clip(rng.beta(2.2 + (0.8 if drifted else 0), 4.8, rows), 0, 1)
    age = rng.integers(21, 76, rows)
    delinquencies = rng.poisson(0.35 + (0.25 if drifted else 0), rows)
    score = 690 - 115 * utilization - 18 * delinquencies + rng.normal(0, 28, rows)
    return pd.DataFrame(
        {
            "application_id": [f"APP-{seed}-{i:06d}" for i in range(rows)],
            "applicant_age": age,
            "annual_income": income,
            "credit_utilization": utilization.round(4),
            "delinquencies_2y": delinquencies,
            "credit_score": np.clip(score, 300, 850).round().astype(int),
            "employment_type": rng.choice(
                ["salaried", "self_employed", "contract"], rows, p=[0.64, 0.25, 0.11]
            ),
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drifted", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generate(args.rows, args.seed, args.drifted).to_csv(args.output, index=False)
