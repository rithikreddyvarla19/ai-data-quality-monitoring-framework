from __future__ import annotations

from typing import Any

import pandas as pd


def to_pandas(frame: Any) -> pd.DataFrame:
    """Accept pandas or PySpark DataFrames and return a pandas DataFrame."""
    if isinstance(frame, pd.DataFrame):
        return frame.copy()
    if hasattr(frame, "toPandas"):
        return frame.toPandas()
    raise TypeError("Expected a pandas DataFrame or PySpark DataFrame-like object.")


def infer_column_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    datetime = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    boolean = df.select_dtypes(include=["bool"]).columns.tolist()
    categorical = [
        col
        for col in df.columns
        if col not in numeric and col not in datetime and col not in boolean
    ]
    return {
        "numeric": numeric,
        "categorical": categorical,
        "datetime": datetime,
        "boolean": boolean,
    }


def quality_score_from_penalty(penalty: float) -> float:
    return round(max(0.0, min(1.0, 1.0 - penalty)), 4)

