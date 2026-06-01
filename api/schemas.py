from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SchemaFieldPayload(BaseModel):
    name: str
    dtype: str
    nullable: bool = True
    required: bool = True
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: list[Any] | None = None
    regex: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ValidationRequest(BaseModel):
    dataset_name: str = "dataset"
    records: list[dict[str, Any]]
    schema_fields: list[SchemaFieldPayload] = Field(default_factory=list)
    target_column: str | None = None


class DriftRequest(BaseModel):
    dataset_name: str = "dataset"
    reference_records: list[dict[str, Any]]
    current_records: list[dict[str, Any]]
    use_evidently: bool = False


class ModelHealthRequest(BaseModel):
    model_name: str = "model"
    y_true: list[Any]
    y_pred: list[Any]
    y_probability: list[float] | list[list[float]] | None = None


class GPTMonitoringRequest(BaseModel):
    system_name: str = "gpt_system"
    records: list[dict[str, Any]]


class MonitoringRunRequest(BaseModel):
    dataset_name: str = "dataset"
    reference_records: list[dict[str, Any]]
    current_records: list[dict[str, Any]]
    schema_fields: list[SchemaFieldPayload] = Field(default_factory=list)
    target_column: str | None = None
    prediction_records: list[dict[str, Any]] = Field(default_factory=list)
    actual_column: str = "actual"
    prediction_column: str = "prediction"
    probability_column: str | None = "probability"

