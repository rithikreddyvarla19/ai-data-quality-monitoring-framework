from __future__ import annotations

from fastapi import FastAPI, File, UploadFile

from api.schemas import (
    DriftRequest,
    GPTMonitoringRequest,
    ModelHealthRequest,
    MonitoringRunRequest,
    ValidationRequest,
)
from api.services import (
    csv_bytes_to_frame,
    run_data_quality,
    run_drift,
    run_full_monitoring,
    run_gpt_monitoring,
    run_model_health,
)

CSV_FILE = File(...)

app = FastAPI(
    title="AI Data Quality Monitoring Framework",
    description="Enterprise MLOps API for data quality, drift, model health, and AI system monitoring.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "ai-data-quality-monitoring-framework"}


@app.post("/validate")
def validate(payload: ValidationRequest) -> dict:
    report = run_data_quality(
        dataset_name=payload.dataset_name,
        records=payload.records,
        schema_fields=[field.model_dump(exclude_none=True) for field in payload.schema_fields],
        target_column=payload.target_column,
    )
    return report.to_dict()


@app.post("/validate-csv")
async def validate_csv(file: UploadFile = CSV_FILE) -> dict:
    df = csv_bytes_to_frame(await file.read())
    report = run_data_quality(dataset_name=file.filename or "uploaded_dataset", records=df.to_dict("records"))
    return report.to_dict()


@app.post("/drift")
def drift(payload: DriftRequest) -> dict:
    report = run_drift(
        dataset_name=payload.dataset_name,
        reference_records=payload.reference_records,
        current_records=payload.current_records,
        use_evidently=payload.use_evidently,
    )
    return report.to_dict()


@app.post("/model-health")
def model_health(payload: ModelHealthRequest) -> dict:
    report = run_model_health(
        model_name=payload.model_name,
        y_true=payload.y_true,
        y_pred=payload.y_pred,
        y_probability=payload.y_probability,
    )
    return report.to_dict()


@app.post("/gpt-monitoring")
def gpt_monitoring(payload: GPTMonitoringRequest) -> dict:
    return run_gpt_monitoring(system_name=payload.system_name, records=payload.records).to_dict()


@app.post("/monitoring-run")
def monitoring_run(payload: MonitoringRunRequest) -> dict:
    return run_full_monitoring(payload.model_dump())
