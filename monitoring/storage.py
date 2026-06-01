from __future__ import annotations

from typing import Any

from monitoring.types import MonitoringReport


class ReportStore:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self._engine = None

    @property
    def engine(self):
        if self._engine is None:
            try:
                from sqlalchemy import create_engine  # type: ignore
            except Exception as exc:
                raise RuntimeError("SQLAlchemy is required for report persistence.") from exc
            self._engine = create_engine(self.database_url, pool_pre_ping=True)
        return self._engine

    def save_report(self, report: MonitoringReport, report_type: str) -> None:
        try:
            from sqlalchemy import text  # type: ignore
        except Exception as exc:
            raise RuntimeError("SQLAlchemy is required for report persistence.") from exc

        payload = report.to_dict()
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    insert into monitoring_reports
                      (run_id, dataset_name, report_type, status, score, payload, created_at)
                    values
                      (:run_id, :dataset_name, :report_type, :status, :score, cast(:payload as jsonb), :created_at)
                    on conflict (run_id, report_type) do update set
                      status = excluded.status,
                      score = excluded.score,
                      payload = excluded.payload,
                      created_at = excluded.created_at
                    """
                ),
                {
                    "run_id": payload["run_id"],
                    "dataset_name": payload["dataset_name"],
                    "report_type": report_type,
                    "status": payload["status"],
                    "score": payload["score"],
                    "payload": _json_dump(payload),
                    "created_at": payload["created_at"],
                },
            )

    def latest_reports(self, limit: int = 25) -> list[dict[str, Any]]:
        try:
            from sqlalchemy import text  # type: ignore
        except Exception as exc:
            raise RuntimeError("SQLAlchemy is required for report persistence.") from exc

        with self.engine.begin() as connection:
            rows = connection.execute(
                text(
                    """
                    select run_id, dataset_name, report_type, status, score, payload, created_at
                    from monitoring_reports
                    order by created_at desc
                    limit :limit
                    """
                ),
                {"limit": limit},
            ).mappings()
            return [dict(row) for row in rows]


def _json_dump(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, default=str)

