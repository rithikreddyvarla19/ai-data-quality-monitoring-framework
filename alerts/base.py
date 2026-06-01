from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from monitoring.types import CheckSeverity, CheckStatus, MonitoringReport


@dataclass(slots=True)
class Alert:
    title: str
    severity: CheckSeverity
    message: str
    run_id: str
    dataset_name: str
    alert_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "title": self.title,
            "severity": str(self.severity),
            "message": self.message,
            "run_id": self.run_id,
            "dataset_name": self.dataset_name,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class AlertRule:
    name: str
    min_severity: CheckSeverity = CheckSeverity.HIGH
    trigger_on_statuses: tuple[CheckStatus, ...] = (CheckStatus.FAIL,)

    def evaluate(self, report: MonitoringReport) -> list[Alert]:
        severity_order = {
            CheckSeverity.INFO: 0,
            CheckSeverity.LOW: 1,
            CheckSeverity.MEDIUM: 2,
            CheckSeverity.HIGH: 3,
            CheckSeverity.CRITICAL: 4,
        }
        min_level = severity_order[self.min_severity]
        alerts: list[Alert] = []
        for check in report.checks:
            if check.status not in self.trigger_on_statuses:
                continue
            if severity_order[check.severity] < min_level:
                continue
            alerts.append(
                Alert(
                    title=f"{self.name}: {check.name}",
                    severity=check.severity,
                    message=f"{check.name} returned {check.status} with score {check.score:.3f}",
                    run_id=report.run_id,
                    dataset_name=report.dataset_name,
                    metadata={"check": check.to_dict()},
                )
            )
        return alerts

