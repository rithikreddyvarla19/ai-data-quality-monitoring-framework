from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from statistics import mean
from typing import Any
from uuid import uuid4


class CheckStatus(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


class CheckSeverity(StrEnum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(slots=True)
class CheckResult:
    name: str
    status: CheckStatus
    score: float
    severity: CheckSeverity = CheckSeverity.INFO
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.score = max(0.0, min(float(self.score), 1.0))

    @property
    def passed(self) -> bool:
        return self.status == CheckStatus.PASS

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = str(self.status)
        payload["severity"] = str(self.severity)
        return payload


@dataclass(slots=True)
class MonitoringReport:
    dataset_name: str
    run_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    checks: list[CheckResult] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    alerts: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> float:
        scored = [check.score for check in self.checks if check.status != CheckStatus.SKIP]
        return round(mean(scored), 4) if scored else 1.0

    @property
    def status(self) -> CheckStatus:
        statuses = {check.status for check in self.checks}
        if CheckStatus.FAIL in statuses:
            return CheckStatus.FAIL
        if CheckStatus.WARN in statuses:
            return CheckStatus.WARN
        if statuses == {CheckStatus.SKIP}:
            return CheckStatus.SKIP
        return CheckStatus.PASS

    def add_check(self, check: CheckResult) -> None:
        self.checks.append(check)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "run_id": self.run_id,
            "created_at": self.created_at.isoformat(),
            "status": str(self.status),
            "score": self.score,
            "checks": [check.to_dict() for check in self.checks],
            "metrics": self.metrics,
            "alerts": self.alerts,
            "metadata": self.metadata,
        }

