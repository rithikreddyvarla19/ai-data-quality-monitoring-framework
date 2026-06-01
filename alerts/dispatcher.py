from __future__ import annotations

from alerts.base import Alert, AlertRule
from alerts.channels import AlertChannel, ConsoleAlertChannel
from monitoring.types import MonitoringReport


class AlertManager:
    def __init__(
        self,
        rules: list[AlertRule] | None = None,
        channels: list[AlertChannel] | None = None,
    ) -> None:
        self.rules = rules or [AlertRule(name="enterprise-monitoring")]
        self.channels = channels or [ConsoleAlertChannel()]

    def evaluate(self, report: MonitoringReport) -> list[Alert]:
        alerts: list[Alert] = []
        for rule in self.rules:
            alerts.extend(rule.evaluate(report))
        report.alerts = [alert.to_dict() for alert in alerts]
        return alerts

    def dispatch(self, alerts: list[Alert]) -> None:
        for alert in alerts:
            for channel in self.channels:
                channel.send(alert)

    def evaluate_and_dispatch(self, report: MonitoringReport) -> list[Alert]:
        alerts = self.evaluate(report)
        self.dispatch(alerts)
        return alerts

