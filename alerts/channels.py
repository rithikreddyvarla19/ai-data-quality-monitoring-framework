from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import requests

from alerts.base import Alert


class AlertChannel(Protocol):
    def send(self, alert: Alert) -> None:
        ...


class ConsoleAlertChannel:
    def send(self, alert: Alert) -> None:
        print(f"[{alert.severity}] {alert.title}: {alert.message}")


@dataclass(slots=True)
class WebhookAlertChannel:
    url: str
    timeout_seconds: float = 5.0

    def send(self, alert: Alert) -> None:
        if not self.url:
            return
        response = requests.post(self.url, json=alert.to_dict(), timeout=self.timeout_seconds)
        response.raise_for_status()

