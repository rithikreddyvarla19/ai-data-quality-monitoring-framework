from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(slots=True)
class MonitoringJob:
    name: str
    cron: str
    callable_path: str
    enabled: bool = True


class MonitoringScheduler:
    def __init__(self) -> None:
        self.jobs: list[MonitoringJob] = []
        self._scheduler = None

    def add_job(self, job: MonitoringJob) -> None:
        self.jobs.append(job)

    def start(self, task_registry: dict[str, Callable[[], None]]) -> None:
        try:
            from apscheduler.schedulers.background import BackgroundScheduler  # type: ignore
            from apscheduler.triggers.cron import CronTrigger  # type: ignore
        except Exception as exc:
            raise RuntimeError("APScheduler is required for scheduled monitoring jobs.") from exc

        scheduler = BackgroundScheduler()
        for job in self.jobs:
            if not job.enabled:
                continue
            if job.callable_path not in task_registry:
                raise ValueError(f"No registered callable for scheduled job: {job.callable_path}")
            scheduler.add_job(
                task_registry[job.callable_path],
                CronTrigger.from_crontab(job.cron),
                id=job.name,
                replace_existing=True,
            )
        scheduler.start()
        self._scheduler = scheduler

    def shutdown(self) -> None:
        if self._scheduler is not None:
            self._scheduler.shutdown()

