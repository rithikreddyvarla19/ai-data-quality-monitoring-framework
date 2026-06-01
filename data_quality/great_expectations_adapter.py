from __future__ import annotations

from typing import Any

import pandas as pd

from monitoring.types import CheckResult, CheckSeverity, CheckStatus


def run_great_expectations_checks(
    df: pd.DataFrame,
    expectations: list[dict[str, Any]] | None = None,
) -> CheckResult:
    """Run a lightweight Great Expectations adapter when GE is installed.

    The project keeps a local implementation for core checks, while this adapter lets
    teams plug in existing GE expectation suites in production.
    """
    if not expectations:
        return CheckResult(
            name="data_quality.great_expectations",
            status=CheckStatus.SKIP,
            score=1.0,
            details={"reason": "no expectations supplied"},
        )

    try:
        import great_expectations as gx  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional install
        return CheckResult(
            name="data_quality.great_expectations",
            status=CheckStatus.WARN,
            score=0.8,
            severity=CheckSeverity.LOW,
            details={"reason": "great_expectations unavailable", "error": str(exc)},
        )

    try:  # pragma: no cover - GE API changes across versions
        context = gx.get_context(mode="ephemeral")
        datasource = context.sources.add_pandas(name="runtime_pandas")
        asset = datasource.add_dataframe_asset(name="monitoring_frame")
        batch_request = asset.build_batch_request(dataframe=df)
        validator = context.get_validator(batch_request=batch_request)

        failures: list[dict[str, Any]] = []
        for expectation in expectations:
            expectation_type = expectation["expectation_type"]
            kwargs = expectation.get("kwargs", {})
            result = getattr(validator, expectation_type)(**kwargs)
            if not result.success:
                failures.append({"expectation_type": expectation_type, "kwargs": kwargs})

        score = 1.0 - (len(failures) / max(len(expectations), 1))
        status = CheckStatus.PASS if not failures else CheckStatus.FAIL
        return CheckResult(
            name="data_quality.great_expectations",
            status=status,
            score=score,
            severity=CheckSeverity.HIGH if failures else CheckSeverity.INFO,
            details={"failures": failures, "expectation_count": len(expectations)},
        )
    except Exception as exc:
        return CheckResult(
            name="data_quality.great_expectations",
            status=CheckStatus.WARN,
            score=0.8,
            severity=CheckSeverity.LOW,
            details={"reason": "GE execution failed", "error": str(exc)},
        )

