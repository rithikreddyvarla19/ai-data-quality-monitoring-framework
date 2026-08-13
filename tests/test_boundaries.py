from unittest.mock import Mock

import pytest


def persist_report(store, report):
    try:
        store.save(report)
    except Exception as exc:
        raise RuntimeError("report persistence failed") from exc


def deliver_report(client, report):
    try:
        response = client.post("https://monitor.invalid/reports", json=report, timeout=5)
    except Exception as exc:
        raise RuntimeError("report delivery failed") from exc
    if response.status_code >= 400:
        raise RuntimeError(f"report delivery returned {response.status_code}")


def test_database_failure_has_stable_context():
    store = Mock()
    store.save.side_effect = OSError("database offline")
    with pytest.raises(RuntimeError, match="persistence failed"):
        persist_report(store, {})


def test_network_failure_has_stable_context():
    client = Mock()
    client.post.side_effect = TimeoutError("timed out")
    with pytest.raises(RuntimeError, match="delivery failed"):
        deliver_report(client, {})


def test_storage_http_failure_is_not_silenced():
    client = Mock()
    client.post.return_value.status_code = 503
    with pytest.raises(RuntimeError, match="503"):
        deliver_report(client, {})
