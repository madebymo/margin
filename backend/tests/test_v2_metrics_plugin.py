"""Runtime construction and readiness reporting for the v2 fleet metrics sink."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from types import ModuleType

import pytest
from fastapi.testclient import TestClient

from tutor.api.app import create_app
from tutor.seed.load_seed import load_graph


class RecordingMetricsSink:
    def __init__(self) -> None:
        self.events: list[tuple[str, int, dict[str, str]]] = []

    def increment(
        self,
        name: str,
        amount: int = 1,
        *,
        dimensions: Mapping[str, str],
    ) -> None:
        self.events.append((name, amount, dict(dimensions)))


def _module(monkeypatch, name: str) -> ModuleType:
    module = ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _configure_pilot(monkeypatch, *, api_session_v2: bool = True) -> None:
    monkeypatch.setenv("TUTOR_PILOT_PRODUCTION", "1")
    monkeypatch.setenv(
        "TUTOR_ENABLE_API_SESSION_V2",
        "1" if api_session_v2 else "0",
    )
    monkeypatch.setenv("TUTOR_ENABLE_CONTENT_ALLOCATION_V2", "1")
    monkeypatch.setenv("TUTOR_ENABLE_DIAGNOSIS_V2", "1")
    monkeypatch.setenv("TUTOR_ENABLE_LESSON_FLOW_V2", "1")
    monkeypatch.setenv("TUTOR_ENABLE_RICH_WIDGETS_V2", "1")
    monkeypatch.setenv("TUTOR_PAUSE_V2_MUTATIONS", "0")
    monkeypatch.setenv("TUTOR_V2_STUDENT_ROLLOUT_PERCENT", "5")


def test_environment_factory_is_constructed_once_and_exports_metrics(monkeypatch):
    module = _module(monkeypatch, "test_runtime_metrics_plugin")
    sink = RecordingMetricsSink()
    factory_calls = 0

    def build_sink() -> RecordingMetricsSink:
        nonlocal factory_calls
        factory_calls += 1
        return sink

    module.build_sink = build_sink
    monkeypatch.setenv(
        "TUTOR_V2_METRICS_SINK_FACTORY",
        f"{module.__name__}:build_sink",
    )

    client = TestClient(create_app(load_graph()))
    assert factory_calls == 1
    readiness = client.get("/healthz").json()["v2_readiness"]
    assert readiness["fleet_metrics_configured"] is True

    assert client.get("/api/v2/goals").status_code == 200
    assert factory_calls == 1
    assert any(name == "catalog_requests" for name, _, _ in sink.events)


def test_explicit_sink_wins_over_an_invalid_environment_factory(monkeypatch):
    sink = RecordingMetricsSink()
    monkeypatch.setenv(
        "TUTOR_V2_METRICS_SINK_FACTORY",
        "private.invalid_metrics_adapter:build_sink",
    )

    client = TestClient(create_app(load_graph(), v2_metrics_sink=sink))
    assert client.get("/healthz").json()["v2_readiness"][
        "fleet_metrics_configured"
    ] is True
    assert client.get("/api/v2/goals").status_code == 200
    assert any(name == "catalog_requests" for name, _, _ in sink.events)


@pytest.mark.parametrize("failure_kind", ["malformed", "factory", "incompatible"])
def test_configured_factory_failures_stop_startup_without_leaking_details(
    monkeypatch,
    caplog,
    failure_kind,
):
    secret = "metrics-api-key=do-not-expose"
    private_spec = "private.metrics_plugin:build_sink"
    if failure_kind == "malformed":
        private_spec = "private-metrics-plugin:build_sink"
    else:
        module = _module(monkeypatch, "private.metrics_plugin")
        if failure_kind == "factory":
            def build_sink():
                raise RuntimeError(secret)

            module.build_sink = build_sink
        else:
            module.build_sink = object
    monkeypatch.setenv("TUTOR_V2_METRICS_SINK_FACTORY", private_spec)

    with pytest.raises(RuntimeError) as caught:
        create_app(load_graph())

    assert str(caught.value) == (
        "configured v2 fleet metrics sink could not be initialized"
    )
    assert caught.value.__cause__ is None
    assert private_spec not in str(caught.value)
    assert private_spec not in caplog.text
    assert secret not in str(caught.value)
    assert secret not in caplog.text


def test_pilot_production_requires_a_metrics_sink(monkeypatch):
    _configure_pilot(monkeypatch)
    monkeypatch.delenv("TUTOR_V2_METRICS_SINK_FACTORY", raising=False)

    class PersistenceStub:
        def __init__(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setattr("tutor.api.app.PersistenceService", PersistenceStub)

    with pytest.raises(
        RuntimeError,
        match="TUTOR_PILOT_PRODUCTION requires a v2 fleet metrics sink",
    ):
        create_app(
            load_graph(),
            database_url="postgresql://pilot.invalid/tutor",
        )


def test_disabled_v2_api_does_not_import_or_require_metrics_plugin(monkeypatch):
    monkeypatch.setenv("TUTOR_ENABLE_API_SESSION_V2", "0")
    monkeypatch.setenv(
        "TUTOR_V2_METRICS_SINK_FACTORY",
        "private.metrics_plugin:build_sink",
    )

    def unexpected_import(name):
        raise AssertionError("disabled v2 API must not import runtime plugins")

    monkeypatch.setattr("tutor.api.runtime_plugins.import_module", unexpected_import)

    client = TestClient(create_app(load_graph()))
    health = client.get("/healthz").json()
    assert health["v2_features"]["api_session_v2"] is False
    assert health["v2_readiness"]["fleet_metrics_configured"] is False
    assert client.get("/api/v2/goals").status_code == 404
