"""Concrete fleet adapters without requiring a live Redis or OTLP service."""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import tutor.api.app as app_module
from tutor.api.v2_admission import (
    NetworkIdentityResolver,
    RedisTokenBucketRequestAdmissionGate,
)
from tutor.api.v2_controls import RedisMutationGate
from tutor.api.v2_metrics import OpenTelemetryMetricsSink, V2MetricDimensions
from tutor.api.v2_quarantine import RedisReleaseQuarantineProvider
from tutor.api.v2_store import V2SessionStore
from tutor.seed.load_seed import load_graph

_NOW = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)


class FakeRedis:
    """Tiny shared fake implementing the commands used by the adapters."""

    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}
        self.buckets: dict[str, float] = {}
        self.get_calls = 0
        self.eval_calls = 0
        self.raise_get: Exception | None = None
        self.raise_eval: Exception | None = None

    def get(self, key: str):
        self.get_calls += 1
        if self.raise_get is not None:
            raise self.raise_get
        return self.values.get(key)

    def eval(self, script: str, key_count: int, key: str, capacity: int, period_ms: int):
        del script, period_ms
        assert key_count == 1
        self.eval_calls += 1
        if self.raise_eval is not None:
            raise self.raise_eval
        tokens = self.buckets.setdefault(key, float(capacity))
        if tokens >= 1:
            self.buckets[key] = tokens - 1
            return [1, 0]
        return [0, 1000]


def test_cached_redis_controls_are_o1_and_fail_closed_on_refresh_failure(caplog):
    redis = FakeRedis()
    mutation = RedisMutationGate(redis, clock=lambda: _NOW)
    quarantine = RedisReleaseQuarantineProvider(redis, clock=lambda: _NOW)

    # Snapshot reads are cache-only and do not touch Redis.
    assert mutation.snapshot().paused is True
    assert quarantine.snapshot().available is False
    assert redis.get_calls == 0

    digest = "a" * 64
    redis.values["tutor:v2:controls:mutations"] = json.dumps(
        {"schema_version": 1, "paused": False, "revision": "operator-7"}
    ).encode()
    redis.values["tutor:v2:controls:quarantine"] = json.dumps(
        {
            "schema_version": 1,
            "revision": "release-4",
            "quarantined_digests": [digest],
        }
    ).encode()
    mutation.refresh_once()
    quarantine.refresh_once()

    assert mutation.snapshot().paused is False
    assert mutation.snapshot().source == "redis_control_plane"
    assert quarantine.snapshot().available is True
    assert quarantine.snapshot().is_quarantined(digest)
    calls_after_refresh = redis.get_calls
    mutation.snapshot()
    quarantine.snapshot()
    assert redis.get_calls == calls_after_refresh

    redis.raise_get = RuntimeError("redis-password=do-not-log")
    mutation.refresh_once()
    quarantine.refresh_once()
    assert mutation.snapshot().paused is True
    assert mutation.snapshot().source == "fail_closed"
    assert quarantine.snapshot().available is False
    assert "redis-password=do-not-log" not in caplog.text


def test_two_control_consumers_observe_one_shared_redis_value():
    redis = FakeRedis()
    key = "tutor:v2:controls:mutations"
    redis.values[key] = json.dumps(
        {"schema_version": 1, "paused": False, "revision": "open-1"}
    ).encode()
    first = RedisMutationGate(redis, clock=lambda: _NOW)
    second = RedisMutationGate(redis, clock=lambda: _NOW)
    first.refresh_once()
    second.refresh_once()
    assert (first.snapshot().paused, second.snapshot().paused) == (False, False)

    redis.values[key] = json.dumps(
        {"schema_version": 1, "paused": True, "revision": "paused-2"}
    ).encode()
    first.refresh_once()
    second.refresh_once()
    assert (first.snapshot().paused, second.snapshot().paused) == (True, True)


def test_network_identity_trusts_forwarding_only_from_configured_proxies():
    resolver = NetworkIdentityResolver(
        b"network-identity-secret-at-least-32-bytes",
        trusted_proxy_cidrs=("10.0.0.0/8",),
    )
    direct = resolver.identity("203.0.113.9")
    spoofed = resolver.identity(
        "203.0.113.9",
        forwarded_for=("198.51.100.4",),
    )
    proxied = resolver.identity(
        "10.1.2.3",
        forwarded_for=("198.51.100.4, 10.8.0.1",),
    )

    assert spoofed == direct
    assert proxied != resolver.identity("10.1.2.3")
    assert "198.51.100.4" not in proxied
    assert len(proxied) == 64


def test_two_admission_gates_share_token_buckets_without_storing_addresses():
    redis = FakeRedis()
    resolver = NetworkIdentityResolver(
        b"network-identity-secret-at-least-32-bytes",
        trusted_proxy_cidrs=("10.0.0.0/8",),
    )
    first = RedisTokenBucketRequestAdmissionGate(redis, resolver)
    second = RedisTokenBucketRequestAdmissionGate(redis, resolver)

    decisions = [
        (first if index % 2 == 0 else second).admit(
            "create",
            peer_host="10.2.0.8",
            forwarded_for=("198.51.100.77",),
        )
        for index in range(11)
    ]

    assert all(decision.allowed for decision in decisions[:10])
    assert decisions[-1].allowed is False
    assert decisions[-1].available is True
    assert decisions[-1].retry_after_seconds == 1
    assert len(redis.buckets) == 1
    key = next(iter(redis.buckets))
    assert "198.51.100.77" not in key
    assert "10.2.0.8" not in key

    redis.raise_eval = TimeoutError("private redis endpoint")
    unavailable = first.admit("action", peer_host="203.0.113.2")
    assert unavailable.allowed is False
    assert unavailable.available is False
    assert unavailable.retry_after_seconds is None


def test_lifecycle_operations_share_one_ten_request_bucket():
    redis = FakeRedis()
    gate = RedisTokenBucketRequestAdmissionGate(
        redis,
        NetworkIdentityResolver(
            b"network-identity-secret-at-least-32-bytes",
            trusted_proxy_cidrs=("10.0.0.0/8",),
        ),
    )

    operations = ("create", "recover", "reset") * 4
    decisions = [
        gate.admit(operation, peer_host="203.0.113.4")
        for operation in operations
    ]

    assert all(decision.allowed for decision in decisions[:10])
    assert all(not decision.allowed for decision in decisions[10:])
    assert len(redis.buckets) == 1
    assert next(iter(redis.buckets)).endswith(":lifecycle")


def test_nonblocking_metrics_queue_bounds_a_wedged_exporter():
    release = threading.Event()
    entered = threading.Event()

    def blocked_emitter(
        name: str,
        amount: int,
        dimensions: Mapping[str, str],
    ) -> None:
        del name, amount, dimensions
        entered.set()
        release.wait(timeout=2)

    sink = OpenTelemetryMetricsSink(blocked_emitter, max_queue_size=64)
    dimensions = {
        "graph_version": "1",
        "item_bank_version": "bank-v1",
        "pedagogy_catalog_version": "packs-v1",
        "learner_parameter_version": "bkt-v2",
        "capability_manifest_version": "widgets-v1",
    }
    sink.increment("first_metric", dimensions=dimensions)
    assert entered.wait(timeout=1)
    for _ in range(1000):
        sink.increment("queued_metric", dimensions=dimensions)

    assert sink.dropped_count > 0
    release.set()
    sink.close(timeout_seconds=1)


def test_metric_dimensions_are_derived_from_the_pinned_checkpoint():
    active = V2MetricDimensions(
        graph_version="2",
        item_bank_version="active-bank",
        pedagogy_catalog_version="active-packs",
        policy_versions=(("lesson", "lesson-v2"),),
        learner_parameter_version="bkt-v2",
        capability_manifest_version="widgets-v2",
        release_digest="b" * 64,
    )
    retained = V2MetricDimensions.from_checkpoint(
        {
            "graph_version": 1,
            "item_bank_version": "retained-bank",
            "pedagogy_catalog_version": "retained-packs",
            "policy_versions": {"lesson": "lesson-v1"},
            "learner_params": {"params_version": 1},
            "widget_capability_manifest": {"version": "widgets-v1"},
        },
        fallback=active,
    )

    assert retained.as_labels() == {
        "graph_version": "1",
        "item_bank_version": "retained-bank",
        "pedagogy_catalog_version": "retained-packs",
        "learner_parameter_version": "bkt-v1",
        "capability_manifest_version": "widgets-v1",
        "policy_lesson_version": "lesson-v1",
    }


def test_retained_session_metrics_use_its_pins_not_the_active_release():
    events: list[tuple[str, dict[str, str]]] = []

    class Sink:
        def increment(
            self,
            name: str,
            amount: int = 1,
            *,
            dimensions: Mapping[str, str],
        ) -> None:
            assert amount == 1
            events.append((name, dict(dimensions)))

    active = V2MetricDimensions(
        graph_version="2",
        item_bank_version="active-bank",
        pedagogy_catalog_version="active-packs",
        policy_versions=(("lesson", "lesson-v2"),),
        learner_parameter_version="bkt-v2",
        capability_manifest_version="widgets-v2",
    )
    retained = SimpleNamespace(
        _graph=SimpleNamespace(graph_version=1),
        _bank=SimpleNamespace(bank_version="retained-bank"),
        pedagogy_catalog_version="retained-packs",
        _policy_versions=lambda: {"lesson": "lesson-v1"},
        _pinned_widget_capabilities={"version": "widgets-v1"},
        learner=SimpleNamespace(params=SimpleNamespace(params_version=1)),
    )
    store = V2SessionStore(
        graph_nodes={},
        metrics_sink=Sink(),
        metric_dimensions=active,
        metric_dimensions_resolver=lambda orchestrator: V2MetricDimensions(
            **{
                **V2MetricDimensions.from_orchestrator(
                    orchestrator,
                    fallback=active,
                ).__dict__,
                "release_digest": "a" * 64,
            }
        ),
    )
    retained_dimensions = store._resolve_metric_dimensions(retained)

    store.record_metric(
        "actions_committed",
        metric_dimensions=retained_dimensions,
    )

    assert events == [
        (
            "actions_committed",
            {
                "graph_version": "1",
                "item_bank_version": "retained-bank",
                "pedagogy_catalog_version": "retained-packs",
                "learner_parameter_version": "bkt-v1",
                "capability_manifest_version": "widgets-v1",
                "policy_lesson_version": "lesson-v1",
                "release_digest": "a" * 64,
            },
        )
    ]


def test_pilot_production_builds_real_adapters_around_one_redis_client(monkeypatch):
    redis = FakeRedis()
    installed: dict = {}

    class Engine:
        def dispose(self) -> None:
            pass

    class Persistence:
        def __init__(self, *args, **kwargs) -> None:
            self.engine = Engine()

    class Sink:
        def increment(self, name, amount=1, *, dimensions):
            pass

        def close(self) -> None:
            pass

    sink = Sink()

    def install(app, graph, **kwargs):
        del app, graph
        installed.update(kwargs)

    for name in (
        "TUTOR_ENABLE_API_SESSION_V2",
        "TUTOR_ENABLE_CONTENT_ALLOCATION_V2",
        "TUTOR_ENABLE_DIAGNOSIS_V2",
        "TUTOR_ENABLE_LESSON_FLOW_V2",
        "TUTOR_ENABLE_RICH_WIDGETS_V2",
    ):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("TUTOR_PILOT_PRODUCTION", "1")
    monkeypatch.setenv("TUTOR_PAUSE_V2_MUTATIONS", "0")
    monkeypatch.setenv("TUTOR_V2_STUDENT_ROLLOUT_PERCENT", "5")
    monkeypatch.setenv("TUTOR_REDIS_URL", "rediss://redis.invalid/0")
    monkeypatch.setenv(
        "TUTOR_NETWORK_HMAC_SECRET",
        "independent-network-secret-at-least-32-bytes",
    )
    monkeypatch.setenv("TUTOR_TRUSTED_PROXY_CIDRS", "10.0.0.0/8")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://otel.invalid")
    for name in (
        "TUTOR_V2_METRICS_SINK_FACTORY",
        "TUTOR_V2_MUTATION_GATE_FACTORY",
        "TUTOR_V2_RELEASE_QUARANTINE_FACTORY",
        "TUTOR_V2_REQUEST_ADMISSION_FACTORY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(app_module, "PersistenceService", Persistence)
    monkeypatch.setattr(
        app_module,
        "schema_migration_status",
        lambda engine: {"reachable": True, "current": True, "head": "head"},
    )
    monkeypatch.setattr(app_module, "create_redis_client", lambda settings: redis)
    monkeypatch.setattr(app_module, "V2PersistenceService", lambda engine: object())
    monkeypatch.setattr(
        app_module.OpenTelemetryMetricsSink,
        "from_environment",
        staticmethod(lambda: sink),
    )
    monkeypatch.setattr(app_module, "install_v2_routes", install)

    app_module.create_app(
        load_graph(),
        database_url="postgresql+psycopg://pilot.invalid/tutor",
    )

    assert installed["metrics_sink"] is sink
    assert isinstance(installed["mutation_gate"], RedisMutationGate)
    assert isinstance(
        installed["release_quarantine"],
        RedisReleaseQuarantineProvider,
    )
    assert isinstance(
        installed["request_admission_gate"],
        RedisTokenBucketRequestAdmissionGate,
    )
    assert installed["mutation_gate"]._redis is redis
    assert installed["release_quarantine"]._redis is redis
    assert installed["request_admission_gate"]._redis is redis


def test_pilot_request_admission_configuration_errors_are_sanitized(
    monkeypatch,
    caplog,
):
    secret = "private-invalid-network-key"

    class Engine:
        def dispose(self) -> None:
            pass

    class Persistence:
        def __init__(self, *args, **kwargs) -> None:
            self.engine = Engine()

    class Sink:
        def increment(self, name, amount=1, *, dimensions):
            pass

    for name in (
        "TUTOR_ENABLE_API_SESSION_V2",
        "TUTOR_ENABLE_CONTENT_ALLOCATION_V2",
        "TUTOR_ENABLE_DIAGNOSIS_V2",
        "TUTOR_ENABLE_LESSON_FLOW_V2",
        "TUTOR_ENABLE_RICH_WIDGETS_V2",
    ):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("TUTOR_PILOT_PRODUCTION", "1")
    monkeypatch.setenv("TUTOR_PAUSE_V2_MUTATIONS", "0")
    monkeypatch.setenv("TUTOR_V2_STUDENT_ROLLOUT_PERCENT", "5")
    monkeypatch.setenv("TUTOR_REDIS_URL", "rediss://redis.invalid/0")
    monkeypatch.setenv("TUTOR_NETWORK_HMAC_SECRET", secret)
    monkeypatch.setenv("TUTOR_TRUSTED_PROXY_CIDRS", "10.0.0.0/8")
    monkeypatch.setattr(app_module, "PersistenceService", Persistence)
    monkeypatch.setattr(
        app_module,
        "schema_migration_status",
        lambda engine: {"reachable": True, "current": True, "head": "head"},
    )
    monkeypatch.setattr(app_module, "create_redis_client", lambda settings: FakeRedis())

    with pytest.raises(
        RuntimeError,
        match="requires Redis request admission",
    ) as caught:
        app_module.create_app(
            load_graph(),
            database_url="postgresql+psycopg://pilot.invalid/tutor",
            v2_metrics_sink=Sink(),
        )

    assert secret not in str(caught.value)
    assert secret not in caplog.text
