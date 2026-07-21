"""Focused production-runtime safety gates that do not require external services."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from sqlalchemy import inspect

from tutor.api.app import create_app
from tutor.api.http_safety import (
    CONTENT_SECURITY_POLICY,
    HttpSecurityHeadersMiddleware,
    RequestBodyLimitMiddleware,
    trusted_hosts_from_environment,
)
from tutor.api.v2 import install_v2_routes
from tutor.api.v2_persistence import V2PersistenceService
from tutor.api.v2_quarantine import (
    ReleaseQuarantineSnapshot,
    release_runtime_digest,
)
from tutor.api.v2_versions import V2VersionRegistry
from tutor.db.migrate_session_v2 import migrate, schema_migration_status
from tutor.db.persistence import PersistenceService
from tutor.db.session import PostgresEngineSettings, get_engine
from tutor.orchestrator.session_v2 import SessionOrchestratorV2
from tutor.schemas.assessment import ItemBankDocument
from tutor.schemas.kc import GraphDocument

from tests.v2_helpers import (
    approved_power_rule_episode_bank,
    approved_power_rule_catalog,
    power_rule_only_graph,
)


class MutableQuarantine:
    def __init__(self) -> None:
        self.digests: frozenset[str] = frozenset()

    def snapshot(self) -> ReleaseQuarantineSnapshot:
        return ReleaseQuarantineSnapshot(
            quarantined_digests=self.digests,
            revision="test-quarantine-v1",
            source="test_control_plane",
            observed_at=datetime.now(timezone.utc),
        )


def test_postgres_engine_uses_bounded_pool_and_server_timeouts(monkeypatch):
    captured: dict = {}

    def capture(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr("tutor.db.session.create_engine", capture)
    settings = PostgresEngineSettings(
        connect_timeout_seconds=4,
        pool_size=7,
        max_overflow=2,
        pool_timeout_seconds=3,
        pool_recycle_seconds=900,
        statement_timeout_ms=2500,
        lock_timeout_ms=750,
        idle_transaction_timeout_ms=4000,
        application_name="tutor-test",
    )

    result = get_engine(
        "postgresql+psycopg://user:private-password@example.invalid/tutor",
        postgres_settings=settings,
    )

    assert result is not None
    assert captured["kwargs"] == {
        "pool_pre_ping": True,
        "pool_size": 7,
        "max_overflow": 2,
        "pool_timeout": 3,
        "pool_recycle": 900,
        "connect_args": {
            "connect_timeout": 4,
            "application_name": "tutor-test",
            "options": (
                "-c statement_timeout=2500 -c lock_timeout=750 "
                "-c idle_in_transaction_session_timeout=4000"
            ),
        },
    }


def test_postgres_settings_reject_invalid_or_unbounded_environment_values():
    with pytest.raises(ValueError, match="TUTOR_DB_POOL_SIZE"):
        PostgresEngineSettings.from_environment({"TUTOR_DB_POOL_SIZE": "0"})
    with pytest.raises(ValueError, match="TUTOR_DB_APPLICATION_NAME"):
        PostgresEngineSettings.from_environment(
            {"TUTOR_DB_APPLICATION_NAME": "private value with spaces"}
        )


def test_migration_head_creates_operational_indexes_and_is_observable():
    persistence = PersistenceService(engine=get_engine("sqlite+pysqlite:///:memory:"))
    assert schema_migration_status(persistence.engine)["current"] is False

    migrate(persistence.engine)
    status = schema_migration_status(persistence.engine)
    assert status["reachable"] is True
    assert status["current"] is True
    inspector = inspect(persistence.engine)
    names = {
        index["name"]
        for table in (
            "evidence_events",
            "resume_tokens",
            "session_checkpoints",
            "session_mutation_receipts",
        )
        for index in inspector.get_indexes(table)
    }
    assert {
        "ix_evidence_learner_time",
        "ix_evidence_episode",
        "ix_resume_tokens_expiry_revoked",
        "ix_resume_tokens_session",
        "ix_session_checkpoint_learner_started",
        "ix_session_checkpoint_updated",
        "ix_session_receipt_request",
    } <= names


def test_liveness_is_cheap_and_default_readiness_fails_closed():
    client = TestClient(create_app())

    assert client.get("/livez").json() == {"status": "ok"}
    readiness = client.get("/readyz")
    assert readiness.status_code == 503
    assert readiness.json()["status"] == "not_ready"
    assert readiness.json()["checks"]["durable_persistence"] is False
    assert set(readiness.json()) == {"status", "checks", "migration_head"}


def test_content_length_body_limit_returns_typed_v2_error_before_validation():
    client = TestClient(create_app())
    response = client.post(
        "/api/v2/sessions",
        content=b"x" * (64 * 1024 + 1),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 413
    assert response.json() == {
        "code": "request_too_large",
        "message": "request body exceeds the 64 KiB limit",
    }


def test_chunked_body_limit_never_calls_downstream_application():
    called = False

    async def downstream(scope, receive, send):
        nonlocal called
        called = True

    middleware = RequestBodyLimitMiddleware(downstream, max_body_bytes=1024)
    chunks = iter(
        (
            {"type": "http.request", "body": b"a" * 700, "more_body": True},
            {"type": "http.request", "body": b"b" * 400, "more_body": False},
        )
    )
    sent: list[dict] = []

    async def receive():
        return next(chunks)

    async def send(message):
        sent.append(message)

    asyncio.run(
        middleware(
            {
                "type": "http",
                "method": "POST",
                "path": "/api/v2/sessions",
                "headers": [],
            },
            receive,
            send,
        )
    )

    assert called is False
    assert sent[0]["status"] == 413


def test_browser_security_policy_covers_pages_api_and_rejected_hosts():
    client = TestClient(create_app())

    page = client.get("/")
    api = client.get("/api/v2/goals")
    rejected = client.get("/livez", headers={"host": "attacker.example"})

    for response in (page, api, rejected):
        assert response.headers["content-security-policy"] == CONTENT_SECURITY_POLICY
        assert response.headers["cross-origin-opener-policy"] == "same-origin"
        assert response.headers["cross-origin-resource-policy"] == "same-origin"
        assert response.headers["permissions-policy"] == (
            "camera=(), microphone=(), geolocation=(), payment=(), usb=()"
        )
        assert response.headers["referrer-policy"] == "no-referrer"
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["x-frame-options"] == "DENY"
        assert "strict-transport-security" not in response.headers
    assert page.headers["cache-control"] == "no-store"
    assert api.headers["cache-control"] == "no-store"
    assert rejected.status_code == 400


def test_secure_transport_policy_adds_hsts_without_overwriting_app_headers():
    app = FastAPI()

    @app.get("/api/example")
    def example():
        return JSONResponse(
            {"ok": True},
            headers={"Referrer-Policy": "same-origin"},
        )

    app.add_middleware(HttpSecurityHeadersMiddleware, secure_transport=True)
    response = TestClient(app).get("/api/example")

    assert response.headers["strict-transport-security"] == (
        "max-age=31536000; includeSubDomains"
    )
    assert response.headers["referrer-policy"] == "same-origin"
    assert response.headers["cache-control"] == "no-store"


def test_production_trusted_hosts_are_explicit_bounded_and_never_global():
    assert trusted_hosts_from_environment(
        pilot_production=False,
        environ={},
    ) == ("testserver", "localhost", "*.localhost", "127.0.0.1")
    assert trusted_hosts_from_environment(
        pilot_production=True,
        environ={"TUTOR_TRUSTED_HOSTS": "Tutor.Example.edu,api.example.edu"},
    ) == ("tutor.example.edu", "api.example.edu")

    with pytest.raises(RuntimeError, match="explicit TUTOR_TRUSTED_HOSTS"):
        trusted_hosts_from_environment(pilot_production=True, environ={})
    with pytest.raises(RuntimeError, match="forbids wildcard"):
        trusted_hosts_from_environment(
            pilot_production=True,
            environ={"TUTOR_TRUSTED_HOSTS": "*"},
        )
    for value in (
        "https://tutor.example.edu",
        "tutor.example.edu:443",
        "tutor.example.edu/path",
        "tutor.example.edu,,api.example.edu",
    ):
        with pytest.raises(ValueError):
            trusted_hosts_from_environment(
                pilot_production=False,
                environ={"TUTOR_TRUSTED_HOSTS": value},
            )


def test_quarantine_blocks_reads_and_committed_action_replays_without_content():
    graph = power_rule_only_graph()
    bank = approved_power_rule_episode_bank()
    catalog = approved_power_rule_catalog()
    registry = V2VersionRegistry()
    release = registry.register(graph, bank, catalog)
    quarantine = MutableQuarantine()
    app = FastAPI()
    install_v2_routes(
        app,
        graph,
        available_targets=("kc.der.power_rule",),
        item_bank=bank,
        pedagogy_catalog=catalog,
        version_registry=registry,
        release_quarantine=quarantine,
        resume_token_secret=b"runtime-hardening-test-secret-32b",
    )
    client = TestClient(app)
    created = client.post(
        "/api/v2/sessions",
        json={
            "request_id": str(uuid4()),
            "goal_id": "goal.der.power_rule",
        },
    )
    assert created.status_code == 200
    view = created.json()
    action = {
        "type": "request_hint",
        "request_id": str(uuid4()),
        "expected_revision": view["revision"],
        "pending_key": view["pending"]["key"],
    }
    committed = client.post(
        f"/api/v2/sessions/{view['session_id']}/actions",
        json=action,
    )
    assert committed.status_code == 200

    quarantine.digests = frozenset(
        {
            release_runtime_digest(
                release,
                SessionOrchestratorV2._policy_versions(),
            )
        }
    )
    for response in (
        client.get("/api/v2/sessions/current"),
        client.post(
            f"/api/v2/sessions/{view['session_id']}/actions",
            json=action,
        ),
    ):
        assert response.status_code == 410
        payload = response.json()
        assert payload["code"] == "release_quarantined"
        assert "session" not in payload
        assert "transcript" not in repr(payload)
        assert "pending" not in repr(payload)


def test_quarantine_reset_moves_an_old_episode_to_a_distinct_safe_release():
    graph_v1 = power_rule_only_graph()
    bank_v1 = approved_power_rule_episode_bank()
    catalog_v1 = approved_power_rule_catalog()
    graph_payload = graph_v1.model_dump(mode="json")
    graph_payload["graph_version"] = 2
    graph_v2 = GraphDocument.model_validate(graph_payload)
    bank_payload = bank_v1.model_dump(mode="json")
    bank_payload["graph_version"] = 2
    bank_payload["bank_version"] = "power-bank-v2-distinct-families"
    for item in bank_payload["items"]:
        item["item_id"] += ".release2"
        item["family_id"] += ".release2"
    bank_v2 = ItemBankDocument.model_validate(bank_payload)
    catalog_v2 = approved_power_rule_catalog(
        graph_version=2,
        catalog_version="power-pedagogy-v2-distinct-families",
    )

    registry_v1 = V2VersionRegistry()
    release_v1 = registry_v1.register(graph_v1, bank_v1, catalog_v1)
    persistence = V2PersistenceService(
        PersistenceService(
            engine=get_engine("sqlite+pysqlite:///:memory:")
        ).engine
    )
    secret = b"quarantine-reset-test-secret-32b"
    old_app = FastAPI()
    install_v2_routes(
        old_app,
        graph_v1,
        persistence=persistence,
        available_targets=("kc.der.power_rule",),
        item_bank=bank_v1,
        pedagogy_catalog=catalog_v1,
        version_registry=registry_v1,
        resume_token_secret=secret,
    )
    old_client = TestClient(old_app)
    created = old_client.post(
        "/api/v2/sessions",
        json={"request_id": str(uuid4()), "goal_id": "goal.der.power_rule"},
    )
    assert created.status_code == 200
    old_token = old_client.cookies.get("tutor_resume_v2")
    assert old_token is not None

    registry_v2 = V2VersionRegistry([(graph_v1, bank_v1, catalog_v1)])
    registry_v2.register(graph_v2, bank_v2, catalog_v2)
    quarantine = MutableQuarantine()
    quarantine.digests = frozenset(
        {
            release_runtime_digest(
                release_v1,
                SessionOrchestratorV2._policy_versions(),
            )
        }
    )
    new_app = FastAPI()
    install_v2_routes(
        new_app,
        graph_v2,
        persistence=persistence,
        available_targets=("kc.der.power_rule",),
        item_bank=bank_v2,
        pedagogy_catalog=catalog_v2,
        version_registry=registry_v2,
        release_quarantine=quarantine,
        resume_token_secret=secret,
    )
    new_client = TestClient(new_app)
    new_client.cookies.set(
        "tutor_resume_v2",
        old_token,
        domain="testserver.local",
        path="/api/v2",
    )

    blocked = new_client.get("/api/v2/sessions/current")
    assert blocked.status_code == 410
    recovery = blocked.json()["quarantine_recovery"]
    assert set(recovery) == {"revision", "reset_key"}

    reset_payload = {
        "request_id": str(uuid4()),
        "expected_revision": recovery["revision"],
        "pending_key": recovery["reset_key"],
    }
    reset = new_client.post(
        "/api/v2/sessions/current/reset",
        json=reset_payload,
    )
    assert reset.status_code == 200
    replacement = reset.json()["session"]
    assert replacement["session_id"] != created.json()["session_id"]
    assert new_client.get("/api/v2/sessions/current").json() == replacement

    new_client.cookies.set(
        "tutor_resume_v2",
        old_token,
        domain="testserver.local",
        path="/api/v2",
    )
    replayed = new_client.post(
        "/api/v2/sessions/current/reset",
        json=reset_payload,
    )
    assert replayed.status_code == 200, replayed.text
    assert replayed.json() == reset.json()
