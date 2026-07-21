"""Emergency mutation controls, resume telemetry, and fleet metric exports."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import ModuleType
from uuid import UUID, uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from tutor.api.app import create_app
from tutor.api.v2 import install_v2_routes
from tutor.api.v2_controls import MutationGate, MutationGateSnapshot
from tutor.api.v2_features import V2FeatureFlags
from tutor.api.v2_persistence import V2PersistenceService
from tutor.db import models as m
from tutor.db.persistence import PersistenceService
from tutor.db.session import get_engine
from tutor.seed.load_seed import load_graph

from tests.v2_helpers import (
    approved_power_rule_catalog,
    approved_power_rule_stress_bank,
    power_rule_only_graph,
)

_SECRET = b"operational-controls-test-secret-32-bytes"


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


class MutableMutationGate:
    def __init__(self, paused: bool = False) -> None:
        self.paused = paused
        self.revision = "operator-1"

    def set(self, paused: bool, revision: str) -> None:
        self.paused = paused
        self.revision = revision

    def snapshot(self) -> MutationGateSnapshot:
        return MutationGateSnapshot(
            paused=self.paused,
            revision=self.revision,
            source="test_control_plane",
            observed_at=datetime.now(timezone.utc),
        )


class BrokenMutationGate:
    def snapshot(self) -> MutationGateSnapshot:
        raise RuntimeError("private-provider-error-do-not-expose")


class FixedMutationGate:
    def __init__(self, snapshot: MutationGateSnapshot) -> None:
        self._snapshot = snapshot

    def snapshot(self) -> MutationGateSnapshot:
        return self._snapshot


def _app(
    *,
    persistence: V2PersistenceService | None = None,
    paused: bool = False,
    metrics_sink: RecordingMetricsSink | None = None,
    mutation_gate: MutationGate | None = None,
    mutation_gate_max_age: timedelta | None = None,
) -> FastAPI:
    app = FastAPI()
    install_v2_routes(
        app,
        power_rule_only_graph(),
        persistence=persistence,
        available_targets=("kc.der.power_rule",),
        item_bank=approved_power_rule_stress_bank(),
        pedagogy_catalog=approved_power_rule_catalog(),
        resume_token_secret=_SECRET,
        feature_flags=V2FeatureFlags(pause_v2_mutations=paused),
        metrics_sink=metrics_sink,
        mutation_gate=mutation_gate,
        mutation_gate_max_age=mutation_gate_max_age,
    )
    return app


def _create_payload(request_id: UUID, *, context: str | None = None) -> dict:
    return {
        "request_id": str(request_id),
        "goal_id": "goal.der.power_rule",
        "context": context,
    }


def _hint(view: dict, request_id: UUID) -> dict:
    return {
        "type": "request_hint",
        "request_id": str(request_id),
        "expected_revision": view["revision"],
        "pending_key": view["pending"]["key"],
    }


def _reset(view: dict, request_id: UUID) -> dict:
    return {
        "request_id": str(request_id),
        "expected_revision": view["revision"],
        "pending_key": view["pending"]["key"] if view["pending"] else None,
    }


def _set_resume_cookie(client: TestClient, raw_token: str) -> None:
    client.cookies.set(
        "tutor_resume_v2",
        raw_token,
        path="/api/v2",
        domain="testserver.local",
    )


def _durable_ledger_snapshot(engine) -> dict:
    """Exclude rolling token expiry, which reads and replays may refresh."""
    with Session(engine) as session:
        checkpoints = session.scalars(
            select(m.SessionCheckpointRow).order_by(m.SessionCheckpointRow.session_id)
        ).all()
        return {
            "checkpoints": [
                (
                    row.session_id,
                    row.revision,
                    deepcopy(row.checkpoint),
                )
                for row in checkpoints
            ],
            "receipts": len(session.scalars(select(m.SessionMutationReceiptRow)).all()),
            "transcript": len(session.scalars(select(m.TranscriptEntryRow)).all()),
            "evidence": len(session.scalars(select(m.EvidenceEventRow)).all()),
            "exposures": len(session.scalars(select(m.ItemExposureRow)).all()),
            "widgets": len(session.scalars(select(m.WidgetAttemptRow)).all()),
        }


def test_mutation_pause_preserves_reads_recovery_and_exact_replays_without_delta():
    legacy = PersistenceService(engine=get_engine("sqlite+pysqlite:///:memory:"))
    persistence = V2PersistenceService(legacy.engine)
    create_id = uuid4()
    action_id = uuid4()

    admitted = TestClient(_app(persistence=persistence))
    create_payload = _create_payload(create_id)
    created = admitted.post("/api/v2/sessions", json=create_payload)
    original_cookie = admitted.cookies.get("tutor_resume_v2")
    action_payload = _hint(created.json(), action_id)
    advanced = admitted.post(
        f"/api/v2/sessions/{created.json()['session_id']}/actions",
        json=action_payload,
    )
    assert advanced.status_code == 200
    before = _durable_ledger_snapshot(legacy.engine)

    paused = TestClient(_app(persistence=persistence, paused=True))
    _set_resume_cookie(paused, original_cookie)
    current = paused.get("/api/v2/sessions/current")
    assert current.status_code == 200
    assert current.json() == advanced.json()

    replayed_action = paused.post(
        f"/api/v2/sessions/{created.json()['session_id']}/actions",
        json=action_payload,
    )
    assert replayed_action.status_code == 200
    assert replayed_action.json() == advanced.json()

    blocked_action = paused.post(
        f"/api/v2/sessions/{created.json()['session_id']}/actions",
        json=_hint(advanced.json(), uuid4()),
    )
    assert blocked_action.status_code == 503
    assert blocked_action.json() == {
        "code": "v2_mutations_paused",
        "message": (
            "Session changes are temporarily paused for a safety check; "
            "retry this same request_id shortly."
        ),
        "session": advanced.json(),
        "retryable": True,
    }

    blocked_reset = paused.post(
        "/api/v2/sessions/current/reset",
        json=_reset(advanced.json(), uuid4()),
    )
    assert blocked_reset.status_code == 503
    assert blocked_reset.json()["code"] == "v2_mutations_paused"
    assert blocked_reset.json()["session"] == advanced.json()

    replay_client = TestClient(_app(persistence=persistence, paused=True))
    replayed_create = replay_client.post("/api/v2/sessions", json=create_payload)
    assert replayed_create.status_code == 200
    assert replayed_create.json() == created.json()

    new_client = TestClient(_app(persistence=persistence, paused=True))
    blocked_create = new_client.post(
        "/api/v2/sessions",
        json=_create_payload(uuid4()),
    )
    assert blocked_create.status_code == 503
    assert blocked_create.json()["retryable"] is True

    recovery_client = TestClient(_app(persistence=persistence, paused=True))
    recovered = recovery_client.post(
        "/api/v2/sessions/recover",
        json={
            "schema_version": 1,
            "operation": "create",
            "request_id": str(create_id),
        },
    )
    assert recovered.status_code == 200
    assert recovered.json()["session_id"] == created.json()["session_id"]

    assert paused.get("/api/v2/sessions/current").json() == advanced.json()
    assert _durable_ledger_snapshot(legacy.engine) == before


def test_mutation_pause_replays_committed_reset_but_blocks_a_new_reset():
    legacy = PersistenceService(engine=get_engine("sqlite+pysqlite:///:memory:"))
    persistence = V2PersistenceService(legacy.engine)
    admitted = TestClient(_app(persistence=persistence))
    created = admitted.post(
        "/api/v2/sessions",
        json=_create_payload(uuid4()),
    )
    old_cookie = admitted.cookies.get("tutor_resume_v2")
    reset_payload = _reset(created.json(), uuid4())
    committed = admitted.post("/api/v2/sessions/current/reset", json=reset_payload)
    assert committed.status_code == 200
    replacement_cookie = admitted.cookies.get("tutor_resume_v2")

    paused = TestClient(_app(persistence=persistence, paused=True))
    _set_resume_cookie(paused, old_cookie)
    replayed = paused.post("/api/v2/sessions/current/reset", json=reset_payload)
    assert replayed.status_code == 200
    assert replayed.json() == committed.json()
    assert paused.cookies.get("tutor_resume_v2") == replacement_cookie

    blocked = paused.post(
        "/api/v2/sessions/current/reset",
        json=_reset(committed.json()["session"], uuid4()),
    )
    assert blocked.status_code == 503
    assert blocked.json()["code"] == "v2_mutations_paused"
    assert paused.get("/api/v2/sessions/current").json() == committed.json()["session"]


def test_live_mutation_gate_toggles_without_restart_and_preserves_all_replays():
    gate = MutableMutationGate()
    client = TestClient(_app(mutation_gate=gate))
    create_payload = _create_payload(uuid4())
    created = client.post("/api/v2/sessions", json=create_payload)
    assert created.status_code == 200
    original_cookie = client.cookies.get("tutor_resume_v2")
    action_payload = _hint(created.json(), uuid4())
    advanced = client.post(
        f"/api/v2/sessions/{created.json()['session_id']}/actions",
        json=action_payload,
    )
    assert advanced.status_code == 200

    gate.set(True, "operator-2-paused")
    assert client.get("/api/v2/goals").json()["rollout"]["status"] == "paused"
    readiness = client.app.state.v2_readiness_provider()
    assert readiness["mutations_paused"] is True
    assert readiness["accepting_mutations"] is False
    assert readiness["mutation_gate"] == {
        "revision": "operator-2-paused",
        "source": "test_control_plane",
        "observed_at": readiness["mutation_gate"]["observed_at"],
    }

    replayed_create = client.post("/api/v2/sessions", json=create_payload)
    assert replayed_create.status_code == 200
    assert replayed_create.json() == created.json()
    replayed_action = client.post(
        f"/api/v2/sessions/{created.json()['session_id']}/actions",
        json=action_payload,
    )
    assert replayed_action.status_code == 200
    assert replayed_action.json() == advanced.json()
    blocked_action = client.post(
        f"/api/v2/sessions/{created.json()['session_id']}/actions",
        json=_hint(advanced.json(), uuid4()),
    )
    assert blocked_action.status_code == 503
    assert blocked_action.json()["code"] == "v2_mutations_paused"

    gate.set(False, "operator-3-open")
    reset_payload = _reset(advanced.json(), uuid4())
    reset = client.post("/api/v2/sessions/current/reset", json=reset_payload)
    assert reset.status_code == 200
    replacement_cookie = client.cookies.get("tutor_resume_v2")

    gate.set(True, "operator-4-paused")
    _set_resume_cookie(client, original_cookie)
    replayed_reset = client.post(
        "/api/v2/sessions/current/reset",
        json=reset_payload,
    )
    assert replayed_reset.status_code == 200
    assert replayed_reset.json() == reset.json()
    assert client.cookies.get("tutor_resume_v2") == replacement_cookie

    new_reset_payload = _reset(reset.json()["session"], uuid4())
    blocked_reset = client.post(
        "/api/v2/sessions/current/reset",
        json=new_reset_payload,
    )
    assert blocked_reset.status_code == 503
    assert blocked_reset.json()["code"] == "v2_mutations_paused"

    gate.set(False, "operator-5-open")
    committed_after_toggle = client.post(
        "/api/v2/sessions/current/reset",
        json=new_reset_payload,
    )
    assert committed_after_toggle.status_code == 200
    assert client.get("/api/v2/goals").json()["rollout"]["status"] == "available"


def test_broken_and_stale_mutation_gate_observations_fail_closed_safely():
    broken_app = create_app(load_graph(), v2_mutation_gate=BrokenMutationGate())
    broken = TestClient(broken_app)
    health = broken.get("/healthz")
    assert health.status_code == 200
    readiness = health.json()["v2_readiness"]
    assert readiness["mutations_paused"] is True
    assert readiness["accepting_mutations"] is False
    assert readiness["mutation_gate"]["revision"] == "fail-closed-v1"
    assert readiness["mutation_gate"]["source"] == "fail_closed"
    assert set(readiness["mutation_gate"]) == {
        "revision",
        "source",
        "observed_at",
    }
    assert "private-provider-error-do-not-expose" not in repr(readiness)
    assert broken.get("/api/v2/goals").json()["rollout"]["status"] == "paused"
    blocked = broken.post(
        "/api/v2/sessions",
        json=_create_payload(uuid4()),
    )
    assert blocked.status_code == 503
    assert blocked.json()["code"] == "v2_mutations_paused"

    stale_gate = FixedMutationGate(
        MutationGateSnapshot(
            paused=False,
            revision="stale-open-state",
            source="test_control_plane",
            observed_at=datetime.now(timezone.utc) - timedelta(minutes=5),
        )
    )
    stale = TestClient(
        _app(
            mutation_gate=stale_gate,
            mutation_gate_max_age=timedelta(seconds=30),
        )
    )
    assert stale.get("/api/v2/goals").json()["rollout"]["status"] == "paused"
    stale_readiness = stale.app.state.v2_readiness_provider()
    assert stale_readiness["mutations_paused"] is True
    assert stale_readiness["mutation_gate"]["revision"] == "fail-closed-v1"
    assert stale_readiness["mutation_gate"]["source"] == "fail_closed"


def test_static_pause_flag_is_a_ceiling_over_an_open_dynamic_gate():
    gate = MutableMutationGate(paused=False)
    client = TestClient(_app(paused=True, mutation_gate=gate))

    assert client.get("/api/v2/goals").json()["rollout"]["status"] == "paused"
    readiness = client.app.state.v2_readiness_provider()
    assert readiness["mutations_paused"] is True
    assert readiness["accepting_mutations"] is False
    assert readiness["mutation_gate"]["revision"] == "operator-1"


def test_create_app_loads_live_mutation_gate_factory_from_environment(monkeypatch):
    gate = MutableMutationGate(paused=True)
    module = ModuleType("test_runtime_mutation_gate_plugin")
    module.build_gate = lambda: gate
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv(
        "TUTOR_V2_MUTATION_GATE_FACTORY",
        f"{module.__name__}:build_gate",
    )

    client = TestClient(create_app(load_graph()))
    paused_health = client.get("/healthz").json()["v2_readiness"]
    assert paused_health["mutations_paused"] is True
    assert paused_health["mutation_gate"]["source"] == "test_control_plane"
    assert client.get("/api/v2/goals").json()["rollout"]["status"] == "paused"

    gate.set(False, "operator-2-open")
    open_health = client.get("/healthz").json()["v2_readiness"]
    assert open_health["mutations_paused"] is False
    assert open_health["content_ready"] is False
    assert open_health["accepting_mutations"] is False
    assert open_health["mutation_gate"]["revision"] == "operator-2-open"
    assert (
        client.get("/api/v2/goals").json()["rollout"]["status"]
        == "content_unavailable"
    )


def test_create_app_plugin_load_failure_pauses_without_leaking_configuration(
    monkeypatch,
    caplog,
):
    private_spec = "private_vendor_control_plane:build_secret_gate"
    monkeypatch.setenv("TUTOR_V2_MUTATION_GATE_FACTORY", private_spec)

    client = TestClient(create_app(load_graph()))
    readiness = client.get("/healthz").json()["v2_readiness"]
    assert readiness["mutations_paused"] is True
    assert readiness["accepting_mutations"] is False
    assert readiness["mutation_gate"]["revision"] == "plugin-load-failed-v1"
    assert readiness["mutation_gate"]["source"] == "fail_closed"
    assert private_spec not in caplog.text


def test_resume_rate_uses_cookie_attempts_and_separates_failure_classes():
    client = TestClient(_app())
    no_cookie = client.get("/api/v2/sessions/current")
    assert no_cookie.status_code == 401
    metrics = client.app.state.v2_store.metrics_snapshot()
    assert metrics["resume_outcomes"]["no_cookie"] == 1
    assert metrics["resume_outcomes"]["cookie_attempts"] == 0
    assert metrics["resume_outcomes"]["eligible_attempts"] == 0
    assert metrics["rollout_gates"]["resume_success_rate"] is None

    _set_resume_cookie(client, "not-a-valid-resume-token")
    malformed = client.get("/api/v2/sessions/current")
    assert malformed.status_code == 401
    metrics = client.app.state.v2_store.metrics_snapshot()
    assert metrics["resume_outcomes"]["invalid"] == 1
    assert metrics["resume_outcomes"]["cookie_attempts"] == 1
    assert metrics["resume_outcomes"]["eligible_attempts"] == 0
    assert metrics["rollout_gates"]["resume_success_rate"] is None

    # A structurally valid but unissued cookie is invalid, not a missing-cookie
    # request, and therefore remains in the cookie-bearing denominator.
    _set_resume_cookie(client, "A" * 43)
    unknown = client.get("/api/v2/sessions/current")
    assert unknown.status_code == 401
    metrics = client.app.state.v2_store.metrics_snapshot()
    assert metrics["resume_outcomes"]["invalid"] == 2
    assert metrics["resume_outcomes"]["cookie_attempts"] == 2
    assert metrics["resume_outcomes"]["eligible_attempts"] == 0

    client.cookies.clear()
    created = client.post(
        "/api/v2/sessions",
        json=_create_payload(uuid4()),
    )
    assert created.status_code == 200
    resumed = client.get("/api/v2/sessions/current")
    assert resumed.status_code == 200
    metrics = client.app.state.v2_store.metrics_snapshot()
    assert metrics["resume_outcomes"]["successes"] == 1
    assert metrics["resume_outcomes"]["cookie_attempts"] == 3
    assert metrics["resume_outcomes"]["eligible_attempts"] == 1
    assert metrics["rollout_gates"]["resume_success_rate"] == 1

    mismatch = client.get("/api/v2/sessions/not-the-current-session")
    assert mismatch.status_code == 404
    metrics = client.app.state.v2_store.metrics_snapshot()
    assert metrics["resume_outcomes"]["session_mismatch"] == 1
    assert metrics["resume_outcomes"]["cookie_attempts"] == 4
    assert metrics["resume_outcomes"]["eligible_attempts"] == 1
    assert metrics["rollout_gates"]["resume_success_rate"] == 1


def test_expired_cookie_has_its_own_resume_outcome():
    legacy = PersistenceService(engine=get_engine("sqlite+pysqlite:///:memory:"))
    persistence = V2PersistenceService(legacy.engine)
    client = TestClient(_app(persistence=persistence))
    created = client.post(
        "/api/v2/sessions",
        json=_create_payload(uuid4()),
    )
    assert created.status_code == 200
    with Session(legacy.engine) as session:
        token = session.scalars(select(m.ResumeTokenRow)).one()
        token.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()
    client.app.state.v2_store._tokens.clear()

    expired = client.get("/api/v2/sessions/current")
    assert expired.status_code == 401
    metrics = client.app.state.v2_store.metrics_snapshot()
    assert metrics["resume_outcomes"]["expired"] == 1
    assert metrics["resume_outcomes"]["invalid"] == 0
    assert metrics["resume_outcomes"]["cookie_attempts"] == 1
    assert metrics["resume_outcomes"]["eligible_attempts"] == 0
    assert metrics["rollout_gates"]["resume_success_rate"] is None


def test_active_token_restore_failure_is_in_resume_reliability_denominator(monkeypatch):
    legacy = PersistenceService(engine=get_engine("sqlite+pysqlite:///:memory:"))
    persistence = V2PersistenceService(legacy.engine)
    client = TestClient(_app(persistence=persistence))
    created = client.post(
        "/api/v2/sessions",
        json=_create_payload(uuid4()),
    )
    assert created.status_code == 200

    def fail_resolve(token_hash):
        raise RuntimeError("database read failed")

    monkeypatch.setattr(persistence, "resolve_resume", fail_resolve)
    failed = client.get("/api/v2/sessions/current")
    assert failed.status_code == 503
    metrics = client.app.state.v2_store.metrics_snapshot()
    assert metrics["resume_outcomes"]["cookie_attempts"] == 1
    assert metrics["resume_outcomes"]["eligible_attempts"] == 1
    assert metrics["resume_outcomes"]["eligible_failures"] == 1
    assert metrics["resume_outcomes"]["restore_failures"] == 1
    assert metrics["rollout_gates"]["resume_success_rate"] == 0


def test_active_token_refresh_failure_is_an_eligible_resume_failure(monkeypatch):
    legacy = PersistenceService(engine=get_engine("sqlite+pysqlite:///:memory:"))
    persistence = V2PersistenceService(legacy.engine)
    client = TestClient(_app(persistence=persistence))
    created = client.post(
        "/api/v2/sessions",
        json=_create_payload(uuid4()),
    )
    assert created.status_code == 200

    def fail_touch(token_hash):
        raise RuntimeError("database write failed")

    monkeypatch.setattr(persistence, "touch_resume", fail_touch)
    failed = client.get("/api/v2/sessions/current")
    assert failed.status_code == 503
    metrics = client.app.state.v2_store.metrics_snapshot()
    assert metrics["resume_outcomes"]["cookie_attempts"] == 1
    assert metrics["resume_outcomes"]["eligible_attempts"] == 1
    assert metrics["resume_outcomes"]["eligible_failures"] == 1
    assert metrics["resume_outcomes"]["refresh_failures"] == 1
    assert metrics["resume_outcomes"]["successes"] == 0
    assert metrics["rollout_gates"]["resume_success_rate"] == 0


def test_metrics_sink_receives_only_stable_release_and_item_dimensions():
    sink = RecordingMetricsSink()
    client = TestClient(_app(metrics_sink=sink))
    private_context = "student private coaching context"
    private_answer = "student private answer"
    created = client.post(
        "/api/v2/sessions",
        json=_create_payload(uuid4(), context=private_context),
    )
    response = client.post(
        f"/api/v2/sessions/{created.json()['session_id']}/actions",
        json={
            "type": "answer",
            "request_id": str(uuid4()),
            "expected_revision": created.json()["revision"],
            "pending_key": created.json()["pending"]["key"],
            "answer": private_answer,
        },
    )
    assert response.status_code == 200
    after_action = client.app.state.v2_store.metrics_snapshot()
    assert after_action["resume_outcomes"]["eligible_attempts"] == 0
    assert after_action["rollout_gates"]["resume_success_rate"] is None

    resumed = client.get("/api/v2/sessions/current")
    assert resumed.status_code == 200
    after_read = client.app.state.v2_store.metrics_snapshot()
    assert after_read["resume_outcomes"]["eligible_attempts"] == 1
    assert after_read["resume_outcomes"]["successes"] == 1
    assert after_read["rollout_gates"]["resume_success_rate"] == 1

    action_events = [event for event in sink.events if event[0] == "actions_committed"]
    assert len(action_events) == 1
    dimensions = action_events[0][2]
    assert dimensions["graph_version"] == str(power_rule_only_graph().graph_version)
    assert (
        dimensions["item_bank_version"]
        == approved_power_rule_stress_bank().bank_version
    )
    assert (
        dimensions["pedagogy_catalog_version"]
        == approved_power_rule_catalog().catalog_version
    )
    assert dimensions["learner_parameter_version"] == "bkt-v2"
    assert dimensions["capability_manifest_version"].startswith(
        "web-widget-capabilities-v2"
    )
    assert len(dimensions["release_digest"]) == 64
    assert dimensions["item_id"].startswith("item.power")
    assert "policy_diagnosis_version" in dimensions
    exported = repr(sink.events)
    assert private_context not in exported
    assert private_answer not in exported
    assert created.json()["session_id"] not in exported

    snapshot = client.app.state.v2_store.metrics_snapshot()
    assert snapshot["counters"]["actions_committed"] == 1
    assert snapshot["actions_by_item_id"][dimensions["item_id"]] == 1


def test_health_reports_operator_active_version_readiness():
    app = create_app(load_graph())
    health = TestClient(app).get("/healthz")
    assert health.status_code == 200
    readiness = health.json()["v2_readiness"]
    versions = readiness["active_versions"]
    assert versions["graph"] == load_graph().graph_version
    assert isinstance(versions["item_bank"], str)
    assert isinstance(versions["pedagogy_catalog"], str)
    assert versions["policies"]["diagnosis"].startswith("diagnosis-v2")
    assert versions["learner_parameters"] == "bkt-v2"
    assert versions["capability_manifest"].startswith("web-widget-capabilities-v2")
    assert readiness["content_ready"] is False
    assert readiness["accepting_mutations"] is False
