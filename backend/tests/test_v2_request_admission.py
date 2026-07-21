"""HTTP wiring for fleet-shared v2 request admission."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from tutor.api.v2 import install_v2_routes
from tutor.api.v2_admission import AdmissionDecision, AdmissionOperation
from tutor.api.v2_quarantine import ReleaseQuarantineSnapshot

from tests.v2_helpers import (
    approved_power_rule_catalog,
    approved_power_rule_stress_bank,
    power_rule_only_graph,
)


class MutableAdmissionGate:
    def __init__(self) -> None:
        self.decisions: dict[str, AdmissionDecision] = {}
        self.calls: list[tuple[str, str | None, tuple[str, ...]]] = []

    def admit(
        self,
        operation: AdmissionOperation,
        *,
        peer_host: str | None,
        forwarded_for: tuple[str, ...] = (),
    ) -> AdmissionDecision:
        self.calls.append((operation, peer_host, forwarded_for))
        return self.decisions.get(operation, AdmissionDecision(allowed=True))


class MutableQuarantine:
    def __init__(self) -> None:
        self.digests: frozenset[str] = frozenset()

    def snapshot(self) -> ReleaseQuarantineSnapshot:
        return ReleaseQuarantineSnapshot(
            quarantined_digests=self.digests,
            revision="admission-order-test-v1",
            source="test_control_plane",
            observed_at=datetime.now(timezone.utc),
        )


def _client(
    gate: MutableAdmissionGate,
    *,
    quarantine: MutableQuarantine | None = None,
) -> TestClient:
    app = FastAPI()
    install_v2_routes(
        app,
        power_rule_only_graph(),
        available_targets=("kc.der.power_rule",),
        item_bank=approved_power_rule_stress_bank(),
        pedagogy_catalog=approved_power_rule_catalog(),
        resume_token_secret=b"request-admission-test-secret-32-bytes",
        request_admission_gate=gate,
        release_quarantine=quarantine,
    )
    return TestClient(app)


def _create(client: TestClient, request_id=None):
    return client.post(
        "/api/v2/sessions",
        json={
            "request_id": str(request_id or uuid4()),
            "goal_id": "goal.der.power_rule",
        },
    )


def test_rate_limit_is_typed_and_includes_retry_after_without_creating_state():
    gate = MutableAdmissionGate()
    gate.decisions["create"] = AdmissionDecision(
        allowed=False,
        retry_after_seconds=17,
    )
    client = _client(gate)

    response = _create(client)

    assert response.status_code == 429
    assert response.headers["retry-after"] == "17"
    assert response.json() == {
        "code": "rate_limited",
        "message": "Too many requests; retry after the indicated delay.",
        "retryable": True,
    }
    assert len(client.app.state.v2_store) == 0


def test_admission_outage_fails_mutations_and_api_reads_closed():
    gate = MutableAdmissionGate()
    gate.decisions["create"] = AdmissionDecision(allowed=False, available=False)
    gate.decisions["read"] = AdmissionDecision(allowed=False, available=False)
    client = _client(gate)

    blocked = _create(client)
    catalog = client.get("/api/v2/goals")

    assert blocked.status_code == 503
    assert blocked.json() == {
        "code": "safety_state_unavailable",
        "message": "Request safety controls are temporarily unavailable; retry shortly.",
        "retryable": True,
    }
    assert catalog.status_code == 503
    assert catalog.json()["code"] == "safety_state_unavailable"
    assert [operation for operation, _, _ in gate.calls] == ["create", "read"]


def test_committed_create_and_action_replays_bypass_exhausted_buckets():
    gate = MutableAdmissionGate()
    client = _client(gate)
    create_id = uuid4()
    created = _create(client, create_id)
    assert created.status_code == 200
    view = created.json()
    action_id = uuid4()
    action = {
        "type": "request_hint",
        "request_id": str(action_id),
        "expected_revision": view["revision"],
        "pending_key": view["pending"]["key"],
    }
    committed = client.post(
        f"/api/v2/sessions/{view['session_id']}/actions",
        json=action,
    )
    assert committed.status_code == 200

    gate.decisions["create"] = AdmissionDecision(
        allowed=False,
        retry_after_seconds=60,
    )
    gate.decisions["action"] = AdmissionDecision(
        allowed=False,
        retry_after_seconds=1,
    )
    replayed_create = _create(client, create_id)
    replayed_action = client.post(
        f"/api/v2/sessions/{view['session_id']}/actions",
        json=action,
    )
    next_action = {
        "type": "request_hint",
        "request_id": str(uuid4()),
        "expected_revision": committed.json()["revision"],
        "pending_key": committed.json()["pending"]["key"],
    }
    blocked = client.post(
        f"/api/v2/sessions/{view['session_id']}/actions",
        json=next_action,
    )

    assert replayed_create.status_code == 200
    assert replayed_create.json() == created.json()
    assert replayed_action.status_code == 200
    assert replayed_action.json() == committed.json()
    assert blocked.status_code == 429
    assert blocked.headers["retry-after"] == "1"


def test_quarantine_precedes_rate_limits_reads_recovery_and_receipt_replay():
    gate = MutableAdmissionGate()
    quarantine = MutableQuarantine()
    client = _client(gate, quarantine=quarantine)
    create_id = uuid4()
    created = _create(client, create_id)
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
    calls_before_quarantine = list(gate.calls)

    gate.decisions = {
        operation: AdmissionDecision(allowed=False, retry_after_seconds=60)
        for operation in ("create", "recover", "read", "action")
    }
    quarantine.digests = frozenset(
        {client.app.state.v2_active_release.release_digest}
    )
    responses = (
        client.get("/api/v2/sessions/current"),
        client.get(f"/api/v2/sessions/{view['session_id']}"),
        client.post(
            f"/api/v2/sessions/{view['session_id']}/actions",
            json=action,
        ),
        _create(client, create_id),
        client.post(
            "/api/v2/sessions/recover",
            json={
                "schema_version": 1,
                "operation": "create",
                "request_id": str(create_id),
            },
        ),
    )

    for response in responses:
        assert response.status_code == 410, response.text
        payload = response.json()
        assert payload["code"] == "release_quarantined"
        assert "session" not in payload
        assert "transcript" not in repr(payload)
        assert "pending" not in repr(payload)
        assert "retry-after" not in response.headers
    assert gate.calls == calls_before_quarantine
