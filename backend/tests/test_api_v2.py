"""API v2: authoritative snapshots, idempotency, locking, and resume tokens."""

import logging
import threading
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from tutor.api.app import create_app
from tutor.api.v2 import (
    _create_rollout_cookie,
    _rollout_assignment,
    install_v2_routes,
)
from tutor.api.v2_features import V2FeatureFlags
from tutor.api.v2_persistence import V2PersistenceService
from tutor.api.v2_store import MutationReceipt
from tutor.db import models as m
from tutor.db.persistence import PersistenceService
from tutor.db.session import get_engine
from tutor.orchestrator.machine import SessionPhase
import tutor.orchestrator.session_v2 as session_v2_module
from tutor.seed.load_seed import load_graph
from tutor.verify.checker import VerificationResult, VerificationStatus

from tests.v2_helpers import (
    approved_power_rule_episode_bank,
    approved_power_rule_catalog,
    approved_power_rule_stress_bank,
    power_rule_only_graph,
)


@pytest.fixture()
def client() -> TestClient:
    return TestClient(_v2_app())


def _v2_app(
    *,
    durable: bool = False,
    persistence: V2PersistenceService | None = None,
    resume_token_secret: bytes | str | None = None,
    feature_flags: V2FeatureFlags | None = None,
    item_bank_override=None,
) -> FastAPI:
    app = FastAPI()
    graph = power_rule_only_graph()
    item_bank = item_bank_override or approved_power_rule_stress_bank()
    if durable and persistence is None:
        legacy = PersistenceService(
            engine=get_engine("sqlite+pysqlite:///:memory:")
        )
        persistence = V2PersistenceService(legacy.engine)
    if persistence is not None:
        app.state.persistence = persistence
    install_v2_routes(
        app,
        graph,
        persistence=persistence,
        available_targets=("kc.der.power_rule",),
        item_bank=item_bank,
        pedagogy_catalog=approved_power_rule_catalog(),
        resume_token_secret=resume_token_secret,
        feature_flags=feature_flags,
    )
    return app


def _create(client: TestClient, request_id=None, **overrides):
    payload = {
        "request_id": str(request_id or uuid4()),
        "goal_id": "goal.der.power_rule",
        **overrides,
    }
    return client.post("/api/v2/sessions", json=payload), payload


def _hint(view: dict, request_id=None, **overrides) -> dict:
    return {
        "type": "request_hint",
        "request_id": str(request_id or uuid4()),
        "expected_revision": view["revision"],
        "pending_key": view["pending"]["key"],
        **overrides,
    }


def _answer(view: dict, answer: str, request_id=None) -> dict:
    return {
        "type": "answer",
        "request_id": str(request_id or uuid4()),
        "expected_revision": view["revision"],
        "pending_key": view["pending"]["key"],
        "answer": answer,
    }


def _widget(view: dict, text: str, request_id=None) -> dict:
    return {
        "type": "widget_attempt",
        "request_id": str(request_id or uuid4()),
        "expected_revision": view["revision"],
        "pending_key": view["pending"]["key"],
        "response": {"text": text},
    }


def _reset(view: dict, request_id=None, **overrides) -> dict:
    return {
        "request_id": str(request_id or uuid4()),
        "expected_revision": view["revision"],
        "pending_key": view["pending"]["key"] if view["pending"] else None,
        **overrides,
    }


def _complete_perfect_session(client: TestClient, view: dict) -> dict:
    for _ in range(6):
        if view["phase"] == "done":
            return view
        handle = client.app.state.v2_store.get(view["session_id"])
        response = client.post(
            f"/api/v2/sessions/{view['session_id']}/actions",
            json=_answer(view, handle.orchestrator.pending_expected),
        )
        assert response.status_code == 200
        view = response.json()
    raise AssertionError("perfect v2 session did not reach done")


def _single_episode_inventory():
    """Keep exactly the bounded inventory needed before the first item is shown."""

    return approved_power_rule_episode_bank()


def _set_resume_cookie(client: TestClient, raw_token: str) -> None:
    client.cookies.set(
        "tutor_resume_v2",
        raw_token,
        domain="testserver.local",
        path="/api/v2",
    )


def test_context_matching_first_truth_skips_that_diagnostic_family(client):
    created, _ = _create(
        client,
        context="My existing notes say the result is 3*x^2.",
    )

    assert created.status_code == 200
    view = created.json()
    handle = client.app.state.v2_store.get(view["session_id"])
    assert view["context"] == "My existing notes say the result is 3*x^2."
    assert handle.orchestrator.pending.item_id == "item.power.diagnostic.quartic"
    assert handle.orchestrator.pending_expected == "4*x^3"
    assert all(
        reservation.family_id != "family.power.diagnostic.cube"
        for reservation in handle.orchestrator.exposure_state.reservations
    )


def test_wrong_answer_collision_skip_survives_durable_process_restore():
    persistence = V2PersistenceService(
        PersistenceService(
            engine=get_engine("sqlite+pysqlite:///:memory:")
        ).engine
    )
    secret = b"visible-ledger-resume-secret-32-bytes"
    first = TestClient(
        _v2_app(persistence=persistence, resume_token_secret=secret)
    )
    created, _ = _create(first)
    view = created.json()
    answered = first.post(
        f"/api/v2/sessions/{view['session_id']}/actions",
        json=_answer(view, "4*x^3"),
    )

    assert answered.status_code == 200
    expected = answered.json()
    first_handle = first.app.state.v2_store.get(view["session_id"])
    assert first_handle.orchestrator.pending.item_id == "item.power.diagnostic.sixth"
    raw_token = first.cookies.get("tutor_resume_v2")

    restarted = TestClient(
        _v2_app(persistence=persistence, resume_token_secret=secret)
    )
    _set_resume_cookie(restarted, raw_token)
    restored = restarted.get("/api/v2/sessions/current")

    assert restored.status_code == 200
    assert restored.json() == expected
    restored_handle = restarted.app.state.v2_store.get(view["session_id"])
    assert restored_handle.orchestrator.pending.item_id == "item.power.diagnostic.sixth"
    assert "4*x^3" in restored_handle.orchestrator._visible_texts


def _set_rollout_cookie(client: TestClient, raw_token: str) -> None:
    client.cookies.set(
        "tutor_rollout_v2",
        raw_token,
        domain="testserver.local",
        path="/api/v2",
    )


def _rollout_cookie_for(
    secret: bytes,
    percentage: int,
    *,
    selected: bool,
) -> str:
    for marker in range(1_000):
        cookie = _create_rollout_cookie(secret, marker.to_bytes(32, "big"))
        assignment = _rollout_assignment(secret, percentage, cookie)
        if assignment.selected is selected:
            return cookie
    raise AssertionError("could not construct the requested deterministic cohort")


def _assert_no_expected_key(value):
    if isinstance(value, dict):
        assert "expected" not in value
        for child in value.values():
            _assert_no_expected_key(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_expected_key(child)


def test_goal_catalog_and_safe_authoritative_create(client):
    catalog = client.get("/api/v2/goals")
    assert catalog.status_code == 200
    assert catalog.json()["goals"]
    assert catalog.json()["rollout"] == {
        "status": "available",
        "reason": "This browser is included in the current pilot rollout.",
        "percentage": 100,
    }
    assert "goal.der.power_rule" in {
        goal["goal_id"] for goal in catalog.json()["goals"]
    }

    response, _ = _create(client, context="Reviewing before a quiz.")
    assert response.status_code == 200
    view = response.json()
    assert view["schema_version"] == 2
    assert view["revision"] == 0
    assert view["phase"] == "diagnose"
    assert view["pending"]["key"]
    assert view["pending"]["skill_name"] == "Power rule"
    assert view["durability"] == "memory_only"
    assert view["context"] == "Reviewing before a quiz."
    assert view["content_mode"] == {
        "requested": "curated",
        "effective": "curated",
        "fallback_reason": None,
    }
    _assert_no_expected_key(view)
    set_cookie = response.headers["set-cookie"].lower()
    assert "httponly" in set_cookie
    assert "samesite=lax" in set_cookie
    assert "path=/api/v2" in set_cookie
    assert "secure" not in set_cookie

    current = client.get("/api/v2/sessions/current")
    assert current.status_code == 200
    assert current.json() == view
    explicit = client.get(f"/api/v2/sessions/{view['session_id']}")
    assert explicit.status_code == 200
    assert explicit.json() == view

    external = TestClient(_v2_app(), base_url="https://tutor.example")
    external_response = external.post(
        "/api/v2/sessions",
        headers={"origin": "https://tutor.example"},
        json={
            "request_id": str(uuid4()),
            "goal_id": "goal.der.power_rule",
        },
    )
    assert external_response.status_code == 200
    assert "secure" in external_response.headers["set-cookie"].lower()


def test_verifier_saturation_returns_retryable_snapshot_without_committing(
    client,
    monkeypatch,
):
    created, _ = _create(client)
    assert created.status_code == 200
    before = created.json()
    handle = client.app.state.v2_store.get(before["session_id"])
    answer = handle.orchestrator.pending_expected
    request_id = uuid4()
    action = _answer(before, answer, request_id=request_id)

    real_verify = session_v2_module.verify_answer

    def saturated(_answer, _given, **_kwargs):
        return VerificationResult(
            status=VerificationStatus.TIMEOUT,
            code="verifier_saturated",
        )

    monkeypatch.setattr(session_v2_module, "verify_answer", saturated)
    failed = client.post(
        f"/api/v2/sessions/{before['session_id']}/actions",
        json=action,
    )

    assert failed.status_code == 503
    assert failed.json()["code"] == "verification_capacity_unavailable"
    assert failed.json()["retryable"] is True
    assert failed.json()["session"] == before
    assert client.app.state.v2_store.view(handle).model_dump(mode="json") == before

    monkeypatch.setattr(session_v2_module, "verify_answer", real_verify)
    retried = client.post(
        f"/api/v2/sessions/{before['session_id']}/actions",
        json=action,
    )
    assert retried.status_code == 200
    assert retried.json()["revision"] == before["revision"] + 1


def test_rich_widget_flag_keeps_core_flow_and_uses_text_guided_practice():
    flags = V2FeatureFlags(rich_widgets=False)
    guarded = TestClient(_v2_app(feature_flags=flags))

    catalog = guarded.get("/api/v2/goals").json()
    assert catalog["goals"]
    assert catalog["rollout"]["status"] == "available"
    manifest = guarded.get("/api/v2/capabilities").json()
    assert "live_input" not in manifest["supported"]
    assert "live_input" in manifest["disabled"]

    created, _ = _create(guarded)
    assert created.status_code == 200
    view = created.json()
    orchestrator = guarded.app.state.v2_store.get(view["session_id"]).orchestrator
    orchestrator._probe_budget = 2
    orchestrator._diag.state.probe_budget = 2
    for _ in range(2):
        response = guarded.post(
            f"/api/v2/sessions/{view['session_id']}/actions",
            json=_answer(view, "0"),
        )
        assert response.status_code == 200
        view = response.json()

    assert view["phase"] == "teach"
    assert view["pending"]["kind"] == "guided_widget"
    assert view["pending"]["input_mode"] == "math"
    assert not any(entry["widget"] for entry in view["transcript"])
    guided_key = view["pending"]["key"]
    guided_answer = guarded.app.state.v2_store.get(
        view["session_id"]
    ).orchestrator.pending_expected
    practiced = guarded.post(
        f"/api/v2/sessions/{view['session_id']}/actions",
        json=_answer(view, guided_answer),
    )
    assert practiced.status_code == 200
    practiced_view = practiced.json()
    assert practiced_view["pending"]["key"] != guided_key
    assert practiced_view["pending"]["kind"] == "checkin"


def test_live_input_stays_disabled_with_rich_flag_and_after_durable_restore():
    secret = b"capability-restore-secret-material-32-bytes"
    persistence = V2PersistenceService(
        PersistenceService(
            engine=get_engine("sqlite+pysqlite:///:memory:")
        ).engine
    )
    first = TestClient(
        _v2_app(
            persistence=persistence,
            resume_token_secret=secret,
            feature_flags=V2FeatureFlags(rich_widgets=True),
        )
    )
    first_manifest = first.get("/api/v2/capabilities").json()
    assert "live_input" not in first_manifest["supported"]
    assert "render semantics" in first_manifest["disabled"]["live_input"]
    created, _ = _create(first)
    view = created.json()
    handle = first.app.state.v2_store.get(view["session_id"])
    handle.orchestrator._probe_budget = 2
    handle.orchestrator._diag.state.probe_budget = 2
    for _ in range(2):
        response = first.post(
            f"/api/v2/sessions/{view['session_id']}/actions",
            json=_answer(view, "0"),
        )
        assert response.status_code == 200
        view = response.json()
    assert view["pending"]["input_mode"] == "math"
    assert not any(entry["widget"] for entry in view["transcript"])
    raw_token = first.cookies.get("tutor_resume_v2")

    restarted = TestClient(
        _v2_app(
            persistence=persistence,
            resume_token_secret=secret,
            feature_flags=V2FeatureFlags(rich_widgets=False),
        )
    )
    restarted.cookies.set("tutor_resume_v2", raw_token, path="/api/v2")
    restored = restarted.get("/api/v2/sessions/current")

    assert restored.status_code == 200
    restored_view = restored.json()
    assert restored_view["pending"]["key"] == view["pending"]["key"]
    assert restored_view["pending"]["input_mode"] == "math"
    manifest = restarted.get("/api/v2/capabilities").json()
    assert "live_input" not in manifest["supported"]
    restored_handle = restarted.app.state.v2_store.get(view["session_id"])
    practiced = restarted.post(
        f"/api/v2/sessions/{view['session_id']}/actions",
        json=_answer(
            restored_view,
            restored_handle.orchestrator.pending_expected,
        ),
    )
    assert practiced.status_code == 200
    assert practiced.json()["pending"]["kind"] == "checkin"


@pytest.mark.parametrize("percentage", [0, 5, 25, 100])
def test_rollout_percentage_environment_accepts_only_release_steps(
    monkeypatch,
    percentage,
):
    monkeypatch.setenv("TUTOR_V2_STUDENT_ROLLOUT_PERCENT", str(percentage))
    assert V2FeatureFlags.from_environment().student_rollout_percent == percentage


def test_pilot_production_requires_every_rollout_switch_to_be_explicit(monkeypatch):
    names = (
        "TUTOR_ENABLE_API_SESSION_V2",
        "TUTOR_ENABLE_CONTENT_ALLOCATION_V2",
        "TUTOR_ENABLE_DIAGNOSIS_V2",
        "TUTOR_ENABLE_LESSON_FLOW_V2",
        "TUTOR_ENABLE_RICH_WIDGETS_V2",
        "TUTOR_PAUSE_V2_MUTATIONS",
        "TUTOR_V2_STUDENT_ROLLOUT_PERCENT",
    )
    monkeypatch.setenv("TUTOR_PILOT_PRODUCTION", "1")
    for name in names:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RuntimeError, match="requires explicit v2 rollout"):
        V2FeatureFlags.from_environment()

    for name in names[:-1]:
        monkeypatch.setenv(
            name,
            "0" if name == "TUTOR_PAUSE_V2_MUTATIONS" else "1",
        )
    monkeypatch.setenv(names[-1], "5")
    flags = V2FeatureFlags.from_environment()
    assert flags.student_rollout_percent == 5
    assert flags.student_stack_enabled is True


@pytest.mark.parametrize("value", ["-1", "1", "50", "101", "five"])
def test_rollout_percentage_environment_rejects_unsafe_values(monkeypatch, value):
    monkeypatch.setenv("TUTOR_V2_STUDENT_ROLLOUT_PERCENT", value)
    with pytest.raises(ValueError, match="must be one of 0, 5, 25, or 100"):
        V2FeatureFlags.from_environment()


@pytest.mark.parametrize("selected", [False, True])
def test_rollout_assignment_is_stable_and_enforced_across_processes(selected):
    secret = b"stable-test-resume-secret-material-32-bytes"
    percentage = 25
    cohort_cookie = _rollout_cookie_for(
        secret,
        percentage,
        selected=selected,
    )
    flags = V2FeatureFlags(student_rollout_percent=percentage)

    client_a = TestClient(
        _v2_app(resume_token_secret=secret, feature_flags=flags)
    )
    _set_rollout_cookie(client_a, cohort_cookie)
    first = client_a.get("/api/v2/goals")
    second = client_a.get("/api/v2/goals")
    assert first.json() == second.json()
    assert client_a.cookies.get("tutor_rollout_v2") == cohort_cookie

    client_b = TestClient(
        _v2_app(resume_token_secret=secret, feature_flags=flags)
    )
    _set_rollout_cookie(client_b, cohort_cookie)
    across_process = client_b.get("/api/v2/goals")
    assert across_process.json() == first.json()

    if selected:
        assert first.json()["rollout"]["status"] == "available"
        assert first.json()["goals"]
        created, _ = _create(client_b)
        assert created.status_code == 200
    else:
        assert first.json()["rollout"]["status"] == "not_selected"
        assert first.json()["goals"] == []
        created, _ = _create(client_b)
        assert created.status_code == 403
        assert created.json()["code"] == "rollout_not_selected"


def test_zero_percent_rollout_does_not_misreport_draft_content_as_review_failure(
    monkeypatch,
):
    monkeypatch.setenv("TUTOR_V2_STUDENT_ROLLOUT_PERCENT", "0")
    guarded = TestClient(create_app(load_graph()))

    catalog = guarded.get("/api/v2/goals").json()

    assert catalog["goals"] == []
    assert catalog["rollout"]["status"] == "not_selected"
    assert catalog["rollout"]["percentage"] == 0
    assert "review" not in catalog["rollout"]["reason"].lower()


def test_rollout_pause_does_not_strand_or_break_exact_replay_of_existing_session():
    legacy = PersistenceService(
        engine=get_engine("sqlite+pysqlite:///:memory:")
    )
    persistence = V2PersistenceService(legacy.engine)
    secret = b"stable-test-resume-secret-material-32-bytes"
    admitted = TestClient(
        _v2_app(
            persistence=persistence,
            resume_token_secret=secret,
            feature_flags=V2FeatureFlags(student_rollout_percent=100),
        )
    )
    created, payload = _create(admitted)
    resume_cookie = admitted.cookies.get("tutor_resume_v2")
    rollout_cookie = admitted.cookies.get("tutor_rollout_v2")

    paused = TestClient(
        _v2_app(
            persistence=persistence,
            resume_token_secret=secret,
            feature_flags=V2FeatureFlags(student_rollout_percent=0),
        )
    )
    _set_resume_cookie(paused, resume_cookie)
    _set_rollout_cookie(paused, rollout_cookie)

    assert paused.get("/api/v2/goals").json()["rollout"]["status"] == "not_selected"
    assert paused.get("/api/v2/sessions/current").json() == created.json()
    replayed = paused.post("/api/v2/sessions", json=payload)
    assert replayed.status_code == 200
    assert replayed.json() == created.json()

    action = paused.post(
        f"/api/v2/sessions/{created.json()['session_id']}/actions",
        json=_hint(created.json()),
    )
    assert action.status_code == 200
    assert action.json()["revision"] == 1


def test_transcript_widget_projection_never_exposes_server_scoring_fields():
    from tutor.api.v2_store import V2SessionStore
    from tutor.orchestrator.machine import Interaction

    [entry] = V2SessionStore._interaction_entries(
        [],
        [
            Interaction(
                key="widget-1",
                kind="lesson",
                text="Practice",
                widget={
                    "widget_type": "live_input",
                    "learning_objective": "Practice",
                    "prompt": "Enter a value",
                    "input_kind": "expression",
                    "checker": {"expected": "secret", "equivalence": "sympy_equiv"},
                    "feedback_rules": [{"if": "x < 2", "say": "secret"}],
                    "render": {
                        "plot": "y = k*x",
                        "var": "k",
                        "expected": "secret",
                    },
                },
            )
        ],
    )

    assert entry.widget == {
        "widget_type": "live_input",
        "learning_objective": "Practice",
        "prompt": "Enter a value",
        "text_fallback": "",
        "input_kind": "expression",
        "render": {"plot": "y = k*x", "var": "k"},
    }
    _assert_no_expected_key(entry.model_dump())


def test_integrated_catalog_hides_goals_without_full_reviewed_closure():
    graph = load_graph()
    client = TestClient(create_app(graph))
    catalog = client.get("/api/v2/goals").json()
    from tutor.content.item_bank import load_item_bank
    from tutor.graph.service import ancestor_subgraph

    released = set(load_item_bank().released_kcs)
    assert released == set()
    assert catalog["goals"] == []
    assert catalog["rollout"]["status"] == "content_unavailable"
    for goal in catalog["goals"]:
        closure = ancestor_subgraph(
            graph, goal["target_kc"], hard_only=True
        ).node_ids()
        assert closure <= released


def test_create_is_idempotent_and_one_episode_is_active(client):
    request_id = uuid4()
    first, payload = _create(client, request_id=request_id)
    second = client.post("/api/v2/sessions", json=payload)
    assert second.status_code == 200
    assert second.json() == first.json()

    conflicting = client.post(
        "/api/v2/sessions",
        json={**payload, "course": "Calculus I"},
    )
    assert conflicting.status_code == 409
    assert conflicting.json()["code"] == "idempotency_conflict"

    another, _ = _create(client)
    assert another.status_code == 409
    assert another.json()["code"] == "active_session_exists"
    assert another.json()["session"]["session_id"] == first.json()["session_id"]


def test_create_replays_when_the_first_response_cookie_was_lost(client):
    request_id = uuid4()
    created, payload = _create(client, request_id=request_id)
    expected_cookie = client.cookies.get("tutor_resume_v2")

    client.cookies.clear()
    replayed = client.post("/api/v2/sessions", json=payload)

    assert replayed.status_code == 200
    assert replayed.json() == created.json()
    assert client.cookies.get("tutor_resume_v2") == expected_cookie

    client.cookies.clear()
    conflict = client.post(
        "/api/v2/sessions", json={**payload, "context": "different payload"}
    )
    assert conflict.status_code == 409
    assert conflict.json()["code"] == "idempotency_conflict"


def test_create_replays_across_processes_with_a_stable_resume_secret():
    legacy = PersistenceService(
        engine=get_engine("sqlite+pysqlite:///:memory:")
    )
    persistence = V2PersistenceService(legacy.engine)
    secret = b"stable-test-resume-secret-material-32-bytes"
    app_a = _v2_app(persistence=persistence, resume_token_secret=secret)
    client_a = TestClient(app_a)
    request_id = uuid4()
    created, payload = _create(client_a, request_id=request_id)
    expected_cookie = client_a.cookies.get("tutor_resume_v2")

    app_b = _v2_app(persistence=persistence, resume_token_secret=secret)
    client_b = TestClient(app_b)
    replayed = client_b.post("/api/v2/sessions", json=payload)

    assert replayed.status_code == 200
    assert replayed.json() == created.json()
    assert client_b.cookies.get("tutor_resume_v2") == expected_cookie


def test_initial_create_response_loss_recovers_without_persisting_create_payload():
    legacy = PersistenceService(
        engine=get_engine("sqlite+pysqlite:///:memory:")
    )
    persistence = V2PersistenceService(legacy.engine)
    secret = b"stable-test-resume-secret-material-32-bytes"
    request_id = uuid4()
    client = TestClient(
        _v2_app(persistence=persistence, resume_token_secret=secret)
    )
    created, _ = _create(
        client,
        request_id=request_id,
        context="private context that is not part of the recovery proof",
    )
    expected_cookie = client.cookies.get("tutor_resume_v2")

    reloaded = TestClient(
        _v2_app(persistence=persistence, resume_token_secret=secret)
    )
    proof = {
        "schema_version": 1,
        "operation": "create",
        "request_id": str(request_id),
    }
    blocked = reloaded.post(
        "/api/v2/sessions/recover",
        json=proof,
        headers={"Origin": "https://attacker.example"},
    )
    recovered = reloaded.post("/api/v2/sessions/recover", json=proof)

    assert blocked.status_code == 403
    assert recovered.status_code == 200
    assert recovered.json()["session_id"] == created.json()["session_id"]
    assert reloaded.cookies.get("tutor_resume_v2") == expected_cookie
    assert reloaded.get("/api/v2/sessions/current").json() == created.json()
    assert "private context" not in str(proof)


def test_completed_episode_reuses_anonymous_learner_and_longitudinal_evidence(client):
    created, _ = _create(client)
    first_view = created.json()
    first_handle = client.app.state.v2_store.get(first_view["session_id"])
    answer = first_handle.orchestrator.pending_expected
    answered = client.post(
        f"/api/v2/sessions/{first_view['session_id']}/actions",
        json=_answer(first_view, answer),
    )
    assert answered.status_code == 200

    first_handle = client.app.state.v2_store.get(first_view["session_id"])
    learner_id = first_handle.orchestrator.learner.learner_id
    historical_events = tuple(first_handle.orchestrator.learner.events)
    assert historical_events
    first_handle.orchestrator.phase = SessionPhase.DONE
    first_handle.orchestrator._pending = None

    next_created, _ = _create(client)
    assert next_created.status_code == 200
    next_handle = client.app.state.v2_store.get(next_created.json()["session_id"])
    assert next_handle.orchestrator.learner.learner_id == learner_id
    assert next_handle.orchestrator.learner.events == historical_events
    assert next_handle.orchestrator.learner.as_of > historical_events[-1].t


def test_terminal_rollover_validates_goal_before_revoking_durable_resume():
    app = _v2_app(durable=True)
    client = TestClient(app)
    created, _ = _create(client)
    terminal = _complete_perfect_session(client, created.json())
    old_cookie = client.cookies.get("tutor_resume_v2")

    rejected, _ = _create(client, goal_id="goal.unknown")

    assert rejected.status_code == 404
    assert client.cookies.get("tutor_resume_v2") == old_cookie
    assert client.get("/api/v2/sessions/current").json() == terminal
    with Session(app.state.persistence.engine) as session:
        tokens = session.scalars(select(m.ResumeTokenRow)).all()
        checkpoints = session.scalars(select(m.SessionCheckpointRow)).all()
        assert len(tokens) == 1
        assert tokens[0].revoked is False
        assert len(checkpoints) == 1


def test_terminal_rollover_persistence_failure_keeps_old_episode_resumable():
    app = _v2_app(durable=True)
    client = TestClient(app)
    created, _ = _create(client)
    terminal = _complete_perfect_session(client, created.json())
    old_cookie = client.cookies.get("tutor_resume_v2")
    durable = app.state.v2_store._persistence

    class _FailRolloverPersistence:
        def __getattr__(self, name):
            return getattr(durable, name)

        def create_session(self, *args, **kwargs):
            if kwargs.get("replace_session_id") is not None:
                raise RuntimeError("injected rollover failure")
            return durable.create_session(*args, **kwargs)

    app.state.v2_store._persistence = _FailRolloverPersistence()
    failed, _ = _create(client)

    assert failed.status_code == 503
    assert client.cookies.get("tutor_resume_v2") == old_cookie
    assert client.get("/api/v2/sessions/current").json() == terminal
    with Session(app.state.persistence.engine) as session:
        tokens = session.scalars(select(m.ResumeTokenRow)).all()
        checkpoints = session.scalars(select(m.SessionCheckpointRow)).all()
        assert len(tokens) == 1
        assert tokens[0].revoked is False
        assert len(checkpoints) == 1


def test_terminal_rollover_exact_retry_with_revoked_old_cookie_replays_new_episode():
    secret = b"stable-test-resume-secret-material-32-bytes"
    app = _v2_app(durable=True, resume_token_secret=secret)
    client = TestClient(app)
    created, _ = _create(client)
    _complete_perfect_session(client, created.json())
    old_cookie = client.cookies.get("tutor_resume_v2")
    request_id = uuid4()

    rollover, payload = _create(client, request_id=request_id)
    assert rollover.status_code == 200
    replacement = rollover.json()
    replacement_cookie = client.cookies.get("tutor_resume_v2")
    assert replacement_cookie != old_cookie

    _set_resume_cookie(client, old_cookie)
    replayed = client.post("/api/v2/sessions", json=payload)

    assert replayed.status_code == 200
    assert replayed.json() == replacement
    assert client.cookies.get("tutor_resume_v2") == replacement_cookie
    with Session(app.state.persistence.engine) as session:
        tokens = session.scalars(
            select(m.ResumeTokenRow).order_by(m.ResumeTokenRow.id)
        ).all()
        assert len(tokens) == 2
        assert tokens[0].revoked is True
        assert tokens[1].revoked is False


def test_terminal_create_response_loss_recovers_with_request_proof_on_reload():
    legacy = PersistenceService(
        engine=get_engine("sqlite+pysqlite:///:memory:")
    )
    persistence = V2PersistenceService(legacy.engine)
    secret = b"stable-test-resume-secret-material-32-bytes"
    app = _v2_app(persistence=persistence, resume_token_secret=secret)
    client = TestClient(app)
    created, _ = _create(client)
    _complete_perfect_session(client, created.json())
    old_cookie = client.cookies.get("tutor_resume_v2")
    private_context = "I am privately worried about this course."

    request_id = uuid4()
    committed, _ = _create(
        client,
        request_id=request_id,
        context=private_context,
    )

    assert committed.status_code == 200
    expected = committed.json()
    replacement_cookie = client.cookies.get("tutor_resume_v2")
    assert replacement_cookie != old_cookie

    # Model a committed response disappearing before its Set-Cookie reaches
    # the browser, followed by a page reload handled by another process.
    reloaded_app = _v2_app(
        persistence=persistence,
        resume_token_secret=secret,
    )
    reloaded_client = TestClient(reloaded_app)
    missing_cookie = reloaded_client.post(
        "/api/v2/sessions/recover",
        json={
            "schema_version": 1,
            "operation": "reset",
            "request_id": str(request_id),
        },
    )
    _set_resume_cookie(reloaded_client, old_cookie)
    unauthorized = reloaded_client.get("/api/v2/sessions/current")
    proof = {
        "schema_version": 1,
        "operation": "create",
        "request_id": str(request_id),
    }
    recovered = reloaded_client.post("/api/v2/sessions/recover", json=proof)
    restored = reloaded_client.get("/api/v2/sessions/current")

    assert missing_cookie.status_code == 401
    assert unauthorized.status_code == 401
    assert recovered.status_code == 200
    assert recovered.json() == {
        "recovered": True,
        "session_id": expected["session_id"],
    }
    assert restored.status_code == 200
    assert restored.json() == expected
    assert reloaded_client.cookies.get("tutor_resume_v2") == replacement_cookie
    assert private_context not in str(proof)


def test_terminal_rollover_carries_revealed_family_retirement():
    app = _v2_app(durable=True)
    client = TestClient(app)
    created, _ = _create(client)
    view = created.json()
    revealed_family = app.state.v2_store.get(
        view["session_id"]
    ).orchestrator.pending.family_id
    for _ in range(3):
        hinted = client.post(
            f"/api/v2/sessions/{view['session_id']}/actions",
            json=_hint(view),
        )
        assert hinted.status_code == 200
        view = hinted.json()

    terminal = _complete_perfect_session(client, view)
    assert terminal["phase"] == "done"
    rollover, _ = _create(client)

    assert rollover.status_code == 200
    replacement = app.state.v2_store.get(rollover.json()["session_id"])
    assert revealed_family in replacement.orchestrator.exposure_state.retired_family_ids
    assert replacement.orchestrator.pending is not None
    assert replacement.orchestrator.pending.family_id != revealed_family
    assert rollover.json()["phase"] == "diagnose"


def test_create_preflight_rejects_incomplete_inventory_without_a_session():
    bank = _single_episode_inventory()
    bank = bank.model_copy(
        update={
            "items": [
                item
                for item in bank.items
                if item.item_id != "item.power.checkin.scaled-sixth"
            ]
        }
    )
    app = _v2_app(item_bank_override=bank)
    client = TestClient(app)

    response, _ = _create(client)

    assert response.status_code == 409
    assert response.json()["code"] == "content_exhausted"
    assert response.json().get("session") is None
    assert client.cookies.get("tutor_resume_v2") is None
    assert not app.state.v2_store._sessions


def test_reset_preflight_is_non_mutating_when_prior_exposure_exhausts_content():
    app = _v2_app(item_bank_override=_single_episode_inventory())
    client = TestClient(app)
    created, _ = _create(client)
    before = created.json()
    raw_token = client.cookies.get("tutor_resume_v2")
    handle = app.state.v2_store.get(before["session_id"])
    checkpoint = handle.orchestrator.export_checkpoint()

    stale = client.post(
        "/api/v2/sessions/current/reset",
        json=_reset(before, expected_revision=before["revision"] + 1),
    )
    assert stale.status_code == 409
    assert stale.json()["code"] == "stale_interaction"

    response = client.post(
        "/api/v2/sessions/current/reset",
        json=_reset(before),
    )

    assert response.status_code == 409
    assert response.json()["code"] == "content_exhausted"
    assert response.json()["session"] == before
    assert client.cookies.get("tutor_resume_v2") == raw_token
    assert (
        app.state.v2_store.get(before["session_id"]).orchestrator.export_checkpoint()
        == checkpoint
    )
    assert len(app.state.v2_store._sessions) == 1


def test_terminal_rollover_preflight_keeps_completed_episode_authoritative():
    app = _v2_app(item_bank_override=_single_episode_inventory())
    client = TestClient(app)
    created, _ = _create(client)
    completed = _complete_perfect_session(client, created.json())
    raw_token = client.cookies.get("tutor_resume_v2")

    response, _ = _create(client)

    assert response.status_code == 409
    assert response.json()["code"] == "content_exhausted"
    assert response.json()["session"] == completed
    assert client.cookies.get("tutor_resume_v2") == raw_token
    assert client.get("/api/v2/sessions/current").json() == completed


def test_hint_action_advances_revision_but_not_pending_answer(client):
    created, _ = _create(client)
    before = created.json()
    assert before["pending"]["hint"] == {
        "available": True,
        "next_index": 0,
        "total": 3,
        "next_reveals_answer": False,
    }
    response = client.post(
        f"/api/v2/sessions/{before['session_id']}/actions",
        json=_hint(before),
    )
    assert response.status_code == 200
    after = response.json()
    assert after["revision"] == before["revision"] + 1
    assert after["pending"]["key"] == before["pending"]["key"]
    assert after["pending"]["hint"]["next_index"] == 1
    assert not after["pending"]["hint"]["next_reveals_answer"]
    assert after["transcript"][-1]["kind"] == "hint"
    assert after["transcript"][-1]["role"] == "tutor"
    assert all(
        entry["role"] != "student" or entry["text"] != "hint"
        for entry in after["transcript"]
    )


def test_revealing_hint_abandons_item_without_mastery_evidence(client):
    created, _ = _create(client)
    view = created.json()
    original_key = view["pending"]["key"]
    handle = client.app.state.v2_store.get(view["session_id"])
    original_family = handle.orchestrator.pending.family_id
    for index in range(3):
        if index == 2:
            assert view["pending"]["hint"]["next_reveals_answer"]
        response = client.post(
            f"/api/v2/sessions/{view['session_id']}/actions",
            json=_hint(view),
        )
        assert response.status_code == 200
        view = response.json()

    handle = client.app.state.v2_store.get(view["session_id"])
    assert view["pending"]["key"] != original_key
    assert handle.orchestrator.pending.family_id != original_family
    assert original_family in handle.orchestrator.exposure_state.retired_family_ids
    assert not handle.orchestrator.learner.events
    assert any(
        "will not be scored" in entry["text"]
        for entry in view["transcript"]
    )


def test_post_diagnosis_unreleased_live_input_uses_text_without_widget_state(client):
    created, _ = _create(client)
    view = created.json()
    orchestrator = client.app.state.v2_store.get(view["session_id"]).orchestrator
    orchestrator._probe_budget = 2
    orchestrator._diag.state.probe_budget = 2

    for _ in range(2):
        response = client.post(
            f"/api/v2/sessions/{view['session_id']}/actions",
            json=_answer(view, "x"),
        )
        assert response.status_code == 200
        view = response.json()

    assert view["phase"] == "teach"
    assert view["pending"]["kind"] == "guided_widget"
    assert view["pending"]["input_mode"] == "math"
    assert not any(entry["widget"] for entry in view["transcript"])

    rejected_widget = client.post(
        f"/api/v2/sessions/{view['session_id']}/actions",
        json=_widget(view, "factorial(5)"),
    )
    assert rejected_widget.status_code == 404
    assert client.get("/api/v2/sessions/current").json() == view

    handle = client.app.state.v2_store.get(view["session_id"])
    assert not any(
        event.surface == "instructional_practice"
        for event in handle.orchestrator.learner.events
    )

    practiced = client.post(
        f"/api/v2/sessions/{view['session_id']}/actions",
        json=_answer(view, handle.orchestrator.pending_expected),
    )
    assert practiced.status_code == 200
    assert practiced.json()["pending"]["kind"] == "checkin"


def test_durable_widget_trajectory_distinguishes_invalid_from_incorrect(monkeypatch):
    # Exercise the dormant widget-attempt ledger with an explicit test-only
    # capability. The production manifest intentionally cannot release this
    # widget until its render semantics have been reviewed end to end.
    import tutor.api.v2 as v2_module

    monkeypatch.setattr(
        v2_module,
        "widget_capability_manifest",
        lambda *, rich_widgets=True: {
            "version": "web-widget-capabilities-v2.1",
            "supported": {
                "mapping": {
                    "keyboard_equivalent": True,
                    "live_visual": False,
                },
                "live_input": {
                    "keyboard_equivalent": True,
                    "live_visual": True,
                },
            },
            "disabled": {
                "slider": "Test-only manifest.",
                "click_region": "Test-only manifest.",
            },
        },
    )
    app = _v2_app(durable=True)
    client = TestClient(app)
    created, _ = _create(client)
    view = created.json()
    orchestrator = app.state.v2_store.get(view["session_id"]).orchestrator
    orchestrator._probe_budget = 2
    orchestrator._diag.state.probe_budget = 2
    for _ in range(2):
        response = client.post(
            f"/api/v2/sessions/{view['session_id']}/actions",
            json=_answer(view, "x"),
        )
        assert response.status_code == 200
        view = response.json()

    invalid = client.post(
        f"/api/v2/sessions/{view['session_id']}/actions",
        json=_widget(view, "factorial(5)"),
    )
    assert invalid.status_code == 200
    with Session(app.state.persistence.engine) as session:
        attempt = session.scalars(select(m.WidgetAttemptRow)).one()
        assert attempt.attempt_number == 1
        assert attempt.verification_status == "invalid"
        assert attempt.counted is False
        assert attempt.correct is False


def test_widget_payload_limits_reject_oversized_nested_state_without_mutation(client):
    created, _ = _create(client)
    view = created.json()
    payload = _widget(view, "0")
    payload["response"] = {"nested": [[[[[[[[["too deep"]]]]]]]]]}

    rejected = client.post(
        f"/api/v2/sessions/{view['session_id']}/actions",
        json=payload,
    )

    assert rejected.status_code == 422
    assert client.get("/api/v2/sessions/current").json() == view


def test_answer_length_matches_verifier_limit_before_transcript_mutation(client):
    created, _ = _create(client)
    view = created.json()

    rejected = client.post(
        f"/api/v2/sessions/{view['session_id']}/actions",
        json=_answer(view, "x" * 257),
    )

    assert rejected.status_code == 422
    assert client.get("/api/v2/sessions/current").json() == view


def test_no_refresh_journey_treats_typed_hint_as_invalid_then_confirms_independently(
    client,
):
    created, _ = _create(client)
    view = created.json()
    session_id = view["session_id"]
    handle = client.app.state.v2_store.get(session_id)
    first_family = handle.orchestrator.pending.family_id
    events_before = len(handle.orchestrator.learner.events)

    typed_hint = client.post(
        f"/api/v2/sessions/{session_id}/actions",
        json=_answer(view, "hint"),
    )
    assert typed_hint.status_code == 200
    view = typed_hint.json()
    assert view["pending"]["key"] == created.json()["pending"]["key"]
    handle = client.app.state.v2_store.get(session_id)
    assert len(handle.orchestrator.learner.events) == events_before

    first_answer = handle.orchestrator.pending_expected
    first = client.post(
        f"/api/v2/sessions/{session_id}/actions",
        json=_answer(view, first_answer),
    )
    assert first.status_code == 200
    view = first.json()
    handle = client.app.state.v2_store.get(session_id)
    assert view["phase"] == "diagnose"
    assert handle.orchestrator.pending.family_id != first_family
    assert handle.orchestrator.pending_expected != first_answer

    confirmed = client.post(
        f"/api/v2/sessions/{session_id}/actions",
        json=_answer(view, handle.orchestrator.pending_expected),
    )
    assert confirmed.status_code == 200
    view = confirmed.json()
    assert view["phase"] == "capstone"
    handle = client.app.state.v2_store.get(session_id)

    completed = client.post(
        f"/api/v2/sessions/{session_id}/actions",
        json=_answer(view, handle.orchestrator.pending_expected),
    )
    assert completed.status_code == 200
    assert completed.json()["phase"] == "done"
    _assert_no_expected_key(completed.json())


def test_failed_capstone_moves_confirmed_strength_back_to_uncertain(client):
    created, _ = _create(client)
    view = created.json()
    session_id = view["session_id"]

    for _ in range(2):
        handle = client.app.state.v2_store.get(session_id)
        response = client.post(
            f"/api/v2/sessions/{session_id}/actions",
            json=_answer(view, handle.orchestrator.pending_expected),
        )
        assert response.status_code == 200
        view = response.json()

    assert view["phase"] == "capstone"
    assert "Power rule" in view["learner_summary"]["confirmed_strengths"]

    failed = client.post(
        f"/api/v2/sessions/{session_id}/actions",
        json=_answer(view, "0"),
    )

    assert failed.status_code == 200
    summary = failed.json()["learner_summary"]
    assert "Power rule" not in summary["confirmed_strengths"]
    assert "Power rule" not in summary["confirmed_gaps"]
    assert "Power rule" in summary["uncertain_skills"]


def test_action_idempotency_and_stale_snapshot_conflicts(client):
    created, _ = _create(client)
    before = created.json()
    request_id = uuid4()
    action = _hint(before, request_id=request_id)
    first = client.post(
        f"/api/v2/sessions/{before['session_id']}/actions", json=action
    )
    repeated = client.post(
        f"/api/v2/sessions/{before['session_id']}/actions", json=action
    )
    assert first.status_code == repeated.status_code == 200
    assert repeated.json() == first.json()

    reused = client.post(
        f"/api/v2/sessions/{before['session_id']}/actions",
        json={**action, "pending_key": "some-other-key"},
    )
    assert reused.status_code == 409
    assert reused.json()["code"] == "idempotency_conflict"
    assert reused.json()["session"] == first.json()

    stale = client.post(
        f"/api/v2/sessions/{before['session_id']}/actions",
        json=_hint(before),
    )
    assert stale.status_code == 409
    assert stale.json()["code"] == "stale_interaction"
    assert stale.json()["session"] == first.json()


def test_operational_metrics_use_stable_ids_without_raw_student_text(client, caplog):
    created, _ = _create(client, context="private coaching context")
    view = created.json()
    caplog.set_level(logging.INFO, logger="tutor.api.v2.operations")
    private_answer = "student-private-answer"

    response = client.post(
        f"/api/v2/sessions/{view['session_id']}/actions",
        json=_answer(view, private_answer),
    )

    assert response.status_code == 200
    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert private_answer not in logs
    assert "private coaching context" not in logs
    assert "item.power" in logs
    metrics = client.app.state.v2_store.metrics_snapshot()
    assert metrics["counters"]["actions_committed"] >= 1
    assert metrics["resume_outcomes"]["eligible_attempts"] == 0
    assert metrics["rollout_gates"]["resume_success_rate"] is None
    assert metrics["rollout_gates"]["action_5xx_rate"] == 0
    assert metrics["rollout_gates"]["duplicate_advances_detected"] == 0
    assert metrics["rollout_gates"]["missing_evidence_detected"] == 0
    assert metrics["rollout_gates"]["commit_integrity_failures"] == 0
    assert any(
        key.startswith("item.power")
        for key in metrics["actions_by_item_id"]
    )


def test_commit_invariant_detects_a_lost_prior_advance(client):
    created, _ = _create(client)
    view = created.json()
    handle = client.app.state.v2_store.get(view["session_id"])
    corrupted_response = client.app.state.v2_store.view(handle).model_copy(
        update={"pending": None}
    )
    handle.receipts["synthetic-corruption"] = MutationReceipt(
        payload_hash="synthetic",
        response=corrupted_response,
        request_payload={
            "type": "answer",
            "pending_key": view["pending"]["key"],
        },
    )

    rejected = client.post(
        f"/api/v2/sessions/{view['session_id']}/actions",
        json=_answer(view, "not-scored"),
    )

    assert rejected.status_code == 500
    assert rejected.json()["code"] == "session_integrity_failure"
    assert client.app.state.v2_store.view(handle).revision == view["revision"]
    metrics = client.app.state.v2_store.metrics_snapshot()
    assert metrics["rollout_gates"]["duplicate_advances_detected"] == 1
    assert metrics["rollout_gates"]["commit_integrity_failures"] == 1
    assert metrics["counters"]["action_5xx"] == 1


def test_commit_invariant_detects_an_advance_without_evidence(client, monkeypatch):
    created, _ = _create(client)
    view = created.json()
    handle = client.app.state.v2_store.get(view["session_id"])

    def corrupt_submit(orchestrator, _answer_text):
        orchestrator._pending = None
        return []

    monkeypatch.setattr(type(handle.orchestrator), "submit", corrupt_submit)
    rejected = client.post(
        f"/api/v2/sessions/{view['session_id']}/actions",
        json=_answer(view, "synthetic"),
    )

    assert rejected.status_code == 500
    assert rejected.json()["code"] == "session_integrity_failure"
    assert handle.revision == view["revision"]
    metrics = client.app.state.v2_store.metrics_snapshot()
    assert metrics["rollout_gates"]["missing_evidence_detected"] == 1
    assert metrics["rollout_gates"]["commit_integrity_failures"] == 1


def test_action_5xx_metric_counts_unexpected_exceptions(monkeypatch):
    crashing = TestClient(_v2_app(), raise_server_exceptions=False)
    created, _ = _create(crashing)
    view = created.json()
    handle = crashing.app.state.v2_store.get(view["session_id"])

    def crash(_orchestrator, _answer_text):
        raise ValueError("synthetic private failure")

    monkeypatch.setattr(type(handle.orchestrator), "submit", crash)
    response = crashing.post(
        f"/api/v2/sessions/{view['session_id']}/actions",
        json=_answer(view, "private-input"),
    )

    assert response.status_code == 500
    metrics = crashing.app.state.v2_store.metrics_snapshot()
    assert metrics["counters"]["action_requests"] == 1
    assert metrics["counters"]["action_5xx"] == 1
    assert metrics["rollout_gates"]["action_5xx_rate"] == 1


def test_action_quota_bounds_storage_but_old_receipt_replay_remains_exact(client):
    client.app.state.v2_store._max_receipts = 1
    created, _ = _create(client)
    initial = created.json()
    first_action = _hint(initial)
    first = client.post(
        f"/api/v2/sessions/{initial['session_id']}/actions",
        json=first_action,
    )
    second = client.post(
        f"/api/v2/sessions/{initial['session_id']}/actions",
        json=_hint(first.json()),
    )
    assert second.status_code == 429
    assert second.json()["code"] == "session_action_limit"

    replay = client.post(
        f"/api/v2/sessions/{initial['session_id']}/actions",
        json=first_action,
    )
    assert replay.status_code == 200
    assert replay.json() == first.json()


def test_durable_invalid_answer_flood_has_a_hard_storage_ceiling():
    app = _v2_app(durable=True)
    app.state.v2_store._max_receipts = 3
    client = TestClient(app)
    created, _ = _create(client)
    initial = created.json()
    view = initial
    first_action = None
    first_response = None

    for attempt in range(3):
        action = _answer(view, "factorial(5)")
        response = client.post(
            f"/api/v2/sessions/{view['session_id']}/actions",
            json=action,
        )
        assert response.status_code == 200
        if attempt == 0:
            first_action = action
            first_response = response.json()
        view = response.json()
        assert view["pending"]["key"] == initial["pending"]["key"]

    transcript_size_at_limit = len(view["transcript"])
    rejected = client.post(
        f"/api/v2/sessions/{view['session_id']}/actions",
        json=_answer(view, "factorial(5)"),
    )
    assert rejected.status_code == 429
    assert rejected.json()["code"] == "session_action_limit"
    current = client.get("/api/v2/sessions/current").json()
    assert current["revision"] == 3
    assert len(current["transcript"]) == transcript_size_at_limit

    assert first_action is not None and first_response is not None
    replay = client.post(
        f"/api/v2/sessions/{view['session_id']}/actions",
        json=first_action,
    )
    assert replay.status_code == 200
    assert replay.json() == first_response
    with Session(app.state.persistence.engine) as session:
        rows = session.scalars(select(m.SessionMutationReceiptRow)).all()
        assert len(rows) == 4  # one creation receipt plus three bounded actions


def test_parallel_actions_advance_at_most_once(client):
    created, _ = _create(client)
    view = created.json()
    action = _hint(view)
    url = f"/api/v2/sessions/{view['session_id']}/actions"

    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(executor.map(lambda _: client.post(url, json=action), range(2)))

    assert [response.status_code for response in responses] == [200, 200]
    assert responses[0].json() == responses[1].json()
    current = client.get("/api/v2/sessions/current").json()
    assert current["revision"] == 1
    assert sum(entry["kind"] == "hint" for entry in current["transcript"]) == 1


def test_parallel_distinct_requests_reject_stale_loser(client):
    created, _ = _create(client)
    view = created.json()
    actions = [_hint(view), _hint(view)]
    url = f"/api/v2/sessions/{view['session_id']}/actions"

    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(executor.map(lambda body: client.post(url, json=body), actions))

    assert sorted(response.status_code for response in responses) == [200, 409]
    conflict = next(response for response in responses if response.status_code == 409)
    assert conflict.json()["code"] == "stale_interaction"
    assert client.get("/api/v2/sessions/current").json()["revision"] == 1


def test_authoritative_view_waits_for_the_session_commit_lock(client):
    created, _ = _create(client)
    handle = client.app.state.v2_store.get(created.json()["session_id"])

    with ThreadPoolExecutor(max_workers=1) as executor:
        with handle.lock:
            projected = executor.submit(client.app.state.v2_store.view, handle)
            assert not projected.done()
        snapshot = projected.result(timeout=1)

    assert snapshot == client.app.state.v2_store.view(handle)


def test_session_ids_are_bound_to_resume_cookie(client):
    created, _ = _create(client)
    response = client.get("/api/v2/sessions/not-the-owned-session")
    assert response.status_code == 404
    assert response.json()["code"] == "session_not_found"

    stranger = TestClient(client.app)
    response = stranger.get(
        f"/api/v2/sessions/{created.json()['session_id']}"
    )
    assert response.status_code == 401
    assert response.json()["code"] == "resume_token_required"


def test_origin_check_and_reset(client):
    # TestClient sends no Origin by default; a foreign browser Origin is rejected.
    response = client.post(
        "/api/v2/sessions",
        headers={"origin": "https://attacker.example"},
        json={"request_id": str(uuid4()), "goal_id": "goal.der.power_rule"},
    )
    assert response.status_code == 403
    assert response.json()["code"] == "origin_not_allowed"

    fetch_metadata_blocked = client.post(
        "/api/v2/sessions",
        headers={"sec-fetch-site": "cross-site"},
        json={"request_id": str(uuid4()), "goal_id": "goal.der.power_rule"},
    )
    assert fetch_metadata_blocked.status_code == 403
    assert fetch_metadata_blocked.json()["code"] == "origin_not_allowed"

    created, _ = _create(client)
    reset = client.post(
        "/api/v2/sessions/current/reset", json=_reset(created.json())
    )
    assert reset.status_code == 200
    assert reset.json()["reset"] is True
    replacement = reset.json()["session"]
    assert replacement["session_id"] != created.json()["session_id"]
    current = client.get("/api/v2/sessions/current")
    assert current.status_code == 200
    assert current.json() == replacement


def test_reset_carries_revealed_family_retirement_into_replacement_episode(client):
    created, _ = _create(client)
    view = created.json()
    handle = client.app.state.v2_store.get(view["session_id"])
    revealed_family = handle.orchestrator.pending.family_id
    for _ in range(3):
        hinted = client.post(
            f"/api/v2/sessions/{view['session_id']}/actions",
            json=_hint(view),
        )
        assert hinted.status_code == 200
        view = hinted.json()

    reset = client.post(
        "/api/v2/sessions/current/reset",
        json=_reset(view),
    )
    assert reset.status_code == 200
    replacement = reset.json()["session"]
    replacement_handle = client.app.state.v2_store.get(replacement["session_id"])
    assert revealed_family in replacement_handle.orchestrator.exposure_state.retired_family_ids
    assert replacement_handle.orchestrator.pending.family_id != revealed_family


def test_reset_is_revision_checked_idempotent_and_conflict_safe(client):
    created, _ = _create(client)
    view = created.json()
    raw_token = client.cookies.get("tutor_resume_v2")
    request_id = uuid4()

    stale = client.post(
        "/api/v2/sessions/current/reset",
        json=_reset(view, request_id=request_id, expected_revision=1),
    )
    assert stale.status_code == 409
    assert stale.json()["code"] == "stale_interaction"
    assert client.cookies.get("tutor_resume_v2") == raw_token

    payload = _reset(view, request_id=request_id)
    committed = client.post("/api/v2/sessions/current/reset", json=payload)
    assert committed.status_code == 200
    assert committed.json()["reset"] is True
    replacement_token = client.cookies.get("tutor_resume_v2")
    assert replacement_token != raw_token
    assert committed.json()["session"]["session_id"] != view["session_id"]

    _set_resume_cookie(client, raw_token)
    replayed = client.post("/api/v2/sessions/current/reset", json=payload)
    assert replayed.status_code == 200
    assert replayed.json() == committed.json()
    assert client.cookies.get("tutor_resume_v2") == replacement_token

    _set_resume_cookie(client, raw_token)
    conflict = client.post(
        "/api/v2/sessions/current/reset",
        json={**payload, "pending_key": "different-interaction"},
    )
    assert conflict.status_code == 409
    assert conflict.json()["code"] == "idempotency_conflict"


def test_reset_starts_fresh_episode_with_same_learner_and_prior_evidence(client):
    created, _ = _create(client)
    before = created.json()
    first_handle = client.app.state.v2_store.get(before["session_id"])
    answered = client.post(
        f"/api/v2/sessions/{before['session_id']}/actions",
        json=_answer(before, first_handle.orchestrator.pending_expected),
    )
    assert answered.status_code == 200
    old_handle = client.app.state.v2_store.get(before["session_id"])
    learner_id = old_handle.learner_id
    historical_events = tuple(old_handle.orchestrator.learner.events)
    assert historical_events

    reset = client.post(
        "/api/v2/sessions/current/reset", json=_reset(answered.json())
    )
    assert reset.status_code == 200
    replacement_view = reset.json()["session"]
    replacement = client.app.state.v2_store.get(replacement_view["session_id"])
    assert replacement.learner_id == learner_id
    assert replacement.orchestrator.learner.events == historical_events
    assert replacement.revision == 0
    assert replacement_view["phase"] == "diagnose"


def test_durable_reset_receipt_replays_after_process_memory_loss():
    app = _v2_app(durable=True)
    client = TestClient(app)
    created, _ = _create(client)
    view = created.json()
    raw_token = client.cookies.get("tutor_resume_v2")
    payload = _reset(view)

    committed = client.post("/api/v2/sessions/current/reset", json=payload)
    assert committed.status_code == 200
    replacement_raw_token = client.cookies.get("tutor_resume_v2")

    with app.state.v2_store._lock:
        app.state.v2_store._sessions.clear()
        app.state.v2_store._tokens.clear()
        app.state.v2_store._reset_receipts.clear()
    _set_resume_cookie(client, raw_token)
    replayed = client.post("/api/v2/sessions/current/reset", json=payload)
    assert replayed.status_code == 200
    assert replayed.json() == committed.json()
    assert client.get("/api/v2/sessions/current").json() == committed.json()["session"]

    with Session(app.state.persistence.engine) as session:
        receipts = session.scalars(select(m.SessionMutationReceiptRow)).all()
        reset_receipts = [
            receipt
            for receipt in receipts
            if receipt.request_payload.get("type") == "reset"
        ]
        assert len(reset_receipts) == 1
        tokens = session.scalars(
            select(m.ResumeTokenRow).order_by(m.ResumeTokenRow.id)
        ).all()
        assert len(tokens) == 2
        assert tokens[0].session_id == view["session_id"]
        assert tokens[0].revoked is True
        assert tokens[1].session_id == committed.json()["session"]["session_id"]
        assert tokens[1].revoked is False
        assert tokens[1].token_hash != replacement_raw_token


def test_reset_response_loss_requires_cookie_and_request_proof_on_reload():
    legacy = PersistenceService(
        engine=get_engine("sqlite+pysqlite:///:memory:")
    )
    persistence = V2PersistenceService(legacy.engine)
    secret = b"stable-test-resume-secret-material-32-bytes"
    app = _v2_app(persistence=persistence, resume_token_secret=secret)
    client = TestClient(app)
    created, _ = _create(client)
    old_cookie = client.cookies.get("tutor_resume_v2")

    request_id = uuid4()
    committed = client.post(
        "/api/v2/sessions/current/reset",
        json=_reset(created.json(), request_id=request_id),
    )

    assert committed.status_code == 200
    expected = committed.json()["session"]
    replacement_cookie = client.cookies.get("tutor_resume_v2")
    assert replacement_cookie != old_cookie

    reloaded_app = _v2_app(
        persistence=persistence,
        resume_token_secret=secret,
    )
    reloaded_client = TestClient(reloaded_app)
    _set_resume_cookie(reloaded_client, old_cookie)
    unauthorized = reloaded_client.get("/api/v2/sessions/current")
    wrong_proof = reloaded_client.post(
        "/api/v2/sessions/recover",
        json={
            "schema_version": 1,
            "operation": "reset",
            "request_id": str(uuid4()),
        },
    )
    recovered = reloaded_client.post(
        "/api/v2/sessions/recover",
        json={
            "schema_version": 1,
            "operation": "reset",
            "request_id": str(request_id),
        },
    )
    restored = reloaded_client.get("/api/v2/sessions/current")

    assert unauthorized.status_code == 401
    assert reloaded_client.cookies.get("tutor_resume_v2") == replacement_cookie
    assert wrong_proof.status_code == 409
    assert wrong_proof.json()["code"] == "recovery_not_committed"
    assert recovered.status_code == 200
    assert restored.status_code == 200
    assert restored.json() == expected
    assert (
        reloaded_app.state.v2_store.metrics_snapshot()["counters"]
        ["reset_responses_recovered"]
        == 1
    )


def test_persistence_is_atomic_and_tokens_are_hashed(monkeypatch):
    app = _v2_app(durable=True)
    client = TestClient(app)
    created, _ = _create(client)
    assert created.status_code == 200
    view = created.json()
    assert view["durability"] == "durable"
    raw_cookie = client.cookies.get("tutor_resume_v2")

    action = _hint(view)
    advanced = client.post(
        f"/api/v2/sessions/{view['session_id']}/actions", json=action
    )
    assert advanced.status_code == 200

    engine = app.state.persistence.engine
    with Session(engine) as session:
        checkpoint = session.scalars(select(m.SessionCheckpointRow)).one()
        receipts = session.scalars(select(m.SessionMutationReceiptRow)).all()
        transcript = session.scalars(select(m.TranscriptEntryRow)).all()
        token = session.scalars(select(m.ResumeTokenRow)).one()
        assert checkpoint.revision == 1
        assert checkpoint.checkpoint["session_view"]["revision"] == 1
        assert checkpoint.pedagogy_catalog_version == "test-approved-pedagogy-v1"
        assert checkpoint.checkpoint["content_release"] == {
            "graph_version": power_rule_only_graph().graph_version,
            "item_bank_version": "test-approved-power-stress-v2",
            "pedagogy_catalog_version": "test-approved-pedagogy-v1",
        }
        assert len(receipts) == 2  # create plus hint
        assert len(transcript) == len(advanced.json()["transcript"])
        assert token.token_hash != raw_cookie
        assert len(token.token_hash) == 64

    class _FailingPersistence:
        def commit_action(self, **kwargs):
            raise RuntimeError("database unavailable")

        def touch_resume(self, token_hash):
            return True

        def revoke_token(self, token_hash):
            raise RuntimeError("database unavailable")

    monkeypatch.setattr(app.state.v2_store, "_persistence", _FailingPersistence())
    before_failure = client.get("/api/v2/sessions/current").json()
    failure = client.post(
        f"/api/v2/sessions/{view['session_id']}/actions",
        json=_hint(before_failure),
    )
    assert failure.status_code == 503
    after_failure = client.get("/api/v2/sessions/current").json()
    assert after_failure == before_failure
    failed_reset = client.post(
        "/api/v2/sessions/current/reset", json=_reset(before_failure)
    )
    assert failed_reset.status_code == 503
    assert client.get("/api/v2/sessions/current").json() == before_failure


def test_v2_evidence_uses_the_same_episode_id_in_memory_checkpoint_and_database():
    app = _v2_app(durable=True)
    client = TestClient(app)
    created, _ = _create(client)
    view = created.json()
    handle = app.state.v2_store.get(view["session_id"])

    response = client.post(
        f"/api/v2/sessions/{view['session_id']}/actions",
        json=_answer(view, handle.orchestrator.pending_expected),
    )
    assert response.status_code == 200

    event = handle.orchestrator.learner.events[-1]
    checkpoint = handle.orchestrator.export_checkpoint()
    assert event.episode_id == view["session_id"]
    assert checkpoint["episode_id"] == view["session_id"]
    assert checkpoint["events"][-1]["episode_id"] == view["session_id"]
    assert event.pedagogy_catalog_version == "test-approved-pedagogy-v1"
    assert event.content_versions["pedagogy_catalog"] == event.pedagogy_catalog_version

    with Session(app.state.persistence.engine) as session:
        row = session.scalars(select(m.EvidenceEventRow)).one()
        assert row.episode_id == view["session_id"]
        assert row.pedagogy_catalog_version == event.pedagogy_catalog_version


def test_resume_expiry_rolls_on_get_new_action_and_receipt_replay(monkeypatch):
    clock = {"now": datetime(2026, 1, 1, 12, tzinfo=timezone.utc)}
    monkeypatch.setattr(
        "tutor.api.v2_store.utcnow", lambda: clock["now"]
    )
    monkeypatch.setattr(
        "tutor.api.v2_persistence._utcnow", lambda: clock["now"]
    )
    app = _v2_app(durable=True)
    client = TestClient(app)
    created, _ = _create(client)
    view = created.json()
    raw_cookie = client.cookies.get("tutor_resume_v2")

    def durable_expiry():
        with Session(app.state.persistence.engine) as session:
            value = session.scalars(select(m.ResumeTokenRow.expires_at)).one()
        return (
            value
            if value.tzinfo is not None
            else value.replace(tzinfo=timezone.utc)
        )

    def assert_cookie_refreshed(response):
        cookie = response.headers.get("set-cookie", "").lower()
        assert f"max-age={30 * 24 * 60 * 60}" in cookie
        assert "httponly" in cookie
        assert "samesite=lax" in cookie
        assert client.cookies.get("tutor_resume_v2") == raw_cookie

    assert durable_expiry() == clock["now"] + timedelta(days=30)

    clock["now"] += timedelta(days=10)
    current = client.get("/api/v2/sessions/current")
    assert current.status_code == 200
    assert_cookie_refreshed(current)
    assert durable_expiry() == clock["now"] + timedelta(days=30)
    [local_token] = app.state.v2_store._tokens.values()
    assert local_token[1] == clock["now"] + timedelta(days=30)

    action = _hint(view)
    clock["now"] += timedelta(days=5)
    advanced = client.post(
        f"/api/v2/sessions/{view['session_id']}/actions", json=action
    )
    assert advanced.status_code == 200
    assert_cookie_refreshed(advanced)
    assert durable_expiry() == clock["now"] + timedelta(days=30)
    [local_token] = app.state.v2_store._tokens.values()
    assert local_token[1] == clock["now"] + timedelta(days=30)

    clock["now"] += timedelta(days=5)
    replayed = client.post(
        f"/api/v2/sessions/{view['session_id']}/actions", json=action
    )
    assert replayed.status_code == 200
    assert replayed.json() == advanced.json()
    assert_cookie_refreshed(replayed)
    assert durable_expiry() == clock["now"] + timedelta(days=30)
    [local_token] = app.state.v2_store._tokens.values()
    assert local_token[1] == clock["now"] + timedelta(days=30)


def test_reset_keeps_cookie_when_durable_authorization_is_unavailable(monkeypatch):
    app = _v2_app(durable=True)
    client = TestClient(app)
    created, _ = _create(client)
    assert created.status_code == 200
    raw_cookie = client.cookies.get("tutor_resume_v2")

    def fail_resume(token_hash):
        raise RuntimeError("database read unavailable")

    monkeypatch.setattr(app.state.v2_persistence, "resolve_resume", fail_resume)
    response = client.post(
        "/api/v2/sessions/current/reset", json=_reset(created.json())
    )

    assert response.status_code == 503
    assert response.json()["code"] == "session_restore_unavailable"
    assert client.cookies.get("tutor_resume_v2") == raw_cookie
    assert "delete_cookie" not in response.headers.get("set-cookie", "")


def test_durable_checkpoint_restores_exact_session_after_memory_loss():
    app = _v2_app(durable=True)
    client = TestClient(app)
    created, _ = _create(client)
    view = created.json()
    advanced = client.post(
        f"/api/v2/sessions/{view['session_id']}/actions",
        json=_hint(view),
    )
    assert advanced.status_code == 200
    expected = advanced.json()

    # Simulate a process losing every live object while retaining its database.
    with app.state.v2_store._lock:
        app.state.v2_store._sessions.clear()
        app.state.v2_store._tokens.clear()

    restored = client.get("/api/v2/sessions/current")
    assert restored.status_code == 200
    assert restored.json() == expected
    assert (
        app.state.v2_store.get(expected["session_id"])
        .orchestrator.export_checkpoint()
        == app.state.v2_persistence.resolve_resume(
            next(iter(app.state.v2_store._tokens))
        )["checkpoint"]["orchestrator"]
    )

    continued = client.post(
        f"/api/v2/sessions/{expected['session_id']}/actions",
        json=_hint(expected),
    )
    assert continued.status_code == 200
    assert continued.json()["revision"] == 2


def test_durable_resume_rejects_visible_ledger_missing_student_answer():
    app = _v2_app(durable=True)
    client = TestClient(app)
    created, _ = _create(client)
    view = created.json()
    answered = client.post(
        f"/api/v2/sessions/{view['session_id']}/actions",
        json=_answer(view, "4*x^3"),
    )
    assert answered.status_code == 200

    with Session(app.state.persistence.engine) as session:
        row = session.get(m.SessionCheckpointRow, view["session_id"])
        payload = deepcopy(row.checkpoint)
        payload["orchestrator"]["visible_texts"].remove("4*x^3")
        row.checkpoint = payload
        session.commit()
    with app.state.v2_store._lock:
        app.state.v2_store._sessions.clear()
        app.state.v2_store._tokens.clear()

    restored = client.get("/api/v2/sessions/current")

    assert restored.status_code == 503
    assert restored.json()["code"] == "session_restore_unavailable"
    assert (
        app.state.v2_store.metrics_snapshot()["counters"]
        ["visible_content_integrity_failures"]
        == 1
    )


def test_durable_resume_fails_closed_when_evidence_ledger_is_missing():
    app = _v2_app(durable=True)
    client = TestClient(app)
    created, _ = _create(client)
    view = created.json()
    handle = app.state.v2_store.get(view["session_id"])
    answered = client.post(
        f"/api/v2/sessions/{view['session_id']}/actions",
        json=_answer(view, handle.orchestrator.pending_expected),
    )
    assert answered.status_code == 200

    with Session(app.state.persistence.engine) as session:
        [event] = session.scalars(
            select(m.EvidenceEventRow).where(
                m.EvidenceEventRow.episode_id == view["session_id"]
            )
        ).all()
        session.delete(event)
        session.commit()
    with app.state.v2_store._lock:
        app.state.v2_store._sessions.clear()
        app.state.v2_store._tokens.clear()

    restored = client.get("/api/v2/sessions/current")

    assert restored.status_code == 503
    assert restored.json()["code"] == "session_restore_unavailable"
    metrics = app.state.v2_store.metrics_snapshot()
    assert metrics["rollout_gates"]["missing_evidence_detected"] == 1
    assert metrics["rollout_gates"]["commit_integrity_failures"] == 1


def test_durable_resume_rejects_a_checkpoint_row_catalog_mismatch():
    app = _v2_app(durable=True)
    client = TestClient(app)
    created, _ = _create(client)
    view = created.json()

    with Session(app.state.persistence.engine) as session:
        row = session.get(m.SessionCheckpointRow, view["session_id"])
        row.pedagogy_catalog_version = "silently-rebound-catalog"
        session.commit()
    with app.state.v2_store._lock:
        app.state.v2_store._sessions.clear()
        app.state.v2_store._tokens.clear()

    restored = client.get("/api/v2/sessions/current")

    assert restored.status_code == 503
    assert restored.json()["code"] == "session_restore_unavailable"
    counters = app.state.v2_store.metrics_snapshot()["counters"]
    assert counters["checkpoint_integrity_failures"] == 1


def test_persistence_rejects_an_unrestorable_checkpoint_before_insert():
    app = _v2_app(durable=True)
    client = TestClient(app)
    created, _ = _create(client)
    handle = app.state.v2_store.get(created.json()["session_id"])
    original = handle.orchestrator
    view = app.state.v2_store.view(handle)

    class MissingCatalogPin:
        def export_checkpoint(self):
            payload = original.export_checkpoint()
            payload.pop("pedagogy_catalog_version")
            return payload

    handle.orchestrator = MissingCatalogPin()

    with pytest.raises(
        ValueError,
        match="trusted pedagogy-catalog version",
    ):
        app.state.v2_persistence._checkpoint(
            handle,
            view,
        )


def test_durable_resume_rejects_internally_inconsistent_evidence_catalog():
    app = _v2_app(durable=True)
    client = TestClient(app)
    created, _ = _create(client)
    view = created.json()
    handle = app.state.v2_store.get(view["session_id"])
    answered = client.post(
        f"/api/v2/sessions/{view['session_id']}/actions",
        json=_answer(view, handle.orchestrator.pending_expected),
    )
    assert answered.status_code == 200

    with Session(app.state.persistence.engine) as session:
        checkpoint = session.get(m.SessionCheckpointRow, view["session_id"])
        payload = deepcopy(checkpoint.checkpoint)
        payload["orchestrator"]["events"][-1][
            "pedagogy_catalog_version"
        ] = "tampered-catalog"
        checkpoint.checkpoint = payload
        event = session.scalars(
            select(m.EvidenceEventRow).where(
                m.EvidenceEventRow.episode_id == view["session_id"]
            )
        ).one()
        event.pedagogy_catalog_version = "tampered-catalog"
        session.commit()
    with app.state.v2_store._lock:
        app.state.v2_store._sessions.clear()
        app.state.v2_store._tokens.clear()

    restored = client.get("/api/v2/sessions/current")

    assert restored.status_code == 503
    assert restored.json()["code"] == "session_restore_unavailable"
    counters = app.state.v2_store.metrics_snapshot()["counters"]
    assert counters["evidence_provenance_integrity_failures"] == 1


def test_create_restores_durable_cookie_before_enforcing_one_active_episode():
    app = _v2_app(durable=True)
    client = TestClient(app)
    request_id = uuid4()
    created, payload = _create(client, request_id=request_id)
    expected = created.json()

    with app.state.v2_store._lock:
        app.state.v2_store._sessions.clear()
        app.state.v2_store._tokens.clear()

    repeated = client.post("/api/v2/sessions", json=payload)
    assert repeated.status_code == 200
    assert repeated.json() == expected

    another, _ = _create(client)
    assert another.status_code == 409
    assert another.json()["code"] == "active_session_exists"
    assert another.json()["session"]["session_id"] == expected["session_id"]

    with Session(app.state.persistence.engine) as session:
        checkpoints = session.scalars(select(m.SessionCheckpointRow)).all()
        assert len(checkpoints) == 1


def test_reset_wins_against_an_action_waiting_to_commit_in_another_process():
    legacy = PersistenceService(
        engine=get_engine("sqlite+pysqlite:///:memory:")
    )
    durable = V2PersistenceService(legacy.engine)
    commit_entered = threading.Event()
    release_commit = threading.Event()

    class _BlockingPersistence:
        @property
        def engine(self):
            return durable.engine

        def create_session(self, *args, **kwargs):
            return durable.create_session(*args, **kwargs)

        def resolve_resume(self, *args, **kwargs):
            return durable.resolve_resume(*args, **kwargs)

        def resume_token_status(self, *args, **kwargs):
            return durable.resume_token_status(*args, **kwargs)

        def touch_resume(self, *args, **kwargs):
            return durable.touch_resume(*args, **kwargs)

        def replay_create(self, **kwargs):
            return durable.replay_create(**kwargs)

        def revoke_token(self, *args, **kwargs):
            return durable.revoke_token(*args, **kwargs)

        def replay_reset(self, **kwargs):
            return durable.replay_reset(**kwargs)

        def commit_reset(self, **kwargs):
            return durable.commit_reset(**kwargs)

        def commit_action(self, **kwargs):
            commit_entered.set()
            assert release_commit.wait(timeout=5)
            return durable.commit_action(**kwargs)

    persistence = _BlockingPersistence()
    app_a = _v2_app(persistence=persistence)
    app_b = _v2_app(persistence=persistence)
    client_a = TestClient(app_a)
    client_b = TestClient(app_b)
    created, _ = _create(client_a)
    view = created.json()
    _set_resume_cookie(
        client_b, client_a.cookies.get("tutor_resume_v2")
    )
    assert client_b.get("/api/v2/sessions/current").status_code == 200

    with ThreadPoolExecutor(max_workers=1) as executor:
        action = executor.submit(
            client_a.post,
            f"/api/v2/sessions/{view['session_id']}/actions",
            json=_hint(view),
        )
        assert commit_entered.wait(timeout=5)
        reset = client_b.post(
            "/api/v2/sessions/current/reset", json=_reset(view)
        )
        release_commit.set()
        rejected = action.result()

    assert reset.status_code == 200
    assert rejected.status_code == 409
    assert rejected.json()["code"] == "session_revoked"
    with Session(legacy.engine) as session:
        checkpoint = session.get(m.SessionCheckpointRow, view["session_id"])
        assert checkpoint is not None
        assert checkpoint.revision == 0
        assert len(session.scalars(select(m.SessionCheckpointRow)).all()) == 2


def test_two_process_caches_converge_on_receipts_revisions_and_revocation():
    legacy = PersistenceService(
        engine=get_engine("sqlite+pysqlite:///:memory:")
    )
    persistence = V2PersistenceService(legacy.engine)
    app_a = _v2_app(persistence=persistence)
    app_b = _v2_app(persistence=persistence)
    client_a = TestClient(app_a)
    client_b = TestClient(app_b)

    created, _ = _create(client_a)
    before = created.json()
    raw_token = client_a.cookies.get("tutor_resume_v2")
    _set_resume_cookie(client_b, raw_token)
    assert client_b.get("/api/v2/sessions/current").json() == before

    action = _hint(before)
    committed = client_a.post(
        f"/api/v2/sessions/{before['session_id']}/actions", json=action
    )
    assert committed.status_code == 200
    replayed = client_b.post(
        f"/api/v2/sessions/{before['session_id']}/actions", json=action
    )
    assert replayed.status_code == 200
    assert replayed.json() == committed.json()
    assert client_b.get("/api/v2/sessions/current").json() == committed.json()

    reset = client_a.post(
        "/api/v2/sessions/current/reset", json=_reset(committed.json())
    )
    assert reset.status_code == 200
    revoked = client_b.get("/api/v2/sessions/current")
    assert revoked.status_code == 401
    assert revoked.json()["code"] == "invalid_resume_token"
