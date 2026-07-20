"""Focused gates for anonymous episode quotas and expired-session retention."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from tutor.api.v2 import install_v2_routes
from tutor.api.v2_persistence import V2PersistenceService
from tutor.db import models as m
from tutor.db.persistence import PersistenceService
from tutor.db.session import get_engine

from tests.v2_helpers import approved_power_rule_bank, power_rule_only_graph

_RESUME_SECRET = b"episode-retention-test-secret-32-bytes"


def _app(persistence: V2PersistenceService | None = None) -> FastAPI:
    app = FastAPI()
    if persistence is not None:
        app.state.persistence = persistence
    install_v2_routes(
        app,
        power_rule_only_graph(),
        persistence=persistence,
        available_targets=("kc.der.power_rule",),
        item_bank=approved_power_rule_bank(),
        resume_token_secret=_RESUME_SECRET,
    )
    return app


def _set_resume_cookie(client: TestClient, raw_token: str) -> None:
    client.cookies.set(
        "tutor_resume_v2",
        raw_token,
        domain="testserver.local",
        path="/api/v2",
    )


def _create(client: TestClient) -> dict:
    response = client.post(
        "/api/v2/sessions",
        json={
            "request_id": str(uuid4()),
            "goal_id": "goal.der.power_rule",
        },
    )
    assert response.status_code == 200
    return response.json()


def _reset_payload(view: dict, *, request_id=None) -> dict:
    return {
        "request_id": str(request_id or uuid4()),
        "expected_revision": view["revision"],
        "pending_key": view["pending"]["key"] if view["pending"] else None,
    }


def test_memory_reset_loop_remains_bounded_after_old_handles_are_evicted() -> None:
    app = _app()
    app.state.v2_store._max_sessions = 1
    app.state.v2_store._max_episodes_per_learner = 3
    client = TestClient(app)
    view = _create(client)

    first = client.post(
        "/api/v2/sessions/current/reset",
        json=_reset_payload(view),
    )
    assert first.status_code == 200
    view = first.json()["session"]

    token_before_final_allowed_reset = client.cookies.get("tutor_resume_v2")
    final_allowed_payload = _reset_payload(view)
    final_allowed = client.post(
        "/api/v2/sessions/current/reset",
        json=final_allowed_payload,
    )
    assert final_allowed.status_code == 200
    final_allowed_response = final_allowed.json()
    replacement_token = client.cookies.get("tutor_resume_v2")

    # Only the newest handle is cached, but the current handle carries all
    # three starts in its anonymous reset chain.
    assert len(app.state.v2_store._sessions) == 1
    replacement = app.state.v2_store.get(
        final_allowed_response["session"]["session_id"]
    )
    assert len(replacement.episode_starts) == 3

    blocked = client.post(
        "/api/v2/sessions/current/reset",
        json=_reset_payload(final_allowed_response["session"]),
    )
    assert blocked.status_code == 429
    assert blocked.json()["code"] == "episode_limit"
    assert client.cookies.get("tutor_resume_v2") == replacement_token

    # A committed request is replayed before quota evaluation, even when its
    # response created the episode that filled the final slot.
    assert token_before_final_allowed_reset is not None
    _set_resume_cookie(client, token_before_final_allowed_reset)
    replayed = client.post(
        "/api/v2/sessions/current/reset",
        json=final_allowed_payload,
    )
    assert replayed.status_code == 200
    assert replayed.json() == final_allowed_response
    assert client.cookies.get("tutor_resume_v2") == replacement_token


def test_durable_quota_is_enforced_after_restore_and_reset_replay_stays_exact() -> None:
    legacy = PersistenceService(
        engine=get_engine("sqlite+pysqlite:///:memory:")
    )
    persistence_a = V2PersistenceService(
        legacy.engine,
        max_episodes_per_learner=2,
    )
    app_a = _app(persistence_a)
    client_a = TestClient(app_a)
    initial = _create(client_a)
    original_token = client_a.cookies.get("tutor_resume_v2")
    assert original_token is not None
    reset_payload = _reset_payload(initial)

    committed = client_a.post(
        "/api/v2/sessions/current/reset",
        json=reset_payload,
    )
    assert committed.status_code == 200
    committed_response = committed.json()
    replacement_token = client_a.cookies.get("tutor_resume_v2")
    assert replacement_token is not None

    # Model a separate process: a fresh persistence facade and empty local
    # store must still use the durable learner-wide count.
    persistence_b = V2PersistenceService(
        legacy.engine,
        max_episodes_per_learner=2,
    )
    app_b = _app(persistence_b)
    client_b = TestClient(app_b)

    _set_resume_cookie(client_b, original_token)
    replayed = client_b.post(
        "/api/v2/sessions/current/reset",
        json=reset_payload,
    )
    assert replayed.status_code == 200
    assert replayed.json() == committed_response
    assert client_b.cookies.get("tutor_resume_v2") == replacement_token

    blocked = client_b.post(
        "/api/v2/sessions/current/reset",
        json=_reset_payload(committed_response["session"]),
    )
    assert blocked.status_code == 429
    assert blocked.json()["code"] == "episode_limit"
    assert blocked.json()["session"] == committed_response["session"]
    assert client_b.cookies.get("tutor_resume_v2") == replacement_token
    assert (
        app_b.state.v2_store.metrics_snapshot()["counters"][
            "episode_resets_rate_limited"
        ]
        == 1
    )

    # Terminal episode rollover uses the same durable gate and exception
    # contract.  Completing the current episode must not turn quota exhaustion
    # into a persistence 503 or revoke its still-valid token.
    completed = committed_response["session"]
    for _ in range(6):
        if completed["phase"] == "done":
            break
        current_handle = app_b.state.v2_store.get(completed["session_id"])
        advanced = client_b.post(
            f"/api/v2/sessions/{completed['session_id']}/actions",
            json={
                "type": "answer",
                "request_id": str(uuid4()),
                "expected_revision": completed["revision"],
                "pending_key": completed["pending"]["key"],
                "answer": current_handle.orchestrator.pending_expected,
            },
        )
        assert advanced.status_code == 200
        completed = advanced.json()
    assert completed["phase"] == "done"

    terminal_rollover = client_b.post(
        "/api/v2/sessions",
        json={
            "request_id": str(uuid4()),
            "goal_id": "goal.der.power_rule",
        },
    )
    assert terminal_rollover.status_code == 429
    assert terminal_rollover.json()["code"] == "episode_limit"
    assert client_b.cookies.get("tutor_resume_v2") == replacement_token

    with Session(legacy.engine) as session:
        assert len(session.scalars(select(m.SessionCheckpointRow)).all()) == 2
        assert len(session.scalars(select(m.ResumeTokenRow)).all()) == 2
        reset_receipts = [
            row
            for row in session.scalars(select(m.SessionMutationReceiptRow)).all()
            if row.request_payload.get("type") == "reset"
        ]
        assert len(reset_receipts) == 1


def test_expiry_purge_waits_for_every_token_and_never_deletes_evidence() -> None:
    legacy = PersistenceService(
        engine=get_engine("sqlite+pysqlite:///:memory:")
    )
    persistence = V2PersistenceService(legacy.engine)
    app = _app(persistence)
    client = TestClient(app)
    initial = _create(client)
    handle = app.state.v2_store.get(initial["session_id"])
    answered = client.post(
        f"/api/v2/sessions/{initial['session_id']}/actions",
        json={
            "type": "answer",
            "request_id": str(uuid4()),
            "expected_revision": initial["revision"],
            "pending_key": initial["pending"]["key"],
            "answer": handle.orchestrator.pending_expected,
        },
    )
    assert answered.status_code == 200

    now = datetime.now(timezone.utc)
    with Session(legacy.engine) as session:
        token = session.scalars(select(m.ResumeTokenRow)).one()
        token.expires_at = now - timedelta(seconds=1)
        session.add(
            m.ResumeTokenRow(
                learner_id=handle.learner_id,
                session_id=initial["session_id"],
                token_hash="b" * 64,
                expires_at=now + timedelta(days=1),
            )
        )
        session.add(
            m.WidgetAttemptRow(
                session_id=initial["session_id"],
                interaction_key="cleanup-test-widget",
                attempt_number=1,
                response={"text": "x"},
                correct=False,
                verification_status="incorrect",
                counted=True,
            )
        )
        session.commit()

    # One future token keeps the complete session resumable.
    assert persistence.purge_expired_anonymous_sessions() == 0
    with Session(legacy.engine) as session:
        assert session.get(m.SessionCheckpointRow, initial["session_id"]) is not None

        tokens = session.scalars(select(m.ResumeTokenRow)).all()
        for token in tokens:
            token.expires_at = now - timedelta(seconds=1)
        session.commit()

    assert persistence.purge_expired_anonymous_sessions() == 1
    with Session(legacy.engine) as session:
        assert session.scalars(select(m.SessionCheckpointRow)).all() == []
        assert session.scalars(select(m.SessionMutationReceiptRow)).all() == []
        assert session.scalars(select(m.TranscriptEntryRow)).all() == []
        assert session.scalars(select(m.ItemExposureRow)).all() == []
        assert session.scalars(select(m.WidgetAttemptRow)).all() == []
        assert session.scalars(select(m.ResumeTokenRow)).all() == []

        # Learner identity and its append-only evidence remain available for
        # longitudinal replay after anonymous resume material expires.
        evidence = session.scalars(select(m.EvidenceEventRow)).all()
        assert len(evidence) == 1
        assert evidence[0].episode_id == initial["session_id"]
        assert session.get(m.LearnerRow, handle.learner_id) is not None
