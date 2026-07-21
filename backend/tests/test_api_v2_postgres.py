"""PostgreSQL contention and transaction-recovery gates for session API v2.

These tests are intentionally opt-in because they require a disposable
PostgreSQL database. Set ``TUTOR_TEST_POSTGRES_URL`` to run them. Each test
creates and drops a private schema, so parallel test processes do not share
session state.
"""

from __future__ import annotations

import os
import threading
from multiprocessing import get_context
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine, event, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from tutor.api.v2 import install_v2_routes
from tutor.api.v2_persistence import V2PersistenceService
from tutor.db import models as m
from tutor.db.session import create_all

from tests.v2_helpers import (
    approved_power_rule_bank,
    approved_power_rule_catalog,
    power_rule_only_graph,
)

_POSTGRES_ENV = "TUTOR_TEST_POSTGRES_URL"
_RESUME_SECRET = b"postgres-integration-resume-secret-32-bytes"


@pytest.fixture()
def postgres_engine() -> Iterator[Engine]:
    """Yield an engine isolated to a temporary schema in the configured database."""
    raw_url = os.environ.get(_POSTGRES_ENV)
    if not raw_url:
        pytest.skip(f"set {_POSTGRES_ENV} to run PostgreSQL integration gates")

    url = make_url(raw_url)
    if url.get_backend_name() != "postgresql":
        pytest.fail(f"{_POSTGRES_ENV} must use PostgreSQL, got {url.get_backend_name()!r}")

    schema = f"tutor_v2_test_{uuid4().hex}"
    admin_engine = create_engine(url, pool_pre_ping=True)
    test_engine: Engine | None = None
    try:
        with admin_engine.begin() as connection:
            connection.execute(text(f'CREATE SCHEMA "{schema}"'))

        test_engine = create_engine(
            url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=5,
        )

        @event.listens_for(test_engine, "connect")
        def _set_search_path(dbapi_connection: Any, _: Any) -> None:
            previous_autocommit = dbapi_connection.autocommit
            dbapi_connection.autocommit = True
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute(f'SET SESSION search_path TO "{schema}"')
                cursor.execute("SET SESSION lock_timeout TO '10s'")
                cursor.execute("SET SESSION statement_timeout TO '30s'")
            finally:
                cursor.close()
                dbapi_connection.autocommit = previous_autocommit

        create_all(test_engine)
        yield test_engine
    finally:
        if test_engine is not None:
            test_engine.dispose()
        try:
            with admin_engine.begin() as connection:
                connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        finally:
            admin_engine.dispose()


def _app(
    engine: Engine,
    *,
    max_episodes_per_learner: int = 32,
) -> FastAPI:
    app = FastAPI()
    persistence = V2PersistenceService(
        engine,
        max_episodes_per_learner=max_episodes_per_learner,
    )
    app.state.persistence = persistence
    install_v2_routes(
        app,
        power_rule_only_graph(),
        persistence=persistence,
        available_targets=("kc.der.power_rule",),
        item_bank=approved_power_rule_bank(),
        pedagogy_catalog=approved_power_rule_catalog(),
        resume_token_secret=_RESUME_SECRET,
    )
    return app


def _kill_after_receipt_insert(
    database_url: str,
    schema: str,
    raw_token: str,
    action_url: str,
    action: dict[str, Any],
) -> None:
    """Child-process target that dies after the final write but before commit."""
    engine = create_engine(database_url, pool_pre_ping=True)

    @event.listens_for(engine, "connect")
    def _set_child_search_path(dbapi_connection: Any, _: Any) -> None:
        previous_autocommit = dbapi_connection.autocommit
        dbapi_connection.autocommit = True
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute(f'SET SESSION search_path TO "{schema}"')
            cursor.execute("SET SESSION lock_timeout TO '10s'")
            cursor.execute("SET SESSION statement_timeout TO '30s'")
        finally:
            cursor.close()
            dbapi_connection.autocommit = previous_autocommit

    @event.listens_for(engine, "after_cursor_execute")
    def _terminate_on_receipt(
        _connection: Any,
        _cursor: Any,
        statement: str,
        _parameters: Any,
        _context: Any,
        _executemany: bool,
    ) -> None:
        normalized = " ".join(statement.lower().split())
        if normalized.startswith("insert into session_mutation_receipts"):
            os._exit(77)

    app = _app(engine)
    with TestClient(app) as client:
        _set_resume_cookie(client, raw_token)
        client.post(action_url, json=action)
    os._exit(78)


def _set_resume_cookie(client: TestClient, raw_token: str) -> None:
    client.cookies.set(
        "tutor_resume_v2",
        raw_token,
        domain="testserver.local",
        path="/api/v2",
    )


def _answer(view: dict[str, Any], answer: str, request_id: UUID) -> dict[str, Any]:
    return {
        "type": "answer",
        "request_id": str(request_id),
        "expected_revision": view["revision"],
        "pending_key": view["pending"]["key"],
        "answer": answer,
    }


@contextmanager
def _two_process_session(
    engine: Engine,
) -> Iterator[tuple[FastAPI, TestClient, FastAPI, TestClient, dict[str, Any], str]]:
    """Create one durable episode and restore it into an independent app store."""
    app_a = _app(engine)
    app_b = _app(engine)
    with TestClient(app_a) as client_a, TestClient(app_b) as client_b:
        created = client_a.post(
            "/api/v2/sessions",
            json={
                "request_id": str(uuid4()),
                "goal_id": "goal.der.power_rule",
            },
        )
        assert created.status_code == 200
        view = created.json()
        raw_token = client_a.cookies.get("tutor_resume_v2")
        assert raw_token

        _set_resume_cookie(client_b, raw_token)
        restored = client_b.get("/api/v2/sessions/current")
        assert restored.status_code == 200
        assert restored.json() == view
        yield app_a, client_a, app_b, client_b, view, raw_token


def _durable_state(engine: Engine, session_id: str) -> dict[str, Any]:
    with Session(engine) as session:
        checkpoints = session.scalars(
            select(m.SessionCheckpointRow).where(
                m.SessionCheckpointRow.session_id == session_id
            )
        ).all()
        assert len(checkpoints) == 1
        checkpoint = checkpoints[0]
        receipts = session.scalars(
            select(m.SessionMutationReceiptRow)
            .where(m.SessionMutationReceiptRow.session_id == session_id)
            .order_by(m.SessionMutationReceiptRow.id)
        ).all()
        transcript = session.scalars(
            select(m.TranscriptEntryRow)
            .where(m.TranscriptEntryRow.session_id == session_id)
            .order_by(m.TranscriptEntryRow.sequence)
        ).all()
        evidence = session.scalars(
            select(m.EvidenceEventRow)
            .where(m.EvidenceEventRow.episode_id == session_id)
            .order_by(m.EvidenceEventRow.id)
        ).all()
        return {
            "checkpoint_count": len(checkpoints),
            "revision": checkpoint.revision,
            "checkpoint_view": checkpoint.checkpoint["session_view"],
            "receipt_request_ids": tuple(row.request_id for row in receipts),
            "transcript_sequences": tuple(row.sequence for row in transcript),
            "transcript_entries": tuple(row.entry for row in transcript),
            "evidence_event_ids": tuple(row.event_id for row in evidence),
        }


def _assert_initial_state(
    state: dict[str, Any],
    view: dict[str, Any],
) -> None:
    assert state["checkpoint_count"] == 1
    assert state["revision"] == 0
    assert state["checkpoint_view"] == view
    assert len(state["receipt_request_ids"]) == 1
    assert state["transcript_sequences"] == tuple(range(len(view["transcript"])))
    assert state["transcript_entries"] == tuple(view["transcript"])
    assert state["evidence_event_ids"] == ()


def _assert_one_answer_committed(
    state: dict[str, Any],
    *,
    view: dict[str, Any],
    request_id: UUID,
) -> None:
    assert state["checkpoint_count"] == 1
    assert state["revision"] == 1
    assert state["checkpoint_view"] == view
    assert len(state["receipt_request_ids"]) == 2
    assert state["receipt_request_ids"].count(str(request_id)) == 1
    assert state["transcript_sequences"] == tuple(range(len(view["transcript"])))
    assert state["transcript_entries"] == tuple(view["transcript"])
    assert len(state["evidence_event_ids"]) == 1


def _post_concurrently(
    requests: tuple[tuple[TestClient, str, dict[str, Any]], ...],
) -> list[Any]:
    barrier = threading.Barrier(len(requests))

    def _post(args: tuple[TestClient, str, dict[str, Any]]) -> Any:
        client, url, payload = args
        barrier.wait(timeout=10)
        return client.post(url, json=payload)

    with ThreadPoolExecutor(max_workers=len(requests)) as executor:
        return list(executor.map(_post, requests))


def test_postgres_duplicate_create_commits_exactly_one_episode(
    postgres_engine: Engine,
) -> None:
    app_a = _app(postgres_engine)
    app_b = _app(postgres_engine)
    payload = {
        "request_id": str(uuid4()),
        "goal_id": "goal.der.power_rule",
    }
    with TestClient(app_a) as client_a, TestClient(app_b) as client_b:
        responses = _post_concurrently(
            (
                (client_a, "/api/v2/sessions", payload),
                (client_b, "/api/v2/sessions", payload),
            )
        )

    assert [response.status_code for response in responses] == [200, 200]
    assert responses[0].json() == responses[1].json()
    with Session(postgres_engine) as session:
        assert len(session.scalars(select(m.SessionCheckpointRow)).all()) == 1
        assert len(session.scalars(select(m.SessionMutationReceiptRow)).all()) == 1
        assert len(session.scalars(select(m.ResumeTokenRow)).all()) == 1


def test_postgres_duplicate_action_commits_exactly_once(
    postgres_engine: Engine,
) -> None:
    with _two_process_session(postgres_engine) as (
        app_a,
        client_a,
        _app_b,
        client_b,
        initial,
        _raw_token,
    ):
        session_id = initial["session_id"]
        baseline = _durable_state(postgres_engine, session_id)
        _assert_initial_state(baseline, initial)

        expected = app_a.state.v2_store.get(
            session_id
        ).orchestrator.pending_expected
        request_id = uuid4()
        action = _answer(initial, expected, request_id)
        url = f"/api/v2/sessions/{session_id}/actions"
        responses = _post_concurrently(
            (
                (client_a, url, action),
                (client_b, url, action),
            )
        )

        assert [response.status_code for response in responses] == [200, 200]
        assert responses[0].json() == responses[1].json()
        committed = responses[0].json()
        _assert_one_answer_committed(
            _durable_state(postgres_engine, session_id),
            view=committed,
            request_id=request_id,
        )


def test_postgres_distinct_actions_serialize_and_stale_requests_do_not_mutate(
    postgres_engine: Engine,
) -> None:
    with _two_process_session(postgres_engine) as (
        app_a,
        client_a,
        _app_b,
        client_b,
        initial,
        _raw_token,
    ):
        session_id = initial["session_id"]
        expected = app_a.state.v2_store.get(
            session_id
        ).orchestrator.pending_expected
        request_ids = (uuid4(), uuid4())
        actions = tuple(
            _answer(initial, expected, request_id) for request_id in request_ids
        )
        url = f"/api/v2/sessions/{session_id}/actions"
        responses = _post_concurrently(
            (
                (client_a, url, actions[0]),
                (client_b, url, actions[1]),
            )
        )

        assert sorted(response.status_code for response in responses) == [200, 409]
        winner = next(response for response in responses if response.status_code == 200)
        loser = next(response for response in responses if response.status_code == 409)
        assert loser.json()["code"] == "stale_interaction"
        assert loser.json()["session"] == winner.json()

        committed_request = request_ids[responses.index(winner)]
        state_after_race = _durable_state(postgres_engine, session_id)
        _assert_one_answer_committed(
            state_after_race,
            view=winner.json(),
            request_id=committed_request,
        )
        uncommitted_request = request_ids[responses.index(loser)]
        assert str(uncommitted_request) not in state_after_race["receipt_request_ids"]

        stale = client_b.post(
            url,
            json=_answer(initial, expected, uuid4()),
        )
        assert stale.status_code == 409
        assert stale.json()["code"] == "stale_interaction"
        assert stale.json()["session"] == winner.json()
        assert _durable_state(postgres_engine, session_id) == state_after_race


def test_postgres_late_transaction_failure_rolls_back_and_replays_exactly(
    postgres_engine: Engine,
) -> None:
    app = _app(postgres_engine)
    with TestClient(app) as client:
        created = client.post(
            "/api/v2/sessions",
            json={
                "request_id": str(uuid4()),
                "goal_id": "goal.der.power_rule",
            },
        )
        assert created.status_code == 200
        initial = created.json()
        session_id = initial["session_id"]
        raw_token = client.cookies.get("tutor_resume_v2")
        assert raw_token
        baseline = _durable_state(postgres_engine, session_id)
        _assert_initial_state(baseline, initial)

        request_id = uuid4()
        expected = app.state.v2_store.get(
            session_id
        ).orchestrator.pending_expected
        action = _answer(initial, expected, request_id)
        url = f"/api/v2/sessions/{session_id}/actions"

        with postgres_engine.begin() as connection:
            connection.execute(
                text(
                    """
                    CREATE FUNCTION tutor_test_reject_receipt()
                    RETURNS trigger
                    LANGUAGE plpgsql
                    AS $$
                    BEGIN
                        RAISE EXCEPTION 'injected receipt failure';
                    END;
                    $$
                    """
                )
            )
            connection.execute(
                text(
                    """
                    CREATE TRIGGER tutor_test_reject_receipt
                    BEFORE INSERT ON session_mutation_receipts
                    FOR EACH ROW EXECUTE FUNCTION tutor_test_reject_receipt()
                    """
                )
            )

        failed = client.post(url, json=action)
        assert failed.status_code == 503
        assert failed.json()["code"] == "persistence_unavailable"
        assert _durable_state(postgres_engine, session_id) == baseline
        assert client.get("/api/v2/sessions/current").json() == initial

        with postgres_engine.begin() as connection:
            connection.execute(
                text(
                    """
                    DROP TRIGGER tutor_test_reject_receipt
                    ON session_mutation_receipts
                    """
                )
            )
            connection.execute(
                text("DROP FUNCTION tutor_test_reject_receipt()")
            )

        committed = client.post(url, json=action)
        assert committed.status_code == 200
        _assert_one_answer_committed(
            _durable_state(postgres_engine, session_id),
            view=committed.json(),
            request_id=request_id,
        )

    recovered_app = _app(postgres_engine)
    with TestClient(recovered_app) as recovered_client:
        _set_resume_cookie(recovered_client, raw_token)
        recovered = recovered_client.get("/api/v2/sessions/current")
        assert recovered.status_code == 200
        assert recovered.json() == committed.json()

        replayed = recovered_client.post(url, json=action)
        assert replayed.status_code == 200
        assert replayed.json() == committed.json()
        _assert_one_answer_committed(
            _durable_state(postgres_engine, session_id),
            view=committed.json(),
            request_id=request_id,
        )


def test_postgres_process_kill_before_commit_rolls_back_then_replays(
    postgres_engine: Engine,
) -> None:
    app = _app(postgres_engine)
    with TestClient(app) as client:
        created = client.post(
            "/api/v2/sessions",
            json={
                "request_id": str(uuid4()),
                "goal_id": "goal.der.power_rule",
            },
        )
        assert created.status_code == 200
        initial = created.json()
        session_id = initial["session_id"]
        raw_token = client.cookies.get("tutor_resume_v2")
        assert raw_token
        baseline = _durable_state(postgres_engine, session_id)

        request_id = uuid4()
        expected = app.state.v2_store.get(
            session_id
        ).orchestrator.pending_expected
        action = _answer(initial, expected, request_id)
        action_url = f"/api/v2/sessions/{session_id}/actions"
        with postgres_engine.connect() as connection:
            schema = connection.scalar(text("SELECT current_schema()"))
        assert isinstance(schema, str)
        database_url = postgres_engine.url.render_as_string(
            hide_password=False
        )

        process = get_context("spawn").Process(
            target=_kill_after_receipt_insert,
            args=(database_url, schema, raw_token, action_url, action),
        )
        process.start()
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
            pytest.fail("child process did not reach the receipt boundary")
        assert process.exitcode == 77
        assert _durable_state(postgres_engine, session_id) == baseline
        assert client.get("/api/v2/sessions/current").json() == initial

        committed = client.post(action_url, json=action)
        assert committed.status_code == 200
        _assert_one_answer_committed(
            _durable_state(postgres_engine, session_id),
            view=committed.json(),
            request_id=request_id,
        )


    recovered_app = _app(postgres_engine)
    with TestClient(recovered_app) as recovered_client:
        _set_resume_cookie(recovered_client, raw_token)
        assert recovered_client.get(
            "/api/v2/sessions/current"
        ).json() == committed.json()
        replayed = recovered_client.post(action_url, json=action)
        assert replayed.status_code == 200
        assert replayed.json() == committed.json()
        _assert_one_answer_committed(
            _durable_state(postgres_engine, session_id),
            view=committed.json(),
            request_id=request_id,
        )


def test_postgres_reset_quota_survives_process_loss_and_preserves_exact_replay(
    postgres_engine: Engine,
) -> None:
    """The durable learner-wide count, not a process cache, owns admission."""
    app_a = _app(postgres_engine, max_episodes_per_learner=2)
    with TestClient(app_a) as client_a:
        initial_response = client_a.post(
            "/api/v2/sessions",
            json={
                "request_id": str(uuid4()),
                "goal_id": "goal.der.power_rule",
            },
        )
        assert initial_response.status_code == 200
        initial = initial_response.json()
        original_token = client_a.cookies.get("tutor_resume_v2")
        assert original_token
        reset_payload = {
            "request_id": str(uuid4()),
            "expected_revision": initial["revision"],
            "pending_key": initial["pending"]["key"],
        }
        committed = client_a.post(
            "/api/v2/sessions/current/reset",
            json=reset_payload,
        )
        assert committed.status_code == 200
        committed_payload = committed.json()
        replacement_token = client_a.cookies.get("tutor_resume_v2")
        assert replacement_token

    # A brand-new app has no in-memory reset history.  The already committed
    # reset must still replay exactly before quota evaluation, while a new
    # reset is rejected by the serialized durable count as a 429 (not a 503).
    app_b = _app(postgres_engine, max_episodes_per_learner=2)
    with TestClient(app_b) as client_b:
        _set_resume_cookie(client_b, original_token)
        replayed = client_b.post(
            "/api/v2/sessions/current/reset",
            json=reset_payload,
        )
        assert replayed.status_code == 200
        assert replayed.json() == committed_payload
        assert client_b.cookies.get("tutor_resume_v2") == replacement_token

        replacement = committed_payload["session"]
        blocked = client_b.post(
            "/api/v2/sessions/current/reset",
            json={
                "request_id": str(uuid4()),
                "expected_revision": replacement["revision"],
                "pending_key": replacement["pending"]["key"],
            },
        )
        assert blocked.status_code == 429
        assert blocked.json()["code"] == "episode_limit"
        assert blocked.json()["session"] == replacement
        assert client_b.cookies.get("tutor_resume_v2") == replacement_token

    with Session(postgres_engine) as session:
        assert len(session.scalars(select(m.SessionCheckpointRow)).all()) == 2
        assert len(session.scalars(select(m.ResumeTokenRow)).all()) == 2
        reset_receipts = [
            row
            for row in session.scalars(select(m.SessionMutationReceiptRow)).all()
            if row.request_payload.get("type") == "reset"
        ]
        assert len(reset_receipts) == 1


def test_postgres_purge_preserves_a_token_refreshed_while_waiting_for_its_lock(
    postgres_engine: Engine,
) -> None:
    app = _app(postgres_engine)
    with TestClient(app) as client:
        created = client.post(
            "/api/v2/sessions",
            json={
                "request_id": str(uuid4()),
                "goal_id": "goal.der.power_rule",
            },
        )
        assert created.status_code == 200
        session_id = created.json()["session_id"]

    now = datetime.now(timezone.utc)
    with Session(postgres_engine) as session:
        token = session.scalars(select(m.ResumeTokenRow)).one()
        token.expires_at = now - timedelta(seconds=1)
        session.commit()

    reached_token_lock = threading.Event()

    def observe_purge_token_lock(
        _connection: Any,
        _cursor: Any,
        statement: str,
        _parameters: Any,
        _context: Any,
        _executemany: bool,
    ) -> None:
        normalized = " ".join(statement.lower().split())
        if "from resume_tokens" in normalized and "for update" in normalized:
            reached_token_lock.set()

    with Session(postgres_engine) as refreshing:
        token = refreshing.scalar(
            select(m.ResumeTokenRow).with_for_update()
        )
        assert token is not None
        token.expires_at = now + timedelta(days=30)
        refreshing.flush()

        event.listen(
            postgres_engine,
            "before_cursor_execute",
            observe_purge_token_lock,
        )
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                purge = executor.submit(
                    app.state.persistence.purge_expired_anonymous_sessions
                )
                assert reached_token_lock.wait(timeout=10)
                assert not purge.done()
                refreshing.commit()
                assert purge.result(timeout=10) == 0
        finally:
            event.remove(
                postgres_engine,
                "before_cursor_execute",
                observe_purge_token_lock,
            )

    with Session(postgres_engine) as session:
        assert session.get(m.SessionCheckpointRow, session_id) is not None
        token = session.scalars(select(m.ResumeTokenRow)).one()
        expires_at = token.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        assert expires_at > now
