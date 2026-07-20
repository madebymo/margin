"""Persistence: evidence round-trip + replay, upserts, episodes, never-block."""

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from tutor.api.app import create_app
from tutor.db import models as m
from tutor.db.persistence import PersistenceService
from tutor.db.session import get_engine
from tutor.learner.service import LearnerModelService
from tutor.orchestrator.machine import SessionOrchestrator, SessionPhase
from tutor.schemas.common import ResponseClass
from tutor.schemas.learner import EvidenceEvent, LearnerProfile
from tutor.seed.load_seed import load_graph

FLOOR = {"Algebra 1", "Algebra 2", "Precalculus"}
PROFILE = LearnerProfile(course="AP Calculus AB", age_band="16-18")


@pytest.fixture(scope="module")
def graph():
    return load_graph()


@pytest.fixture()
def persistence():
    return PersistenceService(engine=get_engine("sqlite+pysqlite:///:memory:"))


def _event(service: LearnerModelService, kc: str, correct: bool) -> EvidenceEvent:
    return EvidenceEvent(
        event_id=uuid4(),
        learner_id=service.learner_id,
        t=datetime.now(timezone.utc),
        item_id="item",
        kc_ids=[kc],
        correct=correct,
        response_class=ResponseClass.SYMBOLIC_ENTRY,
    )


def test_evidence_roundtrip_and_replay_from_db(graph, persistence):
    service = LearnerModelService(graph, assumed_floor_levels=FLOOR)
    persistence.ensure_learner(service.learner_id, PROFILE)
    persistence.ensure_learner(service.learner_id, PROFILE)  # idempotent
    for kc, correct in [
        ("kc.der.chain_rule", False),
        ("kc.der.power_rule", True),
        ("kc.int.u_substitution", False),
    ]:
        event = _event(service, kc, correct)
        service.apply_event(event)
        persistence.record_event(event)

    loaded = persistence.load_events(service.learner_id)
    assert len(loaded) == 3
    assert [event.kc_ids[0] for event in loaded] == [
        "kc.der.chain_rule",
        "kc.der.power_rule",
        "kc.int.u_substitution",
    ]
    # the DB evidence log is authoritative: replaying it reproduces the model
    rebuilt = service.replay(loaded)
    assert rebuilt.snapshot().model_dump() == service.snapshot().model_dump()


def test_v2_evidence_provenance_roundtrip(graph, persistence):
    service = LearnerModelService(graph, assumed_floor_levels=FLOOR)
    persistence.ensure_learner(service.learner_id, PROFILE)
    event = EvidenceEvent(
        event_id=uuid4(),
        learner_id=service.learner_id,
        t=datetime.now(timezone.utc),
        item_id="item.power.checkin.square",
        kc_ids=["kc.der.power_rule"],
        correct=True,
        response_class=ResponseClass.SYMBOLIC_ENTRY,
        episode_id="episode-v2",
        family_id="family.power.checkin.square",
        surface="checkin",
        item_revision=2,
        attempt_number=3,
        policy_version="diagnosis-v2.0",
        learner_params_version="v2",
        content_provenance="reviewed-item-bank",
        learning_opportunity=True,
    )
    persistence.record_event(event)
    assert persistence.load_events(service.learner_id) == [event]


def test_save_derived_upserts_without_duplication(graph, persistence):
    service = LearnerModelService(graph, assumed_floor_levels=FLOOR)
    persistence.ensure_learner(service.learner_id, PROFILE)
    service.apply_event(_event(service, "kc.der.power_rule", True))
    persistence.save_derived(service.snapshot())
    service.apply_event(_event(service, "kc.der.power_rule", True))
    persistence.save_derived(service.snapshot())  # update, not duplicate

    with Session(persistence.engine) as session:
        rows = session.scalars(select(m.DerivedMasteryRow)).all()
        assert len(rows) == 40  # one per KC, stable across saves
        power_rule = next(r for r in rows if r.kc_id == "kc.der.power_rule")
        assert power_rule.observations == 2


def test_episode_lifecycle(persistence):
    learner_id = uuid4()
    persistence.ensure_learner(learner_id, PROFILE)
    episode_id = persistence.start_episode(
        learner_id, "kc.int.u_substitution", {"interaction_budget": 40}
    )
    persistence.update_episode(episode_id, "done", {"interactions_used": 3})
    with Session(persistence.engine) as session:
        row = session.get(m.EpisodeRow, episode_id)
        assert row.state == "done"
        assert row.envelope == {"interactions_used": 3}
    with pytest.raises(KeyError):
        persistence.update_episode(999_999, "done", {})


def test_machine_persists_full_session(graph):
    engine = get_engine("sqlite+pysqlite:///:memory:")
    service = PersistenceService(engine=engine)
    orchestrator = SessionOrchestrator(
        graph, "kc.int.u_substitution", PROFILE, persistence=service
    )
    orchestrator.begin()
    guard = 0
    while orchestrator.phase not in (SessionPhase.DONE, SessionPhase.STOPPED):
        guard += 1
        assert guard < 100
        orchestrator.submit(orchestrator.pending_expected)
    assert orchestrator.phase == SessionPhase.DONE

    with Session(engine) as session:
        learner_row = session.scalars(select(m.LearnerRow)).one()
        assert learner_row.learner_id == str(orchestrator.learner.learner_id)
        events = session.scalars(select(m.EvidenceEventRow)).all()
        assert len(events) == len(orchestrator.learner.events)
        episode = session.scalars(select(m.EpisodeRow)).one()
        assert episode.state == "done"
        assert episode.target_kc == "kc.int.u_substitution"
        derived = session.scalars(select(m.DerivedMasteryRow)).all()
        assert len(derived) == 40


class _ExplodingPersistence:
    """Simulates a dead database: every call raises."""

    def ensure_learner(self, *args, **kwargs):
        raise RuntimeError("db down")

    def start_episode(self, *args, **kwargs):
        raise RuntimeError("db down")

    def record_event(self, *args, **kwargs):
        raise RuntimeError("db down")

    def update_episode(self, *args, **kwargs):
        raise RuntimeError("db down")

    def save_derived(self, *args, **kwargs):
        raise RuntimeError("db down")


def test_persistence_failure_never_blocks_a_session(graph):
    orchestrator = SessionOrchestrator(
        graph, "kc.int.u_substitution", PROFILE, persistence=_ExplodingPersistence()
    )
    orchestrator.begin()
    guard = 0
    while orchestrator.phase not in (SessionPhase.DONE, SessionPhase.STOPPED):
        guard += 1
        assert guard < 100
        orchestrator.submit(orchestrator.pending_expected)
    assert orchestrator.phase == SessionPhase.DONE
    assert len(orchestrator.learner.events) > 0  # in-memory log intact


def test_api_healthz_reports_persistence_flag(graph, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    memory_only = TestClient(create_app(graph))
    assert memory_only.get("/healthz").json()["persistence"] is False
    persistent = TestClient(create_app(graph, database_url="sqlite+pysqlite:///:memory:"))
    assert persistent.get("/healthz").json()["persistence"] is True


def test_pilot_production_requires_postgres(graph, monkeypatch):
    monkeypatch.setenv("TUTOR_PILOT_PRODUCTION", "1")
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(RuntimeError, match="PostgreSQL"):
        create_app(graph)
    with pytest.raises(RuntimeError, match="PostgreSQL"):
        create_app(graph, database_url="sqlite+pysqlite:///:memory:")
