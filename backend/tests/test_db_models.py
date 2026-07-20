"""ORM round-trips and unique constraints against SQLite in-memory."""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from tutor.db import models as m
from tutor.db.session import create_all, get_engine


@pytest.fixture()
def engine():
    engine = get_engine("sqlite+pysqlite:///:memory:")
    create_all(engine)
    return engine


def _learner(session: Session) -> m.LearnerRow:
    learner = m.LearnerRow(
        learner_id=str(uuid4()),
        profile={"course": "AP Calculus AB", "age_band": "16-18"},
    )
    session.add(learner)
    session.flush()
    return learner


def test_round_trip_every_table(engine):
    now = datetime.now(timezone.utc)
    with Session(engine) as session:
        graph_version = m.GraphVersionRow(version=1, status="published")
        session.add(graph_version)
        session.flush()
        session.add(
            m.KCNodeRow(
                graph_version_id=graph_version.id,
                kc_id="kc.alg.factoring",
                name="Factoring",
                description="d",
                course_level="Algebra 1",
                canonical_examples=["e"],
            )
        )
        session.add(
            m.KCEdgeRow(
                graph_version_id=graph_version.id,
                from_kc="kc.alg.factoring",
                to_kc="kc.alg.solve_quadratic",
                type="hard",
                rationale="r",
            )
        )
        session.add(
            m.PedagogyPackRow(
                graph_version_id=graph_version.id,
                kc_id="kc.alg.factoring",
                content={"misconceptions": []},
            )
        )
        learner = _learner(session)
        session.add(
            m.ResumeTokenRow(
                learner_id=learner.learner_id,
                token_hash="hash",
                expires_at=now + timedelta(days=7),
            )
        )
        evidence = m.EvidenceEventRow(
            event_id=str(uuid4()),
            learner_id=learner.learner_id,
            t=now,
            item_id="item",
            kc_ids=["kc.alg.factoring"],
            correct=True,
            response_class="widget",
            episode_id="session-v2",
            family_id="family-1",
            surface="guided_widget",
            item_revision=2,
            attempt_number=3,
            policy_version="diagnosis-v2.0",
            learner_params_version="v2",
            content_provenance="reviewed-item-bank",
        )
        session.add(evidence)
        session.add(
            m.DerivedMasteryRow(
                learner_id=learner.learner_id,
                kc_id="kc.alg.factoring",
                direct=0.6,
                inferred=0.4,
                observations=2,
                params_version=1,
                graph_version=1,
            )
        )
        session.add(
            m.EpisodeRow(
                learner_id=learner.learner_id,
                target_kc="kc.int.u_substitution",
                envelope={"budget": 40},
            )
        )
        session.add(
            m.GenerationJobRow(
                job_id=str(uuid4()),
                idempotency_key="job-1",
                kind="lesson",
                inputs={"graph": 1},
            )
        )
        session.add(
            m.MiniLessonRow(
                kc_id="kc.alg.factoring",
                applicability={"profile_band": "hs"},
                versions={"graph": 1},
                package={"narrative": "n"},
                telemetry_id="tl",
            )
        )
        checkpoint = m.SessionCheckpointRow(
            session_id="session-v2",
            learner_id=learner.learner_id,
            goal_id="goal.alg.factoring",
            target_kc="kc.alg.factoring",
            profile={"course": "Algebra 2", "age_band": "16-18"},
            requested_content_mode="curated",
            effective_content_mode="curated",
            revision=1,
            phase="diagnose",
            checkpoint={"schema_version": 2},
            started_at=now,
        )
        session.add(checkpoint)
        session.flush()
        session.scalars(select(m.ResumeTokenRow)).one().session_id = (
            checkpoint.session_id
        )
        session.add_all(
            [
                m.SessionMutationReceiptRow(
                    session_id=checkpoint.session_id,
                    request_id=str(uuid4()),
                    revision=1,
                    payload_hash="a" * 64,
                    request_payload={"type": "request_hint"},
                    response_payload={"schema_version": 2},
                ),
                m.TranscriptEntryRow(
                    session_id=checkpoint.session_id,
                    sequence=0,
                    entry={"role": "tutor", "text": "Welcome"},
                ),
                m.ItemExposureRow(
                    session_id=checkpoint.session_id,
                    item_id="item",
                    item_revision=2,
                    family_id="family-1",
                    surface="guided_widget",
                    exposure_sequence=0,
                ),
                m.WidgetAttemptRow(
                    session_id=checkpoint.session_id,
                    interaction_key="interaction-1",
                    attempt_number=1,
                    response={"text": "x"},
                    correct=False,
                    verification_status="invalid",
                    counted=False,
                ),
            ]
        )
        session.commit()

        assert session.scalars(select(m.KCNodeRow)).one().kc_id == "kc.alg.factoring"
        stored_evidence = session.scalars(select(m.EvidenceEventRow)).one()
        assert stored_evidence.correct is True
        assert stored_evidence.family_id == "family-1"
        assert stored_evidence.attempt_number == 3
        assert session.scalars(select(m.DerivedMasteryRow)).one().observations == 2
        assert session.scalars(select(m.GenerationJobRow)).one().status == "pending"
        assert session.scalars(select(m.SessionCheckpointRow)).one().revision == 1
        assert session.scalars(select(m.ResumeTokenRow)).one().session_id == "session-v2"
        assert session.scalars(select(m.TranscriptEntryRow)).one().sequence == 0
        assert session.scalars(select(m.ItemExposureRow)).one().item_revision == 2
        stored_widget_attempt = session.scalars(select(m.WidgetAttemptRow)).one()
        assert stored_widget_attempt.correct is False
        assert stored_widget_attempt.verification_status == "invalid"
        assert stored_widget_attempt.counted is False


def test_generation_job_idempotency_key_unique(engine):
    with Session(engine) as session:
        session.add(
            m.GenerationJobRow(
                job_id=str(uuid4()), idempotency_key="dup", kind="lesson", inputs={}
            )
        )
        session.commit()
        session.add(
            m.GenerationJobRow(
                job_id=str(uuid4()), idempotency_key="dup", kind="lesson", inputs={}
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()


def test_derived_mastery_unique_per_learner_kc(engine):
    with Session(engine) as session:
        learner = _learner(session)
        session.add(
            m.DerivedMasteryRow(
                learner_id=learner.learner_id,
                kc_id="kc.der.power_rule",
                direct=0.5,
                inferred=0.5,
                observations=1,
                params_version=1,
                graph_version=1,
            )
        )
        session.commit()
        session.add(
            m.DerivedMasteryRow(
                learner_id=learner.learner_id,
                kc_id="kc.der.power_rule",
                direct=0.7,
                inferred=0.5,
                observations=2,
                params_version=1,
                graph_version=1,
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()
