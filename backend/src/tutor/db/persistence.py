"""Durable persistence: learners, append-only evidence, episodes, derived mastery.

The evidence log is the authoritative record — append-only, never updated.
Derived mastery is a rebuildable cache (see ``LearnerModelService.replay``).
Episodes track each session's phase and routing envelope. Persistence
failures must never block a live session: callers catch, log, and continue
in memory (the orchestrator disables persistence on first failure).
"""

from datetime import timezone
from uuid import UUID

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from tutor.db.models import (
    DerivedMasteryRow,
    EpisodeRow,
    EvidenceEventRow,
    LearnerRow,
)
from tutor.db.session import create_all, get_engine
from tutor.schemas.common import ResponseClass
from tutor.schemas.learner import DerivedLearnerState, EvidenceEvent, LearnerProfile


class PersistenceService:
    """Synchronous persistence facade used by the orchestrator, API, and CLI."""

    def __init__(self, engine: Engine | None = None, url: str | None = None) -> None:
        self._engine = engine or get_engine(url)
        create_all(self._engine)

    @property
    def engine(self) -> Engine:
        """The underlying engine (exposed for tests and tooling)."""
        return self._engine

    # -- learners ---------------------------------------------------------------

    def ensure_learner(self, learner_id: UUID, profile: LearnerProfile) -> None:
        """Insert the learner identity row if missing (idempotent)."""
        with Session(self._engine) as session:
            if session.get(LearnerRow, str(learner_id)) is None:
                session.add(
                    LearnerRow(learner_id=str(learner_id), profile=profile.model_dump())
                )
                session.commit()

    # -- episodes ---------------------------------------------------------------

    def start_episode(self, learner_id: UUID, target_kc: str, envelope: dict) -> int:
        """Create an episode row for a new session; returns its id."""
        with Session(self._engine) as session:
            row = EpisodeRow(
                learner_id=str(learner_id), target_kc=target_kc, envelope=envelope
            )
            session.add(row)
            session.commit()
            return row.id

    def update_episode(self, episode_id: int, state: str, envelope: dict) -> None:
        """Checkpoint an episode's phase and routing envelope."""
        with Session(self._engine) as session:
            row = session.get(EpisodeRow, episode_id)
            if row is None:
                raise KeyError(f"unknown episode: {episode_id}")
            row.state = state
            row.envelope = envelope
            session.commit()

    # -- evidence (append-only) ---------------------------------------------------

    def record_event(self, event: EvidenceEvent) -> None:
        """Append one evidence event. There is deliberately no update path."""
        with Session(self._engine) as session:
            session.add(
                EvidenceEventRow(
                    event_id=str(event.event_id),
                    learner_id=str(event.learner_id),
                    t=event.t,
                    item_id=event.item_id,
                    kc_ids=list(event.kc_ids),
                    correct=event.correct,
                    response_class=event.response_class.value,
                    hints_used=event.hints_used,
                    assisted=event.assisted,
                    misconception_id=event.misconception_id,
                    content_versions=dict(event.content_versions),
                    episode_id=event.episode_id,
                    family_id=event.family_id,
                    surface=event.surface,
                    item_revision=event.item_revision,
                    attempt_number=event.attempt_number,
                    policy_version=event.policy_version,
                    learner_params_version=event.learner_params_version,
                    content_provenance=event.content_provenance,
                    learning_opportunity=event.learning_opportunity,
                )
            )
            session.commit()

    def load_events(self, learner_id: UUID) -> list[EvidenceEvent]:
        """Load a learner's evidence log in append order (for replay).

        SQLite returns naive datetimes; they are normalized to UTC so a
        replayed learner model compares equal to the in-memory one.
        """
        with Session(self._engine) as session:
            rows = session.scalars(
                select(EvidenceEventRow)
                .where(EvidenceEventRow.learner_id == str(learner_id))
                .order_by(EvidenceEventRow.id)
            ).all()
        events: list[EvidenceEvent] = []
        for row in rows:
            timestamp = (
                row.t if row.t.tzinfo is not None else row.t.replace(tzinfo=timezone.utc)
            )
            events.append(
                EvidenceEvent(
                    event_id=row.event_id,
                    learner_id=row.learner_id,
                    t=timestamp,
                    item_id=row.item_id,
                    kc_ids=list(row.kc_ids),
                    correct=row.correct,
                    response_class=ResponseClass(row.response_class),
                    hints_used=row.hints_used,
                    assisted=row.assisted,
                    misconception_id=row.misconception_id,
                    content_versions=dict(row.content_versions or {}),
                    episode_id=row.episode_id,
                    family_id=row.family_id,
                    surface=row.surface or "legacy",
                    item_revision=row.item_revision or 1,
                    attempt_number=row.attempt_number or 1,
                    policy_version=row.policy_version or "legacy",
                    learner_params_version=row.learner_params_version or "v1",
                    content_provenance=row.content_provenance or "legacy",
                    learning_opportunity=bool(row.learning_opportunity),
                )
            )
        return events

    # -- derived mastery (rebuildable cache) ---------------------------------------

    def save_derived(self, state: DerivedLearnerState) -> None:
        """Upsert the derived mastery cache for one learner."""
        with Session(self._engine) as session:
            existing = {
                row.kc_id: row
                for row in session.scalars(
                    select(DerivedMasteryRow).where(
                        DerivedMasteryRow.learner_id == str(state.learner_id)
                    )
                )
            }
            for kc_id, estimate in state.mastery.items():
                row = existing.get(kc_id)
                if row is None:
                    session.add(
                        DerivedMasteryRow(
                            learner_id=str(state.learner_id),
                            kc_id=kc_id,
                            direct=estimate.direct,
                            inferred=estimate.inferred,
                            observations=estimate.observations,
                            last_practiced=estimate.last_practiced,
                            params_version=state.params_version,
                            graph_version=state.graph_version,
                        )
                    )
                else:
                    row.direct = estimate.direct
                    row.inferred = estimate.inferred
                    row.observations = estimate.observations
                    row.last_practiced = estimate.last_practiced
                    row.params_version = state.params_version
                    row.graph_version = state.graph_version
            session.commit()
