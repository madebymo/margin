"""Transactional persistence for API v2 sessions.

The legacy orchestrator persists evidence and episode state in separate
transactions.  V2 instead writes the authoritative checkpoint, transcript,
evidence delta, widget trajectory, and idempotency receipt together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal
from uuid import UUID

from sqlalchemy import Engine, delete, func, inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from tutor.api.v2_schemas import SessionView, TranscriptEntry
from tutor.api.v2_store import (
    SessionConflict,
    SessionRateLimited,
    V2SessionHandle,
)
from tutor.content.visible import canonical_visible_texts
from tutor.db.models import (
    EvidenceEventRow,
    ItemExposureRow,
    LearnerRow,
    ResumeTokenRow,
    SessionCheckpointRow,
    SessionMutationReceiptRow,
    TranscriptEntryRow,
    WidgetAttemptRow,
)
from tutor.schemas.assessment import ContentExposureState
from tutor.schemas.learner import EvidenceEvent

_RESUME_DAYS = 30
_REQUIRED_TABLES = {
    "session_checkpoints",
    "session_mutation_receipts",
    "transcript_entries",
    "item_exposures",
    "widget_attempts",
}
_REQUIRED_EVIDENCE_COLUMNS = {
    "pedagogy_catalog_version",
    "episode_id",
    "family_id",
    "surface",
    "item_revision",
    "attempt_number",
    "policy_version",
    "learner_params_version",
    "content_provenance",
    "learning_opportunity",
}
_REQUIRED_CHECKPOINT_COLUMNS = {"pedagogy_catalog_version"}
_REQUIRED_RESUME_COLUMNS = {"session_id"}
_REQUIRED_WIDGET_ATTEMPT_COLUMNS = {"verification_status", "counted"}
_SUPPORTED_CHECKPOINT_ENVELOPE_SCHEMAS = frozenset({3, 4})


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


class DurableLedgerMismatch(RuntimeError):
    """A normalized append-only ledger disagrees with its latest checkpoint."""

    def __init__(self, metric: str, message: str) -> None:
        super().__init__(message)
        self.metric = metric


@dataclass(frozen=True)
class RetentionBatch:
    """One bounded cursor page from anonymous-session retention.

    ``next_cursor`` is deliberately excluded from representations so routine
    job logging cannot disclose a session identifier. It is an internal
    database-order cursor, not a browser or operator-facing resume token.
    """

    purged: int
    scanned: int
    complete: bool
    next_cursor: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if type(self.purged) is not int or self.purged < 0:
            raise ValueError("purged must be a nonnegative integer")
        if type(self.scanned) is not int or self.scanned < self.purged:
            raise ValueError("scanned must be an integer at least as large as purged")
        if type(self.complete) is not bool:
            raise TypeError("complete must be a boolean")
        if self.complete and self.next_cursor is not None:
            raise ValueError("a complete retention page cannot carry a cursor")


@dataclass(frozen=True)
class ActiveResumeCheckpointPins:
    """Privacy-minimal version pins referenced by one or more live tokens.

    Every field is excluded from ``repr`` so an operational exception cannot
    accidentally print release coordinates or policy configuration. Readiness
    exposes only aggregate counts and booleans derived from these values.
    """

    envelope_schema_version: Any = field(repr=False)
    content_release: Any = field(repr=False)
    orchestrator_schema_version: Any = field(repr=False)
    graph_version: Any = field(repr=False)
    item_bank_version: Any = field(repr=False)
    pedagogy_catalog_version: Any = field(repr=False)
    release_id: Any = field(repr=False)
    release_digest: Any = field(repr=False)
    policy_versions: Any = field(repr=False)

    def restoration_checkpoint(self) -> dict[str, Any]:
        """Return registry input only when the checkpoint layers agree."""

        envelope_schema = self.envelope_schema_version
        if (
            type(envelope_schema) is not int
            or envelope_schema not in _SUPPORTED_CHECKPOINT_ENVELOPE_SCHEMAS
        ):
            raise ValueError("active resume checkpoint envelope is unsupported")
        expected_coordinate_keys = {
            "graph_version",
            "item_bank_version",
            "pedagogy_catalog_version",
        }
        expected_keys = (
            expected_coordinate_keys | {"release_id", "release_digest"}
            if envelope_schema == 4
            else expected_coordinate_keys
        )
        if not isinstance(self.content_release, dict) or set(
            self.content_release
        ) != expected_keys:
            raise ValueError("active resume checkpoint has invalid release pins")
        state = {
            "schema_version": self.orchestrator_schema_version,
            "graph_version": self.graph_version,
            "item_bank_version": self.item_bank_version,
            "pedagogy_catalog_version": self.pedagogy_catalog_version,
            "policy_versions": (
                dict(self.policy_versions)
                if isinstance(self.policy_versions, dict)
                else self.policy_versions
            ),
        }
        state_release = {
            key: state.get(key) for key in expected_coordinate_keys
        }
        if envelope_schema == 4:
            state["release_id"] = self.release_id
            state["release_digest"] = self.release_digest
            state_release.update(
                release_id=self.release_id,
                release_digest=self.release_digest,
            )
        if self.content_release != state_release:
            raise ValueError("active resume checkpoint release pins disagree")
        return state


class V2PersistenceService:
    """Postgres/SQLite-compatible transaction boundary for v2 actions."""

    def __init__(
        self,
        engine: Engine,
        *,
        max_episodes_per_learner: int = 32,
    ) -> None:
        self._engine = engine
        self._max_episodes_per_learner = max_episodes_per_learner
        self._validate_schema()

    @property
    def engine(self) -> Engine:
        return self._engine

    def _validate_schema(self) -> None:
        inspector = inspect(self._engine)
        table_names = set(inspector.get_table_names())
        missing_tables = sorted(_REQUIRED_TABLES - table_names)
        evidence_columns = {
            column["name"]
            for column in inspector.get_columns("evidence_events")
        }
        missing_columns = sorted(_REQUIRED_EVIDENCE_COLUMNS - evidence_columns)
        checkpoint_columns = (
            {
                column["name"]
                for column in inspector.get_columns("session_checkpoints")
            }
            if "session_checkpoints" in table_names
            else set()
        )
        missing_checkpoint_columns = sorted(
            _REQUIRED_CHECKPOINT_COLUMNS - checkpoint_columns
        )
        resume_columns = (
            {
                column["name"]
                for column in inspector.get_columns("resume_tokens")
            }
            if "resume_tokens" in table_names
            else set()
        )
        missing_resume_columns = sorted(_REQUIRED_RESUME_COLUMNS - resume_columns)
        widget_attempt_columns = (
            {
                column["name"]
                for column in inspector.get_columns("widget_attempts")
            }
            if "widget_attempts" in table_names
            else set()
        )
        missing_widget_columns = sorted(
            _REQUIRED_WIDGET_ATTEMPT_COLUMNS - widget_attempt_columns
        )
        if (
            missing_tables
            or missing_columns
            or missing_checkpoint_columns
            or missing_resume_columns
            or missing_widget_columns
        ):
            details = []
            if missing_tables:
                details.append(f"tables={missing_tables}")
            if missing_columns:
                details.append(f"evidence columns={missing_columns}")
            if missing_checkpoint_columns:
                details.append(
                    f"checkpoint columns={missing_checkpoint_columns}"
                )
            if missing_resume_columns:
                details.append(f"resume-token columns={missing_resume_columns}")
            if missing_widget_columns:
                details.append(f"widget-attempt columns={missing_widget_columns}")
            raise RuntimeError(
                "database is not ready for session API v2 "
                f"({'; '.join(details)}); run "
                "`python -m tutor.db.migrate_session_v2` before startup"
            )

    def create_session(
        self,
        handle: V2SessionHandle,
        token_hash: str,
        request_id: UUID,
        *,
        replace_token_hash: str | None = None,
        replace_session_id: str | None = None,
        replace_expected_revision: int | None = None,
    ) -> dict[str, Any] | None:
        """Atomically create identity, resume token, initial view, and transcript."""
        view = self._view_from_create_receipt(handle, request_id)
        receipt = handle.receipts[str(request_id)]
        checkpoint = self._checkpoint(handle, view)
        try:
            with Session(self._engine) as session:
                replacement_token = None
                if replace_session_id is not None:
                    previous = session.scalar(
                        select(SessionCheckpointRow)
                        .where(SessionCheckpointRow.session_id == replace_session_id)
                        .with_for_update()
                    )
                    if previous is None:
                        raise KeyError(f"unknown v2 session: {replace_session_id}")
                    replacement_token = session.scalar(
                        select(ResumeTokenRow)
                        .where(ResumeTokenRow.token_hash == replace_token_hash)
                        .with_for_update()
                    )

                existing = session.scalar(
                    select(SessionMutationReceiptRow)
                    .where(SessionMutationReceiptRow.request_id == str(request_id))
                    .order_by(SessionMutationReceiptRow.id)
                )
                if existing is not None:
                    if existing.payload_hash != receipt.payload_hash:
                        raise SessionConflict(
                            "idempotency_conflict",
                            "request_id was already used with a different payload",
                        )
                    if existing.request_payload.get("type") != "create":
                        raise SessionConflict(
                            "idempotency_conflict",
                            "request_id was already used for another mutation",
                        )
                    return dict(existing.response_payload)

                if replace_session_id is not None:
                    assert previous is not None
                    now = _utcnow()
                    if (
                        replacement_token is None
                        or replacement_token.revoked
                        or replacement_token.session_id != replace_session_id
                        or replacement_token.learner_id != handle.learner_id
                        or _aware(replacement_token.expires_at) <= now
                    ):
                        raise SessionConflict(
                            "session_revoked",
                            "the anonymous session token is no longer active",
                            self._view_from_checkpoint(previous),
                        )
                    if previous.revision != replace_expected_revision:
                        raise SessionConflict(
                            "stale_interaction",
                            "session revision changed; use the authoritative snapshot",
                            self._view_from_checkpoint(previous),
                        )
                    if previous.phase not in {"done", "stopped"}:
                        raise SessionConflict(
                            "active_session_exists",
                            "reset or finish the current session before starting another",
                            self._view_from_checkpoint(previous),
                        )
                    self._enforce_episode_quota(
                        session, handle.learner_id, now=now
                    )

                if session.get(LearnerRow, handle.learner_id) is None:
                    session.add(
                        LearnerRow(
                            learner_id=handle.learner_id,
                            profile=handle.profile.model_dump(mode="json"),
                        )
                    )
                    session.flush()
                session.add(
                    SessionCheckpointRow(
                        session_id=handle.session_id,
                        learner_id=handle.learner_id,
                        goal_id=handle.goal.goal_id,
                        target_kc=handle.goal.target_kc,
                        profile=handle.profile.model_dump(mode="json"),
                        requested_content_mode=handle.content_mode.requested,
                        effective_content_mode=handle.content_mode.effective,
                        fallback_reason=handle.content_mode.fallback_reason,
                        pedagogy_catalog_version=self._catalog_version(handle),
                        revision=handle.revision,
                        phase=view.phase,
                        checkpoint=checkpoint,
                        started_at=handle.started_at,
                        updated_at=handle.updated_at,
                    )
                )
                session.add(
                    ResumeTokenRow(
                        learner_id=handle.learner_id,
                        session_id=handle.session_id,
                        token_hash=token_hash,
                        expires_at=_utcnow() + timedelta(days=_RESUME_DAYS),
                    )
                )
                for entry in handle.transcript:
                    session.add(self._transcript_row(handle.session_id, entry))
                exposure_state = getattr(
                    handle.orchestrator,
                    "exposure_state",
                    getattr(handle.orchestrator, "_exposure_state", None),
                )
                for sequence, exposure in enumerate(
                    getattr(
                        exposure_state,
                        "records",
                        getattr(exposure_state, "exposures", ()),
                    )
                    if exposure_state
                    else ()
                ):
                    session.add(
                        self._exposure_row(handle.session_id, sequence, exposure)
                    )
                session.add(
                    SessionMutationReceiptRow(
                        session_id=handle.session_id,
                        request_id=str(request_id),
                        revision=handle.revision,
                        payload_hash=receipt.payload_hash,
                        request_payload=receipt.request_payload,
                        response_payload=view.model_dump(mode="json"),
                    )
                )
                if replacement_token is not None:
                    replacement_token.revoked = True
                    replacement_token.expires_at = _utcnow() + timedelta(
                        days=_RESUME_DAYS
                    )
                session.commit()
        except IntegrityError:
            replayed = self.replay_create(
                request_id=request_id,
                payload_hash=receipt.payload_hash,
            )
            if replayed is not None:
                return replayed
            raise
        return None

    def commit_action(
        self,
        *,
        handle: V2SessionHandle,
        previous_revision: int,
        request_id: UUID,
        payload_hash: str,
        request_payload: dict[str, Any],
        response_payload: dict[str, Any],
        new_transcript: list[TranscriptEntry],
        new_events: list[Any],
        new_exposures: list[Any],
        widget_attempt: dict[str, Any] | None,
        token_hash: str,
    ) -> dict[str, Any] | None:
        """Commit one action or return a previously committed identical receipt."""
        with Session(self._engine) as session:
            checkpoint_row = session.scalar(
                select(SessionCheckpointRow)
                .where(SessionCheckpointRow.session_id == handle.session_id)
                .with_for_update()
            )
            if checkpoint_row is None:
                raise KeyError(f"unknown v2 session: {handle.session_id}")
            self._validate_checkpoint_release_pin(
                checkpoint_row,
                expected_catalog_version=self._catalog_version(handle),
            )

            token = session.scalar(
                select(ResumeTokenRow)
                .where(ResumeTokenRow.token_hash == token_hash)
                .with_for_update()
            )
            if (
                token is None
                or token.revoked
                or token.learner_id != handle.learner_id
                or token.session_id != handle.session_id
                or _aware(token.expires_at) <= _utcnow()
            ):
                raise SessionConflict(
                    "session_revoked",
                    "the anonymous session token is no longer active",
                    self._view_from_checkpoint(checkpoint_row),
                )

            existing = session.scalar(
                select(SessionMutationReceiptRow).where(
                    SessionMutationReceiptRow.session_id == handle.session_id,
                    SessionMutationReceiptRow.request_id == str(request_id),
                )
            )
            if existing is not None:
                if existing.payload_hash != payload_hash:
                    current = self._view_from_checkpoint(checkpoint_row)
                    raise SessionConflict(
                        "idempotency_conflict",
                        "request_id was already used with a different payload",
                        current,
                    )
                token.expires_at = _utcnow() + timedelta(days=_RESUME_DAYS)
                session.commit()
                return dict(existing.response_payload)

            if checkpoint_row.revision != previous_revision:
                current = self._view_from_checkpoint(checkpoint_row)
                raise SessionConflict(
                    "stale_interaction",
                    "session revision changed; use the authoritative snapshot",
                    current,
                )

            view = SessionView.model_validate(response_payload)
            checkpoint_row.revision = handle.revision
            checkpoint_row.phase = view.phase
            checkpoint_row.checkpoint = self._checkpoint(handle, view)
            checkpoint_row.updated_at = handle.updated_at
            for entry in new_transcript:
                session.add(self._transcript_row(handle.session_id, entry))
            for event in new_events:
                session.add(self._evidence_row(event, handle))
            max_exposure = session.scalar(
                select(func.max(ItemExposureRow.exposure_sequence)).where(
                    ItemExposureRow.session_id == handle.session_id
                )
            )
            existing_exposure = -1 if max_exposure is None else max_exposure
            for offset, exposure in enumerate(new_exposures, start=1):
                session.add(
                    self._exposure_row(
                        handle.session_id, existing_exposure + offset, exposure
                    )
                )
            if widget_attempt is not None:
                attempt_number = (
                    session.scalar(
                        select(func.max(WidgetAttemptRow.attempt_number)).where(
                            WidgetAttemptRow.session_id == handle.session_id,
                            WidgetAttemptRow.interaction_key
                            == widget_attempt["interaction_key"],
                        )
                    )
                    or 0
                ) + 1
                session.add(
                    WidgetAttemptRow(
                        session_id=handle.session_id,
                        interaction_key=widget_attempt["interaction_key"],
                        attempt_number=attempt_number,
                        response=dict(widget_attempt["response"]),
                        correct=bool(widget_attempt["correct"]),
                        verification_status=str(
                            widget_attempt["verification_status"]
                        ),
                        counted=bool(widget_attempt["counted"]),
                    )
                )
            session.add(
                SessionMutationReceiptRow(
                    session_id=handle.session_id,
                    request_id=str(request_id),
                    revision=handle.revision,
                    payload_hash=payload_hash,
                    request_payload=request_payload,
                    response_payload=response_payload,
                )
            )
            token.expires_at = _utcnow() + timedelta(days=_RESUME_DAYS)
            session.commit()
        return None

    def touch_resume(self, token_hash: str) -> bool:
        """Roll one active token's expiry without mutating its session checkpoint."""
        now = _utcnow()
        with Session(self._engine) as session:
            token = session.scalar(
                select(ResumeTokenRow)
                .where(ResumeTokenRow.token_hash == token_hash)
                .with_for_update()
            )
            if (
                token is None
                or token.revoked
                or _aware(token.expires_at) <= now
            ):
                return False
            token.expires_at = now + timedelta(days=_RESUME_DAYS)
            session.commit()
        return True

    def replay_create(
        self, *, request_id: UUID, payload_hash: str
    ) -> dict[str, Any] | None:
        """Return a committed creation receipt without requiring its cookie."""
        with Session(self._engine) as session:
            matching_receipts = session.scalars(
                select(SessionMutationReceiptRow)
                .where(SessionMutationReceiptRow.request_id == str(request_id))
                .order_by(SessionMutationReceiptRow.id)
            ).all()
            receipt = next(
                (
                    row
                    for row in matching_receipts
                    if row.request_payload.get("type") == "create"
                ),
                None,
            )
            if receipt is None:
                if matching_receipts:
                    raise SessionConflict(
                        "idempotency_conflict",
                        "request_id was already used for another mutation",
                    )
                return None
            checkpoint = session.get(SessionCheckpointRow, receipt.session_id)
            if receipt.payload_hash != payload_hash:
                raise SessionConflict(
                    "idempotency_conflict",
                    "request_id was already used with a different payload",
                    self._view_from_checkpoint(checkpoint) if checkpoint else None,
                )
            return dict(receipt.response_payload)

    def commit_reset(
        self,
        *,
        handle: V2SessionHandle,
        token_hash: str,
        request_id: UUID,
        payload_hash: str,
        request_payload: dict[str, Any],
        response_payload: dict[str, Any],
        expected_revision: int,
        pending_key: str | None,
        replacement: V2SessionHandle,
        replacement_token_hash: str,
    ) -> dict[str, Any] | None:
        """Receipt/reset the old episode and create its replacement atomically."""
        now = _utcnow()
        with Session(self._engine) as session:
            checkpoint = session.scalar(
                select(SessionCheckpointRow)
                .where(SessionCheckpointRow.session_id == handle.session_id)
                .with_for_update()
            )
            if checkpoint is None:
                raise KeyError(f"unknown v2 session: {handle.session_id}")
            self._validate_checkpoint_release_pin(
                checkpoint,
                expected_catalog_version=self._catalog_version(handle),
            )

            existing = session.scalar(
                select(SessionMutationReceiptRow).where(
                    SessionMutationReceiptRow.session_id == handle.session_id,
                    SessionMutationReceiptRow.request_id == str(request_id),
                )
            )
            if existing is not None:
                if existing.payload_hash != payload_hash:
                    raise SessionConflict(
                        "idempotency_conflict",
                        "request_id was already used with a different payload",
                        self._view_from_checkpoint(checkpoint),
                    )
                return dict(existing.response_payload)

            token = session.scalar(
                select(ResumeTokenRow)
                .where(ResumeTokenRow.token_hash == token_hash)
                .with_for_update()
            )
            if (
                token is None
                or token.revoked
                or token.learner_id != handle.learner_id
                or token.session_id != handle.session_id
                or _aware(token.expires_at) <= now
            ):
                raise SessionConflict(
                    "session_revoked",
                    "the anonymous session token is no longer active",
                    self._view_from_checkpoint(checkpoint),
                )
            if checkpoint.revision != expected_revision:
                raise SessionConflict(
                    "stale_interaction",
                    "session revision changed; use the authoritative snapshot",
                    self._view_from_checkpoint(checkpoint),
                )
            checkpoint_view = self._view_from_checkpoint(checkpoint)
            authoritative_key = (
                checkpoint_view.pending.key
                if checkpoint_view is not None and checkpoint_view.pending is not None
                else None
            )
            if authoritative_key != pending_key:
                raise SessionConflict(
                    "stale_interaction",
                    "the pending interaction changed; use the authoritative snapshot",
                    checkpoint_view,
                )
            self._enforce_episode_quota(
                session, handle.learner_id, now=now
            )

            session.add(
                SessionMutationReceiptRow(
                    session_id=handle.session_id,
                    request_id=str(request_id),
                    revision=checkpoint.revision,
                    payload_hash=payload_hash,
                    request_payload=request_payload,
                    response_payload=response_payload,
                )
            )
            replacement_view = SessionView.model_validate(
                response_payload["session"]
            )
            session.add(
                SessionCheckpointRow(
                    session_id=replacement.session_id,
                    learner_id=replacement.learner_id,
                    goal_id=replacement.goal.goal_id,
                    target_kc=replacement.goal.target_kc,
                    profile=replacement.profile.model_dump(mode="json"),
                    requested_content_mode=replacement.content_mode.requested,
                    effective_content_mode=replacement.content_mode.effective,
                    fallback_reason=replacement.content_mode.fallback_reason,
                    pedagogy_catalog_version=self._catalog_version(replacement),
                    revision=replacement.revision,
                    phase=replacement_view.phase,
                    checkpoint=self._checkpoint(replacement, replacement_view),
                    started_at=replacement.started_at,
                    updated_at=replacement.updated_at,
                )
            )
            session.flush()
            for entry in replacement.transcript:
                session.add(self._transcript_row(replacement.session_id, entry))
            exposure_state = getattr(
                replacement.orchestrator,
                "exposure_state",
                getattr(replacement.orchestrator, "_exposure_state", None),
            )
            exposures = (
                getattr(
                    exposure_state,
                    "records",
                    getattr(exposure_state, "exposures", ()),
                )
                if exposure_state
                else ()
            )
            for sequence, exposure in enumerate(exposures):
                session.add(
                    self._exposure_row(
                        replacement.session_id, sequence, exposure
                    )
                )
            session.add(
                ResumeTokenRow(
                    learner_id=replacement.learner_id,
                    session_id=replacement.session_id,
                    token_hash=replacement_token_hash,
                    expires_at=now + timedelta(days=_RESUME_DAYS),
                )
            )
            token.revoked = True
            token.expires_at = now + timedelta(days=_RESUME_DAYS)
            session.commit()
        return None

    def replay_reset(
        self,
        *,
        token_hash: str,
        request_id: UUID,
        payload_hash: str,
    ) -> dict[str, Any] | None:
        """Find a reset receipt using a token that is intentionally revoked."""
        with Session(self._engine) as session:
            token = session.scalar(
                select(ResumeTokenRow).where(ResumeTokenRow.token_hash == token_hash)
            )
            if token is None:
                return None
            checkpoint = (
                session.get(SessionCheckpointRow, token.session_id)
                if token.session_id is not None
                else None
            )
            if checkpoint is None:
                return None
            receipt = session.scalar(
                select(SessionMutationReceiptRow).where(
                    SessionMutationReceiptRow.session_id == checkpoint.session_id,
                    SessionMutationReceiptRow.request_id == str(request_id),
                )
            )
            if receipt is None:
                return None
            if receipt.payload_hash != payload_hash:
                raise SessionConflict(
                    "idempotency_conflict",
                    "request_id was already used with a different payload",
                    self._view_from_checkpoint(checkpoint),
                )
            if receipt.request_payload.get("type") != "reset":
                raise SessionConflict(
                    "idempotency_conflict",
                    "request_id was already used for another mutation",
                    self._view_from_checkpoint(checkpoint),
                )
            return dict(receipt.response_payload)

    def recover_create(self, request_id: UUID) -> str | None:
        """Resolve an active create receipt using its client-held request id."""

        now = _utcnow()
        with Session(self._engine) as session:
            receipts = session.scalars(
                select(SessionMutationReceiptRow)
                .where(SessionMutationReceiptRow.request_id == str(request_id))
                .order_by(SessionMutationReceiptRow.id)
            ).all()
            receipt = next(
                (
                    row
                    for row in receipts
                    if row.request_payload.get("type") == "create"
                ),
                None,
            )
            if receipt is None:
                return None
            token = session.scalar(
                select(ResumeTokenRow).where(
                    ResumeTokenRow.session_id == receipt.session_id,
                    ResumeTokenRow.revoked.is_(False),
                    ResumeTokenRow.expires_at > now,
                )
            )
            return receipt.session_id if token is not None else None

    def recover_reset_rotation(
        self, token_hash: str, request_id: UUID
    ) -> str | None:
        """Resolve reset with the revoked cookie hash plus exact request id."""

        now = _utcnow()
        with Session(self._engine) as session:
            token = session.scalar(
                select(ResumeTokenRow).where(ResumeTokenRow.token_hash == token_hash)
            )
            if (
                token is None
                or not token.revoked
                or token.session_id is None
                or _aware(token.expires_at) <= now
            ):
                return None
            receipt = session.scalar(
                select(SessionMutationReceiptRow).where(
                    SessionMutationReceiptRow.session_id == token.session_id,
                    SessionMutationReceiptRow.request_id == str(request_id),
                )
            )
            if receipt is None or receipt.request_payload.get("type") != "reset":
                return None
            replacement = receipt.response_payload.get("session")
            replacement_session_id = (
                replacement.get("session_id")
                if isinstance(replacement, dict)
                else None
            )
            if not isinstance(replacement_session_id, str):
                return None
            successor = session.scalar(
                select(ResumeTokenRow).where(
                    ResumeTokenRow.session_id == replacement_session_id,
                    ResumeTokenRow.revoked.is_(False),
                    ResumeTokenRow.expires_at > now,
                )
            )
            return replacement_session_id if successor is not None else None

    def revoke_token(self, token_hash: str) -> None:
        with Session(self._engine) as session:
            row = session.scalar(
                select(ResumeTokenRow)
                .where(ResumeTokenRow.token_hash == token_hash)
                .with_for_update()
            )
            if row is not None:
                row.revoked = True
                session.commit()

    def resolve_resume(self, token_hash: str) -> dict[str, Any] | None:
        """Resolve a valid token to the latest checkpoint for process recovery."""
        with Session(self._engine) as session:
            token = session.scalar(
                select(ResumeTokenRow).where(
                    ResumeTokenRow.token_hash == token_hash,
                    ResumeTokenRow.revoked.is_(False),
                )
            )
            if token is None or _aware(token.expires_at) <= _utcnow():
                return None
            checkpoint = (
                session.get(SessionCheckpointRow, token.session_id)
                if token.session_id is not None
                else None
            )
            if checkpoint is None:
                return None
            receipts = session.scalars(
                select(SessionMutationReceiptRow)
                .where(SessionMutationReceiptRow.session_id == checkpoint.session_id)
                .order_by(SessionMutationReceiptRow.revision, SessionMutationReceiptRow.id)
            ).all()
            transcript_rows = session.scalars(
                select(TranscriptEntryRow)
                .where(TranscriptEntryRow.session_id == checkpoint.session_id)
                .order_by(TranscriptEntryRow.sequence)
            ).all()
            exposure_rows = session.scalars(
                select(ItemExposureRow)
                .where(ItemExposureRow.session_id == checkpoint.session_id)
                .order_by(ItemExposureRow.exposure_sequence)
            ).all()
            evidence_rows = session.scalars(
                select(EvidenceEventRow)
                .where(EvidenceEventRow.learner_id == checkpoint.learner_id)
                .order_by(EvidenceEventRow.id)
            ).all()
            widget_rows = session.scalars(
                select(WidgetAttemptRow)
                .where(WidgetAttemptRow.session_id == checkpoint.session_id)
                .order_by(WidgetAttemptRow.id)
            ).all()
            ledger = self._validate_resume_ledgers(
                checkpoint=checkpoint,
                receipts=receipts,
                transcript_rows=transcript_rows,
                exposure_rows=exposure_rows,
                evidence_rows=evidence_rows,
                widget_rows=widget_rows,
            )
            return {
                "checkpoint": dict(checkpoint.checkpoint),
                "requests": [
                    dict(receipt.request_payload)
                    for receipt in receipts
                    if receipt.request_payload.get("type") not in {"create", "reset"}
                ],
                "receipts": [
                    {
                        "request_id": receipt.request_id,
                        "payload_hash": receipt.payload_hash,
                        "request_payload": dict(receipt.request_payload),
                        "response_payload": dict(receipt.response_payload),
                    }
                    for receipt in receipts
                    if receipt.request_payload.get("type") != "reset"
                ],
                "ledger": ledger,
            }

    def resume_token_status(
        self, token_hash: str
    ) -> Literal["active", "expired", "invalid"]:
        """Classify a cookie hash without exposing or extending its session.

        This read-only operational seam keeps malformed/revoked/unknown tokens
        separate from rows that are observably past expiry. Retention may have
        already purged an old row, in which case ``invalid`` is the only honest
        classification available.
        """
        with Session(self._engine) as session:
            token = session.scalar(
                select(ResumeTokenRow).where(
                    ResumeTokenRow.token_hash == token_hash
                )
            )
            if token is None or token.revoked:
                return "invalid"
            if _aware(token.expires_at) <= _utcnow():
                return "expired"
            return "active"

    def active_resume_checkpoint_pins(
        self,
        *,
        as_of: datetime | None = None,
    ) -> tuple[ActiveResumeCheckpointPins, ...]:
        """Load only distinct restoration pins referenced by active tokens.

        The projection deliberately excludes token hashes, learner/session
        identifiers, transcript data, context, responses, and the remainder of
        each checkpoint. A legacy token without an exact v2 session binding is
        not resumable and therefore does not contribute a pin set.
        """

        now = as_of or _utcnow()
        checkpoint = SessionCheckpointRow.checkpoint
        orchestrator = checkpoint["orchestrator"]
        statement = (
            select(
                checkpoint["schema_version"].as_integer(),
                checkpoint["content_release"],
                orchestrator["schema_version"].as_integer(),
                orchestrator["graph_version"].as_integer(),
                orchestrator["item_bank_version"].as_string(),
                orchestrator["pedagogy_catalog_version"].as_string(),
                orchestrator["release_id"].as_string(),
                orchestrator["release_digest"].as_string(),
                orchestrator["policy_versions"],
            )
            .join(
                ResumeTokenRow,
                ResumeTokenRow.session_id == SessionCheckpointRow.session_id,
            )
            .where(
                ResumeTokenRow.session_id.is_not(None),
                ResumeTokenRow.revoked.is_(False),
                ResumeTokenRow.expires_at > now,
            )
            .distinct()
        )
        with Session(self._engine) as session:
            rows = session.execute(statement).all()
        return tuple(
            ActiveResumeCheckpointPins(
                envelope_schema_version=row[0],
                content_release=row[1],
                orchestrator_schema_version=row[2],
                graph_version=row[3],
                item_bank_version=row[4],
                pedagogy_catalog_version=row[5],
                release_id=row[6],
                release_digest=row[7],
                policy_versions=row[8],
            )
            for row in rows
        )

    def _enforce_episode_quota(
        self,
        session: Session,
        learner_id: str,
        *,
        now: datetime,
    ) -> None:
        # A checkpoint row only serializes mutations for one episode.  Lock
        # the shared learner identity as well so two independently restored
        # episodes cannot both observe the final free quota slot and create
        # replacements concurrently on PostgreSQL.
        learner = session.scalar(
            select(LearnerRow)
            .where(LearnerRow.learner_id == learner_id)
            .with_for_update()
        )
        if learner is None:
            raise KeyError(f"unknown learner: {learner_id}")
        cutoff = now - timedelta(days=_RESUME_DAYS)
        recent = session.scalar(
            select(func.count(SessionCheckpointRow.session_id)).where(
                SessionCheckpointRow.learner_id == learner_id,
                SessionCheckpointRow.started_at >= cutoff,
            )
        ) or 0
        if recent >= self._max_episodes_per_learner:
            raise SessionRateLimited(
                "this anonymous learner reached the rolling episode limit"
            )

    def purge_expired_anonymous_sessions(self, *, limit: int = 100) -> int:
        """Compatibility wrapper that removes one bounded cursor page."""

        return self.purge_expired_anonymous_sessions_batch(limit=limit).purged

    def purge_expired_anonymous_sessions_batch(
        self,
        *,
        limit: int = 100,
        after_session_id: str | None = None,
    ) -> RetentionBatch:
        """Remove one cursor page after every token for a session expires.

        Learner identity and append-only evidence remain intact for longitudinal
        replay; only anonymous resume/checkpoint/transcript/receipt/widget data
        outside the promised 30-day window is removed.  This maintenance method
        is intentionally never called by the constructor or a request path.

        Candidate eligibility is calculated with ``MAX(expires_at)`` before
        locking, then rechecked under the normal checkpoint-then-token lock
        order. The cursor advances across a raced refresh without repeatedly
        selecting it and starving later eligible sessions.
        """
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("limit must be an integer between 1 and 1000")
        if after_session_id is not None and (
            not isinstance(after_session_id, str)
            or not after_session_id
            or len(after_session_id) > 128
            or not after_session_id.isprintable()
        ):
            raise ValueError("after_session_id is invalid")
        now = _utcnow()
        with Session(self._engine) as session:
            candidates = (
                select(ResumeTokenRow.session_id)
                .where(ResumeTokenRow.session_id.is_not(None))
                .group_by(ResumeTokenRow.session_id)
                .having(func.max(ResumeTokenRow.expires_at) <= now)
            )
            if after_session_id is not None:
                candidates = candidates.where(
                    ResumeTokenRow.session_id > after_session_id
                )
            candidates = candidates.order_by(ResumeTokenRow.session_id).limit(limit)
            candidate_ids = list(
                session.scalars(candidates)
            )
            if not candidate_ids:
                return RetentionBatch(purged=0, scanned=0, complete=True)
            complete = len(candidate_ids) < limit
            next_cursor = None if complete else candidate_ids[-1]

            # Use the same checkpoint-then-token lock order as action/reset
            # commits.  Re-reading token expiry after both locks prevents a
            # rolling touch that wins the race from being purged, while the
            # stable ordering avoids cross-session deadlocks between purgers.
            locked_checkpoints = session.scalars(
                select(SessionCheckpointRow)
                .where(SessionCheckpointRow.session_id.in_(candidate_ids))
                .order_by(SessionCheckpointRow.session_id)
                .with_for_update()
            ).all()
            locked_session_ids = [row.session_id for row in locked_checkpoints]
            if not locked_session_ids:
                return RetentionBatch(
                    purged=0,
                    scanned=len(candidate_ids),
                    complete=complete,
                    next_cursor=next_cursor,
                )
            token_rows = session.scalars(
                select(ResumeTokenRow)
                .where(ResumeTokenRow.session_id.in_(locked_session_ids))
                .order_by(ResumeTokenRow.id)
                .with_for_update()
            ).all()
            by_session: dict[str, list[ResumeTokenRow]] = {}
            for token in token_rows:
                if token.session_id is not None:
                    by_session.setdefault(token.session_id, []).append(token)
            expired_session_ids = [
                session_id
                for session_id, tokens in by_session.items()
                if tokens and all(_aware(token.expires_at) <= now for token in tokens)
            ]
            if not expired_session_ids:
                return RetentionBatch(
                    purged=0,
                    scanned=len(candidate_ids),
                    complete=complete,
                    next_cursor=next_cursor,
                )
            for row_type in (
                WidgetAttemptRow,
                ItemExposureRow,
                TranscriptEntryRow,
                SessionMutationReceiptRow,
                ResumeTokenRow,
            ):
                session.execute(
                    delete(row_type).where(
                        row_type.session_id.in_(expired_session_ids)
                    )
                )
            session.execute(
                delete(SessionCheckpointRow).where(
                    SessionCheckpointRow.session_id.in_(expired_session_ids)
                )
            )
            session.commit()
            return RetentionBatch(
                purged=len(expired_session_ids),
                scanned=len(candidate_ids),
                complete=complete,
                next_cursor=next_cursor,
            )

    @classmethod
    def _validate_resume_ledgers(
        cls,
        *,
        checkpoint: SessionCheckpointRow,
        receipts: list[SessionMutationReceiptRow],
        transcript_rows: list[TranscriptEntryRow],
        exposure_rows: list[ItemExposureRow],
        evidence_rows: list[EvidenceEventRow],
        widget_rows: list[WidgetAttemptRow],
    ) -> dict[str, Any]:
        """Normalize and cross-check every durable append-only session ledger."""
        checkpoint_payload = dict(checkpoint.checkpoint or {})
        view = SessionView.model_validate(checkpoint_payload.get("session_view"))
        state = checkpoint_payload.get("orchestrator")
        if not isinstance(state, dict):
            raise DurableLedgerMismatch(
                "checkpoint_integrity_failures",
                "durable checkpoint has no orchestrator state",
            )
        cls._validate_checkpoint_release_pin(checkpoint)
        if view.revision != checkpoint.revision or view.phase != checkpoint.phase:
            raise DurableLedgerMismatch(
                "checkpoint_integrity_failures",
                "checkpoint row and safe session view disagree",
            )

        action_receipts = [
            receipt
            for receipt in receipts
            if receipt.request_payload.get("type") not in {"create", "reset"}
        ]
        if [receipt.revision for receipt in action_receipts] != list(
            range(1, checkpoint.revision + 1)
        ):
            raise DurableLedgerMismatch(
                "duplicate_advances_detected",
                "mutation receipt revisions are missing, duplicated, or out of order",
            )
        if action_receipts:
            latest = SessionView.model_validate(action_receipts[-1].response_payload)
            if latest != view:
                raise DurableLedgerMismatch(
                    "checkpoint_integrity_failures",
                    "latest mutation receipt and checkpoint view disagree",
                )

        expected_transcript = [
            entry.model_dump(mode="json") for entry in view.transcript
        ]
        actual_transcript = [
            TranscriptEntry.model_validate(row.entry).model_dump(mode="json")
            for row in transcript_rows
        ]
        if [row.sequence for row in transcript_rows] != list(
            range(len(transcript_rows))
        ) or actual_transcript != expected_transcript:
            raise DurableLedgerMismatch(
                "transcript_integrity_failures",
                "append-only transcript and checkpoint transcript disagree",
            )

        public_visible_values: list[Any] = [checkpoint_payload.get("context")]
        for entry in view.transcript:
            public_visible_values.extend(
                (entry.text, entry.prompt_segments, entry.widget)
            )
        if view.pending is not None:
            choice_options = getattr(view.pending.input, "options", ())
            public_visible_values.extend(
                (
                    view.pending.prompt,
                    view.pending.prompt_segments,
                    choice_options,
                )
            )
        expected_visible_texts = canonical_visible_texts(*public_visible_values)
        if state.get("visible_texts") != expected_visible_texts:
            raise DurableLedgerMismatch(
                "visible_content_integrity_failures",
                "checkpoint visible-content ledger and public session view disagree",
            )

        expected_private_inputs = canonical_visible_texts(
            *(row.response for row in widget_rows)
        )
        if state.get("private_visible_inputs") != expected_private_inputs:
            raise DurableLedgerMismatch(
                "visible_content_integrity_failures",
                "checkpoint private visible-input ledger and widget attempts disagree",
            )

        exposure_state = ContentExposureState.model_validate(
            state.get("exposure_state", {})
        )
        expected_exposures = [
            (
                record.item_id,
                record.revision,
                record.variant_id or "base",
                record.family_id,
                record.surface.value,
                record.hints_seen,
                record.solution_exposed,
                record.answer_revealed,
            )
            for record in exposure_state.exposures
        ]
        exposure_order: list[tuple[str, int, str]] = []
        latest_exposures: dict[tuple[str, int, str], tuple[Any, ...]] = {}
        monotonic_exposures = True
        for row in exposure_rows:
            key = (row.item_id, row.item_revision, row.variant_id)
            if key not in latest_exposures:
                exposure_order.append(key)
            transition = (
                row.item_id,
                row.item_revision,
                row.variant_id,
                row.family_id,
                row.surface,
                row.hint_level,
                row.solution_exposed,
                row.answer_revealed,
            )
            previous = latest_exposures.get(key)
            if previous is not None:
                monotonic_exposures = monotonic_exposures and (
                    transition[3:5] == previous[3:5]
                    and int(transition[5]) >= int(previous[5])
                    and (not bool(previous[6]) or bool(transition[6]))
                    and (not bool(previous[7]) or bool(transition[7]))
                )
            latest_exposures[key] = transition
        actual_exposures = [latest_exposures[key] for key in exposure_order]
        if (
            [row.exposure_sequence for row in exposure_rows]
            != list(range(len(exposure_rows)))
            or not monotonic_exposures
            or actual_exposures != expected_exposures
        ):
            raise DurableLedgerMismatch(
                "exposure_integrity_failures",
                "append-only exposure ledger and checkpoint exposure state disagree",
            )

        expected_events = [
            EvidenceEvent.model_validate(payload).model_dump(mode="json")
            for payload in state.get("events", [])
        ]
        actual_events = [
            cls._evidence_payload(row) for row in evidence_rows
        ]
        if actual_events != expected_events:
            raise DurableLedgerMismatch(
                "missing_evidence_detected",
                "append-only evidence log and checkpoint learner state disagree",
            )
        episode_id = state.get("episode_id")
        pinned_catalog_version = state.get("pedagogy_catalog_version")
        for event in actual_events:
            event_catalog_version = event["pedagogy_catalog_version"]
            content_catalog_version = event["content_versions"].get(
                "pedagogy_catalog"
            )
            if (
                event_catalog_version != "legacy"
                and content_catalog_version != event_catalog_version
            ):
                raise DurableLedgerMismatch(
                    "evidence_provenance_integrity_failures",
                    "evidence row and content provenance name different pedagogy catalogs",
                )
            if (
                event["episode_id"] == episode_id
                and event_catalog_version != pinned_catalog_version
            ):
                raise DurableLedgerMismatch(
                    "evidence_provenance_integrity_failures",
                    "current-episode evidence does not use the checkpoint pedagogy catalog",
                )

        widget_receipts = [
            receipt
            for receipt in action_receipts
            if receipt.request_payload.get("type") == "widget_attempt"
        ]
        expected_widget_keys = [
            str(receipt.request_payload.get("pending_key"))
            for receipt in widget_receipts
        ]
        actual_widget_keys = [row.interaction_key for row in widget_rows]
        attempts_by_key: dict[str, int] = {}
        contiguous = True
        for row in widget_rows:
            next_attempt = attempts_by_key.get(row.interaction_key, 0) + 1
            contiguous = contiguous and row.attempt_number == next_attempt
            attempts_by_key[row.interaction_key] = next_attempt
        if actual_widget_keys != expected_widget_keys or not contiguous:
            raise DurableLedgerMismatch(
                "widget_attempt_integrity_failures",
                "widget attempt trajectory and mutation receipts disagree",
            )

        return {
            "transcript_entries": len(actual_transcript),
            "evidence_events": len(actual_events),
            "exposure_transitions": len(exposure_rows),
            "exposed_items": len(actual_exposures),
            "widget_attempts": len(widget_rows),
            "action_receipts": len(action_receipts),
        }

    @staticmethod
    def _evidence_payload(row: EvidenceEventRow) -> dict[str, Any]:
        return EvidenceEvent(
            event_id=row.event_id,
            learner_id=row.learner_id,
            t=_aware(row.t),
            item_id=row.item_id,
            kc_ids=list(row.kc_ids),
            correct=row.correct,
            response_class=row.response_class,
            hints_used=row.hints_used,
            assisted=row.assisted,
            misconception_id=row.misconception_id,
            content_versions=dict(row.content_versions or {}),
            pedagogy_catalog_version=row.pedagogy_catalog_version,
            episode_id=row.episode_id,
            family_id=row.family_id,
            surface=row.surface,
            item_revision=row.item_revision,
            attempt_number=row.attempt_number,
            policy_version=row.policy_version,
            learner_params_version=row.learner_params_version,
            content_provenance=row.content_provenance,
            learning_opportunity=row.learning_opportunity,
        ).model_dump(mode="json")

    @staticmethod
    def _checkpoint(handle: V2SessionHandle, view: SessionView) -> dict[str, Any]:
        export = getattr(handle.orchestrator, "export_checkpoint", None)
        orchestrator_state = export() if callable(export) else None
        if not isinstance(orchestrator_state, dict):
            raise ValueError("v2 orchestrator cannot export a durable checkpoint")
        catalog_version = V2PersistenceService._catalog_version(handle)
        content_release = {
            "graph_version": orchestrator_state.get("graph_version"),
            "item_bank_version": orchestrator_state.get("item_bank_version"),
            "pedagogy_catalog_version": orchestrator_state.get(
                "pedagogy_catalog_version"
            ),
            "release_id": orchestrator_state.get("release_id"),
            "release_digest": orchestrator_state.get("release_digest"),
        }
        payload = {
            "schema_version": 4,
            "content_release": content_release,
            "session_view": view.model_dump(mode="json"),
            "orchestrator": orchestrator_state,
            "goal": handle.goal.model_dump(mode="json"),
            "profile": handle.profile.model_dump(mode="json"),
            "context": handle.context,
            "content_mode": handle.content_mode.model_dump(mode="json"),
            "learner_id": handle.learner_id,
        }
        V2PersistenceService._validate_checkpoint_release_payload(
            payload,
            stored_catalog_version=catalog_version,
            expected_catalog_version=catalog_version,
        )
        return payload

    @staticmethod
    def _validate_checkpoint_release_pin(
        checkpoint: SessionCheckpointRow,
        *,
        expected_catalog_version: str | None = None,
    ) -> None:
        """Require one immutable content triple across every checkpoint layer."""
        V2PersistenceService._validate_checkpoint_release_payload(
            checkpoint.checkpoint,
            stored_catalog_version=checkpoint.pedagogy_catalog_version,
            expected_catalog_version=expected_catalog_version,
        )

    @staticmethod
    def _validate_checkpoint_release_payload(
        payload: Any,
        *,
        stored_catalog_version: str,
        expected_catalog_version: str | None = None,
    ) -> None:
        """Validate a checkpoint before insert as well as after durable read."""
        if not isinstance(payload, dict) or payload.get("schema_version") not in {
            3,
            4,
        }:
            raise DurableLedgerMismatch(
                "checkpoint_integrity_failures",
                "durable checkpoint has no supported content-release pin",
            )
        content_release = payload.get("content_release")
        state = payload.get("orchestrator")
        coordinate_keys = {
            "graph_version",
            "item_bank_version",
            "pedagogy_catalog_version",
        }
        identity_keys = {"release_id", "release_digest"}
        expected_keys = (
            coordinate_keys | identity_keys
            if payload["schema_version"] == 4
            else coordinate_keys
        )
        if (
            not isinstance(content_release, dict)
            or set(content_release) != expected_keys
            or not isinstance(state, dict)
        ):
            raise DurableLedgerMismatch(
                "checkpoint_integrity_failures",
                "durable checkpoint has an invalid content-release pin",
            )
        state_release = {key: state.get(key) for key in expected_keys}
        catalog_version = content_release["pedagogy_catalog_version"]
        view = payload.get("session_view")
        identity_invalid = False
        if payload["schema_version"] == 4:
            release_id = content_release["release_id"]
            release_digest = content_release["release_digest"]
            identity_invalid = (
                not isinstance(release_id, str)
                or not release_id
                or len(release_id) > 128
                or not isinstance(release_digest, str)
                or len(release_digest) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in release_digest
                )
                or not isinstance(view, dict)
                or view.get("release_id") != release_id
                or view.get("release_digest") != release_digest
            )
        if (
            content_release != state_release
            or identity_invalid
            or not isinstance(content_release["graph_version"], int)
            or isinstance(content_release["graph_version"], bool)
            or not isinstance(content_release["item_bank_version"], str)
            or not content_release["item_bank_version"]
            or not isinstance(catalog_version, str)
            or not catalog_version
            or catalog_version == "legacy"
            or stored_catalog_version != catalog_version
            or (
                expected_catalog_version is not None
                and expected_catalog_version != catalog_version
            )
        ):
            raise DurableLedgerMismatch(
                "checkpoint_integrity_failures",
                "durable checkpoint content-release pins disagree",
            )

    @staticmethod
    def _view_from_create_receipt(
        handle: V2SessionHandle, request_id: UUID
    ) -> SessionView:
        receipt = handle.receipts.get(str(request_id))
        if receipt is None:
            raise RuntimeError("initial session receipt is missing")
        return receipt.response

    @staticmethod
    def _view_from_checkpoint(row: SessionCheckpointRow) -> SessionView | None:
        payload = (row.checkpoint or {}).get("session_view")
        return SessionView.model_validate(payload) if payload else None

    @staticmethod
    def _transcript_row(
        session_id: str, entry: TranscriptEntry
    ) -> TranscriptEntryRow:
        return TranscriptEntryRow(
            session_id=session_id,
            sequence=entry.sequence,
            entry=entry.model_dump(mode="json"),
        )

    @staticmethod
    def _evidence_row(event: Any, handle: V2SessionHandle) -> EvidenceEventRow:
        return EvidenceEventRow(
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
            pedagogy_catalog_version=event.pedagogy_catalog_version,
            episode_id=event.episode_id or handle.session_id,
            family_id=event.family_id,
            surface=event.surface,
            item_revision=event.item_revision,
            attempt_number=event.attempt_number,
            policy_version=event.policy_version,
            learner_params_version=event.learner_params_version,
            content_provenance=event.content_provenance,
            learning_opportunity=event.learning_opportunity,
        )

    @staticmethod
    def _catalog_version(handle: V2SessionHandle) -> str:
        version = getattr(
            handle.orchestrator,
            "pedagogy_catalog_version",
            None,
        )
        if (
            not isinstance(version, str)
            or not version
            or version == "legacy"
        ):
            raise ValueError(
                "v2 orchestrator lacks a trusted pedagogy-catalog version"
            )
        return version

    @staticmethod
    def _exposure_row(
        session_id: str, sequence: int, exposure: Any
    ) -> ItemExposureRow:
        if hasattr(exposure, "model_dump"):
            data = exposure.model_dump(mode="json")
        elif isinstance(exposure, dict):
            data = exposure
        else:
            data = vars(exposure)
        surface = data.get("surface", "legacy")
        surface = getattr(surface, "value", surface)
        hints = data.get(
            "hints_seen", data.get("hints_exposed", data.get("hint_level", 0))
        )
        if isinstance(hints, (list, tuple, set)):
            hints = len(hints)
        return ItemExposureRow(
            session_id=session_id,
            item_id=str(data["item_id"]),
            item_revision=int(data.get("revision", 1)),
            variant_id=str(data.get("variant_id") or "base"),
            family_id=str(data["family_id"]),
            surface=str(surface),
            exposure_sequence=sequence,
            solution_exposed=bool(data.get("solution_exposed", False)),
            hint_level=int(hints),
            answer_revealed=bool(data.get("answer_revealed", False)),
        )
