"""SQLAlchemy 2.0 ORM models for the Phase 0 data layer.

Conventions:
- evidence_events is append-only: no updated_at column and no update path.
- UUIDs are stored as String(36) for cross-database compatibility.
- Row classes carry a ``Row`` suffix to distinguish them from Pydantic schemas.
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from tutor.db.base import Base, JSONVariant


def utcnow() -> datetime:
    """Timezone-aware now() used for column defaults."""
    return datetime.now(timezone.utc)


class GraphVersionRow(Base):
    """A published or draft version of the KC graph."""

    __tablename__ = "graph_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version: Mapped[int] = mapped_column(Integer, unique=True, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="draft")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )


class KCNodeRow(Base):
    """A KC node within one graph version."""

    __tablename__ = "kc_nodes"
    __table_args__ = (UniqueConstraint("graph_version_id", "kc_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    graph_version_id: Mapped[int] = mapped_column(
        ForeignKey("graph_versions.id"), nullable=False
    )
    kc_id: Mapped[str] = mapped_column(String(128), nullable=False)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    course_level: Mapped[str] = mapped_column(String(64), nullable=False)
    canonical_examples: Mapped[list] = mapped_column(JSONVariant, nullable=False)


class KCEdgeRow(Base):
    """A prerequisite edge within one graph version."""

    __tablename__ = "kc_edges"
    __table_args__ = (
        UniqueConstraint("graph_version_id", "from_kc", "to_kc"),
        CheckConstraint("from_kc != to_kc", name="no_self_loop"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    graph_version_id: Mapped[int] = mapped_column(
        ForeignKey("graph_versions.id"), nullable=False
    )
    from_kc: Mapped[str] = mapped_column(String(128), nullable=False)
    to_kc: Mapped[str] = mapped_column(String(128), nullable=False)
    type: Mapped[str] = mapped_column(String(8), nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)


class PedagogyPackRow(Base):
    """Cached pedagogy pack content for one KC (JSON payload is the pack)."""

    __tablename__ = "pedagogy_packs"
    __table_args__ = (UniqueConstraint("graph_version_id", "kc_id", "version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    graph_version_id: Mapped[int] = mapped_column(
        ForeignKey("graph_versions.id"), nullable=False
    )
    kc_id: Mapped[str] = mapped_column(String(128), nullable=False)
    content: Mapped[dict] = mapped_column(JSONVariant, nullable=False)
    review_status: Mapped[str] = mapped_column(String(32), nullable=False, default="draft")
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)


class LearnerRow(Base):
    """A learner's internal identity. Never keyed by resume token."""

    __tablename__ = "learners"

    learner_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    profile: Mapped[dict] = mapped_column(JSONVariant, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )


class ResumeTokenRow(Base):
    """A hashed, revocable, expiring token that resolves to a learner."""

    __tablename__ = "resume_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    learner_id: Mapped[str] = mapped_column(ForeignKey("learners.learner_id"), nullable=False)
    token_hash: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )


class EvidenceEventRow(Base):
    """Append-only evidence log. No updated_at by design — events are immutable."""

    __tablename__ = "evidence_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False)
    learner_id: Mapped[str] = mapped_column(ForeignKey("learners.learner_id"), nullable=False)
    t: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    item_id: Mapped[str] = mapped_column(String(128), nullable=False)
    kc_ids: Mapped[list] = mapped_column(JSONVariant, nullable=False)
    correct: Mapped[bool] = mapped_column(Boolean, nullable=False)
    response_class: Mapped[str] = mapped_column(String(32), nullable=False)
    hints_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    assisted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    misconception_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    content_versions: Mapped[dict] = mapped_column(JSONVariant, nullable=False, default=dict)


class DerivedMasteryRow(Base):
    """Derived mastery per (learner, KC): direct vs. inferred, with observation count."""

    __tablename__ = "derived_mastery"
    __table_args__ = (UniqueConstraint("learner_id", "kc_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    learner_id: Mapped[str] = mapped_column(ForeignKey("learners.learner_id"), nullable=False)
    kc_id: Mapped[str] = mapped_column(String(128), nullable=False)
    direct: Mapped[float] = mapped_column(Float, nullable=False)
    inferred: Mapped[float] = mapped_column(Float, nullable=False)
    observations: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_practiced: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    params_version: Mapped[int] = mapped_column(Integer, nullable=False)
    graph_version: Mapped[int] = mapped_column(Integer, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow
    )


class EpisodeRow(Base):
    """A teaching episode and its routing envelope (budgets, retries, resume stack)."""

    __tablename__ = "episodes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    learner_id: Mapped[str] = mapped_column(ForeignKey("learners.learner_id"), nullable=False)
    target_kc: Mapped[str] = mapped_column(String(128), nullable=False)
    state: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    envelope: Mapped[dict] = mapped_column(JSONVariant, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow
    )


class GenerationJobRow(Base):
    """Queue-shaped generation job record (same-process worker in v1)."""

    __tablename__ = "generation_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    inputs: Mapped[dict] = mapped_column(JSONVariant, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    result: Mapped[dict | None] = mapped_column(JSONVariant, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow
    )


class MiniLessonRow(Base):
    """Cached generated mini-lesson package, version-pinned via its JSON payload."""

    __tablename__ = "mini_lessons"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kc_id: Mapped[str] = mapped_column(String(128), nullable=False)
    applicability: Mapped[dict] = mapped_column(JSONVariant, nullable=False)
    versions: Mapped[dict] = mapped_column(JSONVariant, nullable=False)
    package: Mapped[dict] = mapped_column(JSONVariant, nullable=False)
    telemetry_id: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )
