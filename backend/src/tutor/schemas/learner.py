"""Learner model schemas.

The evidence log is append-only and authoritative: EvidenceEvent is frozen
(immutable). Mastery is derived state, rebuildable by replaying the log, and
keeps direct evidence separate from graph-inferred evidence.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from tutor.schemas.common import ResponseClass


class LearnerProfile(BaseModel):
    """Coarse course/age context collected at intake; conditions all content."""

    course: str = Field(min_length=1)
    age_band: str = Field(min_length=1)


class EvidenceEvent(BaseModel):
    """One immutable observation of learner behavior (probe, check-in, or widget)."""

    model_config = ConfigDict(frozen=True)

    event_id: UUID
    learner_id: UUID
    t: datetime
    item_id: str = Field(min_length=1)
    kc_ids: list[str] = Field(min_length=1)
    correct: bool
    response_class: ResponseClass
    hints_used: int = Field(ge=0, default=0)
    assisted: bool = False
    misconception_id: str | None = None
    content_versions: dict[str, str] = Field(default_factory=dict)


class MasteryEstimate(BaseModel):
    """Derived belief for one KC: direct vs. inferred evidence, never merged."""

    direct: float = Field(ge=0.0, le=1.0)
    inferred: float = Field(ge=0.0, le=1.0)
    observations: int = Field(ge=0, default=0)
    last_practiced: datetime | None = None


class DerivedLearnerState(BaseModel):
    """Snapshot of derived mastery, rebuildable from the evidence log."""

    learner_id: UUID
    graph_version: int = Field(ge=1)
    params_version: int = Field(ge=1)
    mastery: dict[str, MasteryEstimate] = Field(default_factory=dict)
    misconception_flags: list[str] = Field(default_factory=list)
