"""Learner model schemas.

The evidence log is append-only and authoritative: EvidenceEvent is frozen
(immutable). Mastery is derived state, rebuildable by replaying the log, and
keeps direct evidence separate from graph-inferred evidence.
"""

from datetime import datetime
from typing import Literal
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
    item_id: str = Field(min_length=1, max_length=128)
    kc_ids: list[str] = Field(min_length=1)
    correct: bool
    response_class: ResponseClass
    hints_used: int = Field(ge=0, default=0)
    assisted: bool = False
    misconception_id: str | None = Field(default=None, max_length=128)
    content_versions: dict[str, str] = Field(default_factory=dict)
    pedagogy_catalog_version: str = Field(
        default="legacy", min_length=1, max_length=128
    )
    episode_id: str | None = Field(default=None, max_length=36)
    family_id: str | None = Field(default=None, max_length=128)
    surface: Literal[
        "diagnostic",
        "guided_widget",
        "checkin",
        "capstone",
        "worked_example",
        "instructional_practice",
        "legacy",
    ] = "legacy"
    item_revision: int = Field(ge=1, default=1)
    attempt_number: int = Field(ge=1, default=1)
    policy_version: str = Field(default="legacy", min_length=1, max_length=64)
    learner_params_version: str = Field(default="v1", min_length=1, max_length=64)
    content_provenance: str = Field(default="legacy", min_length=1, max_length=128)
    learning_opportunity: bool = False


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
