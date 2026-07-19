"""Pedagogy packs: cached, per-KC misconceptions, metaphors, and error patterns.

Packs are data assets built offline (retrieval + LLM + human review) — never
fetched live during a tutoring session.
"""

from pydantic import BaseModel, Field

from tutor.schemas.common import ReviewStatus, WidgetType
from tutor.schemas.kc import KC_ID_PATTERN


class Misconception(BaseModel):
    """A known, named student misconception with a detectable error signature."""

    id: str = Field(pattern=r"^m\.[a-z0-9_.]+$")
    description: str = Field(min_length=1)
    error_signature: str = Field(min_length=1)
    remediation_hint: str = Field(min_length=1)


class Metaphor(BaseModel):
    """A teaching metaphor and the widget types it renders well with."""

    id: str = Field(pattern=r"^met\.[a-z0-9_.]+$")
    description: str = Field(min_length=1)
    widget_affinity: list[WidgetType] = Field(min_length=1)


class PedagogyPack(BaseModel):
    """The complete cached pedagogy asset for one KC."""

    kc_id: str = Field(pattern=KC_ID_PATTERN)
    misconceptions: list[Misconception] = Field(default_factory=list)
    metaphors: list[Metaphor] = Field(default_factory=list)
    error_patterns: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    review_status: ReviewStatus = ReviewStatus.DRAFT
    version: int = 1
