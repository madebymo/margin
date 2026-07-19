"""Mini-lesson package: the fully version-pinned unit of teachable content."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from tutor.schemas.common import ResponseClass, ReviewStatus, VersionPins
from tutor.schemas.kc import KC_ID_PATTERN
from tutor.schemas.widgets import WidgetConfig


class CheckinOption(BaseModel):
    """A multiple-choice option; distractors map to known misconceptions."""

    text: str = Field(min_length=1)
    misconception_id: str | None = None
    rationale: str | None = None


class CheckinItem(BaseModel):
    """A check-in question, usually a near-transfer variation of the last widget."""

    item_id: str = Field(min_length=1)
    stem: str = Field(min_length=1)
    kc_id: str = Field(pattern=KC_ID_PATTERN)
    response_class: ResponseClass
    options: list[CheckinOption] | None = None
    answer: str = Field(min_length=1)
    variation_of: str | None = None
    interleaved_from: str | None = None

    @model_validator(mode="after")
    def _mc_requires_options(self) -> "CheckinItem":
        if self.response_class == ResponseClass.MULTIPLE_CHOICE:
            if self.options is None or len(self.options) < 2:
                raise ValueError("multiple_choice check-ins require at least 2 options")
        return self


class AnswerSemantics(BaseModel):
    """How answers for this lesson's math are judged."""

    equivalence: Literal["sympy_equiv", "numeric"]
    tolerance: float | None = Field(default=None, ge=0)


class MathSpec(BaseModel):
    """Canonical mathematical content of the lesson, in restricted parseable form."""

    canonical_form: str = Field(min_length=1)
    variables: list[str] = Field(default_factory=list)
    assumptions_domains: str | None = None
    answer_semantics: AnswerSemantics


class Applicability(BaseModel):
    """Which learner profile band / difficulty this package targets (cache key)."""

    profile_band: str = Field(min_length=1)
    difficulty: str = Field(min_length=1)


class EntryExit(BaseModel):
    """Entry conditions and the mastery exit criterion for the lesson."""

    entry: str | None = None
    exit_consecutive_correct: int = Field(default=2, ge=1)


class Provenance(BaseModel):
    """Who/what generated this package and its review state."""

    generator: str = Field(min_length=1)
    review_status: ReviewStatus = ReviewStatus.DRAFT
    telemetry_id: str = Field(min_length=1)


class MiniLessonPackage(BaseModel):
    """One teachable unit for one KC: narrative, widgets, check-ins, fallback."""

    kc_id: str = Field(pattern=KC_ID_PATTERN)
    objective: str = Field(min_length=1)
    prerequisite_kcs: list[str] = Field(default_factory=list)
    metaphor_id: str | None = None
    versions: VersionPins
    narrative: str = Field(min_length=1)
    widgets: list[WidgetConfig] = Field(min_length=1)
    checkins: list[CheckinItem] = Field(min_length=1)
    math: MathSpec
    hint_ladder: list[str] = Field(min_length=3, max_length=3)
    text_fallback: str = Field(min_length=1)
    applicability: Applicability
    entry_exit: EntryExit = Field(default_factory=EntryExit)
    provenance: Provenance
