"""Versioned, reviewed assessment content and exposure-state schemas.

The item bank is deliberately separate from KC ``canonical_examples``.  Those
examples are explanatory graph metadata; only items in a validated bank may be
used as mastery-bearing content.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from tutor.schemas.common import ReviewStatus
from tutor.schemas.kc import KC_ID_PATTERN

_CONTENT_ID_PATTERN = r"^[a-z0-9][a-z0-9._-]*$"
_VARIABLE_PATTERN = r"^[A-Za-z][A-Za-z0-9]*$"


class StrictFrozenModel(BaseModel):
    """Base for content contracts: reject drift and make snapshots immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class AssessmentSurface(StrEnum):
    """The mutually isolated places where an authored item may appear."""

    DIAGNOSTIC = "diagnostic"
    CHECKIN = "checkin"
    GUIDED_WIDGET = "guided_widget"
    CAPSTONE = "capstone"
    WORKED_EXAMPLE = "worked_example"


class TextPromptSegment(StrictFrozenModel):
    """Learner-visible prose."""

    kind: Literal["text"] = "text"
    text: str = Field(min_length=1)


class MathPromptSegment(StrictFrozenModel):
    """Learner-visible ASCII math, kept distinct from prose."""

    kind: Literal["math"] = "math"
    expression: str = Field(min_length=1)


class BlankPromptSegment(StrictFrozenModel):
    """One answer location in an assessment prompt."""

    kind: Literal["blank"] = "blank"
    label: str | None = None


PromptSegment = Annotated[
    Union[TextPromptSegment, MathPromptSegment, BlankPromptSegment],
    Field(discriminator="kind"),
]
prompt_segment_adapter: TypeAdapter[PromptSegment] = TypeAdapter(PromptSegment)


class SymbolicAnswerSpec(StrictFrozenModel):
    """A symbolic expression under an explicit variable/function vocabulary."""

    kind: Literal["symbolic"] = "symbolic"
    expected: str = Field(min_length=1)
    variables: list[str] = Field(default_factory=list)
    functions: list[
        Literal["sin", "cos", "tan", "sec", "csc", "cot", "exp", "log", "ln", "sqrt", "Abs"]
    ] = Field(default_factory=list)
    assignment_lhs: str | None = Field(default=None, pattern=r"^[A-Za-z][A-Za-z0-9]*(?:\([A-Za-z][A-Za-z0-9]*\))?$")

    @model_validator(mode="after")
    def _unique_names(self) -> "SymbolicAnswerSpec":
        if len(self.variables) != len(set(self.variables)):
            raise ValueError("symbolic variables must be unique")
        if len(self.functions) != len(set(self.functions)):
            raise ValueError("symbolic functions must be unique")
        return self


class NumericAnswerSpec(StrictFrozenModel):
    """An exact or tolerance-based finite real number."""

    kind: Literal["numeric"] = "numeric"
    expected: str = Field(min_length=1)
    tolerance: float = Field(default=1e-6, ge=0, allow_inf_nan=False)


class FiniteSetAnswerSpec(StrictFrozenModel):
    """An unordered finite set of symbolic values."""

    kind: Literal["finite_set"] = "finite_set"
    expected: list[str] = Field(min_length=1)
    variables: list[str] = Field(default_factory=list)
    functions: list[
        Literal["sin", "cos", "tan", "sec", "csc", "cot", "exp", "log", "ln", "sqrt", "Abs"]
    ] = Field(default_factory=list)


class IntervalSpec(StrictFrozenModel):
    """One real interval, using ``-inf``/``inf`` for unbounded endpoints."""

    lower: str = Field(min_length=1)
    upper: str = Field(min_length=1)
    lower_closed: bool = False
    upper_closed: bool = False

    @model_validator(mode="after")
    def _infinite_endpoints_are_open(self) -> "IntervalSpec":
        if self.lower.strip().lower() in {"-inf", "-infinity"} and self.lower_closed:
            raise ValueError("an interval cannot be closed at -infinity")
        if self.upper.strip().lower() in {"inf", "+inf", "infinity", "+infinity"} and self.upper_closed:
            raise ValueError("an interval cannot be closed at infinity")
        return self


class IntervalSetAnswerSpec(StrictFrozenModel):
    """A union of one or more real intervals."""

    kind: Literal["interval_set"] = "interval_set"
    expected: list[IntervalSpec] = Field(min_length=1)


class OrderedTupleAnswerSpec(StrictFrozenModel):
    """An ordered tuple whose entries are checked symbolically."""

    kind: Literal["ordered_tuple"] = "ordered_tuple"
    expected: list[str] = Field(min_length=1)
    variables: list[str] = Field(default_factory=list)
    functions: list[
        Literal["sin", "cos", "tan", "sec", "csc", "cot", "exp", "log", "ln", "sqrt", "Abs"]
    ] = Field(default_factory=list)


class AntiderivativeAnswerSpec(StrictFrozenModel):
    """An antiderivative family; answers may differ by a constant."""

    kind: Literal["antiderivative"] = "antiderivative"
    expected: str = Field(min_length=1)
    variable: str = Field(default="x", pattern=_VARIABLE_PATTERN)
    variables: list[str] = Field(default_factory=list)
    functions: list[
        Literal["sin", "cos", "tan", "sec", "csc", "cot", "exp", "log", "ln", "sqrt", "Abs"]
    ] = Field(default_factory=list)


class ChoiceAnswerSpec(StrictFrozenModel):
    """A choice token; display labels belong in the prompt or widget config."""

    kind: Literal["choice"] = "choice"
    option_ids: list[str] = Field(min_length=2)
    expected_choice_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _expected_choice_exists(self) -> "ChoiceAnswerSpec":
        if len(self.option_ids) != len(set(self.option_ids)):
            raise ValueError("choice option ids must be unique")
        if self.expected_choice_id not in self.option_ids:
            raise ValueError("expected_choice_id must occur in option_ids")
        return self


AnswerSpec = Annotated[
    Union[
        SymbolicAnswerSpec,
        NumericAnswerSpec,
        FiniteSetAnswerSpec,
        IntervalSetAnswerSpec,
        OrderedTupleAnswerSpec,
        AntiderivativeAnswerSpec,
        ChoiceAnswerSpec,
    ],
    Field(discriminator="kind"),
]
answer_spec_adapter: TypeAdapter[AnswerSpec] = TypeAdapter(AnswerSpec)


class AssessmentHint(StrictFrozenModel):
    """One ordered hint; revealing hints disqualify the family from assessment."""

    text: str = Field(min_length=1)
    revealing: bool = False


class ErrorSignature(StrictFrozenModel):
    """Reviewed deterministic mapping from a wrong form to a suspected cause."""

    expected_wrong: str = Field(min_length=1)
    misconception_id: str | None = Field(default=None, min_length=1)
    implicated_prereq: str | None = Field(default=None, pattern=KC_ID_PATTERN)

    @model_validator(mode="after")
    def _has_consequence(self) -> "ErrorSignature":
        if self.misconception_id is None and self.implicated_prereq is None:
            raise ValueError("an error signature must name a misconception or prerequisite")
        return self


class AssessmentProvenance(StrictFrozenModel):
    """Authorship and review facts for trusted content."""

    # ``source`` is copied into EvidenceEvent.content_provenance, whose durable
    # column is VARCHAR(128). Reject an unrecoverable release at validation time.
    source: str = Field(min_length=1, max_length=128)
    author: str = Field(min_length=1)
    reviewed_by: str | None = Field(default=None, min_length=1)
    reviewed_at: datetime | None = None


class AssessmentItem(StrictFrozenModel):
    """One stable, independently authored assessment or instruction item."""

    item_id: str = Field(max_length=128, pattern=_CONTENT_ID_PATTERN)
    revision: int = Field(default=1, ge=1)
    family_id: str = Field(max_length=128, pattern=_CONTENT_ID_PATTERN)
    kc_id: str = Field(pattern=KC_ID_PATTERN)
    difficulty: Literal["foundation", "core", "stretch"] = "core"
    eligible_surfaces: list[AssessmentSurface] = Field(min_length=1)
    prompt: list[PromptSegment] = Field(min_length=1)
    hints: list[AssessmentHint] = Field(min_length=3, max_length=3)
    answer: AnswerSpec
    review_status: ReviewStatus
    provenance: AssessmentProvenance
    error_signatures: list[ErrorSignature] = Field(default_factory=list)

    @model_validator(mode="after")
    def _content_invariants(self) -> "AssessmentItem":
        if len(self.eligible_surfaces) != len(set(self.eligible_surfaces)):
            raise ValueError("eligible_surfaces must be unique")
        blanks = sum(isinstance(segment, BlankPromptSegment) for segment in self.prompt)
        if self.eligible_surfaces == [AssessmentSurface.WORKED_EXAMPLE]:
            if blanks > 1:
                raise ValueError("a worked example may contain at most one blank")
        elif blanks != 1:
            raise ValueError("an assessment item must contain exactly one blank segment")
        if any(hint.revealing for hint in self.hints[:2]):
            raise ValueError("the first two hints must be non-revealing")
        if not self.hints[2].revealing:
            raise ValueError("the final hint must be revealing")
        return self


class ItemBankDocument(StrictFrozenModel):
    """A complete item-bank release pinned to one KC graph version."""

    schema_version: Literal[2] = 2
    bank_version: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    graph_version: int = Field(ge=1)
    # A bank may carry draft inventory without releasing any KC. Runtime
    # sessions admit only the explicitly released, semantically validated
    # hard-ancestor closure.
    released_kcs: list[str] = Field(default_factory=list)
    items: list[AssessmentItem] = Field(min_length=1)

    @model_validator(mode="after")
    def _identity_invariants(self) -> "ItemBankDocument":
        if len(self.released_kcs) != len(set(self.released_kcs)):
            raise ValueError("released_kcs must be unique")
        identities = [(item.item_id, item.revision) for item in self.items]
        if len(identities) != len(set(identities)):
            raise ValueError("item_id/revision pairs must be unique")
        family_kcs: dict[str, str] = {}
        item_lineages: dict[
            str,
            tuple[str, str, frozenset[AssessmentSurface]],
        ] = {}
        for item in self.items:
            previous = family_kcs.setdefault(item.family_id, item.kc_id)
            if previous != item.kc_id:
                raise ValueError(
                    f"family {item.family_id!r} spans KCs {previous!r} and {item.kc_id!r}"
                )
            lineage = (
                item.kc_id,
                item.family_id,
                frozenset(item.eligible_surfaces),
            )
            previous_lineage = item_lineages.setdefault(item.item_id, lineage)
            if previous_lineage != lineage:
                raise ValueError(
                    f"revisions of item {item.item_id!r} must retain the same "
                    "KC, family, and eligible surfaces"
                )
        return self


class ItemReservation(StrictFrozenModel):
    """A stable item reference reserved before learner-visible generation."""

    item_id: str = Field(min_length=1, max_length=128)
    revision: int = Field(ge=1)
    family_id: str = Field(min_length=1, max_length=128)
    kc_id: str = Field(pattern=KC_ID_PATTERN)
    surface: AssessmentSurface
    variant_id: str | None = Field(default=None, max_length=128)


class ExposureRecord(ItemReservation):
    """What was actually shown for a reservation."""

    hints_seen: int = Field(default=0, ge=0, le=3)
    solution_exposed: bool = False
    answer_revealed: bool = False


class ContentExposureState(StrictFrozenModel):
    """Append-only reservation/exposure ledger stored in an episode checkpoint."""

    reservations: list[ItemReservation] = Field(default_factory=list)
    exposures: list[ExposureRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_reservations(self) -> "ContentExposureState":
        keys = [(record.item_id, record.revision) for record in self.reservations]
        if len(keys) != len(set(keys)):
            raise ValueError("the same item revision cannot be reserved twice")
        families = [record.family_id for record in self.reservations]
        if len(families) != len(set(families)):
            raise ValueError("an assessment family cannot be reserved twice")
        exposure_keys = [(record.item_id, record.revision) for record in self.exposures]
        if len(exposure_keys) != len(set(exposure_keys)):
            raise ValueError("the same item revision cannot be exposed twice")
        if not set(exposure_keys) <= set(keys):
            raise ValueError("every exposure must have a corresponding reservation")
        return self

    @property
    def used_family_ids(self) -> frozenset[str]:
        """All families already committed to this episode, shown or not."""
        return frozenset(record.family_id for record in self.reservations)

    @property
    def retired_family_ids(self) -> frozenset[str]:
        """Families made ineligible by a solution or revealing hint."""
        return frozenset(
            record.family_id
            for record in self.exposures
            if record.solution_exposed or record.answer_revealed or record.hints_seen >= 3
        )


class LessonBundleReservation(StrictFrozenModel):
    """Disjoint content reserved atomically for one KC lesson."""

    worked_example: ItemReservation
    guided_widget: ItemReservation
    checkins: list[ItemReservation] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def _coherent_bundle(self) -> "LessonBundleReservation":
        reservations = [
            self.worked_example,
            self.guided_widget,
            *self.checkins,
        ]
        if len({item.kc_id for item in reservations}) != 1:
            raise ValueError("every lesson-bundle item must measure the same KC")
        expected_surfaces = [
            AssessmentSurface.WORKED_EXAMPLE,
            AssessmentSurface.GUIDED_WIDGET,
            AssessmentSurface.CHECKIN,
            AssessmentSurface.CHECKIN,
            AssessmentSurface.CHECKIN,
        ]
        if [item.surface for item in reservations] != expected_surfaces:
            raise ValueError("lesson-bundle items have invalid surfaces")
        if len({item.family_id for item in reservations}) != len(reservations):
            raise ValueError("lesson-bundle families must be disjoint")
        return self
