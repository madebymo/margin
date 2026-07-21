"""Versioned, reviewed assessment content and exposure-state schemas.

The item bank is deliberately separate from KC ``canonical_examples``.  Those
examples are explanatory graph metadata; only items in a validated bank may be
used as mastery-bearing content.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

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


class AssessmentTaskKind(StrEnum):
    """Whether the learner derives a result or rewrites a supplied expression."""

    SOLVE = "solve"
    TRANSFORM = "transform"


class PromptSemanticRole(StrEnum):
    """Why a structured prompt segment is visible to the learner."""

    INSTRUCTION = "instruction"
    CONTEXT = "context"
    GIVEN = "given"
    RESPONSE = "response"
    WORKED_STEP = "worked_step"
    WORKED_ANSWER = "worked_answer"


class TextPromptSegment(StrictFrozenModel):
    """Learner-visible prose."""

    kind: Literal["text"] = "text"
    # Defaults preserve checkpoints written before semantic roles were added.
    role: PromptSemanticRole = PromptSemanticRole.INSTRUCTION
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def _valid_role(self) -> "TextPromptSegment":
        if self.role not in {
            PromptSemanticRole.INSTRUCTION,
            PromptSemanticRole.CONTEXT,
            PromptSemanticRole.WORKED_STEP,
        }:
            raise ValueError(f"text segments cannot have role {self.role.value!r}")
        return self


class MathPromptSegment(StrictFrozenModel):
    """Learner-visible ASCII math, kept distinct from prose."""

    kind: Literal["math"] = "math"
    role: PromptSemanticRole = PromptSemanticRole.GIVEN
    expression: str = Field(min_length=1)
    # ``None`` keeps schema-v2 checkpoints readable. Schema-v3 banks require
    # reviewed speech text for every learner-visible math segment.
    spoken_text: str | None = Field(default=None, min_length=1)

    @field_validator("spoken_text", mode="before")
    @classmethod
    def _normalize_spoken_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _valid_role(self) -> "MathPromptSegment":
        if self.role not in {
            PromptSemanticRole.GIVEN,
            PromptSemanticRole.WORKED_STEP,
            PromptSemanticRole.WORKED_ANSWER,
        }:
            raise ValueError(f"math segments cannot have role {self.role.value!r}")
        return self


class TablePromptSegment(StrictFrozenModel):
    """An accessible static data table embedded in a prompt."""

    kind: Literal["table"] = "table"
    role: PromptSemanticRole = PromptSemanticRole.CONTEXT
    caption: str = Field(min_length=1)
    column_headers: tuple[str, ...] = Field(min_length=1, max_length=12)
    rows: tuple[tuple[str, ...], ...] = Field(min_length=1, max_length=50)
    spoken_text: str = Field(min_length=1)

    @field_validator("caption", "spoken_text", mode="before")
    @classmethod
    def _normalize_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("column_headers", mode="before")
    @classmethod
    def _normalize_headers(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            return value
        return tuple(cell.strip() if isinstance(cell, str) else cell for cell in value)

    @field_validator("rows", mode="before")
    @classmethod
    def _normalize_rows(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            return value
        return tuple(
            tuple(cell.strip() if isinstance(cell, str) else cell for cell in row)
            if isinstance(row, (list, tuple))
            else row
            for row in value
        )

    @model_validator(mode="after")
    def _valid_table(self) -> "TablePromptSegment":
        if self.role not in {
            PromptSemanticRole.CONTEXT,
            PromptSemanticRole.GIVEN,
            PromptSemanticRole.WORKED_STEP,
            PromptSemanticRole.WORKED_ANSWER,
        }:
            raise ValueError(f"table segments cannot have role {self.role.value!r}")
        if len(self.column_headers) != len(set(self.column_headers)):
            raise ValueError("table column headers must be unique")
        if any(not header for header in self.column_headers):
            raise ValueError("table column headers must be nonblank")
        width = len(self.column_headers)
        if any(len(row) != width for row in self.rows):
            raise ValueError("every table row must match the column-header width")
        if any(not cell for row in self.rows for cell in row):
            raise ValueError("table cells must be nonblank")
        return self


class StaticPlotPoint(StrictFrozenModel):
    """One exact, display-independent point in a reviewed static plot."""

    x: str = Field(min_length=1)
    y: str = Field(min_length=1)


class StaticPlotSeries(StrictFrozenModel):
    """One labeled series in a reviewed static plot."""

    label: str = Field(min_length=1)
    points: tuple[StaticPlotPoint, ...] = Field(min_length=2, max_length=100)


class PlotPromptSegment(StrictFrozenModel):
    """Static plot data with an equivalent text or table representation."""

    kind: Literal["plot"] = "plot"
    role: PromptSemanticRole = PromptSemanticRole.CONTEXT
    title: str = Field(min_length=1)
    x_label: str = Field(min_length=1)
    y_label: str = Field(min_length=1)
    series: tuple[StaticPlotSeries, ...] = Field(min_length=1, max_length=8)
    spoken_text: str = Field(min_length=1)
    equivalent_table: TablePromptSegment | None = None

    @field_validator("title", "x_label", "y_label", "spoken_text", mode="before")
    @classmethod
    def _normalize_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _valid_plot(self) -> "PlotPromptSegment":
        if self.role not in {
            PromptSemanticRole.CONTEXT,
            PromptSemanticRole.GIVEN,
            PromptSemanticRole.WORKED_STEP,
            PromptSemanticRole.WORKED_ANSWER,
        }:
            raise ValueError(f"plot segments cannot have role {self.role.value!r}")
        labels = [series.label for series in self.series]
        if len(labels) != len(set(labels)):
            raise ValueError("plot series labels must be unique")
        point_count = sum(len(series.points) for series in self.series)
        if self.equivalent_table is not None:
            table_cells = sum(len(row) for row in self.equivalent_table.rows)
            if table_cells < point_count:
                raise ValueError(
                    "an equivalent plot table must represent every plotted point"
                )
        elif len(self.spoken_text.split()) < 6:
            raise ValueError(
                "a plot without an equivalent table requires a complete textual description"
            )
        return self


class BlankPromptSegment(StrictFrozenModel):
    """One answer location in an assessment prompt."""

    kind: Literal["blank"] = "blank"
    role: PromptSemanticRole = PromptSemanticRole.RESPONSE
    label: str | None = None

    @model_validator(mode="after")
    def _valid_role(self) -> "BlankPromptSegment":
        if self.role != PromptSemanticRole.RESPONSE:
            raise ValueError("blank segments must have role 'response'")
        return self


class GuidedMappingEntry(StrictFrozenModel):
    """One stable, accessible row or option in a guided mapping activity."""

    entry_id: str = Field(max_length=64, pattern=_CONTENT_ID_PATTERN)
    label: str = Field(min_length=1, max_length=256)
    spoken_text: str = Field(min_length=1, max_length=512)

    @field_validator("label", "spoken_text", mode="before")
    @classmethod
    def _normalize_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class GuidedMappingPresentation(StrictFrozenModel):
    """The safe, deterministic portion of a mapping interaction."""

    prompt: str = Field(min_length=1, max_length=512)
    rows: tuple[GuidedMappingEntry, ...] = Field(min_length=2, max_length=12)
    options: tuple[GuidedMappingEntry, ...] = Field(min_length=2, max_length=12)

    @model_validator(mode="after")
    def _unique_entries(self) -> "GuidedMappingPresentation":
        row_ids = [entry.entry_id for entry in self.rows]
        option_ids = [entry.entry_id for entry in self.options]
        if len(row_ids) != len(set(row_ids)):
            raise ValueError("mapping row ids must be unique")
        if len(option_ids) != len(set(option_ids)):
            raise ValueError("mapping option ids must be unique")
        if set(row_ids) & set(option_ids):
            raise ValueError("mapping row and option ids must be disjoint")
        return self


class GuidedMappingScoring(StrictFrozenModel):
    """Private mapping truth; this model must never enter a SessionView."""

    correct_pairs: tuple[tuple[str, str], ...] = Field(min_length=2, max_length=12)


class GuidedMappingSpec(StrictFrozenModel):
    """A keyboard-operable, one-to-one mapping interaction."""

    kind: Literal["mapping_v1"] = "mapping_v1"
    presentation: GuidedMappingPresentation
    scoring: GuidedMappingScoring

    @model_validator(mode="after")
    def _complete_one_to_one_mapping(self) -> "GuidedMappingSpec":
        row_ids = {entry.entry_id for entry in self.presentation.rows}
        option_ids = {entry.entry_id for entry in self.presentation.options}
        pairs = self.scoring.correct_pairs
        paired_rows = [row_id for row_id, _ in pairs]
        paired_options = [option_id for _, option_id in pairs]
        if set(paired_rows) != row_ids:
            raise ValueError("mapping scoring must pair every public row exactly once")
        if len(paired_rows) != len(set(paired_rows)):
            raise ValueError("mapping scoring repeats a row")
        if len(paired_options) != len(set(paired_options)):
            raise ValueError("mapping scoring repeats an option")
        if not set(paired_options).issubset(option_ids):
            raise ValueError("mapping scoring references an unknown public option")
        return self


class GuidedSliderPresentation(StrictFrozenModel):
    """The public bounded state and accessible labels for a slider activity."""

    prompt: str = Field(min_length=1, max_length=512)
    label: str = Field(min_length=1, max_length=128)
    help_text: str = Field(min_length=1, max_length=512)
    minimum: float = Field(allow_inf_nan=False)
    maximum: float = Field(allow_inf_nan=False)
    step: float = Field(gt=0, allow_inf_nan=False)
    initial_value: float = Field(allow_inf_nan=False)
    value_label: str = Field(min_length=1, max_length=128)
    result_template: str | None = Field(default=None, min_length=1, max_length=256)
    visual_summary: PlotPromptSegment | None = None

    @model_validator(mode="after")
    def _valid_range(self) -> "GuidedSliderPresentation":
        if self.maximum <= self.minimum:
            raise ValueError("slider maximum must be greater than its minimum")
        if not self.minimum <= self.initial_value <= self.maximum:
            raise ValueError("slider initial value must lie within its bounds")
        if self.result_template is not None and "{value}" not in self.result_template:
            raise ValueError("slider result_template must contain the {value} placeholder")
        return self


class GuidedSliderScoring(StrictFrozenModel):
    """Private slider truth; this model must never enter a SessionView."""

    target: float = Field(allow_inf_nan=False)
    tolerance: float = Field(default=0, ge=0, allow_inf_nan=False)


class GuidedSliderSpec(StrictFrozenModel):
    """A bounded numeric slider with private success conditions."""

    kind: Literal["slider_v1"] = "slider_v1"
    presentation: GuidedSliderPresentation
    scoring: GuidedSliderScoring

    @model_validator(mode="after")
    def _target_is_reachable(self) -> "GuidedSliderSpec":
        presentation = self.presentation
        target = self.scoring.target
        if not presentation.minimum <= target <= presentation.maximum:
            raise ValueError("slider target must lie within its public bounds")
        steps = (target - presentation.minimum) / presentation.step
        if abs(steps - round(steps)) > 1e-9:
            raise ValueError("slider target must be reachable on the public step grid")
        return self


GuidedInteractionSpec = Annotated[
    Union[GuidedMappingSpec, GuidedSliderSpec],
    Field(discriminator="kind"),
]
guided_interaction_spec_adapter: TypeAdapter[GuidedInteractionSpec] = TypeAdapter(
    GuidedInteractionSpec
)


DisplayPromptSegment = Annotated[
    Union[
        TextPromptSegment,
        MathPromptSegment,
        TablePromptSegment,
        PlotPromptSegment,
    ],
    Field(discriminator="kind"),
]
display_prompt_segment_adapter: TypeAdapter[DisplayPromptSegment] = TypeAdapter(
    DisplayPromptSegment
)


PromptSegment = Annotated[
    Union[
        TextPromptSegment,
        MathPromptSegment,
        TablePromptSegment,
        PlotPromptSegment,
        BlankPromptSegment,
    ],
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
    # Older releases accepted any representative of the family. Content that
    # explicitly assesses indefinite-integral notation opts into requiring one
    # additive ``+C`` term while retaining exact replay for those releases.
    require_explicit_constant: bool = False
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
    source_id: str | None = Field(
        default=None,
        max_length=128,
        pattern=_CONTENT_ID_PATTERN,
    )
    source_revision: int | None = Field(default=None, ge=1)
    source_digest: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    compiler_version: str | None = Field(
        default=None,
        max_length=128,
        pattern=_CONTENT_ID_PATTERN,
    )

    @field_validator("source", "author", "reviewed_by", mode="before")
    @classmethod
    def _normalize_meaningful_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("reviewed_at")
    @classmethod
    def _review_timestamp_is_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("reviewed_at must include a timezone")
        return value

    @model_validator(mode="after")
    def _review_and_source_bindings_are_complete(self) -> "AssessmentProvenance":
        has_reviewer = self.reviewed_by is not None
        has_review_time = self.reviewed_at is not None
        if has_reviewer != has_review_time:
            raise ValueError("reviewed_by and reviewed_at must be supplied together")
        if (
            self.reviewed_by is not None
            and self.author.casefold() == self.reviewed_by.casefold()
        ):
            raise ValueError("reviewed_by must identify someone other than the author")
        source_binding = (
            self.source_id,
            self.source_revision,
            self.source_digest,
            self.compiler_version,
        )
        if any(value is not None for value in source_binding) and not all(
            value is not None for value in source_binding
        ):
            raise ValueError(
                "source_id, source_revision, source_digest, and compiler_version "
                "must be supplied together"
            )
        return self


class AssessmentItem(StrictFrozenModel):
    """One stable, independently authored assessment or instruction item."""

    item_id: str = Field(max_length=128, pattern=_CONTENT_ID_PATTERN)
    revision: int = Field(default=1, ge=1)
    family_id: str = Field(max_length=128, pattern=_CONTENT_ID_PATTERN)
    kc_id: str = Field(pattern=KC_ID_PATTERN)
    difficulty: Literal["foundation", "core", "stretch"] = "core"
    # Legacy items are ordinary solve tasks. A transform declaration is the
    # narrow exception that permits an equivalent, non-answer ``given``.
    task_kind: AssessmentTaskKind = AssessmentTaskKind.SOLVE
    eligible_surfaces: list[AssessmentSurface] = Field(min_length=1)
    # Kept optional so legacy banks/checkpoints remain parseable. Release
    # validation requires an authored order for every released family.
    allocation_order: int | None = Field(default=None, ge=0)
    prompt: list[PromptSegment] = Field(min_length=1)
    hints: list[AssessmentHint] = Field(min_length=3, max_length=3)
    answer: AnswerSpec
    review_status: ReviewStatus
    provenance: AssessmentProvenance
    error_signatures: list[ErrorSignature] = Field(default_factory=list)
    # The scoring branch is private release content. Runtime projections expose
    # only ``presentation`` and never serialize this object wholesale.
    guided_interaction: GuidedInteractionSpec | None = None

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
        if self.task_kind == AssessmentTaskKind.TRANSFORM:
            givens = sum(
                isinstance(segment, MathPromptSegment)
                and segment.role == PromptSemanticRole.GIVEN
                for segment in self.prompt
            )
            if givens != 1:
                raise ValueError("a transform task must contain exactly one math given")
        guided = AssessmentSurface.GUIDED_WIDGET in self.eligible_surfaces
        if not guided and self.guided_interaction is not None:
            raise ValueError(
                "only guided_widget items may carry a guided_interaction"
            )
        return self


class ItemBankDocument(StrictFrozenModel):
    """A complete item-bank release pinned to one KC graph version."""

    # Version 2 remains parseable for exact replay. Version 3 adds the
    # accessibility contract enforced below and is required by the new
    # publication boundary.
    schema_version: Literal[2, 3] = 2
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
            tuple[
                str,
                str,
                AssessmentTaskKind,
                frozenset[AssessmentSurface],
                int | None,
            ],
        ] = {}
        family_orders: dict[
            tuple[str, frozenset[AssessmentSurface]], int | None
        ] = {}
        for item in self.items:
            if self.schema_version >= 3:
                for segment in item.prompt:
                    if (
                        isinstance(segment, MathPromptSegment)
                        and segment.spoken_text is None
                    ):
                        raise ValueError(
                            f"schema-v3 item {item.item_id!r} has math without spoken_text"
                        )
                is_guided = AssessmentSurface.GUIDED_WIDGET in item.eligible_surfaces
                if is_guided and item.guided_interaction is None:
                    raise ValueError(
                        f"schema-v3 guided item {item.item_id!r} lacks guided_interaction"
                    )
            previous = family_kcs.setdefault(item.family_id, item.kc_id)
            if previous != item.kc_id:
                raise ValueError(
                    f"family {item.family_id!r} spans KCs {previous!r} and {item.kc_id!r}"
                )
            lineage = (
                item.kc_id,
                item.family_id,
                item.task_kind,
                frozenset(item.eligible_surfaces),
                item.allocation_order,
            )
            previous_lineage = item_lineages.setdefault(item.item_id, lineage)
            if previous_lineage != lineage:
                raise ValueError(
                    f"revisions of item {item.item_id!r} must retain the same "
                    "KC, family, task kind, eligible surfaces, and allocation order"
                )
            family_surface = (
                item.family_id,
                frozenset(item.eligible_surfaces),
            )
            if family_surface in family_orders:
                if family_orders[family_surface] != item.allocation_order:
                    raise ValueError(
                        f"variants in family {item.family_id!r} must retain the same "
                        "allocation order"
                    )
            else:
                family_orders[family_surface] = item.allocation_order
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
