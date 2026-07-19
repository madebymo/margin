"""Widget configs — a versioned discriminated union keyed by ``widget_type``.

LLMs emit these as JSON; this module is the validation contract. Unknown
``widget_type`` values are rejected, so new interaction types require an
explicit schema addition (and a bumped ``schema_version``) rather than
slipping through as arbitrary payloads.
"""

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter, model_validator


class WidgetBase(BaseModel):
    """Fields shared by every widget type."""

    schema_version: int = 1
    learning_objective: str = Field(min_length=1)
    metaphor_id: str | None = None
    prompt: str = Field(min_length=1)


class SliderParams(BaseModel):
    """Slider range and optional plot to render behind it."""

    min: float
    max: float
    step: float = Field(gt=0)
    plot: str | None = None
    shade: str | None = None

    @model_validator(mode="after")
    def _range_ok(self) -> "SliderParams":
        if self.max <= self.min:
            raise ValueError(f"slider range invalid: max ({self.max}) <= min ({self.min})")
        return self


class SuccessCondition(BaseModel):
    """Target value the learner must reach, within tolerance."""

    target: float
    tolerance: float = Field(ge=0)


class FeedbackRule(BaseModel):
    """Conditional feedback shown while the learner interacts."""

    when: str = Field(min_length=1)
    say: str = Field(min_length=1)


class SliderWidget(WidgetBase):
    """Drag a value; a plot/shade responds live."""

    widget_type: Literal["slider"] = "slider"
    params: SliderParams
    success_condition: SuccessCondition
    feedback_rules: list[FeedbackRule] = Field(default_factory=list)


class Region(BaseModel):
    """A clickable region rendered on the widget canvas."""

    id: str = Field(min_length=1)
    label: str | None = None
    shape: dict[str, Any] = Field(default_factory=dict)


class ClickRegionWidget(WidgetBase):
    """Click one or more regions (e.g. areas under a curve, unit-circle points)."""

    widget_type: Literal["click_region"] = "click_region"
    regions: list[Region] = Field(min_length=2)
    correct_region_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _correct_ids_exist(self) -> "ClickRegionWidget":
        region_ids = {r.id for r in self.regions}
        missing = [rid for rid in self.correct_region_ids if rid not in region_ids]
        if missing:
            raise ValueError(f"correct_region_ids not present in regions: {missing}")
        return self


class MappingWidget(WidgetBase):
    """Match items between two columns (e.g. functions to their derivatives)."""

    widget_type: Literal["mapping"] = "mapping"
    left: list[str] = Field(min_length=2)
    right: list[str] = Field(min_length=2)
    correct_pairs: list[tuple[str, str]] = Field(min_length=1)

    @model_validator(mode="after")
    def _pairs_exist(self) -> "MappingWidget":
        left_set, right_set = set(self.left), set(self.right)
        for left_item, right_item in self.correct_pairs:
            if left_item not in left_set:
                raise ValueError(f"pair references unknown left item: {left_item!r}")
            if right_item not in right_set:
                raise ValueError(f"pair references unknown right item: {right_item!r}")
        return self


class LiveInputChecker(BaseModel):
    """How a typed answer is judged (symbolic equivalence or numeric tolerance)."""

    equivalence: Literal["sympy_equiv", "numeric"]
    expected: str = Field(min_length=1)
    tolerance: float | None = Field(default=None, ge=0)


class LiveInputWidget(WidgetBase):
    """Type a number/expression; the visual updates live and the answer is checked."""

    widget_type: Literal["live_input"] = "live_input"
    input_kind: Literal["number", "expression"]
    render: dict[str, Any] = Field(default_factory=dict)
    checker: LiveInputChecker


WidgetConfig = Annotated[
    Union[SliderWidget, ClickRegionWidget, MappingWidget, LiveInputWidget],
    Field(discriminator="widget_type"),
]

widget_config_adapter: TypeAdapter[WidgetConfig] = TypeAdapter(WidgetConfig)


def parse_widget_config(data: dict[str, Any]) -> WidgetConfig:
    """Validate a raw dict (e.g. LLM output) into a concrete widget config.

    Raises pydantic.ValidationError on unknown widget_type or invalid payloads.
    """
    return widget_config_adapter.validate_python(data)
