"""Strict source contracts for the pending Fundamental Theorem wave.

The source document stores only bounded mathematical parameters.  Learner-visible
prompts, expected answers, hints, worked steps, widget truth, and error signatures
are deterministic compiler outputs; none can be supplied as free-form source data.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import Field, model_validator

from tutor.schemas.assessment import AssessmentSurface, StrictFrozenModel
from tutor.schemas.kc import KC_ID_PATTERN

_CONTENT_ID_PATTERN = r"^[a-z0-9][a-z0-9._-]*$"


class IntegerPoint(StrictFrozenModel):
    """One exact point used by a reviewed piecewise-linear display."""

    x: int = Field(ge=-20, le=20)
    y: int = Field(ge=-40, le=40)


class PiecewiseLinearSpec(StrictFrozenModel):
    """An exact graph whose segments all have integer slopes."""

    points: tuple[IntegerPoint, ...] = Field(min_length=3, max_length=7)

    @model_validator(mode="after")
    def _ordered_with_integral_slopes(self) -> "PiecewiseLinearSpec":
        xs = [point.x for point in self.points]
        if xs != sorted(xs) or len(xs) != len(set(xs)):
            raise ValueError("piecewise-linear x values must be strictly increasing")
        slopes: list[int] = []
        for left, right in zip(self.points, self.points[1:]):
            rise = right.y - left.y
            run = right.x - left.x
            if rise % run:
                raise ValueError("every piecewise-linear segment must have integer slope")
            slopes.append(rise // run)
        if not any(slopes):
            raise ValueError("piecewise-linear data cannot be constant everywhere")
        return self


class GraphPointValueTask(StrictFrozenModel):
    kind: Literal["graph_point_value"] = "graph_point_value"
    graph: PiecewiseLinearSpec
    point_index: int = Field(ge=0, le=6)

    @model_validator(mode="after")
    def _index_exists(self) -> "GraphPointValueTask":
        if self.point_index >= len(self.graph.points):
            raise ValueError("point_index is outside the graph")
        return self


class GraphInterceptsTask(StrictFrozenModel):
    kind: Literal["graph_intercepts"] = "graph_intercepts"
    graph: PiecewiseLinearSpec

    @model_validator(mode="after")
    def _has_unambiguous_integer_intercepts(self) -> "GraphInterceptsTask":
        points = self.graph.points
        y_intercepts = [point for point in points if point.x == 0]
        x_intercepts = [point for point in points if point.y == 0]
        if len(y_intercepts) != 1 or len(x_intercepts) != 1:
            raise ValueError("intercept tasks require one listed x- and y-intercept")
        if y_intercepts[0].y == 0:
            raise ValueError("the two intercepts must be distinct")
        for left, right in zip(points, points[1:]):
            if left.y * right.y < 0:
                raise ValueError("an unlisted segment crossing would hide an x-intercept")
        return self


class GraphSlopeTask(StrictFrozenModel):
    kind: Literal["graph_slope"] = "graph_slope"
    graph: PiecewiseLinearSpec
    segment_index: int = Field(ge=0, le=5)

    @model_validator(mode="after")
    def _segment_exists(self) -> "GraphSlopeTask":
        if self.segment_index >= len(self.graph.points) - 1:
            raise ValueError("segment_index is outside the graph")
        return self


class GraphBehaviorTask(StrictFrozenModel):
    kind: Literal["graph_behavior"] = "graph_behavior"
    graph: PiecewiseLinearSpec
    behavior: Literal["increasing", "decreasing"]

    @model_validator(mode="after")
    def _one_contiguous_behavior_run(self) -> "GraphBehaviorTask":
        matches = []
        for index, (left, right) in enumerate(
            zip(self.graph.points, self.graph.points[1:])
        ):
            slope = right.y - left.y
            matches.append(slope > 0 if self.behavior == "increasing" else slope < 0)
        selected = [index for index, matches_behavior in enumerate(matches) if matches_behavior]
        if not selected or selected != list(range(selected[0], selected[-1] + 1)):
            raise ValueError("the requested behavior must form one contiguous interval")
        return self


class GraphOrderedReadTask(StrictFrozenModel):
    kind: Literal["graph_ordered_read"] = "graph_ordered_read"
    graph: PiecewiseLinearSpec
    point_indices: tuple[int, int, int]

    @model_validator(mode="after")
    def _indices_exist_and_differ(self) -> "GraphOrderedReadTask":
        if len(set(self.point_indices)) != 3:
            raise ValueError("ordered reads require three distinct points")
        if any(index >= len(self.graph.points) for index in self.point_indices):
            raise ValueError("an ordered-read index is outside the graph")
        return self


class GraphGuidedMappingTask(StrictFrozenModel):
    kind: Literal["graph_guided_mapping"] = "graph_guided_mapping"
    graph: PiecewiseLinearSpec
    point_indices: tuple[int, int, int]

    @model_validator(mode="after")
    def _mapping_is_one_to_one(self) -> "GraphGuidedMappingTask":
        if len(set(self.point_indices)) != 3:
            raise ValueError("guided graph mapping requires three distinct points")
        try:
            outputs = [self.graph.points[index].y for index in self.point_indices]
        except IndexError as exc:
            raise ValueError("a guided graph index is outside the graph") from exc
        if len(outputs) != len(set(outputs)):
            raise ValueError("guided graph outputs must be distinct")
        return self


class RectangleRegion(StrictFrozenModel):
    width: int = Field(ge=1, le=12)
    height: int = Field(ge=1, le=20)


class TriangleRegion(StrictFrozenModel):
    base: int = Field(ge=1, le=12)
    height: int = Field(ge=1, le=20)

    @model_validator(mode="after")
    def _integer_area(self) -> "TriangleRegion":
        if self.base * self.height % 2:
            raise ValueError("triangle parameters must produce an integer area")
        return self


class CompositeRegionSpec(StrictFrozenModel):
    rectangles: tuple[RectangleRegion, ...] = Field(default=(), max_length=3)
    triangles: tuple[TriangleRegion, ...] = Field(default=(), max_length=3)

    @model_validator(mode="after")
    def _has_multiple_regions(self) -> "CompositeRegionSpec":
        if len(self.rectangles) + len(self.triangles) < 2:
            raise ValueError("a composite area requires at least two regions")
        return self


class AreaRectangleTask(StrictFrozenModel):
    kind: Literal["area_rectangle"] = "area_rectangle"
    region: RectangleRegion


class AreaTriangleTask(StrictFrozenModel):
    kind: Literal["area_triangle"] = "area_triangle"
    region: TriangleRegion


class AreaCompositeTask(StrictFrozenModel):
    kind: Literal["area_composite"] = "area_composite"
    region: CompositeRegionSpec


class AreaMissingHeightTask(StrictFrozenModel):
    kind: Literal["area_missing_height"] = "area_missing_height"
    shape: Literal["rectangle", "triangle"]
    width_or_base: int = Field(ge=1, le=12)
    height: int = Field(ge=1, le=20)

    @model_validator(mode="after")
    def _triangle_area_is_integral(self) -> "AreaMissingHeightTask":
        if self.shape == "triangle" and self.width_or_base * self.height % 2:
            raise ValueError("triangle parameters must produce an integer area")
        return self


class AreaOrderedPartsTask(StrictFrozenModel):
    kind: Literal["area_ordered_parts"] = "area_ordered_parts"
    rectangle: RectangleRegion
    triangle: TriangleRegion


class AreaGuidedSliderTask(StrictFrozenModel):
    kind: Literal["area_guided_slider"] = "area_guided_slider"
    region: CompositeRegionSpec


class EndpointTableSpec(StrictFrozenModel):
    """Endpoint samples for equal-width left/right rectangle sums."""

    lower: int = Field(ge=-10, le=20)
    width: int = Field(ge=1, le=5)
    values: tuple[int, ...] = Field(min_length=4, max_length=6)

    @model_validator(mode="after")
    def _nonnegative_nonconstant_values(self) -> "EndpointTableSpec":
        if any(value < 0 or value > 30 for value in self.values):
            raise ValueError("Riemann table values must lie from zero through thirty")
        if len(set(self.values)) == 1:
            raise ValueError("Riemann endpoint values cannot all be equal")
        return self


class MidpointTableSpec(StrictFrozenModel):
    """Midpoint samples with even widths so all displayed inputs are integers."""

    lower: int = Field(ge=-10, le=20)
    width: Literal[2, 4]
    values: tuple[int, ...] = Field(min_length=3, max_length=5)

    @model_validator(mode="after")
    def _nonnegative_nonconstant_values(self) -> "MidpointTableSpec":
        if any(value < 0 or value > 30 for value in self.values):
            raise ValueError("Riemann midpoint values must lie from zero through thirty")
        if len(set(self.values)) == 1:
            raise ValueError("Riemann midpoint values cannot all be equal")
        return self


class RiemannLeftTask(StrictFrozenModel):
    kind: Literal["riemann_left"] = "riemann_left"
    table: EndpointTableSpec


class RiemannRightTask(StrictFrozenModel):
    kind: Literal["riemann_right"] = "riemann_right"
    table: EndpointTableSpec


class RiemannMidpointTask(StrictFrozenModel):
    kind: Literal["riemann_midpoint"] = "riemann_midpoint"
    table: MidpointTableSpec


class RiemannCompareTask(StrictFrozenModel):
    kind: Literal["riemann_compare"] = "riemann_compare"
    table: EndpointTableSpec


class RiemannMissingHeightTask(StrictFrozenModel):
    kind: Literal["riemann_missing_height"] = "riemann_missing_height"
    width: int = Field(ge=1, le=5)
    known_heights: tuple[int, ...] = Field(min_length=2, max_length=4)
    missing_height: int = Field(ge=0, le=30)


class RiemannContributionsTask(StrictFrozenModel):
    kind: Literal["riemann_contributions"] = "riemann_contributions"
    width: int = Field(ge=1, le=5)
    heights: tuple[int, ...] = Field(min_length=3, max_length=5)


class RiemannGuidedMappingTask(StrictFrozenModel):
    kind: Literal["riemann_guided_mapping"] = "riemann_guided_mapping"
    table: EndpointTableSpec

    @model_validator(mode="after")
    def _three_distinct_left_heights(self) -> "RiemannGuidedMappingTask":
        if len(self.table.values) != 4 or len(set(self.table.values[:-1])) != 3:
            raise ValueError("guided Riemann mapping needs three distinct left heights")
        return self


class DefiniteOrientationTask(StrictFrozenModel):
    kind: Literal["definite_orientation"] = "definite_orientation"
    lower: int = Field(ge=-10, le=20)
    upper: int = Field(ge=-10, le=20)
    forward_value: int = Field(ge=-300, le=300)

    @model_validator(mode="after")
    def _ordered_nonzero(self) -> "DefiniteOrientationTask":
        if self.lower >= self.upper or self.forward_value == 0:
            raise ValueError("orientation data require ordered bounds and a nonzero value")
        return self


class DefiniteAdditivityTask(StrictFrozenModel):
    kind: Literal["definite_additivity"] = "definite_additivity"
    lower: int = Field(ge=-10, le=20)
    split: int = Field(ge=-10, le=20)
    upper: int = Field(ge=-10, le=20)
    left_value: int = Field(ge=-300, le=300)
    right_value: int = Field(ge=-300, le=300)

    @model_validator(mode="after")
    def _ordered_bounds(self) -> "DefiniteAdditivityTask":
        if not self.lower < self.split < self.upper:
            raise ValueError("additivity bounds must be strictly ordered")
        return self


class DefiniteMissingPieceTask(StrictFrozenModel):
    kind: Literal["definite_missing_piece"] = "definite_missing_piece"
    lower: int = Field(ge=-10, le=20)
    split: int = Field(ge=-10, le=20)
    upper: int = Field(ge=-10, le=20)
    total_value: int = Field(ge=-300, le=300)
    left_value: int = Field(ge=-300, le=300)

    @model_validator(mode="after")
    def _ordered_bounds(self) -> "DefiniteMissingPieceTask":
        if not self.lower < self.split < self.upper:
            raise ValueError("missing-piece bounds must be strictly ordered")
        return self


class DefiniteSignedRegionsTask(StrictFrozenModel):
    kind: Literal["definite_signed_regions"] = "definite_signed_regions"
    signed_areas: tuple[int, ...] = Field(min_length=3, max_length=5)

    @model_validator(mode="after")
    def _uses_both_signs(self) -> "DefiniteSignedRegionsTask":
        if not any(value > 0 for value in self.signed_areas) or not any(
            value < 0 for value in self.signed_areas
        ):
            raise ValueError("signed accumulation must include positive and negative regions")
        if any(value == 0 or abs(value) > 100 for value in self.signed_areas):
            raise ValueError("signed region values must be modest and nonzero")
        return self


class DefiniteInterpretationTask(StrictFrozenModel):
    kind: Literal["definite_interpretation"] = "definite_interpretation"
    lower: int = Field(ge=-10, le=20)
    upper: int = Field(ge=-10, le=20)
    value: int = Field(ge=-300, le=300)

    @model_validator(mode="after")
    def _different_bounds(self) -> "DefiniteInterpretationTask":
        if self.lower == self.upper:
            raise ValueError("definite-integral bounds must differ")
        return self


class DefiniteTwoOrientationsTask(StrictFrozenModel):
    kind: Literal["definite_two_orientations"] = "definite_two_orientations"
    lower: int = Field(ge=-10, le=20)
    upper: int = Field(ge=-10, le=20)
    forward_value: int = Field(ge=-300, le=300)

    @model_validator(mode="after")
    def _ordered_nonzero(self) -> "DefiniteTwoOrientationsTask":
        if self.lower >= self.upper or self.forward_value == 0:
            raise ValueError("two-orientation data require ordered bounds and a nonzero value")
        return self


class DefiniteGuidedMappingTask(StrictFrozenModel):
    kind: Literal["definite_guided_mapping"] = "definite_guided_mapping"
    lower: int = Field(ge=-10, le=20)
    split: int = Field(ge=-10, le=20)
    upper: int = Field(ge=-10, le=20)
    left_value: int = Field(ge=-300, le=300)
    right_value: int = Field(ge=-300, le=300)

    @model_validator(mode="after")
    def _ordered_distinct_results(self) -> "DefiniteGuidedMappingTask":
        if not self.lower < self.split < self.upper:
            raise ValueError("guided definite-integral bounds must be ordered")
        results = (self.left_value, self.right_value, self.left_value + self.right_value)
        if len(set(results)) != 3:
            raise ValueError("guided definite-integral results must be distinct")
        return self


class PolynomialTerm(StrictFrozenModel):
    """One nonconstant term of an antiderivative polynomial."""

    coefficient: int = Field(ge=-15, le=15)
    exponent: int = Field(ge=1, le=6)

    @model_validator(mode="after")
    def _nonzero(self) -> "PolynomialTerm":
        if self.coefficient == 0:
            raise ValueError("polynomial term coefficient cannot be zero")
        return self


class AntiderivativePolynomialSpec(StrictFrozenModel):
    """F(x); the compiler differentiates it to obtain the learner's given."""

    terms: tuple[PolynomialTerm, ...] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def _unique_descending_powers(self) -> "AntiderivativePolynomialSpec":
        powers = [term.exponent for term in self.terms]
        if powers != sorted(powers, reverse=True) or len(powers) != len(set(powers)):
            raise ValueError("antiderivative powers must be unique and descending")
        return self


class AntiderivativeSingleTask(StrictFrozenModel):
    kind: Literal["antiderivative_single"] = "antiderivative_single"
    polynomial: AntiderivativePolynomialSpec

    @model_validator(mode="after")
    def _one_term(self) -> "AntiderivativeSingleTask":
        if len(self.polynomial.terms) != 1:
            raise ValueError("single-term task requires one term")
        return self


class AntiderivativeBinomialTask(StrictFrozenModel):
    kind: Literal["antiderivative_binomial"] = "antiderivative_binomial"
    polynomial: AntiderivativePolynomialSpec

    @model_validator(mode="after")
    def _two_terms(self) -> "AntiderivativeBinomialTask":
        if len(self.polynomial.terms) != 2:
            raise ValueError("binomial task requires two terms")
        return self


class AntiderivativeTrinomialTask(StrictFrozenModel):
    kind: Literal["antiderivative_trinomial"] = "antiderivative_trinomial"
    polynomial: AntiderivativePolynomialSpec

    @model_validator(mode="after")
    def _three_terms(self) -> "AntiderivativeTrinomialTask":
        if len(self.polynomial.terms) != 3:
            raise ValueError("trinomial task requires three terms")
        return self


class AntiderivativeCorrectionTask(StrictFrozenModel):
    kind: Literal["antiderivative_correction"] = "antiderivative_correction"
    polynomial: AntiderivativePolynomialSpec
    mistake: Literal["kept_power", "did_not_divide", "dropped_term"]


class AntiderivativeDerivativeCheckTask(StrictFrozenModel):
    kind: Literal["antiderivative_derivative_check"] = "antiderivative_derivative_check"
    polynomial: AntiderivativePolynomialSpec


class AntiderivativeCoefficientAuditTask(StrictFrozenModel):
    kind: Literal["antiderivative_coefficient_audit"] = "antiderivative_coefficient_audit"
    polynomial: AntiderivativePolynomialSpec


class AntiderivativeGuidedMappingTask(StrictFrozenModel):
    kind: Literal["antiderivative_guided_mapping"] = "antiderivative_guided_mapping"
    polynomial: AntiderivativePolynomialSpec

    @model_validator(mode="after")
    def _three_terms(self) -> "AntiderivativeGuidedMappingTask":
        if len(self.polynomial.terms) != 3:
            raise ValueError("guided antiderivative mapping requires three terms")
        return self


class FTCSuppliedTask(StrictFrozenModel):
    kind: Literal["ftc_supplied"] = "ftc_supplied"
    polynomial: AntiderivativePolynomialSpec
    lower: int = Field(ge=-6, le=6)
    upper: int = Field(ge=-6, le=6)

    @model_validator(mode="after")
    def _ordered_bounds(self) -> "FTCSuppliedTask":
        if self.lower >= self.upper:
            raise ValueError("FTC bounds must be ordered")
        return self


class FTCDeriveTask(StrictFrozenModel):
    kind: Literal["ftc_derive"] = "ftc_derive"
    polynomial: AntiderivativePolynomialSpec
    lower: int = Field(ge=-6, le=6)
    upper: int = Field(ge=-6, le=6)

    @model_validator(mode="after")
    def _ordered_bounds(self) -> "FTCDeriveTask":
        if self.lower >= self.upper:
            raise ValueError("FTC bounds must be ordered")
        return self


class FTCReversedTask(StrictFrozenModel):
    kind: Literal["ftc_reversed"] = "ftc_reversed"
    polynomial: AntiderivativePolynomialSpec
    lower: int = Field(ge=-6, le=6)
    upper: int = Field(ge=-6, le=6)

    @model_validator(mode="after")
    def _ordered_source_bounds(self) -> "FTCReversedTask":
        if self.lower >= self.upper:
            raise ValueError("FTC source bounds must be ordered")
        return self


class FTCSplitTask(StrictFrozenModel):
    kind: Literal["ftc_split"] = "ftc_split"
    polynomial: AntiderivativePolynomialSpec
    lower: int = Field(ge=-6, le=6)
    split: int = Field(ge=-6, le=6)
    upper: int = Field(ge=-6, le=6)

    @model_validator(mode="after")
    def _ordered_bounds(self) -> "FTCSplitTask":
        if not self.lower < self.split < self.upper:
            raise ValueError("FTC split bounds must be strictly ordered")
        return self


class FTCCorrectionTask(StrictFrozenModel):
    kind: Literal["ftc_correction"] = "ftc_correction"
    polynomial: AntiderivativePolynomialSpec
    lower: int = Field(ge=-6, le=6)
    upper: int = Field(ge=-6, le=6)
    mistake: Literal["added_endpoints", "reversed_subtraction", "used_integrand"]

    @model_validator(mode="after")
    def _ordered_bounds(self) -> "FTCCorrectionTask":
        if self.lower >= self.upper:
            raise ValueError("FTC correction bounds must be ordered")
        return self


class FTCOrderedIntervalsTask(StrictFrozenModel):
    kind: Literal["ftc_ordered_intervals"] = "ftc_ordered_intervals"
    polynomial: AntiderivativePolynomialSpec
    first_bounds: tuple[int, int]
    second_bounds: tuple[int, int]

    @model_validator(mode="after")
    def _valid_distinct_intervals(self) -> "FTCOrderedIntervalsTask":
        for lower, upper in (self.first_bounds, self.second_bounds):
            if not -6 <= lower < upper <= 6:
                raise ValueError("ordered FTC intervals must be increasing and bounded")
        if self.first_bounds == self.second_bounds:
            raise ValueError("ordered FTC intervals must differ")
        return self


class FTCGuidedSliderTask(StrictFrozenModel):
    kind: Literal["ftc_guided_slider"] = "ftc_guided_slider"
    polynomial: AntiderivativePolynomialSpec
    lower: int = Field(ge=-4, le=4)
    upper: int = Field(ge=-4, le=4)

    @model_validator(mode="after")
    def _ordered_bounds(self) -> "FTCGuidedSliderTask":
        if self.lower >= self.upper:
            raise ValueError("guided FTC bounds must be ordered")
        return self


FTCConstructId = Literal[
    "graph.point_value",
    "graph.intercepts",
    "graph.slope",
    "graph.behavior",
    "graph.ordered_read",
    "graph.guided_mapping",
    "area.rectangle",
    "area.triangle",
    "area.composite",
    "area.missing_height",
    "area.ordered_parts",
    "area.guided_slider",
    "riemann.left",
    "riemann.right",
    "riemann.midpoint",
    "riemann.compare",
    "riemann.missing_height",
    "riemann.contributions",
    "riemann.guided_mapping",
    "definite.orientation",
    "definite.additivity",
    "definite.missing_piece",
    "definite.signed_regions",
    "definite.interpretation",
    "definite.two_orientations",
    "definite.guided_mapping",
    "antiderivative.single",
    "antiderivative.binomial",
    "antiderivative.trinomial",
    "antiderivative.correction",
    "antiderivative.derivative_check",
    "antiderivative.coefficient_audit",
    "antiderivative.guided_mapping",
    "ftc.supplied",
    "ftc.derive",
    "ftc.reversed",
    "ftc.split",
    "ftc.correction",
    "ftc.ordered_intervals",
    "ftc.guided_slider",
]


FTCMathTask = Annotated[
    Union[
        GraphPointValueTask,
        GraphInterceptsTask,
        GraphSlopeTask,
        GraphBehaviorTask,
        GraphOrderedReadTask,
        GraphGuidedMappingTask,
        AreaRectangleTask,
        AreaTriangleTask,
        AreaCompositeTask,
        AreaMissingHeightTask,
        AreaOrderedPartsTask,
        AreaGuidedSliderTask,
        RiemannLeftTask,
        RiemannRightTask,
        RiemannMidpointTask,
        RiemannCompareTask,
        RiemannMissingHeightTask,
        RiemannContributionsTask,
        RiemannGuidedMappingTask,
        DefiniteOrientationTask,
        DefiniteAdditivityTask,
        DefiniteMissingPieceTask,
        DefiniteSignedRegionsTask,
        DefiniteInterpretationTask,
        DefiniteTwoOrientationsTask,
        DefiniteGuidedMappingTask,
        AntiderivativeSingleTask,
        AntiderivativeBinomialTask,
        AntiderivativeTrinomialTask,
        AntiderivativeCorrectionTask,
        AntiderivativeDerivativeCheckTask,
        AntiderivativeCoefficientAuditTask,
        AntiderivativeGuidedMappingTask,
        FTCSuppliedTask,
        FTCDeriveTask,
        FTCReversedTask,
        FTCSplitTask,
        FTCCorrectionTask,
        FTCOrderedIntervalsTask,
        FTCGuidedSliderTask,
    ],
    Field(discriminator="kind"),
]


class FTCFamilyBlueprint(StrictFrozenModel):
    blueprint_id: str = Field(max_length=96, pattern=_CONTENT_ID_PATTERN)
    revision: int = Field(default=1, ge=1)
    item_id: str = Field(max_length=128, pattern=_CONTENT_ID_PATTERN)
    family_id: str = Field(max_length=128, pattern=_CONTENT_ID_PATTERN)
    kc_id: str = Field(pattern=KC_ID_PATTERN)
    construct_id: FTCConstructId
    surface: AssessmentSurface
    allocation_order: int = Field(ge=0)
    difficulty: Literal["foundation", "core", "stretch"] = "core"
    task: FTCMathTask


class FTCBlueprintDocument(StrictFrozenModel):
    schema_version: Literal[1] = 1
    blueprint_version: str = Field(pattern=_CONTENT_ID_PATTERN)
    output_bank_version: str = Field(pattern=_CONTENT_ID_PATTERN)
    graph_version: int = Field(ge=1)
    authoring_source: str = Field(min_length=1, max_length=128)
    author: str = Field(min_length=1)
    target_kcs: list[str] = Field(min_length=1)
    released_kcs: list[str] = Field(default_factory=list)
    families: list[FTCFamilyBlueprint] = Field(min_length=1)

    @model_validator(mode="after")
    def _identities_are_unambiguous(self) -> "FTCBlueprintDocument":
        for label, values in (
            ("target_kcs", self.target_kcs),
            ("released_kcs", self.released_kcs),
            ("blueprint identities", [(f.blueprint_id, f.revision) for f in self.families]),
            ("item ids", [f.item_id for f in self.families]),
            ("family ids", [f.family_id for f in self.families]),
            (
                "allocation orders",
                [(f.kc_id, f.surface, f.allocation_order) for f in self.families],
            ),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique")
        if not set(self.released_kcs) <= set(self.target_kcs):
            raise ValueError("released_kcs must be a subset of target_kcs")
        if any(family.kc_id not in self.target_kcs for family in self.families):
            raise ValueError("every family KC must occur in target_kcs")
        return self
