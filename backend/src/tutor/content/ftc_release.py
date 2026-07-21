"""Compile and qualify the pending Fundamental Theorem of Calculus wave.

This module is intentionally deterministic.  Typed source parameters are
compiled through a closed registry into accessible prompts, private answer
contracts, guided interactions, hints, and executable error signatures.  Draft
compilation cannot approve, attest, or release content.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import multiprocessing
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import cast

import sympy
from pydantic import BaseModel

from tutor.content.item_bank import (
    _candidate_answer_texts,
    _candidate_fits_answer_contract,
    load_item_bank,
    render_prompt,
)
from tutor.content.task_compilers import (
    TaskCompilerRegistration,
    TaskCompilerRegistry,
    TaskCompilerRegistryError,
)
from tutor.graph.service import ancestor_subgraph
from tutor.packs.review_compiler import COMPILER_VERSION as PEDAGOGY_COMPILER_VERSION
from tutor.packs.review_compiler import source_digest as pedagogy_source_digest
from tutor.packs.review_compiler import validate_review_bundle
from tutor.schemas.assessment import (
    AnswerSpec,
    AntiderivativeAnswerSpec,
    AssessmentHint,
    AssessmentItem,
    AssessmentProvenance,
    AssessmentSurface,
    BlankPromptSegment,
    ErrorSignature,
    FiniteSetAnswerSpec,
    GuidedInteractionSpec,
    GuidedMappingEntry,
    GuidedMappingPresentation,
    GuidedMappingScoring,
    GuidedMappingSpec,
    GuidedSliderPresentation,
    GuidedSliderScoring,
    GuidedSliderSpec,
    IntervalSetAnswerSpec,
    IntervalSpec,
    ItemBankDocument,
    MathPromptSegment,
    NumericAnswerSpec,
    OrderedTupleAnswerSpec,
    PlotPromptSegment,
    PromptSegment,
    PromptSemanticRole,
    StaticPlotPoint,
    StaticPlotSeries,
    SymbolicAnswerSpec,
    TablePromptSegment,
    TextPromptSegment,
)
from tutor.schemas.common import EdgeType, ReviewStatus
from tutor.schemas.content_authoring import (
    ContentReviewEntry,
    ContentReviewManifest,
    ReviewDecision,
)
from tutor.schemas.ftc_authoring import (
    AntiderivativeBinomialTask,
    AntiderivativeCoefficientAuditTask,
    AntiderivativeCorrectionTask,
    AntiderivativeDerivativeCheckTask,
    AntiderivativeGuidedMappingTask,
    AntiderivativePolynomialSpec,
    AntiderivativeSingleTask,
    AntiderivativeTrinomialTask,
    AreaCompositeTask,
    AreaGuidedSliderTask,
    AreaMissingHeightTask,
    AreaOrderedPartsTask,
    AreaRectangleTask,
    AreaTriangleTask,
    CompositeRegionSpec,
    DefiniteAdditivityTask,
    DefiniteGuidedMappingTask,
    DefiniteInterpretationTask,
    DefiniteMissingPieceTask,
    DefiniteOrientationTask,
    DefiniteSignedRegionsTask,
    DefiniteTwoOrientationsTask,
    EndpointTableSpec,
    FTCBlueprintDocument,
    FTCCorrectionTask,
    FTCDeriveTask,
    FTCFamilyBlueprint,
    FTCGuidedSliderTask,
    FTCMathTask,
    FTCOrderedIntervalsTask,
    FTCReversedTask,
    FTCSplitTask,
    FTCSuppliedTask,
    GraphBehaviorTask,
    GraphGuidedMappingTask,
    GraphInterceptsTask,
    GraphOrderedReadTask,
    GraphPointValueTask,
    GraphSlopeTask,
    MidpointTableSpec,
    PiecewiseLinearSpec,
    RectangleRegion,
    RiemannCompareTask,
    RiemannContributionsTask,
    RiemannGuidedMappingTask,
    RiemannLeftTask,
    RiemannMidpointTask,
    RiemannMissingHeightTask,
    RiemannRightTask,
    TriangleRegion,
)
from tutor.schemas.kc import GraphDocument
from tutor.schemas.pedagogy_authoring import (
    PedagogyReviewDecision,
    PedagogyReviewEntry,
    PedagogyReviewManifest,
    PedagogySourceDocument,
)
from tutor.verify.checker import (
    VerificationStatus,
    _split_container,
    parse_restricted,
    verify_answer,
)

COMPILER_VERSION = "ftc-item-compiler-v1"
AUTHOR = "AI-assisted implementation draft (unreviewed)"

GRAPH_KC = "kc.fun.graph_reading"
AREA_KC = "kc.int.area_under_curve"
RIEMANN_KC = "kc.int.riemann_sums"
DEFINITE_KC = "kc.int.definite_integral"
ANTIDERIVATIVE_KC = "kc.int.antiderivatives"
TARGET_KC = "kc.int.ftc"
TARGET_KCS = frozenset(
    {GRAPH_KC, AREA_KC, RIEMANN_KC, DEFINITE_KC, ANTIDERIVATIVE_KC, TARGET_KC}
)
EXPECTED_CLOSURE = frozenset(
    {
        "kc.alg.exponent_rules",
        "kc.fun.function_notation",
        "kc.der.power_rule",
        "kc.der.sum_constant_rules",
        *TARGET_KCS,
    }
)
EXPECTED_FAMILY_COUNTS = {
    AssessmentSurface.DIAGNOSTIC: 4,
    AssessmentSurface.CHECKIN: 5,
    AssessmentSurface.GUIDED_WIDGET: 1,
    AssessmentSurface.CAPSTONE: 2,
    AssessmentSurface.WORKED_EXAMPLE: 1,
}

EXPECTED_CONSTRUCT_ORDER: dict[str, dict[AssessmentSurface, tuple[str, ...]]] = {
    GRAPH_KC: {
        AssessmentSurface.DIAGNOSTIC: (
            "graph.point_value",
            "graph.intercepts",
            "graph.slope",
            "graph.behavior",
        ),
        AssessmentSurface.CHECKIN: (
            "graph.ordered_read",
            "graph.point_value",
            "graph.slope",
            "graph.behavior",
            "graph.intercepts",
        ),
        AssessmentSurface.GUIDED_WIDGET: ("graph.guided_mapping",),
        AssessmentSurface.CAPSTONE: ("graph.ordered_read", "graph.behavior"),
        AssessmentSurface.WORKED_EXAMPLE: ("graph.slope",),
    },
    AREA_KC: {
        AssessmentSurface.DIAGNOSTIC: (
            "area.rectangle",
            "area.triangle",
            "area.composite",
            "area.missing_height",
        ),
        AssessmentSurface.CHECKIN: (
            "area.ordered_parts",
            "area.rectangle",
            "area.triangle",
            "area.composite",
            "area.missing_height",
        ),
        AssessmentSurface.GUIDED_WIDGET: ("area.guided_slider",),
        AssessmentSurface.CAPSTONE: ("area.composite", "area.ordered_parts"),
        AssessmentSurface.WORKED_EXAMPLE: ("area.composite",),
    },
    RIEMANN_KC: {
        AssessmentSurface.DIAGNOSTIC: (
            "riemann.left",
            "riemann.right",
            "riemann.midpoint",
            "riemann.compare",
        ),
        AssessmentSurface.CHECKIN: (
            "riemann.contributions",
            "riemann.missing_height",
            "riemann.left",
            "riemann.right",
            "riemann.midpoint",
        ),
        AssessmentSurface.GUIDED_WIDGET: ("riemann.guided_mapping",),
        AssessmentSurface.CAPSTONE: ("riemann.compare", "riemann.contributions"),
        AssessmentSurface.WORKED_EXAMPLE: ("riemann.left",),
    },
    DEFINITE_KC: {
        AssessmentSurface.DIAGNOSTIC: (
            "definite.orientation",
            "definite.additivity",
            "definite.signed_regions",
            "definite.interpretation",
        ),
        AssessmentSurface.CHECKIN: (
            "definite.missing_piece",
            "definite.two_orientations",
            "definite.orientation",
            "definite.additivity",
            "definite.signed_regions",
        ),
        AssessmentSurface.GUIDED_WIDGET: ("definite.guided_mapping",),
        AssessmentSurface.CAPSTONE: (
            "definite.missing_piece",
            "definite.two_orientations",
        ),
        AssessmentSurface.WORKED_EXAMPLE: ("definite.additivity",),
    },
    ANTIDERIVATIVE_KC: {
        AssessmentSurface.DIAGNOSTIC: (
            "antiderivative.single",
            "antiderivative.binomial",
            "antiderivative.correction",
            "antiderivative.derivative_check",
        ),
        AssessmentSurface.CHECKIN: (
            "antiderivative.trinomial",
            "antiderivative.coefficient_audit",
            "antiderivative.binomial",
            "antiderivative.correction",
            "antiderivative.derivative_check",
        ),
        AssessmentSurface.GUIDED_WIDGET: ("antiderivative.guided_mapping",),
        AssessmentSurface.CAPSTONE: (
            "antiderivative.trinomial",
            "antiderivative.coefficient_audit",
        ),
        AssessmentSurface.WORKED_EXAMPLE: ("antiderivative.binomial",),
    },
    TARGET_KC: {
        AssessmentSurface.DIAGNOSTIC: (
            "ftc.supplied",
            "ftc.derive",
            "ftc.reversed",
            "ftc.correction",
        ),
        AssessmentSurface.CHECKIN: (
            "ftc.split",
            "ftc.ordered_intervals",
            "ftc.supplied",
            "ftc.derive",
            "ftc.reversed",
        ),
        AssessmentSurface.GUIDED_WIDGET: ("ftc.guided_slider",),
        AssessmentSurface.CAPSTONE: ("ftc.correction", "ftc.ordered_intervals"),
        AssessmentSurface.WORKED_EXAMPLE: ("ftc.supplied",),
    },
}

SEED_DIR = Path(__file__).resolve().parents[1] / "seed"
DEFAULT_SOURCE_PATH = SEED_DIR / "item_blueprints_ftc_v2.json"
DEFAULT_MANIFEST_PATH = SEED_DIR / "item_reviews_ftc_v2.json"
DEFAULT_BANK_PATH = SEED_DIR / "item_bank_ftc_v2.json"
DEFAULT_PEDAGOGY_SOURCE_PATH = SEED_DIR / "pedagogy_pack_sources_ftc_v2.json"
DEFAULT_PEDAGOGY_MANIFEST_PATH = SEED_DIR / "pedagogy_pack_reviews_ftc_v2.json"
DEFAULT_GRAPH_PATH = SEED_DIR / "kc_graph_calc1.json"
PRECEDING_BANK_PATHS = (
    SEED_DIR / "item_bank_product_quotient_v2.json",
    SEED_DIR / "item_bank_chain_rule_v2.json",
    SEED_DIR / "item_bank_solve_quadratics_v2.json",
)


class FTCCompilationError(ValueError):
    """The pending FTC wave cannot be compiled or qualified safely."""


@dataclass(frozen=True)
class InventorySeparationReport:
    answer_pairs_checked: int
    visible_candidate_comparisons_checked: int
    literal_visible_pairs_checked: int
    errors: tuple[str, ...]


@dataclass(frozen=True)
class MappingPlan:
    prompt: str
    rows: tuple[tuple[str, str], ...]
    options: tuple[tuple[str, str], ...]
    correct: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class SliderPlan:
    prompt: str
    label: str
    help_text: str
    target: int
    minimum: int
    maximum: int
    initial: int
    value_label: str


@dataclass(frozen=True)
class DerivedTask:
    instruction: str
    givens: tuple[PromptSegment, ...]
    answer: AnswerSpec
    conceptual_hint: str
    operation_hint: str
    worked_steps: tuple[PromptSegment, ...]
    error_signatures: tuple[ErrorSignature, ...]
    mapping_plan: MappingPlan | None = None
    slider_plan: SliderPlan | None = None

    @property
    def submission(self) -> str:
        return canonical_submission(self.answer)


def _spoken_math(expression: str) -> str:
    spoken = expression
    for source, replacement in (
        ("^", " to the power of "),
        ("*", " times "),
        ("/", " divided by "),
        ("+", " plus "),
        ("-", " minus "),
        ("=", " equals "),
        ("(", " open parenthesis "),
        (")", " close parenthesis "),
        ("[", " from "),
        ("]", " end bounds "),
        (",", " comma "),
    ):
        spoken = spoken.replace(source, replacement)
    return " ".join(spoken.split())


def _math(
    expression: str,
    *,
    role: PromptSemanticRole = PromptSemanticRole.GIVEN,
) -> MathPromptSegment:
    return MathPromptSegment(
        role=role,
        expression=expression,
        spoken_text=_spoken_math(expression),
    )


def _step(expression: str) -> MathPromptSegment:
    return _math(expression, role=PromptSemanticRole.WORKED_STEP)


def _text_step(text: str) -> TextPromptSegment:
    return TextPromptSegment(role=PromptSemanticRole.WORKED_STEP, text=text)


def _signature(
    wrong: str,
    misconception_id: str,
    implicated_prereq: str | None = None,
) -> ErrorSignature:
    return ErrorSignature(
        expected_wrong=wrong,
        misconception_id=misconception_id,
        implicated_prereq=implicated_prereq,
    )


def _tuple_answer(values: tuple[int | str, ...]) -> OrderedTupleAnswerSpec:
    return OrderedTupleAnswerSpec(expected=[str(value) for value in values], variables=["x"])


def canonical_submission(answer: AnswerSpec) -> str:
    if isinstance(
        answer,
        (NumericAnswerSpec, SymbolicAnswerSpec, AntiderivativeAnswerSpec),
    ):
        return answer.expected
    if isinstance(answer, FiniteSetAnswerSpec):
        return "{" + ", ".join(answer.expected) + "}"
    if isinstance(answer, OrderedTupleAnswerSpec):
        return "(" + ", ".join(answer.expected) + ")"
    if isinstance(answer, IntervalSetAnswerSpec):
        parts = []
        for interval in answer.expected:
            left = "[" if interval.lower_closed else "("
            right = "]" if interval.upper_closed else ")"
            parts.append(f"{left}{interval.lower}, {interval.upper}{right}")
        return " U ".join(parts)
    raise FTCCompilationError(f"unsupported FTC answer contract {answer.kind!r}")


def _table(
    caption: str,
    headers: tuple[str, ...],
    rows: tuple[tuple[str, ...], ...],
    spoken_text: str,
) -> TablePromptSegment:
    return TablePromptSegment(
        role=PromptSemanticRole.CONTEXT,
        caption=caption,
        column_headers=headers,
        rows=rows,
        spoken_text=spoken_text,
    )


def _graph_segment(graph: PiecewiseLinearSpec) -> PlotPromptSegment:
    rows = tuple((str(point.x), str(point.y)) for point in graph.points)
    spoken_points = ", ".join(
        f"x {point.x}, f of x {point.y}" for point in graph.points
    )
    equivalent = _table(
        "Exact points on the piecewise-linear graph",
        ("x", "f(x)"),
        rows,
        "The graph joins these exact points in order: " + spoken_points + ".",
    )
    return PlotPromptSegment(
        role=PromptSemanticRole.CONTEXT,
        title="Piecewise-linear graph of f",
        x_label="x",
        y_label="f(x)",
        series=(
            StaticPlotSeries(
                label="f",
                points=tuple(
                    StaticPlotPoint(x=str(point.x), y=str(point.y))
                    for point in graph.points
                ),
            ),
        ),
        spoken_text=(
            "A piecewise-linear graph joins the following exact points from left "
            "to right: "
            + spoken_points
            + "."
        ),
        equivalent_table=equivalent,
    )


def _slope(graph: PiecewiseLinearSpec, index: int) -> tuple[int, int, int]:
    left = graph.points[index]
    right = graph.points[index + 1]
    rise = right.y - left.y
    run = right.x - left.x
    return rise, run, rise // run


def _behavior_interval(task: GraphBehaviorTask) -> tuple[int, int]:
    matches = []
    for index, (left, right) in enumerate(zip(task.graph.points, task.graph.points[1:])):
        slope = right.y - left.y
        if (task.behavior == "increasing" and slope > 0) or (
            task.behavior == "decreasing" and slope < 0
        ):
            matches.append(index)
    return task.graph.points[matches[0]].x, task.graph.points[matches[-1] + 1].x


def _derive_graph(task: FTCMathTask) -> DerivedTask | None:
    prereq = "kc.fun.function_notation"
    if isinstance(task, GraphPointValueTask):
        point = task.graph.points[task.point_index]
        answer = _tuple_answer((point.x, point.y))
        return DerivedTask(
            instruction=(
                f"Read the graph at x = {point.x}. Report the ordered pair "
                "(input, function value)."
            ),
            givens=(_graph_segment(task.graph),),
            answer=answer,
            conceptual_hint="Find the named input on the horizontal axis, then read its height.",
            operation_hint="Keep the input first and the function value second in the pair.",
            worked_steps=(
                _text_step(f"Locate x = {point.x} in the exact point table."),
                _math(f"f({point.x})={point.y}", role=PromptSemanticRole.WORKED_STEP),
            ),
            error_signatures=(
                _signature(
                    f"({point.y}, {point.x})",
                    "m.graph_reading.axes_coordinates",
                    prereq,
                ),
                _signature(
                    f"({point.x}, {-point.y})",
                    "m.graph_reading.intercept_confusion",
                ),
            ),
        )
    if isinstance(task, GraphInterceptsTask):
        x_intercept = next(point.x for point in task.graph.points if point.y == 0)
        y_intercept = next(point.y for point in task.graph.points if point.x == 0)
        return DerivedTask(
            instruction=(
                "Read both intercepts. Report (x-coordinate of the x-intercept, "
                "y-coordinate of the y-intercept)."
            ),
            givens=(_graph_segment(task.graph),),
            answer=_tuple_answer((x_intercept, y_intercept)),
            conceptual_hint="An x-intercept has height zero; a y-intercept has input zero.",
            operation_hint="Read the remaining coordinate from each zero-coordinate point.",
            worked_steps=(
                _math(f"f({x_intercept})=0", role=PromptSemanticRole.WORKED_STEP),
                _math(f"f(0)={y_intercept}", role=PromptSemanticRole.WORKED_STEP),
            ),
            error_signatures=(
                _signature(
                    f"({y_intercept}, {x_intercept})",
                    "m.graph_reading.intercept_confusion",
                ),
                _signature(
                    f"({-x_intercept}, {y_intercept})",
                    "m.graph_reading.axes_coordinates",
                    prereq,
                ),
            ),
        )
    if isinstance(task, GraphSlopeTask):
        rise, run, slope = _slope(task.graph, task.segment_index)
        left = task.graph.points[task.segment_index]
        right = task.graph.points[task.segment_index + 1]
        return DerivedTask(
            instruction=(
                f"For the segment from x = {left.x} to x = {right.x}, report "
                "(rise, run, slope)."
            ),
            givens=(_graph_segment(task.graph),),
            answer=_tuple_answer((rise, run, slope)),
            conceptual_hint="Slope compares the vertical change with the horizontal change.",
            operation_hint="Compute right y minus left y, then right x minus left x.",
            worked_steps=(
                _math(f"rise={right.y}-{left.y}={rise}", role=PromptSemanticRole.WORKED_STEP),
                _math(f"run={right.x}-{left.x}={run}; slope={rise}/{run}={slope}", role=PromptSemanticRole.WORKED_STEP),
            ),
            error_signatures=(
                _signature(
                    f"({run}, {rise}, {run // rise if rise and run % rise == 0 else run})",
                    "m.graph_reading.slope_direction",
                ),
                _signature(
                    f"({-rise}, {run}, {-slope})",
                    "m.graph_reading.axes_coordinates",
                    prereq,
                ),
            ),
        )
    if isinstance(task, GraphBehaviorTask):
        lower, upper = _behavior_interval(task)
        answer = IntervalSetAnswerSpec(
            expected=[
                IntervalSpec(
                    lower=str(lower),
                    upper=str(upper),
                    lower_closed=False,
                    upper_closed=False,
                )
            ]
        )
        return DerivedTask(
            instruction=f"Give the open interval where the graph is {task.behavior}.",
            givens=(_graph_segment(task.graph),),
            answer=answer,
            conceptual_hint=(
                f"Move from left to right and keep the connected segment where y is {task.behavior}."
            ),
            operation_hint="Use the x-values at the two ends and write an open interval.",
            worked_steps=(
                _text_step(f"The graph moves {task.behavior} from x = {lower} to x = {upper}."),
                _math(f"interval=({lower},{upper})", role=PromptSemanticRole.WORKED_STEP),
            ),
            error_signatures=(
                _signature(
                    f"[{lower}, {upper}]",
                    "m.graph_reading.slope_direction",
                ),
                _signature(
                    f"({lower - 1}, {upper})",
                    "m.graph_reading.axes_coordinates",
                    prereq,
                ),
            ),
        )
    if isinstance(task, GraphOrderedReadTask):
        points = [task.graph.points[index] for index in task.point_indices]
        outputs = tuple(point.y for point in points)
        inputs = ", ".join(str(point.x) for point in points)
        return DerivedTask(
            instruction=f"Read f(x) at x = {inputs}, in that order. Report one ordered triple.",
            givens=(_graph_segment(task.graph),),
            answer=_tuple_answer(outputs),
            conceptual_hint="Read each requested input separately from the exact graph data.",
            operation_hint="Keep the outputs in the same order as the three inputs.",
            worked_steps=(
                _math(
                    ", ".join(f"f({point.x})={point.y}" for point in points),
                    role=PromptSemanticRole.WORKED_STEP,
                ),
                _math(
                    "outputs=(" + ",".join(str(value) for value in outputs) + ")",
                    role=PromptSemanticRole.WORKED_STEP,
                ),
            ),
            error_signatures=(
                _signature(
                    "(" + ", ".join(str(value) for value in reversed(outputs)) + ")",
                    "m.graph_reading.axes_coordinates",
                    prereq,
                ),
                _signature(
                    "(" + ", ".join(str(point.x) for point in points) + ")",
                    "m.graph_reading.intercept_confusion",
                ),
            ),
        )
    if isinstance(task, GraphGuidedMappingTask):
        points = [task.graph.points[index] for index in task.point_indices]
        outputs = tuple(point.y for point in points)
        rows = tuple((f"input.{index}", f"f({point.x})") for index, point in enumerate(points))
        options = tuple(
            (f"output.{index}", str(point.y)) for index, point in enumerate(reversed(points))
        )
        correct = tuple(
            (f"input.{index}", f"output.{len(points) - index - 1}")
            for index in range(len(points))
        )
        return DerivedTask(
            instruction="Read the three requested outputs and report them as an ordered triple.",
            givens=(_graph_segment(task.graph),),
            answer=_tuple_answer(outputs),
            conceptual_hint="Each input maps to the height directly above or below it.",
            operation_hint="Match each f(input) row to its exact output, then preserve row order.",
            worked_steps=(
                _text_step("Read each requested row from the equivalent table."),
                _math(
                    "outputs=(" + ",".join(str(value) for value in outputs) + ")",
                    role=PromptSemanticRole.WORKED_STEP,
                ),
            ),
            error_signatures=(
                _signature(
                    "(" + ", ".join(str(value) for value in reversed(outputs)) + ")",
                    "m.graph_reading.axes_coordinates",
                    prereq,
                ),
            ),
            mapping_plan=MappingPlan(
                prompt="Match each function input to its output on the exact graph.",
                rows=rows,
                options=options,
                correct=correct,
            ),
        )
    return None


def _rectangle_area(region: RectangleRegion) -> int:
    return region.width * region.height


def _triangle_area(region: TriangleRegion) -> int:
    return region.base * region.height // 2


def _composite_areas(region: CompositeRegionSpec) -> tuple[int, ...]:
    return tuple(
        [*(_rectangle_area(item) for item in region.rectangles),
         *(_triangle_area(item) for item in region.triangles)]
    )


def _region_table(region: CompositeRegionSpec) -> TablePromptSegment:
    rows = tuple(
        [
            (f"rectangle {index}", str(item.width), str(item.height), "rectangle")
            for index, item in enumerate(region.rectangles, start=1)
        ]
        + [
            (f"triangle {index}", str(item.base), str(item.height), "triangle")
            for index, item in enumerate(region.triangles, start=1)
        ]
    )
    return _table(
        "Nonoverlapping regions above the horizontal axis",
        ("region", "width or base", "height", "shape"),
        rows,
        "The region is partitioned into the listed nonoverlapping rectangles and triangles.",
    )


def _numeric_signatures(
    correct: int,
    wrongs: tuple[tuple[int, str, str | None], ...],
) -> tuple[ErrorSignature, ...]:
    return tuple(
        _signature(str(wrong), misconception, prerequisite)
        for wrong, misconception, prerequisite in wrongs
        if wrong != correct
    )


def _derive_area(task: FTCMathTask) -> DerivedTask | None:
    prereq = GRAPH_KC
    if isinstance(task, AreaRectangleTask):
        result = _rectangle_area(task.region)
        table = _table(
            "Rectangle above the horizontal axis",
            ("width", "height"),
            ((str(task.region.width), str(task.region.height)),),
            f"A rectangle has width {task.region.width} and height {task.region.height}.",
        )
        return DerivedTask(
            instruction="Find the nonnegative area of the rectangle.",
            givens=(table,),
            answer=NumericAnswerSpec(expected=str(result), tolerance=0),
            conceptual_hint="A rectangle fills every unit of its width at the same height.",
            operation_hint="Multiply width by height.",
            worked_steps=(
                _math(f"area={task.region.width}*{task.region.height}", role=PromptSemanticRole.WORKED_STEP),
                _math(f"area={result}", role=PromptSemanticRole.WORKED_STEP),
            ),
            error_signatures=_numeric_signatures(
                result,
                (
                    (task.region.width + task.region.height, "m.area_under_curve.omits_region", prereq),
                    (result // 2, "m.area_under_curve.triangle_factor", None),
                ),
            ),
        )
    if isinstance(task, AreaTriangleTask):
        result = _triangle_area(task.region)
        table = _table(
            "Triangle above the horizontal axis",
            ("base", "height"),
            ((str(task.region.base), str(task.region.height)),),
            f"A triangle has base {task.region.base} and perpendicular height {task.region.height}.",
        )
        return DerivedTask(
            instruction="Find the nonnegative area of the triangle.",
            givens=(table,),
            answer=NumericAnswerSpec(expected=str(result), tolerance=0),
            conceptual_hint="A triangle occupies half of the matching rectangle.",
            operation_hint="Multiply base by height, then divide by two.",
            worked_steps=(
                _math(f"area=({task.region.base}*{task.region.height})/2", role=PromptSemanticRole.WORKED_STEP),
                _math(f"area={result}", role=PromptSemanticRole.WORKED_STEP),
            ),
            error_signatures=_numeric_signatures(
                result,
                (
                    (task.region.base * task.region.height, "m.area_under_curve.triangle_factor", None),
                    (task.region.base + task.region.height, "m.area_under_curve.omits_region", prereq),
                ),
            ),
        )
    if isinstance(task, (AreaCompositeTask, AreaGuidedSliderTask)):
        parts = _composite_areas(task.region)
        result = sum(parts)
        signatures = _numeric_signatures(
            result,
            (
                (sum(parts[:-1]), "m.area_under_curve.omits_region", prereq),
                (
                    sum(_rectangle_area(item) for item in task.region.rectangles)
                    + sum(item.base * item.height for item in task.region.triangles),
                    "m.area_under_curve.triangle_factor",
                    None,
                ),
            ),
        )
        slider = None
        if isinstance(task, AreaGuidedSliderTask):
            spread = max(20, result // 3)
            slider = SliderPlan(
                prompt="Choose the sum of all listed region areas.",
                label="Total accumulated area",
                help_text="Use the arrow keys or slider in whole-number steps, then check.",
                target=result,
                minimum=max(0, result - spread),
                maximum=result + spread,
                initial=max(0, result - spread),
                value_label="Selected total area",
            )
        return DerivedTask(
            instruction="Find the total nonnegative area of all listed, nonoverlapping regions.",
            givens=(_region_table(task.region),),
            answer=NumericAnswerSpec(expected=str(result), tolerance=0),
            conceptual_hint="Partition the whole region into familiar shapes without overlap.",
            operation_hint="Find each rectangle or triangle area, then add every part once.",
            worked_steps=(
                _math("parts=(" + ",".join(str(value) for value in parts) + ")", role=PromptSemanticRole.WORKED_STEP),
                _math("total=" + "+".join(str(value) for value in parts) + f"={result}", role=PromptSemanticRole.WORKED_STEP),
            ),
            error_signatures=signatures,
            slider_plan=slider,
        )
    if isinstance(task, AreaMissingHeightTask):
        factor = 1 if task.shape == "rectangle" else 2
        total = task.width_or_base * task.height // factor
        table = _table(
            f"{task.shape.title()} with one unknown dimension",
            ("shape", "width or base", "area"),
            ((task.shape, str(task.width_or_base), str(total)),),
            f"The {task.shape} has width or base {task.width_or_base} and area {total}.",
        )
        return DerivedTask(
            instruction=(
                "Find the missing perpendicular height. Report "
                "(width or base, given area, missing height)."
            ),
            givens=(table,),
            answer=_tuple_answer((task.width_or_base, total, task.height)),
            conceptual_hint="Undo the area formula for the listed shape.",
            operation_hint=(
                "Divide area by width."
                if task.shape == "rectangle"
                else "Double the area, then divide by the base."
            ),
            worked_steps=(
                _math(
                    f"height={total}/{task.width_or_base}"
                    if task.shape == "rectangle"
                    else f"height=2*{total}/{task.width_or_base}",
                    role=PromptSemanticRole.WORKED_STEP,
                ),
                _math(f"height={task.height}", role=PromptSemanticRole.WORKED_STEP),
            ),
            error_signatures=(
                _signature(
                    f"({task.width_or_base}, {total}, {total // task.width_or_base})",
                    "m.area_under_curve.triangle_factor",
                ),
                _signature(
                    f"({task.width_or_base}, {total}, {total - task.width_or_base})",
                    "m.area_under_curve.uses_endpoint_height",
                    prereq,
                ),
            ),
        )
    if isinstance(task, AreaOrderedPartsTask):
        rectangle = _rectangle_area(task.rectangle)
        triangle = _triangle_area(task.triangle)
        total = rectangle + triangle
        region = CompositeRegionSpec(rectangles=(task.rectangle,), triangles=(task.triangle,))
        return DerivedTask(
            instruction="Report (rectangle area, triangle area, total area).",
            givens=(_region_table(region),),
            answer=_tuple_answer((rectangle, triangle, total)),
            conceptual_hint="Treat the two nonoverlapping shapes separately before combining them.",
            operation_hint="Use width times height for the rectangle and half base times height for the triangle.",
            worked_steps=(
                _math(f"rectangle={rectangle}; triangle={triangle}", role=PromptSemanticRole.WORKED_STEP),
                _math(f"total={rectangle}+{triangle}={total}", role=PromptSemanticRole.WORKED_STEP),
            ),
            error_signatures=(
                _signature(
                    f"({rectangle}, {task.triangle.base * task.triangle.height}, {rectangle + task.triangle.base * task.triangle.height})",
                    "m.area_under_curve.triangle_factor",
                ),
                _signature(
                    f"({rectangle}, {triangle}, {rectangle})",
                    "m.area_under_curve.omits_region",
                    prereq,
                ),
            ),
        )
    return None


def _endpoint_table(table: EndpointTableSpec) -> TablePromptSegment:
    rows = tuple(
        (str(table.lower + index * table.width), str(value))
        for index, value in enumerate(table.values)
    )
    return _table(
        "Endpoint samples on an equal-width partition",
        ("x", "f(x)"),
        rows,
        f"Endpoint samples are spaced {table.width} units apart and listed in increasing x order.",
    )


def _midpoint_table(table: MidpointTableSpec) -> TablePromptSegment:
    rows = tuple(
        (str(table.lower + table.width // 2 + index * table.width), str(value))
        for index, value in enumerate(table.values)
    )
    return _table(
        "Midpoint samples on an equal-width partition",
        ("subinterval midpoint", "f(midpoint)"),
        rows,
        f"The listed midpoints belong to subintervals of width {table.width}.",
    )


def _riemann_total(width: int, heights: tuple[int, ...]) -> int:
    return width * sum(heights)


def _derive_riemann(task: FTCMathTask) -> DerivedTask | None:
    prereq = AREA_KC
    if isinstance(task, (RiemannLeftTask, RiemannRightTask)):
        left = isinstance(task, RiemannLeftTask)
        method = "left" if left else "right"
        heights = task.table.values[:-1] if left else task.table.values[1:]
        result = _riemann_total(task.table.width, heights)
        other = _riemann_total(
            task.table.width,
            task.table.values[1:] if left else task.table.values[:-1],
        )
        return DerivedTask(
            instruction=(
                f"Compute the {method}-endpoint rectangle sum across all subintervals."
            ),
            givens=(_endpoint_table(task.table),),
            answer=NumericAnswerSpec(expected=str(result), tolerance=0),
            conceptual_hint=f"Each rectangle uses the {method} endpoint of its subinterval.",
            operation_hint=f"Add the chosen heights, then multiply by width {task.table.width}.",
            worked_steps=(
                _math(f"heights=({','.join(str(value) for value in heights)})", role=PromptSemanticRole.WORKED_STEP),
                _math(f"sum={task.table.width}*({'+'.join(str(value) for value in heights)})={result}", role=PromptSemanticRole.WORKED_STEP),
            ),
            error_signatures=_numeric_signatures(
                result,
                (
                    (other, "m.riemann_sums.endpoint_choice", prereq),
                    (sum(heights), "m.riemann_sums.omits_width", None),
                ),
            ),
        )
    if isinstance(task, RiemannMidpointTask):
        result = _riemann_total(task.table.width, task.table.values)
        return DerivedTask(
            instruction="Compute the midpoint rectangle sum across all subintervals.",
            givens=(_midpoint_table(task.table),),
            answer=NumericAnswerSpec(expected=str(result), tolerance=0),
            conceptual_hint="Use the sample at the center of each equal-width subinterval.",
            operation_hint=f"Add all midpoint heights, then multiply by width {task.table.width}.",
            worked_steps=(
                _math(f"midpoint heights=({','.join(str(value) for value in task.table.values)})", role=PromptSemanticRole.WORKED_STEP),
                _math(f"sum={task.table.width}*({'+'.join(str(value) for value in task.table.values)})={result}", role=PromptSemanticRole.WORKED_STEP),
            ),
            error_signatures=_numeric_signatures(
                result,
                (
                    (sum(task.table.values), "m.riemann_sums.omits_width", None),
                    (
                        task.table.width * sum(task.table.values[:-1]),
                        "m.riemann_sums.midpoint_confusion",
                        prereq,
                    ),
                ),
            ),
        )
    if isinstance(task, RiemannCompareTask):
        left = _riemann_total(task.table.width, task.table.values[:-1])
        right = _riemann_total(task.table.width, task.table.values[1:])
        return DerivedTask(
            instruction="Report (left-endpoint sum, right-endpoint sum).",
            givens=(_endpoint_table(task.table),),
            answer=_tuple_answer((left, right)),
            conceptual_hint="The two sums use the same widths but shift the chosen endpoint.",
            operation_hint="Use every value except the last for left; every value except the first for right.",
            worked_steps=(
                _math(f"left={task.table.width}*({'+'.join(str(v) for v in task.table.values[:-1])})={left}", role=PromptSemanticRole.WORKED_STEP),
                _math(f"right={task.table.width}*({'+'.join(str(v) for v in task.table.values[1:])})={right}", role=PromptSemanticRole.WORKED_STEP),
            ),
            error_signatures=(
                _signature(f"({right}, {left})", "m.riemann_sums.endpoint_choice", prereq),
                _signature(
                    f"({sum(task.table.values[:-1])}, {sum(task.table.values[1:])})",
                    "m.riemann_sums.omits_width",
                ),
            ),
        )
    if isinstance(task, RiemannMissingHeightTask):
        total = task.width * (sum(task.known_heights) + task.missing_height)
        height_sum = total // task.width
        known_sum = sum(task.known_heights)
        table = _table(
            "Equal-width rectangle data with one missing height",
            ("width", "known heights", "total sum"),
            ((str(task.width), ", ".join(str(v) for v in task.known_heights), str(total)),),
            "The equal rectangle width, all known heights, and the full rectangle sum are listed.",
        )
        return DerivedTask(
            instruction=(
                "Find the one missing rectangle height. Report "
                "(all-heights sum, known-heights sum, missing height)."
            ),
            givens=(table,),
            answer=_tuple_answer((height_sum, known_sum, task.missing_height)),
            conceptual_hint="First convert the total area back into the sum of all heights.",
            operation_hint="Divide by the common width, then subtract the known heights.",
            worked_steps=(
                _math(f"height sum={total}/{task.width}={total // task.width}", role=PromptSemanticRole.WORKED_STEP),
                _math(f"missing={total // task.width}-{sum(task.known_heights)}={task.missing_height}", role=PromptSemanticRole.WORKED_STEP),
            ),
            error_signatures=(
                _signature(
                    f"({total}, {known_sum}, {total - known_sum})",
                    "m.riemann_sums.omits_width",
                ),
                _signature(
                    f"({height_sum}, {known_sum}, {height_sum})",
                    "m.riemann_sums.endpoint_choice",
                    prereq,
                ),
            ),
        )
    if isinstance(task, RiemannContributionsTask):
        contributions = tuple(task.width * value for value in task.heights)
        total = sum(contributions)
        table = _table(
            "Selected rectangle heights",
            ("common width", "selected heights"),
            ((str(task.width), ", ".join(str(value) for value in task.heights)),),
            "All listed heights are used once with the listed common rectangle width.",
        )
        return DerivedTask(
            instruction="Report each rectangle contribution in order, followed by the total.",
            givens=(table,),
            answer=_tuple_answer((*contributions, total)),
            conceptual_hint="Each contribution is one width times its selected height.",
            operation_hint="Multiply every height by the common width, then add the contributions.",
            worked_steps=(
                _math("contributions=(" + ",".join(str(v) for v in contributions) + ")", role=PromptSemanticRole.WORKED_STEP),
                _math(f"total={'+'.join(str(v) for v in contributions)}={total}", role=PromptSemanticRole.WORKED_STEP),
            ),
            error_signatures=(
                _signature(
                    "(" + ", ".join(str(v) for v in (*task.heights, sum(task.heights))) + ")",
                    "m.riemann_sums.omits_width",
                ),
                _signature(
                    "(" + ", ".join(str(v) for v in (*reversed(contributions), total)) + ")",
                    "m.riemann_sums.endpoint_choice",
                    prereq,
                ),
            ),
        )
    if isinstance(task, RiemannGuidedMappingTask):
        heights = task.table.values[:-1]
        total = _riemann_total(task.table.width, heights)
        interval_labels = tuple(
            (
                f"interval.{index}",
                f"[{task.table.lower + index * task.table.width}, "
                f"{task.table.lower + (index + 1) * task.table.width}]",
            )
            for index in range(3)
        )
        option_labels = tuple(
            (f"height.{index}", str(value))
            for index, value in enumerate(reversed(heights))
        )
        correct = tuple(
            (f"interval.{index}", f"height.{2 - index}") for index in range(3)
        )
        return DerivedTask(
            instruction="Use left endpoints and compute the rectangle sum.",
            givens=(_endpoint_table(task.table),),
            answer=NumericAnswerSpec(expected=str(total), tolerance=0),
            conceptual_hint="A left sum assigns each interval the height at its left edge.",
            operation_hint=f"After matching, add the heights and multiply by width {task.table.width}.",
            worked_steps=(
                _math(f"left heights=({','.join(str(v) for v in heights)})", role=PromptSemanticRole.WORKED_STEP),
                _math(f"sum={task.table.width}*({'+'.join(str(v) for v in heights)})={total}", role=PromptSemanticRole.WORKED_STEP),
            ),
            error_signatures=_numeric_signatures(
                total,
                (
                    (_riemann_total(task.table.width, task.table.values[1:]), "m.riemann_sums.endpoint_choice", prereq),
                    (sum(heights), "m.riemann_sums.omits_width", None),
                ),
            ),
            mapping_plan=MappingPlan(
                prompt="Match each subinterval to its left-endpoint height.",
                rows=interval_labels,
                options=option_labels,
                correct=correct,
            ),
        )
    return None


def _integral_notation(lower: int, upper: int, name: str = "f") -> str:
    return f"integral_[{lower},{upper}] {name}(x) dx"


def _derive_definite(task: FTCMathTask) -> DerivedTask | None:
    prereq = RIEMANN_KC
    if isinstance(task, DefiniteOrientationTask):
        result = -task.forward_value
        return DerivedTask(
            instruction="Reverse the bounds and give the resulting accumulated value.",
            givens=(_math(f"{_integral_notation(task.lower, task.upper)}={task.forward_value}"),),
            answer=NumericAnswerSpec(expected=str(result), tolerance=0),
            conceptual_hint="Reversing the direction of accumulation reverses its sign.",
            operation_hint="Multiply the known value by negative one.",
            worked_steps=(
                _math(f"{_integral_notation(task.upper, task.lower)}=-({task.forward_value})", role=PromptSemanticRole.WORKED_STEP),
                _math(f"value={result}", role=PromptSemanticRole.WORKED_STEP),
            ),
            error_signatures=_numeric_signatures(
                result,
                ((task.forward_value, "m.definite_integral.bound_order", prereq),),
            ),
        )
    if isinstance(task, DefiniteAdditivityTask):
        result = task.left_value + task.right_value
        return DerivedTask(
            instruction="Use adjacent-interval additivity to find the whole integral.",
            givens=(
                _math(f"{_integral_notation(task.lower, task.split)}={task.left_value}"),
                _math(f"{_integral_notation(task.split, task.upper)}={task.right_value}"),
            ),
            answer=NumericAnswerSpec(expected=str(result), tolerance=0),
            conceptual_hint="Adjacent intervals cover the whole interval without overlap.",
            operation_hint="Add the two signed accumulated values.",
            worked_steps=(
                _math(f"whole={task.left_value}+({task.right_value})", role=PromptSemanticRole.WORKED_STEP),
                _math(f"whole={result}", role=PromptSemanticRole.WORKED_STEP),
            ),
            error_signatures=_numeric_signatures(
                result,
                (
                    (task.left_value - task.right_value, "m.definite_integral.breaks_additivity", prereq),
                    (abs(task.left_value) + abs(task.right_value), "m.definite_integral.ignores_sign", None),
                ),
            ),
        )
    if isinstance(task, DefiniteMissingPieceTask):
        result = task.total_value - task.left_value
        return DerivedTask(
            instruction="Find the missing signed accumulation on the right subinterval.",
            givens=(
                _math(f"{_integral_notation(task.lower, task.upper)}={task.total_value}"),
                _math(f"{_integral_notation(task.lower, task.split)}={task.left_value}"),
            ),
            answer=NumericAnswerSpec(expected=str(result), tolerance=0),
            conceptual_hint="The left piece plus the missing right piece equals the whole.",
            operation_hint="Subtract the known left piece from the whole value.",
            worked_steps=(
                _math(f"missing={task.total_value}-({task.left_value})", role=PromptSemanticRole.WORKED_STEP),
                _math(f"missing={result}", role=PromptSemanticRole.WORKED_STEP),
            ),
            error_signatures=_numeric_signatures(
                result,
                (
                    (task.total_value + task.left_value, "m.definite_integral.breaks_additivity", prereq),
                    (abs(task.total_value) - abs(task.left_value), "m.definite_integral.ignores_sign", None),
                ),
            ),
        )
    if isinstance(task, DefiniteSignedRegionsTask):
        result = sum(task.signed_areas)
        table = _table(
            "Signed areas from left to right",
            ("region", "signed area"),
            tuple(
                (str(index), str(value))
                for index, value in enumerate(task.signed_areas, start=1)
            ),
            "Positive entries lie above the axis and negative entries lie below it.",
        )
        return DerivedTask(
            instruction="Find the definite integral by adding the signed region areas.",
            givens=(table,),
            answer=NumericAnswerSpec(expected=str(result), tolerance=0),
            conceptual_hint="Area below the horizontal axis contributes negatively.",
            operation_hint="Add every signed entry exactly as written.",
            worked_steps=(
                _math("signed sum=" + "+".join(f"({v})" for v in task.signed_areas), role=PromptSemanticRole.WORKED_STEP),
                _math(f"signed sum={result}", role=PromptSemanticRole.WORKED_STEP),
            ),
            error_signatures=_numeric_signatures(
                result,
                (
                    (sum(abs(value) for value in task.signed_areas), "m.definite_integral.ignores_sign", None),
                    (sum(task.signed_areas[:-1]), "m.definite_integral.breaks_additivity", prereq),
                ),
            ),
        )
    if isinstance(task, DefiniteInterpretationTask):
        return DerivedTask(
            instruction="Report (starting bound, ending bound, signed accumulated value).",
            givens=(_math(f"{_integral_notation(task.lower, task.upper)}={task.value}"),),
            answer=_tuple_answer((task.lower, task.upper, task.value)),
            conceptual_hint="The lower written bound is the start; the upper bound is the end.",
            operation_hint="Keep the two bounds in written order and retain the sign of the value.",
            worked_steps=(
                _text_step(f"The accumulation starts at {task.lower} and ends at {task.upper}."),
                _math(f"interpretation=({task.lower},{task.upper},{task.value})", role=PromptSemanticRole.WORKED_STEP),
            ),
            error_signatures=(
                _signature(
                    f"({task.upper}, {task.lower}, {task.value})",
                    "m.definite_integral.bound_order",
                    prereq,
                ),
                _signature(
                    f"({task.lower}, {task.upper}, {abs(task.value)})",
                    "m.definite_integral.ignores_sign",
                ),
            ),
        )
    if isinstance(task, DefiniteTwoOrientationsTask):
        return DerivedTask(
            instruction="Report (forward value, reverse-bounds value).",
            givens=(_math(f"{_integral_notation(task.lower, task.upper)}={task.forward_value}"),),
            answer=_tuple_answer((task.forward_value, -task.forward_value)),
            conceptual_hint="The same interval in the opposite direction changes only the sign.",
            operation_hint="Keep the supplied forward value, then negate it for reverse bounds.",
            worked_steps=(
                _math(f"forward={task.forward_value}", role=PromptSemanticRole.WORKED_STEP),
                _math(f"reverse={-task.forward_value}", role=PromptSemanticRole.WORKED_STEP),
            ),
            error_signatures=(
                _signature(
                    f"({task.forward_value}, {task.forward_value})",
                    "m.definite_integral.bound_order",
                    prereq,
                ),
                _signature(
                    f"({abs(task.forward_value)}, {abs(task.forward_value)})",
                    "m.definite_integral.ignores_sign",
                ),
            ),
        )
    if isinstance(task, DefiniteGuidedMappingTask):
        whole = task.left_value + task.right_value
        rows = (
            ("piece.left", _integral_notation(task.lower, task.split)),
            ("piece.right", _integral_notation(task.split, task.upper)),
            ("piece.whole", _integral_notation(task.lower, task.upper)),
        )
        options = (
            ("value.whole", str(whole)),
            ("value.left", str(task.left_value)),
            ("value.right", str(task.right_value)),
        )
        correct = (
            ("piece.left", "value.left"),
            ("piece.right", "value.right"),
            ("piece.whole", "value.whole"),
        )
        return DerivedTask(
            instruction="Use the two adjacent pieces and report (left, right, whole).",
            givens=(
                _math(f"left value={task.left_value}; right value={task.right_value}"),
            ),
            answer=_tuple_answer((task.left_value, task.right_value, whole)),
            conceptual_hint="The whole interval is the union of the two adjacent pieces.",
            operation_hint="Keep each supplied piece value and add them for the whole.",
            worked_steps=(
                _math(f"whole={task.left_value}+({task.right_value})", role=PromptSemanticRole.WORKED_STEP),
                _math(f"values=({task.left_value},{task.right_value},{whole})", role=PromptSemanticRole.WORKED_STEP),
            ),
            error_signatures=(
                _signature(
                    f"({task.right_value}, {task.left_value}, {whole})",
                    "m.definite_integral.bound_order",
                    prereq,
                ),
                _signature(
                    f"({task.left_value}, {task.right_value}, {task.left_value - task.right_value})",
                    "m.definite_integral.breaks_additivity",
                ),
            ),
            mapping_plan=MappingPlan(
                prompt="Match each integral interval to its signed value.",
                rows=rows,
                options=options,
                correct=correct,
            ),
        )
    return None


def _render_term(coefficient: int, exponent: int) -> str:
    if exponent == 0:
        return str(coefficient)
    variable = "x" if exponent == 1 else f"x^{exponent}"
    if coefficient == 1:
        return variable
    if coefficient == -1:
        return "-" + variable
    return f"{coefficient}*{variable}"


def _render_terms(terms: tuple[tuple[int, int], ...]) -> str:
    rendered = [_render_term(coefficient, exponent) for coefficient, exponent in terms]
    expression = rendered[0]
    for term in rendered[1:]:
        expression += term if term.startswith("-") else "+" + term
    return expression


def _antiderivative_expression(polynomial: AntiderivativePolynomialSpec) -> str:
    return _render_terms(tuple((term.coefficient, term.exponent) for term in polynomial.terms))


def _derivative_terms(polynomial: AntiderivativePolynomialSpec) -> tuple[tuple[int, int], ...]:
    return tuple(
        (term.coefficient * term.exponent, term.exponent - 1)
        for term in polynomial.terms
    )


def _derivative_expression(polynomial: AntiderivativePolynomialSpec) -> str:
    return _render_terms(_derivative_terms(polynomial))


def _antiderivative_wrong(
    polynomial: AntiderivativePolynomialSpec,
    mistake: str,
) -> str:
    derivative = _derivative_terms(polynomial)
    if mistake == "kept_power":
        return _render_terms(
            tuple(
                (coefficient, term.exponent)
                for (coefficient, _), term in zip(derivative, polynomial.terms, strict=True)
            )
        )
    if mistake == "did_not_divide":
        return _render_terms(
            tuple((coefficient, exponent + 1) for coefficient, exponent in derivative)
        )
    if mistake == "dropped_term":
        terms = polynomial.terms[:-1]
        return (
            _render_terms(tuple((term.coefficient, term.exponent) for term in terms))
            if terms
            else "0"
        )
    raise FTCCompilationError(f"unknown antiderivative mistake {mistake!r}")


def _derive_antiderivative(task: FTCMathTask) -> DerivedTask | None:
    supported = (
        AntiderivativeSingleTask,
        AntiderivativeBinomialTask,
        AntiderivativeTrinomialTask,
        AntiderivativeCorrectionTask,
        AntiderivativeDerivativeCheckTask,
        AntiderivativeCoefficientAuditTask,
        AntiderivativeGuidedMappingTask,
    )
    if not isinstance(task, supported):
        return None
    polynomial = task.polynomial
    integrand = _derivative_expression(polynomial)
    expected = _antiderivative_expression(polynomial)
    answer = AntiderivativeAnswerSpec(expected=f"{expected}+C", variable="x")
    if isinstance(task, AntiderivativeCorrectionTask):
        wrong_candidate = _antiderivative_wrong(polynomial, task.mistake)
        instruction = "Correct the proposed antiderivative and give the full family."
        givens: tuple[PromptSegment, ...] = (
            _math(f"f(x)={integrand}"),
            _math(f"proposed F(x)={wrong_candidate}"),
        )
    elif isinstance(task, AntiderivativeDerivativeCheckTask):
        instruction = "Find an antiderivative, then use differentiation to check it."
        givens = (_math(f"f(x)={integrand}"),)
    elif isinstance(task, AntiderivativeCoefficientAuditTask):
        instruction = "Reverse the power rule term by term and give the antiderivative family."
        givens = (_math(f"f(x)={integrand}"),)
    else:
        instruction = "Find the polynomial antiderivative family."
        givens = (_math(f"f(x)={integrand}"),)
    kept = _antiderivative_wrong(polynomial, "kept_power")
    undivided = _antiderivative_wrong(polynomial, "did_not_divide")
    dropped = _antiderivative_wrong(polynomial, "dropped_term")
    signatures = tuple(
        signature
        for signature in (
            _signature(kept, "m.antiderivatives.keeps_exponent", "kc.der.power_rule"),
            _signature(
                undivided,
                "m.antiderivatives.multiplies_coefficient",
                "kc.der.sum_constant_rules",
            ),
            _signature(
                dropped,
                "m.antiderivatives.drops_term",
                "kc.der.sum_constant_rules",
            ),
        )
        if signature.expected_wrong != answer.expected
    )
    mapping = None
    if isinstance(task, AntiderivativeGuidedMappingTask):
        derivative_terms = _derivative_terms(polynomial)
        rows = tuple(
            (f"derivative.{index}", _render_term(*term))
            for index, term in enumerate(derivative_terms)
        )
        options = tuple(
            (f"anti.{index}", _render_term(term.coefficient, term.exponent))
            for index, term in reversed(list(enumerate(polynomial.terms)))
        )
        correct = tuple(
            (f"derivative.{index}", f"anti.{index}")
            for index in range(len(polynomial.terms))
        )
        mapping = MappingPlan(
            prompt="Match each derivative term to the term that differentiates back to it.",
            rows=rows,
            options=options,
            correct=correct,
        )
    return DerivedTask(
        instruction=instruction,
        givens=givens,
        answer=answer,
        conceptual_hint="An antiderivative reverses differentiation; constants disappear when differentiated.",
        operation_hint="For each x power, raise the exponent by one and divide the coefficient by that new exponent.",
        worked_steps=(
            _text_step(f"Reverse each power-rule term in {integrand}."),
            _text_step("Combine the reversed terms and include one arbitrary constant."),
        ),
        error_signatures=signatures,
        mapping_plan=mapping,
    )


def _evaluate_polynomial(polynomial: AntiderivativePolynomialSpec, value: int) -> int:
    return sum(term.coefficient * value**term.exponent for term in polynomial.terms)


def _ftc_value(polynomial: AntiderivativePolynomialSpec, lower: int, upper: int) -> int:
    return _evaluate_polynomial(polynomial, upper) - _evaluate_polynomial(polynomial, lower)


def _derive_ftc(task: FTCMathTask) -> DerivedTask | None:
    supported = (
        FTCSuppliedTask,
        FTCDeriveTask,
        FTCReversedTask,
        FTCSplitTask,
        FTCCorrectionTask,
        FTCOrderedIntervalsTask,
        FTCGuidedSliderTask,
    )
    if not isinstance(task, supported):
        return None
    polynomial = task.polynomial
    integrand = _derivative_expression(polynomial)
    anti = _antiderivative_expression(polynomial)
    mapping = None
    slider = None
    if isinstance(task, FTCOrderedIntervalsTask):
        first = _ftc_value(polynomial, *task.first_bounds)
        second = _ftc_value(polynomial, *task.second_bounds)
        answer: AnswerSpec = _tuple_answer((first, second))
        instruction = "Evaluate both definite integrals and report their values in the listed order."
        givens = (
            _math(_integral_notation(*task.first_bounds, name="f")),
            _math(_integral_notation(*task.second_bounds, name="f")),
            _math(f"f(x)={integrand}"),
        )
        worked = (
            _math(f"first=F({task.first_bounds[1]})-F({task.first_bounds[0]})={first}", role=PromptSemanticRole.WORKED_STEP),
            _math(f"second=F({task.second_bounds[1]})-F({task.second_bounds[0]})={second}", role=PromptSemanticRole.WORKED_STEP),
        )
        signatures = (
            _signature(f"({second}, {first})", "m.ftc.reverses_subtraction", DEFINITE_KC),
            _signature(f"({-first}, {-second})", "m.ftc.reverses_subtraction", DEFINITE_KC),
        )
        return DerivedTask(
            instruction=instruction,
            givens=givens,
            answer=answer,
            conceptual_hint="Use one antiderivative and evaluate upper minus lower for each interval.",
            operation_hint="Compute F(upper) - F(lower) twice, preserving the listed order.",
            worked_steps=worked,
            error_signatures=signatures,
        )
    lower = task.lower
    upper = task.upper
    forward = _ftc_value(polynomial, lower, upper)
    result = -forward if isinstance(task, FTCReversedTask) else forward
    shown_lower, shown_upper = (
        (upper, lower) if isinstance(task, FTCReversedTask) else (lower, upper)
    )
    integrand_delta = (
        sum(coefficient * upper**exponent for coefficient, exponent in _derivative_terms(polynomial))
        - sum(coefficient * lower**exponent for coefficient, exponent in _derivative_terms(polynomial))
    )
    wrong_added = _evaluate_polynomial(polynomial, shown_upper) + _evaluate_polynomial(
        polynomial, shown_lower
    )
    wrong_reversed = -result
    wrong_integrand = -integrand_delta if isinstance(task, FTCReversedTask) else integrand_delta
    signatures = _numeric_signatures(
        result,
        (
            (wrong_added, "m.ftc.adds_endpoint_values", ANTIDERIVATIVE_KC),
            (wrong_reversed, "m.ftc.reverses_subtraction", DEFINITE_KC),
            (wrong_integrand, "m.ftc.uses_integrand_values", ANTIDERIVATIVE_KC),
        ),
    )
    if isinstance(task, FTCSplitTask):
        left = _ftc_value(polynomial, task.lower, task.split)
        right = _ftc_value(polynomial, task.split, task.upper)
        instruction = "Evaluate the two adjacent pieces, then give the integral over the whole interval."
        givens = (
            _math(f"f(x)={integrand}"),
            _math(f"split point={task.split}; whole bounds=[{lower},{upper}]"),
        )
        worked = (
            _math(f"pieces=({left},{right})", role=PromptSemanticRole.WORKED_STEP),
            _math(f"whole={left}+({right})={result}", role=PromptSemanticRole.WORKED_STEP),
        )
    elif isinstance(task, FTCCorrectionTask):
        wrong_by_kind = {
            "added_endpoints": wrong_added,
            "reversed_subtraction": wrong_reversed,
            "used_integrand": wrong_integrand,
        }
        instruction = "Correct the proposed FTC evaluation and give the definite-integral value."
        givens = (
            _math(f"{_integral_notation(lower, upper)} with f(x)={integrand}"),
            _math(f"proposed value={wrong_by_kind[task.mistake]}"),
        )
        worked = (
            _math(f"F({upper})-F({lower})", role=PromptSemanticRole.WORKED_STEP),
            _math(f"value={result}", role=PromptSemanticRole.WORKED_STEP),
        )
    else:
        supplied = isinstance(task, (FTCSuppliedTask, FTCGuidedSliderTask))
        instruction = (
            "Use the supplied antiderivative to evaluate the definite integral."
            if supplied
            else "Derive a polynomial antiderivative, then evaluate the definite integral."
        )
        givens_list: list[PromptSegment] = [
            _math(f"{_integral_notation(shown_lower, shown_upper)} with f(x)={integrand}")
        ]
        if supplied:
            givens_list.append(_math(f"Use F(x)={anti}"))
        givens = tuple(givens_list)
        worked = (
            _math(f"F(x)={anti}", role=PromptSemanticRole.WORKED_STEP),
            _math(f"F({shown_upper})-F({shown_lower})={result}", role=PromptSemanticRole.WORKED_STEP),
        )
    if isinstance(task, FTCGuidedSliderTask):
        spread = max(20, abs(result) // 3)
        slider = SliderPlan(
            prompt="Choose the upper-endpoint value minus the lower-endpoint value.",
            label="Definite-integral value",
            help_text="Use arrow keys or the slider in whole-number steps, then check.",
            target=result,
            minimum=result - spread,
            maximum=result + spread,
            initial=result - spread,
            value_label="Selected integral value",
        )
    return DerivedTask(
        instruction=instruction,
        givens=givens,
        answer=NumericAnswerSpec(expected=str(result), tolerance=0),
        conceptual_hint="The definite integral equals an antiderivative at the ending bound minus its value at the starting bound.",
        operation_hint=f"Use F({shown_upper}) - F({shown_lower}), keeping the bound order exactly as written.",
        worked_steps=worked,
        error_signatures=signatures,
        mapping_plan=mapping,
        slider_plan=slider,
    )


def _compile_registered(task: BaseModel) -> DerivedTask:
    typed = cast(FTCMathTask, task)
    for compiler in (
        _derive_graph,
        _derive_area,
        _derive_riemann,
        _derive_definite,
        _derive_antiderivative,
        _derive_ftc,
    ):
        derived = compiler(typed)
        if derived is not None:
            return derived
    raise FTCCompilationError(f"no deterministic compiler for {type(task).__name__}")


def _registration(
    task_type: type[BaseModel],
    construct_id: str,
    kc_id: str,
) -> TaskCompilerRegistration:
    kind = task_type.model_fields["kind"].default
    if not isinstance(kind, str):
        raise TypeError(f"{task_type.__name__}.kind must have a string default")
    return TaskCompilerRegistration(
        kind=kind,
        task_type=task_type,
        construct_id=construct_id,
        kc_id=kc_id,
        compile=_compile_registered,
    )


_TASK_COMPILER_REGISTRY = TaskCompilerRegistry(
    (
        _registration(GraphPointValueTask, "graph.point_value", GRAPH_KC),
        _registration(GraphInterceptsTask, "graph.intercepts", GRAPH_KC),
        _registration(GraphSlopeTask, "graph.slope", GRAPH_KC),
        _registration(GraphBehaviorTask, "graph.behavior", GRAPH_KC),
        _registration(GraphOrderedReadTask, "graph.ordered_read", GRAPH_KC),
        _registration(GraphGuidedMappingTask, "graph.guided_mapping", GRAPH_KC),
        _registration(AreaRectangleTask, "area.rectangle", AREA_KC),
        _registration(AreaTriangleTask, "area.triangle", AREA_KC),
        _registration(AreaCompositeTask, "area.composite", AREA_KC),
        _registration(AreaMissingHeightTask, "area.missing_height", AREA_KC),
        _registration(AreaOrderedPartsTask, "area.ordered_parts", AREA_KC),
        _registration(AreaGuidedSliderTask, "area.guided_slider", AREA_KC),
        _registration(RiemannLeftTask, "riemann.left", RIEMANN_KC),
        _registration(RiemannRightTask, "riemann.right", RIEMANN_KC),
        _registration(RiemannMidpointTask, "riemann.midpoint", RIEMANN_KC),
        _registration(RiemannCompareTask, "riemann.compare", RIEMANN_KC),
        _registration(RiemannMissingHeightTask, "riemann.missing_height", RIEMANN_KC),
        _registration(RiemannContributionsTask, "riemann.contributions", RIEMANN_KC),
        _registration(RiemannGuidedMappingTask, "riemann.guided_mapping", RIEMANN_KC),
        _registration(DefiniteOrientationTask, "definite.orientation", DEFINITE_KC),
        _registration(DefiniteAdditivityTask, "definite.additivity", DEFINITE_KC),
        _registration(DefiniteMissingPieceTask, "definite.missing_piece", DEFINITE_KC),
        _registration(DefiniteSignedRegionsTask, "definite.signed_regions", DEFINITE_KC),
        _registration(DefiniteInterpretationTask, "definite.interpretation", DEFINITE_KC),
        _registration(DefiniteTwoOrientationsTask, "definite.two_orientations", DEFINITE_KC),
        _registration(DefiniteGuidedMappingTask, "definite.guided_mapping", DEFINITE_KC),
        _registration(AntiderivativeSingleTask, "antiderivative.single", ANTIDERIVATIVE_KC),
        _registration(AntiderivativeBinomialTask, "antiderivative.binomial", ANTIDERIVATIVE_KC),
        _registration(AntiderivativeTrinomialTask, "antiderivative.trinomial", ANTIDERIVATIVE_KC),
        _registration(AntiderivativeCorrectionTask, "antiderivative.correction", ANTIDERIVATIVE_KC),
        _registration(
            AntiderivativeDerivativeCheckTask,
            "antiderivative.derivative_check",
            ANTIDERIVATIVE_KC,
        ),
        _registration(
            AntiderivativeCoefficientAuditTask,
            "antiderivative.coefficient_audit",
            ANTIDERIVATIVE_KC,
        ),
        _registration(
            AntiderivativeGuidedMappingTask,
            "antiderivative.guided_mapping",
            ANTIDERIVATIVE_KC,
        ),
        _registration(FTCSuppliedTask, "ftc.supplied", TARGET_KC),
        _registration(FTCDeriveTask, "ftc.derive", TARGET_KC),
        _registration(FTCReversedTask, "ftc.reversed", TARGET_KC),
        _registration(FTCSplitTask, "ftc.split", TARGET_KC),
        _registration(FTCCorrectionTask, "ftc.correction", TARGET_KC),
        _registration(FTCOrderedIntervalsTask, "ftc.ordered_intervals", TARGET_KC),
        _registration(FTCGuidedSliderTask, "ftc.guided_slider", TARGET_KC),
    )
)


def derive_task(task: FTCMathTask) -> DerivedTask:
    """Compile one validated source task through the closed registry."""

    derived = _TASK_COMPILER_REGISTRY.compile(task)
    if not isinstance(derived, DerivedTask):
        raise FTCCompilationError(f"task compiler returned {type(derived).__name__}")
    return derived


def _opaque_entry_id(
    family_id: str,
    namespace: str,
    semantic_id: str,
    salt: int,
) -> str:
    payload = f"{family_id}\0{namespace}\0{semantic_id}\0{salt}".encode()
    return "entry." + hashlib.sha256(payload).hexdigest()[:16]


def _opaque_guided_mapping(
    family: FTCFamilyBlueprint,
    plan: MappingPlan,
) -> GuidedMappingSpec:
    """Compile a mapping whose identifiers and ordering do not reveal truth."""

    row_labels = dict(plan.rows)
    option_labels = dict(plan.options)
    truth = dict(plan.correct)
    if set(truth) != set(row_labels) or set(truth.values()) != set(option_labels):
        raise FTCCompilationError(
            f"{family.item_id}: guided semantic truth is not a complete bijection"
        )
    row_semantics = tuple(
        sorted(
            row_labels,
            key=lambda value: hashlib.sha256(
                f"{family.family_id}\0row-order\0{value}".encode()
            ).hexdigest(),
        )
    )
    permutations = [
        candidate
        for candidate in itertools.permutations(option_labels)
        if all(truth[row] != option for row, option in zip(row_semantics, candidate, strict=True))
    ]
    if not permutations:
        raise FTCCompilationError(
            f"{family.item_id}: no non-positional guided ordering exists"
        )
    selector = int(
        hashlib.sha256(f"{family.family_id}\0option-order".encode()).hexdigest(),
        16,
    )
    option_semantics = permutations[selector % len(permutations)]
    for salt in range(256):
        row_ids = {
            semantic: _opaque_entry_id(family.family_id, "row", semantic, salt)
            for semantic in row_labels
        }
        option_ids = {
            semantic: _opaque_entry_id(family.family_id, "option", semantic, salt)
            for semantic in option_labels
        }
        sorted_pairing = dict(
            zip(sorted(row_ids.values()), sorted(option_ids.values()), strict=True)
        )
        actual_pairs = {
            row_ids[row]: option_ids[option] for row, option in truth.items()
        }
        if sorted_pairing != actual_pairs:
            break
    else:  # pragma: no cover - cryptographic ids make this unreachable
        raise FTCCompilationError(f"{family.item_id}: could not hide guided truth")
    return GuidedMappingSpec(
        presentation=GuidedMappingPresentation(
            prompt=plan.prompt,
            rows=tuple(
                GuidedMappingEntry(
                    entry_id=row_ids[semantic],
                    label=row_labels[semantic],
                    spoken_text=_spoken_math(row_labels[semantic]),
                )
                for semantic in row_semantics
            ),
            options=tuple(
                GuidedMappingEntry(
                    entry_id=option_ids[semantic],
                    label=option_labels[semantic],
                    spoken_text=_spoken_math(option_labels[semantic]),
                )
                for semantic in option_semantics
            ),
        ),
        scoring=GuidedMappingScoring(
            correct_pairs=tuple(
                (row_ids[row], option_ids[option]) for row, option in plan.correct
            )
        ),
    )


def _guided_interaction_for(
    family: FTCFamilyBlueprint,
    derived: DerivedTask,
) -> GuidedInteractionSpec | None:
    if family.surface != AssessmentSurface.GUIDED_WIDGET:
        if derived.mapping_plan is not None or derived.slider_plan is not None:
            raise FTCCompilationError(
                f"{family.item_id}: non-guided family compiled private widget truth"
            )
        return None
    plans = [derived.mapping_plan is not None, derived.slider_plan is not None]
    if sum(plans) != 1:
        raise FTCCompilationError(
            f"{family.item_id}: guided family requires exactly one interaction plan"
        )
    if derived.mapping_plan is not None:
        return _opaque_guided_mapping(family, derived.mapping_plan)
    assert derived.slider_plan is not None
    plan = derived.slider_plan
    return GuidedSliderSpec(
        presentation=GuidedSliderPresentation(
            prompt=plan.prompt,
            label=plan.label,
            help_text=plan.help_text,
            minimum=plan.minimum,
            maximum=plan.maximum,
            step=1,
            initial_value=plan.initial,
            value_label=plan.value_label,
            result_template=plan.value_label + ": {value}.",
        ),
        scoring=GuidedSliderScoring(target=plan.target, tolerance=0),
    )


def _review_status_and_provenance(
    source: FTCBlueprintDocument,
    family: FTCFamilyBlueprint,
    review: ContentReviewEntry,
) -> tuple[ReviewStatus, AssessmentProvenance]:
    if review.decision == ReviewDecision.REJECTED:
        raise FTCCompilationError("rejected families cannot be compiled")
    approved = review.decision == ReviewDecision.APPROVED
    if approved:
        if review.reviewed_by is None or review.reviewed_at is None:
            raise FTCCompilationError("approved family lacks review provenance")
        if review.reviewed_by.strip().casefold() == source.author.strip().casefold():
            raise FTCCompilationError("a family author cannot approve their own work")
    return (
        ReviewStatus.HUMAN_APPROVED if approved else ReviewStatus.DRAFT,
        AssessmentProvenance(
            source=source.authoring_source,
            author=source.author,
            reviewed_by=review.reviewed_by if approved else None,
            reviewed_at=review.reviewed_at if approved else None,
            source_id=family.blueprint_id,
            source_revision=family.revision,
            source_digest=review.source_digest,
            compiler_version=COMPILER_VERSION,
        ),
    )


def _validate_executable_signatures(item: AssessmentItem) -> None:
    for signature in item.error_signatures:
        verdict = verify_answer(item.answer, signature.expected_wrong, supervised=False)
        if verdict.status != VerificationStatus.INCORRECT:
            raise FTCCompilationError(
                f"{item.item_id}: error signature is not an executable wrong answer "
                f"({verdict.code})"
            )


def _executable_wrong_signatures(
    answer: AnswerSpec,
    signatures: tuple[ErrorSignature, ...],
) -> list[ErrorSignature]:
    executable: list[ErrorSignature] = []
    for signature in signatures:
        verdict = verify_answer(answer, signature.expected_wrong, supervised=False)
        if verdict.status == VerificationStatus.INCORRECT:
            executable.append(signature)
        elif verdict.status != VerificationStatus.CORRECT:
            raise FTCCompilationError(
                "derived error signature has invalid syntax "
                f"({signature.expected_wrong!r}, {verdict.code})"
            )
    return executable


def _build_item(
    source: FTCBlueprintDocument,
    family: FTCFamilyBlueprint,
    *,
    review_status: ReviewStatus,
    provenance: AssessmentProvenance,
) -> AssessmentItem:
    derived = derive_task(family.task)
    if family.surface == AssessmentSurface.WORKED_EXAMPLE:
        if len(derived.worked_steps) < 2:
            raise FTCCompilationError(
                f"{family.item_id}: worked example lacks a real derivation"
            )
        prompt: list[PromptSegment] = [
            TextPromptSegment(
                role=PromptSemanticRole.INSTRUCTION,
                text="Study this worked example. " + derived.instruction,
            ),
            *derived.givens,
            TextPromptSegment(
                role=PromptSemanticRole.WORKED_STEP,
                text="Follow the intermediate reasoning before reading the result.",
            ),
            *derived.worked_steps,
            MathPromptSegment(
                role=PromptSemanticRole.WORKED_ANSWER,
                expression=derived.submission,
                spoken_text=_spoken_math(derived.submission),
            ),
        ]
    else:
        guided_prefix = (
            "Use the guided activity or enter its equivalent text answer. "
            if family.surface == AssessmentSurface.GUIDED_WIDGET
            else ""
        )
        prompt = [
            TextPromptSegment(
                role=PromptSemanticRole.INSTRUCTION,
                text=guided_prefix + derived.instruction,
            ),
            *derived.givens,
            BlankPromptSegment(label="Answer:"),
        ]
    item = AssessmentItem(
        item_id=family.item_id,
        revision=family.revision,
        family_id=family.family_id,
        kc_id=family.kc_id,
        difficulty=family.difficulty,
        eligible_surfaces=[family.surface],
        allocation_order=family.allocation_order,
        prompt=prompt,
        hints=[
            AssessmentHint(text=derived.conceptual_hint),
            AssessmentHint(text=derived.operation_hint),
            AssessmentHint(
                text=(
                    "Reveal the answer and move to a new problem: "
                    f"{derived.submission}."
                ),
                revealing=True,
            ),
        ],
        answer=derived.answer,
        review_status=review_status,
        provenance=provenance,
        error_signatures=_executable_wrong_signatures(
            derived.answer,
            derived.error_signatures,
        ),
        guided_interaction=_guided_interaction_for(family, derived),
    )
    _validate_executable_signatures(item)
    return item


def _compiled_review_artifact(
    source: FTCBlueprintDocument,
    family: FTCFamilyBlueprint,
) -> dict[str, object]:
    item = _build_item(
        source,
        family,
        review_status=ReviewStatus.DRAFT,
        provenance=AssessmentProvenance(
            source=source.authoring_source,
            author=source.author,
            source_id=family.blueprint_id,
            source_revision=family.revision,
            source_digest="0" * 64,
            compiler_version=COMPILER_VERSION,
        ),
    )
    artifact = item.model_dump(mode="json")
    artifact.pop("review_status")
    artifact_provenance = artifact["provenance"]
    if not isinstance(artifact_provenance, dict):
        raise TypeError("compiled provenance must serialize as an object")
    for field in ("reviewed_by", "reviewed_at", "source_digest"):
        artifact_provenance.pop(field, None)
    return artifact


def family_digest(
    source: FTCBlueprintDocument,
    family: FTCFamilyBlueprint,
) -> str:
    """Bind one family review to source math and exact compiled bytes."""

    canonical = json.dumps(
        {
            "authorship": {
                "author": source.author,
                "authoring_source": source.authoring_source,
            },
            "compiler_version": COMPILER_VERSION,
            "compiled_artifact": _compiled_review_artifact(source, family),
            "family_blueprint": family.model_dump(mode="json"),
            "graph_version": source.graph_version,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_source(path: Path | None = None) -> FTCBlueprintDocument:
    source = path or DEFAULT_SOURCE_PATH
    return FTCBlueprintDocument.model_validate_json(source.read_text(encoding="utf-8"))


def load_manifest(path: Path | None = None) -> ContentReviewManifest:
    source = path or DEFAULT_MANIFEST_PATH
    return ContentReviewManifest.model_validate_json(source.read_text(encoding="utf-8"))


def draft_review_manifest(source: FTCBlueprintDocument) -> ContentReviewManifest:
    return ContentReviewManifest(
        manifest_version="ftc-review-manifest-v2",
        graph_version=source.graph_version,
        compiler_version=COMPILER_VERSION,
        entries=[
            ContentReviewEntry(
                blueprint_id=family.blueprint_id,
                revision=family.revision,
                source_digest=family_digest(source, family),
                decision=ReviewDecision.PENDING,
            )
            for family in source.families
        ],
    )


def draft_pedagogy_review_manifest(
    source: PedagogySourceDocument,
) -> PedagogyReviewManifest:
    return PedagogyReviewManifest(
        manifest_version="ftc-pedagogy-review-manifest-v2",
        graph_version=source.graph_version,
        compiler_version=PEDAGOGY_COMPILER_VERSION,
        entries=tuple(
            PedagogyReviewEntry(
                source_id=pack.source_id,
                revision=pack.revision,
                source_digest=pedagogy_source_digest(pack),
                decision=PedagogyReviewDecision.PENDING,
            )
            for pack in source.pack_sources
        ),
    )


def _segment_fragments(segment: PromptSegment) -> list[str]:
    if isinstance(segment, TextPromptSegment):
        return [segment.text]
    if isinstance(segment, MathPromptSegment):
        return [segment.expression, segment.spoken_text or ""]
    if isinstance(segment, TablePromptSegment):
        return [
            segment.caption,
            segment.spoken_text,
            *segment.column_headers,
            *(cell for row in segment.rows for cell in row),
        ]
    if isinstance(segment, PlotPromptSegment):
        fragments = [
            segment.title,
            segment.x_label,
            segment.y_label,
            segment.spoken_text,
        ]
        fragments.extend(
            value
            for series in segment.series
            for point in series.points
            for value in (point.x, point.y)
        )
        if segment.equivalent_table is not None:
            fragments.extend(_segment_fragments(segment.equivalent_table))
        return fragments
    return []


def _visible_fragments(item: AssessmentItem) -> list[str]:
    fragments = [render_prompt(item)]
    for segment in item.prompt:
        fragments.extend(_segment_fragments(segment))
    fragments.extend(hint.text for hint in item.hints)
    interaction = item.guided_interaction
    if isinstance(interaction, GuidedMappingSpec):
        fragments.append(interaction.presentation.prompt)
        fragments.extend(
            value
            for entry in (
                *interaction.presentation.rows,
                *interaction.presentation.options,
            )
            for value in (entry.label, entry.spoken_text)
        )
    elif isinstance(interaction, GuidedSliderSpec):
        presentation = interaction.presentation
        fragments.extend(
            (
                presentation.prompt,
                presentation.label,
                presentation.help_text,
                presentation.value_label,
                presentation.result_template or "",
                str(presentation.minimum),
                str(presentation.maximum),
                str(presentation.initial_value),
            )
        )
    return list(dict.fromkeys(fragment for fragment in fragments if fragment))


def _candidate_texts_for(target: AssessmentItem, fragments: list[str]) -> list[str]:
    candidates: list[str] = []
    for fragment in fragments:
        stripped = fragment.strip()
        marker = re.search(
            r"(?:answer|result|value|total|whole|sum|area|missing|interval)\s*"
            r"(?:is|=|:)?\s*(.+)$",
            stripped,
            flags=re.IGNORECASE,
        )
        if marker:
            candidates.append(marker.group(1).strip().rstrip(".;:"))
        if isinstance(target.answer, NumericAnswerSpec):
            if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", stripped):
                candidates.append(stripped)
            continue
        candidates.extend(_candidate_answer_texts(stripped))
    return list(
        dict.fromkeys(
            candidate
            for candidate in candidates
            if candidate and _candidate_fits_answer_contract(target, candidate)
        )
    )


def _candidate_contract_key(item: AssessmentItem) -> tuple[object, ...]:
    """Return the fields that affect candidate shape, excluding expected truth."""

    answer = item.answer
    return (
        answer.kind,
        tuple(getattr(answer, "variables", ())),
        tuple(getattr(answer, "functions", ())),
        getattr(answer, "variable", None),
        getattr(answer, "assignment_lhs", None),
    )


def _answer_spec_with_shared_vocabulary(
    left: AnswerSpec,
    right: AnswerSpec,
) -> tuple[AnswerSpec, AnswerSpec]:
    if type(left) is not type(right):
        return left, right
    if isinstance(left, OrderedTupleAnswerSpec) and isinstance(
        right, OrderedTupleAnswerSpec
    ):
        variables = sorted(set(left.variables) | set(right.variables))
        functions = sorted(set(left.functions) | set(right.functions))
        return (
            left.model_copy(update={"variables": variables, "functions": functions}),
            right.model_copy(update={"variables": variables, "functions": functions}),
        )
    return left, right


def _answers_are_equivalent(left: AssessmentItem, right: AssessmentItem) -> bool:
    scalar_types = (NumericAnswerSpec,)
    if isinstance(left.answer, scalar_types) and isinstance(right.answer, scalar_types):
        left_value = sympy.sympify(left.answer.expected)
        right_value = sympy.sympify(right.answer.expected)
        return sympy.cancel(left_value - right_value) == 0
    # A numeric constant authored under a symbolic contract in an earlier wave
    # is still the same mathematical answer and must not be reused.
    if isinstance(left.answer, (NumericAnswerSpec, SymbolicAnswerSpec)) and isinstance(
        right.answer, (NumericAnswerSpec, SymbolicAnswerSpec)
    ):
        variables = set(getattr(left.answer, "variables", ())) | set(
            getattr(right.answer, "variables", ())
        )
        left_value = parse_restricted(
            left.answer.expected,
            allowed_variables=variables,
            allowed_functions=set(),
            allowed_assignment_lhs=None,
        )
        right_value = parse_restricted(
            right.answer.expected,
            allowed_variables=variables,
            allowed_functions=set(),
            allowed_assignment_lhs=None,
        )
        return sympy.cancel(left_value - right_value) == 0
    if type(left.answer) is not type(right.answer):
        return False
    left_spec, right_spec = _answer_spec_with_shared_vocabulary(
        left.answer, right.answer
    )
    left_verdict = verify_answer(
        left_spec,
        canonical_submission(right_spec),
        supervised=False,
    )
    right_verdict = verify_answer(
        right_spec,
        canonical_submission(left_spec),
        supervised=False,
    )
    if any(
        verdict.status not in {VerificationStatus.CORRECT, VerificationStatus.INCORRECT}
        for verdict in (left_verdict, right_verdict)
    ):
        raise FTCCompilationError(
            f"answer comparison for {left.item_id}/{right.item_id} was indeterminate"
        )
    return VerificationStatus.CORRECT in {
        left_verdict.status,
        right_verdict.status,
    }


def _canonical_visible_candidate(
    answer: AnswerSpec,
    candidate: str,
) -> tuple[object, ...] | None:
    """Canonicalize polynomial-era contracts once per visible fragment.

    This is only a fast inequality gate.  A canonical match is still confirmed
    through the verifier before it is reported as leakage.
    """

    try:
        if isinstance(answer, (NumericAnswerSpec, SymbolicAnswerSpec)):
            variables = set(getattr(answer, "variables", ()))
            functions = set(getattr(answer, "functions", ()))
            expression = parse_restricted(
                candidate,
                allowed_variables=variables,
                allowed_functions=functions,
                allowed_assignment_lhs=getattr(answer, "assignment_lhs", None),
            )
            if isinstance(answer, NumericAnswerSpec) and expression.free_symbols:
                return None
            normalized = sympy.expand(sympy.cancel(expression))
            return (answer.kind, sympy.srepr(normalized))
        if isinstance(answer, AntiderivativeAnswerSpec):
            variables = set(answer.variables) | {answer.variable, "C"}
            expression = parse_restricted(
                candidate,
                allowed_variables=variables,
                allowed_functions=set(answer.functions),
                allowed_assignment_lhs=None,
            )
            derivative = sympy.diff(expression, sympy.Symbol(answer.variable))
            normalized = sympy.expand(sympy.cancel(derivative))
            return (answer.kind, sympy.srepr(normalized))
        if isinstance(answer, OrderedTupleAnswerSpec):
            values = _split_container(candidate, "(", ")")
            normalized = tuple(
                sympy.srepr(
                    sympy.expand(
                        sympy.cancel(
                            parse_restricted(
                                value,
                                allowed_variables=set(answer.variables),
                                allowed_functions=set(answer.functions),
                                allowed_assignment_lhs=None,
                            )
                        )
                    )
                )
                for value in values
            )
            return (answer.kind, *normalized)
        if isinstance(answer, FiniteSetAnswerSpec):
            values = _split_container(candidate, "{", "}")
            normalized = sorted(
                {
                    sympy.srepr(
                        sympy.expand(
                            sympy.cancel(
                                parse_restricted(
                                    value,
                                    allowed_variables=set(answer.variables),
                                    allowed_functions=set(answer.functions),
                                    allowed_assignment_lhs=None,
                                )
                            )
                        )
                    )
                    for value in values
                }
            )
            return (answer.kind, *normalized)
    except Exception:  # A non-answer-shaped fragment is definitively not equal.
        return None
    return None


def _visible_separation_worker(
    connection: Connection,
    item_payloads: list[dict[str, object]],
    source_item_ids: tuple[str, ...],
    focus_item_ids: set[str] | None,
) -> None:
    """Check a bounded source slice so symbolic caches cannot grow unbounded."""

    try:
        ordered = [AssessmentItem.model_validate(payload) for payload in item_payloads]
        by_id = {item.item_id: item for item in ordered}
        sources = [by_id[item_id] for item_id in source_item_ids]
        def in_scope(source: AssessmentItem, target: AssessmentItem) -> bool:
            return focus_item_ids is None or bool(
                {source.item_id, target.item_id} & focus_item_ids
            )

        targets_by_source = {
            source.item_id: [
                target
                for target in ordered
                if source.family_id != target.family_id and in_scope(source, target)
            ]
            for source in sources
        }
        representatives: dict[tuple[object, ...], AssessmentItem] = {}
        for targets in targets_by_source.values():
            for target in targets:
                representatives.setdefault(_candidate_contract_key(target), target)
        candidates = {
            (source.item_id, contract): _candidate_texts_for(
                representative,
                _visible_fragments(source),
            )
            for source in sources
            for contract, representative in representatives.items()
            if any(
                _candidate_contract_key(target) == contract
                for target in targets_by_source[source.item_id]
            )
        }
        expected_canonical = {
            target.item_id: _canonical_visible_candidate(
                target.answer,
                canonical_submission(target.answer),
            )
            for targets in targets_by_source.values()
            for target in targets
        }
        candidate_canonical: dict[
            tuple[tuple[object, ...], str], tuple[object, ...] | None
        ] = {}

        comparisons = 0
        errors: list[str] = []
        for source in sources:
            for target in targets_by_source[source.item_id]:
                for candidate in candidates[
                    (source.item_id, _candidate_contract_key(target))
                ]:
                    comparisons += 1
                    contract = _candidate_contract_key(target)
                    cache_key = (contract, candidate)
                    if cache_key not in candidate_canonical:
                        candidate_canonical[cache_key] = _canonical_visible_candidate(
                            target.answer,
                            candidate,
                        )
                    expected = expected_canonical[target.item_id]
                    candidate_value = candidate_canonical[cache_key]
                    if (
                        expected is not None
                        and candidate_value is not None
                        and expected != candidate_value
                    ):
                        continue
                    verdict = verify_answer(target.answer, candidate, supervised=False)
                    if verdict.status == VerificationStatus.CORRECT:
                        errors.append(
                            f"{source.item_id}: visible content leaks the answer "
                            f"for {target.item_id}"
                        )
                        break
                    if verdict.status not in {
                        VerificationStatus.INCORRECT,
                        VerificationStatus.INVALID,
                    }:
                        errors.append(
                            f"{source.item_id}: visible comparison for "
                            f"{target.item_id} was indeterminate ({verdict.code})"
                        )
                        break
        connection.send({"comparisons": comparisons, "errors": errors})
    except BaseException as exc:  # noqa: BLE001 - worker must fail closed
        connection.send({"error": type(exc).__name__})
    finally:
        connection.close()


def _run_visible_separation(
    ordered: list[AssessmentItem],
    focus_item_ids: set[str] | None,
    *,
    chunk_size: int = 12,
    timeout_seconds: float = 120.0,
) -> tuple[int, list[str]]:
    # Fork keeps the offline compiler's read-only item payload copy-on-write;
    # platforms without it retain the portable spawn path.
    method = "fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn"
    context = multiprocessing.get_context(method)
    payloads = [item.model_dump(mode="json") for item in ordered]
    comparisons = 0
    errors: list[str] = []
    source_ids = [item.item_id for item in ordered]
    for offset in range(0, len(source_ids), chunk_size):
        parent, child = context.Pipe(duplex=False)
        process = context.Process(
            target=_visible_separation_worker,
            args=(
                child,
                payloads,
                tuple(source_ids[offset : offset + chunk_size]),
                focus_item_ids,
            ),
            daemon=True,
        )
        process.start()
        child.close()
        try:
            if not parent.poll(timeout_seconds):
                raise FTCCompilationError("visible-separation worker timed out")
            result = parent.recv()
        except EOFError as exc:
            raise FTCCompilationError(
                "visible-separation worker exited without a result"
            ) from exc
        finally:
            parent.close()
            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)
        if "error" in result:
            raise FTCCompilationError(
                f"visible-separation worker failed ({result['error']})"
            )
        comparisons += int(result["comparisons"])
        errors.extend(str(error) for error in result["errors"])
    return comparisons, errors


def validate_inventory_separation(
    items: list[AssessmentItem],
    graph: GraphDocument,
    *,
    focus_item_ids: set[str] | None = None,
) -> InventorySeparationReport:
    """Exhaustively reject reused answers and cross-family visible leakage."""

    ordered = sorted(items, key=lambda item: (item.family_id, item.item_id))
    ids = {item.item_id for item in ordered}
    if focus_item_ids is not None and not focus_item_ids <= ids:
        raise FTCCompilationError("separation focus names an unknown item")

    def in_scope(left: AssessmentItem, right: AssessmentItem) -> bool:
        return focus_item_ids is None or bool(
            {left.item_id, right.item_id} & focus_item_ids
        )

    errors: list[str] = []
    answer_pairs = 0
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            answer_pairs += 1
            if not in_scope(left, right):
                continue
            try:
                equivalent = _answers_are_equivalent(left, right)
            except Exception as exc:  # noqa: BLE001 - publication fails closed
                errors.append(
                    f"answer comparison for {left.item_id}/{right.item_id} "
                    f"was indeterminate ({type(exc).__name__})"
                )
            else:
                if equivalent:
                    errors.append(
                        "expected answer reused across families "
                        f"{left.family_id!r} and {right.family_id!r}"
                    )
    visible_pairs = 0
    for source_item in ordered:
        for target in ordered:
            if source_item.family_id == target.family_id:
                continue
            visible_pairs += 1
    candidate_comparisons, visible_errors = _run_visible_separation(
        ordered,
        focus_item_ids,
    )
    errors.extend(visible_errors)
    graph_fragments = [
        value
        for node in graph.nodes
        if node.id in {item.kc_id for item in ordered}
        for value in (node.name, node.description, *node.canonical_examples)
    ]
    for target in ordered:
        if focus_item_ids is not None and target.item_id not in focus_item_ids:
            continue
        for candidate in _candidate_texts_for(target, graph_fragments):
            candidate_comparisons += 1
            verdict = verify_answer(target.answer, candidate, supervised=False)
            if verdict.status == VerificationStatus.CORRECT:
                errors.append(
                    f"student-visible graph content leaks the answer for {target.item_id}"
                )
                break
    return InventorySeparationReport(
        answer_pairs_checked=answer_pairs,
        visible_candidate_comparisons_checked=candidate_comparisons,
        literal_visible_pairs_checked=visible_pairs,
        errors=tuple(dict.fromkeys(errors)),
    )


def validate_pedagogy_item_separation(
    bank: ItemBankDocument,
    pedagogy: PedagogySourceDocument,
) -> tuple[str, ...]:
    errors: list[str] = []
    for pack in pedagogy.pack_sources:
        fragments = [
            value
            for segment in (*pack.lesson_narrative, *pack.remediation)
            for value in _segment_fragments(cast(PromptSegment, segment))
            if value
        ]
        fragments.extend(metaphor.description for metaphor in pack.metaphors)
        fragments.extend(
            misconception.remediation_hint for misconception in pack.misconceptions
        )
        for target in bank.items:
            for candidate in _candidate_texts_for(target, fragments):
                verdict = verify_answer(target.answer, candidate, supervised=False)
                if verdict.status == VerificationStatus.CORRECT:
                    errors.append(
                        f"{pack.source_id}: pedagogy leaks the answer for {target.item_id}"
                    )
                    break
    return tuple(dict.fromkeys(errors))


def _validate_taxonomy(source: FTCBlueprintDocument) -> None:
    for family in source.families:
        try:
            _TASK_COMPILER_REGISTRY.validate_taxonomy(
                family.task,
                construct_id=family.construct_id,
                kc_id=family.kc_id,
            )
        except TaskCompilerRegistryError as exc:
            raise FTCCompilationError(str(exc)) from exc
    for kc_id, by_surface in EXPECTED_CONSTRUCT_ORDER.items():
        for surface, constructs in by_surface.items():
            actual = tuple(
                (family.allocation_order, family.construct_id)
                for family in sorted(
                    (
                        family
                        for family in source.families
                        if family.kc_id == kc_id and family.surface == surface
                    ),
                    key=lambda family: family.allocation_order,
                )
            )
            expected = tuple(
                ((index + 1) * 10, construct)
                for index, construct in enumerate(constructs)
            )
            if actual != expected:
                raise FTCCompilationError(
                    f"{kc_id}/{surface.value}: construct/order mismatch; "
                    f"expected={expected}, got={actual}"
                )
            if surface in {AssessmentSurface.DIAGNOSTIC, AssessmentSurface.CHECKIN}:
                if len(constructs) != len(set(constructs)):
                    raise FTCCompilationError(
                        f"{kc_id}/{surface.value}: confirmation constructors repeat"
                    )


_MISCONCEPTION_IDS = {
    GRAPH_KC: frozenset(
        {
            "m.graph_reading.axes_coordinates",
            "m.graph_reading.intercept_confusion",
            "m.graph_reading.slope_direction",
        }
    ),
    AREA_KC: frozenset(
        {
            "m.area_under_curve.omits_region",
            "m.area_under_curve.triangle_factor",
            "m.area_under_curve.uses_endpoint_height",
        }
    ),
    RIEMANN_KC: frozenset(
        {
            "m.riemann_sums.endpoint_choice",
            "m.riemann_sums.omits_width",
            "m.riemann_sums.midpoint_confusion",
        }
    ),
    DEFINITE_KC: frozenset(
        {
            "m.definite_integral.bound_order",
            "m.definite_integral.ignores_sign",
            "m.definite_integral.breaks_additivity",
        }
    ),
    ANTIDERIVATIVE_KC: frozenset(
        {
            "m.antiderivatives.keeps_exponent",
            "m.antiderivatives.multiplies_coefficient",
            "m.antiderivatives.drops_term",
        }
    ),
    TARGET_KC: frozenset(
        {
            "m.ftc.adds_endpoint_values",
            "m.ftc.reverses_subtraction",
            "m.ftc.uses_integrand_values",
        }
    ),
}


def _validate_signature_taxonomy(
    items: list[AssessmentItem],
    graph: GraphDocument,
) -> None:
    hard_predecessors = {
        kc_id: {
            edge.from_kc
            for edge in graph.edges
            if edge.to_kc == kc_id and edge.type == EdgeType.HARD
        }
        for kc_id in TARGET_KCS
    }
    for item in items:
        for signature in item.error_signatures:
            if signature.misconception_id not in _MISCONCEPTION_IDS[item.kc_id]:
                raise FTCCompilationError(f"{item.item_id}: unreviewed misconception id")
            if (
                signature.implicated_prereq is not None
                and signature.implicated_prereq not in hard_predecessors[item.kc_id]
            ):
                raise FTCCompilationError(
                    f"{item.item_id}: implicated prerequisite is not direct and hard"
                )


def compile_release_inventory(
    source: FTCBlueprintDocument,
    manifest: ContentReviewManifest,
    graph: GraphDocument,
) -> tuple[ItemBankDocument, InventorySeparationReport]:
    """Compile and qualify all 78 pending FTC-wave families."""

    if source.graph_version != graph.graph_version:
        raise FTCCompilationError("source and graph versions differ")
    if manifest.graph_version != graph.graph_version:
        raise FTCCompilationError("manifest and graph versions differ")
    if manifest.compiler_version != COMPILER_VERSION:
        raise FTCCompilationError("manifest compiler pin is stale")
    if set(source.target_kcs) != set(TARGET_KCS):
        raise FTCCompilationError("source must contain exactly the six FTC-wave KCs")
    closure = ancestor_subgraph(graph, TARGET_KC, hard_only=True).node_ids()
    if closure != set(EXPECTED_CLOSURE):
        raise FTCCompilationError(f"FTC hard closure changed: {sorted(closure)}")
    identities = {
        (family.blueprint_id, family.revision) for family in source.families
    }
    reviews = {
        (entry.blueprint_id, entry.revision): entry for entry in manifest.entries
    }
    if set(reviews) != identities:
        raise FTCCompilationError("review/source identity coverage differs")
    expected_matrix = {
        (kc_id, surface): count
        for kc_id in TARGET_KCS
        for surface, count in EXPECTED_FAMILY_COUNTS.items()
    }
    if dict(Counter((family.kc_id, family.surface) for family in source.families)) != (
        expected_matrix
    ):
        raise FTCCompilationError(
            "family matrix must contain 13 families per KC in the 4/5/1/2/1 split"
        )
    _validate_taxonomy(source)
    items: list[AssessmentItem] = []
    for family in sorted(
        source.families,
        key=lambda entry: (
            entry.kc_id,
            entry.surface.value,
            entry.allocation_order,
            entry.family_id,
        ),
    ):
        review = reviews[(family.blueprint_id, family.revision)]
        if review.source_digest != family_digest(source, family):
            raise FTCCompilationError(
                f"review digest mismatch for {family.blueprint_id}@{family.revision}"
            )
        review_status, provenance = _review_status_and_provenance(
            source, family, review
        )
        item = _build_item(
            source,
            family,
            review_status=review_status,
            provenance=provenance,
        )
        verdict = verify_answer(
            item.answer,
            canonical_submission(item.answer),
            supervised=False,
        )
        if verdict.status != VerificationStatus.CORRECT:
            raise FTCCompilationError(
                f"{item.item_id}: derived truth failed verification ({verdict.code})"
            )
        items.append(item)
    _validate_signature_taxonomy(items, graph)
    approved_kcs = {
        kc_id
        for kc_id in TARGET_KCS
        if all(
            item.review_status == ReviewStatus.HUMAN_APPROVED
            for item in items
            if item.kc_id == kc_id
        )
    }
    if not set(source.released_kcs) <= approved_kcs:
        raise FTCCompilationError(
            "released_kcs contains a KC without complete independent approval"
        )
    bank = ItemBankDocument(
        schema_version=3,
        bank_version=source.output_bank_version,
        graph_version=source.graph_version,
        released_kcs=source.released_kcs,
        items=items,
    )
    report = validate_inventory_separation(items, graph)
    if report.errors:
        raise FTCCompilationError(
            "inventory separation failed: " + "; ".join(report.errors)
        )
    preceding_items = [
        item
        for path in PRECEDING_BANK_PATHS
        for item in load_item_bank(path).items
    ]
    cumulative_report = validate_inventory_separation(
        [*preceding_items, *items],
        graph,
        focus_item_ids={item.item_id for item in items},
    )
    if cumulative_report.errors:
        raise FTCCompilationError(
            "cumulative inventory separation failed: "
            + "; ".join(cumulative_report.errors)
        )
    return bank, report


def _refuse_completed_review_overwrite(
    item_manifest: ContentReviewManifest | None,
    pedagogy_manifest: PedagogyReviewManifest | None,
) -> None:
    if item_manifest is not None and any(
        entry.decision != ReviewDecision.PENDING for entry in item_manifest.entries
    ):
        raise FTCCompilationError("refusing to overwrite a completed item-family review")
    if pedagogy_manifest is not None and any(
        entry.decision != PedagogyReviewDecision.PENDING
        for entry in pedagogy_manifest.entries
    ):
        raise FTCCompilationError("refusing to overwrite a completed pedagogy review")


def _atomic_write_model(path: Path, model: BaseModel) -> None:
    payload = model.model_dump_json(indent=2) + "\n"
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def regenerate_draft_assets(
    *,
    source_path: Path = DEFAULT_SOURCE_PATH,
    item_manifest_path: Path = DEFAULT_MANIFEST_PATH,
    bank_path: Path = DEFAULT_BANK_PATH,
    pedagogy_source_path: Path = DEFAULT_PEDAGOGY_SOURCE_PATH,
    pedagogy_manifest_path: Path = DEFAULT_PEDAGOGY_MANIFEST_PATH,
    graph_path: Path = DEFAULT_GRAPH_PATH,
) -> None:
    """Regenerate only pending manifests and deterministic draft output."""

    existing_items = (
        ContentReviewManifest.model_validate_json(
            item_manifest_path.read_text(encoding="utf-8")
        )
        if item_manifest_path.exists()
        else None
    )
    existing_pedagogy = (
        PedagogyReviewManifest.model_validate_json(
            pedagogy_manifest_path.read_text(encoding="utf-8")
        )
        if pedagogy_manifest_path.exists()
        else None
    )
    _refuse_completed_review_overwrite(existing_items, existing_pedagogy)
    source = load_source(source_path)
    graph = GraphDocument.model_validate_json(graph_path.read_text(encoding="utf-8"))
    pedagogy = PedagogySourceDocument.model_validate_json(
        pedagogy_source_path.read_text(encoding="utf-8")
    )
    item_manifest = draft_review_manifest(source)
    pedagogy_manifest = draft_pedagogy_review_manifest(pedagogy)
    bank, _report = compile_release_inventory(source, item_manifest, graph)
    if validate_pedagogy_item_separation(bank, pedagogy):
        raise FTCCompilationError("pedagogy content leaks a draft assessment answer")
    _atomic_write_model(item_manifest_path, item_manifest)
    _atomic_write_model(pedagogy_manifest_path, pedagogy_manifest)
    _atomic_write_model(bank_path, bank)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH_PATH)
    parser.add_argument(
        "--pedagogy-source", type=Path, default=DEFAULT_PEDAGOGY_SOURCE_PATH
    )
    parser.add_argument(
        "--pedagogy-manifest",
        type=Path,
        default=DEFAULT_PEDAGOGY_MANIFEST_PATH,
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if not args.check and not args.regenerate and args.out is None:
        parser.error("nothing to do: pass --check, --regenerate, and/or --out")
    try:
        if args.regenerate:
            regenerate_draft_assets(
                source_path=args.source,
                item_manifest_path=args.manifest,
                bank_path=args.out or DEFAULT_BANK_PATH,
                pedagogy_source_path=args.pedagogy_source,
                pedagogy_manifest_path=args.pedagogy_manifest,
                graph_path=args.graph,
            )
        source = load_source(args.source)
        manifest = load_manifest(args.manifest)
        graph = GraphDocument.model_validate_json(
            args.graph.read_text(encoding="utf-8")
        )
        bank, report = compile_release_inventory(source, manifest, graph)
        pedagogy = PedagogySourceDocument.model_validate_json(
            args.pedagogy_source.read_text(encoding="utf-8")
        )
        pedagogy_manifest = PedagogyReviewManifest.model_validate_json(
            args.pedagogy_manifest.read_text(encoding="utf-8")
        )
        validate_review_bundle(pedagogy, pedagogy_manifest, graph)
        pedagogy_errors = validate_pedagogy_item_separation(bank, pedagogy)
        if pedagogy_errors:
            raise FTCCompilationError(
                "pedagogy separation failed: " + "; ".join(pedagogy_errors)
            )
        if args.out is not None and not args.regenerate:
            _atomic_write_model(args.out, bank)
    except Exception as exc:  # noqa: BLE001 - offline CLI fail-closed boundary
        print(f"FTC content INVALID: {exc}", file=sys.stderr)
        return 1
    if args.check:
        print(
            "FTC content OK: "
            f"{len(bank.items)} families, "
            f"{report.answer_pairs_checked} answer pairs, "
            f"{report.visible_candidate_comparisons_checked} visible comparisons"
        )
    if args.out is not None:
        print(f"wrote deterministic draft bank to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
