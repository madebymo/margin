"""Compile and qualify the pending three-KC Chain Rule content wave.

The source contains 39 typed family blueprints: 13 each for function
notation, composition, and the chain rule.  Mathematical truth is derived by
closed constructors.  The packaged review manifest remains pending and the
compiled bank releases no KCs until independent review is complete.
"""

from __future__ import annotations

import argparse
import hashlib
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
from typing import Literal

import sympy

from tutor.content.item_bank import (
    _candidate_answer_texts,
    _candidate_fits_answer_contract,
    _segment_visible_fragments,
    render_prompt,
)
from tutor.content.task_compilers import (
    TaskCompilerRegistration,
    TaskCompilerRegistry,
    TaskCompilerRegistryError,
)
from tutor.graph.service import ancestor_subgraph
from tutor.schemas.assessment import (
    AssessmentHint,
    AssessmentItem,
    AssessmentProvenance,
    AssessmentSurface,
    AssessmentTaskKind,
    BlankPromptSegment,
    ErrorSignature,
    GuidedInteractionSpec,
    GuidedMappingSpec,
    GuidedSliderPresentation,
    GuidedSliderScoring,
    GuidedSliderSpec,
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
    answer_spec_adapter,
)
from tutor.schemas.chain_rule_authoring import (
    AffineCompositionTask,
    AffineFunctionValueTask,
    AffinePowerChainTask,
    ChainAtPointTask,
    ChainCorrectionTask,
    ChainFactorTupleTask,
    ChainRuleBlueprintDocument,
    ChainRuleFamilyBlueprint,
    ChainRuleMathTask,
    ChainTableValuesTask,
    CompositionAtPointTask,
    CompositionPairedOrdersTask,
    CompositionPlotSumTask,
    CompositionTablePathTask,
    FunctionExpressionValueTask,
    FunctionOrderedValuesTask,
    FunctionPlotChangeTask,
    FunctionTableCombinationTask,
    QuadraticFunctionValueTask,
    QuadraticOuterCompositionTask,
    QuadraticPowerChainTask,
)
from tutor.schemas.common import ReviewStatus
from tutor.schemas.content_authoring import (
    ContentReviewEntry,
    ContentReviewManifest,
    ReviewDecision,
)
from tutor.schemas.kc import GraphDocument
from tutor.verify.checker import VerificationStatus, parse_restricted, verify_answer

COMPILER_VERSION = "chain-rule-item-compiler-v3"
TARGET_KCS = frozenset(
    {
        "kc.fun.function_notation",
        "kc.fun.composition",
        "kc.der.chain_rule",
    }
)
EXPECTED_CHAIN_CLOSURE = frozenset(
    {
        "kc.alg.exponent_rules",
        "kc.der.power_rule",
        "kc.fun.function_notation",
        "kc.fun.composition",
        "kc.der.chain_rule",
    }
)
EXPECTED_FAMILY_COUNTS = {
    AssessmentSurface.DIAGNOSTIC: 4,
    AssessmentSurface.CHECKIN: 5,
    AssessmentSurface.GUIDED_WIDGET: 1,
    AssessmentSurface.CAPSTONE: 2,
    AssessmentSurface.WORKED_EXAMPLE: 1,
}
EXPECTED_CONSTRUCT_ORDER: dict[
    str, dict[AssessmentSurface, tuple[str, ...]]
] = {
    "kc.fun.function_notation": {
        AssessmentSurface.DIAGNOSTIC: (
            "function.affine_value",
            "function.expression_value",
            "function.table_combination",
            "function.plot_change",
        ),
        AssessmentSurface.CHECKIN: (
            "function.quadratic_value",
            "function.expression_value",
            "function.table_combination",
            "function.plot_change",
            "function.ordered_values",
        ),
        AssessmentSurface.GUIDED_WIDGET: ("function.affine_value",),
        AssessmentSurface.CAPSTONE: (
            "function.expression_value",
            "function.ordered_values",
        ),
        AssessmentSurface.WORKED_EXAMPLE: ("function.affine_value",),
    },
    "kc.fun.composition": {
        AssessmentSurface.DIAGNOSTIC: (
            "composition.affine",
            "composition.quadratic_outer",
            "composition.at_point",
            "composition.table_path",
        ),
        AssessmentSurface.CHECKIN: (
            "composition.quadratic_outer",
            "composition.at_point",
            "composition.table_path",
            "composition.paired_orders",
            "composition.plot_sum",
        ),
        AssessmentSurface.GUIDED_WIDGET: ("composition.at_point",),
        AssessmentSurface.CAPSTONE: (
            "composition.paired_orders",
            "composition.quadratic_outer",
        ),
        AssessmentSurface.WORKED_EXAMPLE: ("composition.affine",),
    },
    "kc.der.chain_rule": {
        AssessmentSurface.DIAGNOSTIC: (
            "chain.affine_power",
            "chain.quadratic_power",
            "chain.at_point",
            "chain.correction",
        ),
        AssessmentSurface.CHECKIN: (
            "chain.affine_power",
            "chain.quadratic_power",
            "chain.at_point",
            "chain.table_values",
            "chain.correction",
        ),
        AssessmentSurface.GUIDED_WIDGET: ("chain.at_point",),
        AssessmentSurface.CAPSTONE: (
            "chain.correction",
            "chain.affine_power",
        ),
        AssessmentSurface.WORKED_EXAMPLE: ("chain.quadratic_power",),
    },
}

SEED_DIR = Path(__file__).resolve().parents[1] / "seed"
DEFAULT_SOURCE_PATH = SEED_DIR / "item_blueprints_chain_rule_v2.json"
DEFAULT_MANIFEST_PATH = SEED_DIR / "item_reviews_chain_rule_v2.json"
DEFAULT_GRAPH_PATH = SEED_DIR / "kc_graph_calc1.json"


class ChainRuleCompilationError(ValueError):
    """The pending Chain Rule inventory cannot be compiled safely."""


@dataclass(frozen=True)
class InventorySeparationReport:
    """Auditable totals from exhaustive answer and visible-content checks."""

    answer_pairs_checked: int
    visible_candidate_comparisons_checked: int
    literal_visible_pairs_checked: int
    errors: tuple[str, ...]


@dataclass(frozen=True)
class DerivedTask:
    """Deterministic constructor result used by item and oracle tests."""

    instruction: str
    givens: tuple[PromptSegment, ...]
    expected: str | tuple[str, ...]
    answer_kind: Literal["numeric", "symbolic", "ordered_tuple"]
    conceptual_hint: str
    operation_hint: str
    worked_step: str | None = None
    error_signatures: tuple[ErrorSignature, ...] = ()

    @property
    def submission(self) -> str:
        """Canonical learner submission used only in private compiled content."""

        if isinstance(self.expected, tuple):
            return "(" + ", ".join(self.expected) + ")"
        return self.expected


def _term(coefficient: int, variable: str) -> str:
    if coefficient == 1:
        return variable
    if coefficient == -1:
        return f"-{variable}"
    return f"{coefficient}*{variable}"


def _append_constant(expression: str, constant: int) -> str:
    if constant > 0:
        return f"{expression}+{constant}"
    if constant < 0:
        return f"{expression}{constant}"
    return expression


def _affine(coefficient: int, constant: int, variable: str = "x") -> str:
    return _append_constant(_term(coefficient, variable), constant)


def _quadratic(quadratic: int, linear: int, constant: int) -> str:
    expression = _term(quadratic, "x^2")
    if linear:
        linear_term = _term(abs(linear), "x")
        expression += f"+{linear_term}" if linear > 0 else f"-{linear_term}"
    return _append_constant(expression, constant)


def _scaled_power(scale: int, inner: str, power: int, constant: int = 0) -> str:
    factor = f"({inner})^{power}"
    expression = _term(scale, factor)
    return _append_constant(expression, constant)


def _spoken_math(expression: str) -> str:
    spoken = expression
    for source, replacement in (
        ("'", " prime "),
        ("^", " to the power of "),
        ("*", " times "),
        ("/", " divided by "),
        ("+", " plus "),
        ("-", " minus "),
        ("=", " equals "),
        ("(", " open parenthesis "),
        (")", " close parenthesis "),
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


def _composition_values(
    *,
    f_coefficient: int,
    f_constant: int,
    g_coefficient: int,
    g_constant: int,
    point: int,
) -> tuple[int, int]:
    f_at_point = f_coefficient * point + f_constant
    g_at_point = g_coefficient * point + g_constant
    return (
        f_coefficient * g_at_point + f_constant,
        g_coefficient * f_at_point + g_constant,
    )


def _derive_unregistered(task: ChainRuleMathTask) -> DerivedTask:
    if isinstance(task, AffineFunctionValueTask):
        function = _affine(task.coefficient, task.constant)
        expected = task.coefficient * task.input_value + task.constant
        return DerivedTask(
            instruction=f"Evaluate f({task.input_value}) by replacing x in the rule.",
            givens=(_math(f"f(x)={function}"),),
            expected=str(expected),
            answer_kind="numeric",
            conceptual_hint="Function notation names the input to substitute; it is not multiplication.",
            operation_hint=(
                f"Replace x with {task.input_value}, keep the grouping, and then simplify."
            ),
            worked_step=_append_constant(
                f"{task.coefficient}*({task.input_value})",
                task.constant,
            ),
        )
    if isinstance(task, QuadraticFunctionValueTask):
        function = _quadratic(
            task.quadratic,
            task.linear,
            task.constant,
        )
        expected = (
            task.quadratic * task.input_value**2
            + task.linear * task.input_value
            + task.constant
        )
        return DerivedTask(
            instruction=f"Evaluate f({task.input_value}) from the polynomial rule.",
            givens=(_math(f"f(x)={function}"),),
            expected=str(expected),
            answer_kind="numeric",
            conceptual_hint="The same input replaces every occurrence of x in the function rule.",
            operation_hint=(
                "Substitute the input into both the squared and linear terms before "
                "combining the signed values."
            ),
            error_signatures=(
                _signature(
                    str(
                        task.quadratic * task.input_value**2
                        + task.linear
                        + task.constant
                    ),
                    "m.function_notation.substitutes_one_occurrence",
                ),
            ),
        )
    if isinstance(task, FunctionExpressionValueTask):
        function = _affine(
            task.function_coefficient,
            task.function_constant,
            variable="t",
        )
        supplied_input = _affine(
            task.input_coefficient,
            task.input_constant,
        )
        expected = _affine(
            task.function_coefficient * task.input_coefficient,
            task.function_coefficient * task.input_constant
            + task.function_constant,
        )
        return DerivedTask(
            instruction="Evaluate f(p(x)) and simplify the result in x.",
            givens=(
                _math(f"f(t)={function}"),
                _math(f"p(x)={supplied_input}"),
            ),
            expected=expected,
            answer_kind="symbolic",
            conceptual_hint="Treat the entire supplied expression as one input for t.",
            operation_hint=(
                "Place the input expression in parentheses, multiply it by the "
                "function coefficient, and then combine constants."
            ),
            error_signatures=(
                _signature(
                    _affine(
                        task.function_coefficient * task.input_coefficient,
                        task.input_constant + task.function_constant,
                    ),
                    "m.function_notation.drops_input_grouping",
                ),
            ),
        )
    if isinstance(task, FunctionTableCombinationTask):
        if task.operation == "sum":
            expected = task.left_output + task.right_output
            operation = "sum"
            notation = "+"
        else:
            expected = task.left_output - task.right_output
            operation = "difference"
            notation = "-"
        table = TablePromptSegment(
            role=PromptSemanticRole.GIVEN,
            caption="Selected values of f",
            column_headers=("input x", "output f(x)"),
            rows=(
                (str(task.left_input), str(task.left_output)),
                (str(task.right_input), str(task.right_output)),
            ),
            spoken_text=(
                f"The table gives f of {task.left_input} as {task.left_output} "
                f"and f of {task.right_input} as {task.right_output}."
            ),
        )
        return DerivedTask(
            instruction=(
                f"Use the table to find the {operation} "
                f"f({task.left_input}) {notation} f({task.right_input})."
            ),
            givens=(table,),
            expected=str(expected),
            answer_kind="numeric",
            conceptual_hint="Each table row pairs one input with its function output.",
            operation_hint=(
                "Read both requested outputs first, then combine them in the "
                "order shown in the expression."
            ),
        )
    if isinstance(task, FunctionPlotChangeTask):
        table = TablePromptSegment(
            role=PromptSemanticRole.CONTEXT,
            caption="Exact plotted points",
            column_headers=("input x", "output f(x)"),
            rows=(
                (str(task.start_input), str(task.start_output)),
                (str(task.end_input), str(task.end_output)),
            ),
            spoken_text=(
                f"The plot contains the points {task.start_input}, {task.start_output} "
                f"and {task.end_input}, {task.end_output}."
            ),
        )
        plot = PlotPromptSegment(
            role=PromptSemanticRole.GIVEN,
            title="Two exact values of f",
            x_label="input x",
            y_label="output f(x)",
            series=(
                StaticPlotSeries(
                    label="f",
                    points=(
                        StaticPlotPoint(
                            x=str(task.start_input),
                            y=str(task.start_output),
                        ),
                        StaticPlotPoint(
                            x=str(task.end_input),
                            y=str(task.end_output),
                        ),
                    ),
                ),
            ),
            spoken_text=(
                f"The static plot shows f({task.start_input})={task.start_output} "
                f"and f({task.end_input})={task.end_output}."
            ),
            equivalent_table=table,
        )
        return DerivedTask(
            instruction=(
                "Use the exact plotted values to find the output change "
                f"f({task.end_input})-f({task.start_input})."
            ),
            givens=(plot,),
            expected=str(task.end_output - task.start_output),
            answer_kind="numeric",
            conceptual_hint="A plotted point (x,y) says that f(x)=y.",
            operation_hint="Read the ending output and subtract the starting output.",
        )
    if isinstance(task, FunctionOrderedValuesTask):
        function = _affine(task.coefficient, task.constant)
        expected = (
            str(task.coefficient * task.first_input + task.constant),
            str(task.coefficient * task.second_input + task.constant),
        )
        return DerivedTask(
            instruction=(
                "Evaluate both requested inputs and enter the ordered tuple "
                f"(f({task.first_input}), f({task.second_input}))."
            ),
            givens=(_math(f"f(x)={function}"),),
            expected=expected,
            answer_kind="ordered_tuple",
            conceptual_hint="The first tuple entry belongs to the first listed input.",
            operation_hint="Substitute and simplify each input separately, then keep their order.",
        )
    if isinstance(task, AffineCompositionTask):
        f_rule = _affine(task.f_coefficient, task.f_constant, variable="u")
        g_rule = _affine(task.g_coefficient, task.g_constant, variable="v")
        f_in_x = _affine(task.f_coefficient, task.f_constant)
        g_in_x = _affine(task.g_coefficient, task.g_constant)
        f_after_g = _affine(
            task.f_coefficient * task.g_coefficient,
            task.f_coefficient * task.g_constant + task.f_constant,
        )
        g_after_f = _affine(
            task.g_coefficient * task.f_coefficient,
            task.g_coefficient * task.f_constant + task.g_constant,
        )
        if task.order == "f_after_g":
            name = "f(g(x))"
            expected = f_after_g
            wrong = g_after_f
            order_hint = "Use g(x) as the complete input to f."
            worked_step = _append_constant(
                f"{task.f_coefficient}*({g_in_x})",
                task.f_constant,
            )
        else:
            name = "g(f(x))"
            expected = g_after_f
            wrong = f_after_g
            order_hint = "Use f(x) as the complete input to g."
            worked_step = _append_constant(
                f"{task.g_coefficient}*({f_in_x})",
                task.g_constant,
            )
        return DerivedTask(
            instruction=f"Build {name} and simplify it as an expression in x.",
            givens=(_math(f"f(u)={f_rule}"), _math(f"g(v)={g_rule}")),
            expected=expected,
            answer_kind="symbolic",
            conceptual_hint="Read composition from the inside outward.",
            operation_hint=order_hint + " Distribute, then combine constants.",
            worked_step=worked_step,
            error_signatures=(
                _signature(
                    wrong,
                    "m.composition.reverses_order",
                    "kc.fun.function_notation",
                ),
                _signature(
                    f"({f_in_x})*({g_in_x})",
                    "m.composition.multiplies_functions",
                    "kc.fun.function_notation",
                ),
            ),
        )
    if isinstance(task, QuadraticOuterCompositionTask):
        f_rule = _scaled_power(
            task.outer_scale,
            "u",
            2,
            task.outer_constant,
        )
        g_rule = _affine(task.inner_coefficient, task.inner_constant)
        expected = _scaled_power(
            task.outer_scale,
            g_rule,
            2,
            task.outer_constant,
        )
        return DerivedTask(
            instruction="Build f(g(x)). Keep the squared inner expression grouped.",
            givens=(_math(f"f(u)={f_rule}"), _math(f"g(x)={g_rule}")),
            expected=expected,
            answer_kind="symbolic",
            conceptual_hint="The complete output of g becomes the input u of f.",
            operation_hint=(
                "Replace u with the parenthesized affine expression and keep the "
                "outer square and scale attached."
            ),
            error_signatures=(
                _signature(
                    f"({_scaled_power(task.outer_scale, 'x', 2, task.outer_constant)})"
                    f"*({g_rule})",
                    "m.composition.multiplies_functions",
                    "kc.fun.function_notation",
                ),
                _signature(
                    _append_constant(
                        f"{_term(task.outer_scale, f'({task.inner_coefficient}*x)^2')}"
                        f"+{task.inner_constant}",
                        task.outer_constant,
                    ),
                    "m.composition.drops_inner_grouping",
                    "kc.fun.function_notation",
                ),
            ),
        )
    if isinstance(task, CompositionAtPointTask):
        f_rule = _affine(task.f_coefficient, task.f_constant)
        g_rule = _affine(task.g_coefficient, task.g_constant)
        f_after_g, g_after_f = _composition_values(
            f_coefficient=task.f_coefficient,
            f_constant=task.f_constant,
            g_coefficient=task.g_coefficient,
            g_constant=task.g_constant,
            point=task.point,
        )
        expected = f_after_g if task.order == "f_after_g" else g_after_f
        wrong = g_after_f if task.order == "f_after_g" else f_after_g
        notation = "f(g(p))" if task.order == "f_after_g" else "g(f(p))"
        return DerivedTask(
            instruction=f"Evaluate {notation} at p={task.point}.",
            givens=(_math(f"f(x)={f_rule}"), _math(f"g(x)={g_rule}")),
            expected=str(expected),
            answer_kind="numeric",
            conceptual_hint="Evaluate the inner function first; its output is the outer input.",
            operation_hint="Compute the inner output, then substitute it into the outer rule.",
            error_signatures=(
                _signature(
                    str(wrong),
                    "m.composition.reverses_order",
                    "kc.fun.function_notation",
                ),
            ),
        )
    if isinstance(task, CompositionTablePathTask):
        g_table = TablePromptSegment(
            role=PromptSemanticRole.GIVEN,
            caption="Selected values of g",
            column_headers=("input", "g(input)"),
            rows=((str(task.point), str(task.g_at_point)),),
            spoken_text=f"The table gives g({task.point})={task.g_at_point}.",
        )
        f_table = TablePromptSegment(
            role=PromptSemanticRole.GIVEN,
            caption="Selected values of f",
            column_headers=("input", "f(input)"),
            rows=(
                (str(task.g_at_point), str(task.f_at_g)),
                (str(task.distractor_input), str(task.f_at_distractor)),
            ),
            spoken_text=(
                f"The table gives f({task.g_at_point})={task.f_at_g} and "
                f"f({task.distractor_input})={task.f_at_distractor}."
            ),
        )
        return DerivedTask(
            instruction=(
                "Follow the table path for f(g(p)), then find "
                f"f(g({task.point}))-g({task.point})."
            ),
            givens=(g_table, f_table),
            expected=str(task.f_at_g - task.g_at_point),
            answer_kind="numeric",
            conceptual_hint="The output from the g table becomes the input used in the f table.",
            operation_hint="Find the final outer output, then subtract the inner output.",
        )
    if isinstance(task, CompositionPairedOrdersTask):
        f_rule = _affine(task.f_coefficient, task.f_constant)
        g_rule = _affine(task.g_coefficient, task.g_constant)
        f_after_g, g_after_f = _composition_values(
            f_coefficient=task.f_coefficient,
            f_constant=task.f_constant,
            g_coefficient=task.g_coefficient,
            g_constant=task.g_constant,
            point=task.point,
        )
        expected = (str(f_after_g), str(g_after_f))
        return DerivedTask(
            instruction=(
                f"At x={task.point}, enter the ordered tuple (f(g(x)), g(f(x)))."
            ),
            givens=(_math(f"f(x)={f_rule}"), _math(f"g(x)={g_rule}")),
            expected=expected,
            answer_kind="ordered_tuple",
            conceptual_hint="The two tuple entries use opposite inner functions.",
            operation_hint="Complete f after g first, then restart and complete g after f.",
            error_signatures=(
                _signature(
                    f"({expected[1]}, {expected[0]})",
                    "m.composition.reverses_order",
                    "kc.fun.function_notation",
                ),
            ),
        )
    if isinstance(task, CompositionPlotSumTask):
        table = TablePromptSegment(
            role=PromptSemanticRole.CONTEXT,
            caption="Exact points shown in both plots",
            column_headers=("function", "input", "output"),
            rows=(
                ("f", str(task.point), str(task.f_at_point)),
                ("f", str(task.g_at_point), str(task.f_at_g)),
                ("g", str(task.point), str(task.g_at_point)),
                ("g", str(task.f_at_point), str(task.g_at_f)),
            ),
            spoken_text=(
                "The table lists every exact point used from the two static plots."
            ),
        )
        plot = PlotPromptSegment(
            role=PromptSemanticRole.GIVEN,
            title="Exact values of f and g",
            x_label="input",
            y_label="output",
            series=(
                StaticPlotSeries(
                    label="f",
                    points=(
                        StaticPlotPoint(x=str(task.point), y=str(task.f_at_point)),
                        StaticPlotPoint(x=str(task.g_at_point), y=str(task.f_at_g)),
                    ),
                ),
                StaticPlotSeries(
                    label="g",
                    points=(
                        StaticPlotPoint(x=str(task.point), y=str(task.g_at_point)),
                        StaticPlotPoint(x=str(task.f_at_point), y=str(task.g_at_f)),
                    ),
                ),
            ),
            spoken_text=(
                f"At input {task.point}, g outputs {task.g_at_point} and f outputs "
                f"{task.f_at_point}. At input {task.g_at_point}, f outputs "
                f"{task.f_at_g}. At input {task.f_at_point}, g outputs {task.g_at_f}."
            ),
            equivalent_table=table,
        )
        return DerivedTask(
            instruction=(
                f"Use the exact plot data to find f(g({task.point}))+g(f({task.point}))."
            ),
            givens=(plot,),
            expected=str(task.f_at_g + task.g_at_f),
            answer_kind="numeric",
            conceptual_hint="Trace each composition from its inner output to the other graph.",
            operation_hint="Find the two final composition outputs, then add them.",
            error_signatures=(
                _signature(
                    str(task.f_at_point * task.g_at_point),
                    "m.composition.multiplies_functions",
                    "kc.fun.function_notation",
                ),
            ),
        )
    if isinstance(task, AffinePowerChainTask):
        inner = _affine(task.inner_coefficient, task.inner_constant)
        given = f"h(x)=({inner})^{task.outer_power}"
        coefficient = task.outer_power * task.inner_coefficient
        expected = _scaled_power(coefficient, inner, task.outer_power - 1)
        return DerivedTask(
            instruction="Differentiate h with respect to x using the chain rule.",
            givens=(_math(given),),
            expected=expected,
            answer_kind="symbolic",
            conceptual_hint=(
                "Differentiate the outer power while keeping the inner expression, "
                "then multiply by the inner derivative."
            ),
            operation_hint=(
                f"The outer exponent becomes {task.outer_power - 1}; multiply the "
                f"front coefficient by the inner derivative {task.inner_coefficient}."
            ),
            error_signatures=(
                _signature(
                    _scaled_power(task.outer_power, inner, task.outer_power - 1),
                    "m.chain_rule.missing_inner_derivative",
                    "kc.fun.composition",
                ),
                _signature(
                    str(task.inner_coefficient),
                    "m.chain_rule.inner_derivative_only",
                    "kc.der.power_rule",
                ),
            ),
        )
    if isinstance(task, QuadraticPowerChainTask):
        inner = _append_constant(
            _term(task.quadratic_coefficient, "x^2"),
            task.inner_constant,
        )
        given = f"h(x)=({inner})^{task.outer_power}"
        coefficient = (
            task.outer_power * 2 * task.quadratic_coefficient
        )
        expected = f"{_term(coefficient, 'x')}*({inner})^{task.outer_power - 1}"
        return DerivedTask(
            instruction="Differentiate h with respect to x using the chain rule.",
            givens=(_math(given),),
            expected=expected,
            answer_kind="symbolic",
            conceptual_hint=(
                "Keep one copy of the inner expression under the reduced outer power, "
                "then multiply by the derivative of the quadratic inner function."
            ),
            operation_hint=(
                f"Differentiate the inner expression to {_term(2 * task.quadratic_coefficient, 'x')} "
                "and multiply that factor by the outer-power derivative."
            ),
            worked_step=(
                f"{task.outer_power}*({inner})^{task.outer_power - 1}"
                f"*({_term(2 * task.quadratic_coefficient, 'x')})"
            ),
            error_signatures=(
                _signature(
                    _scaled_power(task.outer_power, inner, task.outer_power - 1),
                    "m.chain_rule.missing_inner_derivative",
                    "kc.fun.composition",
                ),
                _signature(
                    _term(2 * task.quadratic_coefficient, "x"),
                    "m.chain_rule.inner_derivative_only",
                    "kc.der.power_rule",
                ),
            ),
        )
    if isinstance(task, ChainAtPointTask):
        expected = (
            task.outer_power
            * task.inner_value ** (task.outer_power - 1)
            * task.inner_derivative
        )
        return DerivedTask(
            instruction=(
                f"Let h(x)=(u(x))^{task.outer_power}. Find h'({task.point}) "
                "from the supplied local data."
            ),
            givens=(
                _math(f"u({task.point})={task.inner_value}"),
                _math(f"u'({task.point})={task.inner_derivative}"),
            ),
            expected=str(expected),
            answer_kind="numeric",
            conceptual_hint=(
                "At the point, the chain rule needs both the inner value and the "
                "inner derivative."
            ),
            operation_hint=(
                "Compute n times the inner value to the power n minus one, then "
                "multiply by the supplied inner derivative."
            ),
            error_signatures=(
                _signature(
                    str(task.outer_power * task.inner_value ** (task.outer_power - 1)),
                    "m.chain_rule.missing_inner_derivative",
                    "kc.fun.composition",
                ),
                _signature(
                    str(task.inner_derivative),
                    "m.chain_rule.inner_derivative_only",
                    "kc.der.power_rule",
                ),
            ),
        )
    if isinstance(task, ChainTableValuesTask):
        def derivative(value: int, inner_derivative: int) -> int:
            return (
                task.outer_power
                * value ** (task.outer_power - 1)
                * inner_derivative
            )

        expected = (
            str(derivative(task.first_inner_value, task.first_inner_derivative)),
            str(derivative(task.second_inner_value, task.second_inner_derivative)),
        )
        missing = (
            str(task.outer_power * task.first_inner_value ** (task.outer_power - 1)),
            str(task.outer_power * task.second_inner_value ** (task.outer_power - 1)),
        )
        table = TablePromptSegment(
            role=PromptSemanticRole.GIVEN,
            caption="Local data for u",
            column_headers=("x", "u(x)", "u'(x)"),
            rows=(
                (
                    str(task.first_point),
                    str(task.first_inner_value),
                    str(task.first_inner_derivative),
                ),
                (
                    str(task.second_point),
                    str(task.second_inner_value),
                    str(task.second_inner_derivative),
                ),
            ),
            spoken_text=(
                f"At {task.first_point}, u is {task.first_inner_value} and u prime is "
                f"{task.first_inner_derivative}. At {task.second_point}, u is "
                f"{task.second_inner_value} and u prime is {task.second_inner_derivative}."
            ),
        )
        return DerivedTask(
            instruction=(
                f"For h(x)=(u(x))^{task.outer_power}, use the table to enter "
                f"(h'({task.first_point}), h'({task.second_point}))."
            ),
            givens=(table,),
            expected=expected,
            answer_kind="ordered_tuple",
            conceptual_hint="Each derivative value needs both u(x) and u'(x) from its row.",
            operation_hint="Apply n times u to the n minus one times u prime in each row.",
            error_signatures=(
                _signature(
                    f"({missing[0]}, {missing[1]})",
                    "m.chain_rule.missing_inner_derivative",
                    "kc.fun.composition",
                ),
            ),
        )
    if isinstance(task, ChainFactorTupleTask):
        inner = _affine(task.inner_coefficient, task.inner_constant)
        outer_factor = _scaled_power(task.outer_power, inner, task.outer_power - 1)
        expected = (outer_factor, str(task.inner_coefficient))
        return DerivedTask(
            instruction=(
                "For the displayed h, enter the ordered tuple "
                "(outer derivative factor, inner derivative)."
            ),
            givens=(_math(f"h(x)=({inner})^{task.outer_power}"),),
            expected=expected,
            answer_kind="ordered_tuple",
            conceptual_hint="Differentiate the outer layer first while copying its inner input.",
            operation_hint="The second tuple entry is the derivative of the affine inner rule.",
            error_signatures=(
                _signature(
                    f"({outer_factor}, 1)",
                    "m.chain_rule.missing_inner_derivative",
                    "kc.fun.composition",
                ),
            ),
        )
    if isinstance(task, ChainCorrectionTask):
        if task.inner_kind == "affine":
            inner = _affine(task.inner_coefficient, task.inner_constant)
            inner_derivative = str(task.inner_coefficient)
            expected = _scaled_power(
                task.outer_power * task.inner_coefficient,
                inner,
                task.outer_power - 1,
            )
        else:
            inner = _append_constant(
                _term(task.inner_coefficient, "x^2"),
                task.inner_constant,
            )
            inner_derivative = _term(2 * task.inner_coefficient, "x")
            expected = (
                f"{_term(task.outer_power * 2 * task.inner_coefficient, 'x')}"
                f"*({inner})^{task.outer_power - 1}"
            )
        missing_inner = _scaled_power(
            task.outer_power,
            inner,
            task.outer_power - 1,
        )
        return DerivedTask(
            instruction="Correct the proposed derivative by supplying the missing chain factor.",
            givens=(
                _math(f"h(x)=({inner})^{task.outer_power}"),
                _math(f"proposed h'(x)={missing_inner}"),
            ),
            expected=expected,
            answer_kind="symbolic",
            conceptual_hint="The proposed work differentiates only the outer layer.",
            operation_hint=f"Multiply the proposed expression by {inner_derivative}.",
            error_signatures=(
                _signature(
                    missing_inner,
                    "m.chain_rule.missing_inner_derivative",
                    "kc.fun.composition",
                ),
                _signature(
                    inner_derivative,
                    "m.chain_rule.inner_derivative_only",
                    "kc.der.power_rule",
                ),
            ),
        )
    raise TypeError(f"unsupported Chain Rule task {type(task).__name__}")


_TASK_COMPILER_REGISTRY = TaskCompilerRegistry(
    TaskCompilerRegistration(
        kind=kind,
        task_type=task_type,
        construct_id=construct_id,
        kc_id=kc_id,
        compile=_derive_unregistered,
    )
    for kind, task_type, construct_id, kc_id in (
        (
            "function_affine_value",
            AffineFunctionValueTask,
            "function.affine_value",
            "kc.fun.function_notation",
        ),
        (
            "function_quadratic_value",
            QuadraticFunctionValueTask,
            "function.quadratic_value",
            "kc.fun.function_notation",
        ),
        (
            "function_expression_value",
            FunctionExpressionValueTask,
            "function.expression_value",
            "kc.fun.function_notation",
        ),
        (
            "function_table_combination",
            FunctionTableCombinationTask,
            "function.table_combination",
            "kc.fun.function_notation",
        ),
        (
            "function_plot_change",
            FunctionPlotChangeTask,
            "function.plot_change",
            "kc.fun.function_notation",
        ),
        (
            "function_ordered_values",
            FunctionOrderedValuesTask,
            "function.ordered_values",
            "kc.fun.function_notation",
        ),
        (
            "composition_affine",
            AffineCompositionTask,
            "composition.affine",
            "kc.fun.composition",
        ),
        (
            "composition_quadratic_outer",
            QuadraticOuterCompositionTask,
            "composition.quadratic_outer",
            "kc.fun.composition",
        ),
        (
            "composition_at_point",
            CompositionAtPointTask,
            "composition.at_point",
            "kc.fun.composition",
        ),
        (
            "composition_table_path",
            CompositionTablePathTask,
            "composition.table_path",
            "kc.fun.composition",
        ),
        (
            "composition_paired_orders",
            CompositionPairedOrdersTask,
            "composition.paired_orders",
            "kc.fun.composition",
        ),
        (
            "composition_plot_sum",
            CompositionPlotSumTask,
            "composition.plot_sum",
            "kc.fun.composition",
        ),
        (
            "chain_affine_power",
            AffinePowerChainTask,
            "chain.affine_power",
            "kc.der.chain_rule",
        ),
        (
            "chain_quadratic_power",
            QuadraticPowerChainTask,
            "chain.quadratic_power",
            "kc.der.chain_rule",
        ),
        (
            "chain_at_point",
            ChainAtPointTask,
            "chain.at_point",
            "kc.der.chain_rule",
        ),
        (
            "chain_table_values",
            ChainTableValuesTask,
            "chain.table_values",
            "kc.der.chain_rule",
        ),
        (
            "chain_factor_tuple",
            ChainFactorTupleTask,
            "chain.factor_tuple",
            "kc.der.chain_rule",
        ),
        (
            "chain_correction",
            ChainCorrectionTask,
            "chain.correction",
            "kc.der.chain_rule",
        ),
    )
)


def derive_task(task: ChainRuleMathTask) -> DerivedTask:
    """Compile one typed task through the closed constructor registry."""

    derived = _TASK_COMPILER_REGISTRY.compile(task)
    if not isinstance(derived, DerivedTask):
        raise ChainRuleCompilationError(
            f"task compiler returned unsupported output {type(derived).__name__}"
        )
    return derived


def _guided_interaction_for(
    family: ChainRuleFamilyBlueprint,
    derived: DerivedTask,
) -> GuidedInteractionSpec | None:
    if family.surface != AssessmentSurface.GUIDED_WIDGET:
        return None
    if derived.answer_kind != "numeric" or not isinstance(derived.expected, str):
        raise ChainRuleCompilationError(
            f"guided family {family.item_id!r} must compile to one numeric answer"
        )
    target = float(derived.expected)
    if not -20 <= target <= 20:
        raise ChainRuleCompilationError(
            f"guided family {family.item_id!r} exceeds the fixed slider range"
        )
    return GuidedSliderSpec(
        presentation=GuidedSliderPresentation(
            prompt="Choose the numeric answer to the exact problem shown above.",
            label="Your answer",
            help_text=(
                "Use the arrow keys or slider in steps of one. The text fallback "
                "asks for the same numeric answer."
            ),
            minimum=-20,
            maximum=20,
            step=1,
            initial_value=0,
            value_label="Selected answer",
            result_template="The selected answer is {value}.",
        ),
        scoring=GuidedSliderScoring(target=target, tolerance=0),
    )


def _item_submission(item: AssessmentItem) -> str:
    answer = item.answer
    if isinstance(answer, (NumericAnswerSpec, SymbolicAnswerSpec)):
        return answer.expected
    if isinstance(answer, OrderedTupleAnswerSpec):
        return "(" + ", ".join(answer.expected) + ")"
    raise ChainRuleCompilationError(
        f"{item.item_id}: unsupported cumulative separation contract {answer.kind}"
    )


def load_source(path: Path | None = None) -> ChainRuleBlueprintDocument:
    source = path or DEFAULT_SOURCE_PATH
    return ChainRuleBlueprintDocument.model_validate_json(
        source.read_text(encoding="utf-8")
    )


def load_manifest(path: Path | None = None) -> ContentReviewManifest:
    source = path or DEFAULT_MANIFEST_PATH
    return ContentReviewManifest.model_validate_json(
        source.read_text(encoding="utf-8")
    )


def _review_status_and_provenance(
    source: ChainRuleBlueprintDocument,
    family: ChainRuleFamilyBlueprint,
    review: ContentReviewEntry,
) -> tuple[ReviewStatus, AssessmentProvenance]:
    if review.decision == ReviewDecision.REJECTED:
        raise ChainRuleCompilationError("rejected families cannot be compiled")
    approved = review.decision == ReviewDecision.APPROVED
    if approved:
        if review.reviewed_by is None or review.reviewed_at is None:
            raise ChainRuleCompilationError("approved family lacks review provenance")
        if review.reviewed_by.strip().casefold() == source.author.strip().casefold():
            raise ChainRuleCompilationError("a family author cannot approve their own work")
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


def _build_item(
    source: ChainRuleBlueprintDocument,
    family: ChainRuleFamilyBlueprint,
    *,
    review_status: ReviewStatus,
    provenance: AssessmentProvenance,
) -> AssessmentItem:
    derived = derive_task(family.task)
    givens = list(derived.givens)
    if derived.answer_kind == "numeric" and isinstance(derived.expected, str):
        answer = NumericAnswerSpec(expected=derived.expected, tolerance=0)
    elif derived.answer_kind == "symbolic" and isinstance(derived.expected, str):
        answer = SymbolicAnswerSpec(expected=derived.expected, variables=["x"])
    elif derived.answer_kind == "ordered_tuple" and isinstance(derived.expected, tuple):
        answer = OrderedTupleAnswerSpec(expected=list(derived.expected), variables=["x"])
    else:
        raise ChainRuleCompilationError(
            f"{family.item_id}: constructor answer kind and expected value disagree"
        )
    for signature in derived.error_signatures:
        # This is an offline, closed-constructor check. Runtime learner answers
        # still use the supervised verifier pool; spawning one worker per
        # compiled signature would make deterministic authoring needlessly slow.
        verdict = verify_answer(answer, signature.expected_wrong, supervised=False)
        if verdict.status != VerificationStatus.INCORRECT:
            raise ChainRuleCompilationError(
                f"{family.item_id}: error signature is not an executable wrong answer "
                f"({verdict.code})"
            )
    if family.surface == AssessmentSurface.WORKED_EXAMPLE:
        if derived.worked_step is None:
            raise ChainRuleCompilationError(
                f"{family.item_id}: worked example lacks a task-derived math step"
            )
        prompt = [
            TextPromptSegment(
                role=PromptSemanticRole.INSTRUCTION,
                text="Study this worked example. " + derived.instruction,
            ),
            *givens,
            TextPromptSegment(
                role=PromptSemanticRole.WORKED_STEP,
                text=derived.operation_hint,
            ),
            MathPromptSegment(
                role=PromptSemanticRole.WORKED_STEP,
                expression=derived.worked_step,
                spoken_text=_spoken_math(derived.worked_step),
            ),
            TextPromptSegment(
                role=PromptSemanticRole.WORKED_STEP,
                text="Therefore, the final answer is",
            ),
            MathPromptSegment(
                role=PromptSemanticRole.WORKED_ANSWER,
                expression=derived.submission,
                spoken_text=_spoken_math(derived.submission),
            ),
        ]
    else:
        prefix = (
            "Use the guided activity or its equivalent text fallback. "
            if family.surface == AssessmentSurface.GUIDED_WIDGET
            else ""
        )
        prompt = [
            TextPromptSegment(
                role=PromptSemanticRole.INSTRUCTION,
                text=prefix + derived.instruction,
            ),
            *givens,
            BlankPromptSegment(label="Answer:"),
        ]
    return AssessmentItem(
        item_id=family.item_id,
        revision=family.revision,
        family_id=family.family_id,
        kc_id=family.kc_id,
        difficulty=family.difficulty,
        task_kind=AssessmentTaskKind.SOLVE,
        eligible_surfaces=[family.surface],
        allocation_order=family.allocation_order,
        prompt=prompt,
        hints=[
            AssessmentHint(text=derived.conceptual_hint),
            AssessmentHint(text=derived.operation_hint),
            AssessmentHint(
                text=f"A correct completed form is {derived.submission}.",
                revealing=True,
            ),
        ],
        answer=answer,
        review_status=review_status,
        provenance=provenance,
        error_signatures=list(derived.error_signatures),
        guided_interaction=_guided_interaction_for(family, derived),
    )


def _compiled_review_artifact(
    source: ChainRuleBlueprintDocument,
    family: ChainRuleFamilyBlueprint,
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
    provenance = artifact["provenance"]
    if not isinstance(provenance, dict):
        raise TypeError("compiled provenance must serialize as an object")
    for field in ("reviewed_by", "reviewed_at", "source_digest"):
        provenance.pop(field, None)
    return artifact


def family_digest(
    source: ChainRuleBlueprintDocument,
    family: ChainRuleFamilyBlueprint,
) -> str:
    """Bind approval only to the exact family-local bytes under review.

    Cumulative document coordinates, output-bank coordinates, and release
    membership intentionally do not participate. The inventory CLI reads a
    separately authored review manifest and can write only a compiled bank;
    it has no command or code path that generates or overwrites review records.
    """

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


def _visible_content(item: AssessmentItem) -> list[str]:
    visible = [render_prompt(item)]
    for segment in item.prompt:
        visible.extend(_segment_visible_fragments(segment))
    visible.extend(hint.text for hint in item.hints)
    interaction = item.guided_interaction
    if isinstance(interaction, GuidedMappingSpec):
        visible.append(interaction.presentation.prompt)
        visible.extend(
            text
            for entry in (
                *interaction.presentation.rows,
                *interaction.presentation.options,
            )
            for text in (entry.label, entry.spoken_text)
        )
    elif isinstance(interaction, GuidedSliderSpec):
        presentation = interaction.presentation
        visible.extend(
            [
                presentation.prompt,
                presentation.label,
                presentation.help_text,
                presentation.value_label,
                presentation.result_template or "",
                str(presentation.minimum),
                str(presentation.maximum),
                str(presentation.initial_value),
            ]
        )
    return list(dict.fromkeys(text for text in visible if text))


def _separation_worker(
    connection: Connection,
    expected_payload: list[tuple[str, dict[str, object], str]],
    visible_payload: list[tuple[str, str, str]],
) -> None:
    try:
        answers = {
            item_id: answer_spec_adapter.validate_python(specification)
            for item_id, specification, _submission in expected_payload
        }
        submissions = {
            item_id: submission
            for item_id, _specification, submission in expected_payload
        }
        scalar_kinds = {"numeric", "symbolic"}

        def canonical_scalar(item_id: str, expression: str) -> sympy.Expr:
            specification = answers[item_id]
            variables = set(getattr(specification, "variables", ()))
            functions = set(getattr(specification, "functions", ()))
            parsed = parse_restricted(
                expression,
                allowed_variables=variables,
                allowed_functions=functions,
                allowed_assignment_lhs=getattr(
                    specification,
                    "assignment_lhs",
                    None,
                ),
            )
            if specification.kind == "numeric" and parsed.free_symbols:
                raise ValueError("numeric candidate contains a variable")
            if parsed.has(sympy.zoo, sympy.nan, sympy.oo) or parsed.is_finite is False:
                raise ValueError("candidate is not finite")
            return sympy.cancel(parsed)

        canonical_expected = {
            item_id: canonical_scalar(item_id, submissions[item_id])
            for item_id, answer in answers.items()
            if answer.kind in scalar_kinds
        }
        ordered_ids = sorted(answers)
        reused: list[tuple[str, str]] = []
        pair_indeterminate: list[tuple[str, str]] = []
        comparisons = 0
        for index, left_id in enumerate(ordered_ids):
            for right_id in ordered_ids[index + 1 :]:
                comparisons += 1
                left = answers[left_id]
                right = answers[right_id]
                collection_kinds = {"finite_set", "interval_set", "ordered_tuple"}
                if left.kind != right.kind and (
                    left.kind in collection_kinds or right.kind in collection_kinds
                ):
                    continue
                if left.kind in scalar_kinds and right.kind in scalar_kinds:
                    if sympy.cancel(
                        canonical_expected[left_id] - canonical_expected[right_id]
                    ) == 0:
                        reused.append((left_id, right_id))
                    continue
                left_verdict = verify_answer(
                    left,
                    submissions[right_id],
                    supervised=False,
                )
                right_verdict = verify_answer(
                    right,
                    submissions[left_id],
                    supervised=False,
                )
                if (
                    left_verdict.status == VerificationStatus.CORRECT
                    or right_verdict.status == VerificationStatus.CORRECT
                ):
                    reused.append((left_id, right_id))
                elif not any(
                    verdict.status == VerificationStatus.INCORRECT
                    for verdict in (left_verdict, right_verdict)
                ):
                    pair_indeterminate.append((left_id, right_id))
        leaks: list[tuple[str, str]] = []
        indeterminate: list[tuple[str, str]] = []
        for source_id, target_id, candidate in visible_payload:
            if answers[target_id].kind in scalar_kinds:
                try:
                    parsed = canonical_scalar(target_id, candidate)
                except Exception:
                    indeterminate.append((source_id, target_id))
                    continue
                if sympy.cancel(parsed - canonical_expected[target_id]) == 0:
                    leaks.append((source_id, target_id))
                continue
            verdict = verify_answer(
                answers[target_id],
                candidate,
                supervised=False,
            )
            if verdict.status == VerificationStatus.CORRECT:
                leaks.append((source_id, target_id))
            elif verdict.status != VerificationStatus.INCORRECT:
                indeterminate.append((source_id, target_id))
        connection.send(
            {
                "reused": reused,
                "pair_indeterminate": pair_indeterminate,
                "leaks": leaks,
                "indeterminate": indeterminate,
                "answer_comparisons": comparisons,
                "visible_comparisons": len(visible_payload),
            }
        )
    except BaseException as exc:  # noqa: BLE001 - worker must fail closed
        connection.send({"error": type(exc).__name__})
    finally:
        connection.close()


def _run_math_separation(
    items: list[AssessmentItem],
    *,
    timeout_seconds: float = 15.0,
) -> dict[str, object]:
    expected_payload = [
        (item.item_id, item.answer.model_dump(mode="json"), _item_submission(item))
        for item in items
    ]
    candidates_by_source: dict[str, list[str]] = {}
    for item in items:
        candidates = [
            segment.expression
            for segment in item.prompt
            if isinstance(segment, MathPromptSegment)
        ]
        candidates.extend(
            candidate
            for visible in _visible_content(item)
            for candidate in _candidate_answer_texts(visible)
        )
        candidates_by_source[item.item_id] = list(dict.fromkeys(candidates))
    visible_payload = list(
        dict.fromkeys(
            (source.item_id, target.item_id, candidate)
            for source in items
            for target in items
            if source.family_id != target.family_id
            for candidate in candidates_by_source[source.item_id]
            if _candidate_fits_answer_contract(target, candidate)
        )
    )
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_separation_worker,
        args=(child, expected_payload, visible_payload),
        daemon=True,
    )
    process.start()
    child.close()
    try:
        if not parent.poll(timeout_seconds):
            raise ChainRuleCompilationError("inventory separation worker timed out")
        result = parent.recv()
    except EOFError as exc:
        raise ChainRuleCompilationError(
            "inventory separation worker exited without a result"
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
    if process.is_alive():
        raise ChainRuleCompilationError("inventory separation worker could not stop")
    if "error" in result:
        raise ChainRuleCompilationError(
            f"inventory separation worker failed ({result['error']})"
        )
    return result


def _literal_answer_visible(expected: str, visible: str) -> bool:
    compact_expected = re.sub(r"\s+", "", expected).lower()
    compact_visible = re.sub(r"\s+", "", visible).lower()
    if len(compact_expected) >= 3 and compact_expected in compact_visible:
        return True
    scalar = expected.strip()
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", scalar):
        return False
    return bool(
        re.search(
            rf"(?<![A-Za-z0-9_.^*/+\-]){re.escape(scalar)}"
            rf"(?![A-Za-z0-9_^*/])(?!\.\d)",
            visible,
        )
    )


def _literal_answer_values(item: AssessmentItem) -> tuple[str, ...]:
    answer = item.answer
    if isinstance(answer, (NumericAnswerSpec, SymbolicAnswerSpec)):
        return (answer.expected,)
    if isinstance(answer, OrderedTupleAnswerSpec):
        return tuple(answer.expected)
    raise ChainRuleCompilationError(
        f"{item.item_id}: unsupported literal-separation contract {answer.kind}"
    )


def validate_inventory_separation(
    items: list[AssessmentItem],
    graph: GraphDocument,
    *,
    focus_item_ids: set[str] | None = None,
) -> InventorySeparationReport:
    """Reject answer reuse, leakage, and indeterminate comparisons.

    ``focus_item_ids`` supports cumulative authoring: every comparison is still
    executed and counted, while failures wholly inside an already-qualified
    predecessor bank are left to that predecessor's frozen report.
    """

    ordered = sorted(items, key=lambda item: (item.family_id, item.item_id))
    item_ids = {item.item_id for item in ordered}
    if focus_item_ids is not None and not focus_item_ids <= item_ids:
        raise ChainRuleCompilationError("separation focus names an unknown item")

    def in_scope(left_id: str, right_id: str) -> bool:
        return focus_item_ids is None or bool(
            {left_id, right_id} & focus_item_ids
        )

    errors: list[str] = []
    # Keep the single-wave check fast while allowing the deliberately larger
    # cumulative qualification a bounded amount of time. Every path still has
    # a hard wall-clock limit and a replaceable worker.
    timeout_seconds = max(15.0, min(60.0, len(ordered) * 0.5))
    math_result = _run_math_separation(
        ordered,
        timeout_seconds=timeout_seconds,
    )
    by_id = {item.item_id: item for item in ordered}
    for left_id, right_id in math_result["reused"]:
        if not in_scope(left_id, right_id):
            continue
        errors.append(
            "expected answer reused across families "
            f"{by_id[left_id].family_id!r} and {by_id[right_id].family_id!r}"
        )
    for left_id, right_id in math_result["pair_indeterminate"]:
        if not in_scope(left_id, right_id):
            continue
        errors.append(
            f"answer comparison between {left_id} and {right_id} was indeterminate"
        )
    for source_id, target_id in math_result["leaks"]:
        if not in_scope(source_id, target_id):
            continue
        errors.append(
            f"{source_id}: visible math is equivalent to the answer for {target_id}"
        )
    for source_id, target_id in math_result["indeterminate"]:
        if not in_scope(source_id, target_id):
            continue
        errors.append(
            f"{source_id}: visible comparison for {target_id} was indeterminate"
        )
    literal_pairs = len(ordered) * max(0, len(ordered) - 1)
    for source_item in ordered:
        visible = "\n".join(_visible_content(source_item))
        for target in ordered:
            if (
                target.family_id != source_item.family_id
                and in_scope(source_item.item_id, target.item_id)
                and any(
                    _literal_answer_visible(expected, visible)
                    for expected in _literal_answer_values(target)
                )
            ):
                errors.append(
                    f"{source_item.item_id}: visible content leaks answer for {target.item_id}"
                )
    item_kcs = {item.kc_id for item in ordered}
    graph_visible = "\n".join(
        text
        for node in graph.nodes
        if node.id in item_kcs
        for text in (node.name, node.description, *node.canonical_examples)
    )
    for target in ordered:
        if (
            focus_item_ids is None or target.item_id in focus_item_ids
        ) and any(
            _literal_answer_visible(expected, graph_visible)
            for expected in _literal_answer_values(target)
        ):
            errors.append(
                f"student-visible graph content leaks answer for {target.item_id}"
            )
    return InventorySeparationReport(
        answer_pairs_checked=int(math_result["answer_comparisons"]),
        visible_candidate_comparisons_checked=int(
            math_result["visible_comparisons"]
        ),
        literal_visible_pairs_checked=literal_pairs,
        errors=tuple(errors),
    )


def _validate_taxonomy(source: ChainRuleBlueprintDocument) -> None:
    for family in source.families:
        try:
            registration = _TASK_COMPILER_REGISTRY.resolve(family.task)
            _TASK_COMPILER_REGISTRY.validate_taxonomy(
                family.task,
                construct_id=family.construct_id,
                kc_id=family.kc_id,
            )
        except TaskCompilerRegistryError as exc:
            raise ChainRuleCompilationError(str(exc)) from exc
        if registration.kc_id != family.kc_id:
            raise ChainRuleCompilationError(
                f"{family.blueprint_id}: task belongs to {registration.kc_id}"
            )
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
                raise ChainRuleCompilationError(
                    f"{kc_id}/{surface.value}: construct/order mismatch; "
                    f"expected={expected}, got={actual}"
                )


def compile_release_inventory(
    source: ChainRuleBlueprintDocument,
    manifest: ContentReviewManifest,
    graph: GraphDocument,
) -> tuple[ItemBankDocument, InventorySeparationReport]:
    """Compile, verify, and exhaustively separate all 39 draft families."""

    if source.graph_version != graph.graph_version:
        raise ChainRuleCompilationError("source and graph versions differ")
    if manifest.graph_version != graph.graph_version:
        raise ChainRuleCompilationError("manifest and graph versions differ")
    if manifest.compiler_version != COMPILER_VERSION:
        raise ChainRuleCompilationError("manifest compiler pin is stale")
    if set(source.target_kcs) != set(TARGET_KCS):
        raise ChainRuleCompilationError("source must contain exactly the three Wave 2 KCs")
    closure = ancestor_subgraph(
        graph,
        "kc.der.chain_rule",
        hard_only=True,
    ).node_ids()
    if closure != set(EXPECTED_CHAIN_CLOSURE):
        raise ChainRuleCompilationError(
            f"chain-rule hard closure changed: {sorted(closure)}"
        )
    expected_identities = {
        (family.blueprint_id, family.revision) for family in source.families
    }
    reviews = {
        (entry.blueprint_id, entry.revision): entry for entry in manifest.entries
    }
    if set(reviews) != expected_identities:
        raise ChainRuleCompilationError("review/source identity coverage differs")
    expected_matrix = {
        (kc_id, surface): count
        for kc_id in TARGET_KCS
        for surface, count in EXPECTED_FAMILY_COUNTS.items()
    }
    if dict(Counter((family.kc_id, family.surface) for family in source.families)) != (
        expected_matrix
    ):
        raise ChainRuleCompilationError(
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
            raise ChainRuleCompilationError(
                f"review digest mismatch for {family.blueprint_id}@{family.revision}"
            )
        review_status, provenance = _review_status_and_provenance(
            source,
            family,
            review,
        )
        item = _build_item(
            source,
            family,
            review_status=review_status,
            provenance=provenance,
        )
        verdict = verify_answer(item.answer, _item_submission(item), supervised=True)
        if verdict.status != VerificationStatus.CORRECT:
            raise ChainRuleCompilationError(
                f"{item.item_id}: derived truth failed verification ({verdict.code})"
            )
        items.append(item)

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
        raise ChainRuleCompilationError(
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
        raise ChainRuleCompilationError(
            "inventory separation failed: " + "; ".join(report.errors)
        )
    return bank, report


def _atomic_write_bank(path: Path, bank: ItemBankDocument) -> None:
    payload = bank.model_dump_json(indent=2) + "\n"
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
        ItemBankDocument.model_validate_json(
            temporary_path.read_text(encoding="utf-8")
        )
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH_PATH)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if not args.check and args.out is None:
        parser.error("nothing to do: pass --check and/or --out PATH")
    try:
        source = load_source(args.source)
        manifest = load_manifest(args.manifest)
        graph = GraphDocument.model_validate_json(
            args.graph.read_text(encoding="utf-8")
        )
        bank, report = compile_release_inventory(source, manifest, graph)
    except Exception as exc:  # noqa: BLE001 - fail closed at CLI boundary
        print(f"Chain Rule inventory INVALID: {exc}", file=sys.stderr)
        return 1
    if args.out is not None:
        _atomic_write_bank(args.out, bank)
    status_counts = Counter(item.review_status for item in bank.items)
    print(
        "Chain Rule inventory OK: "
        f"{len(bank.items)} families, "
        f"{status_counts[ReviewStatus.DRAFT]} draft, "
        f"{status_counts[ReviewStatus.HUMAN_APPROVED]} approved, "
        f"{report.answer_pairs_checked} answer comparisons, "
        f"{report.visible_candidate_comparisons_checked} visible comparisons, "
        f"{report.literal_visible_pairs_checked} literal scans, "
        f"released KCs={len(bank.released_kcs)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
