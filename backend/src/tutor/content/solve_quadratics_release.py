"""Compile and qualify the pending Solve Quadratics content wave.

The closed compiler derives prompts, worked derivations, private truth,
guided interactions, and exact misconception signatures from typed source
math. It cannot approve, attest, or publish the unreviewed draft.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import cast

import sympy
from pydantic import BaseModel

from tutor.content.item_bank import (
    _candidate_answer_texts,
    _candidate_fits_answer_contract,
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
    AssessmentHint,
    AssessmentItem,
    AssessmentProvenance,
    AssessmentSurface,
    AssessmentTaskKind,
    BlankPromptSegment,
    ErrorSignature,
    FiniteSetAnswerSpec,
    GuidedInteractionSpec,
    GuidedMappingEntry,
    GuidedMappingPresentation,
    GuidedMappingScoring,
    GuidedMappingSpec,
    ItemBankDocument,
    MathPromptSegment,
    NumericAnswerSpec,
    OrderedTupleAnswerSpec,
    PromptSegment,
    PromptSemanticRole,
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
from tutor.schemas.kc import GraphDocument
from tutor.schemas.pedagogy_authoring import (
    PedagogyReviewDecision,
    PedagogyReviewEntry,
    PedagogyReviewManifest,
    PedagogySourceDocument,
)
from tutor.schemas.solve_quadratics_authoring import (
    BinomialProduct,
    CubicPolynomial,
    FactoringCore,
    FactoringCorrectionPortfolioTask,
    FactoringDirectPortfolioTask,
    FactoringGuidedMatchTask,
    FactoringMissingPortfolioTask,
    FactoringTablePortfolioTask,
    FactoringTransferPortfolioTask,
    FactoringVerificationPortfolioTask,
    GcfFactorSpec,
    LinearCorrectionTask,
    LinearDistributedTask,
    LinearDoubleDistributedTask,
    LinearEquationSpec,
    LinearGroupedTask,
    LinearGuidedBalanceTask,
    LinearOneSideTask,
    LinearReversedTask,
    LinearTwoSidedTask,
    MonicFactorSpec,
    PolynomialCoefficientAuditTask,
    PolynomialCore,
    PolynomialCorrectionPortfolioTask,
    PolynomialDirectPortfolioTask,
    PolynomialGuidedMatchTask,
    PolynomialMixedExpressionTask,
    PolynomialReversePortfolioTask,
    PolynomialSparsePortfolioTask,
    PolynomialTablePortfolioTask,
    QuadraticBothSidesTask,
    QuadraticCorrectionTask,
    QuadraticExpandedTask,
    QuadraticFactoredTask,
    QuadraticGuidedFactorMapTask,
    QuadraticRepeatedTask,
    QuadraticReversedTask,
    QuadraticShiftedTask,
    QuadraticSparseDifferenceTask,
    RootPair,
    SolveQuadraticsBlueprintDocument,
    SolveQuadraticsFamilyBlueprint,
    SolveQuadraticsMathTask,
)
from tutor.verify.checker import VerificationStatus, parse_restricted, verify_answer

COMPILER_VERSION = "solve-quadratics-item-compiler-v2"
AUTHOR = "AI-assisted implementation draft (unreviewed)"
TARGET_KC = "kc.alg.solve_quadratic"
TARGET_KCS = frozenset(
    {
        "kc.alg.polynomial_ops",
        "kc.alg.factoring",
        "kc.alg.solve_linear",
        TARGET_KC,
    }
)
EXPECTED_CLOSURE = TARGET_KCS
EXPECTED_FAMILY_COUNTS = {
    AssessmentSurface.DIAGNOSTIC: 4,
    AssessmentSurface.CHECKIN: 5,
    AssessmentSurface.GUIDED_WIDGET: 1,
    AssessmentSurface.CAPSTONE: 2,
    AssessmentSurface.WORKED_EXAMPLE: 1,
}
CORE_POLYNOMIAL_COVERAGE = frozenset({"add", "subtract", "expand"})
CORE_FACTORING_COVERAGE = frozenset(
    {"gcf", "monic_quadratic", "difference_squares"}
)

EXPECTED_CONSTRUCT_ORDER: dict[str, dict[AssessmentSurface, tuple[str, ...]]] = {
    "kc.alg.polynomial_ops": {
        AssessmentSurface.DIAGNOSTIC: (
            "polynomial.direct_portfolio",
            "polynomial.correction_portfolio",
            "polynomial.sparse_portfolio",
            "polynomial.mixed_expression",
        ),
        AssessmentSurface.CHECKIN: (
            "polynomial.coefficient_audit",
            "polynomial.reverse_portfolio",
            "polynomial.table_portfolio",
            "polynomial.sparse_portfolio",
            "polynomial.mixed_expression",
        ),
        AssessmentSurface.GUIDED_WIDGET: ("polynomial.guided_match",),
        AssessmentSurface.CAPSTONE: (
            "polynomial.reverse_portfolio",
            "polynomial.mixed_expression",
        ),
        AssessmentSurface.WORKED_EXAMPLE: ("polynomial.direct_portfolio",),
    },
    "kc.alg.factoring": {
        AssessmentSurface.DIAGNOSTIC: (
            "factoring.direct_portfolio",
            "factoring.correction_portfolio",
            "factoring.missing_portfolio",
            "factoring.table_portfolio",
        ),
        AssessmentSurface.CHECKIN: (
            "factoring.verification_portfolio",
            "factoring.transfer_portfolio",
            "factoring.table_portfolio",
            "factoring.missing_portfolio",
            "factoring.correction_portfolio",
        ),
        AssessmentSurface.GUIDED_WIDGET: ("factoring.guided_match",),
        AssessmentSurface.CAPSTONE: (
            "factoring.transfer_portfolio",
            "factoring.verification_portfolio",
        ),
        AssessmentSurface.WORKED_EXAMPLE: ("factoring.direct_portfolio",),
    },
    "kc.alg.solve_linear": {
        AssessmentSurface.DIAGNOSTIC: (
            "linear.two_sided",
            "linear.distributed",
            "linear.one_side",
            "linear.correction",
        ),
        AssessmentSurface.CHECKIN: (
            "linear.reversed",
            "linear.grouped",
            "linear.double_distributed",
            "linear.one_side",
            "linear.distributed",
        ),
        AssessmentSurface.GUIDED_WIDGET: ("linear.guided_balance",),
        AssessmentSurface.CAPSTONE: (
            "linear.double_distributed",
            "linear.correction",
        ),
        AssessmentSurface.WORKED_EXAMPLE: ("linear.two_sided",),
    },
    TARGET_KC: {
        AssessmentSurface.DIAGNOSTIC: (
            "quadratic.expanded",
            "quadratic.shifted",
            "quadratic.sparse_difference",
            "quadratic.correction",
        ),
        AssessmentSurface.CHECKIN: (
            "quadratic.reversed",
            "quadratic.factored",
            "quadratic.repeated",
            "quadratic.both_sides",
            "quadratic.expanded",
        ),
        AssessmentSurface.GUIDED_WIDGET: ("quadratic.guided_factor_map",),
        AssessmentSurface.CAPSTONE: (
            "quadratic.both_sides",
            "quadratic.correction",
        ),
        AssessmentSurface.WORKED_EXAMPLE: ("quadratic.expanded",),
    },
}

SEED_DIR = Path(__file__).resolve().parents[1] / "seed"
DEFAULT_SOURCE_PATH = SEED_DIR / "item_blueprints_solve_quadratics_v2.json"
DEFAULT_MANIFEST_PATH = SEED_DIR / "item_reviews_solve_quadratics_v2.json"
DEFAULT_BANK_PATH = SEED_DIR / "item_bank_solve_quadratics_v2.json"
DEFAULT_PEDAGOGY_SOURCE_PATH = (
    SEED_DIR / "pedagogy_pack_sources_solve_quadratics_v2.json"
)
DEFAULT_PEDAGOGY_MANIFEST_PATH = (
    SEED_DIR / "pedagogy_pack_reviews_solve_quadratics_v2.json"
)
DEFAULT_GRAPH_PATH = SEED_DIR / "kc_graph_calc1.json"


class SolveQuadraticsCompilationError(ValueError):
    """The pending wave cannot be compiled or qualified safely."""


@dataclass(frozen=True)
class InventorySeparationReport:
    """Auditable totals from exhaustive answer and visible-content checks."""

    answer_pairs_checked: int
    visible_candidate_comparisons_checked: int
    literal_visible_pairs_checked: int
    errors: tuple[str, ...]


@dataclass(frozen=True)
class GuidedPlan:
    """Semantic guided truth before opaque public identifiers are assigned."""

    rows: tuple[tuple[str, str], ...]
    options: tuple[tuple[str, str], ...]
    correct: tuple[tuple[str, str], ...]
    fallback_expected: tuple[str, ...]


@dataclass(frozen=True)
class DerivedTask:
    instruction: str
    givens: tuple[PromptSegment, ...]
    answer: AnswerSpec
    conceptual_hint: str
    operation_hint: str
    worked_steps: tuple[PromptSegment, ...]
    construct_coverage: frozenset[str]
    error_signatures: tuple[ErrorSignature, ...] = ()
    guided_plan: GuidedPlan | None = None
    task_kind: AssessmentTaskKind = AssessmentTaskKind.SOLVE

    @property
    def submission(self) -> str:
        return _canonical_submission(self.answer)


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
        ("{", " set containing "),
        ("}", " end set "),
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


def _format_fraction(value: Fraction) -> str:
    return (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )


def _canonical_submission(answer: AnswerSpec) -> str:
    if isinstance(answer, (SymbolicAnswerSpec, NumericAnswerSpec)):
        return answer.expected
    if isinstance(answer, OrderedTupleAnswerSpec):
        return "(" + ", ".join(answer.expected) + ")"
    if isinstance(answer, FiniteSetAnswerSpec):
        return "{" + ", ".join(answer.expected) + "}"
    raise TypeError(f"unsupported wave answer {type(answer).__name__}")


def _tuple_answer(
    values: tuple[str, ...],
    *,
    variables: list[str] | None = None,
) -> OrderedTupleAnswerSpec:
    return OrderedTupleAnswerSpec(expected=list(values), variables=variables or [])


def _coefficients(polynomial: CubicPolynomial) -> tuple[int, int, int, int]:
    return (
        polynomial.cubic,
        polynomial.quadratic,
        polynomial.linear,
        polynomial.constant,
    )


def _render_terms(terms: list[tuple[int, int]]) -> str:
    rendered: list[str] = []
    for coefficient, exponent in terms:
        if coefficient == 0:
            continue
        magnitude = abs(coefficient)
        if exponent == 0:
            body = str(magnitude)
        else:
            variable = "x" if exponent == 1 else f"x^{exponent}"
            body = variable if magnitude == 1 else f"{magnitude}*{variable}"
        if not rendered:
            rendered.append(body if coefficient > 0 else f"-{body}")
        else:
            rendered.append(("+" if coefficient > 0 else "-") + body)
    return "".join(rendered) or "0"


def _render_coefficients(coefficients: tuple[int, ...]) -> str:
    degree = len(coefficients) - 1
    return _render_terms(
        [
            (coefficient, degree - index)
            for index, coefficient in enumerate(coefficients)
        ]
    )


def _render_polynomial(polynomial: CubicPolynomial) -> str:
    terms = list(zip(_coefficients(polynomial), (3, 2, 1, 0), strict=True))
    if polynomial.display_order == "ascending":
        terms.reverse()
    elif polynomial.display_order == "interleaved":
        terms = [terms[index] for index in (1, 3, 0, 2)]
    return _render_terms(terms)


def _render_binomial(linear: int, constant: int) -> str:
    return _render_coefficients((linear, constant))


def _root_factor(root: int) -> str:
    """Render ``x-root`` without learner-visible double negatives."""

    return f"(x-{root})" if root >= 0 else f"(x+{abs(root)})"


def _factor_from_roots(lower: int, upper: int) -> str:
    return f"{_root_factor(lower)}*{_root_factor(upper)}"


def _poly_result(
    left: CubicPolynomial,
    right: CubicPolynomial,
    sign: int,
) -> tuple[int, int, int, int]:
    result = tuple(
        left_value + sign * right_value
        for left_value, right_value in zip(
            _coefficients(left),
            _coefficients(right),
            strict=True,
        )
    )
    return cast(tuple[int, int, int, int], result)


def _expand(product: BinomialProduct) -> tuple[int, int, int]:
    return (
        product.left_linear * product.right_linear,
        product.left_linear * product.right_constant
        + product.left_constant * product.right_linear,
        product.left_constant * product.right_constant,
    )


def _portfolio_values(core: PolynomialCore) -> tuple[str, str, str]:
    return (
        _render_coefficients(_poly_result(core.add_left, core.add_right, 1)),
        _render_coefficients(
            _poly_result(core.subtract_left, core.subtract_right, -1)
        ),
        _render_coefficients(_expand(core.expand)),
    )


def _portfolio_givens(core: PolynomialCore) -> tuple[MathPromptSegment, ...]:
    product = core.expand
    return (
        _math(
            f"A=({_render_polynomial(core.add_left)})+"
            f"({_render_polynomial(core.add_right)})"
        ),
        _math(
            f"B=({_render_polynomial(core.subtract_left)})-"
            f"({_render_polynomial(core.subtract_right)})"
        ),
        _math(
            f"C=({_render_binomial(product.left_linear, product.left_constant)})*"
            f"({_render_binomial(product.right_linear, product.right_constant)})"
        ),
    )


def _polynomial_wrong_values(core: PolynomialCore) -> tuple[tuple[str, ...], ...]:
    correct = _portfolio_values(core)
    add = _poly_result(core.add_left, core.add_right, 1)
    unlike = (add[0], add[1] + add[2], 0, add[3])
    subtract_as_add = _poly_result(core.subtract_left, core.subtract_right, 1)
    product = core.expand
    expanded = _expand(product)
    incomplete = (
        expanded[0],
        product.left_linear * product.right_constant,
        expanded[2],
    )
    candidates: list[tuple[str, ...]] = [
        (_render_coefficients(unlike), correct[1], correct[2]),
        (correct[0], _render_coefficients(subtract_as_add), correct[2]),
        (correct[0], correct[1], _render_coefficients(incomplete)),
    ]
    for index, candidate in enumerate(candidates):
        if candidate == correct:
            replacement = list(candidate)
            replacement[index] = f"({replacement[index]})+1"
            candidates[index] = tuple(replacement)
    return tuple(candidates)


def _polynomial_signatures(core: PolynomialCore) -> tuple[ErrorSignature, ...]:
    wrong = _polynomial_wrong_values(core)
    return (
        _signature(
            "(" + ", ".join(wrong[0]) + ")",
            "m.polynomial_ops.unlike_terms",
        ),
        _signature(
            "(" + ", ".join(wrong[1]) + ")",
            "m.polynomial_ops.subtraction_sign",
        ),
        _signature(
            "(" + ", ".join(wrong[2]) + ")",
            "m.polynomial_ops.incomplete_distribution",
        ),
    )


def _polynomial_worked_steps(core: PolynomialCore) -> tuple[PromptSegment, ...]:
    add, subtract, expanded = _portfolio_values(core)
    product = core.expand
    left_cross = product.left_linear * product.right_constant
    right_cross = product.left_constant * product.right_linear
    return (
        _step(f"A={add}"),
        _step(f"B={subtract}"),
        _step(
            f"C={product.left_linear * product.right_linear}*x^2"
            f"+({left_cross}+{right_cross})*x"
            f"+{product.left_constant * product.right_constant}"
        ),
        _step(f"C={expanded}"),
    )


def _normalized_symbolic(expression: sympy.Expr) -> str:
    return str(sympy.expand(expression)).replace("**", "^").replace(" ", "")


def _parse_x(expression: str) -> sympy.Expr:
    return parse_restricted(
        expression,
        allowed_variables={"x"},
        allowed_functions=set(),
        allowed_assignment_lhs=None,
    )


def _derive_polynomial(task: SolveQuadraticsMathTask) -> DerivedTask | None:
    polynomial_types = (
        PolynomialDirectPortfolioTask,
        PolynomialCorrectionPortfolioTask,
        PolynomialSparsePortfolioTask,
        PolynomialMixedExpressionTask,
        PolynomialCoefficientAuditTask,
        PolynomialReversePortfolioTask,
        PolynomialTablePortfolioTask,
        PolynomialGuidedMatchTask,
    )
    if not isinstance(task, polynomial_types):
        return None
    core = task.core
    add, subtract, expanded = _portfolio_values(core)
    common: dict[str, object] = {
        "conceptual_hint": (
            "Align like powers, distribute subtraction to every term, and form "
            "every binomial cross-product before collecting."
        ),
        "operation_hint": (
            "Complete A, B, and C independently, then place their results in the "
            "requested order."
        ),
        "worked_steps": _polynomial_worked_steps(core),
        "construct_coverage": CORE_POLYNOMIAL_COVERAGE,
        "task_kind": AssessmentTaskKind.SOLVE,
    }
    if isinstance(task, PolynomialCoefficientAuditTask):
        results = (
            _poly_result(core.add_left, core.add_right, 1),
            _poly_result(core.subtract_left, core.subtract_right, -1),
            _expand(core.expand),
        )
        expected = (
            str(results[0][3 - task.add_power]),
            str(results[1][3 - task.subtract_power]),
            str(results[2][2 - task.expand_power]),
        )
        return DerivedTask(
            instruction=(
                "Compute only the requested coefficients and enter "
                f"(x^{task.add_power} coefficient in A, x^{task.subtract_power} "
                f"coefficient in B, x^{task.expand_power} coefficient in C)."
            ),
            givens=_portfolio_givens(core),
            answer=_tuple_answer(expected),
            **common,
        )
    if isinstance(task, PolynomialReversePortfolioTask):
        product = core.expand
        expected = (
            _render_polynomial(core.add_right),
            _render_polynomial(core.subtract_right),
            str(product.right_constant),
        )
        return DerivedTask(
            instruction=(
                "Reverse the operations. Enter (P(x), Q(x), b), where P is the "
                "missing addend, Q is the missing subtrahend, and b completes the factor."
            ),
            givens=(
                _math(f"{_render_polynomial(core.add_left)}+P(x)={add}"),
                _math(f"{_render_polynomial(core.subtract_left)}-Q(x)={subtract}"),
                _math(
                    f"({_render_binomial(product.left_linear, product.left_constant)})*"
                    f"({product.right_linear}*x+b)={expanded}"
                ),
            ),
            answer=_tuple_answer(expected, variables=["x"]),
            **common,
        )
    if isinstance(task, PolynomialMixedExpressionTask):
        expected = _normalized_symbolic(
            _parse_x(add) - _parse_x(subtract) + _parse_x(expanded)
        )
        wrong = _polynomial_wrong_values(core)[0]
        wrong_expression = _normalized_symbolic(
            _parse_x(wrong[0]) - _parse_x(subtract) + _parse_x(expanded)
        )
        return DerivedTask(
            instruction=(
                "Simplify A-B+C as one polynomial. All three core operations "
                "contribute to the final result."
            ),
            givens=_portfolio_givens(core),
            answer=SymbolicAnswerSpec(expected=expected, variables=["x"]),
            error_signatures=(
                _signature(wrong_expression, "m.polynomial_ops.unlike_terms"),
            ),
            **common,
        )
    if isinstance(task, PolynomialTablePortfolioTask):
        table = TablePromptSegment(
            role=PromptSemanticRole.GIVEN,
            caption="Coefficients for addition task A",
            column_headers=("power", "first polynomial", "second polynomial"),
            rows=tuple(
                (label, str(left), str(right))
                for label, left, right in zip(
                    ("x^3", "x^2", "x", "constant"),
                    _coefficients(core.add_left),
                    _coefficients(core.add_right),
                    strict=True,
                )
            ),
            spoken_text="The table lists both coefficients for each power in A.",
        )
        return DerivedTask(
            instruction=(
                "Use the table for A and the expressions for B and C. Enter the "
                "complete tuple (A, B, C)."
            ),
            givens=(table, *_portfolio_givens(core)[1:]),
            answer=_tuple_answer((add, subtract, expanded), variables=["x"]),
            error_signatures=_polynomial_signatures(core),
            **common,
        )
    if isinstance(task, PolynomialCorrectionPortfolioTask):
        wrong = _polynomial_wrong_values(core)
        attempted = (wrong[0][0], wrong[1][1], wrong[2][2])
        return DerivedTask(
            instruction=(
                "Correct the three structural errors in the attempted tuple and "
                "return the complete results (A, B, C)."
            ),
            givens=(
                *_portfolio_givens(core),
                TextPromptSegment(
                    role=PromptSemanticRole.CONTEXT,
                    text="A learner supplied this attempted tuple:",
                ),
                _math("(" + ", ".join(attempted) + ")"),
            ),
            answer=_tuple_answer((add, subtract, expanded), variables=["x"]),
            error_signatures=_polynomial_signatures(core),
            **common,
        )
    if isinstance(task, PolynomialGuidedMatchTask):
        mixed = _normalized_symbolic(
            _parse_x(add) - _parse_x(subtract) + _parse_x(expanded)
        )
        rows = (
            ("sum", f"A: {_portfolio_givens(core)[0].expression}"),
            ("difference", f"B: {_portfolio_givens(core)[1].expression}"),
            ("product", f"C: {_portfolio_givens(core)[2].expression}"),
            ("combined", "D: simplify A-B+C"),
        )
        options = (
            ("sum_result", add),
            ("difference_result", subtract),
            ("product_result", expanded),
            ("combined_result", mixed),
        )
        correct = tuple(
            (row[0], option[0])
            for row, option in zip(rows, options, strict=True)
        )
        expected = (add, subtract, expanded, mixed)
        return DerivedTask(
            instruction=(
                "Match each exact task to its result. Text fallback: enter "
                "(A, B, C, D) in that named order."
            ),
            givens=_portfolio_givens(core),
            answer=_tuple_answer(expected, variables=["x"]),
            guided_plan=GuidedPlan(rows, options, correct, expected),
            **common,
        )
    prefix = (
        "The terms are intentionally sparse and reordered. "
        if isinstance(task, PolynomialSparsePortfolioTask)
        else ""
    )
    return DerivedTask(
        instruction=prefix + "Enter the ordered tuple of complete results (A, B, C).",
        givens=_portfolio_givens(core),
        answer=_tuple_answer((add, subtract, expanded), variables=["x"]),
        error_signatures=_polynomial_signatures(core),
        **common,
    )


def _render_gcf_expanded(spec: GcfFactorSpec) -> str:
    return _render_terms(
        [
            (
                spec.common_coefficient * spec.residual_linear,
                spec.common_exponent + 1,
            ),
            (
                spec.common_coefficient * spec.residual_constant,
                spec.common_exponent,
            ),
        ]
    )


def _render_monic(spec: MonicFactorSpec) -> str:
    return _render_coefficients(
        (
            1,
            -(spec.lower_root + spec.upper_root),
            spec.lower_root * spec.upper_root,
        )
    )


def _render_difference(core: FactoringCore) -> str:
    spec = core.difference
    return _render_coefficients(
        (spec.scale, 0, -spec.scale * spec.magnitude**2)
    )


def _factoring_values(core: FactoringCore) -> tuple[str, ...]:
    return (
        str(core.gcf.common_coefficient),
        str(core.gcf.common_exponent),
        str(core.gcf.residual_linear),
        str(core.gcf.residual_constant),
        str(core.monic.lower_root),
        str(core.monic.upper_root),
        str(core.difference.scale),
        str(core.difference.magnitude),
    )


def _factoring_givens(core: FactoringCore) -> tuple[MathPromptSegment, ...]:
    return (
        _math(f"G={_render_gcf_expanded(core.gcf)}"),
        _math(f"M={_render_monic(core.monic)}"),
        _math(f"D={_render_difference(core)}"),
    )


def _factoring_signatures(core: FactoringCore) -> tuple[ErrorSignature, ...]:
    expected = list(_factoring_values(core))
    skipped = expected.copy()
    skipped[:4] = [
        "1",
        "0",
        str(core.gcf.common_coefficient * core.gcf.residual_linear),
        str(core.gcf.common_coefficient * core.gcf.residual_constant),
    ]
    swapped = expected.copy()
    swapped[4:6] = [
        str(core.monic.lower_root + core.monic.upper_root),
        str(core.monic.lower_root * core.monic.upper_root),
    ]
    same_sign = expected.copy()
    same_sign[7] = str(-core.difference.magnitude)
    candidates = (skipped, swapped, same_sign)
    for candidate in candidates:
        if candidate == expected:
            candidate[-1] = str(int(candidate[-1]) + 1)
    return (
        _signature(
            "(" + ", ".join(skipped) + ")",
            "m.factoring.skips_gcf",
            "kc.alg.polynomial_ops",
        ),
        _signature(
            "(" + ", ".join(swapped) + ")",
            "m.factoring.sum_product_reversed",
        ),
        _signature(
            "(" + ", ".join(same_sign) + ")",
            "m.factoring.same_sign_difference",
        ),
    )


def _factoring_worked_steps(core: FactoringCore) -> tuple[PromptSegment, ...]:
    gcf = core.gcf
    monic = core.monic
    difference = core.difference
    return (
        _step(
            f"G={gcf.common_coefficient}*x^{gcf.common_exponent}*"
            f"({_render_binomial(gcf.residual_linear, gcf.residual_constant)})"
        ),
        _step(f"M={_factor_from_roots(monic.lower_root, monic.upper_root)}"),
        _step(
            f"D={difference.scale}*(x-{difference.magnitude})*"
            f"(x+{difference.magnitude})"
        ),
    )


def _derive_factoring(task: SolveQuadraticsMathTask) -> DerivedTask | None:
    factoring_types = (
        FactoringDirectPortfolioTask,
        FactoringCorrectionPortfolioTask,
        FactoringMissingPortfolioTask,
        FactoringTablePortfolioTask,
        FactoringVerificationPortfolioTask,
        FactoringTransferPortfolioTask,
        FactoringGuidedMatchTask,
    )
    if not isinstance(task, factoring_types):
        return None
    core = task.core
    values = _factoring_values(core)
    common: dict[str, object] = {
        "conceptual_hint": (
            "For G remove the signed common monomial; for M use the root sum "
            "and product; for D use conjugate factors."
        ),
        "operation_hint": (
            "Return (g, k, a, b, r, s, d, m), where G=g*x^k*(a*x+b), "
            "M=(x-r)*(x-s), r is no greater than s, and D=d*(x-m)*(x+m)."
        ),
        "worked_steps": _factoring_worked_steps(core),
        "construct_coverage": CORE_FACTORING_COVERAGE,
        "task_kind": AssessmentTaskKind.SOLVE,
    }
    if isinstance(task, FactoringTablePortfolioTask):
        table = TablePromptSegment(
            role=PromptSemanticRole.GIVEN,
            caption="Expanded coefficient forms for G, M, and D",
            column_headers=("task", "expanded polynomial"),
            rows=(
                ("G", _render_gcf_expanded(core.gcf)),
                ("M", _render_monic(core.monic)),
                ("D", _render_difference(core)),
            ),
            spoken_text="The table gives each exact expanded polynomial.",
        )
        return DerivedTask(
            instruction=(
                "Recover all three factorizations from the table and enter "
                "(g, k, a, b, r, s, d, m)."
            ),
            givens=(table,),
            answer=_tuple_answer(values),
            error_signatures=_factoring_signatures(core),
            **common,
        )
    if isinstance(task, FactoringCorrectionPortfolioTask):
        signatures = _factoring_signatures(core)
        attempted = (
            signatures[0].expected_wrong.strip("()")
            + ", "
            + signatures[1].expected_wrong.strip("()")
            + ", "
            + signatures[2].expected_wrong.strip("()")
        )
        return DerivedTask(
            instruction=(
                "Correct the GCF omission, sum-product error, and conjugate-sign "
                "error. Return one normalized eight-entry tuple."
            ),
            givens=(
                *_factoring_givens(core),
                TextPromptSegment(
                    role=PromptSemanticRole.CONTEXT,
                    text="The three attempted parameter tuples are:",
                ),
                _math(attempted),
            ),
            answer=_tuple_answer(values),
            error_signatures=signatures,
            **common,
        )
    if isinstance(task, FactoringMissingPortfolioTask):
        return DerivedTask(
            instruction=(
                "Complete the named parameters in all three forms, then enter "
                "(g, k, a, b, r, s, d, m)."
            ),
            givens=(
                *_factoring_givens(core),
                _math("G=g*x^k*(a*x+b)"),
                _math("M=(x-r)*(x-s)"),
                _math("D=d*(x-m)*(x+m)"),
                TextPromptSegment(
                    role=PromptSemanticRole.CONTEXT,
                    text=(
                        "Use the leading sign in g, keep a positive, order r before "
                        "s, and use a positive m."
                    ),
                ),
            ),
            answer=_tuple_answer(values),
            error_signatures=_factoring_signatures(core),
            **common,
        )
    if isinstance(task, FactoringVerificationPortfolioTask):
        x_value = task.check_value
        x = sympy.Symbol("x")
        checks = tuple(
            str(_parse_x(expression).subs(x, x_value))
            for expression in (
                _render_gcf_expanded(core.gcf),
                _render_monic(core.monic),
                _render_difference(core),
            )
        )
        return DerivedTask(
            instruction=(
                "Factor G, M, and D, then verify each reconstruction at "
                f"x={x_value}. Enter the eight factor parameters followed by the "
                "three check values."
            ),
            givens=_factoring_givens(core),
            answer=_tuple_answer((*values, *checks)),
            **common,
        )
    if isinstance(task, FactoringTransferPortfolioTask):
        return DerivedTask(
            instruction=(
                "The tasks include signed, mixed-sign, and repeated-factor cases. "
                "Normalize and factor all three, then enter the eight parameters."
            ),
            givens=tuple(reversed(_factoring_givens(core))),
            answer=_tuple_answer(values),
            error_signatures=_factoring_signatures(core),
            **common,
        )
    if isinstance(task, FactoringGuidedMatchTask):
        rows = (
            (
                "gcf_coefficient",
                f"Signed common coefficient in G={_render_gcf_expanded(core.gcf)}",
            ),
            ("gcf_exponent", "Lowest common exponent of x in G"),
            (
                "monic_root",
                f"Smaller root parameter in M={_render_monic(core.monic)}",
            ),
            (
                "difference_magnitude",
                f"Positive square magnitude in D={_render_difference(core)}",
            ),
        )
        options = (
            ("gcf_coefficient_result", values[0]),
            ("gcf_exponent_result", values[1]),
            ("monic_root_result", values[4]),
            ("difference_magnitude_result", values[7]),
        )
        correct = tuple(
            (row[0], option[0])
            for row, option in zip(rows, options, strict=True)
        )
        expected = (values[0], values[1], values[4], values[7])
        return DerivedTask(
            instruction=(
                "Match each exact factoring parameter to its value. Text fallback: "
                "enter (signed GCF coefficient, GCF exponent, smaller monic root, "
                "difference-square magnitude)."
            ),
            givens=_factoring_givens(core),
            answer=_tuple_answer(expected),
            guided_plan=GuidedPlan(rows, options, correct, expected),
            **common,
        )
    return DerivedTask(
        instruction=(
            "Factor all three polynomials and enter the normalized tuple "
            "(g, k, a, b, r, s, d, m)."
        ),
        givens=_factoring_givens(core),
        answer=_tuple_answer(values),
        error_signatures=_factoring_signatures(core),
        **common,
    )


def _linear_solution(equation: LinearEquationSpec) -> int:
    return (equation.right_constant - equation.left_constant) // (
        equation.left_coefficient - equation.right_coefficient
    )


def _render_affine(coefficient: int, constant: int) -> str:
    return _render_coefficients((coefficient, constant))


def _linear_signatures(
    equation: LinearEquationSpec,
) -> tuple[ErrorSignature, ...]:
    difference = equation.left_coefficient - equation.right_coefficient
    numerator = equation.right_constant - equation.left_constant
    correct = Fraction(numerator, difference)
    one_sided = Fraction(numerator, equation.left_coefficient or 1)
    sign_transfer = -correct
    partial_division = Fraction(numerator)
    candidates = [one_sided, sign_transfer, partial_division]
    used = {correct}
    for index, candidate in enumerate(candidates, start=1):
        while candidate in used:
            candidate += index
        candidates[index - 1] = candidate
        used.add(candidate)
    return (
        _signature(
            _format_fraction(candidates[0]),
            "m.solve_linear.one_sided_balance",
        ),
        _signature(
            _format_fraction(candidates[1]),
            "m.solve_linear.sign_transfer",
        ),
        _signature(
            _format_fraction(candidates[2]),
            "m.solve_linear.divides_one_term",
        ),
    )


def _linear_worked_steps(
    equation: LinearEquationSpec,
) -> tuple[PromptSegment, ...]:
    difference = equation.left_coefficient - equation.right_coefficient
    numerator = equation.right_constant - equation.left_constant
    return (
        _step(f"{difference}*x={numerator}"),
        _step(f"x={numerator}/{difference}"),
    )


def _linear_derived(
    *,
    instruction: str,
    given: str,
    equation: LinearEquationSpec,
    extra_givens: tuple[PromptSegment, ...] = (),
    guided_plan: GuidedPlan | None = None,
) -> DerivedTask:
    solution = _linear_solution(equation)
    answer: AnswerSpec
    if guided_plan is None:
        answer = NumericAnswerSpec(expected=str(solution), tolerance=0)
    else:
        answer = _tuple_answer(guided_plan.fallback_expected)
    return DerivedTask(
        instruction=instruction,
        givens=(_math(given), *extra_givens),
        answer=answer,
        conceptual_hint=(
            "Preserve equality while collecting variable terms on one side and "
            "constants on the other."
        ),
        operation_hint=(
            "Apply each inverse operation to both complete sides, then divide by "
            "the remaining coefficient of x."
        ),
        worked_steps=_linear_worked_steps(equation),
        construct_coverage=frozenset({"solve_integer_linear"}),
        error_signatures=() if guided_plan else _linear_signatures(equation),
        guided_plan=guided_plan,
    )


def _expanded_distributed(task: LinearDistributedTask) -> LinearEquationSpec:
    return LinearEquationSpec(
        left_coefficient=task.multiplier * task.inner_coefficient,
        left_constant=task.multiplier * task.inner_constant + task.added_constant,
        right_coefficient=task.right_coefficient,
        right_constant=task.right_constant,
    )


def _expanded_grouped(task: LinearGroupedTask) -> LinearEquationSpec:
    return LinearEquationSpec(
        left_coefficient=task.left_multiplier,
        left_constant=task.left_multiplier * task.left_shift,
        right_coefficient=task.right_multiplier,
        right_constant=task.right_multiplier * task.right_shift,
    )


def _expanded_double(task: LinearDoubleDistributedTask) -> LinearEquationSpec:
    return LinearEquationSpec(
        left_coefficient=task.left_multiplier * task.left_coefficient,
        left_constant=task.left_multiplier * task.left_constant,
        right_coefficient=task.right_multiplier * task.right_coefficient,
        right_constant=task.right_multiplier * task.right_constant,
    )


def _derive_linear(task: SolveQuadraticsMathTask) -> DerivedTask | None:
    if isinstance(
        task,
        (LinearTwoSidedTask, LinearReversedTask, LinearCorrectionTask),
    ):
        equation = task.equation
        left = _render_affine(equation.left_coefficient, equation.left_constant)
        right = _render_affine(equation.right_coefficient, equation.right_constant)
        if isinstance(task, LinearReversedTask):
            return _linear_derived(
                instruction=(
                    "Solve for x from the equation written with its sides reversed."
                ),
                given=f"{right}={left}",
                equation=equation,
            )
        if isinstance(task, LinearCorrectionTask):
            signature_by_mistake = {
                "one_sided": _linear_signatures(equation)[0],
                "sign_transfer": _linear_signatures(equation)[1],
                "partial_division": _linear_signatures(equation)[2],
            }
            attempted = signature_by_mistake[task.mistake].expected_wrong
            return _linear_derived(
                instruction=(
                    "Find and correct the balancing error, then solve the original "
                    "equation for x."
                ),
                given=f"{left}={right}",
                equation=equation,
                extra_givens=(
                    TextPromptSegment(
                        role=PromptSemanticRole.CONTEXT,
                        text=f"The attempted final value was x={attempted}.",
                    ),
                ),
            )
        return _linear_derived(
            instruction="Solve the two-sided linear equation for x.",
            given=f"{left}={right}",
            equation=equation,
        )
    if isinstance(task, LinearOneSideTask):
        solution = (task.target - task.constant) // task.coefficient
        equation = (
            LinearEquationSpec(
                left_coefficient=task.coefficient,
                left_constant=task.constant,
                right_coefficient=0,
                right_constant=task.target,
            )
            if task.variable_side == "left"
            else LinearEquationSpec(
                left_coefficient=0,
                left_constant=task.target,
                right_coefficient=task.coefficient,
                right_constant=task.constant,
            )
        )
        left = _render_affine(equation.left_coefficient, equation.left_constant)
        right = _render_affine(equation.right_coefficient, equation.right_constant)
        derived = _linear_derived(
            instruction=(
                "Solve the one-variable equation; the variable may appear on either side."
            ),
            given=f"{left}={right}",
            equation=equation,
        )
        if _linear_solution(equation) != solution:
            raise SolveQuadraticsCompilationError("one-sided solution derivation changed")
        return derived
    if isinstance(task, LinearDistributedTask):
        equation = _expanded_distributed(task)
        inner = _render_affine(task.inner_coefficient, task.inner_constant)
        left = f"{task.multiplier}*({inner})"
        if task.added_constant > 0:
            left += f"+{task.added_constant}"
        elif task.added_constant < 0:
            left += str(task.added_constant)
        right = _render_affine(task.right_coefficient, task.right_constant)
        return _linear_derived(
            instruction=(
                "Distribute first, then solve the resulting linear equation for x."
            ),
            given=f"{left}={right}",
            equation=equation,
        )
    if isinstance(task, LinearGroupedTask):
        equation = _expanded_grouped(task)
        return _linear_derived(
            instruction=(
                "Expand both grouped sides, collect terms, and solve for x."
            ),
            given=(
                f"{task.left_multiplier}*(x+{task.left_shift})="
                f"{task.right_multiplier}*(x+{task.right_shift})"
            ),
            equation=equation,
        )
    if isinstance(task, LinearDoubleDistributedTask):
        equation = _expanded_double(task)
        return _linear_derived(
            instruction=(
                "Distribute both sides, collect the signed terms, and solve for x."
            ),
            given=(
                f"{task.left_multiplier}*"
                f"({_render_affine(task.left_coefficient, task.left_constant)})="
                f"{task.right_multiplier}*"
                f"({_render_affine(task.right_coefficient, task.right_constant)})"
            ),
            equation=equation,
        )
    if isinstance(task, LinearGuidedBalanceTask):
        equation = task.equation
        solution = _linear_solution(equation)
        difference = equation.left_coefficient - equation.right_coefficient
        numerator = equation.right_constant - equation.left_constant
        check_value = equation.left_coefficient * solution + equation.left_constant
        values = (str(difference), str(numerator), str(solution), str(check_value))
        if len(set(values)) != len(values):
            raise SolveQuadraticsCompilationError(
                "guided linear task requires four distinct reachable values"
            )
        rows = (
            ("variable_coefficient", "Coefficient after collecting x terms"),
            ("constant_difference", "Constant after collecting constants"),
            ("solution", "Value of x after division"),
            ("check_value", "Value of either side after substitution"),
        )
        options = tuple(
            (f"result_{label}", value)
            for (label, _), value in zip(rows, values, strict=True)
        )
        correct = tuple(
            (row[0], option[0])
            for row, option in zip(rows, options, strict=True)
        )
        plan = GuidedPlan(rows, options, correct, values)
        left = _render_affine(equation.left_coefficient, equation.left_constant)
        right = _render_affine(equation.right_coefficient, equation.right_constant)
        return _linear_derived(
            instruction=(
                "Match the four stages to their exact values. Text fallback: enter "
                "(collected x coefficient, collected constant, x, substitution check)."
            ),
            given=f"{left}={right}",
            equation=equation,
            guided_plan=plan,
        )
    return None


def _root_values(roots: RootPair) -> tuple[str, ...]:
    if roots.lower == roots.upper:
        return (str(roots.lower),)
    return (str(roots.lower), str(roots.upper))


def _root_answer(roots: RootPair) -> FiniteSetAnswerSpec:
    return FiniteSetAnswerSpec(expected=list(_root_values(roots)))


def _quadratic_expression(roots: RootPair) -> str:
    return _render_coefficients(
        (1, -(roots.lower + roots.upper), roots.lower * roots.upper)
    )


def _quadratic_signatures(
    roots: RootPair,
    *,
    include_not_zero: bool,
) -> tuple[ErrorSignature, ...]:
    correct = tuple(int(value) for value in _root_values(roots))
    correct_set = frozenset(correct)
    one_root = (correct[0],) if len(correct) > 1 else (correct[0] + 1,)
    used = {correct_set, frozenset(one_root)}

    sign_reversed = tuple(sorted({-value for value in correct}))
    if frozenset(sign_reversed) in used:
        for delta in range(1, 10):
            sign_reversed = tuple(value + delta for value in correct)
            if frozenset(sign_reversed) not in used:
                break
    used.add(frozenset(sign_reversed))
    signatures = [
        _signature(
            "{" + ", ".join(str(value) for value in one_root) + "}",
            "m.solve_quadratic.one_root",
            "kc.alg.solve_linear",
        ),
        _signature(
            "{" + ", ".join(str(value) for value in sign_reversed) + "}",
            "m.solve_quadratic.factor_sign",
            "kc.alg.solve_linear",
        ),
    ]
    if include_not_zero:
        shifted: tuple[int, ...] | None = None
        for delta in range(2, 12):
            candidate = tuple(value + delta for value in correct)
            if frozenset(candidate) not in used:
                shifted = candidate
                break
        if shifted is None:  # pragma: no cover - bounded integer roots guarantee one
            raise SolveQuadraticsCompilationError(
                "could not derive a distinct quadratic error signature"
            )
        signatures.insert(
            0,
            _signature(
                "{" + ", ".join(str(value) for value in shifted) + "}",
                "m.solve_quadratic.not_zero",
                "kc.alg.factoring",
            ),
        )
    return tuple(signatures)


def _factor_text(roots: RootPair) -> str:
    return _factor_from_roots(roots.lower, roots.upper)


def _quadratic_worked_steps(roots: RootPair) -> tuple[PromptSegment, ...]:
    return (
        _step(f"p(x)={_factor_text(roots)}"),
        _step(
            f"{_render_affine(1, -roots.lower)}=0 and "
            f"{_render_affine(1, -roots.upper)}=0"
        ),
    )


def _quadratic_derived(
    *,
    instruction: str,
    givens: tuple[PromptSegment, ...],
    roots: RootPair,
    include_not_zero: bool = False,
    guided_plan: GuidedPlan | None = None,
) -> DerivedTask:
    answer: AnswerSpec = (
        _tuple_answer(guided_plan.fallback_expected)
        if guided_plan is not None
        else _root_answer(roots)
    )
    return DerivedTask(
        instruction=instruction,
        givens=givens,
        answer=answer,
        conceptual_hint=(
            "Rewrite the zero-value condition as a product of integer linear "
            "factors, then use the zero-product property."
        ),
        operation_hint=(
            "Set every distinct linear factor equal to zero, solve each linear "
            "equation, and report the complete set without duplicates."
        ),
        worked_steps=_quadratic_worked_steps(roots),
        construct_coverage=frozenset({"factorable_quadratic_roots"}),
        error_signatures=(
            ()
            if guided_plan is not None
            else _quadratic_signatures(roots, include_not_zero=include_not_zero)
        ),
        guided_plan=guided_plan,
    )


def _derive_quadratic(task: SolveQuadraticsMathTask) -> DerivedTask | None:
    if isinstance(task, QuadraticExpandedTask):
        return _quadratic_derived(
            instruction="Find every real zero of the expanded monic polynomial.",
            givens=(_math(f"p(x)={_quadratic_expression(task.roots)}"),),
            roots=task.roots,
        )
    if isinstance(task, QuadraticShiftedTask):
        base = _parse_x(_quadratic_expression(task.roots))
        left = _normalized_symbolic(base + task.right_constant)
        return _quadratic_derived(
            instruction=(
                "Move the nonzero right side first, then factor and solve for every root."
            ),
            givens=(_math(f"{left}={task.right_constant}"),),
            roots=task.roots,
            include_not_zero=True,
        )
    if isinstance(task, QuadraticSparseDifferenceTask):
        roots = RootPair(lower=-task.magnitude, upper=task.magnitude)
        return _quadratic_derived(
            instruction=(
                "Find the zeros of the sparse difference-of-squares polynomial."
            ),
            givens=(_math(f"p(x)=x^2-{task.magnitude**2}"),),
            roots=roots,
        )
    if isinstance(task, QuadraticFactoredTask):
        return _quadratic_derived(
            instruction=(
                "Use the displayed factorization to find the complete zero set."
            ),
            givens=(_math(f"p(x)={_factor_text(task.roots)}"),),
            roots=task.roots,
        )
    if isinstance(task, QuadraticReversedTask):
        return _quadratic_derived(
            instruction=(
                "The zero-value condition is written in reverse. Factor the "
                "polynomial and find every root."
            ),
            givens=(
                TextPromptSegment(
                    role=PromptSemanticRole.CONTEXT,
                    text="The polynomial on the right has value zero.",
                ),
                _math(_quadratic_expression(task.roots)),
            ),
            roots=task.roots,
        )
    if isinstance(task, QuadraticRepeatedTask):
        roots = RootPair(lower=task.root, upper=task.root)
        return _quadratic_derived(
            instruction=(
                "Find the finite zero set of the repeated linear factor. Report "
                "the repeated root only once."
            ),
            givens=(_math(f"p(x)=(x-{task.root})^2"),),
            roots=roots,
        )
    if isinstance(task, QuadraticBothSidesTask):
        target = _parse_x(_quadratic_expression(task.roots))
        right = _parse_x(
            _render_affine(task.right_linear, task.right_constant)
        )
        left = _normalized_symbolic(target + right)
        return _quadratic_derived(
            instruction=(
                "Collect the linear expression from the right, then factor the "
                "resulting monic quadratic and solve."
            ),
            givens=(
                _math(
                    f"{left}={_render_affine(task.right_linear, task.right_constant)}"
                ),
            ),
            roots=task.roots,
            include_not_zero=True,
        )
    if isinstance(task, QuadraticCorrectionTask):
        signatures = _quadratic_signatures(task.roots, include_not_zero=True)
        signature = {
            "not_zero": signatures[0],
            "one_root": signatures[1],
            "factor_sign": signatures[2],
        }[task.mistake]
        return _quadratic_derived(
            instruction=(
                "Correct the attempted root set, then report every root of the "
                "original polynomial."
            ),
            givens=(
                _math(f"p(x)={_quadratic_expression(task.roots)}"),
                TextPromptSegment(
                    role=PromptSemanticRole.CONTEXT,
                    text=f"The attempted root set was {signature.expected_wrong}.",
                ),
            ),
            roots=task.roots,
            include_not_zero=True,
        )
    if isinstance(task, QuadraticGuidedFactorMapTask):
        values = (
            str(-task.roots.lower),
            str(-task.roots.upper),
            str(task.roots.lower),
            str(task.roots.upper),
        )
        if len(set(values)) != len(values):
            raise SolveQuadraticsCompilationError(
                "guided quadratic task requires distinct factor constants and roots"
            )
        rows = (
            ("first_factor_constant", "Constant in the first factor x+c"),
            ("second_factor_constant", "Constant in the second factor x+c"),
            ("first_root", "Root produced by the first factor"),
            ("second_root", "Root produced by the second factor"),
        )
        options = tuple(
            (f"result_{row[0]}", value)
            for row, value in zip(rows, values, strict=True)
        )
        correct = tuple(
            (row[0], option[0])
            for row, option in zip(rows, options, strict=True)
        )
        plan = GuidedPlan(rows, options, correct, values)
        return _quadratic_derived(
            instruction=(
                "Match each factor or root role to its exact value. Text fallback: "
                "enter (first factor constant, second factor constant, first root, "
                "second root)."
            ),
            givens=(_math(f"p(x)={_quadratic_expression(task.roots)}"),),
            roots=task.roots,
            guided_plan=plan,
        )
    return None


def _compile_registered(task: BaseModel) -> DerivedTask:
    typed = cast(SolveQuadraticsMathTask, task)
    for compiler in (
        _derive_polynomial,
        _derive_factoring,
        _derive_linear,
        _derive_quadratic,
    ):
        derived = compiler(typed)
        if derived is not None:
            return derived
    raise SolveQuadraticsCompilationError(
        f"no deterministic compiler for {type(task).__name__}"
    )


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
        _registration(
            PolynomialDirectPortfolioTask,
            "polynomial.direct_portfolio",
            "kc.alg.polynomial_ops",
        ),
        _registration(
            PolynomialCorrectionPortfolioTask,
            "polynomial.correction_portfolio",
            "kc.alg.polynomial_ops",
        ),
        _registration(
            PolynomialSparsePortfolioTask,
            "polynomial.sparse_portfolio",
            "kc.alg.polynomial_ops",
        ),
        _registration(
            PolynomialMixedExpressionTask,
            "polynomial.mixed_expression",
            "kc.alg.polynomial_ops",
        ),
        _registration(
            PolynomialCoefficientAuditTask,
            "polynomial.coefficient_audit",
            "kc.alg.polynomial_ops",
        ),
        _registration(
            PolynomialReversePortfolioTask,
            "polynomial.reverse_portfolio",
            "kc.alg.polynomial_ops",
        ),
        _registration(
            PolynomialTablePortfolioTask,
            "polynomial.table_portfolio",
            "kc.alg.polynomial_ops",
        ),
        _registration(
            PolynomialGuidedMatchTask,
            "polynomial.guided_match",
            "kc.alg.polynomial_ops",
        ),
        _registration(
            FactoringDirectPortfolioTask,
            "factoring.direct_portfolio",
            "kc.alg.factoring",
        ),
        _registration(
            FactoringCorrectionPortfolioTask,
            "factoring.correction_portfolio",
            "kc.alg.factoring",
        ),
        _registration(
            FactoringMissingPortfolioTask,
            "factoring.missing_portfolio",
            "kc.alg.factoring",
        ),
        _registration(
            FactoringTablePortfolioTask,
            "factoring.table_portfolio",
            "kc.alg.factoring",
        ),
        _registration(
            FactoringVerificationPortfolioTask,
            "factoring.verification_portfolio",
            "kc.alg.factoring",
        ),
        _registration(
            FactoringTransferPortfolioTask,
            "factoring.transfer_portfolio",
            "kc.alg.factoring",
        ),
        _registration(
            FactoringGuidedMatchTask,
            "factoring.guided_match",
            "kc.alg.factoring",
        ),
        _registration(LinearTwoSidedTask, "linear.two_sided", "kc.alg.solve_linear"),
        _registration(LinearReversedTask, "linear.reversed", "kc.alg.solve_linear"),
        _registration(LinearCorrectionTask, "linear.correction", "kc.alg.solve_linear"),
        _registration(
            LinearGuidedBalanceTask,
            "linear.guided_balance",
            "kc.alg.solve_linear",
        ),
        _registration(LinearOneSideTask, "linear.one_side", "kc.alg.solve_linear"),
        _registration(
            LinearDistributedTask,
            "linear.distributed",
            "kc.alg.solve_linear",
        ),
        _registration(LinearGroupedTask, "linear.grouped", "kc.alg.solve_linear"),
        _registration(
            LinearDoubleDistributedTask,
            "linear.double_distributed",
            "kc.alg.solve_linear",
        ),
        _registration(
            QuadraticExpandedTask,
            "quadratic.expanded",
            TARGET_KC,
        ),
        _registration(QuadraticShiftedTask, "quadratic.shifted", TARGET_KC),
        _registration(QuadraticFactoredTask, "quadratic.factored", TARGET_KC),
        _registration(QuadraticReversedTask, "quadratic.reversed", TARGET_KC),
        _registration(
            QuadraticSparseDifferenceTask,
            "quadratic.sparse_difference",
            TARGET_KC,
        ),
        _registration(QuadraticRepeatedTask, "quadratic.repeated", TARGET_KC),
        _registration(QuadraticCorrectionTask, "quadratic.correction", TARGET_KC),
        _registration(QuadraticBothSidesTask, "quadratic.both_sides", TARGET_KC),
        _registration(
            QuadraticGuidedFactorMapTask,
            "quadratic.guided_factor_map",
            TARGET_KC,
        ),
    )
)


def derive_task(task: SolveQuadraticsMathTask) -> DerivedTask:
    """Compile one validated source task through the closed registry."""

    derived = _TASK_COMPILER_REGISTRY.compile(task)
    if not isinstance(derived, DerivedTask):
        raise SolveQuadraticsCompilationError(
            f"task compiler returned {type(derived).__name__}"
        )
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
    family: SolveQuadraticsFamilyBlueprint,
    plan: GuidedPlan,
) -> GuidedMappingSpec:
    """Create a public ordering whose ids and positions disclose no pairing."""

    row_labels = dict(plan.rows)
    option_labels = dict(plan.options)
    truth = dict(plan.correct)
    if set(truth) != set(row_labels) or set(truth.values()) != set(option_labels):
        raise SolveQuadraticsCompilationError(
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
        if all(truth[row] != option for row, option in zip(row_semantics, candidate))
        and any(
            truth[row] != option
            for row, option in zip(row_semantics, reversed(candidate))
        )
    ]
    if not permutations:
        raise SolveQuadraticsCompilationError(
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
    else:  # pragma: no cover - cryptographic ids make this unreachable in practice
        raise SolveQuadraticsCompilationError(
            f"{family.item_id}: could not hide guided truth from identifier order"
        )
    presentation = GuidedMappingPresentation(
        prompt=(
            "Match every row to the value that completes the exact task above. "
            "Keyboard selection and the text answer solve the same problem."
        ),
        rows=tuple(
            GuidedMappingEntry(
                entry_id=row_ids[semantic],
                label=row_labels[semantic],
                spoken_text=row_labels[semantic],
            )
            for semantic in row_semantics
        ),
        options=tuple(
            GuidedMappingEntry(
                entry_id=option_ids[semantic],
                label=option_labels[semantic],
                spoken_text=option_labels[semantic],
            )
            for semantic in option_semantics
        ),
    )
    return GuidedMappingSpec(
        presentation=presentation,
        scoring=GuidedMappingScoring(
            correct_pairs=tuple(
                (row_ids[row], option_ids[option]) for row, option in plan.correct
            )
        ),
    )


def _review_status_and_provenance(
    source: SolveQuadraticsBlueprintDocument,
    family: SolveQuadraticsFamilyBlueprint,
    review: ContentReviewEntry,
) -> tuple[ReviewStatus, AssessmentProvenance]:
    if review.decision == ReviewDecision.REJECTED:
        raise SolveQuadraticsCompilationError("rejected families cannot be compiled")
    approved = review.decision == ReviewDecision.APPROVED
    if approved:
        if review.reviewed_by is None or review.reviewed_at is None:
            raise SolveQuadraticsCompilationError("approved family lacks review provenance")
        if review.reviewed_by.strip().casefold() == source.author.strip().casefold():
            raise SolveQuadraticsCompilationError(
                "a family author cannot approve their own work"
            )
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
        verdict = verify_answer(
            item.answer,
            signature.expected_wrong,
            supervised=False,
        )
        if verdict.status != VerificationStatus.INCORRECT:
            raise SolveQuadraticsCompilationError(
                f"{item.item_id}: error signature is not an executable wrong "
                f"answer ({verdict.code})"
            )


def _build_item(
    source: SolveQuadraticsBlueprintDocument,
    family: SolveQuadraticsFamilyBlueprint,
    *,
    review_status: ReviewStatus,
    provenance: AssessmentProvenance,
) -> AssessmentItem:
    derived = derive_task(family.task)
    if family.surface == AssessmentSurface.WORKED_EXAMPLE:
        if len(derived.worked_steps) < 2:
            raise SolveQuadraticsCompilationError(
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
                text="Follow the intermediate steps before reading the result.",
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
    guided: GuidedInteractionSpec | None = None
    if family.surface == AssessmentSurface.GUIDED_WIDGET:
        if derived.guided_plan is None:
            raise SolveQuadraticsCompilationError(
                f"{family.item_id}: guided constructor lacks a mapping plan"
            )
        guided = _opaque_guided_mapping(family, derived.guided_plan)
    elif derived.guided_plan is not None:
        raise SolveQuadraticsCompilationError(
            f"{family.item_id}: non-guided family compiled guided truth"
        )
    item = AssessmentItem(
        item_id=family.item_id,
        revision=family.revision,
        family_id=family.family_id,
        kc_id=family.kc_id,
        difficulty=family.difficulty,
        task_kind=derived.task_kind,
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
        error_signatures=list(derived.error_signatures),
        guided_interaction=guided,
    )
    _validate_executable_signatures(item)
    return item


def _compiled_review_artifact(
    source: SolveQuadraticsBlueprintDocument,
    family: SolveQuadraticsFamilyBlueprint,
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
    source: SolveQuadraticsBlueprintDocument,
    family: SolveQuadraticsFamilyBlueprint,
) -> str:
    """Bind review to only this source family and its compiled artifact."""

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


def load_source(path: Path | None = None) -> SolveQuadraticsBlueprintDocument:
    source = path or DEFAULT_SOURCE_PATH
    return SolveQuadraticsBlueprintDocument.model_validate_json(
        source.read_text(encoding="utf-8")
    )


def load_manifest(path: Path | None = None) -> ContentReviewManifest:
    source = path or DEFAULT_MANIFEST_PATH
    return ContentReviewManifest.model_validate_json(
        source.read_text(encoding="utf-8")
    )


def draft_review_manifest(
    source: SolveQuadraticsBlueprintDocument,
) -> ContentReviewManifest:
    return ContentReviewManifest(
        manifest_version="solve-quadratics-review-manifest-v2",
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
        manifest_version="solve-quadratics-pedagogy-review-manifest-v2",
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


def _visible_fragments(item: AssessmentItem) -> list[str]:
    fragments = [render_prompt(item)]
    for segment in item.prompt:
        if isinstance(segment, TextPromptSegment):
            fragments.append(segment.text)
        elif isinstance(segment, MathPromptSegment):
            fragments.extend((segment.expression, segment.spoken_text or ""))
        elif isinstance(segment, TablePromptSegment):
            fragments.extend(
                (
                    segment.caption,
                    segment.spoken_text,
                    *segment.column_headers,
                    *(cell for row in segment.rows for cell in row),
                )
            )
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
    return list(dict.fromkeys(value for value in fragments if value))


def _candidate_texts_for(
    target: AssessmentItem,
    fragments: list[str],
) -> list[str]:
    candidates: list[str] = []
    for fragment in fragments:
        stripped = fragment.strip()
        marker = re.search(
            r"(?:answer|result|correct(?: completed)? form)\s*(?:is|=|:)?\s*(.+)$",
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


def _answers_are_equivalent(left: AssessmentItem, right: AssessmentItem) -> bool:
    scalar_types = (NumericAnswerSpec, SymbolicAnswerSpec)
    if isinstance(left.answer, scalar_types) and isinstance(right.answer, scalar_types):
        variables = set(getattr(left.answer, "variables", ())) | set(
            getattr(right.answer, "variables", ())
        )
        left_value = parse_restricted(
            _canonical_submission(left.answer),
            allowed_variables=variables,
            allowed_functions=set(),
            allowed_assignment_lhs=None,
        )
        right_value = parse_restricted(
            _canonical_submission(right.answer),
            allowed_variables=variables,
            allowed_functions=set(),
            allowed_assignment_lhs=None,
        )
        return sympy.expand(left_value - right_value) == 0
    if type(left.answer) is not type(right.answer):
        return False
    left_spec = left.answer
    right_spec = right.answer
    if isinstance(left_spec, (OrderedTupleAnswerSpec, FiniteSetAnswerSpec)) and isinstance(
        right_spec,
        type(left_spec),
    ):
        shared_variables = sorted(
            set(left_spec.variables) | set(right_spec.variables)
        )
        shared_functions = sorted(
            set(left_spec.functions) | set(right_spec.functions)
        )
        left_spec = left_spec.model_copy(
            update={
                "variables": shared_variables,
                "functions": shared_functions,
            }
        )
        right_spec = right_spec.model_copy(
            update={
                "variables": shared_variables,
                "functions": shared_functions,
            }
        )
    left_verdict = verify_answer(
        left_spec,
        _canonical_submission(right_spec),
        supervised=False,
    )
    right_verdict = verify_answer(
        right_spec,
        _canonical_submission(left_spec),
        supervised=False,
    )
    if any(
        verdict.status not in {VerificationStatus.CORRECT, VerificationStatus.INCORRECT}
        for verdict in (left_verdict, right_verdict)
    ):
        raise SolveQuadraticsCompilationError(
            f"answer comparison for {left.item_id}/{right.item_id} was indeterminate"
        )
    return VerificationStatus.CORRECT in {
        left_verdict.status,
        right_verdict.status,
    }


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
        raise SolveQuadraticsCompilationError("separation focus names an unknown item")

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
    candidate_comparisons = 0
    visible_pairs = 0
    fragments_by_id = {
        item.item_id: _visible_fragments(item) for item in ordered
    }
    for source_item in ordered:
        for target in ordered:
            if source_item.family_id == target.family_id:
                continue
            visible_pairs += 1
            if not in_scope(source_item, target):
                continue
            for candidate in _candidate_texts_for(
                target,
                fragments_by_id[source_item.item_id],
            ):
                candidate_comparisons += 1
                verdict = verify_answer(
                    target.answer,
                    candidate,
                    supervised=False,
                )
                if verdict.status == VerificationStatus.CORRECT:
                    errors.append(
                        f"{source_item.item_id}: visible content leaks the answer "
                        f"for {target.item_id}"
                    )
                    break
                if verdict.status not in {
                    VerificationStatus.INCORRECT,
                    VerificationStatus.INVALID,
                }:
                    errors.append(
                        f"{source_item.item_id}: visible comparison for "
                        f"{target.item_id} was indeterminate ({verdict.code})"
                    )
                    break
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
    """Reject pedagogy prose or math that discloses an item answer."""

    errors: list[str] = []
    for pack in pedagogy.pack_sources:
        fragments = [
            value
            for segment in (*pack.lesson_narrative, *pack.remediation)
            for value in (
                getattr(segment, "text", ""),
                getattr(segment, "expression", ""),
                getattr(segment, "spoken_text", ""),
            )
            if value
        ]
        fragments.extend(metaphor.description for metaphor in pack.metaphors)
        fragments.extend(
            misconception.remediation_hint for misconception in pack.misconceptions
        )
        for target in bank.items:
            for candidate in _candidate_texts_for(target, fragments):
                verdict = verify_answer(
                    target.answer,
                    candidate,
                    supervised=False,
                )
                if verdict.status == VerificationStatus.CORRECT:
                    errors.append(
                        f"{pack.source_id}: pedagogy leaks the answer for {target.item_id}"
                    )
                    break
    return tuple(dict.fromkeys(errors))


def _validate_taxonomy(source: SolveQuadraticsBlueprintDocument) -> None:
    for family in source.families:
        try:
            _TASK_COMPILER_REGISTRY.validate_taxonomy(
                family.task,
                construct_id=family.construct_id,
                kc_id=family.kc_id,
            )
        except TaskCompilerRegistryError as exc:
            raise SolveQuadraticsCompilationError(str(exc)) from exc
    for kc_id, by_surface in EXPECTED_CONSTRUCT_ORDER.items():
        for surface, constructs in by_surface.items():
            families = sorted(
                (
                    family
                    for family in source.families
                    if family.kc_id == kc_id and family.surface == surface
                ),
                key=lambda family: family.allocation_order,
            )
            actual = tuple(
                (family.allocation_order, family.construct_id) for family in families
            )
            expected = tuple(
                ((index + 1) * 10, construct)
                for index, construct in enumerate(constructs)
            )
            if actual != expected:
                raise SolveQuadraticsCompilationError(
                    f"{kc_id}/{surface.value}: construct/order mismatch; "
                    f"expected={expected}, got={actual}"
                )
            if surface in {AssessmentSurface.DIAGNOSTIC, AssessmentSurface.CHECKIN}:
                if len(constructs) != len(set(constructs)):
                    raise SolveQuadraticsCompilationError(
                        f"{kc_id}/{surface.value}: confirmation constructors repeat"
                    )
    for family in source.families:
        if family.kc_id not in {"kc.alg.polynomial_ops", "kc.alg.factoring"}:
            continue
        if family.surface not in {
            AssessmentSurface.DIAGNOSTIC,
            AssessmentSurface.CHECKIN,
            AssessmentSurface.CAPSTONE,
        }:
            continue
        expected = (
            CORE_POLYNOMIAL_COVERAGE
            if family.kc_id == "kc.alg.polynomial_ops"
            else CORE_FACTORING_COVERAGE
        )
        if derive_task(family.task).construct_coverage != expected:
            raise SolveQuadraticsCompilationError(
                f"{family.item_id}: mastery family does not cover the full construct"
            )


_MISCONCEPTION_IDS = {
    "kc.alg.polynomial_ops": frozenset(
        {
            "m.polynomial_ops.unlike_terms",
            "m.polynomial_ops.subtraction_sign",
            "m.polynomial_ops.incomplete_distribution",
        }
    ),
    "kc.alg.factoring": frozenset(
        {
            "m.factoring.skips_gcf",
            "m.factoring.sum_product_reversed",
            "m.factoring.same_sign_difference",
        }
    ),
    "kc.alg.solve_linear": frozenset(
        {
            "m.solve_linear.one_sided_balance",
            "m.solve_linear.sign_transfer",
            "m.solve_linear.divides_one_term",
        }
    ),
    TARGET_KC: frozenset(
        {
            "m.solve_quadratic.not_zero",
            "m.solve_quadratic.one_root",
            "m.solve_quadratic.factor_sign",
        }
    ),
}


def _validate_signature_taxonomy(items: list[AssessmentItem], graph: GraphDocument) -> None:
    hard_predecessors: dict[str, set[str]] = {
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
                raise SolveQuadraticsCompilationError(
                    f"{item.item_id}: unreviewed misconception id"
                )
            if (
                signature.implicated_prereq is not None
                and signature.implicated_prereq not in hard_predecessors[item.kc_id]
            ):
                raise SolveQuadraticsCompilationError(
                    f"{item.item_id}: implicated prerequisite is not direct and hard"
                )


def compile_release_inventory(
    source: SolveQuadraticsBlueprintDocument,
    manifest: ContentReviewManifest,
    graph: GraphDocument,
) -> tuple[ItemBankDocument, InventorySeparationReport]:
    """Compile and qualify all 52 pending Solve Quadratics families."""

    if source.graph_version != graph.graph_version:
        raise SolveQuadraticsCompilationError("source and graph versions differ")
    if manifest.graph_version != graph.graph_version:
        raise SolveQuadraticsCompilationError("manifest and graph versions differ")
    if manifest.compiler_version != COMPILER_VERSION:
        raise SolveQuadraticsCompilationError("manifest compiler pin is stale")
    if set(source.target_kcs) != set(TARGET_KCS):
        raise SolveQuadraticsCompilationError(
            "source must contain exactly the four Solve Quadratics KCs"
        )
    closure = ancestor_subgraph(graph, TARGET_KC, hard_only=True).node_ids()
    if closure != set(EXPECTED_CLOSURE):
        raise SolveQuadraticsCompilationError(
            f"solve-quadratics hard closure changed: {sorted(closure)}"
        )
    identities = {
        (family.blueprint_id, family.revision) for family in source.families
    }
    reviews = {
        (entry.blueprint_id, entry.revision): entry for entry in manifest.entries
    }
    if set(reviews) != identities:
        raise SolveQuadraticsCompilationError("review/source identity coverage differs")
    expected_matrix = {
        (kc_id, surface): count
        for kc_id in TARGET_KCS
        for surface, count in EXPECTED_FAMILY_COUNTS.items()
    }
    if dict(Counter((family.kc_id, family.surface) for family in source.families)) != (
        expected_matrix
    ):
        raise SolveQuadraticsCompilationError(
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
            raise SolveQuadraticsCompilationError(
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
        verdict = verify_answer(
            item.answer,
            _canonical_submission(item.answer),
            supervised=False,
        )
        if verdict.status != VerificationStatus.CORRECT:
            raise SolveQuadraticsCompilationError(
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
        raise SolveQuadraticsCompilationError(
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
        raise SolveQuadraticsCompilationError(
            "inventory separation failed: " + "; ".join(report.errors)
        )
    return bank, report


def _refuse_completed_review_overwrite(
    item_manifest: ContentReviewManifest | None,
    pedagogy_manifest: PedagogyReviewManifest | None,
) -> None:
    if item_manifest is not None and any(
        entry.decision != ReviewDecision.PENDING for entry in item_manifest.entries
    ):
        raise SolveQuadraticsCompilationError(
            "refusing to overwrite a completed item-family review"
        )
    if pedagogy_manifest is not None and any(
        entry.decision != PedagogyReviewDecision.PENDING
        for entry in pedagogy_manifest.entries
    ):
        raise SolveQuadraticsCompilationError(
            "refusing to overwrite a completed pedagogy review"
        )


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
    """Regenerate only pending manifests and compiled draft output."""

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
        raise SolveQuadraticsCompilationError(
            "pedagogy content leaks a draft assessment answer"
        )
    _atomic_write_model(item_manifest_path, item_manifest)
    _atomic_write_model(pedagogy_manifest_path, pedagogy_manifest)
    _atomic_write_model(bank_path, bank)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH_PATH)
    parser.add_argument("--pedagogy-source", type=Path, default=DEFAULT_PEDAGOGY_SOURCE_PATH)
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
            raise SolveQuadraticsCompilationError("; ".join(pedagogy_errors))
    except Exception as exc:  # noqa: BLE001 - fail closed at CLI boundary
        print(f"Solve Quadratics inventory INVALID: {exc}", file=sys.stderr)
        return 1
    if args.out is not None and not args.regenerate:
        _atomic_write_model(args.out, bank)
    status_counts = Counter(item.review_status for item in bank.items)
    print(
        "Solve Quadratics inventory OK: "
        f"{len(bank.items)} families, "
        f"{status_counts[ReviewStatus.DRAFT]} draft, "
        f"{status_counts[ReviewStatus.HUMAN_APPROVED]} approved, "
        f"{report.answer_pairs_checked} answer comparisons, "
        f"{report.literal_visible_pairs_checked} directed visible checks, "
        f"{report.visible_candidate_comparisons_checked} candidate comparisons, "
        f"released KCs={len(bank.released_kcs)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
