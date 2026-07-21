"""Compile and qualify the pending Product/Quotient Rules content release.

This compiler is intentionally isolated from the active item bank.  Its
packaged source contains 52 independently reviewable family blueprints for the
four-KC hard-prerequisite closure, while ``released_kcs`` remains empty until
the exact source digests receive independent human approval.
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
from fractions import Fraction
from multiprocessing.connection import Connection
from pathlib import Path
from tempfile import NamedTemporaryFile

import sympy

from tutor.content.item_bank import (
    _candidate_answer_texts,
    _candidate_fits_answer_contract,
    render_prompt,
)
from tutor.graph.service import ancestor_subgraph
from tutor.schemas.assessment import (
    AssessmentHint,
    AssessmentItem,
    AssessmentProvenance,
    AssessmentSurface,
    AssessmentTaskKind,
    BlankPromptSegment,
    ItemBankDocument,
    MathPromptSegment,
    NumericAnswerSpec,
    PromptSemanticRole,
    SymbolicAnswerSpec,
    TextPromptSegment,
)
from tutor.schemas.common import ReviewStatus
from tutor.schemas.content_authoring import (
    ContentReviewEntry,
    ContentReviewManifest,
    ReviewDecision,
)
from tutor.schemas.kc import GraphDocument
from tutor.schemas.product_quotient_authoring import (
    ExponentCompoundTask,
    ExponentNegativeTask,
    ExponentPowerTask,
    ExponentProductTask,
    ExponentQuotientTask,
    ExponentZeroTask,
    PolynomialDerivativeTask,
    PolynomialSpec,
    PowerDerivativeTask,
    ProductAtPointTask,
    ProductQuotientBlueprintDocument,
    QuotientAtPointTask,
    RadicalPowerDerivativeTask,
    RationalPowerDerivativeTask,
    ReleaseFamilyBlueprint,
    ReleaseMathTask,
)
from tutor.verify.checker import VerificationStatus, parse_restricted, verify_answer

COMPILER_VERSION = "product-quotient-item-compiler-v2"
TARGET_KC = "kc.der.product_quotient"
TARGET_CLOSURE = frozenset(
    {
        "kc.alg.exponent_rules",
        "kc.der.power_rule",
        "kc.der.sum_constant_rules",
        TARGET_KC,
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
    "kc.alg.exponent_rules": {
        AssessmentSurface.DIAGNOSTIC: (
            "exponent.product",
            "exponent.quotient",
            "exponent.power",
            "exponent.negative",
        ),
        AssessmentSurface.CHECKIN: (
            "exponent.compound",
            "exponent.quotient",
            "exponent.power",
            "exponent.negative",
            "exponent.zero",
        ),
        AssessmentSurface.GUIDED_WIDGET: ("exponent.compound",),
        AssessmentSurface.CAPSTONE: ("exponent.compound", "exponent.compound"),
        AssessmentSurface.WORKED_EXAMPLE: ("exponent.compound",),
    },
    "kc.der.power_rule": {
        # Diagnosis may stop after two successes. The first two families
        # deliberately span rational/radical and signed-exponent scope.
        AssessmentSurface.DIAGNOSTIC: (
            "power.rational",
            "power.reciprocal_radical",
            "power.negative_integer",
            "power.positive_integer",
        ),
        AssessmentSurface.CHECKIN: (
            "power.rational",
            "power.reciprocal_radical",
            "power.positive_integer",
            "power.sqrt",
            "power.negative_integer",
        ),
        AssessmentSurface.GUIDED_WIDGET: ("power.sqrt",),
        AssessmentSurface.CAPSTONE: (
            "power.rational",
            "power.reciprocal_radical",
        ),
        AssessmentSurface.WORKED_EXAMPLE: ("power.sqrt",),
    },
    "kc.der.sum_constant_rules": {
        surface: tuple("sum.polynomial_termwise" for _ in range(count))
        for surface, count in EXPECTED_FAMILY_COUNTS.items()
    },
    TARGET_KC: {
        AssessmentSurface.DIAGNOSTIC: (
            "product_quotient.product_at_point",
            "product_quotient.quotient_at_point",
            "product_quotient.product_at_point",
            "product_quotient.quotient_at_point",
        ),
        AssessmentSurface.CHECKIN: (
            "product_quotient.product_at_point",
            "product_quotient.quotient_at_point",
            "product_quotient.product_at_point",
            "product_quotient.quotient_at_point",
            "product_quotient.product_at_point",
        ),
        AssessmentSurface.GUIDED_WIDGET: (
            "product_quotient.product_at_point",
        ),
        AssessmentSurface.CAPSTONE: (
            "product_quotient.quotient_at_point",
            "product_quotient.product_at_point",
        ),
        AssessmentSurface.WORKED_EXAMPLE: (
            "product_quotient.quotient_at_point",
        ),
    },
}

SEED_DIR = Path(__file__).resolve().parents[1] / "seed"
DEFAULT_SOURCE_PATH = SEED_DIR / "item_blueprints_product_quotient_v1.json"
DEFAULT_MANIFEST_PATH = SEED_DIR / "item_reviews_product_quotient_v1.json"
DEFAULT_GRAPH_PATH = SEED_DIR / "kc_graph_calc1.json"


class ProductQuotientCompilationError(ValueError):
    """The pending release cannot be compiled or qualified safely."""


@dataclass(frozen=True)
class InventorySeparationReport:
    """Auditable totals and failures from exhaustive family separation."""

    answer_pairs_checked: int
    visible_candidate_comparisons_checked: int
    literal_visible_pairs_checked: int
    errors: tuple[str, ...]


@dataclass(frozen=True)
class _SeparationMathResult:
    answer_reuse: tuple[tuple[str, str], ...]
    visible_leakage: tuple[tuple[str, str], ...]
    answer_comparisons: int
    visible_candidate_comparisons: int


def _canonical_release_expression(expression: str) -> sympy.Expr:
    parsed = parse_restricted(
        expression,
        allowed_variables={"x", "z"},
        allowed_functions={"sqrt"},
        allowed_assignment_lhs=None,
    )
    if parsed.has(sympy.zoo, sympy.nan, sympy.oo) or parsed.is_finite is False:
        raise ValueError("release expression is not finite")
    # ``cancel`` gives the pilot's polynomial/rational task language one
    # deterministic normal form.  It is run only inside the supervised batch.
    return sympy.cancel(parsed)


def _release_expressions_equivalent(left: sympy.Expr, right: sympy.Expr) -> bool:
    """Compare exact scalar expressions inside the supervised worker."""
    return sympy.cancel(left - right) == 0


def _separation_math_worker(
    connection: Connection,
    expected_payload: list[tuple[str, str]],
    visible_payload: list[tuple[str, str, str]],
) -> None:
    """Canonicalize once, compare exhaustively, and return only stable ids."""
    try:
        expected = {
            item_id: _canonical_release_expression(expected_text)
            for item_id, expected_text in expected_payload
        }
        ordered_ids = sorted(expected)
        answer_reuse: list[tuple[str, str]] = []
        answer_comparisons = 0
        for index, left_id in enumerate(ordered_ids):
            for right_id in ordered_ids[index + 1 :]:
                answer_comparisons += 1
                if _release_expressions_equivalent(
                    expected[left_id], expected[right_id]
                ):
                    answer_reuse.append((left_id, right_id))
        visible_leakage: list[tuple[str, str]] = []
        for source_id, target_id, candidate in visible_payload:
            parsed_candidate = _canonical_release_expression(candidate)
            if _release_expressions_equivalent(
                parsed_candidate, expected[target_id]
            ):
                visible_leakage.append((source_id, target_id))
        connection.send(
            {
                "answer_reuse": answer_reuse,
                "visible_leakage": visible_leakage,
                "answer_comparisons": answer_comparisons,
                "visible_candidate_comparisons": len(visible_payload),
            }
        )
    except BaseException as exc:  # noqa: BLE001 - worker must fail closed
        connection.send(
            {
                "error": type(exc).__name__,
            }
        )
    finally:
        connection.close()


def _reap_worker(process: multiprocessing.Process) -> None:
    """Join normally, then terminate and finally kill a stuck batch worker."""
    process.join(timeout=1.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
    if process.is_alive():
        process.kill()
        process.join(timeout=1.0)
    if process.is_alive():
        raise ProductQuotientCompilationError(
            "inventory separation worker could not be stopped"
        )


def _supervised_math_separation(
    items: list[AssessmentItem],
    *,
    timeout_seconds: float = 15.0,
) -> _SeparationMathResult:
    """Run all pairwise symbolic work in one replaceable worker process."""
    expected_payload: list[tuple[str, str]] = []
    candidates_by_source: dict[str, list[str]] = {}
    for item in items:
        if not isinstance(item.answer, (SymbolicAnswerSpec, NumericAnswerSpec)):
            raise ProductQuotientCompilationError(
                "inventory separation requires scalar symbolic or numeric answers"
            )
        expected_payload.append((item.item_id, item.answer.expected))
        candidates_by_source[item.item_id] = list(
            dict.fromkeys(
                [
                    segment.expression
                    for segment in item.prompt
                    if isinstance(segment, MathPromptSegment)
                ]
                + [
                    candidate
                    for visible in _visible_content(item)
                    for candidate in _candidate_answer_texts(visible)
                ]
            )
        )

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
        target=_separation_math_worker,
        args=(child, expected_payload, visible_payload),
        daemon=True,
    )
    process.start()
    child.close()
    try:
        if not parent.poll(timeout_seconds):
            raise ProductQuotientCompilationError(
                "inventory separation worker timed out"
            )
        result = parent.recv()
    except EOFError as exc:
        raise ProductQuotientCompilationError(
            "inventory separation worker exited without a result"
        ) from exc
    finally:
        parent.close()
        _reap_worker(process)
    if "error" in result:
        raise ProductQuotientCompilationError(
            f"inventory separation worker failed ({result['error']})"
        )
    return _SeparationMathResult(
        answer_reuse=tuple(tuple(pair) for pair in result["answer_reuse"]),
        visible_leakage=tuple(tuple(pair) for pair in result["visible_leakage"]),
        answer_comparisons=result["answer_comparisons"],
        visible_candidate_comparisons=result["visible_candidate_comparisons"],
    )


def load_source(path: Path | None = None) -> ProductQuotientBlueprintDocument:
    """Load the packaged typed source document or an explicit replacement."""
    source = path or DEFAULT_SOURCE_PATH
    return ProductQuotientBlueprintDocument.model_validate_json(
        source.read_text(encoding="utf-8")
    )


def load_manifest(path: Path | None = None) -> ContentReviewManifest:
    """Load exact review decisions for the typed family source."""
    source = path or DEFAULT_MANIFEST_PATH
    return ContentReviewManifest.model_validate_json(source.read_text(encoding="utf-8"))


def family_digest(
    source: ProductQuotientBlueprintDocument,
    family: ReleaseFamilyBlueprint,
) -> str:
    """Bind review to family math plus the declared source and authorship."""
    canonical = json.dumps(
        {
            "author": source.author,
            "authoring_source": source.authoring_source,
            "blueprint_version": source.blueprint_version,
            "compiler_version": COMPILER_VERSION,
            "compiled_artifact": _compiled_review_artifact(source, family),
            "family": family.model_dump(mode="json"),
            "graph_version": source.graph_version,
            "output_bank_version": source.output_bank_version,
            "released_kcs": sorted(source.released_kcs),
            "schema_version": source.schema_version,
            "target_kcs": sorted(source.target_kcs),
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _render_monomial(coefficient: int, exponent: int, variable: str = "x") -> str:
    if exponent == 0:
        return str(coefficient)
    if exponent == 1:
        variable_factor = variable
    else:
        variable_factor = f"{variable}^{exponent}"
    if coefficient == 1:
        return variable_factor
    if coefficient == -1:
        return f"-{variable_factor}"
    return f"{coefficient}*{variable_factor}"


def _render_terms(terms: list[tuple[int, int]]) -> str:
    rendered: list[str] = []
    for index, (coefficient, exponent) in enumerate(terms):
        monomial = _render_monomial(abs(coefficient), exponent)
        if index == 0:
            rendered.append(monomial if coefficient > 0 else f"-{monomial}")
        else:
            operator = "+" if coefficient > 0 else "-"
            rendered.append(f"{operator}{monomial}")
    return "".join(rendered)


def _render_polynomial(polynomial: PolynomialSpec) -> str:
    return _render_terms(
        [(term.coefficient, term.exponent) for term in polynomial.terms]
    )


def _differentiate_polynomial(polynomial: PolynomialSpec) -> str:
    derivative_terms = [
        (term.coefficient * term.exponent, term.exponent - 1)
        for term in polynomial.terms
        if term.exponent > 0
    ]
    if not derivative_terms:
        return "0"
    return _render_terms(derivative_terms)


def _render_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _render_rational_monomial(coefficient: Fraction, exponent: Fraction) -> str:
    if exponent == 0:
        return _render_fraction(coefficient)
    exponent_text = _render_fraction(exponent)
    variable_factor = "x" if exponent == 1 else f"x^({exponent_text})"
    if coefficient == 1:
        return variable_factor
    if coefficient == -1:
        return f"-{variable_factor}"
    coefficient_text = _render_fraction(coefficient)
    if coefficient.denominator != 1:
        coefficient_text = f"({coefficient_text})"
    return f"{coefficient_text}*{variable_factor}"


def _render_radical_reciprocal(
    coefficient: Fraction,
    *,
    extra_x: bool,
) -> str:
    denominator_factor = "x*sqrt(x)" if extra_x else "sqrt(x)"
    magnitude = abs(coefficient)
    sign = "-" if coefficient < 0 else ""
    if magnitude.denominator == 1:
        numerator_text = str(magnitude.numerator)
        denominator_text = denominator_factor
    else:
        numerator_text = str(magnitude.numerator)
        denominator_text = f"{magnitude.denominator}*{denominator_factor}"
    return f"{sign}{numerator_text}/({denominator_text})"


@dataclass(frozen=True)
class _DerivedTask:
    instruction: str
    givens: tuple[str, ...]
    expected: str
    conceptual_hint: str
    operation_hint: str
    numeric_answer: bool = False
    functions: tuple[str, ...] = ()
    variables: tuple[str, ...] = ("x",)


def _derive_task(task: ReleaseMathTask) -> _DerivedTask:
    """Derive visible structure and truth from the closed typed task union."""
    if isinstance(task, ExponentProductTask):
        given = f"{task.base}^{task.left_exponent}*{task.base}^{task.right_exponent}"
        expected = f"{task.base}^{task.left_exponent + task.right_exponent}"
        return _DerivedTask(
            instruction="Simplify the expression using the product rule for equal bases.",
            givens=(given,),
            expected=expected,
            conceptual_hint="Keep the common base and combine the exponents.",
            operation_hint="A product of equal bases calls for adding their exponents.",
        )
    if isinstance(task, ExponentQuotientTask):
        given = (
            f"{task.base}^{task.numerator_exponent}/"
            f"{task.base}^{task.denominator_exponent}"
        )
        result = task.numerator_exponent - task.denominator_exponent
        expected = _render_monomial(1, result, task.base)
        return _DerivedTask(
            instruction=(
                "Simplify the expression using the quotient rule for equal bases. "
                f"Assume {task.base} is non-zero."
            ),
            givens=(given,),
            expected=expected,
            conceptual_hint="Keep the common base and combine the exponents.",
            operation_hint=(
                "A quotient of equal bases calls for subtracting the denominator exponent."
            ),
        )
    if isinstance(task, ExponentPowerTask):
        given = f"({task.base}^{task.inner_exponent})^{task.outer_exponent}"
        expected = f"{task.base}^{task.inner_exponent * task.outer_exponent}"
        return _DerivedTask(
            instruction="Simplify the power of a power.",
            givens=(given,),
            expected=expected,
            conceptual_hint="Keep the base while combining the nested exponents.",
            operation_hint="A power raised to a power calls for multiplying the exponents.",
        )
    if isinstance(task, ExponentNegativeTask):
        return _DerivedTask(
            instruction=(
                "Rewrite the expression using only positive exponents. "
                f"Assume {task.base} is non-zero."
            ),
            givens=(f"{task.base}^-{task.magnitude}",),
            expected=f"1/{task.base}^{task.magnitude}",
            conceptual_hint=(
                "A negative exponent moves the powered factor across the fraction bar."
            ),
            operation_hint=(
                "Write the powered base in the denominator and make its exponent positive."
            ),
        )
    if isinstance(task, ExponentZeroTask):
        return _DerivedTask(
            instruction=(
                "Simplify using the zero-exponent rule. "
                f"Assume {task.base} is non-zero."
            ),
            givens=(f"{task.base}^0",),
            expected="1",
            conceptual_hint=(
                "Use the multiplicative-identity result of the zero-exponent rule."
            ),
            operation_hint=(
                "The ratio interpretation of a zero exponent has equal numerator and denominator."
            ),
        )
    if isinstance(task, ExponentCompoundTask):
        given = (
            f"({task.base}^0*{task.base}^-{task.negative_magnitude})*"
            f"({task.base}^{task.inner_exponent})^{task.outer_exponent}*"
            f"{task.base}^{task.product_exponent}/{task.base}^{task.denominator_exponent}"
        )
        result = (
            -task.negative_magnitude
            + task.inner_exponent * task.outer_exponent
            + task.product_exponent
            - task.denominator_exponent
        )
        expected = _render_monomial(1, result, task.base)
        return _DerivedTask(
            instruction=(
                f"Simplify the compound expression to one power of {task.base}. "
                f"Assume {task.base} is non-zero."
            ),
            givens=(given,),
            expected=expected,
            conceptual_hint=(
                "Account for the zero and negative exponents before combining "
                f"every {task.base} power."
            ),
            operation_hint=(
                "Multiply nested exponents, then add numerator exponents and subtract "
                "denominator exponents."
            ),
        )
    if isinstance(task, PowerDerivativeTask):
        given = _render_monomial(task.coefficient, task.exponent)
        expected = _render_monomial(
            task.coefficient * task.exponent,
            task.exponent - 1,
        )
        return _DerivedTask(
            instruction="Differentiate with respect to x using the power rule.",
            givens=(given,),
            expected=expected,
            conceptual_hint=(
                "Keep the constant multiple and apply the power rule to the variable factor."
            ),
            operation_hint=(
                "Multiply by the old exponent, then reduce that exponent by one."
            ),
        )
    if isinstance(task, RationalPowerDerivativeTask):
        exponent = Fraction(task.numerator, task.denominator)
        given = _render_rational_monomial(Fraction(task.coefficient), exponent)
        expected = _render_rational_monomial(
            Fraction(task.coefficient) * exponent,
            exponent - 1,
        )
        return _DerivedTask(
            instruction="Differentiate with respect to x using the power rule.",
            givens=(given,),
            expected=expected,
            conceptual_hint=(
                "The power rule applies to reduced rational exponents as well as integers."
            ),
            operation_hint=(
                "Multiply by the rational exponent, then subtract one from that exponent."
            ),
        )
    if isinstance(task, RadicalPowerDerivativeTask):
        magnitude = abs(task.coefficient)
        sign = "-" if task.coefficient < 0 else ""
        coefficient_prefix = "" if magnitude == 1 else f"{magnitude}*"
        radical = f"{sign}{coefficient_prefix}sqrt(x)"
        if task.form == "sqrt":
            given = radical
            expected = _render_radical_reciprocal(
                Fraction(task.coefficient, 2),
                extra_x=False,
            )
            conceptual = "Treat the square root as the one-half power of x."
        else:
            given = f"{sign}{magnitude}/sqrt(x)"
            expected = _render_radical_reciprocal(
                Fraction(-task.coefficient, 2),
                extra_x=True,
            )
            conceptual = "A reciprocal square root is the negative one-half power of x."
        return _DerivedTask(
            instruction="Differentiate with respect to x using the power rule.",
            givens=(given,),
            expected=expected,
            conceptual_hint=conceptual,
            operation_hint=(
                "Rewrite the radical as a one-half power, apply the power rule, "
                "then return to radical form."
            ),
            functions=("sqrt",),
        )
    if isinstance(task, PolynomialDerivativeTask):
        return _DerivedTask(
            instruction="Differentiate the polynomial with respect to x.",
            givens=(_render_polynomial(task.polynomial),),
            expected=_differentiate_polynomial(task.polynomial),
            conceptual_hint="Differentiation distributes across sums and differences.",
            operation_hint=(
                "Apply the constant-multiple and power rules to each nonconstant term."
            ),
        )
    if isinstance(task, ProductAtPointTask):
        data = task.data
        expected = data.f_derivative * data.g_value + data.f_value * data.g_derivative
        return _DerivedTask(
            instruction=(
                "Let h be the product of the opaque functions f and g. "
                "Use only the supplied point data to find h' at that point."
            ),
            givens=(
                "h(x)=f(x)*g(x)",
                f"f({data.point})={data.f_value}",
                f"g({data.point})={data.g_value}",
                f"f'({data.point})={data.f_derivative}",
                f"g'({data.point})={data.g_derivative}",
            ),
            expected=str(expected),
            conceptual_hint="A product derivative uses both values and both derivatives.",
            operation_hint=(
                "At the point, compute f' times g plus f times g'."
            ),
            numeric_answer=True,
        )
    if isinstance(task, QuotientAtPointTask):
        data = task.data
        expected = Fraction(
            data.f_derivative * data.g_value - data.f_value * data.g_derivative,
            data.g_value**2,
        )
        return _DerivedTask(
            instruction=(
                "Let h be the quotient of the opaque functions f and g. "
                "Use only the supplied point data to find h' at that point."
            ),
            givens=(
                "h(x)=f(x)/g(x)",
                f"f({data.point})={data.f_value}",
                f"g({data.point})={data.g_value}",
                f"f'({data.point})={data.f_derivative}",
                f"g'({data.point})={data.g_derivative}",
            ),
            expected=_render_fraction(expected),
            conceptual_hint=(
                "A quotient derivative uses both values and derivatives over g squared."
            ),
            operation_hint=(
                "At the point, compute g times f' minus f times g', then divide by g squared."
            ),
            numeric_answer=True,
        )
    raise TypeError(f"unsupported task {type(task).__name__}")


def _review_status_and_provenance(
    source: ProductQuotientBlueprintDocument,
    family: ReleaseFamilyBlueprint,
    review: ContentReviewEntry,
) -> tuple[ReviewStatus, AssessmentProvenance]:
    if review.decision == ReviewDecision.REJECTED:
        raise ProductQuotientCompilationError("rejected families cannot be compiled")
    approved = review.decision == ReviewDecision.APPROVED
    if approved:
        if review.reviewed_by is None or review.reviewed_at is None:
            raise ProductQuotientCompilationError("approved family lacks review provenance")
        if review.reviewed_by.strip().casefold() == source.author.strip().casefold():
            raise ProductQuotientCompilationError("a family author cannot approve their own work")
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


def _build_family_item(
    source: ProductQuotientBlueprintDocument,
    family: ReleaseFamilyBlueprint,
    *,
    review_status: ReviewStatus,
    provenance: AssessmentProvenance,
) -> AssessmentItem:
    derived = _derive_task(family.task)
    given_segments = [
        MathPromptSegment(role=PromptSemanticRole.GIVEN, expression=given)
        for given in derived.givens
    ]
    if family.surface == AssessmentSurface.WORKED_EXAMPLE:
        worked_instruction = (
            "Study this worked example before trying an independent family. "
            + derived.instruction
        )
        if isinstance(family.task, QuotientAtPointTask):
            worked_instruction += (
                " This quotient example and the guided product practice that follows "
                "use the same opaque point-data model."
            )
        prompt = [
            TextPromptSegment(
                role=PromptSemanticRole.INSTRUCTION,
                text=worked_instruction,
            ),
            *given_segments,
            TextPromptSegment(
                role=PromptSemanticRole.WORKED_STEP,
                text=derived.operation_hint,
            ),
            MathPromptSegment(
                role=PromptSemanticRole.WORKED_ANSWER,
                expression=derived.expected,
            ),
        ]
    else:
        prefix = (
            "Use the guided workspace. "
            if family.surface == AssessmentSurface.GUIDED_WIDGET
            else ""
        )
        prompt = [
            TextPromptSegment(
                role=PromptSemanticRole.INSTRUCTION,
                text=prefix + derived.instruction,
            ),
            *given_segments,
            BlankPromptSegment(label="Answer:"),
        ]
    if derived.numeric_answer:
        answer = NumericAnswerSpec(expected=derived.expected, tolerance=0)
        task_kind = AssessmentTaskKind.SOLVE
    else:
        variables = (
            [family.task.base]
            if isinstance(
                family.task,
                (
                    ExponentProductTask,
                    ExponentQuotientTask,
                    ExponentPowerTask,
                    ExponentNegativeTask,
                    ExponentZeroTask,
                    ExponentCompoundTask,
                ),
            )
            else list(derived.variables)
        )
        answer = SymbolicAnswerSpec(
            expected=derived.expected,
            variables=variables,
            functions=list(derived.functions),
        )
        task_kind = AssessmentTaskKind.TRANSFORM
    return AssessmentItem(
        item_id=family.item_id,
        revision=family.revision,
        family_id=family.family_id,
        kc_id=family.kc_id,
        difficulty=family.difficulty,
        task_kind=task_kind,
        eligible_surfaces=[family.surface],
        allocation_order=family.allocation_order,
        prompt=prompt,
        hints=[
            AssessmentHint(text=derived.conceptual_hint),
            AssessmentHint(text=derived.operation_hint),
            AssessmentHint(
                text=f"A correct completed form is {derived.expected}.",
                revealing=True,
            ),
        ],
        answer=answer,
        review_status=review_status,
        provenance=provenance,
    )


def _compiled_review_artifact(
    source: ProductQuotientBlueprintDocument,
    family: ReleaseFamilyBlueprint,
) -> dict[str, object]:
    """Return deterministic compiled bytes without circular review facts."""
    item = _build_family_item(
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


def _compile_family(
    source: ProductQuotientBlueprintDocument,
    family: ReleaseFamilyBlueprint,
    review: ContentReviewEntry,
) -> AssessmentItem:
    review_status, provenance = _review_status_and_provenance(
        source, family, review
    )
    return _build_family_item(
        source,
        family,
        review_status=review_status,
        provenance=provenance,
    )


def _validate_task_kc(family: ReleaseFamilyBlueprint) -> None:
    task = family.task
    if isinstance(
        task,
        (
            ExponentProductTask,
            ExponentQuotientTask,
            ExponentPowerTask,
            ExponentNegativeTask,
            ExponentZeroTask,
            ExponentCompoundTask,
        ),
    ):
        expected_kc = "kc.alg.exponent_rules"
    elif isinstance(
        task,
        (PowerDerivativeTask, RationalPowerDerivativeTask, RadicalPowerDerivativeTask),
    ):
        expected_kc = "kc.der.power_rule"
    elif isinstance(task, PolynomialDerivativeTask):
        expected_kc = "kc.der.sum_constant_rules"
    else:
        expected_kc = TARGET_KC
    if family.kc_id != expected_kc:
        raise ProductQuotientCompilationError(
            f"{family.blueprint_id}: task type belongs to {expected_kc}, not {family.kc_id}"
        )


def _construct_for_task(task: ReleaseMathTask) -> str:
    if isinstance(task, ExponentProductTask):
        return "exponent.product"
    if isinstance(task, ExponentQuotientTask):
        return "exponent.quotient"
    if isinstance(task, ExponentPowerTask):
        return "exponent.power"
    if isinstance(task, ExponentNegativeTask):
        return "exponent.negative"
    if isinstance(task, ExponentZeroTask):
        return "exponent.zero"
    if isinstance(task, ExponentCompoundTask):
        return "exponent.compound"
    if isinstance(task, PowerDerivativeTask):
        return (
            "power.positive_integer"
            if task.exponent > 0
            else "power.negative_integer"
        )
    if isinstance(task, RationalPowerDerivativeTask):
        return "power.rational"
    if isinstance(task, RadicalPowerDerivativeTask):
        return (
            "power.sqrt"
            if task.form == "sqrt"
            else "power.reciprocal_radical"
        )
    if isinstance(task, PolynomialDerivativeTask):
        return "sum.polynomial_termwise"
    if isinstance(task, ProductAtPointTask):
        return "product_quotient.product_at_point"
    if isinstance(task, QuotientAtPointTask):
        return "product_quotient.quotient_at_point"
    raise TypeError(f"unsupported task {type(task).__name__}")


def _validate_construct_taxonomy(
    source: ProductQuotientBlueprintDocument,
) -> None:
    for family in source.families:
        inferred = _construct_for_task(family.task)
        if family.construct_id != inferred:
            raise ProductQuotientCompilationError(
                f"{family.blueprint_id}: construct {family.construct_id!r} "
                f"does not match typed task construct {inferred!r}"
            )
    for kc_id, by_surface in EXPECTED_CONSTRUCT_ORDER.items():
        for surface, expected_constructs in by_surface.items():
            ordered = sorted(
                (
                    family
                    for family in source.families
                    if family.kc_id == kc_id and family.surface == surface
                ),
                key=lambda family: family.allocation_order,
            )
            actual = tuple(
                (family.allocation_order, family.construct_id)
                for family in ordered
            )
            expected = tuple(
                ((index + 1) * 10, construct_id)
                for index, construct_id in enumerate(expected_constructs)
            )
            if actual != expected:
                raise ProductQuotientCompilationError(
                    f"{kc_id}/{surface.value}: construct/order taxonomy mismatch; "
                    f"expected={expected}, got={actual}"
                )


def _visible_content(item: AssessmentItem) -> list[str]:
    visible = [render_prompt(item)]
    for segment in item.prompt:
        if isinstance(segment, TextPromptSegment):
            visible.append(segment.text)
        elif isinstance(segment, MathPromptSegment):
            visible.append(segment.expression)
    visible.extend(hint.text for hint in item.hints)
    return list(dict.fromkeys(visible))


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


def validate_inventory_separation(
    items: list[AssessmentItem],
    graph: GraphDocument,
) -> InventorySeparationReport:
    """Exhaustively reject answer reuse and cross-family visible leakage."""
    ordered = sorted(items, key=lambda item: (item.family_id, item.item_id))
    errors: list[str] = []
    math_result = _supervised_math_separation(ordered)
    by_item_id = {item.item_id: item for item in ordered}
    for left_id, right_id in math_result.answer_reuse:
        errors.append(
            "expected answer reused across families "
            f"{by_item_id[left_id].family_id!r} and "
            f"{by_item_id[right_id].family_id!r}"
        )
    for source_id, target_id in math_result.visible_leakage:
        errors.append(
            f"{source_id}: visible math is equivalent to the answer for {target_id}"
        )

    literal_visible_pairs = len(ordered) * max(0, len(ordered) - 1)
    for source_item in ordered:
        visible = "\n".join(_visible_content(source_item))
        for target in ordered:
            if target.family_id == source_item.family_id:
                continue
            if not isinstance(target.answer, (SymbolicAnswerSpec, NumericAnswerSpec)):
                errors.append(
                    "inventory separation requires scalar symbolic or numeric answers"
                )
            elif _literal_answer_visible(target.answer.expected, visible):
                errors.append(
                    f"{source_item.item_id}: visible content leaks answer for {target.item_id}"
                )

    graph_visible = "\n".join(
        text
        for node in graph.nodes
        if node.id in TARGET_CLOSURE
        for text in (node.name, node.description)
    )
    for target in ordered:
        if isinstance(
            target.answer, (SymbolicAnswerSpec, NumericAnswerSpec)
        ) and _literal_answer_visible(target.answer.expected, graph_visible):
            errors.append(
                f"student-visible graph content leaks answer for {target.item_id}"
            )
    return InventorySeparationReport(
        answer_pairs_checked=math_result.answer_comparisons,
        visible_candidate_comparisons_checked=(
            math_result.visible_candidate_comparisons
        ),
        literal_visible_pairs_checked=literal_visible_pairs,
        errors=tuple(errors),
    )


def compile_release_inventory(
    source: ProductQuotientBlueprintDocument,
    manifest: ContentReviewManifest,
    graph: GraphDocument,
) -> tuple[ItemBankDocument, InventorySeparationReport]:
    """Compile, verify, and separate all 52 independently reviewable families."""
    if source.graph_version != graph.graph_version:
        raise ProductQuotientCompilationError("source and graph versions differ")
    if manifest.graph_version != graph.graph_version:
        raise ProductQuotientCompilationError("review manifest and graph versions differ")
    if manifest.compiler_version != COMPILER_VERSION:
        raise ProductQuotientCompilationError(
            f"manifest compiler pin must be {COMPILER_VERSION!r}"
        )
    actual_closure = ancestor_subgraph(graph, TARGET_KC, hard_only=True).node_ids()
    if actual_closure != set(TARGET_CLOSURE):
        raise ProductQuotientCompilationError(
            f"graph hard closure changed: expected {sorted(TARGET_CLOSURE)}, "
            f"got {sorted(actual_closure)}"
        )
    if set(source.target_kcs) != set(TARGET_CLOSURE):
        raise ProductQuotientCompilationError("source target_kcs is not the exact hard closure")

    expected_identities = {
        (family.blueprint_id, family.revision) for family in source.families
    }
    reviews = {
        (entry.blueprint_id, entry.revision): entry for entry in manifest.entries
    }
    if set(reviews) != expected_identities:
        missing = sorted(expected_identities - set(reviews))
        extra = sorted(set(reviews) - expected_identities)
        raise ProductQuotientCompilationError(
            f"review/source identity mismatch; missing={missing}, extra={extra}"
        )

    matrix = Counter((family.kc_id, family.surface) for family in source.families)
    expected_matrix = {
        (kc_id, surface): count
        for kc_id in TARGET_CLOSURE
        for surface, count in EXPECTED_FAMILY_COUNTS.items()
    }
    if dict(matrix) != expected_matrix:
        raise ProductQuotientCompilationError(
            "family matrix must contain exactly 13 families per KC: "
            "four diagnostic, five check-in, one guided widget, two capstone, "
            "and one worked example"
        )
    _validate_construct_taxonomy(source)

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
        _validate_task_kc(family)
        review = reviews[(family.blueprint_id, family.revision)]
        if review.source_digest != family_digest(source, family):
            raise ProductQuotientCompilationError(
                f"review digest mismatch for {family.blueprint_id}@{family.revision}"
            )
        item = _compile_family(source, family, review)
        verdict = verify_answer(item.answer, item.answer.expected, supervised=True)
        if verdict.status != VerificationStatus.CORRECT:
            raise ProductQuotientCompilationError(
                f"{item.item_id}: derived expected answer failed verification ({verdict.code})"
            )
        items.append(item)

    approved_kcs = {
        kc_id
        for kc_id in TARGET_CLOSURE
        if all(
            item.review_status == ReviewStatus.HUMAN_APPROVED
            for item in items
            if item.kc_id == kc_id
        )
    }
    if not set(source.released_kcs) <= approved_kcs:
        raise ProductQuotientCompilationError(
            "released_kcs contains a KC without complete independent approval"
        )
    if source.released_kcs and set(source.released_kcs) != set(TARGET_CLOSURE):
        raise ProductQuotientCompilationError(
            "this pilot release is atomic and must publish the exact hard closure"
        )

    bank = ItemBankDocument(
        bank_version=source.output_bank_version,
        graph_version=source.graph_version,
        released_kcs=source.released_kcs,
        items=items,
    )
    report = validate_inventory_separation(items, graph)
    if report.errors:
        raise ProductQuotientCompilationError(
            "inventory separation failed: " + "; ".join(report.errors)
        )
    return bank, report


def _atomic_write_bank(path: Path, bank: ItemBankDocument) -> None:
    """Durably replace a compiled bank only after validating staged bytes."""
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
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    """Check the pending inventory or write its deterministic compiled bank."""
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
        graph = GraphDocument.model_validate_json(args.graph.read_text(encoding="utf-8"))
        bank, report = compile_release_inventory(source, manifest, graph)
    except Exception as exc:  # noqa: BLE001 - fail closed at the CLI boundary
        print(f"Product/Quotient inventory INVALID: {exc}", file=sys.stderr)
        return 1
    if args.out is not None:
        try:
            _atomic_write_bank(args.out, bank)
        except Exception as exc:  # noqa: BLE001 - preserve any prior destination
            print(f"Product/Quotient bank write FAILED: {exc}", file=sys.stderr)
            return 1
    status_counts = Counter(item.review_status for item in bank.items)
    print(
        "Product/Quotient inventory OK: "
        f"{len(bank.items)} total families, "
        f"{status_counts[ReviewStatus.DRAFT]} draft, "
        f"{status_counts[ReviewStatus.HUMAN_APPROVED]} approved, "
        f"{report.answer_pairs_checked} answer comparisons, "
        f"{report.visible_candidate_comparisons_checked} "
        "visible candidate comparisons, "
        f"{report.literal_visible_pairs_checked} literal cross-family scans, "
        f"released KCs={len(bank.released_kcs)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
