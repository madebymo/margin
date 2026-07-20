"""Load and validate the reviewed v2 assessment item bank."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Literal

from tutor.schemas.assessment import (
    AnswerSpec,
    AntiderivativeAnswerSpec,
    AssessmentItem,
    AssessmentSurface,
    AssessmentTaskKind,
    BlankPromptSegment,
    ChoiceAnswerSpec,
    FiniteSetAnswerSpec,
    IntervalSetAnswerSpec,
    ItemBankDocument,
    MathPromptSegment,
    NumericAnswerSpec,
    OrderedTupleAnswerSpec,
    PromptSemanticRole,
    SymbolicAnswerSpec,
    TextPromptSegment,
)
from tutor.schemas.common import EdgeType, ReviewStatus
from tutor.schemas.kc import GraphDocument
from tutor.verify.checker import (
    VerificationResult,
    VerificationStatus,
    verify_answer,
)

DEFAULT_ITEM_BANK_PATH = (
    Path(__file__).resolve().parents[1] / "seed" / "item_bank_v2.json"
)

InputMode = Literal["expression", "number", "choice", "set", "tuple"]

_MINIMUM_FAMILIES: dict[AssessmentSurface, int] = {
    AssessmentSurface.DIAGNOSTIC: 3,
    AssessmentSurface.CHECKIN: 4,
    AssessmentSurface.GUIDED_WIDGET: 1,
    AssessmentSurface.CAPSTONE: 2,
    AssessmentSurface.WORKED_EXAMPLE: 1,
}

_INLINE_ATOM = (
    r"(?:"
    r"[A-Za-z][A-Za-z0-9]*(?:\([^()\n]{1,128}\))?"
    r"|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    r"|\([^()\n]{1,128}\)"
    r")"
)
_INLINE_EXPRESSION = re.compile(
    rf"(?<![A-Za-z0-9_])"
    rf"(?:[+-]\s*)?{_INLINE_ATOM}"
    rf"(?:\s*(?:\*\*|\^|[+\-*/])\s*(?:[+-]\s*)?{_INLINE_ATOM})+"
    rf"(?![A-Za-z0-9_])"
)
# The restricted answer grammar intentionally accepts implicit multiplication,
# so answer-separation must recognize it too.  Keep this extractor narrow: it
# starts with a numeric coefficient and requires a following symbolic or
# parenthesized factor.  Contract-aware identifier filtering below discards
# prose fragments that cannot be submitted for the upcoming item.
_INLINE_IMPLICIT_COEFFICIENT = re.compile(
    rf"(?<![A-Za-z0-9_])"
    rf"(?:[+-]\s*)?"
    rf"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    rf"\s*(?:[A-Za-z][A-Za-z0-9]*(?:\([^()\n]{{1,128}}\))?|\([^()\n]{{1,128}}\))"
    rf"(?:\s*(?:\*\*|\^|[+\-*/])\s*(?:[+-]\s*)?{_INLINE_ATOM})*"
    rf"(?![A-Za-z0-9_])"
)
_INLINE_CONTAINER = re.compile(
    r"(?:\{[^{}\n]{1,256}\}|\[[^\[\]\n]{1,256}\]|"
    r"\([^(),\n]{1,128},[^()\n]{1,128}\))"
)
_INTERVAL_COMPONENT = r"(?:\([^()\n]{1,128},[^()\n]{1,128}\)|\[[^\[\]\n]{1,256}\])"
_INLINE_INTERVAL_SET = re.compile(
    rf"{_INTERVAL_COMPONENT}(?:\s*(?:[Uu]|∪)\s*{_INTERVAL_COMPONENT})+"
)
_SCIENTIFIC_LITERAL = re.compile(
    r"(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))[eE][+-]?\d+"
)
_SHORT_SCALAR = re.compile(
    r"(?:[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?|"
    r"[A-Za-z][A-Za-z0-9]*)"
)
_INLINE_NUMERIC_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_.^*/+\-])"
    r"(?:[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?)"
    r"(?![A-Za-z0-9_^*/])"
)


class _EquivalenceState(StrEnum):
    """The three possible outcomes of an answer-separation comparison."""

    EQUIVALENT = "equivalent"
    UNEQUAL = "unequal"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True)
class _EquivalenceComparison:
    state: _EquivalenceState
    codes: tuple[str, ...] = ()


class AnswerSeparationIndeterminate(RuntimeError):
    """Raised when a boolean helper cannot prove either equality or inequality."""


def _safe_verify(
    answer: AnswerSpec,
    candidate: str,
    *,
    supervised: bool,
) -> VerificationResult:
    """Convert an unexpected verifier failure into a fail-closed typed result."""
    try:
        return verify_answer(answer, candidate, supervised=supervised)
    except Exception:  # noqa: BLE001 - content release must fail closed
        return VerificationResult(
            status=VerificationStatus.INVALID,
            code="verifier_exception",
        )


def _indeterminate_code(verdict: VerificationResult) -> str:
    return f"{verdict.status.value}:{verdict.code}"


def load_item_bank(path: Path | None = None) -> ItemBankDocument:
    """Parse the default packaged bank or an explicitly supplied document."""
    source = path or DEFAULT_ITEM_BANK_PATH
    return ItemBankDocument.model_validate_json(source.read_text())


def render_prompt(item: AssessmentItem, *, blank: str = "____") -> str:
    """Render structured segments without exposing the hidden answer."""
    parts: list[str] = []
    for segment in item.prompt:
        if isinstance(segment, TextPromptSegment):
            parts.append(segment.text)
        elif isinstance(segment, MathPromptSegment):
            parts.append(segment.expression)
        elif isinstance(segment, BlankPromptSegment):
            parts.append(segment.label or blank)
    return " ".join(part.strip() for part in parts if part.strip())


def input_mode_for(item: AssessmentItem) -> InputMode:
    """Map an authored answer contract to a client-safe input control."""
    answer = item.answer
    if isinstance(answer, NumericAnswerSpec):
        return "number"
    if isinstance(answer, ChoiceAnswerSpec):
        return "choice"
    if isinstance(answer, (FiniteSetAnswerSpec, IntervalSetAnswerSpec)):
        return "set"
    if isinstance(answer, OrderedTupleAnswerSpec):
        return "tuple"
    return "expression"


def bundle_leakage_problems(
    visible_texts: Iterable[str],
    upcoming_items: Iterable[AssessmentItem],
    *,
    supervised: bool = True,
) -> list[str]:
    """Find reviewed answers disclosed by generated learner-visible content.

    This is the cross-item gate used after narrative/widget generation but
    before display.  It complements per-item bank validation and returns only
    stable item ids, never the hidden expected values.
    """
    visible = "\n".join(visible_texts)
    compact_visible = _compact(visible)
    errors: list[str] = []
    for item in upcoming_items:
        leaked = False
        indeterminate: set[str] = set()
        for raw_expected in _literal_answer_values(item):
            expected = _compact(raw_expected)
            if (
                len(expected) >= 3
                and expected in compact_visible
                or _short_scalar_is_visible(raw_expected, visible)
            ):
                leaked = True
                break
            if expected and re.search(
                rf"(?:answer|result|correct(?:\s+value)?|=)\s*(?:is\s*)?"
                rf"{re.escape(expected)}(?![a-z0-9])",
                compact_visible,
            ):
                leaked = True
                break
        if not leaked:
            for candidate in _candidate_answer_texts(visible):
                if not _candidate_fits_answer_contract(item, candidate):
                    continue
                verdict = _safe_verify(
                    item.answer,
                    candidate,
                    supervised=supervised,
                )
                if verdict.status == VerificationStatus.CORRECT:
                    leaked = True
                    break
                if verdict.status != VerificationStatus.INCORRECT:
                    indeterminate.add(_indeterminate_code(verdict))
        if leaked:
            errors.append(f"{item.item_id}: expected answer leaks into visible content")
        elif indeterminate:
            errors.append(
                f"{item.item_id}: answer-separation check indeterminate "
                f"({', '.join(sorted(indeterminate))})"
            )
    return errors


def _canonical_submission(item: AssessmentItem) -> str:
    answer = item.answer
    if isinstance(answer, (SymbolicAnswerSpec, NumericAnswerSpec, AntiderivativeAnswerSpec)):
        return answer.expected
    if isinstance(answer, FiniteSetAnswerSpec):
        return "{" + ", ".join(answer.expected) + "}"
    if isinstance(answer, OrderedTupleAnswerSpec):
        return "(" + ", ".join(answer.expected) + ")"
    if isinstance(answer, IntervalSetAnswerSpec):
        intervals: list[str] = []
        for interval in answer.expected:
            left = "[" if interval.lower_closed else "("
            right = "]" if interval.upper_closed else ")"
            intervals.append(f"{left}{interval.lower}, {interval.upper}{right}")
        return " U ".join(intervals)
    if isinstance(answer, ChoiceAnswerSpec):
        return answer.expected_choice_id
    raise TypeError(f"unsupported answer spec {type(answer).__name__}")


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def _short_scalar_is_visible(expected: str, visible: str) -> bool:
    """Match an authored scalar as a token, not inside another math token."""
    scalar = expected.strip()
    if not _SHORT_SCALAR.fullmatch(scalar):
        return False
    return bool(
        re.search(
            rf"(?<![A-Za-z0-9_.^*/+\-]){re.escape(scalar)}"
            rf"(?![A-Za-z0-9_^*/])(?!\.\d)",
            visible,
            flags=re.IGNORECASE,
        )
    )


def _candidate_answer_texts(visible: str) -> list[str]:
    """Return exact and explicitly introduced answer-like fragments.

    Prompt math segments and standalone hints arrive as their own lines.  The
    marker extraction additionally catches equivalent forms embedded in prose,
    such as ``the result is 5*x^4 + 5*x^4``.
    """
    candidates: list[str] = []
    for line in visible.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.fullmatch(rf"(?:[+-]\s*)?{_INLINE_ATOM}", stripped):
            candidates.append(stripped)
        sentence_tail = re.search(
            rf"(?<![A-Za-z0-9_.^*/+\-])((?:[+-]\s*)?{_INLINE_ATOM})"
            rf"\s*[.;:]?$",
            stripped,
        )
        if sentence_tail:
            candidates.append(sentence_tail.group(1))
        marker = re.search(
            r"(?:answer|result|correct(?:\s+value)?)\s*(?:is|=|:)\s*(.+)$",
            stripped,
            flags=re.IGNORECASE,
        )
        if marker:
            candidate = marker.group(1).strip().rstrip(".;:")
            if candidate:
                candidates.append(candidate)
        candidates.extend(
            match.group().strip().rstrip(".;:")
            for pattern in (
                _INLINE_INTERVAL_SET,
                _INLINE_IMPLICIT_COEFFICIENT,
                _INLINE_EXPRESSION,
                _INLINE_CONTAINER,
                _INLINE_NUMERIC_TOKEN,
            )
            for match in pattern.finditer(stripped)
        )
    # Preserve display order while avoiding repeated verifier work.
    return list(dict.fromkeys(candidate for candidate in candidates if candidate))


def _candidate_fits_answer_contract(
    item: AssessmentItem,
    candidate: str,
) -> bool:
    """Skip fragments that cannot be a valid submission for this answer type.

    The equivalence worker should only see answer-shaped fragments. For
    example, derivative notation ``d/dx`` in a prompt is definitively not a
    symbolic answer whose sole declared variable is ``x``; treating its
    expected parser rejection as a verifier outage would reject every normal
    lesson. Once a plausible fragment reaches the worker, however, every
    INVALID or TIMEOUT result is indeterminate and therefore release-blocking.
    """
    stripped = candidate.strip()
    answer = item.answer
    if isinstance(answer, ChoiceAnswerSpec):
        return stripped in answer.option_ids
    if isinstance(answer, FiniteSetAnswerSpec):
        if not (stripped.startswith("{") and stripped.endswith("}")):
            return False
    elif isinstance(answer, OrderedTupleAnswerSpec):
        if not (
            stripped.startswith("(")
            and stripped.endswith(")")
            and "," in stripped
        ):
            return False
    elif isinstance(answer, IntervalSetAnswerSpec):
        if not (
            stripped.startswith(("(", "["))
            and stripped.endswith((")", "]"))
            and "," in stripped
        ):
            return False
    elif stripped.startswith(("{", "[")) or (
        stripped.startswith("(")
        and stripped.endswith(")")
        and "," in stripped
    ):
        return False

    without_scientific = _SCIENTIFIC_LITERAL.sub("0", stripped)
    identifiers = set(re.findall(r"[A-Za-z][A-Za-z0-9]*", without_scientific))
    if isinstance(answer, NumericAnswerSpec):
        allowed_identifiers: set[str] = set()
    elif isinstance(answer, IntervalSetAnswerSpec):
        allowed_identifiers = {"U", "u", "inf", "infinity"}
    else:
        allowed_identifiers = set(getattr(answer, "variables", ()))
        allowed_identifiers.update(getattr(answer, "functions", ()))
        allowed_identifiers.update({"pi", "e"})
        if isinstance(answer, AntiderivativeAnswerSpec):
            allowed_identifiers.update({answer.variable, "C"})
        assignment_lhs = getattr(answer, "assignment_lhs", None)
        if assignment_lhs:
            allowed_identifiers.update(
                re.findall(r"[A-Za-z][A-Za-z0-9]*", assignment_lhs)
            )
    return identifiers <= allowed_identifiers


def _literal_answer_values(item: AssessmentItem) -> list[str]:
    answer = item.answer
    if isinstance(answer, (SymbolicAnswerSpec, NumericAnswerSpec, AntiderivativeAnswerSpec)):
        return [answer.expected]
    if isinstance(answer, (FiniteSetAnswerSpec, OrderedTupleAnswerSpec)):
        return list(answer.expected)
    if isinstance(answer, IntervalSetAnswerSpec):
        return [
            value
            for interval in answer.expected
            for value in (interval.lower, interval.upper)
            if value.lower() not in {"-inf", "inf", "+inf", "-infinity", "infinity"}
        ]
    return []


def _compare_answers(
    left: AssessmentItem,
    right: AssessmentItem,
) -> _EquivalenceComparison:
    """Compare authored truth without collapsing verifier failure to inequality."""
    if isinstance(left.answer, ChoiceAnswerSpec) or isinstance(
        right.answer, ChoiceAnswerSpec
    ):
        # Choice ids are local to their option set; do not equate two generic
        # ids such as "a" unless the complete answer contracts are identical.
        equivalent = (
            isinstance(left.answer, ChoiceAnswerSpec)
            and isinstance(right.answer, ChoiceAnswerSpec)
            and left.answer.model_dump(mode="json")
            == right.answer.model_dump(mode="json")
        )
        return _EquivalenceComparison(
            _EquivalenceState.EQUIVALENT
            if equivalent
            else _EquivalenceState.UNEQUAL
        )

    collection_types = (
        FiniteSetAnswerSpec,
        IntervalSetAnswerSpec,
        OrderedTupleAnswerSpec,
    )
    if type(left.answer) is not type(right.answer) and (
        isinstance(left.answer, collection_types)
        or isinstance(right.answer, collection_types)
    ):
        # Sets, intervals, tuples, and scalar expressions have discriminated,
        # non-interchangeable submission domains. Their parser rejection in a
        # cross-check is expected, not evidence that the verifier is unhealthy.
        return _EquivalenceComparison(_EquivalenceState.UNEQUAL)

    left_submission = _canonical_submission(left)
    right_submission = _canonical_submission(right)
    left_verdict = _safe_verify(
        left.answer,
        right_submission,
        supervised=True,
    )
    if left_verdict.status == VerificationStatus.CORRECT:
        return _EquivalenceComparison(_EquivalenceState.EQUIVALENT)
    right_verdict = _safe_verify(
        right.answer,
        left_submission,
        supervised=True,
    )
    if right_verdict.status == VerificationStatus.CORRECT:
        return _EquivalenceComparison(_EquivalenceState.EQUIVALENT)
    verdicts = (left_verdict, right_verdict)
    if any(verdict.status == VerificationStatus.INCORRECT for verdict in verdicts):
        # At least one answer contract parsed the other canonical submission
        # and definitively disproved equality. A parser-invalid reverse
        # direction does not erase that mathematical result.
        return _EquivalenceComparison(_EquivalenceState.UNEQUAL)
    return _EquivalenceComparison(
        _EquivalenceState.INDETERMINATE,
        tuple(
            sorted(
                {
                    _indeterminate_code(verdict)
                    for verdict in verdicts
                    if verdict.status != VerificationStatus.INCORRECT
                }
            )
        ),
    )


def _answers_equivalent(left: AssessmentItem, right: AssessmentItem) -> bool:
    """Boolean compatibility helper that raises rather than failing open."""
    comparison = _compare_answers(left, right)
    if comparison.state == _EquivalenceState.INDETERMINATE:
        raise AnswerSeparationIndeterminate(
            "answer comparison was indeterminate: " + ", ".join(comparison.codes)
        )
    return comparison.state == _EquivalenceState.EQUIVALENT


def _visible_item_content(item: AssessmentItem) -> list[str]:
    """Return all content that can precede an independent scored item."""
    visible = [render_prompt(item)]
    for segment in item.prompt:
        if isinstance(segment, TextPromptSegment):
            visible.append(segment.text)
        elif isinstance(segment, MathPromptSegment):
            visible.append(segment.expression)
    # The final hint may reveal this item's own answer, but it must never
    # disclose the answer to a different family that could still be scored.
    visible.extend(hint.text for hint in item.hints)
    if AssessmentSurface.WORKED_EXAMPLE in item.eligible_surfaces:
        visible.append(_canonical_submission(item))
    return visible


def _leakage_problems(item: AssessmentItem) -> list[str]:
    """Conservative bundle-independent checks for direct answer disclosure."""
    if item.eligible_surfaces == [AssessmentSurface.WORKED_EXAMPLE]:
        return []
    errors: list[str] = []
    expected_values = [_compact(value) for value in _literal_answer_values(item)]
    visible_text = " ".join(
        segment.text
        for segment in item.prompt
        if isinstance(segment, TextPromptSegment)
    )
    visible_text += " " + " ".join(hint.text for hint in item.hints[:2])
    compact_visible = _compact(visible_text)
    for expected in expected_values:
        if len(expected) >= 3 and expected in compact_visible:
            errors.append("expected answer appears in prompt or non-revealing hint")
            break

    if not isinstance(item.answer, ChoiceAnswerSpec):
        for segment in item.prompt:
            if not isinstance(segment, MathPromptSegment):
                continue
            candidates = list(
                dict.fromkeys(
                    [
                        segment.expression,
                        *_candidate_answer_texts(segment.expression),
                    ]
                )
            )
            for candidate in candidates:
                if not _candidate_fits_answer_contract(item, candidate):
                    continue
                verdict = _safe_verify(
                    item.answer,
                    candidate,
                    supervised=True,
                )
                if verdict.status == VerificationStatus.CORRECT:
                    canonical = _compact(_canonical_submission(item)).replace("**", "^")
                    normalized_given = _compact(segment.expression).replace("**", "^")
                    is_distinct_transform_given = (
                        item.task_kind == AssessmentTaskKind.TRANSFORM
                        and segment.role == PromptSemanticRole.GIVEN
                        and candidate == segment.expression
                        and normalized_given != canonical
                    )
                    if is_distinct_transform_given:
                        continue
                    errors.append(
                        "a visible math segment is equivalent to the expected answer"
                    )
                    break
                if verdict.status != VerificationStatus.INCORRECT:
                    errors.append(
                        "a visible math segment answer-separation check is "
                        f"indeterminate ({_indeterminate_code(verdict)})"
                    )
                    break
            if errors:
                break
    return errors


def _validate_item_bank_uncached(
    bank: ItemBankDocument,
    graph: GraphDocument,
    released_kcs: set[str] | None = None,
    reviewed_misconceptions: dict[str, set[str]] | None = None,
) -> list[str]:
    """Return deterministic release-blocking bank errors.

    Besides schema validation, release validation checks graph pins, reviewed
    provenance, parseability, independent-family coverage, direct leakage, and
    duplicate mathematical answers.
    """
    errors: list[str] = []
    requested = set(bank.released_kcs) if released_kcs is None else set(released_kcs)
    declared_released = set(bank.released_kcs)
    graph_kcs = graph.node_ids()
    if bank.graph_version != graph.graph_version:
        errors.append(
            f"graph version mismatch: bank={bank.graph_version}, graph={graph.graph_version}"
        )
    for kc_id in sorted(requested - graph_kcs):
        errors.append(f"released KC is absent from graph: {kc_id}")
    for kc_id in sorted(declared_released - graph_kcs):
        errors.append(f"bank names unknown released KC: {kc_id}")
    for kc_id in sorted(requested - declared_released):
        errors.append(f"requested KC is not declared released: {kc_id}")

    hard_predecessors: dict[str, set[str]] = defaultdict(set)
    for edge in graph.edges:
        if edge.type == EdgeType.HARD:
            hard_predecessors[edge.to_kc].add(edge.from_kc)
    reviewed_misconceptions = reviewed_misconceptions or {}

    family_surfaces: dict[tuple[str, AssessmentSurface], set[str]] = defaultdict(set)
    release_family_surface: dict[tuple[str, str], AssessmentSurface] = {}
    reported_cross_surface_families: set[tuple[str, str]] = set()
    production_family_surfaces: dict[
        tuple[str, AssessmentSurface], set[str]
    ] = defaultdict(set)
    release_order_families: dict[
        tuple[str, AssessmentSurface, int], set[str]
    ] = defaultdict(set)
    scored_items: list[AssessmentItem] = []
    visible_items: list[AssessmentItem] = []
    item_kcs = {item.kc_id for item in bank.items}
    for missing in sorted(requested - item_kcs):
        errors.append(f"{missing}: no assessment items")

    for item in sorted(bank.items, key=lambda entry: (entry.kc_id, entry.item_id)):
        prefix = f"{item.kc_id}/{item.item_id}@{item.revision}"
        is_release_item = item.kc_id in requested
        if item.kc_id not in graph_kcs:
            errors.append(f"{prefix}: item KC is absent from graph")
        if len(item.eligible_surfaces) != 1:
            errors.append(f"{prefix}: each reviewed family must have exactly one surface")
        for surface in item.eligible_surfaces:
            family_key = (item.kc_id, item.family_id)
            previous_surface = release_family_surface.setdefault(family_key, surface)
            if (
                is_release_item
                and previous_surface != surface
                and family_key not in reported_cross_surface_families
            ):
                reported_cross_surface_families.add(family_key)
                errors.append(
                    f"{item.kc_id}/{item.family_id}: family spans surfaces "
                    f"{previous_surface.value} and {surface.value}"
                )
            family_surfaces[(item.kc_id, surface)].add(item.family_id)
            if not isinstance(item.answer, ChoiceAnswerSpec):
                production_family_surfaces[(item.kc_id, surface)].add(
                    item.family_id
                )
        if is_release_item and item.review_status != ReviewStatus.HUMAN_APPROVED:
            errors.append(f"{prefix}: item is not human_approved")
        if is_release_item and item.allocation_order is None:
            errors.append(f"{prefix}: released item lacks allocation_order")
        elif is_release_item:
            for surface in item.eligible_surfaces:
                release_order_families[
                    (item.kc_id, surface, item.allocation_order)
                ].add(item.family_id)
        if (
            is_release_item
            and (
                item.provenance.reviewed_by is None
                or item.provenance.reviewed_at is None
            )
        ):
            errors.append(f"{prefix}: approved item lacks reviewer provenance")
        if is_release_item and isinstance(item.answer, ChoiceAnswerSpec):
            errors.append(
                f"{prefix}: choice items cannot be released until their semantic "
                "option content participates in answer-reuse validation"
            )

        canonical = _canonical_submission(item)
        verdict = _safe_verify(item.answer, canonical, supervised=True)
        if verdict.status != VerificationStatus.CORRECT:
            errors.append(f"{prefix}: authored expected answer is not verifiable ({verdict.code})")
        for leakage in _leakage_problems(item):
            errors.append(f"{prefix}: {leakage}")

        if is_release_item:
            visible_items.append(item)
            if AssessmentSurface.WORKED_EXAMPLE not in item.eligible_surfaces:
                scored_items.append(item)
        for signature in item.error_signatures:
            if (
                is_release_item
                and signature.misconception_id is not None
                and signature.misconception_id
                not in reviewed_misconceptions.get(item.kc_id, set())
            ):
                errors.append(
                    f"{prefix}: misconception {signature.misconception_id} "
                    "is not in a human-approved pedagogy pack for this KC"
                )
            if (
                signature.implicated_prereq is not None
                and signature.implicated_prereq not in hard_predecessors[item.kc_id]
            ):
                errors.append(
                    f"{prefix}: error signature prerequisite "
                    f"{signature.implicated_prereq} is not a direct hard predecessor"
                )

    for (kc_id, surface, order), families in sorted(
        release_order_families.items(),
        key=lambda entry: (entry[0][0], entry[0][1].value, entry[0][2]),
    ):
        if len(families) > 1:
            errors.append(
                f"{kc_id}/{surface.value}: allocation_order {order} is reused "
                f"across families {sorted(families)}"
            )

    reported_family_pairs: set[tuple[str, str]] = set()
    ordered_scored = sorted(
        scored_items,
        key=lambda item: (item.family_id, item.item_id, item.revision),
    )

    # KC names and descriptions are learner-visible before and during an
    # episode (goal catalog, progress, lesson narrative, and remediation).
    # Treat them as part of the released content bundle so a graph edit cannot
    # disclose the truth for any mastery-bearing or guided assessment family.
    # Canonical examples are deliberately excluded: v2 never renders them and
    # they remain explanatory seed data only.
    for node in sorted(
        (node for node in graph.nodes if node.id in requested),
        key=lambda node: node.id,
    ):
        for problem in bundle_leakage_problems(
            [node.name, node.description],
            ordered_scored,
            supervised=True,
        ):
            errors.append(
                f"{node.id}: student-visible graph content {problem}"
            )

    for index, left in enumerate(ordered_scored):
        for right in ordered_scored[index + 1 :]:
            if left.family_id == right.family_id:
                continue
            family_pair = tuple(sorted((left.family_id, right.family_id)))
            if family_pair in reported_family_pairs:
                continue
            comparison = _compare_answers(left, right)
            if comparison.state == _EquivalenceState.INDETERMINATE:
                reported_family_pairs.add(family_pair)
                errors.append(
                    "expected-answer comparison indeterminate across families "
                    f"{list(family_pair)} ({', '.join(comparison.codes)})"
                )
            elif comparison.state == _EquivalenceState.EQUIVALENT:
                reported_family_pairs.add(family_pair)
                errors.append(
                    "expected answer reused across families "
                    f"{list(family_pair)} (mathematically equivalent)"
                )

    for source in sorted(visible_items, key=lambda item: item.item_id):
        visible = _visible_item_content(source)
        for target in ordered_scored:
            if source.family_id == target.family_id:
                continue
            leakage_problems = bundle_leakage_problems(
                visible,
                [target],
                supervised=True,
            )
            for problem in leakage_problems:
                if "expected answer leaks into visible content" in problem:
                    errors.append(
                        f"{source.kc_id}/{source.item_id}@{source.revision}: "
                        f"visible content leaks scored answer for {target.item_id}"
                    )
                else:
                    errors.append(
                        f"{source.kc_id}/{source.item_id}@{source.revision}: "
                        f"{problem}"
                    )

    for kc_id in sorted(requested):
        for surface, minimum in _MINIMUM_FAMILIES.items():
            actual = len(family_surfaces[(kc_id, surface)])
            if actual < minimum:
                errors.append(
                    f"{kc_id}: {surface.value} has {actual} distinct families; "
                    f"requires {minimum}"
                )
        for surface, minimum in (
            (AssessmentSurface.DIAGNOSTIC, 2),
            (AssessmentSurface.CHECKIN, 4),
            (AssessmentSurface.CAPSTONE, 2),
        ):
            actual = len(production_family_surfaces[(kc_id, surface)])
            if actual < minimum:
                errors.append(
                    f"{kc_id}: {surface.value} has {actual} production families; "
                    f"requires {minimum}"
                )

    item_identity_counts = Counter((item.item_id, item.revision) for item in bank.items)
    for identity, count in sorted(item_identity_counts.items()):
        if count > 1:  # Normally caught by the Pydantic document validator.
            errors.append(f"duplicate item revision: {identity[0]}@{identity[1]}")
    return errors


@lru_cache(maxsize=128)
def _cached_validation(
    bank_json: str,
    graph_json: str,
    released_kcs: tuple[str, ...],
    reviewed_misconceptions: tuple[tuple[str, tuple[str, ...]], ...],
) -> tuple[str, ...]:
    """Memoize immutable release documents across catalog and session gates."""
    bank = ItemBankDocument.model_validate_json(bank_json)
    graph = GraphDocument.model_validate_json(graph_json)
    return tuple(
        _validate_item_bank_uncached(
            bank,
            graph,
            released_kcs=set(released_kcs),
            reviewed_misconceptions={
                kc_id: set(ids) for kc_id, ids in reviewed_misconceptions
            },
        )
    )


def validate_item_bank(
    bank: ItemBankDocument,
    graph: GraphDocument,
    released_kcs: set[str] | None = None,
) -> list[str]:
    """Return cached, deterministic release-blocking bank errors."""
    from tutor.packs.loader import load_packs

    requested = set(bank.released_kcs) if released_kcs is None else set(released_kcs)
    reviewed = tuple(
        sorted(
            (
                kc_id,
                tuple(sorted(misconception.id for misconception in pack.misconceptions)),
            )
            for kc_id, pack in load_packs().items()
            if pack.review_status == ReviewStatus.HUMAN_APPROVED
        )
    )
    return list(
        _cached_validation(
            bank.model_dump_json(),
            graph.model_dump_json(),
            tuple(sorted(requested)),
            reviewed,
        )
    )
