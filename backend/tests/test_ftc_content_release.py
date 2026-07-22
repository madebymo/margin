"""Qualification tests for the pending six-KC Fundamental Theorem wave."""

from __future__ import annotations

import json
from collections import Counter

import pytest
import sympy

from tutor.content.ftc_release import (
    AUTHOR,
    COMPILER_VERSION,
    DEFAULT_BANK_PATH,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_PEDAGOGY_MANIFEST_PATH,
    DEFAULT_PEDAGOGY_SOURCE_PATH,
    DEFAULT_SOURCE_PATH,
    EXPECTED_CLOSURE,
    PRECEDING_BANK_PATHS,
    TARGET_KCS,
    canonical_submission,
    compile_release_inventory,
    draft_pedagogy_review_manifest,
    draft_review_manifest,
    family_digest,
    load_manifest,
    load_source,
    validate_inventory_separation,
    validate_pedagogy_item_separation,
)
from tutor.content.item_bank import load_item_bank
from tutor.graph.service import ancestor_subgraph
from tutor.packs.review_compiler import (
    load_review_manifest as load_pedagogy_reviews,
    load_source_document as load_pedagogy_source,
    source_digest as pedagogy_source_digest,
    validate_review_bundle,
)
from tutor.schemas.assessment import (
    AntiderivativeAnswerSpec,
    AssessmentSurface,
    BlankPromptSegment,
    ChoiceAnswerSpec,
    GuidedMappingSpec,
    GuidedSliderSpec,
    ItemBankDocument,
    MathPromptSegment,
    PlotPromptSegment,
    PromptSemanticRole,
    TablePromptSegment,
)
from tutor.schemas.common import ReviewStatus
from tutor.schemas.content_authoring import ReviewDecision
from tutor.schemas.pedagogy_authoring import PedagogyReviewDecision
from tutor.seed.load_seed import load_graph
from tutor.verify.checker import VerificationStatus, verify_answer


@pytest.fixture(scope="module")
def source():
    return load_source()


@pytest.fixture(scope="module")
def manifest():
    return load_manifest()


@pytest.fixture(scope="module")
def graph():
    return load_graph()


@pytest.fixture(scope="module")
def compiled(source, manifest, graph):
    return compile_release_inventory(source, manifest, graph)


def _tuple(values) -> str:
    return "(" + ", ".join(str(value) for value in values) + ")"


def _polynomial_value(polynomial, value: int) -> int:
    return sum(
        term.coefficient * value**term.exponent for term in polynomial.terms
    )


def _polynomial_expression(polynomial) -> str:
    x = sympy.Symbol("x")
    expression = sum(
        term.coefficient * x**term.exponent for term in polynomial.terms
    )
    return str(sympy.expand(expression)) + " + C"


def _independent_oracle(task) -> str:
    kind = task.kind
    if kind == "graph_point_value":
        point = task.graph.points[task.point_index]
        return _tuple((point.x, point.y))
    if kind == "graph_intercepts":
        x_intercept = next(point.x for point in task.graph.points if point.y == 0)
        y_intercept = next(point.y for point in task.graph.points if point.x == 0)
        return _tuple((x_intercept, y_intercept))
    if kind == "graph_slope":
        left = task.graph.points[task.segment_index]
        right = task.graph.points[task.segment_index + 1]
        rise = right.y - left.y
        run = right.x - left.x
        return _tuple((rise, run, rise // run))
    if kind == "graph_behavior":
        behavior = task.behavior
        selected = []
        for index, (left, right) in enumerate(
            zip(task.graph.points, task.graph.points[1:])
        ):
            rise = right.y - left.y
            if behavior == "increasing" and rise > 0:
                selected.append(index)
            elif behavior == "decreasing" and rise < 0:
                selected.append(index)
        lower = task.graph.points[selected[0]].x
        upper = task.graph.points[selected[-1] + 1].x
        return f"({lower}, {upper})"
    if kind in {"graph_ordered_read", "graph_guided_mapping"}:
        return _tuple(task.graph.points[index].y for index in task.point_indices)
    if kind == "area_rectangle":
        return str(task.region.width * task.region.height)
    if kind == "area_triangle":
        return str(task.region.base * task.region.height // 2)
    if kind in {"area_composite", "area_guided_slider"}:
        parts = [
            *(region.width * region.height for region in task.region.rectangles),
            *(region.base * region.height // 2 for region in task.region.triangles),
        ]
        return str(sum(parts))
    if kind == "area_missing_height":
        divisor = 1 if task.shape == "rectangle" else 2
        area = task.width_or_base * task.height // divisor
        return _tuple((task.width_or_base, area, task.height))
    if kind == "area_ordered_parts":
        rectangle = task.rectangle.width * task.rectangle.height
        triangle = task.triangle.base * task.triangle.height // 2
        return _tuple((rectangle, triangle, rectangle + triangle))
    if kind in {"riemann_left", "riemann_right"}:
        heights = task.table.values[:-1] if kind == "riemann_left" else task.table.values[1:]
        return str(task.table.width * sum(heights))
    if kind == "riemann_midpoint":
        return str(task.table.width * sum(task.table.values))
    if kind == "riemann_compare":
        left = task.table.width * sum(task.table.values[:-1])
        right = task.table.width * sum(task.table.values[1:])
        return _tuple((left, right))
    if kind == "riemann_missing_height":
        known = sum(task.known_heights)
        all_heights = known + task.missing_height
        return _tuple((all_heights, known, task.missing_height))
    if kind == "riemann_contributions":
        contributions = tuple(task.width * height for height in task.heights)
        return _tuple((*contributions, sum(contributions)))
    if kind == "riemann_guided_mapping":
        return str(task.table.width * sum(task.table.values[:-1]))
    if kind == "definite_orientation":
        return str(-task.forward_value)
    if kind == "definite_additivity":
        return str(task.left_value + task.right_value)
    if kind == "definite_missing_piece":
        return str(task.total_value - task.left_value)
    if kind == "definite_signed_regions":
        return str(sum(task.signed_areas))
    if kind == "definite_interpretation":
        return _tuple((task.lower, task.upper, task.value))
    if kind == "definite_two_orientations":
        return _tuple((task.forward_value, -task.forward_value))
    if kind == "definite_guided_mapping":
        return _tuple(
            (task.left_value, task.right_value, task.left_value + task.right_value)
        )
    if kind.startswith("antiderivative_"):
        return _polynomial_expression(task.polynomial)
    if kind == "ftc_ordered_intervals":
        return _tuple(
            _polynomial_value(task.polynomial, upper)
            - _polynomial_value(task.polynomial, lower)
            for lower, upper in (task.first_bounds, task.second_bounds)
        )
    if kind.startswith("ftc_"):
        value = _polynomial_value(task.polynomial, task.upper) - _polynomial_value(
            task.polynomial, task.lower
        )
        return str(-value if kind == "ftc_reversed" else value)
    raise AssertionError(f"missing independent oracle for {kind}")


def test_packaged_manifests_are_exact_pending_drafts(source, manifest):
    assert manifest == draft_review_manifest(source)
    assert source.author == AUTHOR
    assert source.released_kcs == []
    assert len(source.families) == len(manifest.entries) == 78
    reviews = {
        (entry.blueprint_id, entry.revision): entry for entry in manifest.entries
    }
    for family in source.families:
        review = reviews[(family.blueprint_id, family.revision)]
        assert review.source_digest == family_digest(source, family)
        assert review.decision == ReviewDecision.PENDING
        assert review.reviewed_by is None
        assert review.reviewed_at is None

    pedagogy = load_pedagogy_source(DEFAULT_PEDAGOGY_SOURCE_PATH)
    pedagogy_reviews = load_pedagogy_reviews(DEFAULT_PEDAGOGY_MANIFEST_PATH)
    assert pedagogy_reviews == draft_pedagogy_review_manifest(pedagogy)
    assert len(pedagogy_reviews.entries) == 6
    review_by_source = {
        (entry.source_id, entry.revision): entry
        for entry in pedagogy_reviews.entries
    }
    for pack in pedagogy.pack_sources:
        review = review_by_source[(pack.source_id, pack.revision)]
        assert review.source_digest == pedagogy_source_digest(pack)
        assert review.decision == PedagogyReviewDecision.PENDING
        assert review.reviewed_by is None
        assert review.reviewed_at is None


def test_every_family_matches_an_independent_truth_oracle(source, compiled):
    bank, _report = compiled
    items = {item.item_id: item for item in bank.items}
    for family in source.families:
        verdict = verify_answer(
            items[family.item_id].answer,
            _independent_oracle(family.task),
            supervised=False,
        )
        assert verdict.status == VerificationStatus.CORRECT, family.item_id


def test_compiled_bank_is_exact_schema_v3_draft_only(compiled):
    bank, report = compiled
    packaged = ItemBankDocument.model_validate_json(
        DEFAULT_BANK_PATH.read_text(encoding="utf-8")
    )

    assert packaged == bank
    assert DEFAULT_BANK_PATH.read_text(encoding="utf-8") == (
        bank.model_dump_json(indent=2) + "\n"
    )
    assert bank.schema_version == 3
    assert bank.released_kcs == []
    assert len(bank.items) == 78
    assert {item.review_status for item in bank.items} == {ReviewStatus.DRAFT}
    assert all(item.provenance.reviewed_by is None for item in bank.items)
    assert all(item.provenance.compiler_version == COMPILER_VERSION for item in bank.items)
    assert all(not isinstance(item.answer, ChoiceAnswerSpec) for item in bank.items)
    assert Counter(item.answer.kind for item in bank.items) == Counter(
        {
            "numeric": 37,
            "ordered_tuple": 25,
            "antiderivative": 13,
            "interval_set": 3,
        }
    )
    antiderivative_items = [
        item for item in bank.items if isinstance(item.answer, AntiderivativeAnswerSpec)
    ]
    assert len(antiderivative_items) == 13
    assert all(item.answer.require_explicit_constant for item in antiderivative_items)
    for item in antiderivative_items:
        without_constant = item.answer.expected.removesuffix("+C")
        verdict = verify_answer(item.answer, without_constant, supervised=False)
        assert verdict.status == VerificationStatus.INCORRECT
        assert verdict.code == "explicit_constant_required"
    assert report.errors == ()
    assert report.answer_pairs_checked == 78 * 77 // 2 == 3003
    assert report.literal_visible_pairs_checked == 78 * 77 == 6006
    assert report.visible_candidate_comparisons_checked > 0


def test_accessible_segments_widgets_hints_and_worked_answers(compiled):
    bank, _report = compiled
    guided = [item for item in bank.items if item.guided_interaction is not None]
    assert Counter(type(item.guided_interaction) for item in guided) == Counter(
        {GuidedMappingSpec: 4, GuidedSliderSpec: 2}
    )
    for item in bank.items:
        assert [hint.revealing for hint in item.hints] == [False, False, True]
        assert item.hints[-1].text.startswith(
            "Reveal the answer and move to a new problem:"
        )
        for segment in item.prompt:
            if isinstance(segment, MathPromptSegment):
                assert segment.spoken_text
            if isinstance(segment, TablePromptSegment):
                assert segment.spoken_text
            if isinstance(segment, PlotPromptSegment):
                assert segment.spoken_text
                assert segment.equivalent_table is not None
    worked = [
        item
        for item in bank.items
        if item.eligible_surfaces == [AssessmentSurface.WORKED_EXAMPLE]
    ]
    assert len(worked) == 6
    for item in worked:
        answers = [
            segment
            for segment in item.prompt
            if isinstance(segment, MathPromptSegment)
            and segment.role == PromptSemanticRole.WORKED_ANSWER
        ]
        assert len(answers) == 1
        assert answers[0].expression == canonical_submission(item.answer)
        assert not any(isinstance(segment, BlankPromptSegment) for segment in item.prompt)


def test_exact_hard_closure_and_cumulative_separation(compiled, graph):
    bank, _report = compiled
    assert ancestor_subgraph(graph, "kc.int.ftc", hard_only=True).node_ids() == set(
        EXPECTED_CLOSURE
    )
    assert {item.kc_id for item in bank.items} == set(TARGET_KCS)
    preceding = [
        item for path in PRECEDING_BANK_PATHS for item in load_item_bank(path).items
    ]
    report = validate_inventory_separation(
        [*preceding, *bank.items],
        graph,
        focus_item_ids={item.item_id for item in bank.items},
    )
    assert report.errors == ()
    assert report.answer_pairs_checked == 221 * 220 // 2 == 24310
    assert report.literal_visible_pairs_checked == 221 * 220 == 48620


def test_pedagogy_bundle_is_bound_and_does_not_leak_item_answers(compiled, graph):
    bank, _report = compiled
    pedagogy = load_pedagogy_source(DEFAULT_PEDAGOGY_SOURCE_PATH)
    reviews = load_pedagogy_reviews(DEFAULT_PEDAGOGY_MANIFEST_PATH)

    assert validate_review_bundle(pedagogy, reviews, graph) is None
    assert validate_pedagogy_item_separation(bank, pedagogy) == ()


def test_generated_assets_make_no_review_or_release_claim():
    source_text = DEFAULT_SOURCE_PATH.read_text(encoding="utf-8")
    manifest_text = DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8")
    bank_text = DEFAULT_BANK_PATH.read_text(encoding="utf-8")
    pedagogy_manifest_text = DEFAULT_PEDAGOGY_MANIFEST_PATH.read_text(encoding="utf-8")

    assert '"released_kcs": []' in source_text
    assert '"released_kcs": []' in bank_text
    assert '"review_status": "draft"' in bank_text
    assert '"reviewed_by": null' in manifest_text
    assert '"reviewed_at": null' in manifest_text
    assert '"decision": "pending"' in manifest_text
    assert '"decision": "pending"' in pedagogy_manifest_text
    assert "human_approved" not in json.dumps(json.loads(source_text))
