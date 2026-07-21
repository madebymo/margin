"""Qualification tests for the pending four-KC Solve Quadratics wave."""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from fractions import Fraction

import pytest
import sympy
from pydantic import ValidationError

import tutor.content.solve_quadratics_release as solve_release
from tutor.content.item_bank import load_item_bank
from tutor.content.solve_quadratics_release import (
    AUTHOR,
    COMPILER_VERSION,
    DEFAULT_BANK_PATH,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_PEDAGOGY_MANIFEST_PATH,
    DEFAULT_PEDAGOGY_SOURCE_PATH,
    EXPECTED_CONSTRUCT_ORDER,
    EXPECTED_FAMILY_COUNTS,
    TARGET_KCS,
    _canonical_submission,
    _refuse_completed_review_overwrite,
    compile_release_inventory,
    derive_task,
    draft_review_manifest,
    family_digest,
    load_manifest,
    load_source,
    main,
    validate_inventory_separation,
)
from tutor.orchestrator.session_v2 import SessionOrchestratorV2
from tutor.schemas.assessment import (
    AssessmentSurface,
    BlankPromptSegment,
    ChoiceAnswerSpec,
    GuidedMappingSpec,
    ItemBankDocument,
    MathPromptSegment,
    OrderedTupleAnswerSpec,
    PromptSemanticRole,
    TablePromptSegment,
)
from tutor.schemas.common import EdgeType, ReviewStatus
from tutor.schemas.content_authoring import ContentReviewManifest, ReviewDecision
from tutor.schemas.pedagogy_authoring import (
    PedagogyReviewDecision,
    PedagogyReviewManifest,
    PedagogySourceDocument,
)
from tutor.schemas.solve_quadratics_authoring import (
    CubicPolynomial,
    SolveQuadraticsBlueprintDocument,
)
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


def _poly(polynomial: CubicPolynomial) -> sympy.Expr:
    x = sympy.Symbol("x")
    return (
        polynomial.cubic * x**3
        + polynomial.quadratic * x**2
        + polynomial.linear * x
        + polynomial.constant
    )


def _expanded_product(product) -> sympy.Expr:
    x = sympy.Symbol("x")
    return sympy.expand(
        (product.left_linear * x + product.left_constant)
        * (product.right_linear * x + product.right_constant)
    )


def _tuple(values) -> str:
    return "(" + ", ".join(str(value) for value in values) + ")"


def _portfolio_oracle(task) -> str:
    core = task.core
    values = (
        sympy.expand(_poly(core.add_left) + _poly(core.add_right)),
        sympy.expand(_poly(core.subtract_left) - _poly(core.subtract_right)),
        _expanded_product(core.expand),
    )
    if task.kind == "polynomial_mixed_expression":
        return str(sympy.expand(values[0] - values[1] + values[2]))
    if task.kind == "polynomial_coefficient_audit":
        x = sympy.Symbol("x")
        return _tuple(
            (
                sympy.Poly(values[0], x).coeff_monomial(x**task.add_power),
                sympy.Poly(values[1], x).coeff_monomial(x**task.subtract_power),
                sympy.Poly(values[2], x).coeff_monomial(x**task.expand_power),
            )
        )
    if task.kind == "polynomial_reverse_portfolio":
        return _tuple(
            (
                _poly(core.add_right),
                _poly(core.subtract_right),
                core.expand.right_constant,
            )
        )
    if task.kind == "polynomial_guided_match":
        return _tuple((*values, sympy.expand(values[0] - values[1] + values[2])))
    return _tuple(values)


def _factoring_oracle(task) -> str:
    core = task.core
    values = (
        core.gcf.common_coefficient,
        core.gcf.common_exponent,
        core.gcf.residual_linear,
        core.gcf.residual_constant,
        core.monic.lower_root,
        core.monic.upper_root,
        core.difference.scale,
        core.difference.magnitude,
    )
    if task.kind == "factoring_guided_match":
        return _tuple((values[0], values[1], values[4], values[7]))
    if task.kind == "factoring_verification_portfolio":
        x = task.check_value
        checks = (
            core.gcf.common_coefficient
            * x**core.gcf.common_exponent
            * (core.gcf.residual_linear * x + core.gcf.residual_constant),
            (x - core.monic.lower_root) * (x - core.monic.upper_root),
            core.difference.scale
            * (x - core.difference.magnitude)
            * (x + core.difference.magnitude),
        )
        return _tuple((*values, *checks))
    return _tuple(values)


def _linear_coefficients(task) -> tuple[int, int, int, int]:
    if task.kind in {"linear_two_sided", "linear_reversed", "linear_correction", "linear_guided_balance"}:
        equation = task.equation
        return (
            equation.left_coefficient,
            equation.left_constant,
            equation.right_coefficient,
            equation.right_constant,
        )
    if task.kind == "linear_one_side":
        if task.variable_side == "left":
            return task.coefficient, task.constant, 0, task.target
        return 0, task.target, task.coefficient, task.constant
    if task.kind == "linear_distributed":
        return (
            task.multiplier * task.inner_coefficient,
            task.multiplier * task.inner_constant + task.added_constant,
            task.right_coefficient,
            task.right_constant,
        )
    if task.kind == "linear_grouped":
        return (
            task.left_multiplier,
            task.left_multiplier * task.left_shift,
            task.right_multiplier,
            task.right_multiplier * task.right_shift,
        )
    if task.kind == "linear_double_distributed":
        return (
            task.left_multiplier * task.left_coefficient,
            task.left_multiplier * task.left_constant,
            task.right_multiplier * task.right_coefficient,
            task.right_multiplier * task.right_constant,
        )
    raise AssertionError(f"missing linear oracle for {task.kind}")


def _linear_oracle(task) -> str:
    left_coefficient, left_constant, right_coefficient, right_constant = (
        _linear_coefficients(task)
    )
    difference = left_coefficient - right_coefficient
    numerator = right_constant - left_constant
    solution = Fraction(numerator, difference)
    assert solution.denominator == 1
    if task.kind == "linear_guided_balance":
        check = left_coefficient * solution.numerator + left_constant
        return _tuple((difference, numerator, solution.numerator, check))
    return str(solution.numerator)


def _quadratic_oracle(task) -> str:
    if task.kind == "quadratic_sparse_difference":
        roots = (-task.magnitude, task.magnitude)
    elif task.kind == "quadratic_repeated":
        roots = (task.root,)
    else:
        roots = (task.roots.lower, task.roots.upper)
    if task.kind == "quadratic_guided_factor_map":
        return _tuple((-roots[0], -roots[-1], roots[0], roots[-1]))
    return "{" + ", ".join(str(root) for root in dict.fromkeys(roots)) + "}"


def _independent_oracle(task) -> str:
    if task.kind.startswith("polynomial_"):
        return _portfolio_oracle(task)
    if task.kind.startswith("factoring_"):
        return _factoring_oracle(task)
    if task.kind.startswith("linear_"):
        return _linear_oracle(task)
    if task.kind.startswith("quadratic_"):
        return _quadratic_oracle(task)
    raise AssertionError(f"missing independent oracle for {task.kind}")


def test_packaged_source_and_manifest_are_canonical_pending_drafts(source, manifest):
    assert manifest == draft_review_manifest(source)
    assert source.author == AUTHOR
    assert source.released_kcs == []
    assert len(source.families) == len(manifest.entries) == 52
    reviews = {
        (entry.blueprint_id, entry.revision): entry for entry in manifest.entries
    }
    for family in source.families:
        review = reviews[(family.blueprint_id, family.revision)]
        assert review.source_digest == family_digest(source, family)
        assert review.decision == ReviewDecision.PENDING
        assert review.reviewed_by is None
        assert review.reviewed_at is None


def test_source_contract_forbids_free_form_truth_fields(source):
    payload = source.model_dump(mode="json")
    payload["families"][0]["task"]["expected_answer"] = "fabricated"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SolveQuadraticsBlueprintDocument.model_validate(payload)
    assert "expected_wrong" not in json.dumps(payload["families"])


def test_exact_surface_matrix_and_within_surface_construct_independence(source):
    assert set(source.target_kcs) == set(TARGET_KCS)
    assert Counter((family.kc_id, family.surface) for family in source.families) == Counter(
        {
            (kc_id, surface): count
            for kc_id in TARGET_KCS
            for surface, count in EXPECTED_FAMILY_COUNTS.items()
        }
    )
    assert len({family.family_id for family in source.families}) == 52
    assert len({family.item_id for family in source.families}) == 52
    for kc_id in TARGET_KCS:
        for surface in (AssessmentSurface.DIAGNOSTIC, AssessmentSurface.CHECKIN):
            families = [
                family
                for family in source.families
                if family.kc_id == kc_id and family.surface == surface
            ]
            assert len({family.construct_id for family in families}) == len(families)


def test_construct_taxonomy_is_closed_and_ordered(source):
    expected_constructs = set()
    for kc_id, by_surface in EXPECTED_CONSTRUCT_ORDER.items():
        for surface, expected in by_surface.items():
            expected_constructs.update(expected)
            families = sorted(
                (
                    family
                    for family in source.families
                    if family.kc_id == kc_id and family.surface == surface
                ),
                key=lambda family: family.allocation_order,
            )
            assert tuple(family.construct_id for family in families) == expected
    assert {family.construct_id for family in source.families} == expected_constructs
    assert len({family.task.kind for family in source.families}) == 32


def test_family_digest_is_local_stable_and_truth_sensitive(source, monkeypatch):
    family = next(
        family for family in source.families if family.task.kind == "linear_one_side"
    )
    baseline = family_digest(source, family)
    for update in (
        {"blueprint_version": "solve-quadratics-cumulative-v99"},
        {"output_bank_version": "solve-quadratics-bank-v99"},
        {"released_kcs": [family.kc_id]},
        {"target_kcs": [*source.target_kcs, "kc.der.power_rule"]},
    ):
        assert family_digest(source.model_copy(update=update), family) == baseline
    changed_task = family.task.model_copy(update={"target": family.task.target + 3})
    assert family_digest(source, family.model_copy(update={"task": changed_task})) != baseline
    assert family_digest(source.model_copy(update={"author": "another author"}), family) != baseline
    monkeypatch.setattr(solve_release, "COMPILER_VERSION", "compiler-change")
    assert family_digest(source, family) != baseline


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


def test_compiled_bank_is_schema_v3_draft_only_and_exhaustively_separated(compiled):
    bank, report = compiled
    packaged = ItemBankDocument.model_validate_json(
        DEFAULT_BANK_PATH.read_text(encoding="utf-8")
    )
    assert packaged == bank
    assert DEFAULT_BANK_PATH.read_text(encoding="utf-8") == bank.model_dump_json(indent=2) + "\n"
    assert bank.schema_version == 3
    assert bank.released_kcs == []
    assert len(bank.items) == 52
    assert {item.review_status for item in bank.items} == {ReviewStatus.DRAFT}
    assert all(item.provenance.reviewed_by is None for item in bank.items)
    assert all(item.provenance.compiler_version == COMPILER_VERSION for item in bank.items)
    assert all(not isinstance(item.answer, ChoiceAnswerSpec) for item in bank.items)
    assert Counter(item.answer.kind for item in bank.items) == Counter(
        {"ordered_tuple": 25, "numeric": 12, "finite_set": 12, "symbolic": 3}
    )
    assert report.errors == ()
    assert report.answer_pairs_checked == 52 * 51 // 2 == 1326
    assert report.literal_visible_pairs_checked == 52 * 51 == 2652
    assert report.visible_candidate_comparisons_checked > 0


def test_prompt_segments_are_accessible_and_student_copy_is_plain(compiled):
    bank, _report = compiled
    forbidden = ("answer-separated", "formative policy", "knowledge component")
    for item in bank.items:
        assert all(
            segment.spoken_text
            for segment in item.prompt
            if isinstance(segment, (MathPromptSegment, TablePromptSegment))
        )
        visible = " ".join(
            [
                *(getattr(segment, "text", "") for segment in item.prompt),
                *(hint.text for hint in item.hints),
            ]
        ).casefold()
        assert not any(term in visible for term in forbidden)


def test_broad_kcs_are_covered_by_each_mastery_bearing_portfolio(source):
    expected = {
        "kc.alg.polynomial_ops": {"add", "subtract", "expand"},
        "kc.alg.factoring": {"gcf", "monic_quadratic", "difference_squares"},
        "kc.alg.solve_linear": {"solve_integer_linear"},
        "kc.alg.solve_quadratic": {"factorable_quadratic_roots"},
    }
    for family in source.families:
        if family.surface in {
            AssessmentSurface.DIAGNOSTIC,
            AssessmentSurface.CHECKIN,
            AssessmentSurface.CAPSTONE,
        }:
            assert set(derive_task(family.task).construct_coverage) == expected[
                family.kc_id
            ]


def test_tasks_cover_signed_sparse_and_structurally_distinct_cases(source):
    linear = [family.task for family in source.families if family.kc_id == "kc.alg.solve_linear"]
    solutions = [_linear_oracle(task) for task in linear if task.kind != "linear_guided_balance"]
    assert any(int(value) < 0 for value in solutions)
    assert any(int(value) > 0 for value in solutions)
    assert max(abs(int(value)) for value in solutions) <= 29
    assert len(set(solutions)) == len(solutions)

    polynomial = [
        family.task for family in source.families if family.kc_id == "kc.alg.polynomial_ops"
    ]
    assert any(
        0 in (
            task.core.add_left.cubic,
            task.core.add_left.quadratic,
            task.core.add_left.linear,
            task.core.add_left.constant,
        )
        for task in polynomial
    )
    assert any(
        task.core.add_left.display_order != "descending" for task in polynomial
    )

    factoring = [
        family.task for family in source.families if family.kc_id == "kc.alg.factoring"
    ]
    assert {task.core.gcf.common_coefficient < 0 for task in factoring} == {False, True}
    assert any(task.core.monic.lower_root == task.core.monic.upper_root for task in factoring)
    assert {task.core.difference.scale < 0 for task in factoring} == {False, True}

    quadratics = [
        family.task
        for family in source.families
        if family.kc_id == "kc.alg.solve_quadratic"
    ]
    root_sets = [set(map(int, _quadratic_oracle(task).strip("{}").split(", "))) for task in quadratics if task.kind != "quadratic_guided_factor_map"]
    assert any(all(root > 0 for root in roots) for roots in root_sets)
    assert any(all(root < 0 for root in roots) for roots in root_sets)
    assert any(min(roots) < 0 < max(roots) for roots in root_sets)
    assert any(len(roots) == 1 for roots in root_sets)


def test_guided_mappings_are_opaque_nonpositional_and_text_equivalent(source, compiled):
    bank, _report = compiled
    families = {family.item_id: family for family in source.families}
    guided = [
        item
        for item in bank.items
        if item.eligible_surfaces == [AssessmentSurface.GUIDED_WIDGET]
    ]
    assert len(guided) == 4
    for item in guided:
        interaction = item.guided_interaction
        assert isinstance(interaction, GuidedMappingSpec)
        assert isinstance(item.answer, OrderedTupleAnswerSpec)
        presentation = interaction.presentation
        assert len(presentation.rows) == len(presentation.options) == 4
        assert len(interaction.scoring.correct_pairs) == 4
        assert all(entry.entry_id.startswith("entry.") for entry in (*presentation.rows, *presentation.options))
        public = presentation.model_dump_json()
        assert "correct_pairs" not in public
        assert "expected" not in public
        positional = dict(
            zip(
                sorted(entry.entry_id for entry in presentation.rows),
                sorted(entry.entry_id for entry in presentation.options),
                strict=True,
            )
        )
        assert positional != dict(interaction.scoring.correct_pairs)
        plan = derive_task(families[item.item_id].task).guided_plan
        assert plan is not None
        assert item.answer.expected == list(plan.fallback_expected)
        assert verify_answer(
            item.answer,
            _canonical_submission(item.answer),
            supervised=False,
        ).status == VerificationStatus.CORRECT
        assert any(isinstance(segment, BlankPromptSegment) for segment in item.prompt)


def test_error_signatures_are_executable_and_use_reviewed_taxonomy(compiled, graph):
    bank, _report = compiled
    pedagogy = PedagogySourceDocument.model_validate_json(
        DEFAULT_PEDAGOGY_SOURCE_PATH.read_text(encoding="utf-8")
    )
    reviewed = {
        pack.kc_id: frozenset(item.id for item in pack.misconceptions)
        for pack in pedagogy.pack_sources
    }
    matcher = object.__new__(SessionOrchestratorV2)
    matcher._reviewed_misconceptions = reviewed
    predecessors = {
        kc_id: {
            edge.from_kc
            for edge in graph.edges
            if edge.type == EdgeType.HARD and edge.to_kc == kc_id
        }
        for kc_id in TARGET_KCS
    }
    signatures = [
        (item, signature)
        for item in bank.items
        for signature in item.error_signatures
    ]
    assert len(signatures) == 116
    for item, signature in signatures:
        assert signature.misconception_id in reviewed[item.kc_id]
        if signature.implicated_prereq is not None:
            assert signature.implicated_prereq in predecessors[item.kc_id]
        assert verify_answer(
            item.answer,
            signature.expected_wrong,
            supervised=False,
        ).status == VerificationStatus.INCORRECT
        assert matcher._match_error_signature(
            item,
            signature.expected_wrong,
            False,
        ) == (signature.implicated_prereq, signature.misconception_id)
        assert matcher._match_error_signature(item, signature.expected_wrong, True) == (
            None,
            None,
        )


def test_worked_examples_have_real_steps_and_one_final_answer(compiled):
    bank, _report = compiled
    worked = [
        item
        for item in bank.items
        if item.eligible_surfaces == [AssessmentSurface.WORKED_EXAMPLE]
    ]
    assert len(worked) == 4
    for item in worked:
        steps = [
            segment
            for segment in item.prompt
            if isinstance(segment, MathPromptSegment)
            and segment.role == PromptSemanticRole.WORKED_STEP
        ]
        answers = [
            segment
            for segment in item.prompt
            if isinstance(segment, MathPromptSegment)
            and segment.role == PromptSemanticRole.WORKED_ANSWER
        ]
        assert len(steps) >= 2
        assert len(answers) == 1
        assert answers[0].expression == _canonical_submission(item.answer)
        assert not any(isinstance(segment, BlankPromptSegment) for segment in item.prompt)


def test_wave_is_separated_from_product_quotient_and_chain_rule(compiled, graph):
    bank, _report = compiled
    preceding = [
        *load_item_bank(
            DEFAULT_BANK_PATH.parent / "item_bank_product_quotient_v2.json"
        ).items,
        *load_item_bank(
            DEFAULT_BANK_PATH.parent / "item_bank_chain_rule_v2.json"
        ).items,
    ]
    report = validate_inventory_separation(
        [*preceding, *bank.items],
        graph,
        focus_item_ids={item.item_id for item in bank.items},
    )
    assert report.errors == ()
    assert report.answer_pairs_checked == 143 * 142 // 2 == 10153
    assert report.literal_visible_pairs_checked == 143 * 142 == 20306


def test_cli_checks_assets_and_bank_output_cannot_rewrite_reviews(tmp_path, capsys):
    before = DEFAULT_MANIFEST_PATH.read_bytes()
    destination = tmp_path / "solve-quadratics-bank.json"
    assert main(["--check", "--out", str(destination)]) == 0
    assert destination.read_bytes() == DEFAULT_BANK_PATH.read_bytes()
    assert DEFAULT_MANIFEST_PATH.read_bytes() == before
    assert "52 families" in capsys.readouterr().out


def test_stale_family_digest_fails_closed(source, manifest, graph):
    payload = manifest.model_dump(mode="json")
    payload["entries"][0]["source_digest"] = "0" * 64
    stale = ContentReviewManifest.model_validate(payload)
    with pytest.raises(ValueError, match="review digest mismatch"):
        compile_release_inventory(source, stale, graph)


def test_regeneration_guards_each_completed_review_manifest_independently(manifest):
    item_payload = manifest.model_dump(mode="json")
    item_payload["entries"][0].update(
        {
            "decision": "approved",
            "reviewed_by": "Independent human reviewer",
            "reviewed_at": datetime(2026, 7, 21, tzinfo=UTC),
        }
    )
    approved_items = ContentReviewManifest.model_validate(item_payload)
    pedagogy = PedagogyReviewManifest.model_validate_json(
        DEFAULT_PEDAGOGY_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    with pytest.raises(ValueError, match="completed item-family review"):
        _refuse_completed_review_overwrite(approved_items, None)

    pedagogy_payload = pedagogy.model_dump(mode="json")
    pedagogy_payload["entries"][0].update(
        {
            "decision": PedagogyReviewDecision.APPROVED,
            "reviewed_by": "Independent human reviewer",
            "reviewed_at": datetime(2026, 7, 21, tzinfo=UTC),
        }
    )
    approved_pedagogy = PedagogyReviewManifest.model_validate(pedagogy_payload)
    with pytest.raises(ValueError, match="completed pedagogy review"):
        _refuse_completed_review_overwrite(None, approved_pedagogy)
