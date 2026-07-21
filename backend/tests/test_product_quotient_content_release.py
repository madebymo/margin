"""Qualification tests for the pending Product/Quotient content inventory."""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from dataclasses import replace

import sympy
import pytest

import tutor.content.product_quotient_release as release_module
from tutor.content.item_bank import load_item_bank
from tutor.content.product_quotient_release import (
    COMPILER_VERSION,
    EXPECTED_CONSTRUCT_ORDER,
    EXPECTED_FAMILY_COUNTS,
    TARGET_CLOSURE,
    ProductQuotientCompilationError,
    compile_release_inventory,
    family_digest,
    load_manifest,
    load_source,
    main,
    validate_inventory_separation,
)
from tutor.schemas.assessment import (
    AssessmentHint,
    AssessmentSurface,
    AssessmentTaskKind,
    BlankPromptSegment,
    MathPromptSegment,
    NumericAnswerSpec,
    PromptSemanticRole,
    SymbolicAnswerSpec,
    TextPromptSegment,
)
from tutor.schemas.common import ReviewStatus
from tutor.schemas.content_authoring import ContentReviewManifest
from tutor.schemas.product_quotient_authoring import (
    ExponentCompoundTask,
    ExponentNegativeTask,
    ExponentPowerTask,
    ExponentProductTask,
    ExponentQuotientTask,
    ExponentZeroTask,
    PolynomialDerivativeTask,
    PowerDerivativeTask,
    ProductAtPointTask,
    ProductQuotientBlueprintDocument,
    QuotientAtPointTask,
    RadicalPowerDerivativeTask,
    RationalPowerDerivativeTask,
)
from tutor.seed.load_seed import load_graph
from tutor.verify.checker import VerificationStatus, parse_restricted, verify_answer


def _sleeping_worker(connection, expected_payload, visible_payload) -> None:
    del connection, expected_payload, visible_payload
    time.sleep(60)


def _abnormal_exit_worker(connection, expected_payload, visible_payload) -> None:
    del connection, expected_payload, visible_payload
    os._exit(7)


class _StubbornProcess:
    def __init__(self):
        self.alive = True
        self.terminated = False
        self.killed = False
        self.joins = 0

    def join(self, timeout):
        assert timeout == 1.0
        self.joins += 1

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True
        self.alive = False


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


def _manifest_for_source(
    source,
    manifest,
    *,
    approved=True,
    reviewer="Independent mathematics reviewer",
):
    payload = manifest.model_dump(mode="json")
    families = {
        (family.blueprint_id, family.revision): family
        for family in source.families
    }
    for entry in payload["entries"]:
        identity = (entry["blueprint_id"], entry["revision"])
        entry["source_digest"] = family_digest(source, families[identity])
        if approved:
            entry.update({
                "decision": "approved",
                "reviewed_by": reviewer,
                "reviewed_at": "2026-07-20T16:00:00Z",
            })
        else:
            entry.update({"decision": "pending"})
            entry.pop("reviewed_by", None)
            entry.pop("reviewed_at", None)
    return ContentReviewManifest.model_validate(payload)


def test_packaged_inventory_is_exactly_52_pending_candidate_families(
    source, compiled
):
    bank, report = compiled

    assert source.author == "AI-assisted implementation draft (unreviewed)"
    assert source.released_kcs == []
    assert bank.released_kcs == []
    assert len(source.families) == len(bank.items) == 52
    assert len({family.family_id for family in source.families}) == 52
    assert len({family.item_id for family in source.families}) == 52
    assert {item.review_status for item in bank.items} == {ReviewStatus.DRAFT}
    assert all(item.provenance.reviewed_by is None for item in bank.items)
    assert all(item.provenance.source_id for item in bank.items)
    assert all(item.provenance.source_revision == 1 for item in bank.items)
    assert all(item.provenance.source_digest for item in bank.items)
    assert all(
        item.provenance.compiler_version == COMPILER_VERSION
        for item in bank.items
    )
    assert report.errors == ()
    assert report.answer_pairs_checked == 52 * 51 // 2 == 1326
    assert report.literal_visible_pairs_checked == 52 * 51 == 2652
    assert report.visible_candidate_comparisons_checked == 5541


def test_exact_surface_matrix_exists_for_every_kc(source):
    counts = Counter((family.kc_id, family.surface) for family in source.families)

    assert set(source.target_kcs) == set(TARGET_CLOSURE)
    assert counts == Counter(
        {
            (kc_id, surface): count
            for kc_id in TARGET_CLOSURE
            for surface, count in EXPECTED_FAMILY_COUNTS.items()
        }
    )


def test_exact_construct_and_order_taxonomy_is_explicit(source):
    for kc_id, by_surface in EXPECTED_CONSTRUCT_ORDER.items():
        for surface, expected_constructs in by_surface.items():
            families = sorted(
                (
                    family
                    for family in source.families
                    if family.kc_id == kc_id and family.surface == surface
                ),
                key=lambda family: family.allocation_order,
            )
            assert tuple(family.construct_id for family in families) == (
                expected_constructs
            )
            assert tuple(family.allocation_order for family in families) == tuple(
                range(10, 10 * len(families) + 1, 10)
            )


def test_normal_exponent_lesson_bundle_uses_coherent_compound_tasks(
    source, compiled
):
    expected_ids = {
        (AssessmentSurface.WORKED_EXAMPLE, 10),
        (AssessmentSurface.GUIDED_WIDGET, 10),
        (AssessmentSurface.CHECKIN, 10),
    }
    normal_bundle = [
        family
        for family in source.families
        if family.kc_id == "kc.alg.exponent_rules"
        and (family.surface, family.allocation_order) in expected_ids
    ]

    assert len(normal_bundle) == 3
    assert all(family.construct_id == "exponent.compound" for family in normal_bundle)
    assert all(isinstance(family.task, ExponentCompoundTask) for family in normal_bundle)
    assert all(family.task.negative_magnitude > 0 for family in normal_bundle)
    bank, _report = compiled
    items = {item.item_id: item for item in bank.items}
    for family in normal_bundle:
        item = items[family.item_id]
        assert isinstance(item.answer, SymbolicAnswerSpec)
        assert item.answer.variables == ["z"]
        visible_text = " ".join(
            [hint.text for hint in item.hints]
            + [
                segment.text
                for segment in item.prompt
                if hasattr(segment, "text")
            ]
        )
        assert "power of x" not in visible_text
        assert "z power" in visible_text


def test_exponent_prompts_preserve_nonzero_domain_on_every_affected_surface(
    source, compiled
):
    bank, _report = compiled
    items = {item.item_id: item for item in bank.items}
    domain_restricted_types = (
        ExponentQuotientTask,
        ExponentNegativeTask,
        ExponentZeroTask,
        ExponentCompoundTask,
    )
    affected = [
        family
        for family in source.families
        if isinstance(family.task, domain_restricted_types)
    ]

    assert affected
    assert {family.surface for family in affected} >= {
        AssessmentSurface.DIAGNOSTIC,
        AssessmentSurface.CHECKIN,
        AssessmentSurface.GUIDED_WIDGET,
        AssessmentSurface.CAPSTONE,
        AssessmentSurface.WORKED_EXAMPLE,
    }
    for family in affected:
        instruction = " ".join(
            segment.text
            for segment in items[family.item_id].prompt
            if isinstance(segment, TextPromptSegment)
            and segment.role == PromptSemanticRole.INSTRUCTION
        )
        assert f"Assume {family.task.base} is non-zero." in instruction


def test_power_diagnosis_spans_broad_scope_before_easy_integer_repetition(source):
    diagnostics = sorted(
        (
            family
            for family in source.families
            if family.kc_id == "kc.der.power_rule"
            and family.surface == AssessmentSurface.DIAGNOSTIC
        ),
        key=lambda family: family.allocation_order,
    )

    assert [family.construct_id for family in diagnostics] == [
        "power.rational",
        "power.reciprocal_radical",
        "power.negative_integer",
        "power.positive_integer",
    ]
    assert isinstance(diagnostics[0].task, RationalPowerDerivativeTask)
    assert isinstance(diagnostics[1].task, RadicalPowerDerivativeTask)
    assert diagnostics[1].task.form == "reciprocal_sqrt"
    assert isinstance(diagnostics[2].task, PowerDerivativeTask)
    assert diagnostics[2].task.exponent < 0

    checkins = sorted(
        (
            family
            for family in source.families
            if family.kc_id == "kc.der.power_rule"
            and family.surface == AssessmentSurface.CHECKIN
        ),
        key=lambda family: family.allocation_order,
    )
    assert [family.construct_id for family in checkins[:2]] == [
        "power.rational",
        "power.reciprocal_radical",
    ]


def test_product_quotient_items_use_only_opaque_point_data_and_numeric_truth(
    source, compiled
):
    bank, _report = compiled
    target_families = [
        family
        for family in source.families
        if family.kc_id == "kc.der.product_quotient"
    ]
    target_items = {
        item.item_id: item
        for item in bank.items
        if item.kc_id == "kc.der.product_quotient"
    }

    assert [family.construct_id for family in target_families[:2]] == [
        "product_quotient.product_at_point",
        "product_quotient.quotient_at_point",
    ]
    for family in target_families:
        assert isinstance(family.task, (ProductAtPointTask, QuotientAtPointTask))
        item = target_items[family.item_id]
        assert isinstance(item.answer, NumericAnswerSpec)
        assert item.answer.tolerance == 0
        assert item.task_kind == AssessmentTaskKind.SOLVE
        givens = [
            segment.expression
            for segment in item.prompt
            if isinstance(segment, MathPromptSegment)
            and segment.role == PromptSemanticRole.GIVEN
        ]
        assert len(givens) == 5
        assert all("^" not in given for given in givens)
        assert any("f'(" in given for given in givens)
        assert any("g'(" in given for given in givens)


def test_truth_is_compiler_derived_not_stored_as_freeform_source(source):
    raw = json.loads(release_module.DEFAULT_SOURCE_PATH.read_text(encoding="utf-8"))

    def keys(value):
        if isinstance(value, dict):
            return set(value) | {key for child in value.values() for key in keys(child)}
        if isinstance(value, list):
            return {key for child in value for key in keys(child)}
        return set()

    assert "expected" not in keys(raw)
    assert "answer" not in keys(raw)
    assert all(family.task is not None for family in source.families)


def test_compiler_derives_representative_truth_for_each_typed_builder(compiled):
    bank, _report = compiled
    expected_by_id = {
        item.item_id: item.answer.expected
        for item in bank.items
    }

    assert expected_by_id["item.pqv1.exponent_rules.diagnostic.01"] == "z^5"
    assert expected_by_id["item.pqv1.exponent_rules.checkin.01"] == "z^13"
    assert (
        expected_by_id["item.pqv1.power_rule.diagnostic.01"]
        == "(15/4)*x^(-1/4)"
    )
    assert expected_by_id["item.pqv1.power_rule.diagnostic.02"] == (
        "-3/(2*x*sqrt(x))"
    )
    assert expected_by_id["item.pqv1.power_rule.diagnostic.04"] == "28*x^3"
    assert (
        expected_by_id["item.pqv1.sum_constant_rules.diagnostic.01"]
        == "6*x^2+10*x"
    )
    assert expected_by_id["item.pqv1.product_quotient.diagnostic.01"] == "22"
    assert expected_by_id["item.pqv1.product_quotient.diagnostic.02"] == "7/9"


def test_every_derived_answer_is_parseable_under_its_exact_contract(compiled):
    bank, _report = compiled

    for item in bank.items:
        assert isinstance(item.answer, (SymbolicAnswerSpec, NumericAnswerSpec))
        verdict = verify_answer(item.answer, item.answer.expected, supervised=True)
        assert verdict.status == VerificationStatus.CORRECT, (item.item_id, verdict)


def _independent_truth(task):
    x = sympy.Symbol("x")
    z = sympy.Symbol("z")
    if isinstance(task, ExponentProductTask):
        return z ** (task.left_exponent + task.right_exponent)
    if isinstance(task, ExponentQuotientTask):
        return z ** (task.numerator_exponent - task.denominator_exponent)
    if isinstance(task, ExponentPowerTask):
        return z ** (task.inner_exponent * task.outer_exponent)
    if isinstance(task, ExponentNegativeTask):
        return z ** (-task.magnitude)
    if isinstance(task, ExponentZeroTask):
        return sympy.Integer(1)
    if isinstance(task, ExponentCompoundTask):
        exponent = (
            -task.negative_magnitude
            + task.inner_exponent * task.outer_exponent
            + task.product_exponent
            - task.denominator_exponent
        )
        return z**exponent
    if isinstance(task, PowerDerivativeTask):
        return task.coefficient * task.exponent * x ** (task.exponent - 1)
    if isinstance(task, RationalPowerDerivativeTask):
        exponent = sympy.Rational(task.numerator, task.denominator)
        return task.coefficient * exponent * x ** (exponent - 1)
    if isinstance(task, RadicalPowerDerivativeTask):
        if task.form == "sqrt":
            return sympy.Rational(task.coefficient, 2) / sympy.sqrt(x)
        return -sympy.Rational(task.coefficient, 2) / (x * sympy.sqrt(x))
    if isinstance(task, PolynomialDerivativeTask):
        return sum(
            term.coefficient * term.exponent * x ** (term.exponent - 1)
            for term in task.polynomial.terms
            if term.exponent > 0
        )
    if isinstance(task, ProductAtPointTask):
        data = task.data
        return sympy.Integer(
            data.f_derivative * data.g_value
            + data.f_value * data.g_derivative
        )
    if isinstance(task, QuotientAtPointTask):
        data = task.data
        return sympy.Rational(
            data.f_derivative * data.g_value
            - data.f_value * data.g_derivative,
            data.g_value**2,
        )
    raise AssertionError(type(task).__name__)


def test_every_derived_answer_matches_an_independent_truth_oracle(
    source, compiled
):
    """Independently derive all 52 truths from source parameters."""
    bank, _report = compiled
    items = {item.item_id: item for item in bank.items}

    for family in source.families:
        item = items[family.item_id]
        expected = parse_restricted(
            item.answer.expected,
            allowed_variables={"x", "z"},
            allowed_functions={"sqrt"},
            allowed_assignment_lhs=None,
        )
        truth = _independent_truth(family.task)
        assert sympy.cancel(expected - truth) == 0, family.item_id


def test_prompt_shape_separates_worked_examples_from_independent_items(compiled):
    bank, _report = compiled

    for item in bank.items:
        blanks = sum(isinstance(segment, BlankPromptSegment) for segment in item.prompt)
        if item.eligible_surfaces == [AssessmentSurface.WORKED_EXAMPLE]:
            assert blanks == 0
            assert any(
                isinstance(segment, MathPromptSegment)
                and segment.role == PromptSemanticRole.WORKED_ANSWER
                for segment in item.prompt
            )
        else:
            assert blanks == 1
            visible_before_hints = " ".join(
                getattr(segment, "text", getattr(segment, "expression", ""))
                for segment in item.prompt
            )
            assert item.answer.expected not in visible_before_hints
        assert not item.hints[0].revealing
        assert not item.hints[1].revealing
        assert item.hints[2].revealing


def test_manifest_binds_every_exact_family_and_remains_pending(source, manifest):
    reviews = {
        (entry.blueprint_id, entry.revision): entry for entry in manifest.entries
    }

    assert manifest.compiler_version == COMPILER_VERSION
    assert len(reviews) == 52
    for family in source.families:
        review = reviews[(family.blueprint_id, family.revision)]
        assert review.source_digest == family_digest(source, family)
        assert review.decision.value == "pending"
        assert review.reviewed_by is None
        assert review.reviewed_at is None


def test_digest_binds_every_release_and_promotion_input(source):
    family = source.families[0]
    baseline = family_digest(source, family)

    for field, changed in (
        ("blueprint_version", "product-quotient-release-v1.0.1"),
        ("output_bank_version", "draft-product-quotient-release-v1.0.1"),
        ("graph_version", source.graph_version + 1),
        ("authoring_source", source.authoring_source + "-changed"),
        ("author", "A different unreviewed author"),
        ("schema_version", source.schema_version + 1),
    ):
        assert family_digest(source.model_copy(update={field: changed}), family) != baseline

    promoted = source.model_copy(update={"released_kcs": list(source.target_kcs)})
    assert family_digest(promoted, family) != baseline


def test_any_source_math_change_invalidates_the_bound_review(source, manifest, graph):
    payload = source.model_dump(mode="json")
    payload["families"][0]["task"]["left_exponent"] += 1
    changed = ProductQuotientBlueprintDocument.model_validate(payload)

    with pytest.raises(ProductQuotientCompilationError, match="review digest mismatch"):
        compile_release_inventory(changed, manifest, graph)


def test_compiler_rendering_drift_invalidates_bound_reviews(
    monkeypatch, source, manifest, graph
):
    original = release_module._derive_task

    def drifted_rendering(task):
        derived = original(task)
        return replace(
            derived,
            conceptual_hint=derived.conceptual_hint + " Unversioned rendering drift.",
        )

    monkeypatch.setattr(release_module, "_derive_task", drifted_rendering)

    with pytest.raises(ProductQuotientCompilationError, match="review digest mismatch"):
        compile_release_inventory(source, manifest, graph)


def test_schema_rejects_a_freeform_truth_escape_hatch(source):
    payload = source.model_dump(mode="json")
    payload["families"][0]["task"]["expected"] = "unreviewed truth"

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        ProductQuotientBlueprintDocument.model_validate(payload)


def test_compiler_rejects_construct_metadata_that_disagrees_with_typed_task(
    source, manifest, graph
):
    payload = source.model_dump(mode="json")
    payload["families"][0]["construct_id"] = "exponent.zero"
    changed = ProductQuotientBlueprintDocument.model_validate(payload)
    changed_manifest = _manifest_for_source(changed, manifest, approved=False)

    with pytest.raises(ProductQuotientCompilationError, match="does not match"):
        compile_release_inventory(changed, changed_manifest, graph)


def test_schema_rejects_noncanonical_polynomials(source):
    payload = source.model_dump(mode="json")
    polynomial = next(
        family
        for family in payload["families"]
        if family["task"]["kind"] == "polynomial_derivative"
    )
    polynomial["task"]["polynomial"]["terms"].reverse()

    with pytest.raises(ValueError, match="descending exponents"):
        ProductQuotientBlueprintDocument.model_validate(payload)


def test_release_promotion_requires_complete_independent_review(
    source, manifest, graph
):
    promoted = source.model_copy(update={"released_kcs": list(source.target_kcs)})
    promoted_manifest = _manifest_for_source(
        promoted,
        manifest,
        approved=False,
    )

    with pytest.raises(
        ProductQuotientCompilationError,
        match="without complete independent approval",
    ):
        compile_release_inventory(promoted, promoted_manifest, graph)


def test_complete_approval_can_only_promote_the_atomic_hard_closure(
    source, manifest, graph
):
    partial = source.model_copy(update={"released_kcs": [source.target_kcs[0]]})
    partial_approval = _manifest_for_source(partial, manifest)
    with pytest.raises(ProductQuotientCompilationError, match="atomic"):
        compile_release_inventory(partial, partial_approval, graph)

    promoted = source.model_copy(update={"released_kcs": list(source.target_kcs)})
    promoted_approval = _manifest_for_source(promoted, manifest)
    bank, report = compile_release_inventory(promoted, promoted_approval, graph)
    assert set(bank.released_kcs) == set(TARGET_CLOSURE)
    assert {item.review_status for item in bank.items} == {
        ReviewStatus.HUMAN_APPROVED
    }
    assert report.errors == ()


def test_declared_author_cannot_approve_their_own_inventory(
    source, manifest, graph
):
    self_reviewed = _manifest_for_source(
        source,
        manifest,
        reviewer=source.author,
    )

    with pytest.raises(ProductQuotientCompilationError, match="cannot approve"):
        compile_release_inventory(source, self_reviewed, graph)


def test_exact_graph_hard_closure_is_pinned(source, manifest, graph):
    payload = graph.model_dump(mode="json")
    payload["edges"].append(
        {
            "from_kc": "kc.alg.arith_fractions",
            "to_kc": "kc.alg.exponent_rules",
            "type": "hard",
            "rationale": "Test-only closure expansion.",
        }
    )
    expanded = type(graph).model_validate(payload)

    with pytest.raises(ProductQuotientCompilationError, match="hard closure changed"):
        compile_release_inventory(source, manifest, expanded)


def test_exhaustive_gate_detects_equivalent_answers(compiled, graph):
    bank, _report = compiled
    changed = list(bank.items)
    changed[1] = changed[1].model_copy(update={"answer": changed[0].answer})

    report = validate_inventory_separation(changed, graph)

    assert any("expected answer reused across families" in error for error in report.errors)
    assert report.answer_pairs_checked == 1326


def test_exhaustive_gate_detects_visible_cross_family_leakage(compiled, graph):
    bank, _report = compiled
    changed = list(bank.items)
    source_item = changed[0]
    target = changed[1]
    changed[0] = source_item.model_copy(
        update={
            "hints": [
                AssessmentHint(
                    text=f"An unrelated upcoming answer is {target.answer.expected}."
                ),
                source_item.hints[1],
                source_item.hints[2],
            ]
        }
    )

    report = validate_inventory_separation(changed, graph)

    assert any(
        source_item.item_id in error and target.item_id in error
        for error in report.errors
    )


def test_exhaustive_gate_detects_equivalent_math_embedded_in_visible_text(
    compiled, graph
):
    bank, _report = compiled
    changed = list(bank.items)
    source_index, source_item = next(
        (index, item)
        for index, item in enumerate(changed)
        if isinstance(item.answer, NumericAnswerSpec)
    )
    target = next(item for item in changed if item.answer.expected == "z^5")
    changed[source_index] = source_item.model_copy(
        update={
            "hints": [
                AssessmentHint(text="An unrelated result is z^2*z^3."),
                source_item.hints[1],
                source_item.hints[2],
            ]
        }
    )
    report = validate_inventory_separation(changed, graph)

    assert any(
        source_item.item_id in error
        and target.item_id in error
        and "visible math is equivalent" in error
        for error in report.errors
    )


def test_exhaustive_gate_detects_reused_numeric_truth(compiled, graph):
    bank, _report = compiled
    changed = list(bank.items)
    numeric_indexes = [
        index
        for index, item in enumerate(changed)
        if isinstance(item.answer, NumericAnswerSpec)
    ]
    left_index, right_index = numeric_indexes[:2]
    changed[right_index] = changed[right_index].model_copy(
        update={"answer": changed[left_index].answer}
    )

    report = validate_inventory_separation(changed, graph)

    assert any("expected answer reused" in error for error in report.errors)


def test_exhaustive_gate_detects_adjacent_implicit_group_leakage(compiled, graph):
    bank, _report = compiled
    changed = list(bank.items)
    source_item = changed[0]
    target = next(item for item in changed if item.answer.expected == "28*x^3")
    changed[0] = source_item.model_copy(
        update={
            "hints": [
                AssessmentHint(text="An unrelated result is (4*x^2)(7*x)."),
                source_item.hints[1],
                source_item.hints[2],
            ]
        }
    )

    report = validate_inventory_separation(changed, graph)

    assert any(
        source_item.item_id in error
        and target.item_id in error
        and "visible math is equivalent" in error
        for error in report.errors
    )


def test_separation_worker_timeout_is_fail_closed_and_reaped(
    monkeypatch, compiled
):
    bank, _report = compiled
    before = {child.pid for child in release_module.multiprocessing.active_children()}
    monkeypatch.setattr(
        release_module,
        "_separation_math_worker",
        _sleeping_worker,
    )

    with pytest.raises(ProductQuotientCompilationError, match="timed out"):
        release_module._supervised_math_separation(
            list(bank.items[:2]),
            timeout_seconds=0.05,
        )

    after = {
        child.pid
        for child in release_module.multiprocessing.active_children()
        if child.is_alive()
    }
    assert after <= before


def test_separation_worker_abnormal_exit_is_fail_closed(monkeypatch, compiled):
    bank, _report = compiled
    monkeypatch.setattr(
        release_module,
        "_separation_math_worker",
        _abnormal_exit_worker,
    )

    with pytest.raises(ProductQuotientCompilationError, match="without a result"):
        release_module._supervised_math_separation(
            list(bank.items[:2]),
            timeout_seconds=2.0,
        )


def test_stubborn_separation_worker_is_killed_after_terminate():
    process = _StubbornProcess()

    release_module._reap_worker(process)

    assert process.terminated is True
    assert process.killed is True
    assert process.joins == 3


def test_pending_inventory_does_not_replace_the_active_bank(compiled):
    bank, _report = compiled
    active = load_item_bank()

    assert active.bank_version != bank.bank_version
    assert all(not item.item_id.startswith("item.pqv1.") for item in active.items)


def test_cli_check_reports_actual_qualification_work(capsys):
    assert main(["--check"]) == 0
    output = capsys.readouterr().out
    assert "52 total families, 52 draft, 0 approved" in output
    assert "1326 answer comparisons" in output
    assert "5541 visible candidate comparisons" in output
    assert "2652 literal cross-family scans" in output
    assert "released KCs=0" in output


def test_atomic_output_is_complete_and_schema_valid(tmp_path):
    output = tmp_path / "compiled-bank.json"

    assert main(["--check", "--out", str(output)]) == 0

    payload = output.read_text(encoding="utf-8")
    assert payload.endswith("\n")
    bank = release_module.ItemBankDocument.model_validate_json(payload)
    assert len(bank.items) == 52


def test_failed_atomic_replace_preserves_existing_destination(
    monkeypatch, tmp_path, compiled
):
    bank, _report = compiled
    output = tmp_path / "compiled-bank.json"
    output.write_text("existing-good-bytes\n", encoding="utf-8")

    def fail_replace(source, destination):
        del source, destination
        raise OSError("simulated replace failure")

    monkeypatch.setattr(release_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        release_module._atomic_write_bank(output, bank)

    assert output.read_text(encoding="utf-8") == "existing-good-bytes\n"
    assert not list(tmp_path.glob(".compiled-bank.json.*.tmp"))
