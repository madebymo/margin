"""Qualification tests for the pending three-KC Chain Rule content wave."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pytest
import sympy
import tutor.content.chain_rule_release as chain_rule_release

from tutor.content.chain_rule_release import (
    COMPILER_VERSION,
    EXPECTED_FAMILY_COUNTS,
    TARGET_KCS,
    ChainRuleCompilationError,
    compile_release_inventory,
    derive_task,
    family_digest,
    load_manifest,
    load_source,
    main as chain_release_main,
    validate_inventory_separation,
)
from tutor.content.item_bank import _validate_item_bank_uncached, load_item_bank
from tutor.orchestrator.session_v2 import SessionOrchestratorV2
from tutor.schemas.assessment import (
    AssessmentSurface,
    BlankPromptSegment,
    ChoiceAnswerSpec,
    GuidedMappingSpec,
    GuidedSliderSpec,
    ItemBankDocument,
    MathPromptSegment,
    NumericAnswerSpec,
    OrderedTupleAnswerSpec,
    PlotPromptSegment,
    PromptSemanticRole,
    SymbolicAnswerSpec,
    TablePromptSegment,
)
from tutor.schemas.chain_rule_authoring import (
    AffineCompositionTask,
    AffineFunctionValueTask,
    AffinePowerChainTask,
    ChainAtPointTask,
    ChainCorrectionTask,
    ChainFactorTupleTask,
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
from tutor.schemas.common import EdgeType, ReviewStatus
from tutor.schemas.content_authoring import ReviewDecision
from tutor.schemas.pedagogy_authoring import PedagogySourceDocument
from tutor.seed.load_seed import load_graph
from tutor.verify.checker import VerificationStatus, verify_answer

SEED_DIR = Path(__file__).resolve().parents[1] / "src" / "tutor" / "seed"
BANK_PATH = SEED_DIR / "item_bank_chain_rule_v2.json"
PEDAGOGY_PATH = SEED_DIR / "pedagogy_pack_sources_chain_rule_v2.json"


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


def _assert_no_source_authored_expected(value) -> None:
    if isinstance(value, dict):
        assert "expected" not in value
        assert "expected_wrong" not in value
        for child in value.values():
            _assert_no_source_authored_expected(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_source_authored_expected(child)


def _affine(coefficient: int, constant: int, value):
    return coefficient * value + constant


def _oracle_submission(task) -> str:
    """Independent truth oracle: never reads compiler output or bank truth."""

    x = sympy.Symbol("x")
    if isinstance(task, AffineFunctionValueTask):
        return str(_affine(task.coefficient, task.constant, task.input_value))
    if isinstance(task, QuadraticFunctionValueTask):
        return str(
            task.quadratic * task.input_value**2
            + task.linear * task.input_value
            + task.constant
        )
    if isinstance(task, FunctionExpressionValueTask):
        supplied = task.input_coefficient * x + task.input_constant
        return str(sympy.expand(task.function_coefficient * supplied + task.function_constant))
    if isinstance(task, FunctionTableCombinationTask):
        value = (
            task.left_output + task.right_output
            if task.operation == "sum"
            else task.left_output - task.right_output
        )
        return str(value)
    if isinstance(task, FunctionPlotChangeTask):
        return str(task.end_output - task.start_output)
    if isinstance(task, FunctionOrderedValuesTask):
        values = (
            _affine(task.coefficient, task.constant, task.first_input),
            _affine(task.coefficient, task.constant, task.second_input),
        )
        return f"({values[0]}, {values[1]})"
    if isinstance(task, AffineCompositionTask):
        f = task.f_coefficient * x + task.f_constant
        g = task.g_coefficient * x + task.g_constant
        value = (
            task.f_coefficient * g + task.f_constant
            if task.order == "f_after_g"
            else task.g_coefficient * f + task.g_constant
        )
        return str(sympy.expand(value))
    if isinstance(task, QuadraticOuterCompositionTask):
        inner = task.inner_coefficient * x + task.inner_constant
        return str(sympy.expand(task.outer_scale * inner**2 + task.outer_constant))
    if isinstance(task, CompositionAtPointTask):
        f_at_point = _affine(task.f_coefficient, task.f_constant, task.point)
        g_at_point = _affine(task.g_coefficient, task.g_constant, task.point)
        value = (
            _affine(task.f_coefficient, task.f_constant, g_at_point)
            if task.order == "f_after_g"
            else _affine(task.g_coefficient, task.g_constant, f_at_point)
        )
        return str(value)
    if isinstance(task, CompositionTablePathTask):
        return str(task.f_at_g - task.g_at_point)
    if isinstance(task, CompositionPairedOrdersTask):
        f_at_point = _affine(task.f_coefficient, task.f_constant, task.point)
        g_at_point = _affine(task.g_coefficient, task.g_constant, task.point)
        values = (
            _affine(task.f_coefficient, task.f_constant, g_at_point),
            _affine(task.g_coefficient, task.g_constant, f_at_point),
        )
        return f"({values[0]}, {values[1]})"
    if isinstance(task, CompositionPlotSumTask):
        return str(task.f_at_g + task.g_at_f)
    if isinstance(task, AffinePowerChainTask):
        inner = task.inner_coefficient * x + task.inner_constant
        return str(sympy.diff(inner**task.outer_power, x))
    if isinstance(task, QuadraticPowerChainTask):
        inner = task.quadratic_coefficient * x**2 + task.inner_constant
        return str(sympy.diff(inner**task.outer_power, x))
    if isinstance(task, ChainAtPointTask):
        return str(
            task.outer_power
            * task.inner_value ** (task.outer_power - 1)
            * task.inner_derivative
        )
    if isinstance(task, ChainTableValuesTask):
        def derivative(value: int, rate: int) -> int:
            return task.outer_power * value ** (task.outer_power - 1) * rate

        return (
            f"({derivative(task.first_inner_value, task.first_inner_derivative)}, "
            f"{derivative(task.second_inner_value, task.second_inner_derivative)})"
        )
    if isinstance(task, ChainFactorTupleTask):
        inner = task.inner_coefficient * x + task.inner_constant
        outer = task.outer_power * inner ** (task.outer_power - 1)
        return f"({outer}, {task.inner_coefficient})"
    if isinstance(task, ChainCorrectionTask):
        inner = (
            task.inner_coefficient * x + task.inner_constant
            if task.inner_kind == "affine"
            else task.inner_coefficient * x**2 + task.inner_constant
        )
        return str(sympy.diff(inner**task.outer_power, x))
    raise AssertionError(f"missing independent oracle for {type(task).__name__}")


def test_source_is_exactly_39_unreleased_typed_families(source):
    assert source.author == "AI-assisted implementation draft (unreviewed)"
    assert set(source.target_kcs) == set(TARGET_KCS)
    assert source.released_kcs == []
    assert len(source.families) == 39
    assert len({family.family_id for family in source.families}) == 39
    assert len({family.item_id for family in source.families}) == 39
    _assert_no_source_authored_expected(source.model_dump(mode="json"))


def test_surface_matrix_and_within_surface_constructs_are_independent(source):
    assert Counter(
        (family.kc_id, family.surface) for family in source.families
    ) == Counter(
        {
            (kc_id, surface): count
            for kc_id in TARGET_KCS
            for surface, count in EXPECTED_FAMILY_COUNTS.items()
        }
    )
    for kc_id in TARGET_KCS:
        for surface, count in (
            (AssessmentSurface.DIAGNOSTIC, 4),
            (AssessmentSurface.CHECKIN, 5),
        ):
            constructs = [
                family.construct_id
                for family in source.families
                if family.kc_id == kc_id and family.surface == surface
            ]
            assert len(constructs) == len(set(constructs)) == count


def test_pending_manifest_binds_every_compiled_family(source, manifest):
    assert manifest.compiler_version == COMPILER_VERSION
    entries = {
        (entry.blueprint_id, entry.revision): entry
        for entry in manifest.entries
    }
    assert len(entries) == 39
    for family in source.families:
        entry = entries[(family.blueprint_id, family.revision)]
        assert entry.source_digest == family_digest(source, family)
        assert entry.decision == ReviewDecision.PENDING
        assert entry.reviewed_by is None
        assert entry.reviewed_at is None


def test_family_digest_is_local_stable_and_truth_sensitive(source, monkeypatch):
    family = source.families[0]
    baseline = family_digest(source, family)

    unrelated_document_changes = (
        {"blueprint_version": "chain-rule-wave-cumulative-v99"},
        {"output_bank_version": "draft-cumulative-bank-v99"},
        {"released_kcs": ["kc.fun.function_notation"]},
        {"target_kcs": [*source.target_kcs, "kc.der.power_rule"]},
    )
    for update in unrelated_document_changes:
        assert family_digest(source.model_copy(update=update), family) == baseline

    changed_task = family.task.model_copy(
        update={"constant": family.task.constant - 1}
    )
    assert family_digest(
        source,
        family.model_copy(update={"task": changed_task}),
    ) != baseline
    assert family_digest(
        source,
        family.model_copy(update={"difficulty": "stretch"}),
    ) != baseline
    assert family_digest(
        source.model_copy(update={"author": "Different local author"}),
        family,
    ) != baseline
    assert family_digest(
        source.model_copy(update={"authoring_source": "assessment-draft/other"}),
        family,
    ) != baseline
    assert family_digest(
        source.model_copy(update={"graph_version": source.graph_version + 1}),
        family,
    ) != baseline

    monkeypatch.setattr(
        chain_rule_release,
        "COMPILER_VERSION",
        "chain-rule-item-compiler-test-change",
    )
    assert family_digest(source, family) != baseline


def test_every_authored_family_matches_an_independent_truth_oracle(source, compiled):
    bank, _report = compiled
    items = {item.item_id: item for item in bank.items}
    for family in source.families:
        oracle = _oracle_submission(family.task)
        verdict = verify_answer(items[family.item_id].answer, oracle, supervised=False)
        assert verdict.status == VerificationStatus.CORRECT, family.item_id


def test_truth_oracles_cover_both_orders_negatives_and_schema_bounds(source):
    orders = {
        family.task.order
        for family in source.families
        if isinstance(family.task, (AffineCompositionTask, CompositionAtPointTask))
    }
    assert orders == {"f_after_g", "g_after_f"}
    assert any(
        value < 0
        for family in source.families
        for value in family.task.model_dump().values()
        if isinstance(value, int)
    )
    boundary_tasks = (
        AffineFunctionValueTask(coefficient=-12, constant=-30, input_value=-20),
        QuadraticFunctionValueTask(
            quadratic=-6,
            linear=-12,
            constant=-30,
            input_value=-6,
        ),
        CompositionAtPointTask(
            f_coefficient=-10,
            f_constant=0,
            g_coefficient=1,
            g_constant=-20,
            point=-10,
            order="f_after_g",
        ),
        ChainAtPointTask(
            point=-20,
            inner_value=-2,
            inner_derivative=-9,
            outer_power=4,
        ),
        ChainFactorTupleTask(
            inner_coefficient=-10,
            inner_constant=20,
            outer_power=8,
        ),
    )
    for task in boundary_tasks:
        derived = derive_task(task)
        if derived.answer_kind == "numeric":
            answer = NumericAnswerSpec(expected=derived.submission, tolerance=0)
        elif derived.answer_kind == "symbolic":
            answer = SymbolicAnswerSpec(expected=derived.submission, variables=["x"])
        else:
            assert isinstance(derived.expected, tuple)
            answer = OrderedTupleAnswerSpec(
                expected=list(derived.expected),
                variables=["x"],
            )
        assert verify_answer(
            answer,
            _oracle_submission(task),
            supervised=False,
        ).status == VerificationStatus.CORRECT
    with pytest.raises(ValueError, match="absolute value at most 300"):
        ChainAtPointTask(
            point=0,
            inner_value=6,
            inner_derivative=12,
            outer_power=5,
        )


def test_numeric_production_never_exceeds_small_arithmetic_bound(compiled):
    bank, _report = compiled
    for item in bank.items:
        if isinstance(item.answer, NumericAnswerSpec):
            assert abs(float(item.answer.expected)) <= 300
        elif isinstance(item.answer, OrderedTupleAnswerSpec):
            for entry in item.answer.expected:
                parsed = sympy.sympify(entry)
                if not parsed.free_symbols:
                    assert abs(float(parsed)) <= 300


def test_compiled_bank_is_schema_v3_accessible_and_unreleased(compiled):
    bank, report = compiled
    assert bank.schema_version == 3
    assert bank.released_kcs == []
    assert len(bank.items) == 39
    assert {item.review_status for item in bank.items} == {ReviewStatus.DRAFT}
    assert all(item.provenance.reviewed_by is None for item in bank.items)
    assert all(
        segment.spoken_text
        for item in bank.items
        for segment in item.prompt
        if isinstance(segment, (MathPromptSegment, TablePromptSegment, PlotPromptSegment))
    )
    assert all(not isinstance(item.answer, ChoiceAnswerSpec) for item in bank.items)
    assert all(
        isinstance(
            item.answer,
            (NumericAnswerSpec, SymbolicAnswerSpec, OrderedTupleAnswerSpec),
        )
        for item in bank.items
    )
    assert any(
        isinstance(segment, TablePromptSegment)
        for item in bank.items
        for segment in item.prompt
    )
    assert any(
        isinstance(segment, PlotPromptSegment)
        for item in bank.items
        for segment in item.prompt
    )
    assert report.errors == ()
    assert report.answer_pairs_checked == 39 * 38 // 2 == 741
    assert report.literal_visible_pairs_checked == 39 * 38 == 1482
    assert report.visible_candidate_comparisons_checked > 0


def test_draft_bank_passes_runtime_direct_content_gates(compiled, graph):
    bank, _report = compiled
    assert _validate_item_bank_uncached(
        bank,
        graph,
        released_kcs=set(),
        reviewed_misconceptions={},
    ) == []


def test_guided_slider_mechanics_are_answer_neutral_and_text_equivalent(compiled):
    bank, _report = compiled
    guided = [
        item
        for item in bank.items
        if item.eligible_surfaces == [AssessmentSurface.GUIDED_WIDGET]
    ]
    assert len(guided) == 3
    assert all(not isinstance(item.guided_interaction, GuidedMappingSpec) for item in guided)
    assert all(isinstance(item.guided_interaction, GuidedSliderSpec) for item in guided)
    presentations = {
        item.guided_interaction.presentation.model_dump_json()  # type: ignore[union-attr]
        for item in guided
    }
    targets = {
        item.guided_interaction.scoring.target  # type: ignore[union-attr]
        for item in guided
    }
    assert len(presentations) == 1
    assert len(targets) == 3
    public = next(iter(presentations))
    assert '"target"' not in public
    assert "correct_pairs" not in public
    for target in targets:
        assert str(int(target)) not in public
    for item in guided:
        interaction = item.guided_interaction
        assert isinstance(interaction, GuidedSliderSpec)
        assert isinstance(item.answer, NumericAnswerSpec)
        assert interaction.scoring.target == float(item.answer.expected)
        assert sum(isinstance(segment, BlankPromptSegment) for segment in item.prompt) == 1
        assert (interaction.presentation.minimum, interaction.presentation.maximum) == (-20, 20)
        assert interaction.presentation.step == 1
        assert interaction.presentation.initial_value == 0
        stops = (
            (interaction.presentation.maximum - interaction.presentation.minimum)
            / interaction.presentation.step
            + 1
        )
        keyboard_distance = abs(
            interaction.scoring.target - interaction.presentation.initial_value
        ) / interaction.presentation.step
        assert stops <= 41
        assert keyboard_distance <= 20


def test_guided_tasks_use_only_small_arithmetic(source):
    guided = [
        family
        for family in source.families
        if family.surface == AssessmentSurface.GUIDED_WIDGET
    ]
    assert len(guided) == 3
    for family in guided:
        numeric_parameters = [
            value
            for value in family.task.model_dump().values()
            if isinstance(value, int)
        ]
        assert max(abs(value) for value in numeric_parameters) <= 5
        assert abs(int(derive_task(family.task).submission)) <= 20


def test_error_signatures_are_task_derived_and_match_the_runtime_exact_path(compiled):
    bank, _report = compiled
    pedagogy = PedagogySourceDocument.model_validate_json(PEDAGOGY_PATH.read_text())
    reviewed_ids = {
        source.kc_id: frozenset(item.id for item in source.misconceptions)
        for source in pedagogy.pack_sources
    }
    matcher = object.__new__(SessionOrchestratorV2)
    matcher._reviewed_misconceptions = reviewed_ids
    hard_predecessors = {
        kc_id: {
            edge.from_kc
            for edge in load_graph().edges
            if edge.type == EdgeType.HARD and edge.to_kc == kc_id
        }
        for kc_id in TARGET_KCS
    }
    signatures = [
        (item, signature)
        for item in bank.items
        for signature in item.error_signatures
    ]
    assert len(signatures) >= 30
    for item, signature in signatures:
        assert signature.misconception_id in reviewed_ids[item.kc_id]
        if signature.implicated_prereq is not None:
            assert signature.implicated_prereq in hard_predecessors[item.kc_id]
        verdict = verify_answer(item.answer, signature.expected_wrong, supervised=False)
        assert verdict.status == VerificationStatus.INCORRECT
        assert matcher._match_error_signature(
            item,
            signature.expected_wrong,
            False,
        ) == (signature.implicated_prereq, signature.misconception_id)
        assert matcher._match_error_signature(item, signature.expected_wrong, True) == (
            None,
            None,
        )


def test_each_worked_answer_is_rendered_once(compiled):
    bank, _report = compiled
    worked = [
        item
        for item in bank.items
        if item.eligible_surfaces == [AssessmentSurface.WORKED_EXAMPLE]
    ]
    assert len(worked) == 3
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
        assert len(steps) == 1
        assert len(answers) == 1
        assert answers[0].expression == item.answer.expected
        assert steps[0].expression != answers[0].expression
        assert item.prompt.index(steps[0]) < item.prompt.index(answers[0])
        assert verify_answer(
            item.answer,
            steps[0].expression,
            supervised=False,
        ).status == VerificationStatus.CORRECT


def test_packaged_bank_is_exact_deterministic_compiler_output(compiled):
    bank, _report = compiled
    packaged = ItemBankDocument.model_validate_json(BANK_PATH.read_text(encoding="utf-8"))
    assert packaged == bank
    assert BANK_PATH.read_text(encoding="utf-8") == bank.model_dump_json(indent=2) + "\n"


def test_wave_is_separated_from_preceding_product_quotient_draft(compiled, graph):
    bank, _report = compiled
    preceding = load_item_bank(SEED_DIR / "item_bank_product_quotient_v2.json")
    report = validate_inventory_separation(
        [*preceding.items, *bank.items],
        graph,
        focus_item_ids={item.item_id for item in bank.items},
    )
    assert report.errors == ()
    assert report.answer_pairs_checked == 91 * 90 // 2 == 4095
    assert report.literal_visible_pairs_checked == 91 * 90 == 8190


def test_construct_tampering_fails_before_publication(source, manifest, graph):
    first = source.families[0]
    tampered = first.model_copy(update={"construct_id": "function.quadratic_value"})
    bad_source = source.model_copy(update={"families": [tampered, *source.families[1:]]})
    with pytest.raises(ChainRuleCompilationError, match="construct"):
        compile_release_inventory(bad_source, manifest, graph)


def test_source_json_contains_no_hidden_review_or_truth_fields():
    raw = json.loads((SEED_DIR / "item_blueprints_chain_rule_v2.json").read_text())
    _assert_no_source_authored_expected(raw)
    assert "reviewed_by" not in json.dumps(raw)
    assert raw["released_kcs"] == []


def test_bank_generation_reads_but_never_rewrites_review_manifest(tmp_path, capsys):
    manifest_path = tmp_path / "independently-maintained-reviews.json"
    manifest_path.write_bytes(
        (SEED_DIR / "item_reviews_chain_rule_v2.json").read_bytes()
    )
    before = manifest_path.read_bytes()
    bank_path = tmp_path / "compiled-bank.json"

    assert chain_release_main(
        [
            "--manifest",
            str(manifest_path),
            "--out",
            str(bank_path),
        ]
    ) == 0

    assert bank_path.exists()
    assert manifest_path.read_bytes() == before
    assert "inventory OK" in capsys.readouterr().out


def test_pedagogy_math_segments_contain_math_not_arrow_prose():
    document = PedagogySourceDocument.model_validate_json(PEDAGOGY_PATH.read_text())
    forbidden = re.compile(r"->|\b(?:apply|replace|outer input box|unchanged inner form)\b")
    for source in document.pack_sources:
        for segment in (*source.lesson_narrative, *source.remediation):
            if isinstance(segment, MathPromptSegment):
                assert not forbidden.search(segment.expression.lower())
