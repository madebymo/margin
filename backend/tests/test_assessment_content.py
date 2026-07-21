"""Trusted item-bank validation and pure exposure allocation."""

from copy import deepcopy

import pytest

import tutor.content.item_bank as item_bank_module
from tutor.content.exposure import AllocationError, ItemAllocator
from tutor.content.item_bank import (
    AnswerSeparationIndeterminate,
    _answers_equivalent,
    _leakage_problems,
    bundle_leakage_problems,
    input_mode_for,
    load_item_bank,
    render_prompt,
    validate_item_bank,
)
from tutor.schemas.assessment import (
    AssessmentProvenance,
    AssessmentTaskKind,
    AssessmentSurface,
    BlankPromptSegment,
    ContentExposureState,
    FiniteSetAnswerSpec,
    ItemBankDocument,
    MathPromptSegment,
    NumericAnswerSpec,
    OrderedTupleAnswerSpec,
    PromptSemanticRole,
    TextPromptSegment,
)
from tutor.schemas.learner import EvidenceEvent
from tutor.seed.load_seed import load_graph
from tutor.verify.checker import VerificationResult, VerificationStatus

from tests.v2_helpers import (
    approved_power_rule_bank,
    approved_power_rule_catalog,
    empty_pedagogy_catalog,
    power_rule_only_graph,
)


@pytest.fixture(scope="module")
def bank():
    return approved_power_rule_bank()


@pytest.fixture(scope="module")
def graph():
    return load_graph()


@pytest.fixture(scope="module")
def catalog():
    return approved_power_rule_catalog()


def test_explicitly_approved_power_rule_fixture_is_release_valid(
    bank, graph, catalog
):
    assert len(bank.items) == 11
    assert validate_item_bank(bank, graph, catalog) == []


def test_assessment_provenance_binds_complete_reviewed_source_identity():
    provenance = AssessmentProvenance(
        source="assessment-source",
        author="Author One",
        reviewed_by="Reviewer Two",
        reviewed_at="2026-07-20T12:00:00Z",
        source_id="blueprint.power-rule.diagnostic",
        source_revision=2,
        source_digest="a" * 64,
        compiler_version="content-compiler-v2",
    )

    assert provenance.source_digest == "a" * 64

    with pytest.raises(ValueError, match="must be supplied together"):
        AssessmentProvenance(
            source="assessment-source",
            author="Author One",
            source_id="blueprint.power-rule.diagnostic",
        )
    with pytest.raises(ValueError, match="someone other than the author"):
        AssessmentProvenance(
            source="assessment-source",
            author=" Author One ",
            reviewed_by="author one",
            reviewed_at="2026-07-20T12:00:00Z",
        )
    with pytest.raises(ValueError, match="timezone"):
        AssessmentProvenance(
            source="assessment-source",
            author="Author One",
            reviewed_by="Reviewer Two",
            reviewed_at="2026-07-20T12:00:00",
        )


def test_release_requires_reviewed_catalog_coverage(bank, graph):
    errors = validate_item_bank(bank, graph, empty_pedagogy_catalog())

    assert any(
        "required KC has no reviewed pedagogy pack: kc.der.power_rule" in error
        for error in errors
    )


def test_release_rejects_a_catalog_pinned_to_another_graph(bank, graph, catalog):
    payload = catalog.model_dump(mode="json")
    payload["graph_version"] = graph.graph_version + 1
    mismatched = type(catalog).model_validate(payload)

    errors = validate_item_bank(bank, graph, mismatched)

    assert any("pedagogy catalog: graph version mismatch" in error for error in errors)


def test_release_gate_treats_graph_descriptions_as_student_visible_content(
    bank, catalog
):
    graph = power_rule_only_graph()
    payload = graph.model_dump(mode="json")
    payload["nodes"][0]["description"] = (
        "For the guided task, the answer is 12*x^3."
    )
    poisoned_graph = type(graph).model_validate(payload)

    errors = validate_item_bank(bank, poisoned_graph, catalog)

    assert any(
        "student-visible graph content" in error
        and "item.power.widget.scaled-quartic" in error
        for error in errors
    )


def test_packaged_bank_is_an_unreleased_honest_draft(graph):
    draft = load_item_bank()

    assert draft.released_kcs == []
    assert {item.review_status.value for item in draft.items} == {"draft"}
    assert all(item.provenance.reviewed_by is None for item in draft.items)
    assert all(item.provenance.reviewed_at is None for item in draft.items)
    assert validate_item_bank(draft, graph, empty_pedagogy_catalog()) == []


def test_legacy_prompt_and_task_fields_receive_safe_defaults():
    text = TextPromptSegment.model_validate({"kind": "text", "text": "Explain."})
    math = MathPromptSegment.model_validate({"kind": "math", "expression": "x^2"})
    blank = BlankPromptSegment.model_validate({"kind": "blank", "label": "Answer:"})

    assert text.role == PromptSemanticRole.INSTRUCTION
    assert math.role == PromptSemanticRole.GIVEN
    assert blank.role == PromptSemanticRole.RESPONSE
    assert all(item.task_kind == AssessmentTaskKind.SOLVE for item in load_item_bank().items)


@pytest.mark.parametrize(
    ("segment_type", "payload"),
    [
        (TextPromptSegment, {"kind": "text", "role": "response", "text": "Bad"}),
        (
            MathPromptSegment,
            {"kind": "math", "role": "instruction", "expression": "x"},
        ),
        (BlankPromptSegment, {"kind": "blank", "role": "given"}),
    ],
)
def test_prompt_segment_kind_role_matrix_is_enforced(segment_type, payload):
    with pytest.raises(ValueError, match="segments|blank"):
        segment_type.model_validate(payload)


def test_given_role_does_not_waive_self_leakage_for_solve_task(bank):
    item = bank.items[0]
    assert item.task_kind == AssessmentTaskKind.SOLVE
    exact_answer_prompt = item.model_copy(
        update={
            "prompt": [
                TextPromptSegment(text="Solve."),
                MathPromptSegment(
                    role=PromptSemanticRole.GIVEN,
                    expression=item.answer.expected,
                ),
                BlankPromptSegment(label="Answer:"),
            ]
        }
    )

    assert _leakage_problems(exact_answer_prompt) == [
        "a visible math segment is equivalent to the expected answer"
    ]


def test_transform_task_allows_only_a_distinct_equivalent_given(bank):
    item_payload = bank.items[0].model_dump(mode="json")
    item_payload["task_kind"] = "transform"
    item_payload["prompt"] = [
        {"kind": "text", "role": "instruction", "text": "Rewrite."},
        {"kind": "math", "role": "given", "expression": "x^2 * 3"},
        {"kind": "blank", "role": "response", "label": "Answer:"},
    ]
    transform = type(bank.items[0]).model_validate(item_payload)

    assert _leakage_problems(transform) == []

    item_payload["prompt"][1]["expression"] = " 3 * x ** 2 "
    relabeled_answer = type(bank.items[0]).model_validate(item_payload)
    assert _leakage_problems(relabeled_answer) == [
        "a visible math segment is equivalent to the expected answer"
    ]


def test_durable_identifier_and_provenance_widths_fail_at_schema_boundary(bank):
    item_payload = bank.items[0].model_dump(mode="json")
    for field in ("item_id", "family_id"):
        invalid = deepcopy(item_payload)
        invalid[field] = "a" * 129
        with pytest.raises(ValueError, match="at most 128"):
            type(bank.items[0]).model_validate(invalid)

    invalid = deepcopy(item_payload)
    invalid["provenance"]["source"] = "s" * 129
    with pytest.raises(ValueError, match="at most 128"):
        type(bank.items[0]).model_validate(invalid)

    event = {
        "event_id": "00000000-0000-0000-0000-000000000001",
        "learner_id": "00000000-0000-0000-0000-000000000002",
        "t": "2026-01-01T00:00:00Z",
        "item_id": "item",
        "kc_ids": ["kc.der.power_rule"],
        "correct": True,
        "response_class": "production",
        "policy_version": "p" * 65,
    }
    with pytest.raises(ValueError, match="at most 64"):
        EvidenceEvent.model_validate(event)


def test_incomplete_requested_kc_is_release_blocked(bank, graph, catalog):
    errors = validate_item_bank(
        bank,
        graph,
        catalog,
        {"kc.der.chain_rule"},
    )
    assert any("no assessment items" in error for error in errors)
    assert any("diagnostic has 0" in error for error in errors)


def test_structured_prompt_renderer_never_appends_expected(bank):
    item = next(
        candidate
        for candidate in bank.items
        if AssessmentSurface.DIAGNOSTIC in candidate.eligible_surfaces
    )
    prompt = render_prompt(item)
    assert "____" not in prompt  # the authored label replaces the generic blank
    assert item.answer.expected not in prompt
    assert input_mode_for(item) == "expression"


def test_lesson_bundle_reserves_five_disjoint_families(bank):
    result = ItemAllocator(bank).reserve_lesson_bundle(
        ContentExposureState(),
        "kc.der.power_rule",
    )
    reservations = [
        result.bundle.worked_example,
        result.bundle.guided_widget,
        *result.bundle.checkins,
    ]
    assert len({reservation.family_id for reservation in reservations}) == 5
    assert len(result.state.reservations) == 5


def test_diagnostic_confirmations_use_distinct_families_and_exhaust(bank):
    allocator = ItemAllocator(bank)
    state = ContentExposureState()
    families = set()
    for _ in range(3):
        result = allocator.reserve_item(
            state,
            kc_id="kc.der.power_rule",
            surface=AssessmentSurface.DIAGNOSTIC,
        )
        state = result.state
        families.add(result.reservation.family_id)
    assert len(families) == 3
    with pytest.raises(AllocationError):
        allocator.reserve_item(
            state,
            kc_id="kc.der.power_rule",
            surface=AssessmentSurface.DIAGNOSTIC,
        )


def test_allocator_uses_authored_order_before_lexical_family_id(bank):
    payload = bank.model_dump(mode="json")
    diagnostics = sorted(
        (
            item
            for item in payload["items"]
            if item["eligible_surfaces"] == ["diagnostic"]
        ),
        key=lambda item: item["family_id"],
    )
    for index, item in enumerate(reversed(diagnostics), start=1):
        item["allocation_order"] = index * 10
    ordered = ItemBankDocument.model_validate(payload)

    result = ItemAllocator(ordered).reserve_item(
        ContentExposureState(),
        kc_id="kc.der.power_rule",
        surface=AssessmentSurface.DIAGNOSTIC,
    )

    assert result.item.family_id == diagnostics[-1]["family_id"]


def test_allocator_skips_leaking_first_ordered_candidate(bank):
    allocator = ItemAllocator(bank)
    first = allocator.reserve_item(
        ContentExposureState(),
        kc_id="kc.der.power_rule",
        surface=AssessmentSurface.DIAGNOSTIC,
    )
    safe = allocator.reserve_item(
        ContentExposureState(),
        kc_id="kc.der.power_rule",
        surface=AssessmentSurface.DIAGNOSTIC,
        visible_texts=[f"The correct answer is {first.item.answer.expected}."],
    )

    assert safe.item.family_id != first.item.family_id


def test_allocator_serves_only_latest_revision_of_a_stable_item(bank):
    payload = bank.model_dump(mode="json")
    revised = deepcopy(payload["items"][0])
    revised["revision"] = 2
    payload["items"].append(revised)
    versioned = ItemBankDocument.model_validate(payload)

    allocator = ItemAllocator(versioned)
    state = ContentExposureState()
    reservations = []
    for _ in range(3):
        allocation = allocator.reserve_item(
            state,
            kc_id="kc.der.power_rule",
            surface=AssessmentSurface.DIAGNOSTIC,
        )
        state = allocation.state
        reservations.append(allocation.reservation)

    matching = [
        reservation
        for reservation in reservations
        if reservation.item_id == revised["item_id"]
    ]
    assert [reservation.revision for reservation in matching] == [2]


@pytest.mark.parametrize(
    "updates",
    [
        {"family_id": "family.power.revision-drift"},
        {
            "kc_id": "kc.der.chain_rule",
            "family_id": "family.power.revision-drift",
        },
        {"eligible_surfaces": ["checkin"]},
        {"task_kind": "transform"},
        {"allocation_order": 999},
    ],
)
def test_item_revisions_cannot_change_identity_lineage(bank, updates):
    payload = bank.model_dump(mode="json")
    revised = deepcopy(payload["items"][0])
    revised["revision"] = 2
    revised.update(updates)
    payload["items"].append(revised)

    with pytest.raises(ValueError, match="revisions of item"):
        ItemBankDocument.model_validate(payload)


def test_release_requires_unique_authored_family_orders(bank, graph, catalog):
    missing_payload = bank.model_dump(mode="json")
    missing_payload["items"][0]["allocation_order"] = None
    missing_errors = validate_item_bank(
        ItemBankDocument.model_validate(missing_payload),
        graph,
        catalog,
    )
    assert any("released item lacks allocation_order" in error for error in missing_errors)

    duplicate_payload = bank.model_dump(mode="json")
    diagnostics = [
        item
        for item in duplicate_payload["items"]
        if item["eligible_surfaces"] == ["diagnostic"]
    ]
    diagnostics[1]["allocation_order"] = diagnostics[0]["allocation_order"]
    duplicate_errors = validate_item_bank(
        ItemBankDocument.model_validate(duplicate_payload),
        graph,
        catalog,
    )
    assert any(
        "allocation_order" in error and "is reused across families" in error
        for error in duplicate_errors
    )


def test_same_family_revisions_may_share_allocation_order(bank, graph, catalog):
    payload = bank.model_dump(mode="json")
    revised = deepcopy(payload["items"][0])
    revised["revision"] = 2
    payload["items"].append(revised)
    versioned = ItemBankDocument.model_validate(payload)

    assert not any(
        "allocation_order" in error
        for error in validate_item_bank(versioned, graph, catalog)
    )


def test_monotonic_exposure_updates_preserve_durable_ledger_order(bank):
    allocator = ItemAllocator(bank)
    first = allocator.reserve_item(
        ContentExposureState(),
        kc_id="kc.der.power_rule",
        surface=AssessmentSurface.DIAGNOSTIC,
    )
    first_shown = allocator.record_exposure(first.state, first.reservation)
    second = allocator.reserve_item(
        first_shown,
        kc_id="kc.der.power_rule",
        surface=AssessmentSurface.DIAGNOSTIC,
    )
    both_shown = allocator.record_exposure(second.state, second.reservation)

    identical = allocator.update_exposure(
        both_shown,
        first.reservation,
        hints_seen=0,
    )
    assert identical is both_shown

    advanced = allocator.update_exposure(
        identical,
        first.reservation,
        hints_seen=1,
    )
    assert [record.item_id for record in advanced.exposures] == [
        first.reservation.item_id,
        second.reservation.item_id,
    ]
    assert advanced.exposures[0].hints_seen == 1


def test_exposure_updates_monotonically_and_third_hint_retires_family(bank):
    allocator = ItemAllocator(bank)
    allocated = allocator.reserve_item(
        ContentExposureState(),
        kc_id="kc.der.power_rule",
        surface=AssessmentSurface.DIAGNOSTIC,
    )
    shown = allocator.record_exposure(
        allocated.state,
        allocated.reservation,
        hints_seen=0,
    )
    hinted = allocator.update_exposure(
        shown,
        allocated.reservation,
        hints_seen=3,
        answer_revealed=True,
    )
    assert allocated.reservation.family_id in hinted.retired_family_ids
    assert len(hinted.exposures) == 1
    with pytest.raises(AllocationError):
        allocator.update_exposure(
            hinted,
            allocated.reservation,
            hints_seen=2,
            answer_revealed=True,
        )


def test_validator_detects_answer_reuse_across_families(bank, graph, catalog):
    payload = bank.model_dump(mode="json")
    clone = deepcopy(payload["items"][0])
    clone["item_id"] = "item.power.diagnostic.cloned"
    clone["family_id"] = "family.power.diagnostic.cloned"
    payload["items"].append(clone)
    duplicated = ItemBankDocument.model_validate(payload)
    errors = validate_item_bank(duplicated, graph, catalog)
    assert any("expected answer reused across families" in error for error in errors)


def test_validator_requires_production_inventory_for_mastery_surfaces(
    bank, graph, catalog
):
    payload = bank.model_dump(mode="json")
    checkins = [
        item
        for item in payload["items"]
        if item["eligible_surfaces"] == ["checkin"]
    ]
    for index, item in enumerate(checkins):
        item["answer"] = {
            "kind": "choice",
            "option_ids": [f"wrong-{index}", f"right-{index}"],
            "expected_choice_id": f"right-{index}",
        }

    errors = validate_item_bank(
        ItemBankDocument.model_validate(payload), graph, catalog
    )

    assert any(
        "checkin has 0 production families; requires 4" in error
        for error in errors
    )


def test_validator_blocks_choice_items_until_semantic_answer_reuse_is_checkable(
    bank, graph, catalog
):
    payload = bank.model_dump(mode="json")
    payload["items"][0]["answer"] = {
        "kind": "choice",
        "option_ids": ["distractor", "correct"],
        "expected_choice_id": "correct",
    }

    errors = validate_item_bank(
        ItemBankDocument.model_validate(payload), graph, catalog
    )

    assert any(
        "choice items cannot be released" in error
        for error in errors
    )


def test_validator_rejects_families_that_span_assessment_surfaces(
    bank, graph, catalog
):
    payload = bank.model_dump(mode="json")
    reused_families = [
        item["family_id"]
        for item in payload["items"]
        if item["eligible_surfaces"] in (["diagnostic"], ["guided_widget"])
    ]
    checkins = [
        item
        for item in payload["items"]
        if item["eligible_surfaces"] == ["checkin"]
    ]
    for item, family_id in zip(checkins, reused_families, strict=True):
        item["family_id"] = family_id

    errors = validate_item_bank(
        ItemBankDocument.model_validate(payload), graph, catalog
    )

    assert any("family spans surfaces" in error for error in errors)


def test_validator_rejects_unreviewed_misconception_ids(bank, graph, catalog):
    payload = bank.model_dump(mode="json")
    payload["items"][0]["error_signatures"] = [
        {
            "expected_wrong": "0",
            "misconception_id": "m.power.invented",
        }
    ]

    errors = validate_item_bank(
        ItemBankDocument.model_validate(payload), graph, catalog
    )

    assert any(
        "misconception m.power.invented is not in a human-approved pedagogy pack"
        in error
        for error in errors
    )


def test_validator_detects_equivalent_answer_reuse_across_families(
    bank, graph, catalog
):
    payload = bank.model_dump(mode="json")
    clone = deepcopy(payload["items"][0])
    clone["item_id"] = "item.power.diagnostic.equivalent_clone"
    clone["family_id"] = "family.power.diagnostic.equivalent_clone"
    clone["answer"]["expected"] = "x^2 + x^2 + x^2"
    payload["items"].append(clone)

    errors = validate_item_bank(
        ItemBankDocument.model_validate(payload), graph, catalog
    )

    assert any(
        "expected answer reused across families" in error
        and "mathematically equivalent" in error
        for error in errors
    )


def test_validator_detects_literal_cross_family_hint_leak(bank, graph, catalog):
    payload = bank.model_dump(mode="json")
    source = next(
        item
        for item in payload["items"]
        if item["item_id"] == "item.power.diagnostic.cube"
    )
    source["hints"][0]["text"] = "For the later check, the answer is 10*x^4."

    errors = validate_item_bank(
        ItemBankDocument.model_validate(payload), graph, catalog
    )

    assert any(
        "item.power.diagnostic.cube" in error
        and (
            "visible content leaks scored answer for "
            "item.power.checkin.scaled-quintic"
        )
        in error
        for error in errors
    )


def test_validator_checks_revealing_hint_against_other_scored_families(
    bank, graph, catalog
):
    payload = bank.model_dump(mode="json")
    source = next(
        item
        for item in payload["items"]
        if item["item_id"] == "item.power.diagnostic.cube"
    )
    source["hints"][2]["text"] = "For the later check, the answer is 10*x^4."

    errors = validate_item_bank(
        ItemBankDocument.model_validate(payload), graph, catalog
    )

    assert any(
        "item.power.diagnostic.cube" in error
        and "item.power.checkin.scaled-quintic" in error
        for error in errors
    )


def test_validator_detects_equivalent_cross_family_hint_leak(
    bank, graph, catalog
):
    payload = bank.model_dump(mode="json")
    source = next(
        item
        for item in payload["items"]
        if item["item_id"] == "item.power.diagnostic.cube"
    )
    source["hints"][0]["text"] = "The result is 5*x^4 + 5*x^4."

    errors = validate_item_bank(
        ItemBankDocument.model_validate(payload), graph, catalog
    )

    assert any(
        "item.power.diagnostic.cube" in error
        and (
            "visible content leaks scored answer for "
            "item.power.checkin.scaled-quintic"
        )
        in error
        for error in errors
    )


def test_bundle_gate_catches_generated_narrative_answer_leak(bank):
    upcoming = next(
        item
        for item in bank.items
        if item.item_id == "item.power.diagnostic.cube"
    )
    assert bundle_leakage_problems(
        ["Remember: the correct result is 3*x^2."],
        [upcoming],
    ) == [
        "item.power.diagnostic.cube: expected answer leaks into visible content"
    ]
    assert bundle_leakage_problems(
        ["Remember to multiply by the exponent, then reduce it by one."],
        [upcoming],
    ) == []


def test_bundle_gate_catches_equivalent_math_embedded_in_ordinary_prose(bank):
    upcoming = next(
        item
        for item in bank.items
        if item.item_id == "item.power.diagnostic.cube"
    )

    assert bundle_leakage_problems(
        ["After simplification, we get x^2*3."],
        [upcoming],
    ) == [
        "item.power.diagnostic.cube: expected answer leaks into visible content"
    ]


def test_bundle_gate_catches_implicit_multiplication_embedded_in_prose(bank):
    upcoming = next(
        item
        for item in bank.items
        if item.item_id == "item.power.diagnostic.cube"
    )

    assert bundle_leakage_problems(
        ["After simplifying, write 3x^2 and continue."],
        [upcoming],
        supervised=False,
    ) == [
        "item.power.diagnostic.cube: expected answer leaks into visible content"
    ]


@pytest.mark.parametrize(
    "visible",
    [
        "An equivalent form is (4*x^2)(3/4).",
        "An equivalent form is (3*x)x.",
    ],
)
def test_bundle_gate_catches_implicit_group_products_embedded_in_prose(
    bank,
    visible,
):
    upcoming = next(
        item
        for item in bank.items
        if item.item_id == "item.power.diagnostic.cube"
    )

    assert bundle_leakage_problems(
        [visible],
        [upcoming],
        supervised=False,
    ) == [
        "item.power.diagnostic.cube: expected answer leaks into visible content"
    ]


def test_bundle_gate_fails_closed_when_implicit_product_check_times_out(
    bank,
    monkeypatch,
):
    upcoming = next(
        item
        for item in bank.items
        if item.item_id == "item.power.diagnostic.cube"
    )

    def timed_out(answer, given, **_kwargs):
        assert given == "3x^2"
        return VerificationResult(
            status=VerificationStatus.TIMEOUT,
            code="equivalence_timeout",
        )

    monkeypatch.setattr(item_bank_module, "verify_answer", timed_out)

    assert bundle_leakage_problems(
        ["After simplifying, write 3x^2 and continue."],
        [upcoming],
    ) == [
        "item.power.diagnostic.cube: answer-separation check indeterminate "
        "(timeout:equivalence_timeout)"
    ]


def test_finite_set_answer_reuse_ignores_duplicate_members(bank):
    cube = next(
        item for item in bank.items if item.item_id == "item.power.diagnostic.cube"
    )
    quartic = next(
        item
        for item in bank.items
        if item.item_id == "item.power.diagnostic.quartic"
    )
    singleton = cube.model_copy(
        update={"answer": FiniteSetAnswerSpec(expected=["x"], variables=["x"])}
    )
    duplicated = quartic.model_copy(
        update={
            "answer": FiniteSetAnswerSpec(
                expected=["x", "2*x/2"],
                variables=["x"],
            )
        }
    )

    assert _answers_equivalent(singleton, duplicated)


@pytest.mark.parametrize(
    "visible",
    [
        "It becomes 2.",
        "The value becomes 2.",
        "So use 2.",
        "The final value becomes 2.0.",
        "The value becomes 2.0, which closes the example.",
    ],
)
def test_bundle_gate_catches_short_numeric_atom_in_prose(bank, visible):
    template = next(
        item
        for item in bank.items
        if item.item_id == "item.power.diagnostic.cube"
    )
    upcoming = template.model_copy(
        update={"answer": NumericAnswerSpec(expected="2")}
    )

    assert bundle_leakage_problems([visible], [upcoming]) == [
        "item.power.diagnostic.cube: expected answer leaks into visible content"
    ]


@pytest.mark.parametrize("visible", ["Use 20.", "Use x2.", "Study x^2.", "Use 2*x."])
def test_short_numeric_literal_gate_respects_math_token_boundaries(bank, visible):
    template = next(
        item
        for item in bank.items
        if item.item_id == "item.power.diagnostic.cube"
    )
    upcoming = template.model_copy(
        update={"answer": NumericAnswerSpec(expected="2")}
    )

    assert bundle_leakage_problems([visible], [upcoming]) == []


def test_release_gate_fails_closed_when_prompt_leakage_check_times_out(
    bank,
    graph,
    catalog,
    monkeypatch,
):
    payload = bank.model_dump(mode="json")
    payload["bank_version"] = "test-prompt-timeout"
    isolated_bank = ItemBankDocument.model_validate(payload)

    def timed_out_prompt(answer, given, **_kwargs):
        matches = given == getattr(answer, "expected", None)
        if given == "x^3":
            return VerificationResult(
                status=VerificationStatus.TIMEOUT,
                code="equivalence_timeout",
            )
        return VerificationResult(
            status=(
                VerificationStatus.CORRECT
                if matches
                else VerificationStatus.INCORRECT
            ),
            code="equivalent" if matches else "not_equivalent",
        )

    monkeypatch.setattr(item_bank_module, "verify_answer", timed_out_prompt)

    errors = validate_item_bank(isolated_bank, graph, catalog)

    assert any(
        "visible math segment answer-separation check is indeterminate"
        in error
        and "timeout:equivalence_timeout" in error
        for error in errors
    )


def test_release_and_boolean_helpers_fail_closed_on_worker_error(
    bank,
    graph,
    catalog,
    monkeypatch,
):
    payload = bank.model_dump(mode="json")
    payload["bank_version"] = "test-cross-family-worker-error"
    isolated_bank = ItemBankDocument.model_validate(payload)
    cube = next(
        item
        for item in isolated_bank.items
        if item.item_id == "item.power.diagnostic.cube"
    )
    quartic = next(
        item
        for item in isolated_bank.items
        if item.item_id == "item.power.diagnostic.quartic"
    )

    def worker_error(answer, given, **_kwargs):
        if {
            str(getattr(answer, "expected", None)),
            given,
        } == {"3*x^2", "4*x^3"}:
            return VerificationResult(
                status=VerificationStatus.INVALID,
                code="worker_unavailable",
            )
        matches = given == getattr(answer, "expected", None)
        return VerificationResult(
            status=(
                VerificationStatus.CORRECT
                if matches
                else VerificationStatus.INCORRECT
            ),
            code="equivalent" if matches else "not_equivalent",
        )

    monkeypatch.setattr(item_bank_module, "verify_answer", worker_error)

    with pytest.raises(AnswerSeparationIndeterminate, match="worker_unavailable"):
        _answers_equivalent(cube, quartic)
    errors = validate_item_bank(isolated_bank, graph, catalog)

    assert any(
        "expected-answer comparison indeterminate across families" in error
        and "invalid:worker_unavailable" in error
        for error in errors
    )


def test_cross_answer_domains_distinguish_inequality_from_indeterminate(bank):
    cube = next(
        item for item in bank.items if item.item_id == "item.power.diagnostic.cube"
    )
    quartic = next(
        item
        for item in bank.items
        if item.item_id == "item.power.diagnostic.quartic"
    )
    numeric = quartic.model_copy(update={"answer": NumericAnswerSpec(expected="2")})
    finite_set = cube.model_copy(
        update={"answer": FiniteSetAnswerSpec(expected=["1"])}
    )
    ordered_tuple = quartic.model_copy(
        update={"answer": OrderedTupleAnswerSpec(expected=["1"])}
    )

    assert not _answers_equivalent(cube, numeric)
    assert not _answers_equivalent(finite_set, ordered_tuple)
