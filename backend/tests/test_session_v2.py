"""End-to-end trustworthy item allocation and lesson sequencing."""

from copy import deepcopy
from datetime import timedelta

import pytest

import tutor.content.item_bank as item_bank_module
from tutor.orchestrator.session_v2 import SessionOrchestratorV2
from tutor.runtime_capabilities import widget_capability_manifest
from tutor.schemas.assessment import AssessmentSurface, ItemBankDocument
from tutor.schemas.learner import LearnerProfile
from tutor.seed.load_seed import load_graph
from tutor.verify.checker import VerificationResult, VerificationStatus

from tests.v2_helpers import (
    POWER_RULE_KC,
    approved_power_rule_bank,
    approved_power_rule_catalog,
    power_rule_only_graph,
)

TARGET = POWER_RULE_KC
PROFILE = LearnerProfile(course="AP Calculus AB", age_band="16-18")


def _session(*, budget: int = 8) -> SessionOrchestratorV2:
    return SessionOrchestratorV2(
        power_rule_only_graph(),
        TARGET,
        PROFILE,
        item_bank=approved_power_rule_bank(),
        probe_budget=budget,
        pedagogy_catalog=approved_power_rule_catalog(),
    )


def test_perfect_path_confirms_with_distinct_family_then_unseen_capstone():
    session = _session()
    session.begin()
    first_family = session.pending.family_id
    first_answer = session.pending_expected
    first_result = session.submit(first_answer)
    assert "remains uncertain" in first_result[0].text

    assert session.pending.kind == "probe"
    assert session.pending.family_id != first_family
    assert session.pending_expected != first_answer
    session.submit(session.pending_expected)

    assert session.phase.value == "capstone", (
        session.summary(),
        session.learner.routing_score(TARGET),
        session.learner.recent_independent_counts(TARGET),
        [
            (event.surface, event.correct, event.family_id)
            for event in session.learner.events
        ],
    )
    assert session.pending.kind == "capstone"
    assert not session.pending.can_hint
    assert session.hint() is None
    used = [reservation.family_id for reservation in session.exposure_state.reservations]
    assert len(used) == len(set(used))
    session.submit(session.pending_expected)
    assert session.phase.value == "done"


def test_wrong_answer_matching_next_family_truth_skips_that_family():
    session = _session()
    session.begin()
    assert session.pending.item_id == "item.power.diagnostic.cube"

    # Wrong for d/dx(x^3), but exactly the authored truth for the otherwise
    # deterministic next diagnostic family d/dx(x^4).
    session.submit("4*x^3")

    assert "4*x^3" in session._visible_texts
    assert session.pending.item_id == "item.power.diagnostic.sixth"
    assert session.pending_expected == "6*x^5"
    assert all(
        reservation.family_id != "family.power.diagnostic.quartic"
        for reservation in session.exposure_state.reservations
    )


def test_visible_text_ledger_round_trips_exactly_with_pending_skip():
    session = _session()
    session.begin()
    session.submit("4*x^3")
    checkpoint = session.export_checkpoint()

    restored = SessionOrchestratorV2.restore(
        power_rule_only_graph(),
        checkpoint,
        item_bank=approved_power_rule_bank(),
        pedagogy_catalog=approved_power_rule_catalog(),
    )

    assert restored._visible_texts == session._visible_texts
    assert restored._private_visible_inputs == session._private_visible_inputs
    assert restored.pending.item_id == "item.power.diagnostic.sixth"


def test_guided_practice_is_followed_by_two_independent_unseen_checks():
    session = _session(budget=2)
    session.begin()
    session.submit("0")
    session.submit("0")

    assert session.phase.value == "teach"
    assert session.pending.kind == "guided_widget"
    guided_answer = session.pending_expected
    interactions = session.submit(guided_answer)
    assert interactions
    assert session.pending.kind == "checkin"

    checkin_families = []
    for _ in range(3):
        checkin_families.append(session.pending.family_id)
        session.submit(session.pending_expected)
    assert len(set(checkin_families)) == 3
    assert session.learner.routing_score(TARGET) >= 0.9, session.learner.routing_score(
        TARGET
    )
    assert session.learner.recent_independent_counts(TARGET)[0] >= 2
    assert session.phase.value == "capstone", (
        session.summary(),
        session.learner.routing_score(TARGET),
        session.learner.recent_independent_counts(TARGET),
        [
            (event.surface, event.correct, event.family_id)
            for event in session.learner.events
        ],
    )
    assert all(
        event.surface != "guided_widget"
        or session.learner.mastery_status(TARGET) == "confirmed_mastered"
        for event in session.learner.events
    )
    transitions = [
        event
        for event in session.learner.events
        if event.surface == "instructional_practice"
    ]
    assert len(transitions) == 1
    assert transitions[0].learning_opportunity
    assert all(
        not event.learning_opportunity
        for event in session.learner.events
        if event.surface == "checkin"
    )


def test_dynamic_collision_replaces_only_the_unsafe_queued_checkin():
    session = _session(budget=2)
    session.begin()
    session.submit("0")
    session.submit("0")
    assert session.pending.kind == "guided_widget"
    original_checkins = list(session._current_bundle.checkins)
    colliding = original_checkins[0]
    colliding_answer = session._item_for(colliding).answer.expected

    session.submit(colliding_answer)
    session.submit(session.pending_expected)

    assert session.pending.kind == "checkin"
    assert session.pending.family_id != colliding.family_id
    assert session._current_bundle.checkins[1:] == original_checkins[1:]
    assert session._current_bundle.checkins[0] == session.pending.reservation
    assert colliding in session.exposure_state.reservations
    assert all(
        exposure.family_id != colliding.family_id
        for exposure in session.exposure_state.exposures
    )


def test_episode_capability_pin_cannot_be_widened_by_runtime_enable():
    session = SessionOrchestratorV2(
        power_rule_only_graph(),
        TARGET,
        PROFILE,
        item_bank=approved_power_rule_bank(),
        probe_budget=2,
        pedagogy_catalog=approved_power_rule_catalog(),
        widget_capabilities=widget_capability_manifest(rich_widgets=False),
    )
    session.set_runtime_widget_capabilities(
        widget_capability_manifest(rich_widgets=True)
    )
    session.begin()
    session.submit("0")
    interactions = session.submit("0")

    assert session.pending.kind == "guided_widget"
    assert session.pending.delivery_mode == "text"
    assert session.pending.input_mode == "math"
    assert not any(interaction.widget for interaction in interactions)


def test_text_fallback_requires_genuine_practice_before_learning_transition():
    legacy_live_input_manifest = {
        "version": "web-widget-capabilities-v2.1",
        "supported": {
            "mapping": {
                "keyboard_equivalent": True,
                "live_visual": False,
            },
            "live_input": {
                "keyboard_equivalent": True,
                "live_visual": True,
            },
        },
        "disabled": {
            "slider": "Not pinned for this test episode.",
            "click_region": "Not pinned for this test episode.",
        },
    }
    session = SessionOrchestratorV2(
        power_rule_only_graph(),
        TARGET,
        PROFILE,
        item_bank=approved_power_rule_bank(),
        probe_budget=2,
        pedagogy_catalog=approved_power_rule_catalog(),
        widget_capabilities=legacy_live_input_manifest,
    )
    session.begin()
    session.submit("0")
    session.submit("0")

    pending_key = session.pending.key
    events_before_switch = len(session.learner.events)
    session.use_text_fallback()

    assert session.pending.key == pending_key
    assert session.pending.delivery_mode == "text"
    assert session.pending.input_mode == "math"
    assert len(session.learner.events) == events_before_switch
    assert [
        event.surface for event in session.learner.events
    ].count("instructional_practice") == 0

    session.submit("0")
    assert session.pending.key == pending_key
    assert session.pending.kind == "guided_widget"
    assert [
        event.surface for event in session.learner.events
    ].count("instructional_practice") == 0

    session.submit(session.pending_expected)
    assert session.pending.kind == "checkin"
    assert [
        event.surface for event in session.learner.events
    ].count("instructional_practice") == 1


def test_runtime_gate_rejects_lesson_content_that_leaks_a_capstone_answer():
    payload = approved_power_rule_bank().model_dump(mode="json")
    worked = next(
        item
        for item in payload["items"]
        if item["item_id"] == "item.power.worked.eighth"
    )
    worked["prompt"][0]["text"] += (
        " A later goal result is -4*x^(-2)+5*x^4."
    )
    poisoned = ItemBankDocument.model_validate(payload)
    with pytest.raises(ValueError, match="visible content leaks scored answer"):
        SessionOrchestratorV2(
            power_rule_only_graph(),
            TARGET,
            PROFILE,
            item_bank=poisoned,
            probe_budget=2,
            pedagogy_catalog=approved_power_rule_catalog(),
        )


def test_runtime_lesson_gate_catches_graph_text_leaking_guided_answer():
    session = _session(budget=2)
    session._nodes[TARGET] = session._nodes[TARGET].model_copy(
        update={
            "description": "For the guided task, the answer is 12*x^3.",
        }
    )
    session.begin()
    session.submit("0")

    interactions = session.submit("0")

    assert session.phase.value == "stopped"
    assert any("answer-separation gate" in item.text for item in interactions)
    assert all("12*x^3" not in item.text for item in interactions)


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (VerificationStatus.INVALID, "worker_error"),
        (VerificationStatus.TIMEOUT, "equivalence_timeout"),
    ],
)
def test_runtime_bundle_gate_stops_on_indeterminate_worker_result(
    monkeypatch,
    status,
    code,
):
    session = _session(budget=2)
    session.begin()
    session.submit("0")

    def worker_error(_answer, _given, **_kwargs):
        return VerificationResult(
            status=status,
            code=code,
        )

    monkeypatch.setattr(item_bank_module, "verify_answer", worker_error)

    interactions = session.submit("0")

    assert session.phase.value == "stopped"
    assert any(
        "answer-separation gate" in interaction.text
        for interaction in interactions
    )


def test_revealing_hint_retires_family_and_forces_fresh_check():
    session = _session(budget=2)
    session.begin()
    session.submit("0")
    session.submit("0")
    session.submit(session.pending_expected)
    assisted_family = session.pending.family_id
    before = len(session.learner.events)
    assert session.hint()
    assert session.hint()
    revealing = session.hint()

    assert revealing.interactions
    assert len(session.learner.events) == before
    assert assisted_family in session.exposure_state.retired_family_ids
    assert session.pending.kind == "checkin"
    assert session.pending.family_id != assisted_family


def test_revealed_family_remains_retired_in_a_fresh_episode():
    session = _session()
    session.begin()
    revealed_family = session.pending.family_id
    assert session.hint()
    assert session.hint()
    assert session.hint()

    fresh = session.fresh_episode(as_of=session.learner.as_of + timedelta(seconds=1))
    fresh.begin()

    assert revealed_family in fresh.exposure_state.retired_family_ids
    assert fresh.pending.family_id != revealed_family


def test_conceptual_hints_do_not_assist_but_revealing_hint_does():
    session = _session()
    session.begin()

    assert session.hint()
    assert session.hint()
    assert not session.pending.assisted
    session.submit(session.pending_expected)
    assert session.learner.events[-1].hints_used == 2
    assert not session.learner.events[-1].assisted

    revealed_family = session.pending.family_id
    before = len(session.learner.events)
    assert session.hint()
    assert session.hint()
    assert not session.pending.assisted
    revealing = session.hint()
    assert revealing.interactions
    assert len(session.learner.events) == before
    assert session.pending.family_id != revealed_family
    assert revealed_family in session.exposure_state.retired_family_ids


def test_target_conflict_gets_fresh_verification_before_any_lesson():
    session = _session()
    session.begin()
    session.submit(session.pending_expected)
    session.submit("0")
    session.submit(session.pending_expected)

    assert session.summary()["plan_step"] == "verify_uncertain"
    assert session.pending.kind == "checkin"
    assert session._current_bundle is None
    verification_family = session.pending.family_id
    diagnostic_families = {
        event.family_id
        for event in session.learner.events
        if event.surface == "diagnostic"
    }
    assert verification_family not in diagnostic_families

    session.submit(session.pending_expected)
    assert session.phase.value == "capstone"


def test_invalid_input_keeps_pending_and_creates_no_evidence():
    session = _session()
    session.begin()
    key = session.pending.key
    before = len(session.learner.events)
    outputs = session.submit("factorial(999)")

    assert session.pending.key == key
    assert len(session.learner.events) == before
    assert "Nothing was graded" in outputs[0].text


def test_checkpoint_restore_is_exact_and_continues_without_reallocation():
    session = _session()
    session.begin()
    session.submit(session.pending_expected)
    checkpoint = session.export_checkpoint()

    restored = SessionOrchestratorV2.restore(
        power_rule_only_graph(),
        checkpoint,
        item_bank=approved_power_rule_bank(),
        pedagogy_catalog=approved_power_rule_catalog(),
    )
    assert restored.pending.model_dump() == session.pending.model_dump()
    assert restored.exposure_state == session.exposure_state
    assert restored.summary() == session.summary()
    restored.submit(restored.pending_expected)
    assert restored.phase.value == "capstone"


def test_checkpoint_restore_rejects_private_expected_answer_tampering():
    session = _session()
    session.begin()
    checkpoint = session.export_checkpoint()
    exposure_ledger = deepcopy(checkpoint["exposure_state"])
    pending_identity = deepcopy(
        {
            key: checkpoint["pending"][key]
            for key in (
                "key",
                "kind",
                "kc_id",
                "item_id",
                "item_revision",
                "family_id",
                "reservation",
                "prompt",
                "prompt_segments",
                "hints",
                "revealing_hints",
            )
        }
    )

    # Change only the private scoring truth. The learner-visible pending state
    # and the reservation/exposure ledgers remain byte-for-byte equivalent.
    checkpoint["pending"]["answer_spec"]["expected"] = "999*x"

    assert checkpoint["exposure_state"] == exposure_ledger
    assert {
        key: checkpoint["pending"][key] for key in pending_identity
    } == pending_identity
    with pytest.raises(
        ValueError,
        match="pending authored content does not match the pinned item bank",
    ):
        SessionOrchestratorV2.restore(
            power_rule_only_graph(),
            checkpoint,
            item_bank=approved_power_rule_bank(),
            pedagogy_catalog=approved_power_rule_catalog(),
        )


@pytest.mark.parametrize(
    ("exposure_field", "value", "message"),
    [
        ("hints_seen", 3, "hint position does not match its exposure"),
        ("answer_revealed", True, "already revealed its scoring truth"),
        ("solution_exposed", True, "already revealed its scoring truth"),
    ],
)
def test_checkpoint_restore_rejects_pending_assistance_exposure_drift(
    exposure_field,
    value,
    message,
):
    session = _session()
    session.begin()
    checkpoint = session.export_checkpoint()
    pending = checkpoint["pending"]
    exposure = next(
        record
        for record in checkpoint["exposure_state"]["exposures"]
        if record["item_id"] == pending["item_id"]
        and record["revision"] == pending["item_revision"]
    )

    exposure[exposure_field] = value

    with pytest.raises(ValueError, match=message):
        SessionOrchestratorV2.restore(
            power_rule_only_graph(),
            checkpoint,
            item_bank=approved_power_rule_bank(),
            pedagogy_catalog=approved_power_rule_catalog(),
        )


@pytest.mark.parametrize(
    "authored_field",
    [
        "identity",
        "rendered_prompt",
        "prompt_segments",
        "hints",
        "revealing_hints",
    ],
)
def test_checkpoint_restore_rejects_other_pending_authored_content_drift(
    authored_field,
):
    session = _session()
    session.begin()
    checkpoint = session.export_checkpoint()
    pending = checkpoint["pending"]

    if authored_field == "identity":
        pending["family_id"] = "family.power.diagnostic.corrupt"
    elif authored_field == "rendered_prompt":
        pending["prompt"] += " Corrupted prompt."
    elif authored_field == "prompt_segments":
        text_segment = next(
            segment
            for segment in pending["prompt_segments"]
            if segment["kind"] == "text"
        )
        text_segment["text"] += " Corrupted segment."
    elif authored_field == "hints":
        pending["hints"][0] += " Corrupted hint."
    else:
        pending["revealing_hints"][0] = True

    with pytest.raises(
        ValueError,
        match="pending authored content does not match the pinned item bank",
    ):
        SessionOrchestratorV2.restore(
            power_rule_only_graph(),
            checkpoint,
            item_bank=approved_power_rule_bank(),
            pedagogy_catalog=approved_power_rule_catalog(),
        )


@pytest.mark.parametrize("corruption", ["bundle_reference", "queue_order"])
def test_checkpoint_restore_rejects_corrupt_lesson_content_references(corruption):
    session = _session(budget=2)
    session.begin()
    session.submit("0")
    session.submit("0")
    assert session.pending.kind == "guided_widget"
    checkpoint = session.export_checkpoint()

    if corruption == "bundle_reference":
        checkpoint["current_bundle"]["guided_widget"]["variant_id"] = "corrupt"
    else:
        checkpoint["checkin_queue"].reverse()

    with pytest.raises(ValueError, match="checkpoint"):
        SessionOrchestratorV2.restore(
            power_rule_only_graph(),
            checkpoint,
            item_bank=approved_power_rule_bank(),
            pedagogy_catalog=approved_power_rule_catalog(),
        )


def test_checkpoint_restore_requires_exact_pinned_policy_versions():
    session = _session()
    session.begin()
    checkpoint = session.export_checkpoint()

    assert checkpoint["policy_versions"] == session.summary()["policy_versions"]
    assert checkpoint["widget_capability_manifest"]["version"]

    missing_versions = dict(checkpoint)
    missing_versions.pop("policy_versions")
    with pytest.raises(ValueError, match="policy implementation is unavailable"):
        SessionOrchestratorV2.restore(
            power_rule_only_graph(),
            missing_versions,
            item_bank=approved_power_rule_bank(),
            pedagogy_catalog=approved_power_rule_catalog(),
        )

    changed_versions = dict(checkpoint)
    changed_versions["policy_versions"] = {
        **checkpoint["policy_versions"],
        "diagnosis": "future-policy",
    }
    with pytest.raises(ValueError, match="policy implementation is unavailable"):
        SessionOrchestratorV2.restore(
            power_rule_only_graph(),
            changed_versions,
            item_bank=approved_power_rule_bank(),
            pedagogy_catalog=approved_power_rule_catalog(),
        )

    missing_capabilities = dict(checkpoint)
    missing_capabilities.pop("widget_capability_manifest")
    with pytest.raises(ValueError, match="capability manifest version is unavailable"):
        SessionOrchestratorV2.restore(
            power_rule_only_graph(),
            missing_capabilities,
            item_bank=approved_power_rule_bank(),
            pedagogy_catalog=approved_power_rule_catalog(),
        )


def test_capstone_retry_uses_another_family():
    session = _session()
    session.begin()
    session.submit(session.pending_expected)
    session.submit(session.pending_expected)
    first_family = session.pending.family_id
    interactions = session.submit("0")

    assert session.pending.kind == "capstone"
    assert session.pending.family_id != first_family
    assert session.phase.value == "capstone"
    assert any("worked pattern" in interaction.text for interaction in interactions)
    assert any("target is now uncertain" in interaction.text for interaction in interactions)
    summary = session.summary()
    assert TARGET not in summary["confirmed_mastery"]
    assert TARGET not in summary["confirmed_gaps"]
    assert TARGET in summary["uncertain"]


def test_capstone_graph_remediation_is_gated_before_display():
    session = _session()
    worked = session._allocator.reserve_item(
        session.exposure_state,
        kc_id=TARGET,
        surface=AssessmentSurface.WORKED_EXAMPLE,
    )
    session.exposure_state = worked.state
    session._nodes[TARGET] = session._nodes[TARGET].model_copy(
        update={
            "description": "The retry goal answer is -4*x^(-2) + 5*x^4.",
        }
    )
    session.begin()
    session.submit(session.pending_expected)
    session.submit(session.pending_expected)

    interactions = session.submit("0")

    assert session.phase.value == "stopped"
    assert any("answer-separation gate" in item.text for item in interactions)
    assert all("-4*x^(-2) + 5*x^4" not in item.text for item in interactions)


def test_runtime_rejects_packaged_draft_bank():
    from tutor.content.item_bank import load_item_bank

    with pytest.raises(ValueError, match="hard-ancestor closure is not fully released"):
        SessionOrchestratorV2(
            power_rule_only_graph(),
            TARGET,
            PROFILE,
            item_bank=load_item_bank(),
            pedagogy_catalog=approved_power_rule_catalog(),
        )


def test_runtime_rejects_incomplete_hard_ancestor_release():
    with pytest.raises(ValueError, match="kc.alg.exponent_rules"):
        SessionOrchestratorV2(
            load_graph(),
            TARGET,
            PROFILE,
            item_bank=approved_power_rule_bank(),
            pedagogy_catalog=approved_power_rule_catalog(),
        )


def test_runtime_rejects_unreviewed_content_declared_released():
    from tutor.content.item_bank import load_item_bank

    payload = load_item_bank().model_dump(mode="json")
    payload["released_kcs"] = [TARGET]
    unreviewed = ItemBankDocument.model_validate(payload)

    with pytest.raises(ValueError, match="not trusted.*not human_approved"):
        SessionOrchestratorV2(
            power_rule_only_graph(),
            TARGET,
            PROFILE,
            item_bank=unreviewed,
            pedagogy_catalog=approved_power_rule_catalog(),
        )
