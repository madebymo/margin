"""Confirmation and uncertainty invariants for diagnosis policy v2."""

from tutor.learner.service_v2 import LearnerModelServiceV2
from tutor.orchestrator.diagnosis_v2 import (
    DIAGNOSIS_POLICY_VERSION,
    PINNED_IMPACT_DECAY,
    PINNED_IMPACT_LAMBDA,
    DiagnosticObservation,
    DiagnosisControllerV2,
)
from tutor.schemas.common import ResponseClass
from tutor.seed.load_seed import load_graph

TARGET = "kc.int.u_substitution"
FLOOR = {"Algebra 1", "Algebra 2", "Precalculus"}


def _controller(budget: int = 8) -> DiagnosisControllerV2:
    graph = load_graph()
    learner = LearnerModelServiceV2(graph, assumed_floor_levels=FLOOR)
    return DiagnosisControllerV2(graph, TARGET, learner, probe_budget=budget)


def _record(
    controller: DiagnosisControllerV2,
    family: str,
    correct: bool,
    kc: str = TARGET,
) -> None:
    controller.record_result(
        DiagnosticObservation(kc_id=kc, family_id=family, correct=correct)
    )


def test_release_policy_uses_the_grid_sweep_winner():
    controller = _controller()
    assert DIAGNOSIS_POLICY_VERSION == "diagnosis-v2.1"
    assert controller.state.impact_lambda == PINNED_IMPACT_LAMBDA == 0
    assert controller.state.impact_decay == PINNED_IMPACT_DECAY == 0.25


def test_target_success_requires_distinct_family_confirmation():
    controller = _controller()
    assert controller.next_probe().reason == "target_first"
    _record(controller, "target-a", True)

    confirmation = controller.next_probe()
    assert confirmation.kc_id == TARGET
    assert confirmation.reason == "independent_confirmation"
    _record(controller, "target-b", True)

    assert controller.next_probe() is None
    assert controller.state.stop_reason == "target_confirmed_mastered"
    assert controller.status(TARGET) == "confirmed_mastered"
    assert controller.probes_issued == 2


def test_choice_successes_never_confirm_production_mastery():
    controller = _controller()
    controller.next_probe()
    controller.record_result(
        DiagnosticObservation(
            kc_id=TARGET,
            family_id="choice-a",
            correct=True,
            response_class=ResponseClass.MULTIPLE_CHOICE,
        )
    )
    controller.next_probe()
    controller.record_result(
        DiagnosticObservation(
            kc_id=TARGET,
            family_id="choice-b",
            correct=True,
            response_class=ResponseClass.MULTIPLE_CHOICE,
        )
    )

    assert controller.status(TARGET) == "uncertain"
    assert controller.learner_summary()["confirmed_mastered"] == []


def test_symmetric_miss_confirmation_and_honest_plan():
    controller = _controller()
    controller.next_probe()
    _record(controller, "target-a", False)
    assert controller.next_probe().kc_id == TARGET
    _record(controller, "target-b", False)

    assert controller.status(TARGET) == "confirmed_gap"
    next_probe = controller.next_probe()
    assert next_probe is not None
    assert next_probe.reason == "information_gain"

    kinds = [step.kind for step in controller.learning_plan()]
    assert kinds[-2:] == ["teach_confirmed_gap", "practice_target"]


def test_conflict_requires_third_family_when_budget_allows():
    controller = _controller()
    controller.next_probe()
    _record(controller, "target-a", True)
    controller.next_probe()
    _record(controller, "target-b", False)

    third = controller.next_probe()
    assert third.kc_id == TARGET
    assert third.reason == "independent_confirmation"
    _record(controller, "target-c", True)

    assert controller.next_probe() is None
    assert controller.state.stop_reason == "target_conflict_uncertain"
    assert controller.status(TARGET) == "uncertain"
    assert controller.probes_issued == 3
    assert controller.learning_plan()[0].kind == "verify_uncertain"


def test_reusing_a_family_is_rejected():
    controller = _controller()
    controller.next_probe()
    _record(controller, "target-a", False)
    controller.next_probe()

    try:
        _record(controller, "target-a", False)
    except ValueError as exc:
        assert "already been used" in str(exc)
    else:  # pragma: no cover - guards the safety invariant explicitly
        raise AssertionError("family reuse was accepted")


def test_budget_exhaustion_preserves_uncertainty():
    controller = _controller(budget=1)
    controller.next_probe()
    _record(controller, "target-a", True)

    assert controller.next_probe() is None
    assert controller.state.stop_reason == "budget_exhausted"
    assert controller.status(TARGET) == "uncertain"
    assert controller.learner_summary()["uncertain"]


def test_assisted_attempts_consume_families_without_looping_on_one_kc():
    controller = _controller()
    for index in range(3):
        selection = controller.next_probe()
        assert selection is not None
        assert selection.kc_id == TARGET
        controller.record_result(
            DiagnosticObservation(
                kc_id=TARGET,
                family_id=f"assisted-{index}",
                correct=True,
                assisted=True,
            )
        )

    next_selection = controller.next_probe()
    assert next_selection is not None
    assert next_selection.kc_id != TARGET
