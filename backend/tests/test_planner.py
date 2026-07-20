"""Lesson planner: gates, evaluator verdicts, repair loop, fallback, machine wiring."""

import math

import pytest

from tutor.llm.client import LLMError
from tutor.llm.evaluator import LLMEvaluator
from tutor.llm.interaction import LLMInteractionGenerator
from tutor.orchestrator.machine import SessionOrchestrator, SessionPhase
from tutor.orchestrator.planner import (
    LessonPlanner,
    deterministic_gates,
)
from tutor.schemas.learner import LearnerProfile
from tutor.schemas.widgets import LiveInputWidget, SliderWidget
from tutor.seed.load_seed import load_graph
from tutor.verify.checker import check_answer

PROFILE = LearnerProfile(course="AP Calculus AB", age_band="16-18")

VALID_WIDGET = {
    "widget_type": "live_input",
    "learning_objective": "practice the power rule",
    "prompt": "Differentiate x^4 with respect to x.",
    "input_kind": "expression",
    "checker": {"equivalence": "sympy_equiv", "expected": "4*x^3"},
}
SECOND_WIDGET = {**VALID_WIDGET, "prompt": "Differentiate x^6 with respect to x.",
                 "checker": {"equivalence": "sympy_equiv", "expected": "6*x^5"}}
ACCEPT_VERDICT = {
    "hard": {"correctness": True, "alignment": True, "consistency": True, "safety": True},
    "soft": {"clarity": 5, "scaffolding": 4, "cognitive_load": 4, "engagement": 4, "age_fit": 5},
    "abstain": False,
    "feedback": "solid",
}
REJECT_VERDICT = {
    "hard": {"correctness": False, "alignment": True, "consistency": True, "safety": True},
    "soft": {"clarity": 5, "scaffolding": 5, "cognitive_load": 5, "engagement": 5, "age_fit": 5},
    "abstain": False,
    "feedback": "the expected derivative is wrong",
}


class FakeLLM:
    """Tag-prefix-routed fake. Values: dict, list of dicts (consumed), or Exception."""

    def __init__(self, handlers: dict[str, object]) -> None:
        self._handlers = handlers

    def complete_json(self, *, system: str, user: str, tag: str) -> dict:
        for prefix, response in self._handlers.items():
            if tag.startswith(prefix):
                if isinstance(response, Exception):
                    raise response
                if isinstance(response, list):
                    if not response:
                        raise LLMError("handler exhausted")
                    return response.pop(0)
                return dict(response)
        raise LLMError(f"no handler for {tag}")


@pytest.fixture(scope="module")
def graph():
    return load_graph()


def _node(graph, kc_id):
    return next(node for node in graph.nodes if node.id == kc_id)


def test_template_planner_produces_gated_widget(graph):
    planner = LessonPlanner()
    planned = planner.plan_lesson(_node(graph, "kc.der.power_rule"))
    assert not planned.fallback_used
    assert planned.widget is not None
    assert deterministic_gates(planned.widget) == []
    # the widget's hidden answer is verifiable math
    assert check_answer(planned.widget.checker.expected, planned.widget.checker.expected)


def test_deterministic_gate_catches_answer_leak():
    leaky = LiveInputWidget(
        learning_objective="leak",
        prompt="The answer is 4*x^3 — type 4*x^3.",
        input_kind="expression",
        checker={"equivalence": "sympy_equiv", "expected": "4*x^3"},
    )
    assert any("leak" in problem for problem in deterministic_gates(leaky))


@pytest.mark.parametrize(
    ("prompt", "say"),
    [
        ("Set the slope to 1.5.", "Move the line toward the marker."),
        ("Move the line through the marker.", "The target is 1.50."),
        ("Move the line through the marker.", "Aim for 3/2 next."),
    ],
)
def test_deterministic_gate_catches_slider_target_leaks(prompt, say):
    leaky = SliderWidget(
        learning_objective="Interpret slope",
        prompt=prompt,
        params={
            "min": -1,
            "max": 4,
            "step": 0.1,
            "plot": "y = m*x",
            "shade": "point(2, 3)",
        },
        success_condition={"target": 1.5, "tolerance": 0.1},
        feedback_rules=[{"when": "m < 1.5", "say": say}],
    )

    assert any("target leaks" in problem for problem in deterministic_gates(leaky))


def test_deterministic_gate_allows_hidden_rule_threshold_and_goal_marker():
    safe = SliderWidget(
        learning_objective="Interpret slope",
        prompt="Move the line through the marker at (2, 3).",
        params={
            "min": -1,
            "max": 4,
            "step": 0.1,
            "plot": "y = m*x",
            "shade": "point(2, 3)",
        },
        success_condition={"target": 1.5, "tolerance": 0.1},
        feedback_rules=[
            {
                "when": "m < 1.5",
                "say": "The line rises too slowly; increase the slope.",
            }
        ],
    )

    assert deterministic_gates(safe) == []


@pytest.mark.parametrize(
    ("target", "say"),
    [
        (1000.0, "The answer is 1e3."),
        (1000.0, "The answer is 1,000."),
        (math.pi / 2, "The answer is pi/2."),
        (math.sqrt(2), "The answer is sqrt(2)."),
    ],
)
def test_deterministic_gate_catches_formatted_slider_target_leaks(target, say):
    leaky = SliderWidget(
        learning_objective="Interpret slope",
        prompt="Move the line through the marker.",
        params={"min": -2000, "max": 2000, "step": 0.1, "plot": "y = m*x"},
        success_condition={"target": target, "tolerance": 0.1},
        feedback_rules=[{"when": "m < 0", "say": say}],
    )

    assert any("target leaks" in problem for problem in deterministic_gates(leaky))


def test_deterministic_gate_exempts_authorized_marker_coordinate_collision():
    safe = SliderWidget(
        learning_objective="Interpret slope",
        prompt="Move the line through the marker at (2, 0).",
        params={
            "min": -1,
            "max": 1,
            "step": 0.1,
            "plot": "y = m*x",
            "shade": "point(2, 0)",
        },
        success_condition={"target": 0, "tolerance": 0.1},
        feedback_rules=[
            {
                "when": "m < 0",
                "say": "The line misses the marker at (2, 0); increase the slope.",
            }
        ],
    )

    assert deterministic_gates(safe) == []


def test_llm_generator_with_accepting_judge(graph):
    planner = LessonPlanner(
        generator=LLMInteractionGenerator(FakeLLM({"interaction:": {"candidates": [VALID_WIDGET]}})),
        evaluator=LLMEvaluator(FakeLLM({"evaluate:": ACCEPT_VERDICT})),
    )
    planned = planner.plan_lesson(_node(graph, "kc.der.power_rule"))
    assert not planned.fallback_used
    assert planned.widget is not None
    assert planned.widget.checker.expected == "4*x^3"


def test_reject_then_repair_uses_second_candidate(graph):
    generator = LLMInteractionGenerator(
        FakeLLM({"interaction:": [
            {"candidates": [VALID_WIDGET]},
            {"candidates": [SECOND_WIDGET]},
        ]})
    )
    evaluator = LLMEvaluator(FakeLLM({"evaluate:": [REJECT_VERDICT, ACCEPT_VERDICT]}))
    planner = LessonPlanner(generator=generator, evaluator=evaluator)
    planned = planner.plan_lesson(_node(graph, "kc.der.power_rule"))
    assert not planned.fallback_used
    assert planned.widget.checker.expected == "6*x^5"
    assert any("hard gate failed" in item for item in planned.evaluator_feedback)


def test_all_rejected_falls_back_to_worked_example(graph):
    planner = LessonPlanner(
        generator=LLMInteractionGenerator(FakeLLM({"interaction:": {"candidates": [VALID_WIDGET]}})),
        evaluator=LLMEvaluator(FakeLLM({"evaluate:": REJECT_VERDICT})),
    )
    planned = planner.plan_lesson(_node(graph, "kc.der.power_rule"))
    assert planned.fallback_used
    assert planned.widget is None
    assert "worked example" in planned.narrative.lower()


def test_abstention_and_low_soft_scores_reject(graph):
    node = _node(graph, "kc.der.power_rule")
    widget = LiveInputWidget.model_validate(VALID_WIDGET)
    abstain = LLMEvaluator(FakeLLM({"evaluate:": {**ACCEPT_VERDICT, "abstain": True}}))
    assert abstain.evaluate(node, "narrative", widget).accepted is False
    low_soft = LLMEvaluator(
        FakeLLM({"evaluate:": {**ACCEPT_VERDICT, "soft": {"clarity": 2, "scaffolding": 5,
                 "cognitive_load": 5, "engagement": 5, "age_fit": 5}}})
    )
    assert low_soft.evaluate(node, "narrative", widget).accepted is False
    unavailable = LLMEvaluator(FakeLLM({"evaluate:": LLMError("down")}))
    verdict = unavailable.evaluate(node, "narrative", widget)
    assert verdict.accepted is False
    assert "unavailable" in verdict.feedback


def test_machine_lessons_carry_widgets_in_template_mode(graph):
    orchestrator = SessionOrchestrator(graph, "kc.der.chain_rule", PROFILE)
    outputs = list(orchestrator.begin())
    lesson_widgets = []
    guard = 0
    while orchestrator.phase not in (SessionPhase.DONE, SessionPhase.STOPPED):
        guard += 1
        assert guard < 100
        if (
            orchestrator.phase == SessionPhase.DIAGNOSE
            and orchestrator.pending_kc == "kc.der.chain_rule"
        ):
            answer = "totally wrong"
        else:
            answer = orchestrator.pending_expected
        outputs = orchestrator.submit(answer)
        lesson_widgets.extend(
            item.widget for item in outputs if item.kind == "lesson" and item.widget
        )
    assert orchestrator.phase == SessionPhase.DONE
    assert lesson_widgets, "teach-loop lessons should carry an interactive widget"
    assert all("widget_type" in widget for widget in lesson_widgets)
