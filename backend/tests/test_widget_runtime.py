"""Widget runtime: server-side scoring, formative evidence, API endpoint."""

import pytest
from fastapi.testclient import TestClient

from tutor.api.app import create_app
from tutor.orchestrator.machine import SessionOrchestrator, SessionPhase, _score_widget
from tutor.schemas.common import ResponseClass
from tutor.schemas.learner import LearnerProfile
from tutor.schemas.widgets import (
    ClickRegionWidget,
    LiveInputWidget,
    MappingWidget,
    SliderWidget,
)
from tutor.seed.load_seed import load_graph

PROFILE = LearnerProfile(course="AP Calculus AB", age_band="16-18")


@pytest.fixture(scope="module")
def graph():
    return load_graph()


def test_score_slider():
    widget = SliderWidget(
        learning_objective="o",
        prompt="drag",
        params={"min": 0, "max": 4, "step": 0.1},
        success_condition={"target": 2.0, "tolerance": 0.1},
    )
    assert _score_widget(widget, {"value": 2.05})
    assert not _score_widget(widget, {"value": 3.0})
    assert not _score_widget(widget, {"value": "not a number"})
    assert not _score_widget(widget, {})


def test_score_click_region():
    widget = ClickRegionWidget(
        learning_objective="o",
        prompt="click",
        regions=[{"id": "a"}, {"id": "b"}, {"id": "c"}],
        correct_region_ids=["a", "b"],
    )
    assert _score_widget(widget, {"selected": ["b", "a"]})
    assert not _score_widget(widget, {"selected": ["a"]})
    assert not _score_widget(widget, {"selected": ["a", "b", "c"]})
    assert not _score_widget(widget, {"selected": "a"})


def test_score_mapping():
    widget = MappingWidget(
        learning_objective="o",
        prompt="match",
        left=["p1", "p2"],
        right=["r1", "r2"],
        correct_pairs=[("p1", "r1"), ("p2", "r2")],
    )
    assert _score_widget(widget, {"pairs": [["p2", "r2"], ["p1", "r1"]]})
    assert not _score_widget(widget, {"pairs": [["p1", "r2"], ["p2", "r1"]]})
    assert not _score_widget(widget, {"pairs": [["p1", "r1"]]})
    assert not _score_widget(widget, {"pairs": "p1r1"})


def test_score_live_input():
    widget = LiveInputWidget(
        learning_objective="o",
        prompt="differentiate x^4",
        input_kind="expression",
        checker={"equivalence": "sympy_equiv", "expected": "4*x^3"},
    )
    assert _score_widget(widget, {"text": "4x^3"})
    assert not _score_widget(widget, {"text": "3x^4"})
    assert not _score_widget(widget, {"text": "   "})
    assert not _score_widget(widget, {})


def _drive_to_teach(orchestrator: SessionOrchestrator) -> list:
    orchestrator.begin()
    outputs = []
    guard = 0
    while orchestrator.phase == SessionPhase.DIAGNOSE:
        guard += 1
        assert guard < 50
        if orchestrator.pending_kc == "kc.der.chain_rule":
            answer = "totally wrong"
        else:
            answer = orchestrator.pending_expected
        outputs = orchestrator.submit(answer)
    assert orchestrator.phase == SessionPhase.TEACH
    return outputs


def test_machine_widget_records_single_evidence_event(graph):
    orchestrator = SessionOrchestrator(graph, "kc.der.chain_rule", PROFILE)
    outputs = _drive_to_teach(orchestrator)
    lesson = next(i for i in outputs if i.kind == "lesson" and i.widget)
    assert lesson.widget["widget_type"] == "live_input"

    events_before = len(orchestrator.learner.events)
    correct, message = orchestrator.answer_widget(
        lesson.key, {"text": lesson.widget["checker"]["expected"]}
    )
    assert correct
    assert message
    assert len(orchestrator.learner.events) == events_before + 1
    event = orchestrator.learner.events[-1]
    assert event.response_class == ResponseClass.WIDGET
    assert event.correct is True

    # retries are scored but never re-recorded (no mastery pumping)
    correct_again, _ = orchestrator.answer_widget(lesson.key, {"text": "garbage"})
    assert correct_again is False
    assert len(orchestrator.learner.events) == events_before + 1

    with pytest.raises(KeyError):
        orchestrator.answer_widget("nonexistent-key", {})


def test_widget_endpoint_over_http(graph):
    client = TestClient(create_app(graph))
    created = client.post("/sessions", json={"target_kc": "kc.der.chain_rule"}).json()
    session_id = created["session_id"]
    orchestrator = client.app.state.store.get(session_id)

    data = created
    guard = 0
    while data["phase"] == "diagnose":
        guard += 1
        assert guard < 50
        if orchestrator.pending_kc == "kc.der.chain_rule":
            answer = "totally wrong"
        else:
            answer = orchestrator.pending_expected
        data = client.post(f"/sessions/{session_id}/answer", json={"answer": answer}).json()

    lesson = next(i for i in data["interactions"] if i["kind"] == "lesson" and i["widget"])
    response = client.post(
        f"/sessions/{session_id}/widget",
        json={"key": lesson["key"], "response": {"text": lesson["widget"]["checker"]["expected"]}},
    )
    assert response.status_code == 200
    assert response.json()["correct"] is True

    wrong = client.post(
        f"/sessions/{session_id}/widget",
        json={"key": lesson["key"], "response": {"text": "nope"}},
    )
    assert wrong.json()["correct"] is False

    missing = client.post(
        f"/sessions/{session_id}/widget", json={"key": "zzz", "response": {}}
    )
    assert missing.status_code == 404
