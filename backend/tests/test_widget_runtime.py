"""Widget runtime: server-side scoring, formative evidence, API endpoint."""

import logging
import math
from typing import get_args

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from tutor.api.app import create_app
from tutor.orchestrator.machine import SessionOrchestrator, SessionPhase, _score_widget
from tutor.schemas.common import ResponseClass
from tutor.schemas.learner import LearnerProfile
from tutor.schemas.widgets import (
    ClickRegionWidget,
    LiveInputWidget,
    MappingWidget,
    SliderWidget,
    WidgetConfig,
)
from tutor.seed.load_seed import load_graph

PROFILE = LearnerProfile(course="AP Calculus AB", age_band="16-18")
COMMON_CLIENT_FIELDS = {
    "schema_version",
    "learning_objective",
    "metaphor_id",
    "prompt",
    "widget_type",
}
CLIENT_SAFE_FIELD_PATHS = {
    SliderWidget: COMMON_CLIENT_FIELDS
    | {"params", "params.min", "params.max", "params.step", "params.plot", "params.shade"},
    ClickRegionWidget: COMMON_CLIENT_FIELDS
    | {"regions", "regions.id", "regions.label", "regions.shape"},
    MappingWidget: COMMON_CLIENT_FIELDS | {"left", "right"},
    LiveInputWidget: COMMON_CLIENT_FIELDS | {"input_kind", "render"},
}
ANSWER_BEARING_FIELD_PATHS = {
    SliderWidget: {
        "success_condition",
        "success_condition.target",
        "success_condition.tolerance",
        "feedback_rules",
        "feedback_rules.when",
        "feedback_rules.say",
    },
    ClickRegionWidget: {"correct_region_ids"},
    MappingWidget: {"correct_pairs"},
    LiveInputWidget: {
        "checker",
        "checker.equivalence",
        "checker.expected",
        "checker.tolerance",
    },
}


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


class _SingleWidgetGenerator:
    def __init__(self, widget):
        self.widget = widget

    def candidates(self, node, attempt, feedback):
        return [self.widget]


_INCORRECT_WIDGET_MESSAGE = "Not yet — adjust your answer and try again."


def _slider(
    *,
    plot="y = m*x",
    feedback_rules=None,
    target=10.0,
    tolerance=0.01,
):
    return SliderWidget(
        learning_objective="Interpret a changing graph",
        prompt="Adjust the graph.",
        params={"min": -100, "max": 100, "step": 0.01, "plot": plot},
        success_condition={"target": target, "tolerance": tolerance},
        feedback_rules=feedback_rules or [],
    )


def _answer_slider(graph, widget, value):
    orchestrator = SessionOrchestrator(graph, "kc.der.chain_rule", PROFILE)
    key = "slider-feedback-test"
    orchestrator._active_widgets[key] = ("kc.der.chain_rule", widget)
    return orchestrator.answer_widget(key, {"value": value})


def _nested_models(annotation):
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        yield annotation
        return
    for argument in get_args(annotation):
        yield from _nested_models(argument)


def _field_paths(model, prefix=""):
    assert model.model_config.get("extra") != "allow"
    for name, info in model.model_fields.items():
        path = f"{prefix}.{name}" if prefix else name
        yield path
        for nested in _nested_models(info.annotation):
            yield from _field_paths(nested, path)
    for name in model.model_computed_fields:
        yield f"{prefix}.{name}" if prefix else name


def _paths_in(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else key
            yield path
            yield from _paths_in(item, path)
    elif isinstance(value, list):
        for item in value:
            yield from _paths_in(item, prefix)


def test_every_widget_field_has_an_explicit_client_classification():
    widget_union = get_args(WidgetConfig)[0]
    widget_models = set(get_args(widget_union))

    assert widget_models == set(CLIENT_SAFE_FIELD_PATHS)
    assert widget_models == set(ANSWER_BEARING_FIELD_PATHS)
    for model in widget_models:
        safe = CLIENT_SAFE_FIELD_PATHS[model]
        answer_bearing = ANSWER_BEARING_FIELD_PATHS[model]
        assert safe.isdisjoint(answer_bearing)
        assert set(_field_paths(model)) == safe | answer_bearing


@pytest.mark.parametrize(
    ("widget", "correct_response"),
    [
        (
            SliderWidget(
                learning_objective="o",
                prompt="drag",
                params={"min": 0, "max": 4, "step": 0.1},
                success_condition={"target": 2.0, "tolerance": 0.1},
                feedback_rules=[
                    {"when": "x < 2", "say": "Move right."},
                    {"when": "x > 2", "say": "Move left."},
                ],
            ),
            {"value": 2.0},
        ),
        (
            ClickRegionWidget(
                learning_objective="o",
                prompt="click",
                regions=[{"id": "a"}, {"id": "b"}],
                correct_region_ids=["a"],
            ),
            {"selected": ["a"]},
        ),
        (
            MappingWidget(
                learning_objective="o",
                prompt="match",
                left=["p1", "p2"],
                right=["r1", "r2"],
                correct_pairs=[("p1", "r1"), ("p2", "r2")],
            ),
            {"pairs": [["p1", "r1"], ["p2", "r2"]]},
        ),
        (
            LiveInputWidget(
                learning_objective="o",
                prompt="differentiate x^4",
                input_kind="expression",
                checker={"equivalence": "sympy_equiv", "expected": "4*x^3"},
            ),
            {"text": "4*x^3"},
        ),
    ],
)
def test_machine_redacts_widget_answers_but_keeps_server_scoring(
    graph, widget, correct_response
):
    orchestrator = SessionOrchestrator(
        graph,
        "kc.der.chain_rule",
        PROFILE,
        interaction_generator=_SingleWidgetGenerator(widget),
    )
    outputs = _drive_to_teach(orchestrator)
    lesson = next(item for item in outputs if item.kind == "lesson" and item.widget)
    client_widget = lesson.model_dump(mode="json")["widget"]

    client_paths = set(_paths_in(client_widget))
    assert client_paths == CLIENT_SAFE_FIELD_PATHS[type(widget)]
    answer_paths = ANSWER_BEARING_FIELD_PATHS[type(widget)]
    assert answer_paths.isdisjoint(client_paths)
    correct, _ = orchestrator.answer_widget(lesson.key, correct_response)
    assert correct is True


@pytest.mark.parametrize(
    ("widget", "expected_light_field"),
    [
        (
            ClickRegionWidget(
                learning_objective="o",
                prompt="click",
                regions=[
                    {
                        "id": "r1",
                        "shape": {
                            "type": "point",
                            "x": "sqrt(2)",
                            "y": 1,
                            "expected": "SECRET",
                            "checker": {"expected": "SECRET"},
                        },
                    },
                    {"id": "r2", "shape": {"type": "point", "x": 0, "y": 0}},
                ],
                correct_region_ids=["r1"],
            ),
            {"type": "point", "x": "sqrt(2)", "y": 1},
        ),
        (
            LiveInputWidget(
                learning_objective="o",
                prompt="type",
                input_kind="expression",
                render={
                    "plot": "y = k*x",
                    "var": "k",
                    "expected": "SECRET",
                    "checker": {"expected": "SECRET"},
                },
                checker={"equivalence": "numeric", "expected": "2"},
            ),
            {"plot": "y = k*x", "var": "k"},
        ),
    ],
)
def test_machine_projects_free_form_light_fields_without_forwarding_invented_keys(
    graph, widget, expected_light_field
):
    orchestrator = SessionOrchestrator(
        graph,
        "kc.der.chain_rule",
        PROFILE,
        interaction_generator=_SingleWidgetGenerator(widget),
    )
    outputs = _drive_to_teach(orchestrator)
    lesson = next(item for item in outputs if item.kind == "lesson" and item.widget)
    client_widget = lesson.model_dump(mode="json")["widget"]

    if isinstance(widget, ClickRegionWidget):
        assert client_widget["regions"][0]["shape"] == expected_light_field
    else:
        assert client_widget["render"] == expected_light_field
    assert "SECRET" not in str(client_widget)
    assert "expected" not in str(client_widget)
    assert "checker" not in str(client_widget)


@pytest.mark.parametrize(
    "shape",
    [
        {"type": "polygon", "points": [[0, 0]], "expected": "SECRET"},
        {"type": "point", "x": {"expected": "SECRET"}, "y": 0},
        {"type": "point", "x": 10**400, "y": 0},
        {"type": "rect", "x": 0, "y": 0, "w": 1},
    ],
)
def test_machine_replaces_invalid_free_form_shapes_with_native_fallback(
    graph, shape
):
    widget = ClickRegionWidget(
        learning_objective="o",
        prompt="click",
        regions=[{"id": "r1", "shape": shape}, {"id": "r2"}],
        correct_region_ids=["r1"],
    )
    orchestrator = SessionOrchestrator(
        graph,
        "kc.der.chain_rule",
        PROFILE,
        interaction_generator=_SingleWidgetGenerator(widget),
    )
    outputs = _drive_to_teach(orchestrator)
    lesson = next(item for item in outputs if item.kind == "lesson" and item.widget)

    assert lesson.widget["regions"][0]["shape"] == {}
    assert "SECRET" not in str(lesson.widget)


@pytest.mark.parametrize(
    ("condition", "value", "matches"),
    [
        ("m < 2", 1.99, True),
        ("m < 2", 2.0, False),
        ("m <= 2", 2.0, True),
        ("m <= 2", 2.01, False),
        ("m > 2", 2.01, True),
        ("m > 2", 2.0, False),
        ("m >= 2", 2.0, True),
        ("m >= 2", 1.99, False),
    ],
)
def test_slider_feedback_comparators_respect_boundaries(graph, condition, value, matches):
    widget = _slider(
        feedback_rules=[{"when": condition, "say": "Boundary hint."}],
    )

    correct, message = _answer_slider(graph, widget, value)

    assert correct is False
    expected = (
        f"{_INCORRECT_WIDGET_MESSAGE} Boundary hint."
        if matches
        else _INCORRECT_WIDGET_MESSAGE
    )
    assert message == expected


@pytest.mark.parametrize(
    ("value", "expected_hint"),
    [
        (math.sqrt(2), "At or below sqrt(2)."),
        (math.pi / 2, "At or above pi/2."),
    ],
)
def test_slider_feedback_accepts_exact_constant_thresholds(graph, value, expected_hint):
    widget = _slider(
        feedback_rules=[
            {"when": "m <= sqrt(2)", "say": "At or below sqrt(2)."},
            {"when": "m >= pi/2", "say": "At or above pi/2."},
        ]
    )

    correct, message = _answer_slider(graph, widget, value)

    assert correct is False
    assert message == f"{_INCORRECT_WIDGET_MESSAGE} {expected_hint}"


@pytest.mark.parametrize(
    ("parameter", "plot"),
    [
        ("theta", "y = theta*sin(x)"),
        ("rate", "y = rate*x + 1e3"),
        ("speed", "y = speed*x + 1e-3"),
        ("a1", "y = a1*sin(x)"),
    ],
)
def test_slider_feedback_infers_lexical_parameter_and_returns_first_match(
    graph, parameter, plot
):
    widget = _slider(
        plot=plot,
        feedback_rules=[
            {"when": f"{parameter} > 1", "say": "First matching hint."},
            {
                "when": f"{parameter} >= sqrt(2)",
                "say": "Second matching hint.",
            },
        ],
    )

    correct, message = _answer_slider(graph, widget, math.sqrt(2))

    assert correct is False
    assert message == f"{_INCORRECT_WIDGET_MESSAGE} First matching hint."
    assert "Second matching hint." not in message


def test_slider_feedback_is_suppressed_for_correct_attempt(graph):
    widget = _slider(
        target=2.0,
        tolerance=0.1,
        feedback_rules=[{"when": "m >= 2", "say": "This must not be returned."}],
    )

    correct, message = _answer_slider(graph, widget, 2.05)

    assert correct is True
    assert message == "Nice — that's it."


def test_invalid_slider_feedback_rules_are_logged_skipped_and_never_exposed(
    graph, caplog
):
    caplog.set_level(logging.WARNING, logger="tutor.orchestrator")
    widget = _slider(
        feedback_rules=[
            {"when": "m == 2", "say": "Equality is unsupported."},
            {"when": "2 < m", "say": "The parameter must be on the left."},
            {"when": "a < 2", "say": "Wrong parameter."},
            {"when": "m < q", "say": "Symbolic threshold."},
            {"when": "m < 1/0", "say": "Infinite threshold."},
            {"when": "m < sqrt(-1)", "say": "Complex threshold."},
            {"when": "m < y=2", "say": "Assignment threshold."},
            {"when": "m < 1,2", "say": "Tuple threshold."},
            {"when": "m < 3", "say": "Valid fallback hint."},
        ]
    )

    correct, message = _answer_slider(graph, widget, 2.0)

    assert correct is False
    assert message == f"{_INCORRECT_WIDGET_MESSAGE} Valid fallback hint."
    assert all(rule.say not in message for rule in widget.feedback_rules[:-1])
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert logged.count("ignoring slider feedback rule") == 8
    assert "expected one comparison" in logged
    assert "does not match slider parameter" in logged
    assert "must not contain free symbols" in logged
    assert "threshold must be finite" in logged
    assert "threshold must be real" in logged
    assert "not an assignment" in logged


@pytest.mark.parametrize(
    "plot",
    [
        None,
        "m*x",
        "z = m*x",
        "y = x",
        "y = m",
        "y = m*x + b",
        "y = m*x = 2",
        "y = m*x, 2",
        "y = m*x + @",
    ],
)
def test_slider_feedback_logs_and_ignores_invalid_parameter_plots(graph, caplog, plot):
    caplog.set_level(logging.WARNING, logger="tutor.orchestrator")
    widget = _slider(
        plot=plot,
        feedback_rules=[{"when": "m < 3", "say": "Must stay server-only."}],
    )

    correct, message = _answer_slider(graph, widget, 2.0)

    assert correct is False
    assert message == _INCORRECT_WIDGET_MESSAGE
    assert "Must stay server-only." not in message
    assert any(
        "ignoring slider feedback rules" in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.parametrize("value", [None, "not a number", "nan", "inf", "-inf"])
def test_slider_feedback_requires_a_finite_submitted_value(graph, value):
    widget = _slider(
        feedback_rules=[{"when": "m < 3", "say": "Must not match invalid input."}]
    )

    correct, message = _answer_slider(graph, widget, value)

    assert correct is False
    assert message == _INCORRECT_WIDGET_MESSAGE


def test_widget_endpoint_returns_submit_only_slider_feedback(graph):
    app = create_app(graph, allow_v1_session_creation=True)
    client = TestClient(app)
    orchestrator = SessionOrchestrator(graph, "kc.der.chain_rule", PROFILE)
    orchestrator.begin()
    key = "http-slider-feedback"
    orchestrator._active_widgets[key] = (
        "kc.der.chain_rule",
        _slider(
            feedback_rules=[
                {"when": "m < pi/2", "say": "Increase the slope toward the marker."}
            ]
        ),
    )
    session_id = app.state.store.create(orchestrator)

    response = client.post(
        f"/sessions/{session_id}/widget",
        json={"key": key, "response": {"value": 1.0}},
    )

    assert response.status_code == 200
    assert response.json() == {
        "correct": False,
        "message": (
            f"{_INCORRECT_WIDGET_MESSAGE} Increase the slope toward the marker."
        ),
    }


def test_machine_widget_records_complete_formative_attempt_trajectory(graph):
    orchestrator = SessionOrchestrator(graph, "kc.der.chain_rule", PROFILE)
    outputs = _drive_to_teach(orchestrator)
    lesson = next(i for i in outputs if i.kind == "lesson" and i.widget)
    assert lesson.widget["widget_type"] == "live_input"
    _, server_widget = orchestrator._active_widgets[lesson.key]
    assert isinstance(server_widget, LiveInputWidget)

    events_before = len(orchestrator.learner.events)
    correct, message = orchestrator.answer_widget(
        lesson.key, {"text": server_widget.checker.expected}
    )
    assert correct
    assert message
    assert len(orchestrator.learner.events) == events_before + 1
    event = orchestrator.learner.events[-1]
    assert event.response_class == ResponseClass.WIDGET
    assert event.correct is True

    # Every formative attempt is retained; v2 learner models exclude widget
    # events from mastery rather than discarding the trajectory.
    correct_again, _ = orchestrator.answer_widget(lesson.key, {"text": "garbage"})
    assert correct_again is False
    assert len(orchestrator.learner.events) == events_before + 2
    assert orchestrator.learner.events[-1].attempt_number == 2
    assert orchestrator.learner.events[-1].surface == "guided_widget"

    with pytest.raises(KeyError):
        orchestrator.answer_widget("nonexistent-key", {})


def test_widget_endpoint_over_http(graph):
    client = TestClient(create_app(graph, allow_v1_session_creation=True))
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
    _, server_widget = orchestrator._active_widgets[lesson["key"]]
    assert isinstance(server_widget, LiveInputWidget)
    client_paths = set(_paths_in(lesson["widget"]))
    assert client_paths == CLIENT_SAFE_FIELD_PATHS[type(server_widget)]
    answer_paths = ANSWER_BEARING_FIELD_PATHS[type(server_widget)]
    assert answer_paths.isdisjoint(client_paths)
    response = client.post(
        f"/sessions/{session_id}/widget",
        json={"key": lesson["key"], "response": {"text": server_widget.checker.expected}},
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
