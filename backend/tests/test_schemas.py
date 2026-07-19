"""Schema validation contracts: widgets, evidence, mastery, lessons."""

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from tutor.schemas.learner import EvidenceEvent, MasteryEstimate
from tutor.schemas.lesson import CheckinItem, MiniLessonPackage
from tutor.schemas.widgets import (
    ClickRegionWidget,
    LiveInputWidget,
    MappingWidget,
    SliderWidget,
    parse_widget_config,
)

BASE = {"learning_objective": "objective", "prompt": "do the thing"}

VALID_SLIDER = {
    **BASE,
    "widget_type": "slider",
    "params": {"min": 0, "max": 4, "step": 0.1, "plot": "x^2"},
    "success_condition": {"target": 2.0, "tolerance": 0.1},
}
VALID_CLICK_REGION = {
    **BASE,
    "widget_type": "click_region",
    "regions": [{"id": "a"}, {"id": "b"}],
    "correct_region_ids": ["a"],
}
VALID_MAPPING = {
    **BASE,
    "widget_type": "mapping",
    "left": ["x^2", "sin(x)"],
    "right": ["2x", "cos(x)"],
    "correct_pairs": [["x^2", "2x"], ["sin(x)", "cos(x)"]],
}
VALID_LIVE_INPUT = {
    **BASE,
    "widget_type": "live_input",
    "input_kind": "expression",
    "checker": {"equivalence": "sympy_equiv", "expected": "2*x"},
}


@pytest.mark.parametrize(
    ("payload", "expected_type"),
    [
        (VALID_SLIDER, SliderWidget),
        (VALID_CLICK_REGION, ClickRegionWidget),
        (VALID_MAPPING, MappingWidget),
        (VALID_LIVE_INPUT, LiveInputWidget),
    ],
)
def test_every_widget_type_parses(payload, expected_type):
    assert isinstance(parse_widget_config(payload), expected_type)


def test_unknown_widget_type_rejected():
    with pytest.raises(ValidationError):
        parse_widget_config({**BASE, "widget_type": "hologram"})


def test_slider_step_must_be_positive():
    payload = {**VALID_SLIDER, "params": {"min": 0, "max": 4, "step": 0}}
    with pytest.raises(ValidationError):
        parse_widget_config(payload)


def test_mapping_pair_must_reference_existing_items():
    payload = {**VALID_MAPPING, "correct_pairs": [["x^3", "2x"]]}
    with pytest.raises(ValidationError):
        parse_widget_config(payload)


def _evidence_event() -> EvidenceEvent:
    return EvidenceEvent(
        event_id=uuid4(),
        learner_id=uuid4(),
        t=datetime.now(timezone.utc),
        item_id="item-1",
        kc_ids=["kc.der.chain_rule"],
        correct=False,
        response_class="symbolic_entry",
        hints_used=1,
        assisted=True,
        misconception_id="m.usub.forget_dx",
    )


def test_evidence_event_is_immutable():
    event = _evidence_event()
    with pytest.raises(ValidationError):
        event.correct = True  # type: ignore[misc]


def test_mastery_estimate_rejects_out_of_range():
    with pytest.raises(ValidationError):
        MasteryEstimate(direct=1.5, inferred=0.2)


def test_multiple_choice_checkin_requires_options():
    with pytest.raises(ValidationError):
        CheckinItem(
            item_id="c1",
            stem="pick one",
            kc_id="kc.int.u_substitution",
            response_class="multiple_choice",
            options=None,
            answer="a",
        )


def _lesson_payload(hint_ladder: list[str]) -> dict:
    return {
        "kc_id": "kc.int.u_substitution",
        "objective": "reverse the chain rule",
        "versions": {"graph": 1},
        "narrative": "story",
        "widgets": [VALID_LIVE_INPUT],
        "checkins": [
            {
                "item_id": "c1",
                "stem": "integrate 2x cos(x^2)",
                "kc_id": "kc.int.u_substitution",
                "response_class": "symbolic_entry",
                "answer": "sin(x**2) + C",
            }
        ],
        "math": {
            "canonical_form": "Integral(2*x*cos(x**2), x)",
            "answer_semantics": {"equivalence": "sympy_equiv"},
        },
        "hint_ladder": hint_ladder,
        "text_fallback": "worked example text",
        "applicability": {"profile_band": "hs-calc", "difficulty": "core"},
        "provenance": {"generator": "test", "telemetry_id": "tl-1"},
    }


def test_mini_lesson_accepts_exactly_three_hints():
    lesson = MiniLessonPackage.model_validate(_lesson_payload(["a", "b", "c"]))
    assert lesson.entry_exit.exit_consecutive_correct == 2


def test_mini_lesson_rejects_wrong_hint_count():
    with pytest.raises(ValidationError):
        MiniLessonPackage.model_validate(_lesson_payload(["a", "b"]))
