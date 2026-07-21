"""Student-safe v2 input and release-identity contract tests."""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tutor.api.v2_schemas import PendingView, SessionView
from tutor.api.v2_store import V2SessionStore


@pytest.mark.parametrize(
    "answer_kind",
    [
        "symbolic",
        "numeric",
        "finite_set",
        "interval_set",
        "ordered_tuple",
        "antiderivative",
    ],
)
def test_each_production_answer_contract_has_a_typed_text_view(answer_kind):
    public = V2SessionStore._pending_input(
        input_mode="math",
        answer_spec=SimpleNamespace(kind=answer_kind, expected="private"),
        widget=None,
        widget_state=None,
    )

    assert public["type"] == "text"
    assert public["answer_kind"] == answer_kind
    assert public["label"]
    assert public["placeholder"]
    assert public["help_text"]
    assert public["max_length"] == 256
    assert "expected" not in public


def test_pending_input_models_cannot_represent_private_text_or_widget_truth():
    common = {
        "key": "pending-1",
        "kind": "probe",
        "kc_id": "kc.der.power_rule",
        "skill_name": "Power rule",
        "prompt": "Differentiate.",
        "prompt_segments": [],
        "hint": {
            "available": False,
            "next_index": 0,
            "total": 0,
            "next_reveals_answer": False,
        },
    }
    with pytest.raises(ValidationError, match="extra_forbidden"):
        PendingView.model_validate(
            {
                **common,
                "input": {
                    "type": "text",
                    "answer_kind": "symbolic",
                    "label": "Your expression",
                    "placeholder": "Enter an expression",
                    "help_text": "Use standard notation.",
                    "max_length": 256,
                    "expected": "3*x^2",
                },
            }
        )

    with pytest.raises(ValidationError, match="extra_forbidden"):
        PendingView.model_validate(
            {
                **common,
                "input": {
                    "type": "mapping_v1",
                    "prompt": "Match each row.",
                    "rows": [
                        {
                            "entry_id": "row.a",
                            "label": "A",
                            "spoken_text": "row A",
                        },
                        {
                            "entry_id": "row.b",
                            "label": "B",
                            "spoken_text": "row B",
                        },
                    ],
                    "options": [
                        {
                            "entry_id": "option.a",
                            "label": "One",
                            "spoken_text": "option one",
                        },
                        {
                            "entry_id": "option.b",
                            "label": "Two",
                            "spoken_text": "option two",
                        },
                    ],
                    "correct_pairs": [["row.a", "option.a"]],
                },
            }
        )


def test_legacy_pending_checkpoint_is_upgraded_without_loose_fields():
    pending = PendingView.model_validate(
        {
            "key": "legacy-1",
            "kind": "checkin",
            "kc_id": "kc.der.power_rule",
            "skill_name": "Power rule",
            "input_mode": "choice",
            "prompt": "Choose.",
            "choice_options": ["option.a", "option.b"],
            "widget": None,
            "widget_state": None,
            "hint": {
                "available": True,
                "next_index": 0,
                "total": 3,
                "next_reveals_answer": False,
            },
        }
    )

    payload = pending.model_dump(mode="json")
    assert payload["input"] == {
        "type": "legacy_choice",
        "label": "Choose an answer",
        "options": ["option.a", "option.b"],
    }
    assert "input_mode" not in payload
    assert "choice_options" not in payload
    assert "widget" not in payload


def test_schema_two_session_receipt_upgrades_to_typed_wire_version():
    receipt = SessionView.model_validate(
        {
            "schema_version": 2,
            "session_id": "legacy-session",
            "revision": 1,
            "phase": "diagnose",
            "durability": "durable",
            "goal": {
                "goal_id": "goal.der.power_rule",
                "target_kc": "kc.der.power_rule",
                "title": "Power rule",
                "description": "Differentiate powers.",
                "course_level": "calculus_1",
            },
            "profile": {"course": "calculus_1", "age_band": "legacy"},
            "content_mode": {
                "requested": "curated",
                "effective": "curated",
            },
            "transcript": [],
            "pending": None,
            "progress": {"phase": "diagnose"},
            "learner_summary": {},
            "started_at": "2026-07-20T12:00:00Z",
            "updated_at": "2026-07-20T12:00:01Z",
        }
    )

    assert receipt.schema_version == 3
