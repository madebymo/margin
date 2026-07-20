"""Versioned prompt contracts that protect widget answer boundaries."""

import json

from tutor.llm.prompts import (
    EVALUATOR_SYSTEM,
    INTERACTION_SYSTEM,
    PROMPT_VERSION,
    interaction_user,
)
from tutor.schemas.widgets import FeedbackRule
from tutor.seed.load_seed import load_graph


def test_p5_uses_opaque_region_ids_and_server_side_feedback():
    assert PROMPT_VERSION == "p5"
    worked_examples = INTERACTION_SYSTEM.split(
        "TWO WORKED EXAMPLES OF THE TARGET CALIBRE:\n\n", 1
    )[1]
    click_example = worked_examples.split("\n\nclick_region:\n", 1)[1]
    click_payload = json.loads(click_example.rsplit("\n\nWrite the candidates JSON now.", 1)[0])
    assert [region["id"] for region in click_payload["regions"]] == ["r1", "r2", "r3", "r4"]
    assert click_payload["correct_region_ids"] == ["r2"]

    assert "Click-region ids reach the student client" in INTERACTION_SYSTEM
    assert "Click-region ids must be opaque neutral tokens" in EVALUATOR_SYSTEM

    assert "frontend evaluates it live" not in INTERACTION_SYSTEM.lower()
    assert "server must evaluate it after an incorrect submission" in INTERACTION_SYSTEM
    assert "server-evaluated feedback_rules" in EVALUATOR_SYSTEM
    description = FeedbackRule.model_fields["when"].description or ""
    assert "server-side" in description
    assert "client widget config" in description


def test_p5_requests_directional_feedback_for_sliders_only():
    node = load_graph().nodes[0]
    prompt = interaction_user(node, attempt=0, feedback=[])
    directive = next(
        line for line in prompt.splitlines() if line.startswith("Make each widget self-describing")
    )

    assert "for sliders, directional feedback_rules" in directive
    assert "live_input" not in directive
