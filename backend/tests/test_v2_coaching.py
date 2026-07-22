"""OpenAI coaching stays visible but cannot influence trusted session state."""

import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tutor.api.v2 import install_v2_routes
from tutor.llm.coaching import (
    CoachingContext,
    CoachingMessage,
    CoachingOutput,
    OpenAIResponsesCoach,
)

from tests.v2_helpers import (
    approved_power_rule_catalog,
    approved_power_rule_stress_bank,
    power_rule_only_graph,
)


def _context() -> CoachingContext:
    return CoachingContext(
        phase="diagnose",
        surface="diagnostic",
        skill_label="Power rule",
        outcome="incorrect",
        assisted=False,
        attempt_number=1,
        mastery_status="uncertain",
        transition="continue_assessment",
        reviewed_misconception=None,
        reviewed_remediation=None,
    )


class _ResponsesStub:
    def __init__(self, output=None, error: Exception | None = None) -> None:
        self.output = output
        self.error = error
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(output_parsed=self.output, output=[])


class _SDKStub:
    def __init__(self, responses: _ResponsesStub) -> None:
        self.responses = responses
        self.options: list[dict] = []

    def with_options(self, **kwargs):
        self.options.append(kwargs)
        return self


def test_responses_coach_uses_structured_stateless_bounded_request():
    responses = _ResponsesStub(
        CoachingOutput(
            message="Pause and identify the operation before choosing a strategy.",
            focus="strategy",
        )
    )
    sdk = _SDKStub(responses)
    coach = OpenAIResponsesCoach(client=sdk, timeout_seconds=2.5)

    result = coach.coach(
        _context(),
        safety_identifier="a" * 64,
    )

    assert result == CoachingMessage(
        text="Pause and identify the operation before choosing a strategy.",
        focus="strategy",
        model="gpt-5.6-terra",
    )
    assert sdk.options == [{"timeout": 2.5, "max_retries": 0}]
    request = responses.calls[0]
    assert request["model"] == "gpt-5.6-terra"
    assert request["reasoning"] == {"effort": "low"}
    assert request["text_format"] is CoachingOutput
    assert request["store"] is False
    assert request["tools"] == []
    assert request["safety_identifier"] == "a" * 64
    assert "previous_response_id" not in request
    assert set(json.loads(request["input"])) == set(_context().model_dump())


def test_responses_coach_routes_deep_explanations_to_sol_and_falls_back():
    success = _ResponsesStub(
        CoachingOutput(message="Connect this idea to the worked example.", focus="concept")
    )
    coach = OpenAIResponsesCoach(client=_SDKStub(success))
    message = coach.coach(
        _context(),
        safety_identifier="b" * 64,
        deep_explanation=True,
    )

    assert message is not None
    assert message.model == "gpt-5.6-sol"
    assert success.calls[0]["reasoning"] == {"effort": "medium"}

    failed = OpenAIResponsesCoach(
        client=_SDKStub(_ResponsesStub(error=RuntimeError("private provider body")))
    )
    assert failed.coach(_context(), safety_identifier="c" * 64) is None


def test_coaching_output_rejects_extra_fields():
    with pytest.raises(ValidationError):
        CoachingOutput.model_validate(
            {
                "message": "Keep going.",
                "focus": "next_step",
                "score": 1,
            }
        )


class _CoachStub:
    provider = "openai"
    models = ("gpt-5.6-terra", "gpt-5.6-sol")
    policy_version = "coach-v1"

    def __init__(self, text: str | None, *, raises: bool = False) -> None:
        self.text = text
        self.raises = raises
        self.calls: list[tuple[CoachingContext, str, bool]] = []

    def coach(
        self,
        context: CoachingContext,
        *,
        safety_identifier: str,
        deep_explanation: bool = False,
    ) -> CoachingMessage | None:
        self.calls.append((context, safety_identifier, deep_explanation))
        if self.raises:
            raise RuntimeError("private provider body")
        if self.text is None:
            return None
        return CoachingMessage(
            text=self.text,
            focus="strategy",
            model="gpt-5.6-sol" if deep_explanation else "gpt-5.6-terra",
        )


def _app(coach=None) -> FastAPI:
    app = FastAPI()
    install_v2_routes(
        app,
        power_rule_only_graph(),
        available_targets=("kc.der.power_rule",),
        item_bank=approved_power_rule_stress_bank(),
        pedagogy_catalog=approved_power_rule_catalog(),
        resume_token_secret=b"coaching-test-secret-that-is-long-enough",
        coach=coach,
    )
    return app


def _create(client: TestClient, *, content_mode: str = "llm_coaching") -> dict:
    response = client.post(
        "/api/v2/sessions",
        json={
            "request_id": str(uuid4()),
            "goal_id": "goal.der.power_rule",
            "content_mode": content_mode,
            "context": "raw learner context must stay private",
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _answer(view: dict, answer: str, request_id) -> dict:
    return {
        "type": "answer",
        "request_id": str(request_id),
        "expected_revision": view["revision"],
        "pending_key": view["pending"]["key"],
        "answer": answer,
    }


def test_configured_coach_is_attributed_atomic_and_not_recalled_on_replay():
    coach = _CoachStub(
        "You identified the operation. Keep naming the pattern before acting."
    )
    client = TestClient(_app(coach))
    catalog = client.get("/api/v2/goals").json()
    assert catalog["coaching"] == {
        "available": True,
        "provider": "openai",
        "model": "gpt-5.6",
        "reason": "GPT-5.6 coaching is available; scoring remains deterministic.",
    }
    view = _create(client)
    assert view["content_mode"] == {
        "requested": "llm_coaching",
        "effective": "llm_coaching",
        "fallback_reason": None,
    }
    handle = client.app.state.v2_store.get(view["session_id"])
    raw_answer = handle.orchestrator.pending_expected
    raw_prompt = view["pending"]["prompt_segments"]
    request_id = uuid4()
    action = _answer(view, raw_answer, request_id)
    before_events = len(handle.orchestrator.learner.events)

    response = client.post(
        f"/api/v2/sessions/{view['session_id']}/actions",
        json=action,
    )

    assert response.status_code == 200, response.text
    after = response.json()
    coach_entries = [entry for entry in after["transcript"] if entry["kind"] == "coach"]
    assert len(coach_entries) == 1
    assert coach_entries[0]["generated_by"] == {
        "provider": "openai",
        "model": "gpt-5.6-terra",
        "policy_version": "coach-v1",
        "focus": "strategy",
    }
    assert after["revision"] == view["revision"] + 1
    assert len(handle.orchestrator.learner.events) == before_events + 1
    assert len(coach.calls) == 1
    sent_context, safety_identifier, _ = coach.calls[0]
    serialized = sent_context.model_dump_json()
    assert raw_answer not in serialized
    assert json.dumps(raw_prompt) not in serialized
    assert "raw learner context" not in serialized
    assert len(safety_identifier) == 64
    assert handle.learner_id not in safety_identifier

    replay = client.post(
        f"/api/v2/sessions/{view['session_id']}/actions",
        json=action,
    )
    assert replay.status_code == 200
    assert replay.json() == after
    assert len(coach.calls) == 1
    assert len(handle.orchestrator.learner.events) == before_events + 1


def test_provider_failure_keeps_curated_state_transition():
    coach = _CoachStub(None, raises=True)
    client = TestClient(_app(coach))
    view = _create(client)
    handle = client.app.state.v2_store.get(view["session_id"])
    action = _answer(view, handle.orchestrator.pending_expected, uuid4())

    response = client.post(
        f"/api/v2/sessions/{view['session_id']}/actions",
        json=action,
    )

    assert response.status_code == 200
    after = response.json()
    assert after["revision"] == view["revision"] + 1
    assert not any(entry["kind"] == "coach" for entry in after["transcript"])
    assert len(handle.orchestrator.learner.events) == 1
    assert len(coach.calls) == 1


def test_coaching_that_leaks_the_new_pending_answer_is_rejected():
    coach = _CoachStub("The next answer is 4*x^3.")
    client = TestClient(_app(coach))
    view = _create(client)
    handle = client.app.state.v2_store.get(view["session_id"])

    response = client.post(
        f"/api/v2/sessions/{view['session_id']}/actions",
        json=_answer(view, handle.orchestrator.pending_expected, uuid4()),
    )

    assert response.status_code == 200
    after = response.json()
    assert handle.orchestrator.pending_expected == "4*x^3"
    assert not any(entry["kind"] == "coach" for entry in after["transcript"])
    assert client.app.state.v2_store.metrics_snapshot()["counters"][
        "coaching_leakage_rejections"
    ] == 1


def test_unconfigured_coaching_request_is_honestly_curated():
    client = TestClient(_app())
    catalog = client.get("/api/v2/goals").json()
    assert catalog["coaching"]["available"] is False

    view = _create(client)
    assert view["content_mode"]["requested"] == "llm_coaching"
    assert view["content_mode"]["effective"] == "curated"
    assert "unavailable" in view["content_mode"]["fallback_reason"].lower()


def test_v2_rejects_an_unimplemented_coaching_provider():
    client = TestClient(_app(_CoachStub("Keep going.")))

    response = client.post(
        "/api/v2/sessions",
        json={
            "request_id": str(uuid4()),
            "goal_id": "goal.der.power_rule",
            "content_mode": "llm_coaching",
            "provider": "anthropic",
        },
    )

    assert response.status_code == 422
