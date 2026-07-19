"""Provider client adapters, tested against stubbed SDK objects."""

from types import SimpleNamespace

import pytest

from tutor.llm.client import LLMError, OpenAILLMClient


def _stub_openai(text: str, capture: dict):
    def create(**kwargs):
        capture.update(kwargs)
        message = SimpleNamespace(content=text)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_openai_client_parses_json_and_records_call():
    capture: dict = {}
    client = OpenAILLMClient(client=_stub_openai('{"a": 1}', capture), model="test-model")
    result = client.complete_json(system="sys", user="usr", tag="probe:kc.der.power_rule")
    assert result == {"a": 1}
    assert capture["model"] == "test-model"
    assert capture["response_format"] == {"type": "json_object"}
    assert capture["messages"][0] == {"role": "system", "content": "sys"}
    assert len(client.calls) == 1
    assert client.calls[0].tag == "probe:kc.der.power_rule"
    assert client.calls[0].model == "test-model"


def test_openai_client_wraps_bad_output_in_llm_error():
    capture: dict = {}
    client = OpenAILLMClient(client=_stub_openai("not json at all", capture), model="m")
    with pytest.raises(LLMError):
        client.complete_json(system="s", user="u", tag="t")


def test_openai_client_normalizes_provider_errors():
    def create(**kwargs):
        raise RuntimeError("rate limited")

    stub = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client = OpenAILLMClient(client=stub, model="m")
    with pytest.raises(LLMError):
        client.complete_json(system="s", user="u", tag="t")
