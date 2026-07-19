"""Thin, synchronous, JSON-only LLM client adapters.

OpenAILLMClient (default provider) uses Chat Completions JSON mode;
AnthropicLLMClient marks the static system prompt for prompt caching. Both
record per-call metadata (tag, model, latency) for observability. Any other
provider can be wired in by satisfying the LLMClient protocol.
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger("tutor.llm")

DEFAULT_OPENAI_MODEL = "gpt-5.5"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5"


class LLMError(RuntimeError):
    """Raised when an LLM call fails or returns unusable output."""


@runtime_checkable
class LLMClient(Protocol):
    """One JSON-producing completion per call, addressed by an observability tag."""

    def complete_json(self, *, system: str, user: str, tag: str) -> dict[str, Any]:
        """Run one completion and return the parsed JSON object."""
        ...


@dataclass
class LLMCall:
    """Observability record for one LLM call."""

    tag: str
    model: str
    latency_ms: float
    input_chars: int
    output_chars: int


def extract_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from model output (code fences tolerated)."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise LLMError(f"no JSON object in model output: {text[:200]!r}")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMError(f"invalid JSON in model output: {exc}") from exc
    if not isinstance(data, dict):
        raise LLMError("model output JSON is not an object")
    return data


class OpenAILLMClient:
    """OpenAI Chat Completions adapter (JSON mode).

    Prompt caching is automatic on OpenAI's side for repeated prefixes, so the
    static system prompt benefits without explicit cache markers. Temperature
    is left at the model default — the gpt-5 family restricts overrides.
    """

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int = 1024,
        client: Any | None = None,
    ) -> None:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise LLMError(
                    "openai package not installed — pip install -e 'backend[llm]'"
                ) from exc
            if not os.environ.get("OPENAI_API_KEY"):
                raise LLMError("OPENAI_API_KEY is not set")
            client = OpenAI()
        self._client = client
        self._model = model or os.environ.get("TUTOR_LLM_MODEL", DEFAULT_OPENAI_MODEL)
        self._max_tokens = max_tokens
        self.calls: list[LLMCall] = []

    def complete_json(self, *, system: str, user: str, tag: str) -> dict[str, Any]:
        """One JSON-producing completion via Chat Completions JSON mode."""
        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                max_completion_tokens=self._max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except Exception as exc:  # noqa: BLE001 — normalize provider errors
            raise LLMError(f"LLM request failed for {tag}: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000
        text = response.choices[0].message.content or ""
        self.calls.append(
            LLMCall(
                tag=tag,
                model=self._model,
                latency_ms=latency_ms,
                input_chars=len(system) + len(user),
                output_chars=len(text),
            )
        )
        logger.info("llm call %s model=%s latency_ms=%.0f", tag, self._model, latency_ms)
        return extract_json(text)


class AnthropicLLMClient:
    """Anthropic Messages API adapter with prompt caching on the system block."""

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        client: Any | None = None,
    ) -> None:
        if client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise LLMError(
                    "anthropic package not installed — pip install -e 'backend[llm-anthropic]'"
                ) from exc
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise LLMError("ANTHROPIC_API_KEY is not set")
            client = anthropic.Anthropic()
        self._client = client
        self._model = model or os.environ.get("TUTOR_LLM_MODEL", DEFAULT_ANTHROPIC_MODEL)
        self._max_tokens = max_tokens
        self._temperature = temperature
        self.calls: list[LLMCall] = []

    def complete_json(self, *, system: str, user: str, tag: str) -> dict[str, Any]:
        """One JSON-producing completion; the system prompt is cache-marked."""
        started = time.perf_counter()
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=[
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user}],
            )
        except Exception as exc:  # noqa: BLE001 — normalize provider errors
            raise LLMError(f"LLM request failed for {tag}: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000
        text = "".join(
            block.text
            for block in response.content
            if getattr(block, "type", "") == "text"
        )
        self.calls.append(
            LLMCall(
                tag=tag,
                model=self._model,
                latency_ms=latency_ms,
                input_chars=len(system) + len(user),
                output_chars=len(text),
            )
        )
        logger.info("llm call %s model=%s latency_ms=%.0f", tag, self._model, latency_ms)
        return extract_json(text)
