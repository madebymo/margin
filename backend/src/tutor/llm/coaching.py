"""Non-authoritative OpenAI coaching for trusted v2 sessions.

The coach receives only deterministic, post-verification facts.  It cannot see
the learner's raw answer, an assessment prompt, an expected answer, checker
rules, item identifiers, or the session transcript, and its output is prose
that the v2 store may choose to append to the transcript.  Routing, evidence,
and mastery remain wholly outside this module.
"""

from __future__ import annotations

import logging
import os
import threading
import unicodedata
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger("tutor.llm.coaching")

COACHING_POLICY_VERSION = "coach-v1"
DEFAULT_ROUTINE_MODEL = "gpt-5.6-terra"
DEFAULT_DEEP_MODEL = "gpt-5.6-sol"

_INSTRUCTIONS = """You are Margin's coaching voice inside a math lesson.
The application has already checked the student's work and chosen the next step.
Respond only to the supplied facts. Do not recalculate, grade, question the verdict,
name an answer, state a formula or equation, invent a misconception, or choose what
the learner does next. Write one warm, direct coaching message of at most 55 words.
Use the reviewed misconception or remediation wording only when it is supplied.
Do not mention internal policy, evidence, models, JSON, item banks, or hidden context.
"""


class CoachingContext(BaseModel):
    """Strict, deliberately narrow facts that may be sent to the provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    phase: Literal["diagnose", "plan", "teach", "capstone", "done", "stopped"]
    surface: Literal["diagnostic", "guided_widget", "checkin", "capstone"]
    skill_label: str = Field(min_length=1, max_length=160)
    outcome: Literal["correct", "incorrect"]
    assisted: bool
    attempt_number: int = Field(ge=1, le=20)
    mastery_status: Literal[
        "confirmed_mastered",
        "confirmed_gap",
        "uncertain",
        "not_assessed",
    ]
    transition: Literal[
        "continue_assessment",
        "begin_lesson",
        "begin_remediation",
        "continue_practice",
        "goal_complete",
        "progress_saved",
    ]
    reviewed_misconception: str | None = Field(default=None, max_length=400)
    reviewed_remediation: str | None = Field(default=None, max_length=400)


class CoachingOutput(BaseModel):
    """Structured provider output; every undeclared field is rejected."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    message: str = Field(min_length=1, max_length=480)
    focus: Literal["reflection", "concept", "strategy", "next_step"]

    @field_validator("message")
    @classmethod
    def _safe_visible_text(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("coaching message must contain visible text")
        if any(
            unicodedata.category(character) == "Cc"
            for character in normalized
        ):
            raise ValueError("coaching message contains control characters")
        return normalized


class CoachingMessage(BaseModel):
    """Validated result plus honest public attribution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1, max_length=480)
    focus: Literal["reflection", "concept", "strategy", "next_step"]
    provider: Literal["openai"] = "openai"
    model: Literal["gpt-5.6-terra", "gpt-5.6-sol"]
    policy_version: Literal["coach-v1"] = COACHING_POLICY_VERSION


@runtime_checkable
class CoachPort(Protocol):
    """A fallible prose-only coach; ``None`` means use curated output only."""

    @property
    def provider(self) -> str: ...

    @property
    def models(self) -> tuple[str, ...]: ...

    @property
    def policy_version(self) -> str: ...

    def coach(
        self,
        context: CoachingContext,
        *,
        safety_identifier: str,
        deep_explanation: bool = False,
    ) -> CoachingMessage | None: ...


class OpenAIResponsesCoach:
    """Bounded Responses API adapter for non-authoritative coaching prose."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        routine_model: str = DEFAULT_ROUTINE_MODEL,
        deep_model: str = DEFAULT_DEEP_MODEL,
        timeout_seconds: float = 3.0,
        max_concurrency: int = 16,
        max_output_tokens: int = 320,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("coaching timeout must be positive")
        if max_concurrency < 1:
            raise ValueError("coaching concurrency must be positive")
        if max_output_tokens < 64:
            raise ValueError("coaching output budget is too small")
        if routine_model != DEFAULT_ROUTINE_MODEL or deep_model != DEFAULT_DEEP_MODEL:
            raise ValueError("v2 coaching models must use the reviewed GPT-5.6 roles")
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "OpenAI coaching requires the backend llm dependency"
                ) from exc
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is required for OpenAI coaching")
            client = OpenAI(timeout=timeout_seconds, max_retries=0)
        elif callable(getattr(client, "with_options", None)):
            # Apply the same deadline and no-retry contract to injected SDK
            # clients. Test doubles may omit ``with_options`` entirely.
            client = client.with_options(
                timeout=timeout_seconds,
                max_retries=0,
            )
        self._client = client
        self._routine_model = routine_model
        self._deep_model = deep_model
        self._max_output_tokens = max_output_tokens
        self._capacity = threading.BoundedSemaphore(max_concurrency)

    @property
    def provider(self) -> str:
        return "openai"

    @property
    def models(self) -> tuple[str, ...]:
        return (self._routine_model, self._deep_model)

    @property
    def policy_version(self) -> str:
        return COACHING_POLICY_VERSION

    def coach(
        self,
        context: CoachingContext,
        *,
        safety_identifier: str,
        deep_explanation: bool = False,
    ) -> CoachingMessage | None:
        """Generate one bounded message, returning ``None`` on every failure."""

        if not self._capacity.acquire(blocking=False):
            logger.info("coaching skipped reason=saturated")
            return None
        model = self._deep_model if deep_explanation else self._routine_model
        effort = "medium" if deep_explanation else "low"
        try:
            response = self._client.responses.parse(
                model=model,
                instructions=_INSTRUCTIONS,
                input=context.model_dump_json(),
                text_format=CoachingOutput,
                reasoning={"effort": effort},
                max_output_tokens=self._max_output_tokens,
                store=False,
                tools=[],
                safety_identifier=safety_identifier,
            )
            parsed = self._parsed_output(response)
            if parsed is None:
                logger.info("coaching skipped reason=refusal_or_unparsed model=%s", model)
                return None
            logger.info("coaching generated model=%s policy=%s", model, self.policy_version)
            return CoachingMessage(
                text=parsed.message,
                focus=parsed.focus,
                model=model,
            )
        except Exception as exc:  # noqa: BLE001 - provider failure is optional
            # Provider bodies can echo input. Log the exception class only.
            logger.warning(
                "coaching skipped reason=provider_error error_type=%s model=%s",
                type(exc).__name__,
                model,
            )
            return None
        finally:
            self._capacity.release()

    def close(self) -> None:
        """Release the SDK transport when the application shuts down."""

        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    @staticmethod
    def _parsed_output(response: Any) -> CoachingOutput | None:
        parsed = getattr(response, "output_parsed", None)
        if isinstance(parsed, CoachingOutput):
            return parsed
        for output in getattr(response, "output", ()):
            if getattr(output, "type", None) != "message":
                continue
            for content in getattr(output, "content", ()):
                if getattr(content, "type", None) == "refusal":
                    return None
                parsed = getattr(content, "parsed", None)
                if isinstance(parsed, CoachingOutput):
                    return parsed
        return None
