"""Public, student-safe schemas for the versioned session API.

The v2 API returns one authoritative session snapshot after every successful
mutation.  Expected answers, checker configuration, and server-only widget
rules deliberately have no representation in these models.
"""

import json
import math
from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GoalView(BaseModel):
    """One curated goal available to start from the intake screen."""

    goal_id: str
    target_kc: str
    title: str
    description: str
    course_level: str


class CatalogRolloutView(BaseModel):
    """Why this anonymous browser can or cannot start a new pilot session."""

    status: Literal[
        "available",
        "not_selected",
        "paused",
        "content_unavailable",
    ]
    reason: str
    percentage: Literal[0, 5, 25, 100]


class GoalCatalog(BaseModel):
    """Versioned list of goals whose content is ready for student use."""

    catalog_version: int = 1
    goals: list[GoalView]
    rollout: CatalogRolloutView


class WidgetCapabilityView(BaseModel):
    keyboard_equivalent: bool
    live_visual: bool


class WidgetCapabilityManifestView(BaseModel):
    """Versioned backend/frontend agreement about enabled widget semantics."""

    version: str
    supported: dict[str, WidgetCapabilityView]
    disabled: dict[str, str]


class SessionProfile(BaseModel):
    """Non-identifying profile information used to adapt presentation."""

    course: str
    age_band: str


class ContentModeView(BaseModel):
    """Requested and effective content modes, including an honest fallback."""

    requested: Literal["curated", "llm_coaching"]
    effective: Literal["curated", "llm_coaching"]
    fallback_reason: str | None = None


class TranscriptEntry(BaseModel):
    """A safe transcript record rendered by the client."""

    sequence: int = Field(ge=0)
    role: Literal["tutor", "student", "system"]
    kind: Literal[
        "message",
        "probe",
        "lesson",
        "checkin",
        "capstone",
        "hint",
        "widget_attempt",
        "widget_feedback",
    ]
    text: str
    interaction_key: str | None = None
    kc_id: str | None = None
    prompt_segments: list[dict[str, Any]] | None = None
    widget: dict[str, Any] | None = None
    widget_status: Literal[
        "active",
        "invalid",
        "attempted",
        "solved",
        "remediated",
        "text_fallback",
    ] | None = None
    widget_attempt_number: int | None = Field(default=None, ge=1)


class PendingView(BaseModel):
    """The interaction awaiting an action, without its expected answer."""

    key: str
    kind: str
    kc_id: str
    skill_name: str
    input_mode: Literal["math", "choice", "widget", "none"] = "math"
    prompt: str
    prompt_segments: list[dict[str, Any]] = Field(default_factory=list)
    choice_options: list[str] = Field(default_factory=list)
    can_hint: bool = True


class ProgressView(BaseModel):
    """Small, honest progress summary for the student."""

    phase: str
    current_skill: str | None = None
    plan_step: str | None = None
    diagnosis_probes_used: int = Field(ge=0, default=0)
    diagnosis_probe_budget: int | None = Field(default=None, ge=0)
    interactions_used: int = Field(ge=0, default=0)


class LearnerSummaryView(BaseModel):
    """Typed diagnosis summary that keeps uncertainty explicit."""

    confirmed_strengths: list[str] = Field(default_factory=list)
    confirmed_gaps: list[str] = Field(default_factory=list)
    uncertain_skills: list[str] = Field(default_factory=list)


class SessionView(BaseModel):
    """Complete authoritative state returned by all v2 reads and mutations."""

    schema_version: Literal[2] = 2
    session_id: str
    revision: int = Field(ge=0)
    phase: str
    durability: Literal["durable", "memory_only"]
    goal: GoalView
    profile: SessionProfile
    context: str | None = None
    content_mode: ContentModeView
    transcript: list[TranscriptEntry]
    pending: PendingView | None
    progress: ProgressView
    learner_summary: LearnerSummaryView
    started_at: datetime
    updated_at: datetime


class CreateSessionV2Request(BaseModel):
    """Create one anonymous, resumable tutoring episode."""

    model_config = ConfigDict(extra="forbid")

    request_id: UUID
    goal_id: str
    course: str = Field(default="AP Calculus AB", min_length=1, max_length=100)
    age_band: str = Field(default="16-18", min_length=1, max_length=32)
    content_mode: Literal["curated", "llm_coaching"] = "curated"
    context: str | None = Field(default=None, max_length=2000)
    provider: Literal["openai", "anthropic"] = "openai"


class _ActionBase(BaseModel):
    """Optimistic-concurrency and idempotency fields shared by all actions."""

    model_config = ConfigDict(extra="forbid")

    request_id: UUID
    expected_revision: int = Field(ge=0)
    pending_key: str = Field(min_length=1, max_length=128)


class AnswerAction(_ActionBase):
    type: Literal["answer"]
    answer: str = Field(min_length=1, max_length=256)


class HintAction(_ActionBase):
    type: Literal["request_hint"]


class WidgetAttemptAction(_ActionBase):
    type: Literal["widget_attempt"]
    response: dict[str, Any]

    @model_validator(mode="after")
    def _bounded_response(self) -> "WidgetAttemptAction":
        nodes = 0

        def visit(value: Any, depth: int) -> None:
            nonlocal nodes
            nodes += 1
            if nodes > 128:
                raise ValueError("widget response exceeds 128 values")
            if depth > 8:
                raise ValueError("widget response exceeds depth 8")
            if value is None or isinstance(value, (bool, int)):
                return
            if isinstance(value, float):
                if not math.isfinite(value):
                    raise ValueError("widget response numbers must be finite")
                return
            if isinstance(value, str):
                if len(value) > 1024:
                    raise ValueError("widget response text exceeds 1024 characters")
                return
            if isinstance(value, list):
                if len(value) > 64:
                    raise ValueError("widget response list exceeds 64 values")
                for item in value:
                    visit(item, depth + 1)
                return
            if isinstance(value, dict):
                if len(value) > 32:
                    raise ValueError("widget response object exceeds 32 fields")
                for key, item in value.items():
                    if not isinstance(key, str) or len(key) > 64:
                        raise ValueError("widget response keys must be short strings")
                    visit(item, depth + 1)
                return
            raise ValueError("widget response contains an unsupported value")

        visit(self.response, 0)
        encoded = json.dumps(
            self.response,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        if len(encoded.encode("utf-8")) > 4096:
            raise ValueError("widget response exceeds 4096 bytes")
        return self


class TextFallbackAction(_ActionBase):
    type: Literal["use_text_fallback"]


SessionAction = Annotated[
    AnswerAction | HintAction | WidgetAttemptAction | TextFallbackAction,
    Field(discriminator="type"),
]


class SessionActionRequest(BaseModel):
    """Request wrapper that gives OpenAPI a stable discriminated-union shape."""

    model_config = ConfigDict(extra="forbid")

    action: SessionAction


class ResetSessionV2Request(BaseModel):
    """Idempotent, revision-checked revocation of the current episode."""

    model_config = ConfigDict(extra="forbid")

    request_id: UUID
    expected_revision: int = Field(ge=0)
    pending_key: str | None = Field(default=None, max_length=128)


class ResetResponse(BaseModel):
    """Result of replacing the current episode with a fresh one."""

    reset: bool
    session: SessionView


class RecoverSessionV2Request(BaseModel):
    """Second proof used only to recover a lost token-rotation response."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    operation: Literal["create", "reset"]
    request_id: UUID


class RecoverSessionV2Response(BaseModel):
    """Acknowledgement that the replacement cookie reached this response."""

    recovered: Literal[True] = True
    session_id: str


class APIError(BaseModel):
    """Typed error returned for v2 conflict and availability failures."""

    code: str
    message: str
    session: SessionView | None = None
