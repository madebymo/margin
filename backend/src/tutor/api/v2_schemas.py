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

from tutor.content.release_identity import (
    LEGACY_RELEASE_DIGEST,
    LEGACY_RELEASE_ID,
)
from tutor.schemas.assessment import PromptSegment


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


class TranscriptContentBlock(BaseModel):
    """One typed, student-safe block in a tutor transcript entry."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal[
        "text",
        "narrative",
        "prompt",
        "worked_example",
        "remediation",
    ]
    text: str | None = None
    segments: list[PromptSegment] = Field(default_factory=list)

    @model_validator(mode="after")
    def _content_matches_kind(self) -> "TranscriptContentBlock":
        if self.kind in {"text", "narrative"} and not self.text:
            raise ValueError(f"{self.kind} blocks require text")
        if self.kind in {"prompt", "worked_example"} and not self.segments:
            raise ValueError(f"{self.kind} blocks require prompt segments")
        if self.kind == "remediation" and not (self.text or self.segments):
            raise ValueError("remediation blocks require text or prompt segments")
        return self


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
    content_blocks: list[TranscriptContentBlock] = Field(default_factory=list)
    interaction_key: str | None = None
    kc_id: str | None = None
    prompt_segments: list[dict[str, Any]] | None = None
    widget: dict[str, Any] | None = None
    widget_state: dict[str, Any] | None = None
    widget_status: Literal[
        "active",
        "invalid",
        "attempted",
        "solved",
        "remediated",
        "text_fallback",
    ] | None = None
    widget_attempt_number: int | None = Field(default=None, ge=1)


class PendingHintView(BaseModel):
    """What the next hint action will do, without exposing hint content."""

    available: bool
    next_index: int = Field(ge=0)
    total: int = Field(ge=0)
    next_reveals_answer: bool = False

    @model_validator(mode="after")
    def _consistent_position(self) -> "PendingHintView":
        if self.next_index > self.total:
            raise ValueError("next hint index cannot exceed the hint count")
        if not self.available and self.next_reveals_answer:
            raise ValueError("an unavailable hint cannot reveal an answer")
        return self


class _StrictPendingInput(BaseModel):
    """Base for public input contracts; private scoring has no field here."""

    model_config = ConfigDict(extra="forbid")


class TextPendingInputView(_StrictPendingInput):
    """One of the six production answer contracts rendered as text input."""

    type: Literal["text"] = "text"
    answer_kind: Literal[
        "symbolic",
        "numeric",
        "finite_set",
        "interval_set",
        "ordered_tuple",
        "antiderivative",
    ]
    label: str = Field(min_length=1, max_length=128)
    placeholder: str = Field(min_length=1, max_length=128)
    help_text: str = Field(min_length=1, max_length=512)
    max_length: int = Field(default=256, ge=1, le=256)


class MappingInputRowView(_StrictPendingInput):
    """One public row plus its resumable learner selection."""

    entry_id: str = Field(min_length=1, max_length=64)
    label: str = Field(min_length=1, max_length=256)
    spoken_text: str = Field(min_length=1, max_length=512)
    selected_option_id: str | None = Field(default=None, max_length=64)


class MappingInputOptionView(_StrictPendingInput):
    """One public mapping option, never its correct row."""

    entry_id: str = Field(min_length=1, max_length=64)
    label: str = Field(min_length=1, max_length=256)
    spoken_text: str = Field(min_length=1, max_length=512)


class MappingPendingInputView(_StrictPendingInput):
    """Safe public and resumable state for a reviewed mapping interaction."""

    type: Literal["mapping_v1"] = "mapping_v1"
    prompt: str = Field(min_length=1, max_length=512)
    rows: list[MappingInputRowView] = Field(min_length=2, max_length=12)
    options: list[MappingInputOptionView] = Field(min_length=2, max_length=12)

    @model_validator(mode="after")
    def _valid_public_state(self) -> "MappingPendingInputView":
        row_ids = [row.entry_id for row in self.rows]
        option_ids = [option.entry_id for option in self.options]
        if len(row_ids) != len(set(row_ids)):
            raise ValueError("mapping input rows must have unique ids")
        if len(option_ids) != len(set(option_ids)):
            raise ValueError("mapping input options must have unique ids")
        if set(row_ids) & set(option_ids):
            raise ValueError("mapping row and option ids must be disjoint")
        unknown = {
            row.selected_option_id
            for row in self.rows
            if row.selected_option_id is not None
        } - set(option_ids)
        if unknown:
            raise ValueError("mapping state references an unknown public option")
        return self


class SliderPendingInputView(_StrictPendingInput):
    """Safe public and resumable state for a bounded numeric slider."""

    type: Literal["slider_v1"] = "slider_v1"
    prompt: str = Field(min_length=1, max_length=512)
    label: str = Field(min_length=1, max_length=128)
    help_text: str = Field(min_length=1, max_length=512)
    minimum: float = Field(allow_inf_nan=False)
    maximum: float = Field(allow_inf_nan=False)
    step: float = Field(gt=0, allow_inf_nan=False)
    initial_value: float = Field(allow_inf_nan=False)
    current_value: float = Field(allow_inf_nan=False)
    value_label: str = Field(min_length=1, max_length=128)
    result_template: str | None = Field(default=None, min_length=1, max_length=256)

    @model_validator(mode="after")
    def _valid_public_range(self) -> "SliderPendingInputView":
        if self.maximum <= self.minimum:
            raise ValueError("slider maximum must be greater than minimum")
        if not self.minimum <= self.initial_value <= self.maximum:
            raise ValueError("slider initial value must lie within its bounds")
        if not self.minimum <= self.current_value <= self.maximum:
            raise ValueError("slider current value must lie within its bounds")
        if self.result_template is not None and "{value}" not in self.result_template:
            raise ValueError("slider result_template must contain {value}")
        return self


class LegacyTextPendingInputView(_StrictPendingInput):
    """Safe adapter for checkpoints written before answer-kind projection."""

    type: Literal["legacy_text"] = "legacy_text"
    label: str = "Your answer"
    placeholder: str = "Type your answer"
    help_text: str = "Enter your answer using the format requested in the problem."
    max_length: int = Field(default=256, ge=1, le=256)


class LegacyChoicePendingInputView(_StrictPendingInput):
    """Read-only compatibility for old choice-bearing episode checkpoints."""

    type: Literal["legacy_choice"] = "legacy_choice"
    label: str = "Choose an answer"
    options: list[str] = Field(min_length=2, max_length=24)


PendingInputView = Annotated[
    TextPendingInputView
    | MappingPendingInputView
    | SliderPendingInputView
    | LegacyTextPendingInputView
    | LegacyChoicePendingInputView,
    Field(discriminator="type"),
]


class PendingView(BaseModel):
    """The interaction awaiting an action, without its expected answer."""

    model_config = ConfigDict(extra="forbid")

    key: str
    kind: str
    kc_id: str
    skill_name: str
    prompt: str
    prompt_segments: list[PromptSegment] = Field(default_factory=list)
    input: PendingInputView
    hint: PendingHintView
    # Compatibility field for clients predating the structured hint contract.
    can_hint: bool = True

    @model_validator(mode="before")
    @classmethod
    def _upgrade_legacy_input(cls, value: Any) -> Any:
        """Parse old safe checkpoint views, then serialize only the typed union."""

        if not isinstance(value, dict) or "input" in value:
            return value
        data = dict(value)
        mode = data.pop("input_mode", "math")
        choice_options = data.pop("choice_options", [])
        widget = data.pop("widget", None)
        widget_state = data.pop("widget_state", None)
        if mode == "choice" and isinstance(choice_options, list):
            data["input"] = {
                "type": "legacy_choice",
                "options": choice_options,
            }
            return data
        if mode == "widget" and isinstance(widget, dict):
            presentation = widget.get("presentation")
            widget_type = widget.get("interaction_version") or widget.get(
                "widget_type"
            )
            if widget_type == "mapping_v1" and isinstance(presentation, dict):
                selections = {
                    row.get("id"): row.get("value") or None
                    for row in (
                        widget_state.get("rows", [])
                        if isinstance(widget_state, dict)
                        else []
                    )
                    if isinstance(row, dict)
                }
                data["input"] = {
                    "type": "mapping_v1",
                    "prompt": presentation.get("prompt"),
                    "rows": [
                        {
                            **row,
                            "selected_option_id": selections.get(
                                row.get("entry_id")
                            ),
                        }
                        for row in presentation.get("rows", [])
                        if isinstance(row, dict)
                    ],
                    "options": presentation.get("options", []),
                }
                return data
            if widget_type == "slider_v1" and isinstance(presentation, dict):
                current = (
                    widget_state.get("value")
                    if isinstance(widget_state, dict)
                    else None
                )
                data["input"] = {
                    "type": "slider_v1",
                    **{
                        key: presentation.get(key)
                        for key in (
                            "prompt",
                            "label",
                            "help_text",
                            "minimum",
                            "maximum",
                            "step",
                            "initial_value",
                            "value_label",
                            "result_template",
                        )
                    },
                    "current_value": (
                        presentation.get("initial_value")
                        if current is None
                        else current
                    ),
                }
                return data
        data["input"] = {"type": "legacy_text"}
        return data


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

    schema_version: Literal[3] = 3
    session_id: str
    release_id: str = Field(
        default=LEGACY_RELEASE_ID,
        min_length=1,
        max_length=128,
    )
    release_digest: str = Field(
        default=LEGACY_RELEASE_DIGEST,
        pattern=r"^[0-9a-f]{64}$",
    )
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

    @model_validator(mode="before")
    @classmethod
    def _upgrade_legacy_wire_contract(cls, value: Any) -> Any:
        """Keep schema-v2 receipts readable after the typed-input cutover."""

        if not isinstance(value, dict) or value.get("schema_version") != 2:
            return value
        data = dict(value)
        data["schema_version"] = 3
        return data


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


class QuarantineRecoveryView(BaseModel):
    """Content-free capability for atomically restarting a quarantined episode."""

    revision: int = Field(ge=0)
    reset_key: str = Field(min_length=43, max_length=128)


class APIError(BaseModel):
    """Typed error returned for v2 conflict and availability failures."""

    code: str
    message: str
    session: SessionView | None = None
    retryable: bool | None = None
    quarantine_recovery: QuarantineRecoveryView | None = None
