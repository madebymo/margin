"""Shared enums and version-pin model used across all schemas."""

from enum import StrEnum

from pydantic import BaseModel


class EdgeType(StrEnum):
    """Prerequisite edge strength: hard edges gate path planning; soft edges inform it."""

    HARD = "hard"
    SOFT = "soft"


class ResponseClass(StrEnum):
    """How the student produced an answer; guess/slip parameters vary by class."""

    SYMBOLIC_ENTRY = "symbolic_entry"
    MULTIPLE_CHOICE = "multiple_choice"
    WIDGET = "widget"


class ReviewStatus(StrEnum):
    """Provenance state of generated or imported content."""

    DRAFT = "draft"
    LLM_GENERATED = "llm_generated"
    HUMAN_APPROVED = "human_approved"


class JobStatus(StrEnum):
    """Lifecycle of a generation job (in-process worker in v1, real broker later)."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    STALE = "stale"


class EpisodeState(StrEnum):
    """Terminal and non-terminal states of a teaching episode."""

    ACTIVE = "active"
    MASTERED = "mastered"
    FALLBACK = "fallback"
    DEFERRED = "deferred"
    STOPPED = "stopped"


class WidgetType(StrEnum):
    """The v1 interaction vocabulary."""

    SLIDER = "slider"
    CLICK_REGION = "click_region"
    MAPPING = "mapping"
    LIVE_INPUT = "live_input"


class VersionPins(BaseModel):
    """Pins generated content to the exact versions that produced it."""

    schema_version: int = 1
    graph: int
    pack: int | None = None
    prompt: str | None = None
    model: str | None = None
    evaluator: str | None = None
