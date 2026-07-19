"""Episode envelope: the explicit routing state with global progress guards.

All counters persist for the life of the episode (across detours and
resumes), so oscillation cannot reset them. The envelope is only ever updated
through ``tutor.orchestrator.routing.route`` (plus the machine's resume pop),
never mutated in place by callers.
"""

from enum import StrEnum

from pydantic import BaseModel, Field


class RoutingAction(StrEnum):
    """Possible outcomes of routing one check-in result."""

    ADVANCE = "advance"
    CONTINUE = "continue"
    RETRY = "retry"
    DESCEND = "descend"
    FALLBACK = "fallback"
    STOP = "stop"
    DUPLICATE = "duplicate"


class CheckinOutcome(BaseModel):
    """One scored check-in interaction, as seen by the router."""

    kc_id: str
    correct: bool
    interaction_key: str
    consecutive_correct: int = Field(default=0, ge=0)
    implicated_prereq: str | None = None


class RoutingDecision(BaseModel):
    """The router's verdict for one outcome."""

    action: RoutingAction
    descend_to: str | None = None
    reason: str


class EpisodeEnvelope(BaseModel):
    """Explicit episode state consumed and produced by the pure router."""

    target_kc: str
    interaction_budget: int = Field(default=40, ge=1)
    interactions_used: int = Field(default=0, ge=0)
    max_retries_per_kc: int = Field(default=2, ge=0)
    max_inserts: int = Field(default=5, ge=0)
    max_detour_depth: int = Field(default=3, ge=1)
    retries: dict[str, int] = Field(default_factory=dict)
    inserted: list[str] = Field(default_factory=list)
    resume_stack: list[str] = Field(default_factory=list)
    seen_interaction_keys: list[str] = Field(default_factory=list)
