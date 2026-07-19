"""BKT-lite parameters — external, versioned, and replayable.

Mastery values computed with these parameters are routing scores, not
calibrated probabilities. Guess/slip vary by response class; assisted
responses receive discounted credit; graph-propagated (inferred) evidence is
capped below the mastery threshold so it can never confirm mastery alone.
"""

from pydantic import BaseModel, Field

from tutor.schemas.common import ResponseClass


class ResponseClassParams(BaseModel):
    """Guess and slip rates for one response class."""

    guess: float = Field(ge=0.0, le=1.0)
    slip: float = Field(ge=0.0, le=1.0)


def _default_response_class_params() -> dict[ResponseClass, ResponseClassParams]:
    return {
        ResponseClass.MULTIPLE_CHOICE: ResponseClassParams(guess=0.25, slip=0.10),
        ResponseClass.SYMBOLIC_ENTRY: ResponseClassParams(guess=0.10, slip=0.15),
        ResponseClass.WIDGET: ResponseClassParams(guess=0.15, slip=0.10),
    }


class BKTParams(BaseModel):
    """All tunable knobs of the learner model update, pinned by params_version."""

    params_version: int = 1
    learn: float = Field(default=0.15, ge=0.0, le=1.0)
    prior_default: float = Field(default=0.5, ge=0.0, le=1.0)
    prior_assumed_floor: float = Field(default=0.75, ge=0.0, le=1.0)
    response_class: dict[ResponseClass, ResponseClassParams] = Field(
        default_factory=_default_response_class_params
    )
    assisted_credit: float = Field(default=0.5, ge=0.0, le=1.0)
    propagation_strength: float = Field(default=0.25, ge=0.0, le=1.0)
    propagation_decay: float = Field(default=0.5, ge=0.0, le=1.0)
    inferred_cap: float = Field(default=0.65, ge=0.0, le=1.0)
    mastery_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


DEFAULT_PARAMS_V1 = BKTParams()
