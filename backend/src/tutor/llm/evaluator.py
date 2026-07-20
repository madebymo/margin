"""LLM-backed evaluator: conjunctive hard gates + scored soft axes + abstention.

Acceptance rule (from the audited plan): every hard gate passes, no
abstention, no soft axis below 3, and the soft-axis mean is at least 4.
Evaluator unavailability counts as rejection — the planner's fallback keeps
the session moving.
"""

import json
import logging

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from tutor.llm import prompts
from tutor.llm.client import LLMClient, LLMError
from tutor.orchestrator.planner import EvaluationVerdict
from tutor.schemas.kc import KCNode
from tutor.schemas.widgets import WidgetConfig

logger = logging.getLogger("tutor.llm")

class _StrictEvaluationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class EvaluationHardGates(_StrictEvaluationModel):
    """The exact conjunctive evaluator hard gates."""

    correctness: bool
    alignment: bool
    consistency: bool
    safety: bool


class EvaluationSoftAxes(_StrictEvaluationModel):
    """The exact integer-valued evaluator rubric axes."""

    clarity: int = Field(ge=1, le=5)
    scaffolding: int = Field(ge=1, le=5)
    cognitive_load: int = Field(ge=1, le=5)
    engagement: int = Field(ge=1, le=5)
    age_fit: int = Field(ge=1, le=5)


class EvaluationPayload(_StrictEvaluationModel):
    """Schema for untrusted JSON returned by the LLM evaluator."""

    hard: EvaluationHardGates
    soft: EvaluationSoftAxes
    abstain: bool
    feedback: str = ""


class LLMEvaluator:
    """EvaluatorPort implementation backed by an LLM judge."""

    def __init__(
        self,
        client: LLMClient,
        min_soft: float = 3.0,
        mean_soft: float = 4.0,
    ) -> None:
        self._client = client
        self._min_soft = min_soft
        self._mean_soft = mean_soft

    def evaluate(
        self, node: KCNode, narrative: str, widget: WidgetConfig
    ) -> EvaluationVerdict:
        """Judge one candidate; any doubt, unavailability, or abstention rejects."""
        try:
            data = self._client.complete_json(
                system=prompts.EVALUATOR_SYSTEM,
                user=prompts.evaluator_user(
                    node, narrative, json.dumps(widget.model_dump(), sort_keys=True)
                ),
                tag=f"evaluate:{node.id}",
            )
        except LLMError as exc:
            logger.warning(
                "evaluation failed for %s (%s)", node.id, type(exc).__name__
            )
            return EvaluationVerdict(
                accepted=False, feedback="evaluator unavailable"
            )
        except Exception as exc:  # noqa: BLE001 - client failure cannot escape the planner
            logger.warning(
                "evaluation crashed for %s (%s)", node.id, type(exc).__name__
            )
            return EvaluationVerdict(
                accepted=False, feedback="evaluator unavailable"
            )

        try:
            payload = EvaluationPayload.model_validate(data)
        except ValidationError as exc:
            logger.warning(
                "invalid evaluator payload for %s (errors=%d)",
                node.id,
                exc.error_count(),
            )
            return EvaluationVerdict(
                accepted=False,
                feedback="evaluator payload failed strict schema validation",
            )

        feedback = payload.feedback.strip()
        if payload.abstain:
            return EvaluationVerdict(
                accepted=False, feedback=feedback or "evaluator abstained"
            )
        failed_gates = [
            field
            for field, passed in payload.hard.model_dump().items()
            if passed is not True
        ]
        if failed_gates:
            return EvaluationVerdict(
                accepted=False,
                feedback=f"hard gate failed: {', '.join(failed_gates)}. {feedback}".strip(),
            )
        scores = list(payload.soft.model_dump().values())
        if min(scores) < self._min_soft or sum(scores) / len(scores) < self._mean_soft:
            return EvaluationVerdict(
                accepted=False, feedback=feedback or "soft-axis scores below threshold"
            )
        return EvaluationVerdict(accepted=True, feedback=feedback)
