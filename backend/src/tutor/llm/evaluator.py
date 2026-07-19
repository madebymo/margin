"""LLM-backed evaluator: conjunctive hard gates + scored soft axes + abstention.

Acceptance rule (from the audited plan): every hard gate passes, no
abstention, no soft axis below 3, and the soft-axis mean is at least 4.
Evaluator unavailability counts as rejection — the planner's fallback keeps
the session moving.
"""

import json
import logging

from tutor.llm import prompts
from tutor.llm.client import LLMClient, LLMError
from tutor.orchestrator.planner import EvaluationVerdict
from tutor.schemas.kc import KCNode
from tutor.schemas.widgets import WidgetConfig

logger = logging.getLogger("tutor.llm")

_HARD_GATES = ("correctness", "alignment", "consistency", "safety")


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
            logger.warning("evaluation failed for %s: %s", node.id, exc)
            return EvaluationVerdict(accepted=False, feedback=f"evaluator unavailable: {exc}")

        feedback = str(data.get("feedback", "")).strip()
        if data.get("abstain") is True:
            return EvaluationVerdict(
                accepted=False, feedback=feedback or "evaluator abstained"
            )
        hard = data.get("hard", {})
        failed_gates = [gate for gate in _HARD_GATES if hard.get(gate) is not True]
        if failed_gates:
            return EvaluationVerdict(
                accepted=False,
                feedback=f"hard gate failed: {', '.join(failed_gates)}. {feedback}".strip(),
            )
        soft = data.get("soft", {})
        try:
            scores = [float(value) for value in soft.values()]
        except (TypeError, ValueError):
            return EvaluationVerdict(accepted=False, feedback="soft scores unreadable")
        if not scores:
            return EvaluationVerdict(accepted=False, feedback="soft scores missing")
        if min(scores) < self._min_soft or sum(scores) / len(scores) < self._mean_soft:
            return EvaluationVerdict(
                accepted=False, feedback=feedback or "soft-axis scores below threshold"
            )
        return EvaluationVerdict(accepted=True, feedback=feedback)
