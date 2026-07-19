"""LLM-backed interaction generator: widget-config candidates, schema-gated.

Candidates that fail discriminated-union validation are dropped silently;
the LessonPlanner applies deterministic gates and the evaluator verdict on
whatever survives, and falls back to a worked example when nothing does.
"""

import logging

from pydantic import ValidationError

from tutor.llm import prompts
from tutor.llm.client import LLMClient, LLMError
from tutor.schemas.kc import KCNode
from tutor.schemas.learner import LearnerProfile
from tutor.schemas.pedagogy import PedagogyPack
from tutor.schemas.widgets import WidgetConfig, parse_widget_config

logger = logging.getLogger("tutor.llm")


class LLMInteractionGenerator:
    """InteractionGeneratorPort implementation backed by an LLM client."""

    def __init__(
        self,
        client: LLMClient,
        packs: dict[str, PedagogyPack] | None = None,
        profile: LearnerProfile | None = None,
        coverage: dict[str, list[str]] | None = None,
        max_candidates: int = 3,
    ) -> None:
        self._client = client
        self._packs = packs or {}
        self._profile = profile
        self._coverage = coverage or {}
        self._max_candidates = max_candidates

    def candidates(
        self, node: KCNode, attempt: int, feedback: list[str]
    ) -> list[WidgetConfig]:
        """Generate schema-valid widget candidates (empty list on failure)."""
        try:
            data = self._client.complete_json(
                system=prompts.INTERACTION_SYSTEM,
                user=prompts.interaction_user(
                    node,
                    attempt,
                    feedback,
                    preferred_types=self._coverage.get(node.id),
                    pack=self._packs.get(node.id),
                    profile=self._profile,
                ),
                tag=f"interaction:{node.id}",
            )
        except LLMError as exc:
            logger.warning("interaction generation failed for %s: %s", node.id, exc)
            return []
        raw_candidates = data.get("candidates", [])
        if not isinstance(raw_candidates, list):
            return []
        results: list[WidgetConfig] = []
        for raw in raw_candidates[: self._max_candidates]:
            try:
                results.append(parse_widget_config(raw))
            except ValidationError as exc:
                logger.warning("dropping invalid widget candidate for %s: %s", node.id, exc)
        return results
