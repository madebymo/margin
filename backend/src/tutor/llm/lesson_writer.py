"""LLM-backed lesson writer: narrative + check-in variations, gated and safe.

Narratives use the KC's pedagogy-pack metaphor when available (one metaphor
per path, chosen upstream). Check-in items pass the correctness gate before
display. All failures fall back to the deterministic template writer.
"""

import logging

from pydantic import ValidationError

from tutor.llm import prompts
from tutor.llm.client import LLMClient, LLMError
from tutor.orchestrator.ports import (
    LessonWriterPort,
    PracticeItem,
    TemplateLessonWriter,
)
from tutor.schemas.kc import KCNode
from tutor.schemas.learner import LearnerProfile
from tutor.schemas.pedagogy import PedagogyPack
from tutor.verify.checker import MathVerificationError, parse_restricted

logger = logging.getLogger("tutor.llm")

_GENERIC_HINT = "Take it one step at a time."


class LLMLessonWriter:
    """LessonWriterPort implementation backed by an LLM client."""

    def __init__(
        self,
        client: LLMClient,
        packs: dict[str, PedagogyPack] | None = None,
        profile: LearnerProfile | None = None,
        fallback: LessonWriterPort | None = None,
        max_attempts: int = 2,
    ) -> None:
        self._client = client
        self._packs = packs or {}
        self._profile = profile
        self._fallback = fallback or TemplateLessonWriter()
        self._max_attempts = max_attempts

    def lesson_text(self, node: KCNode) -> str:
        """Generate the mini-lesson narrative; fall back on failure."""
        pack = self._packs.get(node.id)
        metaphor = pack.metaphors[0] if pack and pack.metaphors else None
        misconceptions = pack.misconceptions if pack else []
        for _ in range(self._max_attempts):
            try:
                data = self._client.complete_json(
                    system=prompts.LESSON_SYSTEM,
                    user=prompts.lesson_user(node, self._profile, metaphor, misconceptions),
                    tag=f"lesson:{node.id}",
                )
                narrative = data.get("narrative")
                if isinstance(narrative, str) and narrative.strip():
                    return narrative.strip()
                raise LLMError("narrative missing or empty")
            except LLMError as exc:
                logger.warning("lesson generation failed for %s: %s", node.id, exc)
        return self._fallback.lesson_text(node)

    def checkin_item(self, node: KCNode, attempt: int) -> PracticeItem:
        """Generate a gated near-transfer check-in; fall back on failure."""
        pack = self._packs.get(node.id)
        for _ in range(self._max_attempts):
            try:
                data = self._client.complete_json(
                    system=prompts.CHECKIN_SYSTEM,
                    user=prompts.checkin_user(node, attempt, self._profile, pack),
                    tag=f"checkin:{node.id}",
                )
                return self._validated_item(data)
            except (LLMError, ValidationError, MathVerificationError) as exc:
                logger.warning("check-in generation failed for %s: %s", node.id, exc)
        return self._fallback.checkin_item(node, attempt)

    def _validated_item(self, data: dict) -> PracticeItem:
        prompt_text = str(data.get("prompt", "")).strip()
        expected = str(data.get("expected", "")).strip()
        if not prompt_text:
            raise LLMError("check-in prompt missing or empty")
        checker = data.get("checker", "sympy_equiv")
        if checker not in ("sympy_equiv", "numeric"):
            checker = "sympy_equiv"
        hints = [str(hint).strip() for hint in data.get("hints", []) if str(hint).strip()]
        while len(hints) < 3:
            hints.append(_GENERIC_HINT)
        item = PracticeItem(
            prompt=prompt_text, expected=expected, checker=checker, hints=hints[:3]
        )
        parse_restricted(item.expected)  # correctness gate
        if len(item.expected) >= 3 and item.expected in item.prompt:
            raise MathVerificationError("expected answer leaked into the prompt")
        return item
