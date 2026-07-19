"""LLM-backed diagnostician: probe generation + real error analysis.

Every probe passes the correctness gate (restricted SymPy parse of the
expected answer, blank normalization, answer-leak check) before reaching a
student. Failures retry, then fall back to the deterministic template
diagnostician. Error-analysis outputs are membership-validated: misconception
ids must come from the KC's pedagogy pack and implicated prerequisites from
the KC's hard predecessors — the model cannot invent either.
"""

import logging
from collections import defaultdict

from pydantic import ValidationError

from tutor.llm import prompts
from tutor.llm.client import LLMClient, LLMError
from tutor.orchestrator.ports import (
    DiagnosticianPort,
    ErrorAnalysis,
    TemplateDiagnostician,
)
from tutor.schemas.common import EdgeType
from tutor.schemas.kc import GraphDocument, KCNode
from tutor.schemas.learner import LearnerProfile
from tutor.schemas.pedagogy import PedagogyPack
from tutor.schemas.probe import DiagnosticProbe
from tutor.verify.checker import MathVerificationError, parse_restricted

logger = logging.getLogger("tutor.llm")


class LLMDiagnostician:
    """DiagnosticianPort implementation backed by an LLM client."""

    def __init__(
        self,
        client: LLMClient,
        graph: GraphDocument,
        packs: dict[str, PedagogyPack] | None = None,
        profile: LearnerProfile | None = None,
        fallback: DiagnosticianPort | None = None,
        max_attempts: int = 2,
    ) -> None:
        self._client = client
        self._packs = packs or {}
        self._profile = profile
        self._fallback = fallback or TemplateDiagnostician()
        self._max_attempts = max_attempts
        self._hard_preds: dict[str, list[str]] = defaultdict(list)
        for edge in graph.edges:
            if edge.type == EdgeType.HARD:
                self._hard_preds[edge.to_kc].append(edge.from_kc)

    def generate_probe(self, node: KCNode) -> DiagnosticProbe:
        """Generate, validate, and gate a probe; fall back on repeated failure."""
        for _ in range(self._max_attempts):
            try:
                data = self._client.complete_json(
                    system=prompts.PROBE_SYSTEM,
                    user=prompts.probe_user(node, self._profile, self._packs.get(node.id)),
                    tag=f"probe:{node.id}",
                )
                return self._validated_probe(node, data)
            except (LLMError, ValidationError, MathVerificationError) as exc:
                logger.warning("probe generation failed for %s: %s", node.id, exc)
        return self._fallback.generate_probe(node)

    def _validated_probe(self, node: KCNode, data: dict) -> DiagnosticProbe:
        payload = {**data, "probe_id": f"probe.llm.{node.id}", "kc_id": node.id}
        probe = DiagnosticProbe.model_validate(payload)
        parse_restricted(probe.expected)  # correctness gate
        probe.scaffold_steps[probe.blank_index] = "____"  # normalize the blank
        visible = "\n".join(
            step
            for index, step in enumerate(probe.scaffold_steps)
            if index != probe.blank_index
        )
        leak_surface = f"{visible}\n" + "\n".join(probe.hint_ladder[:2])
        if len(probe.expected.strip()) >= 3 and probe.expected.strip() in leak_surface:
            raise MathVerificationError("expected answer leaked into visible probe text")
        return probe

    def analyze_error(
        self, node: KCNode, prompt: str, expected: str, answer: str
    ) -> ErrorAnalysis:
        """Classify a wrong answer; ids are membership-validated, never invented."""
        pack = self._packs.get(node.id)
        misconceptions = pack.misconceptions if pack else []
        candidates = self._hard_preds.get(node.id, [])
        try:
            data = self._client.complete_json(
                system=prompts.ERROR_SYSTEM,
                user=prompts.error_user(
                    node, prompt, expected, answer, misconceptions, candidates
                ),
                tag=f"error:{node.id}",
            )
        except LLMError as exc:
            logger.warning("error analysis failed for %s: %s", node.id, exc)
            return self._fallback.analyze_error(node, prompt, expected, answer)
        misconception = data.get("misconception_id")
        if misconception not in {m.id for m in misconceptions}:
            misconception = None
        prereq = data.get("implicated_prereq")
        if prereq not in set(candidates):
            prereq = None
        return ErrorAnalysis(misconception_id=misconception, implicated_prereq=prereq)
