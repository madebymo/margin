"""Shared construction of LLM-backed ports for the CLI and the API."""

from dataclasses import dataclass
from pathlib import Path

import tutor.packs
from tutor.llm.diagnostician import LLMDiagnostician
from tutor.llm.evaluator import LLMEvaluator
from tutor.llm.interaction import LLMInteractionGenerator
from tutor.llm.lesson_writer import LLMLessonWriter
from tutor.packs.import_csv import parse_pack_csv
from tutor.schemas.kc import GraphDocument
from tutor.schemas.learner import LearnerProfile
from tutor.schemas.pedagogy import PedagogyPack
from tutor.seed.load_seed import load_coverage


def load_template_packs() -> dict[str, PedagogyPack]:
    """Load the bundled pedagogy packs keyed by kc id."""
    template_csv = Path(tutor.packs.__file__).resolve().parent / "template.csv"
    return {pack.kc_id: pack for pack in parse_pack_csv(template_csv)}


@dataclass
class LLMPorts:
    """All four LLM call sites, ready to wire into the orchestrator."""

    diagnostician: LLMDiagnostician
    lesson_writer: LLMLessonWriter
    interaction_generator: LLMInteractionGenerator
    evaluator: LLMEvaluator


def build_llm_ports(
    graph: GraphDocument, profile: LearnerProfile, provider: str = "openai"
) -> LLMPorts:
    """Build all four LLM ports; raises LLMError when unavailable."""
    if provider == "anthropic":
        from tutor.llm.client import AnthropicLLMClient as client_class
    else:
        from tutor.llm.client import OpenAILLMClient as client_class

    client = client_class()
    packs = load_template_packs()
    coverage = {
        kc: list(entry.get("widget_types", []))
        for kc, entry in load_coverage().items()
    }
    return LLMPorts(
        diagnostician=LLMDiagnostician(client, graph=graph, packs=packs, profile=profile),
        lesson_writer=LLMLessonWriter(client, packs=packs, profile=profile),
        interaction_generator=LLMInteractionGenerator(
            client, packs=packs, profile=profile, coverage=coverage
        ),
        evaluator=LLMEvaluator(client),
    )
