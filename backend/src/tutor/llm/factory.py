"""Shared construction of LLM-backed ports for the CLI and the API."""

from pathlib import Path

import tutor.packs
from tutor.llm.diagnostician import LLMDiagnostician
from tutor.llm.lesson_writer import LLMLessonWriter
from tutor.packs.import_csv import parse_pack_csv
from tutor.schemas.kc import GraphDocument
from tutor.schemas.learner import LearnerProfile
from tutor.schemas.pedagogy import PedagogyPack


def load_template_packs() -> dict[str, PedagogyPack]:
    """Load the bundled pedagogy packs keyed by kc id."""
    template_csv = Path(tutor.packs.__file__).resolve().parent / "template.csv"
    return {pack.kc_id: pack for pack in parse_pack_csv(template_csv)}


def build_llm_ports(
    graph: GraphDocument, profile: LearnerProfile, provider: str = "openai"
) -> tuple[LLMDiagnostician, LLMLessonWriter]:
    """Build (diagnostician, lesson_writer); raises LLMError when unavailable."""
    if provider == "anthropic":
        from tutor.llm.client import AnthropicLLMClient as client_class
    else:
        from tutor.llm.client import OpenAILLMClient as client_class

    client = client_class()
    packs = load_template_packs()
    return (
        LLMDiagnostician(client, graph=graph, packs=packs, profile=profile),
        LLMLessonWriter(client, packs=packs, profile=profile),
    )
