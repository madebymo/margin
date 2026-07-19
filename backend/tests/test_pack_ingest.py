"""RAG pack ingestion: chunking, retrieval, id normalization, pipeline, loader merge."""

import json

import pytest

from tutor.llm.client import LLMError
from tutor.packs.ingest import (
    Bm25Index,
    build_pack,
    chunk_text,
    kc_query,
    load_corpus,
    run_ingest,
    _validated_pack,
)
from tutor.packs.loader import load_packs
from tutor.schemas.common import ReviewStatus
from tutor.schemas.pedagogy import PedagogyPack
from tutor.seed.load_seed import load_graph

CHAIN_RULE_DOC = """# The chain rule

The chain rule handles composite functions: the derivative of f(g(x)) is
f'(g(x)) times g'(x).

A very common student misconception is differentiating only the outer
function and forgetting to multiply by the derivative of the inner function.
For example students write d/dx sin(x^2) = cos(x^2), dropping the 2x factor.

Teachers often use the metaphor of nested gears: the outer gear's speed is
scaled by the inner gear's speed.
"""

USUB_DOC = """# U-substitution

Substitution reverses the chain rule for integrals. Students frequently
substitute u for the inner function but forget to convert dx into du,
leaving a mixed integrand.

Another pattern: in definite integrals students keep the original x bounds
after substituting.
"""

COOKING_DOC = """# Sourdough basics

Preheat the oven with the dutch oven inside. Score the loaf and bake covered
for twenty minutes, then uncovered until deep brown.
"""

GOOD_DRAFT = {
    "misconceptions": [
        {
            "slug": "Outer Only!",
            "description": "Differentiates the outer function and drops the inner derivative",
            "error_signature": "answer missing the inner derivative factor",
            "remediation_hint": "Multiply by the derivative of the inside function",
        }
    ],
    "metaphors": [
        {
            "slug": "nested gears",
            "description": "Nested gears: outer speed scaled by inner speed",
            "widget_affinity": ["slider", "banana"],
        }
    ],
    "error_patterns": ["drops the 2x factor", "  "],
}


class FakeLLM:
    def __init__(self, response: object) -> None:
        self._response = response
        self.calls: list[str] = []

    def complete_json(self, *, system: str, user: str, tag: str) -> dict:
        self.calls.append(tag)
        if isinstance(self._response, Exception):
            raise self._response
        return json.loads(json.dumps(self._response))


@pytest.fixture(scope="module")
def graph():
    return load_graph()


@pytest.fixture()
def corpus_dir(tmp_path):
    (tmp_path / "chain_rule.md").write_text(CHAIN_RULE_DOC)
    (tmp_path / "u_sub.md").write_text(USUB_DOC)
    (tmp_path / "cooking.txt").write_text(COOKING_DOC)
    return tmp_path


def _node(graph, kc_id):
    return next(node for node in graph.nodes if node.id == kc_id)


def test_chunker_respects_bounds():
    text = "\n\n".join(f"Paragraph {i} " + "x" * 300 for i in range(10))
    chunks = chunk_text(text, max_chars=800)
    assert len(chunks) > 1
    assert all(chunks)
    assert all(len(chunk) <= 800 for chunk in chunks)
    assert "Paragraph 0" in chunks[0] and "Paragraph 9" in chunks[-1]


def test_bm25_ranks_relevant_source_first(graph, corpus_dir):
    index = Bm25Index(load_corpus(corpus_dir))
    hits = index.search(kc_query(_node(graph, "kc.der.chain_rule")), top_k=3)
    assert hits, "expected at least one hit"
    assert hits[0][0].source == "chain_rule.md"
    assert all(chunk.source != "cooking.txt" for chunk, _ in hits)


def test_validated_pack_normalizes_ids_and_affinity(graph):
    node = _node(graph, "kc.der.chain_rule")
    pack = _validated_pack(node, GOOD_DRAFT, ["chain_rule.md", "chain_rule.md"])
    assert pack.review_status == ReviewStatus.LLM_GENERATED
    assert pack.misconceptions[0].id == "m.chain_rule.outer_only"
    assert pack.metaphors[0].id == "met.nested_gears"
    assert pack.metaphors[0].widget_affinity == ["slider"]  # invalid entries dropped
    assert pack.error_patterns == ["drops the 2x factor"]  # blanks removed
    assert pack.sources == ["chain_rule.md"]  # deduplicated


def test_build_pack_skips_never_fabricates(graph, corpus_dir):
    index = Bm25Index(load_corpus(corpus_dir))
    node = _node(graph, "kc.der.chain_rule")
    empty = FakeLLM({"misconceptions": []})
    assert build_pack(node, empty, index) is None
    dead = FakeLLM(LLMError("down"))
    assert build_pack(node, dead, index) is None


def test_run_ingest_writes_valid_packs_and_report(graph, corpus_dir, tmp_path):
    out_dir = tmp_path / "generated"
    fake = FakeLLM(GOOD_DRAFT)
    report = run_ingest(
        graph,
        corpus_dir,
        out_dir,
        fake,
        kc_ids={"kc.der.chain_rule", "kc.int.u_substitution"},
    )
    assert sorted(report.generated) == ["kc.der.chain_rule", "kc.int.u_substitution"]
    assert report.skipped == []
    assert report.source_count == 3
    files = sorted(path.name for path in out_dir.glob("*.json"))
    assert files == ["kc_der_chain_rule.json", "kc_int_u_substitution.json"]
    for path in out_dir.glob("*.json"):
        pack = PedagogyPack.model_validate_json(path.read_text())
        assert pack.review_status == ReviewStatus.LLM_GENERATED


def test_loader_merges_and_template_wins(graph, corpus_dir, tmp_path):
    out_dir = tmp_path / "generated"
    run_ingest(
        graph,
        corpus_dir,
        out_dir,
        FakeLLM(GOOD_DRAFT),
        kc_ids={"kc.der.chain_rule", "kc.int.u_substitution"},
    )
    packs = load_packs(generated_dir=out_dir)
    # generated draft fills the gap for chain rule
    assert packs["kc.der.chain_rule"].review_status == ReviewStatus.LLM_GENERATED
    # human-authored template pack wins for u-substitution
    usub_ids = {m.id for m in packs["kc.int.u_substitution"].misconceptions}
    assert "m.usub.forget_dx" in usub_ids
    assert packs["kc.int.u_substitution"].review_status == ReviewStatus.DRAFT
