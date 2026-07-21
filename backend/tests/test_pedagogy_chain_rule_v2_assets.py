"""The Wave 2 pedagogy assets remain complete, exact, and unapproved."""

from __future__ import annotations

import json
from pathlib import Path

from tutor.packs.review_compiler import source_digest, validate_review_bundle
from tutor.schemas.common import WidgetType
from tutor.schemas.kc import GraphDocument
from tutor.schemas.pedagogy_authoring import (
    PedagogyReviewDecision,
    PedagogyReviewManifest,
    PedagogySourceDocument,
)

SEED_DIR = Path(__file__).resolve().parents[1] / "src" / "tutor" / "seed"
SOURCE_PATH = SEED_DIR / "pedagogy_pack_sources_chain_rule_v2.json"
REVIEW_PATH = SEED_DIR / "pedagogy_pack_reviews_chain_rule_v2.json"
GRAPH_PATH = SEED_DIR / "kc_graph_calc1.json"
EXPECTED_KCS = {
    "kc.fun.function_notation",
    "kc.fun.composition",
    "kc.der.chain_rule",
}


def _source_document() -> PedagogySourceDocument:
    return PedagogySourceDocument.model_validate_json(
        SOURCE_PATH.read_text(encoding="utf-8")
    )


def _review_manifest() -> PedagogyReviewManifest:
    return PedagogyReviewManifest.model_validate_json(
        REVIEW_PATH.read_text(encoding="utf-8")
    )


def test_chain_rule_sources_have_the_exact_reviewable_pack_contract():
    document = _source_document()

    assert document.schema_version == 2
    assert document.graph_version == 2
    assert len(document.pack_sources) == 3
    assert {source.kc_id for source in document.pack_sources} == EXPECTED_KCS
    for source in document.pack_sources:
        assert source.revision == 1
        assert source.author == "AI-assisted implementation draft (unreviewed)"
        assert len(source.misconceptions) == 3
        assert len(source.metaphors) == 1
        assert len(source.error_patterns) == 3
        assert len(source.sources) == 2
        assert all(
            citation.startswith(("OpenStax", "College Board"))
            for citation in source.sources
        )
        assert source.lesson_narrative
        assert source.remediation


def test_chain_rule_instructional_math_has_reviewable_spoken_text():
    document = _source_document()

    for source in document.pack_sources:
        segments = (*source.lesson_narrative, *source.remediation)
        assert all(
            segment.kind != "math" or segment.spoken_text
            for segment in segments
        )
        assert all(
            WidgetType.CLICK_REGION not in metaphor.widget_affinity
            and WidgetType.LIVE_INPUT not in metaphor.widget_affinity
            for metaphor in source.metaphors
        )


def test_chain_rule_review_manifest_is_digest_exact_and_pending():
    document = _source_document()
    manifest = _review_manifest()
    graph = GraphDocument.model_validate_json(GRAPH_PATH.read_text(encoding="utf-8"))

    assert manifest.graph_version == document.graph_version == graph.graph_version == 2
    entries = {
        (entry.source_id, entry.revision): entry
        for entry in manifest.entries
    }
    assert len(entries) == 3
    for source in document.pack_sources:
        entry = entries[(source.source_id, source.revision)]
        assert entry.source_digest == source_digest(source)
        assert entry.decision == PedagogyReviewDecision.PENDING
        assert entry.reviewed_by is None
        assert entry.reviewed_at is None

    assert validate_review_bundle(document, manifest, graph) is None


def test_chain_rule_authoring_assets_claim_no_human_review():
    source_text = SOURCE_PATH.read_text(encoding="utf-8")
    review_payload = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    assert "reviewed_by" not in source_text
    assert "reviewed_at" not in source_text
    assert {entry["decision"] for entry in review_payload["entries"]} == {
        "pending"
    }
    assert all("reviewed_by" not in entry for entry in review_payload["entries"])
    assert all("reviewed_at" not in entry for entry in review_payload["entries"])
