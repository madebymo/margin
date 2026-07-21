"""Graph-v2 Product/Quotient pedagogy assets remain honest review inputs."""

from __future__ import annotations

from pathlib import Path

from tutor.packs.review_compiler import source_digest, validate_review_bundle
from tutor.schemas.common import WidgetType
from tutor.schemas.kc import GraphDocument
from tutor.schemas.pedagogy import PedagogyPackCatalog
from tutor.schemas.pedagogy_authoring import (
    PedagogyReviewDecision,
    PedagogyReviewManifest,
    PedagogySourceDocument,
)


SEED_DIR = Path(__file__).resolve().parents[1] / "src" / "tutor" / "seed"
SOURCE_PATH = SEED_DIR / "pedagogy_pack_sources_product_quotient_v2.json"
REVIEW_PATH = SEED_DIR / "pedagogy_pack_reviews_product_quotient_v2.json"
CATALOG_PATH = SEED_DIR / "pedagogy_catalog_v2.json"
GRAPH_PATH = SEED_DIR / "kc_graph_calc1.json"
EXPECTED_KCS = {
    "kc.alg.exponent_rules",
    "kc.der.power_rule",
    "kc.der.sum_constant_rules",
    "kc.der.product_quotient",
}


def _source_document() -> PedagogySourceDocument:
    return PedagogySourceDocument.model_validate_json(SOURCE_PATH.read_text(encoding="utf-8"))


def _review_manifest() -> PedagogyReviewManifest:
    return PedagogyReviewManifest.model_validate_json(REVIEW_PATH.read_text(encoding="utf-8"))


def test_product_quotient_v2_sources_have_complete_reviewable_inventory():
    document = _source_document()

    assert document.schema_version == 2
    assert document.graph_version == 2
    assert {source.kc_id for source in document.pack_sources} == EXPECTED_KCS
    assert len(document.pack_sources) == 4

    disabled_widgets = {WidgetType.CLICK_REGION, WidgetType.LIVE_INPUT}
    for source in document.pack_sources:
        assert source.revision == 2
        assert source.author == "AI-assisted implementation draft (unreviewed)"
        assert len(source.misconceptions) == 3
        assert len(source.metaphors) == 1
        assert len(source.error_patterns) == 3
        assert len(source.sources) == 2
        assert source.lesson_narrative
        assert source.remediation
        assert all(
            disabled_widgets.isdisjoint(metaphor.widget_affinity)
            for metaphor in source.metaphors
        )
        segments = (*source.lesson_narrative, *source.remediation)
        assert all(
            segment.kind != "math" or segment.spoken_text
            for segment in segments
        )


def test_product_quotient_v2_review_manifest_is_exact_and_pending():
    document = _source_document()
    manifest = _review_manifest()
    graph = GraphDocument.model_validate_json(GRAPH_PATH.read_text(encoding="utf-8"))

    assert manifest.graph_version == document.graph_version == graph.graph_version == 2
    assert len(manifest.entries) == len(document.pack_sources)
    entries = {
        (entry.source_id, entry.revision): entry
        for entry in manifest.entries
    }
    for source in document.pack_sources:
        entry = entries[(source.source_id, source.revision)]
        assert entry.source_digest == source_digest(source)
        assert entry.decision == PedagogyReviewDecision.PENDING
        assert entry.reviewed_by is None
        assert entry.reviewed_at is None

    assert validate_review_bundle(document, manifest, graph) is None


def test_graph_v2_catalog_is_empty_until_human_review_completes():
    catalog = PedagogyPackCatalog.model_validate_json(
        CATALOG_PATH.read_text(encoding="utf-8")
    )

    assert catalog.schema_version == 2
    assert catalog.graph_version == 2
    assert catalog.packs == ()
