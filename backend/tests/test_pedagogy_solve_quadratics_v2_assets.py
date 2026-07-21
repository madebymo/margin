"""The Solve Quadratics pedagogy wave remains a complete, pending review input."""

from __future__ import annotations

from tutor.content.solve_quadratics_release import (
    AUTHOR,
    DEFAULT_BANK_PATH,
    DEFAULT_PEDAGOGY_MANIFEST_PATH,
    DEFAULT_PEDAGOGY_SOURCE_PATH,
    TARGET_KCS,
    draft_pedagogy_review_manifest,
    validate_pedagogy_item_separation,
)
from tutor.packs.review_compiler import source_digest, validate_review_bundle
from tutor.schemas.assessment import ItemBankDocument, MathPromptSegment
from tutor.schemas.common import WidgetType
from tutor.schemas.kc import GraphDocument
from tutor.schemas.pedagogy_authoring import (
    PedagogyReviewDecision,
    PedagogyReviewManifest,
    PedagogySourceDocument,
)
from tutor.seed.load_seed import GRAPH_PATH


def _source() -> PedagogySourceDocument:
    return PedagogySourceDocument.model_validate_json(
        DEFAULT_PEDAGOGY_SOURCE_PATH.read_text(encoding="utf-8")
    )


def _manifest() -> PedagogyReviewManifest:
    return PedagogyReviewManifest.model_validate_json(
        DEFAULT_PEDAGOGY_MANIFEST_PATH.read_text(encoding="utf-8")
    )


def test_packaged_pedagogy_is_the_canonical_four_pack_schema_v2_draft():
    source = _source()

    assert source.schema_version == 2
    assert source.graph_version == 2
    assert {pack.kc_id for pack in source.pack_sources} == set(TARGET_KCS)
    assert len(source.pack_sources) == 4
    for pack in source.pack_sources:
        assert pack.author == AUTHOR
        assert pack.revision == 1
        assert len(pack.misconceptions) == 3
        assert len(pack.metaphors) == 1
        assert len(pack.error_patterns) == 3
        assert len(pack.sources) == 2
        assert pack.lesson_narrative
        assert pack.remediation
        assert all(
            WidgetType.CLICK_REGION not in metaphor.widget_affinity
            and WidgetType.LIVE_INPUT not in metaphor.widget_affinity
            for metaphor in pack.metaphors
        )
        assert all(
            segment.spoken_text
            for segment in (*pack.lesson_narrative, *pack.remediation)
            if isinstance(segment, MathPromptSegment)
        )


def test_pedagogy_manifest_exactly_binds_every_source_as_pending():
    source = _source()
    manifest = _manifest()
    graph = GraphDocument.model_validate_json(GRAPH_PATH.read_text(encoding="utf-8"))

    assert manifest == draft_pedagogy_review_manifest(source)
    assert len(manifest.entries) == 4
    reviews = {(entry.source_id, entry.revision): entry for entry in manifest.entries}
    for pack in source.pack_sources:
        review = reviews[(pack.source_id, pack.revision)]
        assert review.source_digest == source_digest(pack)
        assert review.decision == PedagogyReviewDecision.PENDING
        assert review.reviewed_by is None
        assert review.reviewed_at is None
    assert validate_review_bundle(source, manifest, graph) is None


def test_pedagogy_blocks_disclose_no_draft_item_answer():
    bank = ItemBankDocument.model_validate_json(
        DEFAULT_BANK_PATH.read_text(encoding="utf-8")
    )

    assert validate_pedagogy_item_separation(bank, _source()) == ()


def test_pedagogy_sources_make_no_review_or_release_claim():
    source_text = DEFAULT_PEDAGOGY_SOURCE_PATH.read_text(encoding="utf-8")
    manifest_text = DEFAULT_PEDAGOGY_MANIFEST_PATH.read_text(encoding="utf-8")

    assert "human_approved" not in source_text
    assert '"reviewed_by": null' in manifest_text
    assert '"reviewed_at": null' in manifest_text
    assert '"decision": "pending"' in manifest_text
