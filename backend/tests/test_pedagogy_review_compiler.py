"""Review-bound pedagogy authoring and deterministic catalog publication."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from tutor.packs.review_compiler import (
    COMPILER_VERSION,
    PedagogyReviewError,
    compile_pedagogy_catalog,
    load_review_manifest,
    load_source_document,
    main,
    source_digest,
    validate_compiled_catalog_provenance,
    validate_review_bundle,
)
from tutor.schemas.common import ReviewStatus
from tutor.schemas.kc import GraphDocument
from tutor.schemas.pedagogy import PedagogyPackCatalog
from tutor.schemas.pedagogy_authoring import (
    PedagogyPublicationMetadata,
    PedagogyReviewEntry,
    PedagogyReviewManifest,
    PedagogySourceDocument,
)
from tutor.seed.load_seed import load_graph

EXPECTED_KCS = {
    "kc.alg.exponent_rules",
    "kc.der.power_rule",
    "kc.der.sum_constant_rules",
    "kc.der.product_quotient",
}


@pytest.fixture(scope="module")
def source_document() -> PedagogySourceDocument:
    return load_source_document()


@pytest.fixture(scope="module")
def manifest() -> PedagogyReviewManifest:
    return load_review_manifest()


@pytest.fixture(scope="module")
def graph() -> GraphDocument:
    return load_graph()


def _approved_manifest(
    manifest: PedagogyReviewManifest,
    *,
    reviewer: str = "Independent pedagogy review board",
) -> PedagogyReviewManifest:
    payload = manifest.model_dump(mode="json")
    for entry in payload["entries"]:
        entry.update(
            {
                "decision": "approved",
                "reviewed_by": reviewer,
                "reviewed_at": "2026-07-20T15:00:00Z",
            }
        )
    return PedagogyReviewManifest.model_validate(payload)


def _publication() -> PedagogyPublicationMetadata:
    return PedagogyPublicationMetadata(
        catalog_version="product-quotient-pedagogy-test-v1",
        published_by="Test release manager",
        published_at=datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc),
    )


def test_packaged_sources_are_complete_honest_four_kc_drafts(source_document):
    assert {source.kc_id for source in source_document.pack_sources} == EXPECTED_KCS
    assert len(source_document.pack_sources) == 4
    for source in source_document.pack_sources:
        assert source.author == "AI-assisted implementation draft (unreviewed)"
        assert len(source.misconceptions) == 3
        assert source.metaphors
        assert source.error_patterns
        assert source.sources
        assert all(len(citation) >= 12 for citation in source.sources)
        payload = source.model_dump(mode="json")
        assert "review_status" not in payload
        assert "reviewed_by" not in payload
        assert "reviewed_at" not in payload


def test_packaged_pending_manifest_exactly_covers_and_hashes_sources(
    source_document,
    manifest,
    graph,
):
    assert manifest.compiler_version == COMPILER_VERSION
    assert {entry.decision.value for entry in manifest.entries} == {"pending"}
    assert all(entry.reviewed_by is None for entry in manifest.entries)
    assert all(entry.reviewed_at is None for entry in manifest.entries)

    reviews = {
        (entry.source_id, entry.revision): entry
        for entry in manifest.entries
    }
    assert set(reviews) == {
        (source.source_id, source.revision)
        for source in source_document.pack_sources
    }
    for source in source_document.pack_sources:
        assert reviews[(source.source_id, source.revision)].source_digest == source_digest(
            source
        )
    assert validate_review_bundle(source_document, manifest, graph) is None


def test_authoring_contracts_are_strict_and_frozen(source_document, manifest):
    with pytest.raises(ValidationError, match="frozen"):
        source_document.source_version = "replacement"
    with pytest.raises(ValidationError, match="frozen"):
        manifest.manifest_version = "replacement"

    source_payload = source_document.model_dump(mode="json")
    source_payload["unexpected"] = True
    with pytest.raises(ValidationError, match="extra_forbidden"):
        PedagogySourceDocument.model_validate(source_payload)

    manifest_payload = manifest.model_dump(mode="json")
    manifest_payload["entries"][0]["unexpected"] = True
    with pytest.raises(ValidationError, match="extra_forbidden"):
        PedagogyReviewManifest.model_validate(manifest_payload)


@pytest.mark.parametrize(
    "mutation",
    [
        {"decision": "approved"},
        {
            "decision": "approved",
            "reviewed_by": "Independent reviewer",
            "reviewed_at": "2026-07-20T15:00:00",
        },
        {
            "decision": "pending",
            "reviewed_by": "Independent reviewer",
            "reviewed_at": "2026-07-20T15:00:00Z",
        },
    ],
)
def test_review_decisions_require_truthful_aware_provenance(manifest, mutation):
    payload = manifest.entries[0].model_dump(mode="json")
    payload.update(mutation)
    with pytest.raises(ValidationError, match="requires|timezone|cannot claim"):
        PedagogyReviewEntry.model_validate(payload)


def test_pending_and_rejected_sources_cannot_compile(source_document, manifest, graph):
    with pytest.raises(PedagogyReviewError, match="require approval"):
        compile_pedagogy_catalog(
            source_document,
            manifest,
            graph,
            _publication(),
        )

    rejected_payload = manifest.model_dump(mode="json")
    rejected_payload["entries"][0].update(
        {
            "decision": "rejected",
            "reviewed_by": "Independent pedagogy reviewer",
            "reviewed_at": "2026-07-20T15:00:00Z",
        }
    )
    rejected = PedagogyReviewManifest.model_validate(rejected_payload)
    with pytest.raises(PedagogyReviewError, match="rejected"):
        compile_pedagogy_catalog(source_document, rejected, graph, _publication())


def test_exact_identity_coverage_rejects_missing_and_extra_reviews(
    source_document,
    manifest,
    graph,
):
    missing_payload = manifest.model_dump(mode="json")
    missing_payload["entries"].pop()
    missing = PedagogyReviewManifest.model_validate(missing_payload)
    with pytest.raises(PedagogyReviewError, match="missing review entries"):
        validate_review_bundle(source_document, missing, graph)

    extra_payload = manifest.model_dump(mode="json")
    extra = deepcopy(extra_payload["entries"][0])
    extra["source_id"] = "pedagogy.source.unmatched"
    extra_payload["entries"].append(extra)
    extra_manifest = PedagogyReviewManifest.model_validate(extra_payload)
    with pytest.raises(PedagogyReviewError, match="have no source"):
        validate_review_bundle(source_document, extra_manifest, graph)


def test_source_or_manifest_tampering_invalidates_exact_digest(
    source_document,
    manifest,
    graph,
):
    source_payload = source_document.model_dump(mode="json")
    source_payload["pack_sources"][0]["misconceptions"][0]["description"] += " Changed."
    tampered_source = PedagogySourceDocument.model_validate(source_payload)
    with pytest.raises(PedagogyReviewError, match="digest mismatch"):
        validate_review_bundle(tampered_source, manifest, graph)

    manifest_payload = manifest.model_dump(mode="json")
    manifest_payload["entries"][0]["source_digest"] = "f" * 64
    tampered_manifest = PedagogyReviewManifest.model_validate(manifest_payload)
    with pytest.raises(PedagogyReviewError, match="digest mismatch"):
        validate_review_bundle(source_document, tampered_manifest, graph)


def test_graph_and_compiler_pins_fail_closed(source_document, manifest, graph):
    source_v2 = source_document.model_copy(
        update={"graph_version": graph.graph_version + 1}
    )
    with pytest.raises(PedagogyReviewError, match="source/graph version mismatch"):
        validate_review_bundle(source_v2, manifest, graph)

    manifest_v2 = manifest.model_copy(
        update={"graph_version": graph.graph_version + 1}
    )
    with pytest.raises(PedagogyReviewError, match="manifest/graph version mismatch"):
        validate_review_bundle(source_document, manifest_v2, graph)

    wrong_compiler = manifest.model_copy(
        update={"compiler_version": "pedagogy-review-compiler-v999"}
    )
    with pytest.raises(PedagogyReviewError, match="expected"):
        validate_review_bundle(source_document, wrong_compiler, graph)


def test_unknown_pack_kc_fails_before_review_can_promote_it(
    source_document,
    manifest,
    graph,
):
    payload = source_document.model_dump(mode="json")
    payload["pack_sources"][0]["kc_id"] = "kc.der.unknown_skill"
    unknown = PedagogySourceDocument.model_validate(payload)
    with pytest.raises(PedagogyReviewError, match="unknown KCs"):
        validate_review_bundle(unknown, manifest, graph)


def test_source_author_cannot_self_review(
    source_document,
    manifest,
    graph,
):
    author = source_document.pack_sources[0].author
    for reviewer in (author, f" {author.upper()} "):
        self_reviewed = _approved_manifest(manifest, reviewer=reviewer)
        with pytest.raises(PedagogyReviewError, match="cannot review itself"):
            validate_review_bundle(source_document, self_reviewed, graph)


def test_approved_bundle_still_requires_explicit_publication_metadata(
    source_document,
    manifest,
    graph,
):
    approved = _approved_manifest(manifest)
    with pytest.raises(PedagogyReviewError, match="explicit publication metadata"):
        compile_pedagogy_catalog(source_document, approved, graph, None)

    with pytest.raises(ValidationError, match="timezone"):
        PedagogyPublicationMetadata(
            catalog_version="invalid-publication-v1",
            published_by="Release manager",
            published_at="2026-07-21T12:00:00",
        )


def test_publication_cannot_predate_any_approval(source_document, manifest, graph):
    approved = _approved_manifest(manifest)
    too_early = PedagogyPublicationMetadata(
        catalog_version="too-early-publication-v1",
        published_by="Release manager",
        published_at="2026-07-20T14:59:59Z",
    )
    with pytest.raises(PedagogyReviewError, match="cannot precede"):
        compile_pedagogy_catalog(source_document, approved, graph, too_early)

    exactly_at_review = PedagogyPublicationMetadata(
        catalog_version="same-time-publication-v1",
        published_by="Release manager",
        published_at="2026-07-20T15:00:00Z",
    )
    catalog = compile_pedagogy_catalog(
        source_document,
        approved,
        graph,
        exactly_at_review,
    )
    assert catalog.published_at == exactly_at_review.published_at

    catalog_payload = catalog.model_dump(mode="json")
    catalog_payload["published_at"] = "2026-07-20T14:59:59Z"
    tampered_catalog = PedagogyPackCatalog.model_validate(catalog_payload)
    with pytest.raises(PedagogyReviewError, match="predates"):
        validate_compiled_catalog_provenance(
            tampered_catalog,
            source_document,
            approved,
        )


def test_approved_compilation_is_deterministic_reviewed_and_sorted(
    source_document,
    manifest,
    graph,
):
    approved = _approved_manifest(manifest)
    publication = _publication()
    first = compile_pedagogy_catalog(
        source_document,
        approved,
        graph,
        publication,
    )
    second = compile_pedagogy_catalog(
        source_document.model_copy(
            update={"pack_sources": tuple(reversed(source_document.pack_sources))}
        ),
        approved.model_copy(update={"entries": tuple(reversed(approved.entries))}),
        graph,
        publication,
    )

    assert first.model_dump_json() == second.model_dump_json()
    assert first.catalog_version == publication.catalog_version
    assert first.published_by == publication.published_by
    assert first.published_at == publication.published_at
    assert [pack.kc_id for pack in first.packs] == sorted(EXPECTED_KCS)
    assert {pack.review_status for pack in first.packs} == {
        ReviewStatus.HUMAN_APPROVED
    }
    assert all(pack.provenance is not None for pack in first.packs)
    assert all(pack.provenance.author != pack.provenance.reviewed_by for pack in first.packs)
    assert all(len(pack.misconceptions) == 3 for pack in first.packs)
    assert all(pack.metaphors for pack in first.packs)
    assert validate_compiled_catalog_provenance(first, source_document, approved) is None

    sources = {source.kc_id: source for source in source_document.pack_sources}
    for pack in first.packs:
        source = sources[pack.kc_id]
        assert pack.sources == sorted(source.sources)
        assert pack.provenance is not None
        assert pack.provenance.source_id == source.source_id
        assert pack.provenance.source_revision == source.revision
        assert pack.provenance.source_digest == source_digest(source)
        assert pack.provenance.compiler_version == COMPILER_VERSION


def test_compiled_provenance_binding_rejects_partial_mismatch_and_content_tampering(
    source_document,
    manifest,
    graph,
):
    approved = _approved_manifest(manifest)
    catalog = compile_pedagogy_catalog(
        source_document,
        approved,
        graph,
        _publication(),
    )

    partial_payload = catalog.model_dump(mode="json")
    partial_provenance = partial_payload["packs"][0]["provenance"]
    partial_provenance["source_id"] = None
    partial_provenance["source_revision"] = None
    partial_provenance["source_digest"] = None
    partial_provenance["compiler_version"] = None
    partial = PedagogyPackCatalog.model_validate(partial_payload)
    with pytest.raises(PedagogyReviewError, match="all-or-none"):
        validate_compiled_catalog_provenance(partial, source_document, approved)

    absent_payload = catalog.model_dump(mode="json")
    for pack in absent_payload["packs"]:
        provenance = pack["provenance"]
        provenance["source_id"] = None
        provenance["source_revision"] = None
        provenance["source_digest"] = None
        provenance["compiler_version"] = None
    absent = PedagogyPackCatalog.model_validate(absent_payload)
    with pytest.raises(PedagogyReviewError, match="lacks source provenance"):
        validate_compiled_catalog_provenance(absent, source_document, approved)

    incomplete_payload = catalog.model_dump(mode="json")
    incomplete_payload["packs"][0]["provenance"]["source_digest"] = None
    with pytest.raises(ValidationError, match="must be supplied together"):
        PedagogyPackCatalog.model_validate(incomplete_payload)

    mismatch_payload = catalog.model_dump(mode="json")
    mismatch_payload["packs"][0]["provenance"]["source_digest"] = "f" * 64
    mismatch = PedagogyPackCatalog.model_validate(mismatch_payload)
    with pytest.raises(PedagogyReviewError, match="binding mismatch"):
        validate_compiled_catalog_provenance(mismatch, source_document, approved)

    content_payload = catalog.model_dump(mode="json")
    content_payload["packs"][0]["error_patterns"][0] += " Altered."
    altered_content = PedagogyPackCatalog.model_validate(content_payload)
    with pytest.raises(PedagogyReviewError, match="error-pattern content mismatch"):
        validate_compiled_catalog_provenance(
            altered_content,
            source_document,
            approved,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("misconceptions", [], "at least 3"),
        ("metaphors", [], "at least 1"),
        ("sources", ["short"], "meaningful citations"),
    ],
)
def test_source_schema_enforces_reviewable_content(
    source_document,
    field,
    value,
    message,
):
    payload = source_document.model_dump(mode="json")
    payload["pack_sources"][0][field] = value
    with pytest.raises(ValidationError, match=message):
        PedagogySourceDocument.model_validate(payload)


def test_cli_check_accepts_pending_but_out_refuses_it(tmp_path, capsys):
    assert main(["--check"]) == 0
    output = capsys.readouterr()
    assert "pending=4, approved=0, rejected=0" in output.out

    destination = tmp_path / "must-not-publish.json"
    assert main(["--out", str(destination)]) == 1
    output = capsys.readouterr()
    assert "require approval" in output.err
    assert not destination.exists()


def test_cli_out_writes_only_an_approved_explicit_release(
    tmp_path,
    source_document,
    manifest,
):
    approved_path = tmp_path / "approved-manifest.json"
    approved_path.write_text(
        _approved_manifest(manifest).model_dump_json(),
        encoding="utf-8",
    )
    destination = tmp_path / "catalog.json"

    assert main(
        [
            "--manifest",
            str(approved_path),
            "--out",
            str(destination),
            "--catalog-version",
            "product-quotient-pedagogy-cli-test-v1",
            "--published-by",
            "CLI test release manager",
            "--published-at",
            "2026-07-21T12:00:00Z",
        ]
    ) == 0

    catalog = PedagogyPackCatalog.model_validate_json(destination.read_text())
    assert catalog.catalog_version == "product-quotient-pedagogy-cli-test-v1"
    assert {pack.kc_id for pack in catalog.packs} == EXPECTED_KCS
