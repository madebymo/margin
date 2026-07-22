"""Pending-only scaffolding for the exact Product/Quotient release review."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from tutor.content.product_quotient_attestations import (
    PRODUCT_QUOTIENT_MASTERY_CLAIMS,
    ProductQuotientAttestationError,
    build_product_quotient_review_scaffold,
    finalize_product_quotient_review_scaffold,
    main,
)
from tutor.content.product_quotient_release import (
    TARGET_CLOSURE,
    family_digest,
    load_manifest,
    load_source,
    promote_reviewed_inventory,
)
from tutor.content.publication import ReleasePublicationError, publish_release
from tutor.content.review_artifacts import canonical_json_bytes
from tutor.packs.review_compiler import (
    compile_pedagogy_catalog,
    load_review_manifest as load_pedagogy_reviews,
    load_source_document as load_pedagogy_source,
)
from tutor.schemas.content_authoring import ContentReviewManifest
from tutor.schemas.pedagogy_authoring import (
    PedagogyPublicationMetadata,
    PedagogyReviewManifest,
)
from tutor.schemas.release_authoring import (
    ReleasePublicationMetadata,
    ReleaseReviewManifest,
    ReleaseReviewScaffold,
)
from tutor.seed.load_seed import load_graph

_FAMILY_REVIEWED_AT = "2026-07-20T15:00:00Z"
_KC_REVIEWED_AT = "2026-07-20T17:00:00Z"
_RELEASE_REVIEWED_AT = "2026-07-20T18:00:00Z"


def _approved_assessment_reviews(source) -> ContentReviewManifest:
    payload = load_manifest().model_dump(mode="json")
    families = {
        (family.blueprint_id, family.revision): family
        for family in source.families
    }
    for entry in payload["entries"]:
        identity = (entry["blueprint_id"], entry["revision"])
        assert entry["source_digest"] == family_digest(source, families[identity])
        entry.update(
            {
                "decision": "approved",
                "reviewed_by": "Independent test mathematics reviewer",
                "reviewed_at": _FAMILY_REVIEWED_AT,
                "notes": "Test-only approval fixture.",
            }
        )
    return ContentReviewManifest.model_validate(payload)


def _approved_pedagogy_reviews() -> PedagogyReviewManifest:
    payload = load_pedagogy_reviews().model_dump(mode="json")
    for entry in payload["entries"]:
        entry.update(
            {
                "decision": "approved",
                "reviewed_by": "Independent test pedagogy reviewer",
                "reviewed_at": _FAMILY_REVIEWED_AT,
                "notes": "Test-only approval fixture.",
            }
        )
    return PedagogyReviewManifest.model_validate(payload)


@pytest.fixture(scope="module")
def exact_candidate():
    graph = load_graph()
    draft_source = load_source()
    assessment_reviews = _approved_assessment_reviews(draft_source)
    source, item_bank, _report = promote_reviewed_inventory(
        draft_source,
        assessment_reviews,
        graph,
        bank_version="test-product-quotient-release-v1",
    )
    pedagogy_source = load_pedagogy_source()
    pedagogy_reviews = _approved_pedagogy_reviews()
    pedagogy_catalog = compile_pedagogy_catalog(
        pedagogy_source,
        pedagogy_reviews,
        graph,
        PedagogyPublicationMetadata(
            catalog_version="test-product-quotient-pedagogy-v1",
            published_by="Test-only catalog publisher",
            published_at=datetime(2026, 7, 20, 16, 0, tzinfo=timezone.utc),
        ),
    )
    return (
        graph,
        source,
        assessment_reviews,
        item_bank,
        pedagogy_source,
        pedagogy_reviews,
        pedagogy_catalog,
    )


def _build(exact_candidate) -> ReleaseReviewScaffold:
    return build_product_quotient_review_scaffold(
        *exact_candidate,
        release_id="release.product-quotient-test-v1",
    )


@pytest.fixture(scope="module")
def exact_scaffold(exact_candidate) -> ReleaseReviewScaffold:
    return _build(exact_candidate)


def _fill_with_test_approvals(scaffold: ReleaseReviewScaffold) -> ReleaseReviewScaffold:
    payload = scaffold.model_dump(mode="json")
    for family in payload["family_attestations"]:
        family.update(
            {
                "decision": "approved",
                "mathematical_correctness": True,
                "accessibility": True,
                "instructional_clarity": True,
                "notes": "Explicit test-only final family attestation.",
            }
        )
    for kc in payload["kc_attestations"]:
        kc.update(
            {
                "decision": "approved",
                "prepared_by": "Test-only curriculum preparer",
                "reviewed_by": "Independent test-only KC reviewer",
                "reviewed_at": _KC_REVIEWED_AT,
                "construct_coverage": True,
                "family_independence": True,
                "difficulty_progression": True,
                "first_two_paths_reviewed": True,
                "notes": "Explicit test-only KC attestation.",
            }
        )
    payload["release_attestation"].update(
        {
            "decision": "approved",
            "prepared_by": "Test-only release preparer",
            "reviewed_by": "Independent test-only release reviewer",
            "reviewed_at": _RELEASE_REVIEWED_AT,
            "cross_component_compatibility": True,
            "complete_hard_closure": True,
            "exact_bytes_reviewed": True,
            "notes": "Explicit test-only release attestation.",
        }
    )
    return ReleaseReviewScaffold.model_validate(payload)


def test_scaffold_is_deterministic_exact_and_non_publishable(
    exact_candidate,
    exact_scaffold,
):
    first = exact_scaffold
    second = _build(exact_candidate)

    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert first.artifact_kind == "pending_release_review_scaffold"
    assert len(first.family_attestations) == 52
    assert len(first.kc_attestations) == 4
    assert {item.decision.value for item in first.family_attestations} == {"pending"}
    assert {item.decision.value for item in first.kc_attestations} == {"pending"}
    assert first.release_attestation.decision.value == "pending"
    assert all(item.mathematical_correctness is None for item in first.family_attestations)
    assert all(item.prepared_by is None for item in first.kc_attestations)
    assert first.release_attestation.prepared_by is None
    assert {
        item.kc_id: item.mastery_claim for item in first.kc_attestations
    } == PRODUCT_QUOTIENT_MASTERY_CLAIMS
    assert all(item.construct_ids for item in first.kc_attestations)
    with pytest.raises(ValidationError):
        ReleaseReviewManifest.model_validate(first.model_dump(mode="json"))
    with pytest.raises(ProductQuotientAttestationError, match="remains pending"):
        finalize_product_quotient_review_scaffold(first, second)


def test_scaffold_requires_completed_source_manifests(exact_candidate):
    values = list(exact_candidate)
    values[2] = load_manifest()

    with pytest.raises(
        ProductQuotientAttestationError,
        match="assessment source reviews are not complete",
    ):
        build_product_quotient_review_scaffold(
            *values,
            release_id="release.product-quotient-test-v1",
        )


def test_actual_manifest_path_finalizes_and_publishes_immutable_release(
    exact_candidate,
    exact_scaffold,
    tmp_path,
):
    expected = exact_scaffold
    filled = _fill_with_test_approvals(expected)
    reviews = finalize_product_quotient_review_scaffold(filled, expected)
    graph, _source, _assessment, bank, _ped_source, _ped_reviews, catalog = (
        exact_candidate
    )

    assert reviews.schema_version == 2
    assert len(reviews.family_attestations) == 52
    assert len(reviews.kc_attestations) == 4
    assert all(item.mastery_claim for item in reviews.kc_attestations)
    assert all(item.construct_ids for item in reviews.kc_attestations)
    publication = ReleasePublicationMetadata(
        published_by="Test-only publisher",
        published_at=datetime(2026, 7, 20, 19, 0, tzinfo=timezone.utc),
    )
    destination = tmp_path / "product-quotient-release"
    manifest = publish_release(
        destination,
        graph,
        bank,
        catalog,
        reviews,
        publication,
    )
    assert manifest.bundle_sha256 == expected.release_attestation.bundle_sha256
    assert manifest.bank_version == bank.bank_version
    assert manifest.catalog_version == catalog.catalog_version
    assert manifest.released_kcs == tuple(sorted(TARGET_CLOSURE))
    assert {path.name for path in destination.iterdir()} == {
        "bundle.json",
        "bundle.sha256",
        "release-manifest.json",
        "release-reviews.json",
    }

    published_bytes = {path.name: path.read_bytes() for path in destination.iterdir()}
    with pytest.raises(ReleasePublicationError, match="already exists"):
        publish_release(
            destination,
            graph,
            bank,
            catalog,
            reviews,
            publication,
        )
    assert {
        path.name: path.read_bytes() for path in destination.iterdir()
    } == published_bytes


def test_finalizer_rejects_changed_exact_binding(exact_scaffold):
    expected = exact_scaffold
    payload = _fill_with_test_approvals(expected).model_dump(mode="json")
    payload["release_attestation"]["bundle_sha256"] = "0" * 64
    tampered = ReleaseReviewScaffold.model_validate(payload)

    with pytest.raises(ProductQuotientAttestationError, match="release binding"):
        finalize_product_quotient_review_scaffold(tampered, expected)


def test_cli_writes_pending_scaffold_only(exact_candidate, tmp_path, capsys):
    (
        graph,
        source,
        assessment_reviews,
        bank,
        pedagogy_source,
        pedagogy_reviews,
        catalog,
    ) = exact_candidate
    paths = {
        "graph": (tmp_path / "graph.json", graph),
        "assessment-source": (tmp_path / "assessment-source.json", source),
        "assessment-reviews": (tmp_path / "assessment-reviews.json", assessment_reviews),
        "item-bank": (tmp_path / "item-bank.json", bank),
        "pedagogy-source": (tmp_path / "pedagogy-source.json", pedagogy_source),
        "pedagogy-reviews": (tmp_path / "pedagogy-reviews.json", pedagogy_reviews),
        "pedagogy-catalog": (tmp_path / "pedagogy-catalog.json", catalog),
    }
    arguments = ["--check", "--release-id", "release.product-quotient-test-v1"]
    for flag, (path, value) in paths.items():
        path.write_text(value.model_dump_json(indent=2) + "\n", encoding="utf-8")
        arguments.extend((f"--{flag}", str(path)))
    output = tmp_path / "review-scaffold.json"
    arguments.extend(("--out", str(output)))

    assert main(arguments) == 0
    written = ReleaseReviewScaffold.model_validate_json(output.read_text())
    assert written.release_attestation.decision.value == "pending"
    assert "state=pending" in capsys.readouterr().out


def test_schema_v2_manifest_requires_claim_and_constructor_coverage(exact_scaffold):
    expected = exact_scaffold
    reviews = finalize_product_quotient_review_scaffold(
        _fill_with_test_approvals(expected),
        expected,
    )
    payload = reviews.model_dump(mode="json")
    payload["kc_attestations"][0]["mastery_claim"] = None

    with pytest.raises(ValidationError, match="mastery claim and constructor coverage"):
        ReleaseReviewManifest.model_validate(payload)
