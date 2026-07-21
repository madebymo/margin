"""Validate attestations and atomically publish one immutable v2 release directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from tutor.content.item_bank import (
    bundle_leakage_problems,
    render_prompt_segments,
    validate_item_bank,
)
from tutor.content.review_artifacts import (
    canonical_digest,
    canonical_json_bytes,
    compiled_family_digest,
    family_attestation_set_digest,
    kc_attestation_set_digest,
)
from tutor.graph.service import ancestor_subgraph
from tutor.schemas.assessment import (
    AssessmentItem,
    AssessmentSurface,
    ChoiceAnswerSpec,
    ItemBankDocument,
)
from tutor.schemas.kc import GraphDocument
from tutor.schemas.pedagogy import PedagogyPackCatalog
from tutor.schemas.release_authoring import (
    PublishedReleaseManifest,
    ReleasePublicationMetadata,
    ReleaseReviewManifest,
)

_EXACT_FAMILY_MATRIX = {
    AssessmentSurface.DIAGNOSTIC: 4,
    AssessmentSurface.CHECKIN: 5,
    AssessmentSurface.GUIDED_WIDGET: 1,
    AssessmentSurface.CAPSTONE: 2,
    AssessmentSurface.WORKED_EXAMPLE: 1,
}


class ReleasePublicationError(ValueError):
    """A candidate cannot cross the immutable publication boundary."""


@dataclass(frozen=True)
class ReleaseCandidate:
    """Canonical candidate bytes and component identities shown to reviewers."""

    bundle_payload: dict[str, object]
    bundle_bytes: bytes
    bundle_sha256: str
    graph_digest: str
    bank_digest: str
    catalog_digest: str


def prepare_release_candidate(
    graph: GraphDocument,
    item_bank: ItemBankDocument,
    pedagogy_catalog: PedagogyPackCatalog,
) -> ReleaseCandidate:
    """Create exact deterministic release bytes without promoting them."""
    payload: dict[str, object] = {
        "schema_version": 2,
        "graph": graph.model_dump(mode="json"),
        "item_bank": item_bank.model_dump(mode="json"),
        "pedagogy_catalog": pedagogy_catalog.model_dump(mode="json"),
    }
    bundle_bytes = canonical_json_bytes(payload, trailing_newline=True)
    return ReleaseCandidate(
        bundle_payload=payload,
        bundle_bytes=bundle_bytes,
        bundle_sha256=hashlib.sha256(bundle_bytes).hexdigest(),
        graph_digest=canonical_digest(graph),
        bank_digest=canonical_digest(item_bank),
        catalog_digest=canonical_digest(pedagogy_catalog),
    )


def _validate_modern_content(
    graph: GraphDocument,
    item_bank: ItemBankDocument,
    pedagogy_catalog: PedagogyPackCatalog,
) -> None:
    if item_bank.schema_version < 3:
        raise ReleasePublicationError(
            "publication requires an accessibility-complete schema-v3 item bank"
        )
    if pedagogy_catalog.schema_version < 2:
        raise ReleasePublicationError(
            "publication requires a schema-v2 reviewed pedagogy catalog"
        )
    released = set(item_bank.released_kcs)
    if not released:
        raise ReleasePublicationError("a published release must contain released KCs")
    catalog_kcs = set(pedagogy_catalog.pack_by_kc)
    if catalog_kcs != released:
        raise ReleasePublicationError(
            "pedagogy catalog coverage must exactly match released_kcs"
        )
    for kc_id in sorted(released):
        closure = ancestor_subgraph(graph, kc_id, hard_only=True).node_ids()
        missing = closure - released
        if missing:
            raise ReleasePublicationError(
                f"released KC {kc_id!r} has an incomplete hard closure: {sorted(missing)}"
            )

    families: dict[tuple[str, AssessmentSurface], set[str]] = defaultdict(set)
    for item in item_bank.items:
        if item.kc_id not in released:
            continue
        if isinstance(item.answer, ChoiceAnswerSpec):
            raise ReleasePublicationError("choice answers are not supported by the pilot")
        if len(item.eligible_surfaces) != 1:
            raise ReleasePublicationError(
                f"released family {item.family_id!r} must have exactly one surface"
            )
        families[(item.kc_id, item.eligible_surfaces[0])].add(item.family_id)
    for kc_id in sorted(released):
        for surface, expected in _EXACT_FAMILY_MATRIX.items():
            actual = len(families[(kc_id, surface)])
            if actual != expected:
                raise ReleasePublicationError(
                    f"{kc_id}/{surface.value} has {actual} families; requires exactly {expected}"
                )

    validation_errors = validate_item_bank(item_bank, graph, pedagogy_catalog)
    if validation_errors:
        raise ReleasePublicationError(
            "release content validation failed: " + "; ".join(validation_errors[:5])
        )

    scored_items = [
        item
        for item in item_bank.items
        if item.kc_id in released
        and AssessmentSurface.WORKED_EXAMPLE not in item.eligible_surfaces
    ]
    for pack in pedagogy_catalog.packs:
        for label, block in (
            ("lesson_narrative", pack.lesson_narrative),
            ("remediation", pack.remediation),
        ):
            leakage = bundle_leakage_problems(
                [render_prompt_segments(block)],
                scored_items,
                supervised=True,
            )
            if leakage:
                raise ReleasePublicationError(
                    f"{pack.kc_id}/{label} leaks released answers: {leakage[:3]}"
                )


def _items_by_family(item_bank: ItemBankDocument) -> dict[str, list[AssessmentItem]]:
    released = set(item_bank.released_kcs)
    result: dict[str, list[AssessmentItem]] = defaultdict(list)
    for item in item_bank.items:
        if item.kc_id in released:
            result[item.family_id].append(item)
    return dict(result)


def validate_release_reviews(
    graph: GraphDocument,
    item_bank: ItemBankDocument,
    pedagogy_catalog: PedagogyPackCatalog,
    reviews: ReleaseReviewManifest,
    publication: ReleasePublicationMetadata,
) -> tuple[ReleaseCandidate, PublishedReleaseManifest]:
    """Fail closed unless every exact artifact has independent human approval."""
    _validate_modern_content(graph, item_bank, pedagogy_catalog)
    candidate = prepare_release_candidate(graph, item_bank, pedagogy_catalog)
    released = tuple(sorted(item_bank.released_kcs))
    items_by_family = _items_by_family(item_bank)
    family_reviews = {
        item.family_id: item for item in reviews.family_attestations
    }
    if set(family_reviews) != set(items_by_family):
        raise ReleasePublicationError(
            "family attestations must exactly cover released families"
        )

    family_kcs: dict[str, str] = {}
    for family_id in sorted(items_by_family):
        items = items_by_family[family_id]
        review = family_reviews[family_id]
        family_kcs[family_id] = items[0].kc_id
        provenance_values = {
            (
                item.provenance.author,
                item.provenance.reviewed_by,
                item.provenance.reviewed_at,
                item.provenance.source_id,
                item.provenance.source_revision,
                item.provenance.source_digest,
                item.provenance.compiler_version,
            )
            for item in items
        }
        if len(provenance_values) != 1:
            raise ReleasePublicationError(
                f"family {family_id!r} has inconsistent compiled provenance"
            )
        (
            author,
            reviewed_by,
            reviewed_at,
            source_id,
            source_revision,
            source_digest,
            compiler_version,
        ) = next(iter(provenance_values))
        if None in {
            reviewed_by,
            reviewed_at,
            source_id,
            source_revision,
            source_digest,
            compiler_version,
        }:
            raise ReleasePublicationError(
                f"family {family_id!r} lacks complete compiled review provenance"
            )
        expected_binding = (
            source_id,
            source_revision,
            source_digest,
            compiler_version,
            graph.graph_version,
            author,
            reviewed_by,
            reviewed_at,
        )
        actual_binding = (
            review.source_id,
            review.source_revision,
            review.source_digest,
            review.compiler_version,
            review.graph_version,
            review.author,
            review.reviewed_by,
            review.reviewed_at,
        )
        if actual_binding != expected_binding:
            raise ReleasePublicationError(
                f"family attestation binding mismatch for {family_id!r}"
            )
        if review.compiled_artifact_digest != compiled_family_digest(items):
            raise ReleasePublicationError(
                f"compiled artifact digest mismatch for family {family_id!r}"
            )

    kc_reviews = {item.kc_id: item for item in reviews.kc_attestations}
    if set(kc_reviews) != set(released):
        raise ReleasePublicationError("KC attestations must exactly cover released_kcs")
    for kc_id in released:
        review = kc_reviews[kc_id]
        expected_families = tuple(
            sorted(
                family_id
                for family_id, family_kc in family_kcs.items()
                if family_kc == kc_id
            )
        )
        if review.family_ids != expected_families:
            raise ReleasePublicationError(
                f"KC attestation family coverage mismatch for {kc_id!r}"
            )
        family_set = [family_reviews[family_id] for family_id in expected_families]
        if review.family_attestation_digest != family_attestation_set_digest(family_set):
            raise ReleasePublicationError(
                f"KC attestation digest mismatch for {kc_id!r}"
            )

    release_review = reviews.release_attestation
    expected_release_binding = (
        graph.graph_version,
        candidate.graph_digest,
        item_bank.bank_version,
        candidate.bank_digest,
        pedagogy_catalog.catalog_version,
        candidate.catalog_digest,
        released,
        kc_attestation_set_digest(reviews.kc_attestations),
        candidate.bundle_sha256,
    )
    actual_release_binding = (
        release_review.graph_version,
        release_review.graph_digest,
        release_review.bank_version,
        release_review.bank_digest,
        release_review.catalog_version,
        release_review.catalog_digest,
        release_review.released_kcs,
        release_review.kc_attestation_digest,
        release_review.bundle_sha256,
    )
    if actual_release_binding != expected_release_binding:
        raise ReleasePublicationError("release attestation does not bind exact candidate bytes")

    latest_review = max(
        [release_review.reviewed_at]
        + [item.reviewed_at for item in reviews.family_attestations]
        + [item.reviewed_at for item in reviews.kc_attestations]
    )
    if publication.published_at < latest_review:
        raise ReleasePublicationError(
            "published_at cannot precede the latest approval timestamp"
        )

    manifest = PublishedReleaseManifest(
        release_id=release_review.release_id,
        bundle_sha256=candidate.bundle_sha256,
        reviews_sha256=hashlib.sha256(
            canonical_json_bytes(reviews, trailing_newline=True)
        ).hexdigest(),
        graph_version=graph.graph_version,
        graph_digest=candidate.graph_digest,
        bank_version=item_bank.bank_version,
        bank_digest=candidate.bank_digest,
        catalog_version=pedagogy_catalog.catalog_version,
        catalog_digest=candidate.catalog_digest,
        released_kcs=released,
        family_attestation_ids=tuple(
            item.attestation_id
            for item in sorted(
                reviews.family_attestations,
                key=lambda item: (item.family_id, item.attestation_id),
            )
        ),
        kc_attestation_ids=tuple(
            item.attestation_id
            for item in sorted(
                reviews.kc_attestations,
                key=lambda item: (item.kc_id, item.attestation_id),
            )
        ),
        release_attestation_id=release_review.attestation_id,
        release_attestation_digest=canonical_digest(release_review),
        published_by=publication.published_by,
        published_at=publication.published_at,
    )
    return candidate, manifest


def _write_fsynced(path: Path, payload: bytes) -> None:
    with path.open("xb") as destination:
        destination.write(payload)
        destination.flush()
        os.fsync(destination.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_release(
    destination: Path,
    graph: GraphDocument,
    item_bank: ItemBankDocument,
    pedagogy_catalog: PedagogyPackCatalog,
    reviews: ReleaseReviewManifest,
    publication: ReleasePublicationMetadata,
) -> PublishedReleaseManifest:
    """Atomically expose one new immutable release directory."""
    candidate, manifest = validate_release_reviews(
        graph,
        item_bank,
        pedagogy_catalog,
        reviews,
        publication,
    )
    destination = Path(destination)
    if destination.exists():
        raise ReleasePublicationError(
            "release destination already exists; version reuse is forbidden"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
        )
    )
    try:
        _write_fsynced(staging / "bundle.json", candidate.bundle_bytes)
        _write_fsynced(
            staging / "release-reviews.json",
            canonical_json_bytes(reviews, trailing_newline=True),
        )
        _write_fsynced(
            staging / "release-manifest.json",
            canonical_json_bytes(manifest, trailing_newline=True),
        )
        _write_fsynced(
            staging / "bundle.sha256",
            f"{candidate.bundle_sha256}  bundle.json\n".encode("ascii"),
        )

        staged_bundle = (staging / "bundle.json").read_bytes()
        if hashlib.sha256(staged_bundle).hexdigest() != candidate.bundle_sha256:
            raise ReleasePublicationError("staged bundle digest changed before publication")
        staged_reviews = (staging / "release-reviews.json").read_bytes()
        if hashlib.sha256(staged_reviews).hexdigest() != manifest.reviews_sha256:
            raise ReleasePublicationError(
                "staged review attestations changed before publication"
            )
        ReleaseReviewManifest.model_validate_json(staged_reviews)
        staged_payload = json.loads(staged_bundle)
        if staged_payload != candidate.bundle_payload:
            raise ReleasePublicationError("staged bundle content changed before publication")
        PublishedReleaseManifest.model_validate_json(
            (staging / "release-manifest.json").read_bytes()
        )
        _fsync_directory(staging)
        os.replace(staging, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _load_json(path: Path, model_type):
    return model_type.model_validate_json(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    """Check or publish one exact reviewed content release."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--item-bank", type=Path, required=True)
    parser.add_argument("--pedagogy-catalog", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--published-by", required=True)
    parser.add_argument("--published-at", required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    if not args.check and args.out_dir is None:
        parser.error("nothing to do: pass --check and/or --out-dir")
    try:
        graph = _load_json(args.graph, GraphDocument)
        bank = _load_json(args.item_bank, ItemBankDocument)
        catalog = _load_json(args.pedagogy_catalog, PedagogyPackCatalog)
        reviews = _load_json(args.reviews, ReleaseReviewManifest)
        publication = ReleasePublicationMetadata(
            published_by=args.published_by,
            published_at=args.published_at,
        )
        candidate, _manifest = validate_release_reviews(
            graph, bank, catalog, reviews, publication
        )
        if args.out_dir is not None:
            publish_release(
                args.out_dir,
                graph,
                bank,
                catalog,
                reviews,
                publication,
            )
    except Exception as exc:  # noqa: BLE001 - fail closed at the CLI boundary
        print(f"release publication INVALID: {exc}", file=sys.stderr)
        return 1
    counts = Counter(item.kc_id for item in reviews.kc_attestations)
    print(
        f"release publication OK: {reviews.release_attestation.release_id}, "
        f"{len(counts)} KCs, {len(reviews.family_attestations)} families, "
        f"sha256={candidate.bundle_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
