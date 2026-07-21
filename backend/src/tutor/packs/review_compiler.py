"""Validate reviewed pedagogy sources and publish an immutable catalog.

``--check`` accepts a digest-matched pending bundle as an honest authoring
state. Publication is stricter: every exact source revision must be approved
by someone other than its author, and callers must supply deterministic
publication metadata explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from tutor.schemas.common import ReviewStatus
from tutor.schemas.kc import GraphDocument
from tutor.schemas.pedagogy import (
    PedagogyPack,
    PedagogyPackCatalog,
    PedagogyPackProvenance,
)
from tutor.schemas.pedagogy_authoring import (
    PedagogyPackSource,
    PedagogyPublicationMetadata,
    PedagogyReviewDecision,
    PedagogyReviewEntry,
    PedagogyReviewManifest,
    PedagogySourceDocument,
)

COMPILER_VERSION = "pedagogy-review-compiler-v1"
SEED_DIR = Path(__file__).resolve().parents[1] / "seed"
DEFAULT_SOURCE_PATH = SEED_DIR / "pedagogy_pack_sources_product_quotient_v2.json"
DEFAULT_MANIFEST_PATH = SEED_DIR / "pedagogy_pack_reviews_product_quotient_v2.json"
DEFAULT_GRAPH_PATH = SEED_DIR / "kc_graph_calc1.json"


class PedagogyReviewError(ValueError):
    """The authoring or review bundle cannot cross the publication boundary."""


def load_source_document(path: Path | None = None) -> PedagogySourceDocument:
    """Parse the packaged draft sources or an explicit source document."""
    source = path or DEFAULT_SOURCE_PATH
    return PedagogySourceDocument.model_validate_json(source.read_text(encoding="utf-8"))


def load_review_manifest(path: Path | None = None) -> PedagogyReviewManifest:
    """Parse the packaged pending manifest or an explicit review manifest."""
    source = path or DEFAULT_MANIFEST_PATH
    return PedagogyReviewManifest.model_validate_json(source.read_text(encoding="utf-8"))


def source_digest(source: PedagogyPackSource) -> str:
    """Return the canonical SHA-256 identity reviewed by the manifest."""
    payload = source.model_dump(mode="json")
    # Preserve exact schema-v1 review identities. The additive instructional
    # blocks are absent from those source files and are enforced only by a
    # schema-v2 source document.
    if not source.lesson_narrative:
        payload.pop("lesson_narrative", None)
    if not source.remediation:
        payload.pop("remediation", None)
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _review_map(
    manifest: PedagogyReviewManifest,
) -> dict[tuple[str, int], PedagogyReviewEntry]:
    return {
        (entry.source_id, entry.revision): entry
        for entry in manifest.entries
    }


def validate_review_bundle(
    source_document: PedagogySourceDocument,
    manifest: PedagogyReviewManifest,
    graph: GraphDocument,
) -> None:
    """Validate pins, exact identity coverage, digests, and review independence."""
    if source_document.graph_version != graph.graph_version:
        raise PedagogyReviewError(
            "source/graph version mismatch: "
            f"source={source_document.graph_version}, graph={graph.graph_version}"
        )
    if manifest.graph_version != graph.graph_version:
        raise PedagogyReviewError(
            "manifest/graph version mismatch: "
            f"manifest={manifest.graph_version}, graph={graph.graph_version}"
        )
    if manifest.compiler_version != COMPILER_VERSION:
        raise PedagogyReviewError(
            f"manifest pins compiler {manifest.compiler_version!r}; "
            f"expected {COMPILER_VERSION!r}"
        )

    graph_kcs = graph.node_ids()
    unknown_kcs = {
        source.kc_id
        for source in source_document.pack_sources
        if source.kc_id not in graph_kcs
    }
    if unknown_kcs:
        raise PedagogyReviewError(f"source packs name unknown KCs: {sorted(unknown_kcs)}")

    sources = {
        (source.source_id, source.revision): source
        for source in source_document.pack_sources
    }
    reviews = _review_map(manifest)
    source_identities = set(sources)
    review_identities = set(reviews)
    missing = source_identities - review_identities
    extra = review_identities - source_identities
    if missing:
        raise PedagogyReviewError(f"missing review entries: {sorted(missing)}")
    if extra:
        raise PedagogyReviewError(f"review entries have no source: {sorted(extra)}")

    for identity in sorted(source_identities):
        source = sources[identity]
        review = reviews[identity]
        if review.source_digest != source_digest(source):
            raise PedagogyReviewError(
                f"review digest mismatch for {source.source_id}@{source.revision}"
            )
        if (
            review.reviewed_by is not None
            and review.reviewed_by.strip().casefold()
            == source.author.strip().casefold()
        ):
            raise PedagogyReviewError(
                f"source {source.source_id}@{source.revision} cannot review itself"
            )


def validate_compiled_catalog_provenance(
    catalog: PedagogyPackCatalog,
    source_document: PedagogySourceDocument,
    manifest: PedagogyReviewManifest,
) -> None:
    """Verify every compiled pack remains exactly bound to its reviewed source.

    Bindings are all-or-none across a catalog, exactly one per pack, and carry
    the source id, revision, canonical digest, and compiler pin. The associated
    pack content and human-review provenance must still match that source and
    manifest; copying valid provenance onto altered content therefore fails.
    """
    if catalog.graph_version != source_document.graph_version:
        raise PedagogyReviewError("compiled catalog/source graph pin mismatch")
    if manifest.graph_version != source_document.graph_version:
        raise PedagogyReviewError("compiled catalog/manifest graph pin mismatch")
    if manifest.compiler_version != COMPILER_VERSION:
        raise PedagogyReviewError("compiled catalog has an unsupported compiler pin")

    source_by_kc = {source.kc_id: source for source in source_document.pack_sources}
    pack_by_kc = {pack.kc_id: pack for pack in catalog.packs}
    if set(pack_by_kc) != set(source_by_kc):
        raise PedagogyReviewError("compiled catalog/source KC coverage mismatch")

    reviews = _review_map(manifest)
    source_identities = {
        (source.source_id, source.revision)
        for source in source_document.pack_sources
    }
    if set(reviews) != source_identities:
        raise PedagogyReviewError("compiled catalog review identity coverage mismatch")
    review_timestamps = [entry.reviewed_at for entry in reviews.values()]
    if any(
        entry.decision != PedagogyReviewDecision.APPROVED
        for entry in reviews.values()
    ) or any(timestamp is None for timestamp in review_timestamps):
        raise PedagogyReviewError("compiled catalog requires complete approved reviews")
    latest_review = max(
        timestamp for timestamp in review_timestamps if timestamp is not None
    )
    if catalog.published_at < latest_review:
        raise PedagogyReviewError(
            "compiled catalog publication predates its latest approval"
        )

    bindings = {
        pack.kc_id: (
            pack.provenance.source_id,
            pack.provenance.source_revision,
            pack.provenance.source_digest,
            pack.provenance.compiler_version,
        )
        if pack.provenance is not None
        else (None, None, None, None)
        for pack in catalog.packs
    }
    has_binding = {
        kc_id: all(value is not None for value in values)
        for kc_id, values in bindings.items()
    }
    if any(has_binding.values()) and not all(has_binding.values()):
        raise PedagogyReviewError("compiled source bindings must be all-or-none")
    if not any(has_binding.values()):
        raise PedagogyReviewError("compiled catalog lacks source provenance bindings")

    for kc_id in sorted(source_by_kc):
        source = source_by_kc[kc_id]
        pack = pack_by_kc[kc_id]
        identity = (source.source_id, source.revision)
        review = reviews[identity]
        if review.source_digest != source_digest(source):
            raise PedagogyReviewError(f"compiled source digest mismatch for {kc_id}")
        expected_binding = (
            source.source_id,
            source.revision,
            source_digest(source),
            manifest.compiler_version,
        )
        if bindings[kc_id] != expected_binding:
            raise PedagogyReviewError(f"compiled source binding mismatch for {kc_id}")
        if pack.version != source.revision:
            raise PedagogyReviewError(f"compiled source revision mismatch for {kc_id}")
        if tuple(pack.misconceptions) != tuple(
            sorted(source.misconceptions, key=lambda item: item.id)
        ):
            raise PedagogyReviewError(f"compiled misconception content mismatch for {kc_id}")
        if tuple(pack.metaphors) != tuple(
            sorted(source.metaphors, key=lambda item: item.id)
        ):
            raise PedagogyReviewError(f"compiled metaphor content mismatch for {kc_id}")
        if tuple(pack.error_patterns) != tuple(sorted(source.error_patterns)):
            raise PedagogyReviewError(f"compiled error-pattern content mismatch for {kc_id}")
        if tuple(pack.sources) != tuple(sorted(source.sources)):
            raise PedagogyReviewError(f"compiled citation content mismatch for {kc_id}")
        if tuple(pack.lesson_narrative) != tuple(source.lesson_narrative):
            raise PedagogyReviewError(
                f"compiled lesson-narrative content mismatch for {kc_id}"
            )
        if tuple(pack.remediation) != tuple(source.remediation):
            raise PedagogyReviewError(
                f"compiled remediation content mismatch for {kc_id}"
            )
        if review.decision != PedagogyReviewDecision.APPROVED:
            raise PedagogyReviewError(f"compiled source review is not approved for {kc_id}")
        if pack.provenance is None:
            raise PedagogyReviewError(f"compiled pack lacks provenance for {kc_id}")
        if (
            pack.provenance.author != source.author
            or pack.provenance.reviewed_by != review.reviewed_by
            or pack.provenance.reviewed_at != review.reviewed_at
        ):
            raise PedagogyReviewError(f"compiled review provenance mismatch for {kc_id}")


def compile_pedagogy_catalog(
    source_document: PedagogySourceDocument,
    manifest: PedagogyReviewManifest,
    graph: GraphDocument,
    publication: PedagogyPublicationMetadata | None,
) -> PedagogyPackCatalog:
    """Publish a deterministic catalog only from complete independent approval."""
    validate_review_bundle(source_document, manifest, graph)
    reviews = _review_map(manifest)
    incomplete = [
        f"{source.source_id}@{source.revision}:{reviews[(source.source_id, source.revision)].decision.value}"
        for source in source_document.pack_sources
        if reviews[(source.source_id, source.revision)].decision
        != PedagogyReviewDecision.APPROVED
    ]
    if incomplete:
        raise PedagogyReviewError(
            "all source revisions require approval before publication: "
            + ", ".join(sorted(incomplete))
        )
    if publication is None:
        raise PedagogyReviewError("explicit publication metadata is required")
    review_timestamps = [
        reviews[(source.source_id, source.revision)].reviewed_at
        for source in source_document.pack_sources
    ]
    if any(timestamp is None for timestamp in review_timestamps):
        raise PedagogyReviewError("approved reviews require timestamps")
    latest_review = max(
        timestamp for timestamp in review_timestamps if timestamp is not None
    )
    if publication.published_at < latest_review:
        raise PedagogyReviewError(
            "published_at cannot precede the latest approval reviewed_at"
        )

    packs: list[PedagogyPack] = []
    for source in sorted(source_document.pack_sources, key=lambda item: item.kc_id):
        review = reviews[(source.source_id, source.revision)]
        # The manifest schema guarantees these for approved entries. Keep the
        # runtime check explicit in case a future schema becomes more permissive.
        if review.reviewed_by is None or review.reviewed_at is None:
            raise PedagogyReviewError(
                f"approved source {source.source_id}@{source.revision} lacks provenance"
            )
        packs.append(
            PedagogyPack(
                kc_id=source.kc_id,
                misconceptions=sorted(source.misconceptions, key=lambda item: item.id),
                metaphors=sorted(source.metaphors, key=lambda item: item.id),
                error_patterns=sorted(source.error_patterns),
                sources=sorted(source.sources),
                lesson_narrative=source.lesson_narrative,
                remediation=source.remediation,
                review_status=ReviewStatus.HUMAN_APPROVED,
                version=source.revision,
                provenance=PedagogyPackProvenance(
                    author=source.author,
                    reviewed_by=review.reviewed_by,
                    reviewed_at=review.reviewed_at,
                    source_id=source.source_id,
                    source_revision=source.revision,
                    source_digest=source_digest(source),
                    compiler_version=manifest.compiler_version,
                ),
            )
        )

    catalog = PedagogyPackCatalog(
        schema_version=source_document.schema_version,
        catalog_version=publication.catalog_version,
        graph_version=graph.graph_version,
        published_by=publication.published_by,
        published_at=publication.published_at,
        packs=tuple(packs),
    )
    validate_compiled_catalog_provenance(catalog, source_document, manifest)
    return catalog


def _publication_from_args(args: argparse.Namespace) -> PedagogyPublicationMetadata | None:
    supplied = (args.catalog_version, args.published_by, args.published_at)
    if not any(value is not None for value in supplied):
        return None
    return PedagogyPublicationMetadata(
        catalog_version=args.catalog_version,
        published_by=args.published_by,
        published_at=args.published_at,
    )


def main(argv: list[str] | None = None) -> int:
    """Check a review bundle or publish its approved catalog."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH_PATH)
    parser.add_argument("--check", action="store_true", help="validate the review bundle")
    parser.add_argument("--out", type=Path, default=None, help="write an approved catalog")
    parser.add_argument("--catalog-version", default=None)
    parser.add_argument("--published-by", default=None)
    parser.add_argument("--published-at", default=None)
    args = parser.parse_args(argv)
    if not args.check and args.out is None:
        parser.error("nothing to do: pass --check and/or --out PATH")

    try:
        source_document = load_source_document(args.source)
        manifest = load_review_manifest(args.manifest)
        graph = GraphDocument.model_validate_json(args.graph.read_text(encoding="utf-8"))
        validate_review_bundle(source_document, manifest, graph)
        if args.check:
            counts = {
                decision.value: sum(
                    entry.decision == decision for entry in manifest.entries
                )
                for decision in PedagogyReviewDecision
            }
            print(
                f"pedagogy review bundle OK: {len(source_document.pack_sources)} "
                f"sources; pending={counts['pending']}, approved={counts['approved']}, "
                f"rejected={counts['rejected']}"
            )
        if args.out is not None:
            catalog = compile_pedagogy_catalog(
                source_document,
                manifest,
                graph,
                _publication_from_args(args),
            )
            args.out.write_text(catalog.model_dump_json(indent=2) + "\n", encoding="utf-8")
            print(f"wrote {args.out}")
    except Exception as exc:  # noqa: BLE001 - CLI boundary reports safe failure
        print(f"pedagogy review bundle INVALID: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
