"""Scaffold exact Product/Quotient release attestations without approving them.

The generated JSON is deliberately *not* a ``ReleaseReviewManifest`` and
cannot be consumed by publication.  It binds completed source reviews to an
exact item bank, pedagogy catalog, graph, and candidate bundle while leaving
all final human judgments pending.  After reviewers fill those judgments, the
same command can validate the immutable bindings and emit a schema-v2 approval
manifest.  It never infers an approval from a source-review decision.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections import defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile

from tutor.content.product_quotient_release import (
    DEFAULT_GRAPH_PATH,
    DEFAULT_MANIFEST_PATH as DEFAULT_ASSESSMENT_REVIEW_PATH,
    DEFAULT_SOURCE_PATH as DEFAULT_ASSESSMENT_SOURCE_PATH,
    TARGET_CLOSURE,
    compile_release_inventory,
    load_manifest as load_assessment_reviews,
    load_source as load_assessment_source,
)
from tutor.content.publication import (
    prepare_release_candidate,
    validate_release_candidate_content,
)
from tutor.content.review_artifacts import (
    canonical_digest,
    canonical_json_bytes,
    compiled_family_digest,
    family_attestation_set_digest,
    kc_attestation_set_digest,
)
from tutor.packs.review_compiler import (
    DEFAULT_MANIFEST_PATH as DEFAULT_PEDAGOGY_REVIEW_PATH,
    DEFAULT_SOURCE_PATH as DEFAULT_PEDAGOGY_SOURCE_PATH,
    load_review_manifest as load_pedagogy_reviews,
    load_source_document as load_pedagogy_source,
    validate_compiled_catalog_provenance,
)
from tutor.schemas.assessment import ItemBankDocument
from tutor.schemas.content_authoring import ContentReviewManifest, ReviewDecision
from tutor.schemas.kc import GraphDocument
from tutor.schemas.pedagogy import PedagogyPackCatalog
from tutor.schemas.pedagogy_authoring import (
    PedagogyReviewDecision,
    PedagogyReviewManifest,
    PedagogySourceDocument,
)
from tutor.schemas.product_quotient_authoring import ProductQuotientBlueprintDocument
from tutor.schemas.release_authoring import (
    AttestationReviewDecision,
    FamilyApprovalAttestation,
    FamilyApprovalAttestationInput,
    KCApprovalAttestation,
    KCApprovalAttestationInput,
    ReleaseApprovalAttestation,
    ReleaseApprovalAttestationInput,
    ReleaseReviewManifest,
    ReleaseReviewScaffold,
)


class ProductQuotientAttestationError(ValueError):
    """The exact candidate cannot enter or complete human attestation."""


# These are the deliberately narrow claims in docs/pilot-curriculum-scope.md.
# They are included in the pending scaffold for explicit human confirmation;
# their presence does not confer approval.
PRODUCT_QUOTIENT_MASTERY_CLAIMS = {
    "kc.alg.exponent_rules": (
        "Simplify products, quotients, powers, zero exponents, and negative "
        "integer exponents for one symbolic base."
    ),
    "kc.der.power_rule": (
        "Differentiate integer powers, including zero and negative integer "
        "exponents, using symbolic exponent notation."
    ),
    "kc.der.sum_constant_rules": (
        "Differentiate integer-coefficient polynomial sums, differences, and "
        "constant multiples term by term."
    ),
    "kc.der.product_quotient": (
        "Apply product and quotient rules to reviewed point-value data or "
        "polynomial factors whose arithmetic yields an integer/symbolic "
        "result without unrelated simplification."
    ),
}

_FAMILY_REVIEW_FIELDS = {
    "decision",
    "mathematical_correctness",
    "accessibility",
    "instructional_clarity",
    "notes",
}
_KC_REVIEW_FIELDS = {
    "decision",
    "prepared_by",
    "reviewed_by",
    "reviewed_at",
    "construct_coverage",
    "family_independence",
    "difficulty_progression",
    "first_two_paths_reviewed",
    "notes",
}
_RELEASE_REVIEW_FIELDS = {
    "decision",
    "prepared_by",
    "reviewed_by",
    "reviewed_at",
    "cross_component_compatibility",
    "complete_hard_closure",
    "exact_bytes_reviewed",
    "notes",
}


def _stable_attestation_id(kind: str, release_id: str, subject_id: str) -> str:
    digest = hashlib.sha256(
        f"{kind}:{release_id}:{subject_id}".encode("utf-8")
    ).hexdigest()[:32]
    return f"attestation.{kind}.{digest}"


def _require_completed_source_reviews(
    assessment_reviews: ContentReviewManifest,
    pedagogy_reviews: PedagogyReviewManifest,
) -> None:
    pending_assessment = [
        entry.blueprint_id
        for entry in assessment_reviews.entries
        if entry.decision != ReviewDecision.APPROVED
    ]
    if pending_assessment:
        raise ProductQuotientAttestationError(
            "assessment source reviews are not complete; pending or rejected: "
            f"{sorted(pending_assessment)}"
        )
    pending_pedagogy = [
        entry.source_id
        for entry in pedagogy_reviews.entries
        if entry.decision != PedagogyReviewDecision.APPROVED
    ]
    if pending_pedagogy:
        raise ProductQuotientAttestationError(
            "pedagogy source reviews are not complete; pending or rejected: "
            f"{sorted(pending_pedagogy)}"
        )


def _assert_exact_candidate(
    graph: GraphDocument,
    assessment_source: ProductQuotientBlueprintDocument,
    assessment_reviews: ContentReviewManifest,
    item_bank: ItemBankDocument,
    pedagogy_source: PedagogySourceDocument,
    pedagogy_reviews: PedagogyReviewManifest,
    pedagogy_catalog: PedagogyPackCatalog,
) -> None:
    _require_completed_source_reviews(assessment_reviews, pedagogy_reviews)
    if set(assessment_source.released_kcs) != set(TARGET_CLOSURE):
        raise ProductQuotientAttestationError(
            "the attestation candidate must release the exact Product/Quotient closure"
        )
    compiled_bank, _report = compile_release_inventory(
        assessment_source,
        assessment_reviews,
        graph,
    )
    if canonical_digest(compiled_bank) != canonical_digest(item_bank):
        raise ProductQuotientAttestationError(
            "item bank bytes do not match the completed assessment source reviews"
        )
    validate_compiled_catalog_provenance(
        pedagogy_catalog,
        pedagogy_source,
        pedagogy_reviews,
    )
    validate_release_candidate_content(graph, item_bank, pedagogy_catalog)


def build_product_quotient_review_scaffold(
    graph: GraphDocument,
    assessment_source: ProductQuotientBlueprintDocument,
    assessment_reviews: ContentReviewManifest,
    item_bank: ItemBankDocument,
    pedagogy_source: PedagogySourceDocument,
    pedagogy_reviews: PedagogyReviewManifest,
    pedagogy_catalog: PedagogyPackCatalog,
    *,
    release_id: str,
) -> ReleaseReviewScaffold:
    """Bind an exact candidate while leaving every final judgment pending."""

    _assert_exact_candidate(
        graph,
        assessment_source,
        assessment_reviews,
        item_bank,
        pedagogy_source,
        pedagogy_reviews,
        pedagogy_catalog,
    )
    candidate = prepare_release_candidate(graph, item_bank, pedagogy_catalog)
    items_by_family: dict[str, list] = defaultdict(list)
    for item in item_bank.items:
        if item.kc_id in item_bank.released_kcs:
            items_by_family[item.family_id].append(item)

    family_inputs: list[FamilyApprovalAttestationInput] = []
    for family_id in sorted(items_by_family):
        items = items_by_family[family_id]
        provenance = items[0].provenance
        binding = (
            provenance.source_id,
            provenance.source_revision,
            provenance.source_digest,
            provenance.compiler_version,
            provenance.reviewed_by,
            provenance.reviewed_at,
        )
        if any(value is None for value in binding):
            raise ProductQuotientAttestationError(
                f"family {family_id!r} lacks completed source-review provenance"
            )
        if any(item.provenance != provenance for item in items[1:]):
            raise ProductQuotientAttestationError(
                f"family {family_id!r} has inconsistent provenance"
            )
        family_inputs.append(
            FamilyApprovalAttestationInput(
                attestation_id=_stable_attestation_id(
                    "family", release_id, family_id
                ),
                family_id=family_id,
                source_id=provenance.source_id,
                source_revision=provenance.source_revision,
                source_digest=provenance.source_digest,
                compiled_artifact_digest=compiled_family_digest(items),
                compiler_version=provenance.compiler_version,
                graph_version=graph.graph_version,
                author=provenance.author,
                reviewed_by=provenance.reviewed_by,
                reviewed_at=provenance.reviewed_at,
            )
        )

    source_families_by_kc: dict[str, list] = defaultdict(list)
    for family in assessment_source.families:
        source_families_by_kc[family.kc_id].append(family)
    family_kc = {
        item.family_id: item.kc_id
        for item in item_bank.items
        if item.kc_id in item_bank.released_kcs
    }
    if set(PRODUCT_QUOTIENT_MASTERY_CLAIMS) != set(item_bank.released_kcs):
        raise ProductQuotientAttestationError(
            "pilot mastery-claim coverage does not match released KCs"
        )

    kc_inputs: list[KCApprovalAttestationInput] = []
    for kc_id in sorted(item_bank.released_kcs):
        family_ids = tuple(
            sorted(
                family_id
                for family_id, family_kc_id in family_kc.items()
                if family_kc_id == kc_id
            )
        )
        construct_ids = tuple(
            sorted({family.construct_id for family in source_families_by_kc[kc_id]})
        )
        kc_inputs.append(
            KCApprovalAttestationInput(
                attestation_id=_stable_attestation_id("kc", release_id, kc_id),
                kc_id=kc_id,
                family_ids=family_ids,
                mastery_claim=PRODUCT_QUOTIENT_MASTERY_CLAIMS[kc_id],
                construct_ids=construct_ids,
            )
        )

    return ReleaseReviewScaffold(
        assessment_source_digest=canonical_digest(assessment_source),
        assessment_review_manifest_digest=canonical_digest(assessment_reviews),
        pedagogy_source_digest=canonical_digest(pedagogy_source),
        pedagogy_review_manifest_digest=canonical_digest(pedagogy_reviews),
        family_attestations=tuple(family_inputs),
        kc_attestations=tuple(kc_inputs),
        release_attestation=ReleaseApprovalAttestationInput(
            attestation_id=_stable_attestation_id(
                "release", release_id, candidate.bundle_sha256
            ),
            release_id=release_id,
            graph_version=graph.graph_version,
            graph_digest=candidate.graph_digest,
            bank_version=item_bank.bank_version,
            bank_digest=candidate.bank_digest,
            catalog_version=pedagogy_catalog.catalog_version,
            catalog_digest=candidate.catalog_digest,
            released_kcs=tuple(sorted(item_bank.released_kcs)),
            bundle_sha256=candidate.bundle_sha256,
        ),
    )


def _fixed_payload(model, mutable_fields: set[str]) -> dict[str, object]:
    return model.model_dump(mode="json", exclude=mutable_fields)


def _validate_scaffold_bindings(
    supplied: ReleaseReviewScaffold,
    expected: ReleaseReviewScaffold,
) -> None:
    for field in (
        "schema_version",
        "artifact_kind",
        "warning",
        "assessment_source_digest",
        "assessment_review_manifest_digest",
        "pedagogy_source_digest",
        "pedagogy_review_manifest_digest",
    ):
        if getattr(supplied, field) != getattr(expected, field):
            raise ProductQuotientAttestationError(
                f"filled scaffold changed immutable field {field!r}"
            )

    supplied_families = {item.family_id: item for item in supplied.family_attestations}
    expected_families = {item.family_id: item for item in expected.family_attestations}
    if set(supplied_families) != set(expected_families):
        raise ProductQuotientAttestationError("filled scaffold changed family coverage")
    for family_id in sorted(expected_families):
        if _fixed_payload(
            supplied_families[family_id], _FAMILY_REVIEW_FIELDS
        ) != _fixed_payload(expected_families[family_id], _FAMILY_REVIEW_FIELDS):
            raise ProductQuotientAttestationError(
                f"filled scaffold changed immutable family binding {family_id!r}"
            )

    supplied_kcs = {item.kc_id: item for item in supplied.kc_attestations}
    expected_kcs = {item.kc_id: item for item in expected.kc_attestations}
    if set(supplied_kcs) != set(expected_kcs):
        raise ProductQuotientAttestationError("filled scaffold changed KC coverage")
    for kc_id in sorted(expected_kcs):
        if _fixed_payload(supplied_kcs[kc_id], _KC_REVIEW_FIELDS) != _fixed_payload(
            expected_kcs[kc_id], _KC_REVIEW_FIELDS
        ):
            raise ProductQuotientAttestationError(
                f"filled scaffold changed immutable KC binding {kc_id!r}"
            )

    if _fixed_payload(
        supplied.release_attestation, _RELEASE_REVIEW_FIELDS
    ) != _fixed_payload(expected.release_attestation, _RELEASE_REVIEW_FIELDS):
        raise ProductQuotientAttestationError(
            "filled scaffold changed immutable release binding"
        )


def finalize_product_quotient_review_scaffold(
    supplied: ReleaseReviewScaffold,
    expected: ReleaseReviewScaffold,
) -> ReleaseReviewManifest:
    """Validate explicit approvals and materialize their dependent digests."""

    _validate_scaffold_bindings(supplied, expected)
    family_attestations: list[FamilyApprovalAttestation] = []
    for item in sorted(supplied.family_attestations, key=lambda value: value.family_id):
        if item.decision != AttestationReviewDecision.APPROVED:
            raise ProductQuotientAttestationError(
                f"family attestation remains {item.decision.value}: {item.family_id}"
            )
        family_attestations.append(
            FamilyApprovalAttestation(
                attestation_id=item.attestation_id,
                family_id=item.family_id,
                source_id=item.source_id,
                source_revision=item.source_revision,
                source_digest=item.source_digest,
                compiled_artifact_digest=item.compiled_artifact_digest,
                compiler_version=item.compiler_version,
                graph_version=item.graph_version,
                author=item.author,
                reviewed_by=item.reviewed_by,
                reviewed_at=item.reviewed_at,
                mathematical_correctness=item.mathematical_correctness,
                accessibility=item.accessibility,
                instructional_clarity=item.instructional_clarity,
            )
        )

    families_by_kc: dict[str, list[FamilyApprovalAttestation]] = defaultdict(list)
    # Use the KC records as the authoritative partition and reject overlap.
    family_kc: dict[str, str] = {}
    for kc_input in supplied.kc_attestations:
        for family_id in kc_input.family_ids:
            if family_id in family_kc:
                raise ProductQuotientAttestationError(
                    f"family {family_id!r} appears in more than one KC attestation"
                )
            family_kc[family_id] = kc_input.kc_id
    if set(family_kc) != {item.family_id for item in family_attestations}:
        raise ProductQuotientAttestationError(
            "KC attestations do not partition the approved families"
        )
    for item in family_attestations:
        families_by_kc[family_kc[item.family_id]].append(item)

    kc_attestations: list[KCApprovalAttestation] = []
    for item in sorted(supplied.kc_attestations, key=lambda value: value.kc_id):
        if item.decision != AttestationReviewDecision.APPROVED:
            raise ProductQuotientAttestationError(
                f"KC attestation remains {item.decision.value}: {item.kc_id}"
            )
        kc_attestations.append(
            KCApprovalAttestation(
                attestation_id=item.attestation_id,
                kc_id=item.kc_id,
                family_ids=item.family_ids,
                family_attestation_digest=family_attestation_set_digest(
                    families_by_kc[item.kc_id]
                ),
                mastery_claim=item.mastery_claim,
                construct_ids=item.construct_ids,
                prepared_by=item.prepared_by,
                reviewed_by=item.reviewed_by,
                reviewed_at=item.reviewed_at,
                construct_coverage=item.construct_coverage,
                family_independence=item.family_independence,
                difficulty_progression=item.difficulty_progression,
                first_two_paths_reviewed=item.first_two_paths_reviewed,
            )
        )

    release = supplied.release_attestation
    if release.decision != AttestationReviewDecision.APPROVED:
        raise ProductQuotientAttestationError(
            f"release attestation remains {release.decision.value}"
        )
    release_attestation = ReleaseApprovalAttestation(
        attestation_id=release.attestation_id,
        release_id=release.release_id,
        graph_version=release.graph_version,
        graph_digest=release.graph_digest,
        bank_version=release.bank_version,
        bank_digest=release.bank_digest,
        catalog_version=release.catalog_version,
        catalog_digest=release.catalog_digest,
        released_kcs=release.released_kcs,
        kc_attestation_digest=kc_attestation_set_digest(kc_attestations),
        bundle_sha256=release.bundle_sha256,
        prepared_by=release.prepared_by,
        reviewed_by=release.reviewed_by,
        reviewed_at=release.reviewed_at,
        cross_component_compatibility=release.cross_component_compatibility,
        complete_hard_closure=release.complete_hard_closure,
        exact_bytes_reviewed=release.exact_bytes_reviewed,
    )
    return ReleaseReviewManifest(
        schema_version=2,
        family_attestations=tuple(family_attestations),
        kc_attestations=tuple(kc_attestations),
        release_attestation=release_attestation,
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _load(path: Path, model_type):
    return model_type.model_validate_json(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    """Write/check a pending scaffold or finalize a human-filled scaffold."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH_PATH)
    parser.add_argument(
        "--assessment-source", type=Path, default=DEFAULT_ASSESSMENT_SOURCE_PATH
    )
    parser.add_argument(
        "--assessment-reviews", type=Path, default=DEFAULT_ASSESSMENT_REVIEW_PATH
    )
    parser.add_argument("--item-bank", type=Path, required=True)
    parser.add_argument(
        "--pedagogy-source", type=Path, default=DEFAULT_PEDAGOGY_SOURCE_PATH
    )
    parser.add_argument(
        "--pedagogy-reviews", type=Path, default=DEFAULT_PEDAGOGY_REVIEW_PATH
    )
    parser.add_argument("--pedagogy-catalog", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument(
        "--filled-scaffold",
        type=Path,
        default=None,
        help="validate explicit human decisions and emit ReleaseReviewManifest",
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if not args.check and args.out is None:
        parser.error("nothing to do: pass --check and/or --out PATH")

    try:
        graph = _load(args.graph, GraphDocument)
        assessment_source = load_assessment_source(args.assessment_source)
        assessment_reviews = load_assessment_reviews(args.assessment_reviews)
        item_bank = _load(args.item_bank, ItemBankDocument)
        pedagogy_source = load_pedagogy_source(args.pedagogy_source)
        pedagogy_reviews = load_pedagogy_reviews(args.pedagogy_reviews)
        pedagogy_catalog = _load(args.pedagogy_catalog, PedagogyPackCatalog)
        expected = build_product_quotient_review_scaffold(
            graph,
            assessment_source,
            assessment_reviews,
            item_bank,
            pedagogy_source,
            pedagogy_reviews,
            pedagogy_catalog,
            release_id=args.release_id,
        )
        if args.filled_scaffold is None:
            result = expected
            state = "pending"
        else:
            supplied = _load(args.filled_scaffold, ReleaseReviewScaffold)
            result = finalize_product_quotient_review_scaffold(supplied, expected)
            state = "approved"
        payload = canonical_json_bytes(result, trailing_newline=True)
        if args.out is not None:
            _atomic_write(args.out, payload)
    except Exception as exc:  # noqa: BLE001 - fail closed at the CLI boundary
        print(f"Product/Quotient attestation scaffold INVALID: {exc}", file=sys.stderr)
        return 1

    if args.check:
        print(
            "Product/Quotient attestation scaffold OK: "
            f"state={state}, families={len(expected.family_attestations)}, "
            f"KCs={len(expected.kc_attestations)}, "
            f"bundle_sha256={expected.release_attestation.bundle_sha256}, "
            f"artifact_sha256={hashlib.sha256(payload).hexdigest()}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
