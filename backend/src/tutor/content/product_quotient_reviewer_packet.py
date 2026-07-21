"""Build the private review packet for the pending Product/Quotient draft.

This command deliberately stops before either approval or publication. It
validates the exact assessment and pedagogy source/review manifests, compiles
the draft assessment items, and emits truth-bearing JSON/HTML for an
independent human reviewer. The resulting directory contains expected answers
and private widget scoring data and must never be mounted by the web app.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from tutor.content.product_quotient_release import (
    DEFAULT_GRAPH_PATH,
    DEFAULT_MANIFEST_PATH as DEFAULT_ASSESSMENT_REVIEW_PATH,
    DEFAULT_SOURCE_PATH as DEFAULT_ASSESSMENT_SOURCE_PATH,
    compile_release_inventory,
    family_digest,
    load_manifest,
    load_source,
)
from tutor.content.review_artifacts import canonical_digest
from tutor.content.reviewer_packet import (
    ReviewerPacketError,
    _allocation_paths,
    _family_entries,
    _review_rendering,
    _similarity_findings,
    write_reviewer_packet,
)
from tutor.packs.review_compiler import (
    DEFAULT_MANIFEST_PATH as DEFAULT_PEDAGOGY_REVIEW_PATH,
    DEFAULT_SOURCE_PATH as DEFAULT_PEDAGOGY_SOURCE_PATH,
    load_review_manifest as load_pedagogy_reviews,
    load_source_document as load_pedagogy_source,
    source_digest as pedagogy_source_digest,
    validate_review_bundle,
)
from tutor.schemas.common import EdgeType, ReviewStatus
from tutor.schemas.content_authoring import ContentReviewManifest, ReviewDecision
from tutor.schemas.kc import GraphDocument
from tutor.schemas.pedagogy_authoring import (
    PedagogyReviewDecision,
    PedagogyReviewManifest,
    PedagogySourceDocument,
)
from tutor.schemas.product_quotient_authoring import ProductQuotientBlueprintDocument


def _require_pending_review_state(
    assessment_source: ProductQuotientBlueprintDocument,
    assessment_reviews: ContentReviewManifest,
    pedagogy_reviews: PedagogyReviewManifest,
) -> None:
    """Keep this path on the non-promoting side of the review boundary."""
    if assessment_source.released_kcs:
        raise ReviewerPacketError("pending reviewer packets require released_kcs to remain empty")
    nonpending_assessment = [
        entry.blueprint_id
        for entry in assessment_reviews.entries
        if entry.decision != ReviewDecision.PENDING
    ]
    if nonpending_assessment:
        raise ReviewerPacketError(
            "pending reviewer packet cannot consume completed assessment decisions: "
            f"{sorted(nonpending_assessment)}"
        )
    nonpending_pedagogy = [
        entry.source_id
        for entry in pedagogy_reviews.entries
        if entry.decision != PedagogyReviewDecision.PENDING
    ]
    if nonpending_pedagogy:
        raise ReviewerPacketError(
            "pending reviewer packet cannot consume completed pedagogy decisions: "
            f"{sorted(nonpending_pedagogy)}"
        )


def _decision_counts(values: list[str]) -> dict[str, int]:
    counts = Counter(values)
    return {key: counts[key] for key in sorted(counts)}


def _knowledge_component_context(
    graph: GraphDocument,
    kc_ids: set[str],
) -> list[dict[str, object]]:
    nodes = {node.id: node for node in graph.nodes}
    return [
        {
            "kc_id": kc_id,
            "name": nodes[kc_id].name,
            "description": nodes[kc_id].description,
            "course_level": nodes[kc_id].course_level,
            "hard_prerequisite_ids": sorted(
                edge.from_kc
                for edge in graph.edges
                if edge.to_kc == kc_id and edge.type == EdgeType.HARD and edge.from_kc in kc_ids
            ),
            "canonical_examples": list(nodes[kc_id].canonical_examples),
            "canonical_examples_role": "explanatory_seed_only_never_scored",
        }
        for kc_id in sorted(kc_ids)
    ]


def build_pending_product_quotient_packet(
    graph: GraphDocument,
    assessment_source: ProductQuotientBlueprintDocument,
    assessment_reviews: ContentReviewManifest,
    pedagogy_source: PedagogySourceDocument,
    pedagogy_reviews: PedagogyReviewManifest,
) -> dict[str, object]:
    """Compile exact pending inputs into a deterministic, private review packet."""
    _require_pending_review_state(
        assessment_source,
        assessment_reviews,
        pedagogy_reviews,
    )
    validate_review_bundle(pedagogy_source, pedagogy_reviews, graph)
    bank, separation = compile_release_inventory(
        assessment_source,
        assessment_reviews,
        graph,
    )
    if bank.released_kcs:
        raise ReviewerPacketError("draft compilation unexpectedly released content")
    if any(item.review_status != ReviewStatus.DRAFT for item in bank.items):
        raise ReviewerPacketError("draft compilation unexpectedly approved an item")

    assessment_kcs = set(assessment_source.target_kcs)
    pedagogy_kcs = {source.kc_id for source in pedagogy_source.pack_sources}
    if pedagogy_kcs != assessment_kcs:
        raise ReviewerPacketError("assessment and pedagogy review inputs cover different KCs")

    blueprints_by_family = {family.family_id: family for family in assessment_source.families}
    construct_ids = {family.family_id: family.construct_id for family in assessment_source.families}
    assessment_review_map = {
        (entry.blueprint_id, entry.revision): entry for entry in assessment_reviews.entries
    }
    families, shapes = _family_entries(bank, construct_ids)
    for family_entry in families:
        family_id = str(family_entry["family_id"])
        blueprint = blueprints_by_family[family_id]
        review = assessment_review_map[(blueprint.blueprint_id, blueprint.revision)]
        exact_source_digest = family_digest(assessment_source, blueprint)
        if review.source_digest != exact_source_digest:
            # The compiler already enforces this. Retain an explicit assertion
            # at packet assembly so a future compiler refactor cannot weaken
            # the truth-bearing review artifact.
            raise ReviewerPacketError(
                f"assessment review digest mismatch for {blueprint.blueprint_id}"
            )
        family_entry["source_blueprint"] = blueprint.model_dump(mode="json")
        family_entry["source_review"] = review.model_dump(mode="json")
        family_entry["review_requirements"] = [
            "mathematical_correctness",
            "accessibility",
            "instructional_clarity",
            "construct_coverage",
            "evidence_family_independence",
        ]

    near_isomorphic, similarity_warnings = _similarity_findings(families, shapes)

    pedagogy_review_map = {
        (entry.source_id, entry.revision): entry for entry in pedagogy_reviews.entries
    }
    pedagogy_packs: list[dict[str, object]] = []
    for source in sorted(pedagogy_source.pack_sources, key=lambda item: item.kc_id):
        review = pedagogy_review_map[(source.source_id, source.revision)]
        exact_source_digest = pedagogy_source_digest(source)
        if review.source_digest != exact_source_digest:
            raise ReviewerPacketError(f"pedagogy review digest mismatch for {source.source_id}")
        pedagogy_packs.append(
            {
                "kc_id": source.kc_id,
                "version": source.revision,
                "review_status": ReviewStatus.DRAFT.value,
                "author": source.author,
                "source_id": source.source_id,
                "source_revision": source.revision,
                "source_digest": exact_source_digest,
                "compiler_version": pedagogy_reviews.compiler_version,
                "source_review": review.model_dump(mode="json"),
                "lesson_narrative": [
                    segment.model_dump(mode="json") for segment in source.lesson_narrative
                ],
                "lesson_narrative_rendering": _review_rendering(source.lesson_narrative),
                "remediation": [segment.model_dump(mode="json") for segment in source.remediation],
                "remediation_rendering": _review_rendering(source.remediation),
                "misconceptions": [item.model_dump(mode="json") for item in source.misconceptions],
                "metaphors": [item.model_dump(mode="json") for item in source.metaphors],
                "error_patterns": list(source.error_patterns),
                "citations": list(source.sources),
                "review_requirements": [
                    "mathematical_correctness",
                    "accessibility",
                    "instructional_clarity",
                    "citation_relevance",
                ],
            }
        )

    graph_digest = canonical_digest(graph)
    bank_digest = canonical_digest(bank)
    assessment_source_document_digest = canonical_digest(assessment_source)
    assessment_manifest_digest = canonical_digest(assessment_reviews)
    pedagogy_source_document_digest = canonical_digest(pedagogy_source)
    pedagogy_manifest_digest = canonical_digest(pedagogy_reviews)
    draft_compilation_digest = canonical_digest(
        {
            "graph_digest": graph_digest,
            "bank_digest": bank_digest,
            "assessment_source_digest": assessment_source_document_digest,
            "assessment_review_manifest_digest": assessment_manifest_digest,
            "pedagogy_source_digest": pedagogy_source_document_digest,
            "pedagogy_review_manifest_digest": pedagogy_manifest_digest,
        }
    )

    packet: dict[str, object] = {
        "schema_version": 2,
        "artifact_kind": "pending_product_quotient_review",
        "warning": (
            "PRIVATE OFFLINE REVIEW ARTIFACT: DRAFT AND UNRELEASED. Contains expected "
            "answers and private scoring data; never serve to learners."
        ),
        "workflow_state": "pending_independent_human_review",
        "publication_eligible": False,
        "review_workflow": {
            "changes_review_manifests": False,
            "creates_approval_records": False,
            "creates_release_artifacts": False,
            "assessment_decisions": _decision_counts(
                [entry.decision.value for entry in assessment_reviews.entries]
            ),
            "pedagogy_decisions": _decision_counts(
                [entry.decision.value for entry in pedagogy_reviews.entries]
            ),
            "required_next_action": (
                "Independent humans review the exact bound source and compiled bytes, then "
                "record decisions in the separate manifests. Packet generation itself "
                "confers no approval."
            ),
        },
        "graph_version": graph.graph_version,
        "graph_digest": graph_digest,
        "bank_version": bank.bank_version,
        "bank_digest": bank_digest,
        "assessment_source_version": assessment_source.blueprint_version,
        "assessment_source_digest": assessment_source_document_digest,
        "assessment_review_manifest_version": assessment_reviews.manifest_version,
        "assessment_review_manifest_digest": assessment_manifest_digest,
        "pedagogy_source_version": pedagogy_source.source_version,
        "pedagogy_source_digest": pedagogy_source_document_digest,
        "pedagogy_review_manifest_version": pedagogy_reviews.manifest_version,
        "pedagogy_review_manifest_digest": pedagogy_manifest_digest,
        "draft_compilation_digest": draft_compilation_digest,
        "released_kcs": [],
        "warnings": [
            f"{len(families)} assessment families remain draft and pending review",
            f"{len(pedagogy_packs)} pedagogy packs remain draft and pending review",
            (
                f"{len(similarity_warnings)} similarity findings require explicit human "
                "family-independence judgment"
            ),
        ],
        "knowledge_components": _knowledge_component_context(graph, assessment_kcs),
        "first_two_paths": _allocation_paths(bank),
        "near_isomorphic_clusters": near_isomorphic,
        "similarity_warnings": similarity_warnings,
        "separation_report": {
            "answer_pairs_checked": separation.answer_pairs_checked,
            "visible_candidate_comparisons_checked": (
                separation.visible_candidate_comparisons_checked
            ),
            "literal_visible_pairs_checked": separation.literal_visible_pairs_checked,
            "errors": list(separation.errors),
        },
        "families": families,
        "pedagogy_packs": pedagogy_packs,
    }
    packet["packet_digest"] = canonical_digest(packet)
    return packet


def main(argv: list[str] | None = None) -> int:
    """Validate or write the exact pending Product/Quotient review packet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assessment-source",
        type=Path,
        default=DEFAULT_ASSESSMENT_SOURCE_PATH,
    )
    parser.add_argument(
        "--assessment-reviews",
        type=Path,
        default=DEFAULT_ASSESSMENT_REVIEW_PATH,
    )
    parser.add_argument(
        "--pedagogy-source",
        type=Path,
        default=DEFAULT_PEDAGOGY_SOURCE_PATH,
    )
    parser.add_argument(
        "--pedagogy-reviews",
        type=Path,
        default=DEFAULT_PEDAGOGY_REVIEW_PATH,
    )
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH_PATH)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    if not args.check and args.out_dir is None:
        parser.error("nothing to do: pass --check and/or --out-dir PATH")

    try:
        graph = GraphDocument.model_validate_json(args.graph.read_text(encoding="utf-8"))
        packet = build_pending_product_quotient_packet(
            graph,
            load_source(args.assessment_source),
            load_manifest(args.assessment_reviews),
            load_pedagogy_source(args.pedagogy_source),
            load_pedagogy_reviews(args.pedagogy_reviews),
        )
        if args.out_dir is not None:
            write_reviewer_packet(args.out_dir, packet)
    except Exception as exc:  # noqa: BLE001 - offline CLI fail-closed boundary
        print(f"Product/Quotient review packet INVALID: {exc}", file=sys.stderr)
        return 1

    if args.check:
        print(
            "Product/Quotient review packet OK: "
            f"{len(packet['families'])} draft families, "
            f"{len(packet['pedagogy_packs'])} draft packs, "
            f"digest={packet['packet_digest']}"
        )
    if args.out_dir is not None:
        print(f"wrote private review packet to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
