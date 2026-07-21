"""Shared builder for truth-bearing, draft-only human reviewer packets.

Wave-specific wrappers supply a typed source loader, deterministic compiler,
and family digest function. This module owns the review-state boundary and the
packet shape so every content wave is reviewed through the same serializer.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import Any, Protocol

from pydantic import BaseModel

from tutor.content.review_artifacts import canonical_digest
from tutor.content.reviewer_packet import (
    ReviewerPacketError,
    _allocation_paths,
    _family_entries,
    _review_rendering,
    _similarity_findings,
)
from tutor.packs.review_compiler import source_digest as pedagogy_source_digest
from tutor.packs.review_compiler import validate_review_bundle
from tutor.schemas.assessment import ItemBankDocument
from tutor.schemas.common import EdgeType, ReviewStatus
from tutor.schemas.content_authoring import ContentReviewManifest, ReviewDecision
from tutor.schemas.kc import GraphDocument
from tutor.schemas.pedagogy_authoring import (
    PedagogyReviewDecision,
    PedagogyReviewManifest,
    PedagogySourceDocument,
)


class SeparationReport(Protocol):
    answer_pairs_checked: int
    visible_candidate_comparisons_checked: int
    literal_visible_pairs_checked: int
    errors: tuple[str, ...]


CompileInventory = Callable[
    [Any, ContentReviewManifest, GraphDocument],
    tuple[ItemBankDocument, SeparationReport],
]
FamilyDigest = Callable[[Any, Any], str]


def _require_pending_review_state(
    assessment_source: Any,
    assessment_reviews: ContentReviewManifest,
    pedagogy_reviews: PedagogyReviewManifest,
) -> None:
    """Keep packet generation strictly on the non-promoting review side."""

    if assessment_source.released_kcs:
        raise ReviewerPacketError(
            "pending reviewer packets require released_kcs to remain empty"
        )
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
    missing = kc_ids - set(nodes)
    if missing:
        raise ReviewerPacketError(
            f"assessment source names unknown KCs: {sorted(missing)}"
        )
    return [
        {
            "kc_id": kc_id,
            "name": nodes[kc_id].name,
            "description": nodes[kc_id].description,
            "course_level": nodes[kc_id].course_level,
            "hard_prerequisite_ids": sorted(
                edge.from_kc
                for edge in graph.edges
                if edge.to_kc == kc_id
                and edge.type == EdgeType.HARD
                and edge.from_kc in kc_ids
            ),
            "canonical_examples": list(nodes[kc_id].canonical_examples),
            "canonical_examples_role": "explanatory_seed_only_never_scored",
        }
        for kc_id in sorted(kc_ids)
    ]


def _source_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, BaseModel):
        raise ReviewerPacketError("assessment sources must be validated models")
    return value.model_dump(mode="json")


def build_pending_reviewer_packet(
    graph: GraphDocument,
    assessment_source: Any,
    assessment_reviews: ContentReviewManifest,
    pedagogy_source: PedagogySourceDocument,
    pedagogy_reviews: PedagogyReviewManifest,
    *,
    artifact_kind: str,
    compile_inventory: CompileInventory,
    family_digest: FamilyDigest,
) -> dict[str, object]:
    """Compile exact pending inputs into one deterministic private packet."""

    if not artifact_kind.startswith("pending_") or not artifact_kind.endswith("_review"):
        raise ReviewerPacketError(
            "draft packet artifact_kind must be pending_<wave>_review"
        )
    _require_pending_review_state(
        assessment_source,
        assessment_reviews,
        pedagogy_reviews,
    )
    validate_review_bundle(pedagogy_source, pedagogy_reviews, graph)
    bank, separation = compile_inventory(
        assessment_source,
        assessment_reviews,
        graph,
    )
    if bank.released_kcs:
        raise ReviewerPacketError("draft compilation unexpectedly released content")
    if any(item.review_status != ReviewStatus.DRAFT for item in bank.items):
        raise ReviewerPacketError("draft compilation unexpectedly approved an item")
    if separation.errors:
        raise ReviewerPacketError("draft compilation has unresolved separation errors")

    assessment_kcs = set(assessment_source.target_kcs)
    pedagogy_kcs = {source.kc_id for source in pedagogy_source.pack_sources}
    if pedagogy_kcs != assessment_kcs:
        raise ReviewerPacketError(
            "assessment and pedagogy review inputs cover different KCs"
        )

    blueprints_by_family = {
        family.family_id: family for family in assessment_source.families
    }
    if len(blueprints_by_family) != len(assessment_source.families):
        raise ReviewerPacketError("assessment family ids must be unique")
    construct_ids = {
        family.family_id: family.construct_id
        for family in assessment_source.families
    }
    assessment_review_map = {
        (entry.blueprint_id, entry.revision): entry
        for entry in assessment_reviews.entries
    }
    families, shapes = _family_entries(bank, construct_ids)
    for family_entry in families:
        family_id = str(family_entry["family_id"])
        try:
            blueprint = blueprints_by_family[family_id]
            review = assessment_review_map[
                (blueprint.blueprint_id, blueprint.revision)
            ]
        except KeyError as exc:
            raise ReviewerPacketError(
                f"compiled family {family_id!r} lacks exact source/review binding"
            ) from exc
        exact_source_digest = family_digest(assessment_source, blueprint)
        if review.source_digest != exact_source_digest:
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

    near_isomorphic, similarity_warnings = _similarity_findings(
        families,
        shapes,
    )
    pedagogy_review_map = {
        (entry.source_id, entry.revision): entry
        for entry in pedagogy_reviews.entries
    }
    pedagogy_packs: list[dict[str, object]] = []
    for source in sorted(
        pedagogy_source.pack_sources,
        key=lambda item: item.kc_id,
    ):
        try:
            review = pedagogy_review_map[(source.source_id, source.revision)]
        except KeyError as exc:
            raise ReviewerPacketError(
                f"pedagogy source {source.source_id!r} lacks exact review binding"
            ) from exc
        exact_source_digest = pedagogy_source_digest(source)
        if review.source_digest != exact_source_digest:
            raise ReviewerPacketError(
                f"pedagogy review digest mismatch for {source.source_id}"
            )
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
                    segment.model_dump(mode="json")
                    for segment in source.lesson_narrative
                ],
                "lesson_narrative_rendering": _review_rendering(
                    source.lesson_narrative
                ),
                "remediation": [
                    segment.model_dump(mode="json") for segment in source.remediation
                ],
                "remediation_rendering": _review_rendering(source.remediation),
                "misconceptions": [
                    item.model_dump(mode="json") for item in source.misconceptions
                ],
                "metaphors": [
                    item.model_dump(mode="json") for item in source.metaphors
                ],
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
        "artifact_kind": artifact_kind,
        "warning": (
            "PRIVATE OFFLINE REVIEW ARTIFACT: DRAFT AND UNRELEASED. Contains "
            "expected answers and private scoring data; never serve to learners."
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
                "Independent humans review the exact bound source and compiled "
                "bytes, then record decisions in the separate manifests. Packet "
                "generation itself confers no approval."
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
                f"{len(similarity_warnings)} similarity findings require explicit "
                "human family-independence judgment"
            ),
        ],
        "knowledge_components": _knowledge_component_context(
            graph,
            assessment_kcs,
        ),
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
    # Validate that the source is a normal Pydantic document before hashing the
    # packet. This guards a future wrapper from passing a permissive mock whose
    # dynamic fields differ between access and serialization.
    _source_payload(assessment_source)
    packet["packet_digest"] = canonical_digest(packet)
    return packet
