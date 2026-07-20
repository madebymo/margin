"""Deterministic assessment-family authoring and review-manifest tests."""

from copy import deepcopy

import pytest

from tutor.content.compiler import (
    COMPILER_VERSION,
    CompilationError,
    blueprint_digest,
    compile_blueprints,
    load_blueprints,
    load_review_manifest,
    main,
)
from tutor.schemas.assessment import (
    AssessmentTaskKind,
    PromptSemanticRole,
    SymbolicAnswerSpec,
)
from tutor.schemas.common import ReviewStatus
from tutor.schemas.content_authoring import (
    ContentReviewManifest,
    ItemBlueprintDocument,
)
from tutor.schemas.kc import GraphDocument
from tutor.seed.load_seed import load_graph
from tutor.verify.checker import VerificationStatus, verify_answer


@pytest.fixture(scope="module")
def source() -> ItemBlueprintDocument:
    return load_blueprints()


@pytest.fixture(scope="module")
def manifest() -> ContentReviewManifest:
    return load_review_manifest()


@pytest.fixture(scope="module")
def graph() -> GraphDocument:
    return load_graph()


def test_packaged_exponent_prototypes_compile_deterministically(source, manifest, graph):
    first = compile_blueprints(source, manifest, graph)
    second = compile_blueprints(source, manifest, graph)

    assert first.model_dump_json() == second.model_dump_json()
    assert first.released_kcs == []
    assert len(first.items) == 6
    assert {item.family_id for item in first.items} == {
        "family.exponent.product_same_base",
        "family.exponent.quotient_same_base",
        "family.exponent.power_of_power",
    }
    assert {item.review_status for item in first.items} == {ReviewStatus.DRAFT}
    assert all(item.task_kind == AssessmentTaskKind.TRANSFORM for item in first.items)
    assert all(item.allocation_order in {10, 20, 30} for item in first.items)

    for item in first.items:
        assert isinstance(item.answer, SymbolicAnswerSpec)
        assert verify_answer(
            item.answer,
            item.answer.expected,
            supervised=False,
        ).status == VerificationStatus.CORRECT
        assert [segment.role for segment in item.prompt] == [
            PromptSemanticRole.INSTRUCTION,
            PromptSemanticRole.GIVEN,
            PromptSemanticRole.RESPONSE,
        ]


def test_parameter_variants_share_family_and_authored_order(source, manifest, graph):
    bank = compile_blueprints(source, manifest, graph)

    for blueprint in source.family_blueprints:
        variants = [item for item in bank.items if item.family_id == blueprint.family_id]
        assert len(variants) == len(blueprint.cases) == 2
        assert {item.allocation_order for item in variants} == {
            blueprint.allocation_order
        }
        assert all(item.item_id.startswith("item.exponent.") for item in variants)


def test_manifest_digest_and_compiler_pin_fail_closed(source, manifest, graph):
    source_payload = source.model_dump(mode="json")
    source_payload["family_blueprints"][0]["cases"][0]["left_exponent"] = 4
    changed_source = ItemBlueprintDocument.model_validate(source_payload)

    with pytest.raises(CompilationError, match="review digest mismatch"):
        compile_blueprints(changed_source, manifest, graph)

    changed_manifest = manifest.model_copy(
        update={"compiler_version": "content-compiler-v999"}
    )
    with pytest.raises(CompilationError, match="unsupported compiler version"):
        compile_blueprints(source, changed_manifest, graph)


def test_approval_is_derived_only_from_exact_review_manifest(source, manifest, graph):
    payload = manifest.model_dump(mode="json")
    payload["entries"][0].update(
        {
            "decision": "approved",
            "reviewed_by": "Test mathematics reviewer",
            "reviewed_at": "2026-07-20T12:00:00Z",
        }
    )
    approved_manifest = ContentReviewManifest.model_validate(payload)

    bank = compile_blueprints(source, approved_manifest, graph)
    approved_family = source.family_blueprints[0].family_id
    approved_items = [item for item in bank.items if item.family_id == approved_family]

    assert {item.review_status for item in approved_items} == {
        ReviewStatus.HUMAN_APPROVED
    }
    assert all(
        item.provenance.reviewed_by == "Test mathematics reviewer"
        for item in approved_items
    )
    assert all(
        item.review_status == ReviewStatus.DRAFT
        for item in bank.items
        if item.family_id != approved_family
    )


def test_blueprint_author_cannot_approve_their_own_family(source, manifest, graph):
    payload = manifest.model_dump(mode="json")
    payload["entries"][0].update(
        {
            "decision": "approved",
            "reviewed_by": source.family_blueprints[0].author,
            "reviewed_at": "2026-07-20T12:00:00Z",
        }
    )
    self_reviewed = ContentReviewManifest.model_validate(payload)

    with pytest.raises(CompilationError, match="cannot be approved by its author"):
        compile_blueprints(source, self_reviewed, graph)


def test_authoring_contracts_reject_unknown_fields(source):
    payload = source.model_dump(mode="json")
    payload["family_blueprints"][0]["unreviewed_escape_hatch"] = True

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        ItemBlueprintDocument.model_validate(payload)


def test_compiler_never_reads_graph_canonical_examples(source, manifest, graph):
    baseline = compile_blueprints(source, manifest, graph)
    graph_payload = deepcopy(graph.model_dump(mode="json"))
    exponent_node = next(
        node for node in graph_payload["nodes"] if node["id"] == "kc.alg.exponent_rules"
    )
    exponent_node["canonical_examples"] = ["deliberately unrelated seed prose"]
    changed_graph = GraphDocument.model_validate(graph_payload)

    assert (
        compile_blueprints(source, manifest, changed_graph).model_dump_json()
        == baseline.model_dump_json()
    )


def test_packaged_manifest_tracks_exact_source_digests(source, manifest):
    reviews = {
        (entry.blueprint_id, entry.revision): entry
        for entry in manifest.entries
    }

    assert manifest.compiler_version == COMPILER_VERSION
    for blueprint in source.family_blueprints:
        review = reviews[(blueprint.blueprint_id, blueprint.revision)]
        assert review.source_digest == blueprint_digest(blueprint)
        assert review.decision.value == "pending"
        assert review.reviewed_by is None
        assert review.reviewed_at is None


def test_compiler_check_cli_accepts_packaged_pending_prototypes(capsys):
    assert main(["--check"]) == 0
    assert "3 families, 6 items, released KCs=0" in capsys.readouterr().out
