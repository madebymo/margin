"""Private, deterministic review packets for the pending vertical slice."""

from __future__ import annotations

import json

import pytest

from tutor.content.product_quotient_release import (
    DEFAULT_MANIFEST_PATH as ASSESSMENT_REVIEW_PATH,
    DEFAULT_SOURCE_PATH as ASSESSMENT_SOURCE_PATH,
    load_manifest,
    load_source,
)
from tutor.content.review_artifacts import canonical_digest, canonical_json_bytes
from tutor.content.product_quotient_reviewer_packet import (
    build_pending_product_quotient_packet,
    main,
)
from tutor.content.reviewer_packet import ReviewerPacketError, render_reviewer_html
from tutor.packs.review_compiler import (
    DEFAULT_MANIFEST_PATH as PEDAGOGY_REVIEW_PATH,
    DEFAULT_SOURCE_PATH as PEDAGOGY_SOURCE_PATH,
    load_review_manifest as load_pedagogy_reviews,
    load_source_document as load_pedagogy_source,
)
from tutor.schemas.content_authoring import ReviewDecision
from tutor.schemas.pedagogy_authoring import PedagogyReviewDecision
from tutor.seed.load_seed import load_graph


@pytest.fixture(scope="module")
def pending_inputs():
    return (
        load_graph(),
        load_source(),
        load_manifest(),
        load_pedagogy_source(),
        load_pedagogy_reviews(),
    )


@pytest.fixture(scope="module")
def packet(pending_inputs):
    return build_pending_product_quotient_packet(*pending_inputs)


def test_pending_packet_is_private_complete_and_truth_bearing(packet):
    assert packet["schema_version"] == 2
    assert packet["artifact_kind"] == "pending_product_quotient_review"
    assert packet["workflow_state"] == "pending_independent_human_review"
    assert packet["publication_eligible"] is False
    assert packet["released_kcs"] == []
    assert packet["review_workflow"]["creates_approval_records"] is False
    assert packet["review_workflow"]["creates_release_artifacts"] is False
    assert packet["review_workflow"]["assessment_decisions"] == {"pending": 52}
    assert packet["review_workflow"]["pedagogy_decisions"] == {"pending": 4}
    assert len(packet["families"]) == 52
    assert len(packet["pedagogy_packs"]) == 4
    assert len(packet["knowledge_components"]) == 4
    assert len(packet["first_two_paths"]) == 20
    assert packet["similarity_warnings"]
    assert packet["separation_report"] == {
        "answer_pairs_checked": 1326,
        "visible_candidate_comparisons_checked": 5601,
        "literal_visible_pairs_checked": 2652,
        "errors": [],
    }

    guided_kinds = set()
    for family in packet["families"]:
        assert family["review_status"] == "draft"
        assert family["reviewed_by"] is None
        assert family["source_review"]["decision"] == "pending"
        assert family["source_review"]["reviewed_by"] is None
        assert family["source_digest"] == family["source_review"]["source_digest"]
        assert family["construct_id"] == family["source_blueprint"]["construct_id"]
        assert family["allocation_order"] == family["source_blueprint"]["allocation_order"]
        item = family["items"][0]
        assert item["answer_spec"]["expected"]
        assert item["rendering"]["production_text_fallback"] == item["prompt_text"]
        assert len(item["rendering"]["segments"]) == len(item["prompt_segments"])
        assert all(
            segment["exact_visual_text"] and segment["exact_spoken_text"]
            for segment in item["rendering"]["segments"]
        )
        assert [hint["revealing"] for hint in item["hints"]] == [False, False, True]
        assert item["revealing_hint_behavior"] == {
            "hint_index": 3,
            "marks_attempt_assisted": True,
            "retires_family": True,
            "requires_fresh_independent_item": True,
        }
        if item["guided_interaction"] is not None:
            guided_kinds.add(item["guided_interaction"]["kind"])
            assert item["guided_interaction"]["public_presentation"]
            assert item["guided_interaction"]["private_scoring"]
            assert item["guided_interaction"]["equivalent_text_fallback"]
    assert guided_kinds == {"mapping_v1", "slider_v1"}

    for pack in packet["pedagogy_packs"]:
        assert pack["review_status"] == "draft"
        assert pack["source_review"]["decision"] == "pending"
        assert pack["source_review"]["reviewed_by"] is None
        assert len(pack["citations"]) == 2
        assert len(pack["misconceptions"]) == 3
        assert len(pack["error_patterns"]) == 3
        assert pack["lesson_narrative_rendering"]["production_text_fallback"]
        assert pack["remediation_rendering"]["production_text_fallback"]


def test_pending_packet_bytes_are_deterministic(packet, pending_inputs):
    rebuilt = build_pending_product_quotient_packet(*pending_inputs)

    assert rebuilt == packet
    assert canonical_json_bytes(rebuilt, trailing_newline=True) == canonical_json_bytes(
        packet,
        trailing_newline=True,
    )
    assert "candidate_bundle_sha256" not in packet
    assert "release_id" not in packet
    unsigned_packet = dict(packet)
    packet_digest = unsigned_packet.pop("packet_digest")
    assert canonical_digest(unsigned_packet) == packet_digest


def test_pending_packet_rejects_release_or_completed_review_state(pending_inputs):
    graph, source, reviews, pedagogy_source, pedagogy_reviews = pending_inputs
    released_source = source.model_copy(update={"released_kcs": ["kc.der.product_quotient"]})
    with pytest.raises(ReviewerPacketError, match="released_kcs"):
        build_pending_product_quotient_packet(
            graph,
            released_source,
            reviews,
            pedagogy_source,
            pedagogy_reviews,
        )

    completed_entry = reviews.entries[0].model_copy(update={"decision": ReviewDecision.APPROVED})
    completed_reviews = reviews.model_copy(
        update={"entries": [completed_entry, *reviews.entries[1:]]}
    )
    with pytest.raises(ReviewerPacketError, match="completed assessment decisions"):
        build_pending_product_quotient_packet(
            graph,
            source,
            completed_reviews,
            pedagogy_source,
            pedagogy_reviews,
        )

    completed_pack_entry = pedagogy_reviews.entries[0].model_copy(
        update={"decision": PedagogyReviewDecision.APPROVED}
    )
    completed_pedagogy_reviews = pedagogy_reviews.model_copy(
        update={
            "entries": [completed_pack_entry, *pedagogy_reviews.entries[1:]],
        }
    )
    with pytest.raises(ReviewerPacketError, match="completed pedagogy decisions"):
        build_pending_product_quotient_packet(
            graph,
            source,
            reviews,
            pedagogy_source,
            completed_pedagogy_reviews,
        )


def test_pending_packet_cli_writes_private_artifact_without_mutating_inputs(
    tmp_path,
    capsys,
):
    source_paths = (
        ASSESSMENT_SOURCE_PATH,
        ASSESSMENT_REVIEW_PATH,
        PEDAGOGY_SOURCE_PATH,
        PEDAGOGY_REVIEW_PATH,
    )
    before = {path: path.read_bytes() for path in source_paths}
    destination = tmp_path / "product-quotient-review"

    assert main(["--check", "--out-dir", str(destination)]) == 0

    after = {path: path.read_bytes() for path in source_paths}
    assert after == before
    payload = json.loads((destination / "review-packet.json").read_text())
    html = (destination / "review-packet.html").read_text()
    assert payload["publication_eligible"] is False
    assert payload["released_kcs"] == []
    assert "PRIVATE OFFLINE REVIEW ARTIFACT: DRAFT AND UNRELEASED" in html
    assert "Exact visual and spoken rendering" in html
    assert "Expected answer contract" in html
    assert "Similarity warnings requiring human judgment" in html
    assert "52 draft families" in capsys.readouterr().out


def test_html_escapes_truth_bearing_content(packet):
    altered = dict(packet)
    altered["warning"] = "<script>alert('private')</script>"

    rendered = render_reviewer_html(altered)

    assert "<script>alert" not in rendered
    assert "&lt;script&gt;alert" in rendered
