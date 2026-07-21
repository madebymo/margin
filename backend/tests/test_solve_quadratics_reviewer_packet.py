"""Private deterministic review packets for the pending Solve Quadratics wave."""

from __future__ import annotations

import json

import pytest

from tutor.content.review_artifacts import canonical_digest, canonical_json_bytes
from tutor.content.solve_quadratics_release import load_manifest, load_source
from tutor.content.solve_quadratics_reviewer_packet import (
    DEFAULT_ASSESSMENT_REVIEW_PATH,
    DEFAULT_ASSESSMENT_SOURCE_PATH,
    DEFAULT_PEDAGOGY_REVIEW_PATH,
    DEFAULT_PEDAGOGY_SOURCE_PATH,
    build_pending_solve_quadratics_packet,
    main,
)
from tutor.packs.review_compiler import (
    load_review_manifest as load_pedagogy_reviews,
    load_source_document as load_pedagogy_source,
)
from tutor.seed.load_seed import load_graph


@pytest.fixture(scope="module")
def pending_inputs():
    return (
        load_graph(),
        load_source(),
        load_manifest(),
        load_pedagogy_source(DEFAULT_PEDAGOGY_SOURCE_PATH),
        load_pedagogy_reviews(DEFAULT_PEDAGOGY_REVIEW_PATH),
    )


@pytest.fixture(scope="module")
def packet(pending_inputs):
    return build_pending_solve_quadratics_packet(*pending_inputs)


def test_solve_packet_is_private_complete_and_truth_bearing(packet):
    assert packet["schema_version"] == 2
    assert packet["artifact_kind"] == "pending_solve_quadratics_review"
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
    assert all(
        "evidence_family_independence" in family["review_requirements"]
        for family in packet["families"]
    )
    assert packet["separation_report"] == {
        "answer_pairs_checked": 1326,
        "visible_candidate_comparisons_checked": 2651,
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
        item = family["items"][0]
        assert item["answer_spec"]["expected"]
        assert item["rendering"]["production_text_fallback"] == item["prompt_text"]
        assert [hint["revealing"] for hint in item["hints"]] == [False, False, True]
        if item["guided_interaction"] is not None:
            guided_kinds.add(item["guided_interaction"]["kind"])
            assert item["guided_interaction"]["public_presentation"]
            assert item["guided_interaction"]["private_scoring"]
            assert item["guided_interaction"]["equivalent_text_fallback"]
    assert guided_kinds == {"mapping_v1"}

    for pack in packet["pedagogy_packs"]:
        assert pack["review_status"] == "draft"
        assert pack["source_review"]["decision"] == "pending"
        assert pack["source_review"]["reviewed_by"] is None
        assert len(pack["citations"]) == 2
        assert len(pack["misconceptions"]) == 3
        assert len(pack["metaphors"]) == 1
        assert len(pack["error_patterns"]) == 3
        assert pack["lesson_narrative_rendering"]["production_text_fallback"]
        assert pack["remediation_rendering"]["production_text_fallback"]


def test_solve_packet_bytes_are_deterministic(packet, pending_inputs):
    rebuilt = build_pending_solve_quadratics_packet(*pending_inputs)

    assert rebuilt == packet
    assert canonical_json_bytes(
        rebuilt,
        trailing_newline=True,
    ) == canonical_json_bytes(packet, trailing_newline=True)
    assert "candidate_bundle_sha256" not in packet
    assert "release_id" not in packet
    unsigned_packet = dict(packet)
    packet_digest = unsigned_packet.pop("packet_digest")
    assert canonical_digest(unsigned_packet) == packet_digest


def test_solve_packet_cli_writes_without_mutating_review_inputs(tmp_path, capsys):
    source_paths = (
        DEFAULT_ASSESSMENT_SOURCE_PATH,
        DEFAULT_ASSESSMENT_REVIEW_PATH,
        DEFAULT_PEDAGOGY_SOURCE_PATH,
        DEFAULT_PEDAGOGY_REVIEW_PATH,
    )
    before = {path: path.read_bytes() for path in source_paths}
    destination = tmp_path / "solve-quadratics-review"

    assert main(["--check", "--out-dir", str(destination)]) == 0

    assert {path: path.read_bytes() for path in source_paths} == before
    payload = json.loads((destination / "review-packet.json").read_text())
    html = (destination / "review-packet.html").read_text()
    assert payload["publication_eligible"] is False
    assert payload["released_kcs"] == []
    assert "PRIVATE OFFLINE REVIEW ARTIFACT: DRAFT AND UNRELEASED" in html
    assert "Exact visual and spoken rendering" in html
    assert "Expected answer contract" in html
    assert "52 draft families" in capsys.readouterr().out
