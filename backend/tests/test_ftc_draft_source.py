"""Qualification of the explicit, unreviewed FTC assessment source."""

from __future__ import annotations

import json
from collections import Counter

from tutor.content.ftc_draft_catalog import AUTHOR, build_draft_source
from tutor.content.ftc_release import (
    DEFAULT_SOURCE_PATH,
    EXPECTED_CONSTRUCT_ORDER,
    EXPECTED_FAMILY_COUNTS,
    TARGET_KCS,
    _TASK_COMPILER_REGISTRY,
)
from tutor.schemas.assessment import AssessmentSurface
from tutor.schemas.ftc_authoring import FTCBlueprintDocument


def test_packaged_source_is_exact_typed_catalog_output():
    packaged = FTCBlueprintDocument.model_validate_json(
        DEFAULT_SOURCE_PATH.read_text(encoding="utf-8")
    )
    authored = build_draft_source()

    assert packaged == authored
    assert DEFAULT_SOURCE_PATH.read_text(encoding="utf-8") == (
        authored.model_dump_json(indent=2) + "\n"
    )
    assert packaged.author == AUTHOR
    assert packaged.released_kcs == []


def test_source_has_exact_six_kc_family_matrix_and_stable_identities():
    source = build_draft_source()

    assert set(source.target_kcs) == set(TARGET_KCS)
    assert len(source.families) == 78
    assert Counter((family.kc_id, family.surface) for family in source.families) == Counter(
        {
            (kc_id, surface): count
            for kc_id in TARGET_KCS
            for surface, count in EXPECTED_FAMILY_COUNTS.items()
        }
    )
    assert len({family.family_id for family in source.families}) == 78
    assert len({family.item_id for family in source.families}) == 78
    assert len({(family.blueprint_id, family.revision) for family in source.families}) == 78


def test_construct_taxonomy_is_exact_ordered_and_confirmation_distinct():
    source = build_draft_source()
    for family in source.families:
        _TASK_COMPILER_REGISTRY.validate_taxonomy(
            family.task,
            construct_id=family.construct_id,
            kc_id=family.kc_id,
        )
    for kc_id, by_surface in EXPECTED_CONSTRUCT_ORDER.items():
        for surface, expected in by_surface.items():
            families = sorted(
                (
                    family
                    for family in source.families
                    if family.kc_id == kc_id and family.surface == surface
                ),
                key=lambda family: family.allocation_order,
            )
            assert tuple(family.construct_id for family in families) == expected
            assert tuple(family.allocation_order for family in families) == tuple(
                range(10, 10 * (len(expected) + 1), 10)
            )
            if surface in {AssessmentSurface.DIAGNOSTIC, AssessmentSurface.CHECKIN}:
                assert len(expected) == len(set(expected))


def test_source_contains_no_expected_answers_scoring_or_review_claims():
    raw = json.loads(DEFAULT_SOURCE_PATH.read_text(encoding="utf-8"))
    serialized = json.dumps(raw)

    assert '"expected"' not in serialized
    assert "expected_answer" not in serialized
    assert "expected_wrong" not in serialized
    assert "correct_pairs" not in serialized
    assert "reviewed_by" not in serialized
    assert "human_approved" not in serialized
    assert raw["released_kcs"] == []
