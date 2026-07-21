"""Contract tests for the typed U-substitution source."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from tutor.schemas.u_sub_authoring import (
    InnerPolynomialSpec,
    USubBlueprintDocument,
    USubMathTask,
    USubstitutionSpec,
)


TASK_ADAPTER = TypeAdapter(USubMathTask)


def test_source_document_forbids_authored_expected_answer_fields():
    payload = {
        "schema_version": 1,
        "blueprint_version": "u-sub-wave-v2.1.0",
        "output_bank_version": "draft-u-sub-wave-v2.1.0",
        "graph_version": 2,
        "authoring_source": "assessment-draft/u-sub-wave-v2.1",
        "author": "AI-assisted implementation draft (unreviewed)",
        "target_kcs": ["kc.der.differentials"],
        "released_kcs": [],
        "families": [
            {
                "blueprint_id": "blueprint.usv2.differentials.diagnostic.01",
                "item_id": "item.usv2.differentials.diagnostic.01",
                "family_id": "family.usv2.differentials.diagnostic.01",
                "kc_id": "kc.der.differentials",
                "construct_id": "differential.affine",
                "surface": "diagnostic",
                "allocation_order": 10,
                "task": {
                    "kind": "differential_affine",
                    "inner": {
                        "terms": [
                            {"coefficient": 7, "exponent": 1},
                            {"coefficient": 3, "exponent": 0},
                        ]
                    },
                    "expected_answer": "fabricated",
                },
            }
        ],
    }

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        USubBlueprintDocument.model_validate(payload)


def test_inner_polynomial_must_be_ordered_unique_and_nonconstant():
    with pytest.raises(ValidationError, match="unique and descending"):
        InnerPolynomialSpec.model_validate(
            {
                "terms": [
                    {"coefficient": 3, "exponent": 1},
                    {"coefficient": 2, "exponent": 2},
                ]
            }
        )
    with pytest.raises(ValidationError, match="nonconstant"):
        InnerPolynomialSpec.model_validate(
            {"terms": [{"coefficient": 5, "exponent": 0}]}
        )


def test_substitution_source_derives_full_differential_scale():
    substitution = USubstitutionSpec.model_validate(
        {
            "inner": {
                "terms": [
                    {"coefficient": 4, "exponent": 2},
                    {"coefficient": -3, "exponent": 0},
                ]
            },
            "outer_power": 4,
            "result_coefficient": -7,
        }
    )
    assert substitution.result_coefficient * (substitution.outer_power + 1) == -35

    parsed = TASK_ADAPTER.validate_python(
        {
            "kind": "u_sub_quadratic",
            "substitution": substitution.model_dump(mode="json"),
        }
    )
    assert parsed.kind == "u_sub_quadratic"
