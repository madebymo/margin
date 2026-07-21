"""Contract tests for the typed Fundamental Theorem wave source."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from tutor.schemas.ftc_authoring import (
    AntiderivativePolynomialSpec,
    FTCBlueprintDocument,
    FTCMathTask,
    GraphBehaviorTask,
    GraphGuidedMappingTask,
    PiecewiseLinearSpec,
)


TASK_ADAPTER = TypeAdapter(FTCMathTask)


def test_source_document_forbids_authored_truth_and_unknown_fields():
    payload = {
        "schema_version": 1,
        "blueprint_version": "ftc-wave-v2.1.0",
        "output_bank_version": "draft-ftc-wave-v2.1.0",
        "graph_version": 2,
        "authoring_source": "assessment-draft/ftc-wave-v2.1",
        "author": "AI-assisted implementation draft (unreviewed)",
        "target_kcs": ["kc.fun.graph_reading"],
        "released_kcs": [],
        "families": [
            {
                "blueprint_id": "blueprint.ftcv2.graph_reading.diagnostic.01",
                "revision": 1,
                "item_id": "item.ftcv2.graph_reading.diagnostic.01",
                "family_id": "family.ftcv2.graph_reading.diagnostic.01",
                "kc_id": "kc.fun.graph_reading",
                "construct_id": "graph.point_value",
                "surface": "diagnostic",
                "allocation_order": 10,
                "task": {
                    "kind": "graph_point_value",
                    "graph": {
                        "points": [
                            {"x": -1, "y": 2},
                            {"x": 0, "y": 4},
                            {"x": 2, "y": 8},
                        ]
                    },
                    "point_index": 2,
                    "expected_answer": "fabricated",
                },
            }
        ],
    }

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        FTCBlueprintDocument.model_validate(payload)


def test_piecewise_graph_rejects_hidden_fractional_slope():
    with pytest.raises(ValidationError, match="integer slope"):
        PiecewiseLinearSpec.model_validate(
            {
                "points": [
                    {"x": 0, "y": 0},
                    {"x": 2, "y": 1},
                    {"x": 4, "y": 4},
                ]
            }
        )


def test_behavior_and_guided_mapping_require_unambiguous_graph_data():
    graph = PiecewiseLinearSpec.model_validate(
        {
            "points": [
                {"x": -2, "y": 7},
                {"x": 0, "y": 3},
                {"x": 2, "y": 5},
                {"x": 4, "y": 9},
            ]
        }
    )
    behavior = GraphBehaviorTask(graph=graph, behavior="increasing")
    assert behavior.behavior == "increasing"
    mapping = GraphGuidedMappingTask(graph=graph, point_indices=(0, 1, 3))
    assert mapping.point_indices == (0, 1, 3)


def test_polynomial_source_requires_unique_descending_nonconstant_terms():
    with pytest.raises(ValidationError, match="unique and descending"):
        AntiderivativePolynomialSpec.model_validate(
            {
                "terms": [
                    {"coefficient": 2, "exponent": 2},
                    {"coefficient": 3, "exponent": 3},
                ]
            }
        )

    parsed = TASK_ADAPTER.validate_python(
        {
            "kind": "ftc_derive",
            "polynomial": {
                "terms": [
                    {"coefficient": 4, "exponent": 3},
                    {"coefficient": -7, "exponent": 1},
                ]
            },
            "lower": -1,
            "upper": 2,
        }
    )
    assert parsed.kind == "ftc_derive"
