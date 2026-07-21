"""Truth-oracle tests for deterministic FTC-wave task compilers."""

from __future__ import annotations

from tutor.content.ftc_release import (
    _TASK_COMPILER_REGISTRY,
    canonical_submission,
    derive_task,
)
from tutor.schemas.assessment import PlotPromptSegment
from tutor.schemas.ftc_authoring import (
    AntiderivativePolynomialSpec,
    AntiderivativeBinomialTask,
    AreaCompositeTask,
    CompositeRegionSpec,
    DefiniteAdditivityTask,
    FTCSuppliedTask,
    GraphPointValueTask,
    PiecewiseLinearSpec,
    RectangleRegion,
    RiemannLeftTask,
    EndpointTableSpec,
    TriangleRegion,
)
from tutor.verify.checker import VerificationStatus, verify_answer


def _assert_compiled_truth(task, independent_submission: str):
    derived = derive_task(task)
    assert canonical_submission(derived.answer)
    assert (
        verify_answer(
            derived.answer,
            independent_submission,
            supervised=False,
        ).status
        == VerificationStatus.CORRECT
    )
    for signature in derived.error_signatures:
        verdict = verify_answer(
            derived.answer,
            signature.expected_wrong,
            supervised=False,
        )
        assert verdict.status in {
            VerificationStatus.CORRECT,
            VerificationStatus.INCORRECT,
        }
    return derived


def test_closed_registry_covers_all_forty_typed_constructs():
    registrations = _TASK_COMPILER_REGISTRY.registrations
    assert len(registrations) == 40
    assert len({registration.kind for registration in registrations}) == 40
    assert len({registration.task_type for registration in registrations}) == 40


def test_graph_compiler_preserves_exact_visual_and_tabular_representation():
    task = GraphPointValueTask(
        graph=PiecewiseLinearSpec.model_validate(
            {
                "points": [
                    {"x": -2, "y": 4},
                    {"x": 0, "y": 8},
                    {"x": 2, "y": 12},
                ]
            }
        ),
        point_index=1,
    )
    derived = _assert_compiled_truth(task, "(0, 8)")

    plot = next(segment for segment in derived.givens if isinstance(segment, PlotPromptSegment))
    assert plot.spoken_text
    assert plot.equivalent_table is not None
    assert plot.equivalent_table.rows == (("-2", "4"), ("0", "8"), ("2", "12"))


def test_area_and_riemann_compilers_use_independent_arithmetic_oracles():
    area = AreaCompositeTask(
        region=CompositeRegionSpec(
            rectangles=(RectangleRegion(width=10, height=10),),
            triangles=(TriangleRegion(base=10, height=10),),
        )
    )
    _assert_compiled_truth(area, "150")

    riemann = RiemannLeftTask(
        table=EndpointTableSpec(lower=0, width=2, values=(10, 12, 14, 16))
    )
    _assert_compiled_truth(riemann, "72")


def test_definite_antiderivative_and_ftc_compilers_have_distinct_contracts():
    definite = DefiniteAdditivityTask(
        lower=0,
        split=2,
        upper=5,
        left_value=101,
        right_value=203,
    )
    _assert_compiled_truth(definite, "304")

    polynomial = AntiderivativePolynomialSpec.model_validate(
        {
            "terms": [
                {"coefficient": 2, "exponent": 3},
                {"coefficient": -5, "exponent": 1},
            ]
        }
    )
    antiderivative = AntiderivativeBinomialTask(polynomial=polynomial)
    _assert_compiled_truth(antiderivative, "2*x^3 - 5*x + C")

    ftc = FTCSuppliedTask(polynomial=polynomial, lower=1, upper=3)
    _assert_compiled_truth(ftc, "42")
