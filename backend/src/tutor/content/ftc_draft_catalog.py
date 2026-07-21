"""Explicit, unreviewed source catalog for the six-KC FTC content wave.

This module is an authoring aid, not a publication boundary.  It contains only
typed mathematical parameters and stable family metadata.  Running it writes
the canonical source JSON; the release compiler derives all learner-visible
content and mathematical truth from these objects.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from tutor.schemas.assessment import AssessmentSurface
from tutor.schemas.ftc_authoring import (
    AntiderivativeBinomialTask,
    AntiderivativeCoefficientAuditTask,
    AntiderivativeCorrectionTask,
    AntiderivativeDerivativeCheckTask,
    AntiderivativeGuidedMappingTask,
    AntiderivativePolynomialSpec,
    AntiderivativeSingleTask,
    AntiderivativeTrinomialTask,
    AreaCompositeTask,
    AreaGuidedSliderTask,
    AreaMissingHeightTask,
    AreaOrderedPartsTask,
    AreaRectangleTask,
    AreaTriangleTask,
    CompositeRegionSpec,
    DefiniteAdditivityTask,
    DefiniteGuidedMappingTask,
    DefiniteInterpretationTask,
    DefiniteMissingPieceTask,
    DefiniteOrientationTask,
    DefiniteSignedRegionsTask,
    DefiniteTwoOrientationsTask,
    EndpointTableSpec,
    FTCBlueprintDocument,
    FTCCorrectionTask,
    FTCDeriveTask,
    FTCFamilyBlueprint,
    FTCGuidedSliderTask,
    FTCMathTask,
    FTCOrderedIntervalsTask,
    FTCReversedTask,
    FTCSplitTask,
    FTCSuppliedTask,
    GraphBehaviorTask,
    GraphGuidedMappingTask,
    GraphInterceptsTask,
    GraphOrderedReadTask,
    GraphPointValueTask,
    GraphSlopeTask,
    MidpointTableSpec,
    PiecewiseLinearSpec,
    RectangleRegion,
    RiemannCompareTask,
    RiemannContributionsTask,
    RiemannGuidedMappingTask,
    RiemannLeftTask,
    RiemannMidpointTask,
    RiemannMissingHeightTask,
    RiemannRightTask,
    TriangleRegion,
)

AUTHOR = "AI-assisted implementation draft (unreviewed)"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1] / "seed" / "item_blueprints_ftc_v2.json"
)


def _graph(*points: tuple[int, int]) -> PiecewiseLinearSpec:
    return PiecewiseLinearSpec.model_validate(
        {"points": [{"x": x, "y": y} for x, y in points]}
    )


def _regions(
    *,
    rectangles: tuple[tuple[int, int], ...] = (),
    triangles: tuple[tuple[int, int], ...] = (),
) -> CompositeRegionSpec:
    return CompositeRegionSpec(
        rectangles=tuple(
            RectangleRegion(width=width, height=height)
            for width, height in rectangles
        ),
        triangles=tuple(
            TriangleRegion(base=base, height=height)
            for base, height in triangles
        ),
    )


def _polynomial(*terms: tuple[int, int]) -> AntiderivativePolynomialSpec:
    return AntiderivativePolynomialSpec.model_validate(
        {
            "terms": [
                {"coefficient": coefficient, "exponent": exponent}
                for coefficient, exponent in terms
            ]
        }
    )


def _family_rows(
    kc_id: str,
    slug: str,
    rows: tuple[tuple[AssessmentSurface, str, FTCMathTask, str], ...],
) -> list[FTCFamilyBlueprint]:
    counters: defaultdict[AssessmentSurface, int] = defaultdict(int)
    families: list[FTCFamilyBlueprint] = []
    for surface, construct_id, task, difficulty in rows:
        counters[surface] += 1
        number = counters[surface]
        identity = f"{slug}.{surface.value}.{number:02d}"
        families.append(
            FTCFamilyBlueprint(
                blueprint_id=f"blueprint.ftcv2.{identity}",
                item_id=f"item.ftcv2.{identity}",
                family_id=f"family.ftcv2.{identity}",
                kc_id=kc_id,
                construct_id=construct_id,
                surface=surface,
                allocation_order=number * 10,
                difficulty=difficulty,
                task=task,
            )
        )
    return families


def _graph_reading_families() -> list[FTCFamilyBlueprint]:
    diagnostic = AssessmentSurface.DIAGNOSTIC
    checkin = AssessmentSurface.CHECKIN
    guided = AssessmentSurface.GUIDED_WIDGET
    capstone = AssessmentSurface.CAPSTONE
    worked = AssessmentSurface.WORKED_EXAMPLE
    return _family_rows(
        "kc.fun.graph_reading",
        "graph_reading",
        (
            (
                diagnostic,
                "graph.point_value",
                GraphPointValueTask(
                    graph=_graph((-6, 2), (-2, 6), (2, 10)),
                    point_index=2,
                ),
                "foundation",
            ),
            (
                diagnostic,
                "graph.intercepts",
                GraphInterceptsTask(graph=_graph((-6, 0), (0, 6), (4, 10))),
                "core",
            ),
            (
                diagnostic,
                "graph.slope",
                GraphSlopeTask(
                    graph=_graph((-5, 4), (-1, 12), (3, 8)),
                    segment_index=0,
                ),
                "core",
            ),
            (
                diagnostic,
                "graph.behavior",
                GraphBehaviorTask(
                    graph=_graph((-10, 14), (-4, 8), (0, 4), (4, 12)),
                    behavior="increasing",
                ),
                "core",
            ),
            (
                checkin,
                "graph.ordered_read",
                GraphOrderedReadTask(
                    graph=_graph((-7, 3), (-3, 11), (1, 15), (5, 7)),
                    point_indices=(0, 2, 3),
                ),
                "core",
            ),
            (
                checkin,
                "graph.point_value",
                GraphPointValueTask(
                    graph=_graph((-6, 16), (-2, 12), (2, 8)),
                    point_index=1,
                ),
                "foundation",
            ),
            (
                checkin,
                "graph.slope",
                GraphSlopeTask(
                    graph=_graph((-10, 14), (-6, 2), (-2, 6)),
                    segment_index=0,
                ),
                "core",
            ),
            (
                checkin,
                "graph.behavior",
                GraphBehaviorTask(
                    graph=_graph((-10, 2), (-6, 6), (-2, 10), (2, 6)),
                    behavior="decreasing",
                ),
                "core",
            ),
            (
                checkin,
                "graph.intercepts",
                GraphInterceptsTask(graph=_graph((-4, 0), (0, 12), (4, 16))),
                "core",
            ),
            (
                guided,
                "graph.guided_mapping",
                GraphGuidedMappingTask(
                    graph=_graph((-6, 4), (-2, 12), (2, 16), (6, 8)),
                    point_indices=(0, 1, 3),
                ),
                "foundation",
            ),
            (
                capstone,
                "graph.ordered_read",
                GraphOrderedReadTask(
                    graph=_graph((-10, 16), (-4, 4), (0, 8), (4, 12)),
                    point_indices=(0, 2, 3),
                ),
                "stretch",
            ),
            (
                capstone,
                "graph.behavior",
                GraphBehaviorTask(
                    graph=_graph((-6, 14), (-2, 10), (2, 6), (6, 10)),
                    behavior="decreasing",
                ),
                "stretch",
            ),
            (
                worked,
                "graph.slope",
                GraphSlopeTask(
                    graph=_graph((-10, 15), (-4, 3), (2, 9)),
                    segment_index=0,
                ),
                "foundation",
            ),
        ),
    )


def _area_families() -> list[FTCFamilyBlueprint]:
    diagnostic = AssessmentSurface.DIAGNOSTIC
    checkin = AssessmentSurface.CHECKIN
    guided = AssessmentSurface.GUIDED_WIDGET
    capstone = AssessmentSurface.CAPSTONE
    worked = AssessmentSurface.WORKED_EXAMPLE
    return _family_rows(
        "kc.int.area_under_curve",
        "area_under_curve",
        (
            (
                diagnostic,
                "area.rectangle",
                AreaRectangleTask(region=RectangleRegion(width=11, height=13)),
                "foundation",
            ),
            (
                diagnostic,
                "area.triangle",
                AreaTriangleTask(region=TriangleRegion(base=12, height=18)),
                "foundation",
            ),
            (
                diagnostic,
                "area.composite",
                AreaCompositeTask(
                    region=_regions(rectangles=((8, 16),), triangles=((10, 14),))
                ),
                "core",
            ),
            (
                diagnostic,
                "area.missing_height",
                AreaMissingHeightTask(
                    shape="triangle", width_or_base=12, height=16
                ),
                "core",
            ),
            (
                checkin,
                "area.ordered_parts",
                AreaOrderedPartsTask(
                    rectangle=RectangleRegion(width=8, height=14),
                    triangle=TriangleRegion(base=10, height=12),
                ),
                "core",
            ),
            (
                checkin,
                "area.rectangle",
                AreaRectangleTask(region=RectangleRegion(width=9, height=18)),
                "foundation",
            ),
            (
                checkin,
                "area.triangle",
                AreaTriangleTask(region=TriangleRegion(base=10, height=20)),
                "foundation",
            ),
            (
                checkin,
                "area.composite",
                AreaCompositeTask(
                    region=_regions(rectangles=((12, 15),), triangles=((8, 16),))
                ),
                "core",
            ),
            (
                checkin,
                "area.missing_height",
                AreaMissingHeightTask(
                    shape="rectangle", width_or_base=11, height=19
                ),
                "core",
            ),
            (
                guided,
                "area.guided_slider",
                AreaGuidedSliderTask(
                    region=_regions(rectangles=((11, 16),), triangles=((12, 12),))
                ),
                "core",
            ),
            (
                capstone,
                "area.composite",
                AreaCompositeTask(
                    region=_regions(rectangles=((11, 18),), triangles=((12, 16),))
                ),
                "stretch",
            ),
            (
                capstone,
                "area.ordered_parts",
                AreaOrderedPartsTask(
                    rectangle=RectangleRegion(width=12, height=14),
                    triangle=TriangleRegion(base=8, height=18),
                ),
                "stretch",
            ),
            (
                worked,
                "area.composite",
                AreaCompositeTask(
                    region=_regions(rectangles=((11, 16),), triangles=((10, 18),))
                ),
                "foundation",
            ),
        ),
    )


def _riemann_families() -> list[FTCFamilyBlueprint]:
    diagnostic = AssessmentSurface.DIAGNOSTIC
    checkin = AssessmentSurface.CHECKIN
    guided = AssessmentSurface.GUIDED_WIDGET
    capstone = AssessmentSurface.CAPSTONE
    worked = AssessmentSurface.WORKED_EXAMPLE
    return _family_rows(
        "kc.int.riemann_sums",
        "riemann_sums",
        (
            (
                diagnostic,
                "riemann.left",
                RiemannLeftTask(
                    table=EndpointTableSpec(
                        lower=0, width=4, values=(18, 20, 21, 24)
                    )
                ),
                "foundation",
            ),
            (
                diagnostic,
                "riemann.right",
                RiemannRightTask(
                    table=EndpointTableSpec(
                        lower=1, width=5, values=(12, 15, 18, 20)
                    )
                ),
                "foundation",
            ),
            (
                diagnostic,
                "riemann.midpoint",
                RiemannMidpointTask(
                    table=MidpointTableSpec(lower=0, width=4, values=(12, 18, 24))
                ),
                "core",
            ),
            (
                diagnostic,
                "riemann.compare",
                RiemannCompareTask(
                    table=EndpointTableSpec(
                        lower=2, width=3, values=(14, 16, 20, 24)
                    )
                ),
                "core",
            ),
            (
                checkin,
                "riemann.contributions",
                RiemannContributionsTask(width=4, heights=(12, 16, 20)),
                "core",
            ),
            (
                checkin,
                "riemann.missing_height",
                RiemannMissingHeightTask(
                    width=3,
                    known_heights=(12, 14, 16),
                    missing_height=20,
                ),
                "core",
            ),
            (
                checkin,
                "riemann.left",
                RiemannLeftTask(
                    table=EndpointTableSpec(
                        lower=-2, width=3, values=(20, 24, 26, 28)
                    )
                ),
                "foundation",
            ),
            (
                checkin,
                "riemann.right",
                RiemannRightTask(
                    table=EndpointTableSpec(
                        lower=0, width=4, values=(14, 18, 21, 26)
                    )
                ),
                "core",
            ),
            (
                checkin,
                "riemann.midpoint",
                RiemannMidpointTask(
                    table=MidpointTableSpec(lower=1, width=2, values=(24, 26, 28))
                ),
                "core",
            ),
            (
                guided,
                "riemann.guided_mapping",
                RiemannGuidedMappingTask(
                    table=EndpointTableSpec(
                        lower=0, width=5, values=(12, 16, 20, 24)
                    )
                ),
                "core",
            ),
            (
                capstone,
                "riemann.compare",
                RiemannCompareTask(
                    table=EndpointTableSpec(
                        lower=-3, width=5, values=(14, 18, 20, 24)
                    )
                ),
                "stretch",
            ),
            (
                capstone,
                "riemann.contributions",
                RiemannContributionsTask(width=3, heights=(16, 20, 24, 28)),
                "stretch",
            ),
            (
                worked,
                "riemann.left",
                RiemannLeftTask(
                    table=EndpointTableSpec(
                        lower=0, width=4, values=(15, 18, 20, 24)
                    )
                ),
                "foundation",
            ),
        ),
    )


def _definite_families() -> list[FTCFamilyBlueprint]:
    diagnostic = AssessmentSurface.DIAGNOSTIC
    checkin = AssessmentSurface.CHECKIN
    guided = AssessmentSurface.GUIDED_WIDGET
    capstone = AssessmentSurface.CAPSTONE
    worked = AssessmentSurface.WORKED_EXAMPLE
    return _family_rows(
        "kc.int.definite_integral",
        "definite_integral",
        (
            (
                diagnostic,
                "definite.orientation",
                DefiniteOrientationTask(lower=0, upper=4, forward_value=311),
                "foundation",
            ),
            (
                diagnostic,
                "definite.additivity",
                DefiniteAdditivityTask(
                    lower=1,
                    split=5,
                    upper=9,
                    left_value=312,
                    right_value=313,
                ),
                "core",
            ),
            (
                diagnostic,
                "definite.signed_regions",
                DefiniteSignedRegionsTask(signed_areas=(96, -42, 98)),
                "core",
            ),
            (
                diagnostic,
                "definite.interpretation",
                DefiniteInterpretationTask(lower=-2, upper=6, value=316),
                "foundation",
            ),
            (
                checkin,
                "definite.missing_piece",
                DefiniteMissingPieceTask(
                    lower=0,
                    split=4,
                    upper=8,
                    total_value=701,
                    left_value=317,
                ),
                "core",
            ),
            (
                checkin,
                "definite.two_orientations",
                DefiniteTwoOrientationsTask(lower=2, upper=7, forward_value=318),
                "foundation",
            ),
            (
                checkin,
                "definite.orientation",
                DefiniteOrientationTask(lower=-3, upper=5, forward_value=319),
                "core",
            ),
            (
                checkin,
                "definite.additivity",
                DefiniteAdditivityTask(
                    lower=0,
                    split=6,
                    upper=12,
                    left_value=320,
                    right_value=321,
                ),
                "core",
            ),
            (
                checkin,
                "definite.signed_regions",
                DefiniteSignedRegionsTask(signed_areas=(92, -46, 98)),
                "core",
            ),
            (
                guided,
                "definite.guided_mapping",
                DefiniteGuidedMappingTask(
                    lower=1,
                    split=6,
                    upper=11,
                    left_value=324,
                    right_value=325,
                ),
                "core",
            ),
            (
                capstone,
                "definite.missing_piece",
                DefiniteMissingPieceTask(
                    lower=-2,
                    split=3,
                    upper=8,
                    total_value=702,
                    left_value=326,
                ),
                "stretch",
            ),
            (
                capstone,
                "definite.two_orientations",
                DefiniteTwoOrientationsTask(lower=4, upper=10, forward_value=327),
                "stretch",
            ),
            (
                worked,
                "definite.additivity",
                DefiniteAdditivityTask(
                    lower=0,
                    split=5,
                    upper=10,
                    left_value=328,
                    right_value=329,
                ),
                "foundation",
            ),
        ),
    )


def _antiderivative_families() -> list[FTCFamilyBlueprint]:
    diagnostic = AssessmentSurface.DIAGNOSTIC
    checkin = AssessmentSurface.CHECKIN
    guided = AssessmentSurface.GUIDED_WIDGET
    capstone = AssessmentSurface.CAPSTONE
    worked = AssessmentSurface.WORKED_EXAMPLE
    return _family_rows(
        "kc.int.antiderivatives",
        "antiderivatives",
        (
            (
                diagnostic,
                "antiderivative.single",
                AntiderivativeSingleTask(polynomial=_polynomial((11, 6))),
                "foundation",
            ),
            (
                diagnostic,
                "antiderivative.binomial",
                AntiderivativeBinomialTask(polynomial=_polynomial((13, 5), (-7, 2))),
                "core",
            ),
            (
                diagnostic,
                "antiderivative.correction",
                AntiderivativeCorrectionTask(
                    polynomial=_polynomial((9, 4), (8, 3)),
                    mistake="kept_power",
                ),
                "core",
            ),
            (
                diagnostic,
                "antiderivative.derivative_check",
                AntiderivativeDerivativeCheckTask(
                    polynomial=_polynomial((12, 6), (-5, 3), (6, 1))
                ),
                "core",
            ),
            (
                checkin,
                "antiderivative.trinomial",
                AntiderivativeTrinomialTask(
                    polynomial=_polynomial((14, 5), (9, 3), (-8, 2))
                ),
                "core",
            ),
            (
                checkin,
                "antiderivative.coefficient_audit",
                AntiderivativeCoefficientAuditTask(
                    polynomial=_polynomial((10, 6), (-11, 4), (7, 1))
                ),
                "core",
            ),
            (
                checkin,
                "antiderivative.binomial",
                AntiderivativeBinomialTask(polynomial=_polynomial((-13, 6), (12, 2))),
                "core",
            ),
            (
                checkin,
                "antiderivative.correction",
                AntiderivativeCorrectionTask(
                    polynomial=_polynomial((15, 5), (-9, 2)),
                    mistake="did_not_divide",
                ),
                "core",
            ),
            (
                checkin,
                "antiderivative.derivative_check",
                AntiderivativeDerivativeCheckTask(
                    polynomial=_polynomial((-12, 5), (11, 3), (-6, 1))
                ),
                "stretch",
            ),
            (
                guided,
                "antiderivative.guided_mapping",
                AntiderivativeGuidedMappingTask(
                    polynomial=_polynomial((8, 6), (13, 4), (-10, 2))
                ),
                "core",
            ),
            (
                capstone,
                "antiderivative.trinomial",
                AntiderivativeTrinomialTask(
                    polynomial=_polynomial((15, 6), (-14, 3), (9, 1))
                ),
                "stretch",
            ),
            (
                capstone,
                "antiderivative.coefficient_audit",
                AntiderivativeCoefficientAuditTask(
                    polynomial=_polynomial((-14, 6), (10, 5), (-11, 2))
                ),
                "stretch",
            ),
            (
                worked,
                "antiderivative.binomial",
                AntiderivativeBinomialTask(polynomial=_polynomial((7, 5), (15, 3))),
                "foundation",
            ),
        ),
    )


def _ftc_families() -> list[FTCFamilyBlueprint]:
    diagnostic = AssessmentSurface.DIAGNOSTIC
    checkin = AssessmentSurface.CHECKIN
    guided = AssessmentSurface.GUIDED_WIDGET
    capstone = AssessmentSurface.CAPSTONE
    worked = AssessmentSurface.WORKED_EXAMPLE
    return _family_rows(
        "kc.int.ftc",
        "ftc",
        (
            (
                diagnostic,
                "ftc.supplied",
                FTCSuppliedTask(polynomial=_polynomial((9, 4), (7, 1)), lower=1, upper=3),
                "foundation",
            ),
            (
                diagnostic,
                "ftc.derive",
                FTCDeriveTask(polynomial=_polynomial((8, 5), (-6, 2)), lower=0, upper=3),
                "core",
            ),
            (
                diagnostic,
                "ftc.reversed",
                FTCReversedTask(polynomial=_polynomial((7, 4), (5, 2)), lower=1, upper=4),
                "core",
            ),
            (
                diagnostic,
                "ftc.correction",
                FTCCorrectionTask(
                    polynomial=_polynomial((11, 3), (-8, 1)),
                    lower=-1,
                    upper=3,
                    mistake="used_integrand",
                ),
                "core",
            ),
            (
                checkin,
                "ftc.split",
                FTCSplitTask(
                    polynomial=_polynomial((12, 3), (7, 2)),
                    lower=0,
                    split=2,
                    upper=4,
                ),
                "core",
            ),
            (
                checkin,
                "ftc.ordered_intervals",
                FTCOrderedIntervalsTask(
                    polynomial=_polynomial((6, 4), (13, 1)),
                    first_bounds=(0, 3),
                    second_bounds=(1, 4),
                ),
                "stretch",
            ),
            (
                checkin,
                "ftc.supplied",
                FTCSuppliedTask(polynomial=_polynomial((10, 5), (9, 1)), lower=1, upper=3),
                "core",
            ),
            (
                checkin,
                "ftc.derive",
                FTCDeriveTask(polynomial=_polynomial((-9, 4), (14, 2)), lower=-1, upper=3),
                "core",
            ),
            (
                checkin,
                "ftc.reversed",
                FTCReversedTask(polynomial=_polynomial((13, 3), (-7, 1)), lower=0, upper=4),
                "core",
            ),
            (
                guided,
                "ftc.guided_slider",
                FTCGuidedSliderTask(polynomial=_polynomial((8, 3), (5, 1)), lower=1, upper=4),
                "core",
            ),
            (
                capstone,
                "ftc.correction",
                FTCCorrectionTask(
                    polynomial=_polynomial((15, 4), (-10, 1)),
                    lower=-1,
                    upper=2,
                    mistake="added_endpoints",
                ),
                "stretch",
            ),
            (
                capstone,
                "ftc.ordered_intervals",
                FTCOrderedIntervalsTask(
                    polynomial=_polynomial((14, 3), (9, 1)),
                    first_bounds=(0, 2),
                    second_bounds=(2, 4),
                ),
                "stretch",
            ),
            (
                worked,
                "ftc.supplied",
                FTCSuppliedTask(polynomial=_polynomial((11, 4), (12, 2)), lower=0, upper=3),
                "foundation",
            ),
        ),
    )


def build_draft_source() -> FTCBlueprintDocument:
    """Return the exact 78-family, unreviewed source document."""

    families = [
        *_graph_reading_families(),
        *_area_families(),
        *_riemann_families(),
        *_definite_families(),
        *_antiderivative_families(),
        *_ftc_families(),
    ]
    return FTCBlueprintDocument(
        schema_version=1,
        blueprint_version="ftc-wave-v2.1.0",
        output_bank_version="draft-ftc-wave-v2.1.0",
        graph_version=2,
        authoring_source="assessment-draft/ftc-wave-v2.1",
        author=AUTHOR,
        target_kcs=[
            "kc.fun.graph_reading",
            "kc.int.area_under_curve",
            "kc.int.riemann_sums",
            "kc.int.definite_integral",
            "kc.int.antiderivatives",
            "kc.int.ftc",
        ],
        released_kcs=[],
        families=families,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    from tutor.content.ftc_release import _atomic_write_model

    source = build_draft_source()
    _atomic_write_model(args.out, source)
    print(f"wrote {len(source.families)} unreviewed FTC family sources to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
