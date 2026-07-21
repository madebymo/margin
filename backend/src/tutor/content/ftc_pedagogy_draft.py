"""Explicit, unreviewed pedagogy sources for the FTC content wave."""

from __future__ import annotations

import argparse
from pathlib import Path

from tutor.schemas.assessment import (
    MathPromptSegment,
    PromptSemanticRole,
    TextPromptSegment,
)
from tutor.schemas.common import WidgetType
from tutor.schemas.pedagogy import Metaphor, Misconception
from tutor.schemas.pedagogy_authoring import (
    PedagogyPackSource,
    PedagogySourceDocument,
)

AUTHOR = "AI-assisted implementation draft (unreviewed)"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "seed"
    / "pedagogy_pack_sources_ftc_v2.json"
)


def _text(text: str) -> TextPromptSegment:
    return TextPromptSegment(role=PromptSemanticRole.INSTRUCTION, text=text)


def _math(expression: str, spoken_text: str) -> MathPromptSegment:
    return MathPromptSegment(
        role=PromptSemanticRole.WORKED_STEP,
        expression=expression,
        spoken_text=spoken_text,
    )


def _misconception(
    identifier: str,
    description: str,
    error_signature: str,
    remediation_hint: str,
) -> Misconception:
    return Misconception(
        id=identifier,
        description=description,
        error_signature=error_signature,
        remediation_hint=remediation_hint,
    )


def _graph_pack() -> PedagogyPackSource:
    return PedagogyPackSource(
        source_id="pedagogy.source.graph_reading",
        revision=1,
        kc_id="kc.fun.graph_reading",
        author=AUTHOR,
        misconceptions=(
            _misconception(
                "m.graph_reading.axes_coordinates",
                "Reverses horizontal inputs and vertical outputs.",
                "Reports a plotted point with its coordinates in the opposite order.",
                "Name the horizontal input first and the vertical function value second.",
            ),
            _misconception(
                "m.graph_reading.intercept_confusion",
                "Uses the wrong zero-coordinate test for an intercept.",
                "Calls a point an x-intercept when its x-coordinate, rather than its height, is zero.",
                "For an x-intercept set the height to zero; for a y-intercept set the input to zero.",
            ),
            _misconception(
                "m.graph_reading.slope_direction",
                "Reads vertical change without keeping a consistent left-to-right direction.",
                "Rise and run are subtracted in opposite directions, producing the wrong sign.",
                "Choose left and right endpoints once, then subtract right minus left for both coordinates.",
            ),
        ),
        metaphors=(
            Metaphor(
                id="met.graph_reading.terrain_map",
                description=(
                    "A function graph is a terrain route: an input names a horizontal "
                    "location, the height names the output, and slope records the climb "
                    "per horizontal step."
                ),
                widget_affinity=[WidgetType.MAPPING, WidgetType.SLIDER],
            ),
        ),
        error_patterns=(
            "graph_coordinate_swap: reports output before input",
            "graph_intercept_axis_swap: tests the wrong coordinate for zero",
            "graph_slope_direction_mismatch: rise and run use opposite subtraction orders",
        ),
        sources=(
            "OpenStax Precalculus 2e, Section 1.1: Functions and Function Notation.",
            "OpenStax Calculus Volume 1, Section 1.1: Review of Functions.",
        ),
        lesson_narrative=(
            _text(
                "Read a graph from left to right: x is the input, f(x) is the height, and each exact point connects those two values."
            ),
            _math(
                "slope=(right y-left y)/(right x-left x)",
                "Slope is right y minus left y divided by right x minus left x.",
            ),
            _text(
                "Intercepts are zero-coordinate points, while increasing and decreasing intervals describe how height changes as x moves right."
            ),
        ),
        remediation=(
            _text(
                "Label the horizontal and vertical axes, then copy the exact endpoint coordinates before doing any arithmetic."
            ),
            _math(
                "point=(input,output)",
                "A plotted point is written as input first, output second.",
            ),
            _text(
                "For slope, draw the horizontal run and vertical rise separately and subtract both in the same direction."
            ),
        ),
    )


def _area_pack() -> PedagogyPackSource:
    return PedagogyPackSource(
        source_id="pedagogy.source.area_under_curve",
        revision=1,
        kc_id="kc.int.area_under_curve",
        author=AUTHOR,
        misconceptions=(
            _misconception(
                "m.area_under_curve.omits_region",
                "Adds only some of the nonoverlapping regions.",
                "The reported total equals a proper subset of the listed rectangle and triangle areas.",
                "Mark every region after its area has been included exactly once.",
            ),
            _misconception(
                "m.area_under_curve.triangle_factor",
                "Treats a triangle as a full rectangle.",
                "Uses base times height without the one-half factor.",
                "Pair the triangle with its matching rectangle and take half of that rectangle area.",
            ),
            _misconception(
                "m.area_under_curve.uses_endpoint_height",
                "Uses a listed dimension as the area or as an unrelated endpoint value.",
                "The answer copies a width or height without applying the shape formula.",
                "Name the shape first, then write its complete area formula before substituting dimensions.",
            ),
        ),
        metaphors=(
            Metaphor(
                id="met.area_under_curve.floor_tiles",
                description=(
                    "Accumulated area is a tiled floor: partition the region into "
                    "nonoverlapping familiar tiles, find each tile's area, and count "
                    "every tile once."
                ),
                widget_affinity=[WidgetType.SLIDER, WidgetType.MAPPING],
            ),
        ),
        error_patterns=(
            "area_region_omitted: one listed nonoverlapping shape is absent from the sum",
            "area_triangle_not_halved: base times height is used as the triangle area",
            "area_dimension_copied: a width or height is reported without applying an area formula",
        ),
        sources=(
            "OpenStax Calculus Volume 1, Section 5.1: Approximating Areas.",
            "OpenStax Precalculus 2e, Section 1.6: Absolute Value Functions.",
        ),
        lesson_narrative=(
            _text(
                "For a nonnegative piecewise-linear region, partition the space under the graph into rectangles and triangles."
            ),
            _math(
                "A_rectangle=w*h; A_triangle=b*h/2",
                "Rectangle area is width times height; triangle area is base times height divided by two.",
            ),
            _text(
                "Because the pieces do not overlap and stay above the axis, the total area is the ordinary sum of their areas."
            ),
        ),
        remediation=(
            _text(
                "Sketch a boundary around each rectangle or triangle and write one formula beside each piece."
            ),
            _math(
                "total area=sum of nonoverlapping piece areas",
                "Total area is the sum of all nonoverlapping piece areas.",
            ),
            _text(
                "Check off each piece after adding it and verify that every triangle contribution includes one-half."
            ),
        ),
    )


def _riemann_pack() -> PedagogyPackSource:
    return PedagogyPackSource(
        source_id="pedagogy.source.riemann_sums",
        revision=1,
        kc_id="kc.int.riemann_sums",
        author=AUTHOR,
        misconceptions=(
            _misconception(
                "m.riemann_sums.endpoint_choice",
                "Chooses heights from the wrong side of each subinterval.",
                "A left sum uses right endpoints, or a right sum uses left endpoints.",
                "Write each subinterval first and circle the endpoint named by the method.",
            ),
            _misconception(
                "m.riemann_sums.omits_width",
                "Adds heights but does not turn them into rectangle areas.",
                "The reported value is the sum of heights without the common-width factor.",
                "Write width times height for every rectangle before adding.",
            ),
            _misconception(
                "m.riemann_sums.midpoint_confusion",
                "Treats a midpoint sample as an endpoint sample.",
                "The selected x-values sit at subinterval edges rather than their centers.",
                "Mark both endpoints of each subinterval, then locate the point halfway between them."
            ),
        ),
        metaphors=(
            Metaphor(
                id="met.riemann_sums.fence_panels",
                description=(
                    "A Riemann sum is a row of equal-width fence panels: the method "
                    "chooses each panel's height, and width times height gives its contribution."
                ),
                widget_affinity=[WidgetType.MAPPING, WidgetType.SLIDER],
            ),
        ),
        error_patterns=(
            "riemann_wrong_endpoint: samples the opposite edge of each subinterval",
            "riemann_width_omitted: adds heights without multiplying by common width",
            "riemann_midpoint_shift: uses an edge in place of the subinterval center",
        ),
        sources=(
            "OpenStax Calculus Volume 1, Section 5.1: Approximating Areas.",
            "OpenStax Calculus Volume 1, Section 5.2: The Definite Integral.",
        ),
        lesson_narrative=(
            _text(
                "A Riemann sum partitions an interval into equal widths and assigns one sampled height to each rectangle."
            ),
            _math(
                "rectangle contribution=common width*selected height",
                "Each rectangle contribution is the common width times its selected height.",
            ),
            _text(
                "Left, right, and midpoint methods differ only in which sample supplies each height."
            ),
        ),
        remediation=(
            _text(
                "List the subintervals in order and write L, R, or M beside the selected point for each one."
            ),
            _math(
                "sum=width*(height one+height two+more heights)",
                "The rectangle sum is width times the sum of all selected heights.",
            ),
            _text(
                "Count selected heights and subintervals; the counts must match before you multiply."
            ),
        ),
    )


def _definite_pack() -> PedagogyPackSource:
    return PedagogyPackSource(
        source_id="pedagogy.source.definite_integral",
        revision=1,
        kc_id="kc.int.definite_integral",
        author=AUTHOR,
        misconceptions=(
            _misconception(
                "m.definite_integral.bound_order",
                "Treats reversed bounds as the same direction.",
                "The value keeps its sign when the starting and ending bounds are exchanged.",
                "Read the lower written bound as the start and negate the value when direction reverses.",
            ),
            _misconception(
                "m.definite_integral.ignores_sign",
                "Counts every geometric region as positive.",
                "Areas below the horizontal axis are added by magnitude instead of contributing negatively.",
                "Attach a positive sign above the axis and a negative sign below it before adding."
            ),
            _misconception(
                "m.definite_integral.breaks_additivity",
                "Combines adjacent intervals with subtraction or drops one piece.",
                "The whole-interval value is not the sum of the two consistently oriented pieces.",
                "Write the two adjacent intervals end to end and add their signed values."
            ),
        ),
        metaphors=(
            Metaphor(
                id="met.definite_integral.trip_ledger",
                description=(
                    "A definite integral is a directed trip ledger: contributions in "
                    "the chosen direction carry signs, adjacent trip segments add, and "
                    "reversing the trip reverses the total."
                ),
                widget_affinity=[WidgetType.MAPPING, WidgetType.SLIDER],
            ),
        ),
        error_patterns=(
            "definite_bounds_reversed_without_sign: direction changes but sign does not",
            "definite_unsigned_area: negative contributions are replaced by magnitudes",
            "definite_additivity_break: adjacent signed values are not added",
        ),
        sources=(
            "OpenStax Calculus Volume 1, Section 5.2: The Definite Integral.",
            "OpenStax Calculus Volume 1, Section 5.3: The Fundamental Theorem of Calculus.",
        ),
        lesson_narrative=(
            _text(
                "A definite integral is signed accumulation from its starting bound to its ending bound."
            ),
            _math(
                "integral_[b,a] f(x) dx=-integral_[a,b] f(x) dx",
                "The integral from b to a equals the negative of the integral from a to b.",
            ),
            _text(
                "Adjacent intervals add when their directions agree; contributions below the axis remain negative."
            ),
        ),
        remediation=(
            _text(
                "Write an arrow from the starting bound to the ending bound and keep that direction through every piece."
            ),
            _math(
                "whole signed accumulation=left piece+right piece",
                "The whole signed accumulation equals the left piece plus the right piece.",
            ),
            _text(
                "Before adding, label each region or supplied value with its sign and verify that adjacent endpoints meet."
            ),
        ),
    )


def _antiderivative_pack() -> PedagogyPackSource:
    return PedagogyPackSource(
        source_id="pedagogy.source.antiderivatives",
        revision=1,
        kc_id="kc.int.antiderivatives",
        author=AUTHOR,
        misconceptions=(
            _misconception(
                "m.antiderivatives.keeps_exponent",
                "Changes a coefficient but leaves the power unchanged.",
                "Differentiating the proposed term produces a power one lower than the required term.",
                "Raise the exponent by one before adjusting the coefficient."
            ),
            _misconception(
                "m.antiderivatives.multiplies_coefficient",
                "Multiplies by the new exponent instead of dividing by it.",
                "The derivative of the proposed term has an extra factor of the new exponent.",
                "After raising the power, divide the old coefficient by that new power."
            ),
            _misconception(
                "m.antiderivatives.drops_term",
                "Antidifferentiates only part of a polynomial.",
                "Differentiating the proposed result fails to reproduce one given term.",
                "Process every polynomial term separately, then combine the results."
            ),
        ),
        metaphors=(
            Metaphor(
                id="met.antiderivatives.reverse_machine",
                description=(
                    "Differentiation is a machine with a reversible polynomial setting: "
                    "an antiderivative runs each power-rule step backward, while an "
                    "arbitrary constant records what differentiation cannot recover."
                ),
                widget_affinity=[WidgetType.MAPPING, WidgetType.SLIDER],
            ),
        ),
        error_patterns=(
            "antiderivative_power_unchanged: output exponent does not increase",
            "antiderivative_coefficient_multiplied: coefficient is multiplied by the new exponent",
            "antiderivative_term_dropped: derivative of the response omits a source term",
        ),
        sources=(
            "OpenStax Calculus Volume 1, Section 4.10: Antiderivatives.",
            "OpenStax Calculus Volume 1, Section 5.3: The Fundamental Theorem of Calculus.",
        ),
        lesson_narrative=(
            _text(
                "An antiderivative is checked by differentiation, so each polynomial term reverses one power-rule step."
            ),
            _math(
                "new power=old power+1; new coefficient=old coefficient/new power",
                "The new power is the old power plus one, and the new coefficient is the old coefficient divided by the new power."
            ),
            _text(
                "One arbitrary constant is included because every constant has derivative zero."
            ),
        ),
        remediation=(
            _text(
                "Make a two-column table with each derivative term on the left and its reversed power-rule term on the right."
            ),
            _math(
                "differentiate the candidate -> recover the given polynomial",
                "Differentiate the candidate and confirm that it recovers the given polynomial."
            ),
            _text(
                "If the check fails, compare exponents first, then coefficients, and finally confirm that no term was omitted."
            ),
        ),
    )


def _ftc_pack() -> PedagogyPackSource:
    return PedagogyPackSource(
        source_id="pedagogy.source.ftc",
        revision=1,
        kc_id="kc.int.ftc",
        author=AUTHOR,
        misconceptions=(
            _misconception(
                "m.ftc.adds_endpoint_values",
                "Adds the two antiderivative endpoint values.",
                "The result uses ending value plus starting value instead of their directed difference.",
                "Write ending value minus starting value before substituting either number."
            ),
            _misconception(
                "m.ftc.reverses_subtraction",
                "Subtracts endpoint values in the opposite order.",
                "The response is the negative of the correctly oriented result.",
                "Match the first evaluation to the upper bound and subtract the evaluation at the lower bound."
            ),
            _misconception(
                "m.ftc.uses_integrand_values",
                "Evaluates the integrand at the endpoints instead of an antiderivative.",
                "The work substitutes bounds into f rather than into a function whose derivative is f.",
                "Name and verify an antiderivative F before evaluating either endpoint."
            ),
        ),
        metaphors=(
            Metaphor(
                id="met.ftc.odometer",
                description=(
                    "An antiderivative acts like an accumulation odometer: the change "
                    "in its reading from the starting bound to the ending bound gives "
                    "the accumulated amount between them."
                ),
                widget_affinity=[WidgetType.SLIDER, WidgetType.MAPPING],
            ),
        ),
        error_patterns=(
            "ftc_endpoint_addition: endpoint antiderivative values are added",
            "ftc_subtraction_reversed: lower value is reduced by upper value",
            "ftc_integrand_substitution: bounds are substituted into the integrand",
        ),
        sources=(
            "OpenStax Calculus Volume 1, Section 5.3: The Fundamental Theorem of Calculus.",
            "OpenStax Calculus Volume 1, Section 4.10: Antiderivatives.",
        ),
        lesson_narrative=(
            _text(
                "The Fundamental Theorem turns a definite integral into an endpoint change in any antiderivative of the integrand."
            ),
            _math(
                "integral_[a,b] f(x) dx=F(b)-F(a)",
                "The integral from a to b of f of x equals F of b minus F of a."
            ),
            _text(
                "The antiderivative supplies F, and the written bound order determines ending value minus starting value."
            ),
        ),
        remediation=(
            _text(
                "Separate the work into three lines: identify F, evaluate F at the ending bound, then subtract F at the starting bound."
            ),
            _math(
                "endpoint change=ending antiderivative value-starting antiderivative value",
                "Endpoint change is ending antiderivative value minus starting antiderivative value."
            ),
            _text(
                "Differentiate F once to verify the integrand before using either endpoint."
            ),
        ),
    )


def build_draft_pedagogy() -> PedagogySourceDocument:
    return PedagogySourceDocument(
        schema_version=2,
        source_version="ftc-wave-pedagogy-draft-v2",
        graph_version=2,
        pack_sources=(
            _graph_pack(),
            _area_pack(),
            _riemann_pack(),
            _definite_pack(),
            _antiderivative_pack(),
            _ftc_pack(),
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    from tutor.content.ftc_release import _atomic_write_model

    source = build_draft_pedagogy()
    _atomic_write_model(args.out, source)
    print(f"wrote {len(source.pack_sources)} unreviewed FTC pedagogy sources to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
