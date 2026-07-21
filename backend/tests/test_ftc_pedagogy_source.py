"""Qualification of the six pending FTC-wave pedagogy sources."""

from __future__ import annotations

from tutor.content.ftc_pedagogy_draft import AUTHOR, build_draft_pedagogy
from tutor.content.ftc_release import DEFAULT_PEDAGOGY_SOURCE_PATH, TARGET_KCS
from tutor.schemas.assessment import MathPromptSegment
from tutor.schemas.common import WidgetType
from tutor.schemas.pedagogy_authoring import PedagogySourceDocument


def test_packaged_pedagogy_is_exact_schema_v2_authoring_output():
    packaged = PedagogySourceDocument.model_validate_json(
        DEFAULT_PEDAGOGY_SOURCE_PATH.read_text(encoding="utf-8")
    )
    authored = build_draft_pedagogy()

    assert packaged == authored
    assert DEFAULT_PEDAGOGY_SOURCE_PATH.read_text(encoding="utf-8") == (
        authored.model_dump_json(indent=2) + "\n"
    )
    assert packaged.schema_version == 2
    assert packaged.graph_version == 2
    assert {pack.kc_id for pack in packaged.pack_sources} == set(TARGET_KCS)


def test_each_pack_has_complete_reviewable_instructional_content():
    source = build_draft_pedagogy()

    assert len(source.pack_sources) == 6
    assert sum(len(pack.misconceptions) for pack in source.pack_sources) == 18
    assert sum(len(pack.metaphors) for pack in source.pack_sources) == 6
    assert sum(len(pack.error_patterns) for pack in source.pack_sources) == 18
    assert sum(len(pack.sources) for pack in source.pack_sources) == 12
    for pack in source.pack_sources:
        assert pack.author == AUTHOR
        assert pack.revision == 1
        assert len(pack.misconceptions) == 3
        assert len(pack.metaphors) == 1
        assert len(pack.error_patterns) == 3
        assert len(pack.sources) == 2
        assert pack.lesson_narrative
        assert pack.remediation
        assert all(
            WidgetType.CLICK_REGION not in metaphor.widget_affinity
            and WidgetType.LIVE_INPUT not in metaphor.widget_affinity
            for metaphor in pack.metaphors
        )
        assert all(
            segment.spoken_text
            for segment in (*pack.lesson_narrative, *pack.remediation)
            if isinstance(segment, MathPromptSegment)
        )


def test_misconception_taxonomy_matches_deterministic_error_signatures():
    source = build_draft_pedagogy()
    expected = {
        "kc.fun.graph_reading": {
            "m.graph_reading.axes_coordinates",
            "m.graph_reading.intercept_confusion",
            "m.graph_reading.slope_direction",
        },
        "kc.int.area_under_curve": {
            "m.area_under_curve.omits_region",
            "m.area_under_curve.triangle_factor",
            "m.area_under_curve.uses_endpoint_height",
        },
        "kc.int.riemann_sums": {
            "m.riemann_sums.endpoint_choice",
            "m.riemann_sums.omits_width",
            "m.riemann_sums.midpoint_confusion",
        },
        "kc.int.definite_integral": {
            "m.definite_integral.bound_order",
            "m.definite_integral.ignores_sign",
            "m.definite_integral.breaks_additivity",
        },
        "kc.int.antiderivatives": {
            "m.antiderivatives.keeps_exponent",
            "m.antiderivatives.multiplies_coefficient",
            "m.antiderivatives.drops_term",
        },
        "kc.int.ftc": {
            "m.ftc.adds_endpoint_values",
            "m.ftc.reverses_subtraction",
            "m.ftc.uses_integrand_values",
        },
    }

    assert {
        pack.kc_id: {item.id for item in pack.misconceptions}
        for pack in source.pack_sources
    } == expected
