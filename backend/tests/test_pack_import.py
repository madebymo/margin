"""Pack CSV import surface: template must round-trip into a valid PedagogyPack."""

from pathlib import Path

import tutor.packs
from tutor.packs.import_csv import parse_pack_csv
from tutor.schemas.common import ReviewStatus, WidgetType

TEMPLATE = Path(tutor.packs.__file__).resolve().parent / "template.csv"


def test_template_parses_into_valid_pack():
    packs = parse_pack_csv(TEMPLATE)
    assert [p.kc_id for p in packs] == ["kc.int.u_substitution"]

    pack = packs[0]
    assert pack.review_status == ReviewStatus.DRAFT

    misconception_ids = {m.id for m in pack.misconceptions}
    assert "m.usub.forget_dx" in misconception_ids
    assert "m.usub.bounds_unchanged" in misconception_ids

    affinities = {wt for met in pack.metaphors for wt in met.widget_affinity}
    assert WidgetType.SLIDER in affinities

    assert pack.error_patterns
    assert pack.sources
