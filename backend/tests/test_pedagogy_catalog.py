"""Reviewed pedagogy catalogs are explicit, immutable release documents."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

import tutor.packs.loader as pack_loader
from tutor.packs.catalog import (
    reviewed_misconception_ids,
    validate_pedagogy_catalog,
)
from tutor.packs.loader import load_pedagogy_catalog, load_template_packs
from tutor.schemas.common import ReviewStatus
from tutor.schemas.kc import GraphDocument
from tutor.schemas.pedagogy import (
    Metaphor,
    Misconception,
    PedagogyPack,
    PedagogyPackCatalog,
    PedagogyPackProvenance,
)
from tutor.seed.load_seed import load_graph


def _provenance() -> PedagogyPackProvenance:
    return PedagogyPackProvenance(
        author="Curriculum author",
        reviewed_by="Independent reviewer",
        reviewed_at=datetime(2026, 7, 20, tzinfo=timezone.utc),
    )


def _approved_pack(
    kc_id: str = "kc.der.power_rule",
    *,
    misconception_id: str = "m.power.multiply_exponent",
    metaphor_id: str = "met.power_machine",
) -> PedagogyPack:
    return PedagogyPack(
        kc_id=kc_id,
        misconceptions=[
            Misconception(
                id=misconception_id,
                description="Forgets to multiply by the exponent.",
                error_signature="The exponent is changed without a coefficient.",
                remediation_hint="Bring the exponent down first.",
            )
        ],
        metaphors=[
            Metaphor(
                id=metaphor_id,
                description="A power rule input-output machine.",
                widget_affinity=["mapping"],
            )
        ],
        sources=["Reviewed internal curriculum"],
        review_status=ReviewStatus.HUMAN_APPROVED,
        version=1,
        provenance=_provenance(),
    )


def _catalog(
    *packs: PedagogyPack,
    graph_version: int | None = None,
) -> PedagogyPackCatalog:
    return PedagogyPackCatalog(
        catalog_version="test-pedagogy-v1",
        graph_version=(
            load_graph().graph_version
            if graph_version is None
            else graph_version
        ),
        published_by="Release manager",
        published_at=datetime(2026, 7, 20, tzinfo=timezone.utc),
        packs=list(packs),
    )


def test_packaged_catalog_is_an_honest_empty_release():
    catalog = load_pedagogy_catalog()

    assert catalog.schema_version == 2
    assert catalog.catalog_version == "pedagogy-catalog-empty-v2"
    assert catalog.graph_version == load_graph().graph_version
    assert catalog.packs == ()
    assert validate_pedagogy_catalog(catalog, load_graph()) == []

    # The ambient CSV contains a useful authoring draft, but loading the exact
    # catalog neither merges it nor promotes it to reviewed content.
    assert load_template_packs()
    assert {pack.review_status for pack in load_template_packs().values()} == {
        ReviewStatus.DRAFT
    }


def test_loader_parses_only_the_explicit_catalog_document(tmp_path, monkeypatch):
    catalog = _catalog(_approved_pack())
    path = tmp_path / "release.json"
    path.write_text(catalog.model_dump_json(), encoding="utf-8")

    def ambient_merge_must_not_run(*args, **kwargs):
        raise AssertionError("ambient draft merge was called")

    monkeypatch.setattr(pack_loader, "load_packs", ambient_merge_must_not_run)
    assert load_pedagogy_catalog(path) == catalog


def test_reviewed_pack_requires_strict_timezone_aware_provenance():
    payload = _approved_pack().model_dump(mode="json")
    payload.pop("provenance")
    with pytest.raises(ValidationError, match="requires reviewed provenance"):
        PedagogyPack.model_validate(payload)

    provenance = _provenance().model_dump(mode="json")
    provenance["unexpected"] = "ignored data must not cross the boundary"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        PedagogyPackProvenance.model_validate(provenance)

    provenance = _provenance().model_dump(mode="json")
    provenance["reviewed_at"] = "2026-07-20T00:00:00"
    with pytest.raises(ValidationError, match="include a timezone"):
        PedagogyPackProvenance.model_validate(provenance)


@pytest.mark.parametrize(
    ("author", "reviewed_by"),
    [
        ("Same Person", "Same Person"),
        ("  Same Person  ", "same person"),
        ("SAME PERSON", " same person "),
    ],
)
def test_review_provenance_requires_an_independent_reviewer(
    author: str,
    reviewed_by: str,
):
    with pytest.raises(ValidationError, match="someone other than the author"):
        PedagogyPackProvenance(
            author=author,
            reviewed_by=reviewed_by,
            reviewed_at=datetime(2026, 7, 20, tzinfo=timezone.utc),
        )


@pytest.mark.parametrize("field", ["author", "reviewed_by"])
def test_review_provenance_rejects_whitespace_only_people(field: str):
    payload = _provenance().model_dump(mode="json")
    payload[field] = " \t "
    with pytest.raises(ValidationError, match="at least 1 character"):
        PedagogyPackProvenance.model_validate(payload)


def test_review_provenance_trims_people_before_storing_them():
    provenance = PedagogyPackProvenance(
        author="  Curriculum author ",
        reviewed_by=" Independent reviewer  ",
        reviewed_at=datetime(2026, 7, 20, tzinfo=timezone.utc),
    )

    assert provenance.author == "Curriculum author"
    assert provenance.reviewed_by == "Independent reviewer"


def test_human_approved_pack_requires_at_least_one_meaningful_source():
    payload = _approved_pack().model_dump(mode="json")
    payload["sources"] = []
    with pytest.raises(ValidationError, match="requires at least one source"):
        PedagogyPack.model_validate(payload)

    payload["sources"] = [" \t "]
    with pytest.raises(ValidationError, match="meaningful nonblank strings"):
        PedagogyPack.model_validate(payload)

    draft = PedagogyPack(kc_id="kc.der.power_rule", sources=[])
    assert draft.review_status == ReviewStatus.DRAFT
    assert draft.sources == []


def test_catalog_is_strict_frozen_and_accepts_only_reviewed_packs():
    catalog = _catalog(_approved_pack())
    with pytest.raises(ValidationError, match="frozen"):
        catalog.catalog_version = "replacement"

    payload = catalog.model_dump(mode="json")
    payload["unexpected"] = True
    with pytest.raises(ValidationError, match="extra_forbidden"):
        PedagogyPackCatalog.model_validate(payload)

    draft_payload = _approved_pack().model_dump(mode="json")
    draft_payload["review_status"] = "draft"
    draft_payload["provenance"] = None
    with pytest.raises(ValidationError, match="not human_approved"):
        _catalog(PedagogyPack.model_validate(draft_payload))


def test_catalog_rejects_duplicate_pack_and_content_identities():
    power = _approved_pack()
    with pytest.raises(ValidationError, match="one pack per KC"):
        _catalog(power, power)

    chain = _approved_pack(
        "kc.der.chain_rule",
        misconception_id=power.misconceptions[0].id,
        metaphor_id="met.chain_gears",
    )
    with pytest.raises(ValidationError, match="misconception id"):
        _catalog(power, chain)

    chain = _approved_pack(
        "kc.der.chain_rule",
        misconception_id="m.chain.outer_only",
        metaphor_id=power.metaphors[0].id,
    )
    with pytest.raises(ValidationError, match="metaphor id"):
        _catalog(power, chain)


def test_catalog_validation_checks_graph_pin_inventory_and_required_coverage():
    graph = load_graph()
    catalog = _catalog(_approved_pack())
    assert validate_pedagogy_catalog(
        catalog, graph, required_kcs={"kc.der.power_rule"}
    ) == []

    graph_payload = graph.model_dump(mode="json")
    graph_payload["graph_version"] = graph.graph_version + 1
    mismatched_graph = GraphDocument.model_validate(graph_payload)
    assert validate_pedagogy_catalog(catalog, mismatched_graph) == [
        "graph version mismatch: "
        f"catalog={graph.graph_version}, graph={graph.graph_version + 1}"
    ]

    assert validate_pedagogy_catalog(
        catalog,
        graph,
        required_kcs={"kc.der.chain_rule", "kc.der.unknown_skill"},
    ) == [
        "required catalog KC is absent from graph: kc.der.unknown_skill",
        "required KC has no reviewed pedagogy pack: kc.der.chain_rule",
        "required KC has no reviewed pedagogy pack: kc.der.unknown_skill",
    ]

    unknown_catalog = _catalog(_approved_pack("kc.der.unknown_skill"))
    assert validate_pedagogy_catalog(unknown_catalog, graph) == [
        "catalog pack KC is absent from graph: kc.der.unknown_skill"
    ]


def test_reviewed_misconception_membership_comes_only_from_catalog_snapshot():
    catalog = _catalog(_approved_pack())

    assert reviewed_misconception_ids(catalog) == {
        "kc.der.power_rule": frozenset({"m.power.multiply_exponent"})
    }
