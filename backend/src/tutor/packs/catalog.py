"""Pure validation helpers for reviewed pedagogy catalog releases."""

from __future__ import annotations

from collections.abc import Iterable

from tutor.schemas.kc import GraphDocument
from tutor.schemas.pedagogy import PedagogyPackCatalog


def validate_pedagogy_catalog(
    catalog: PedagogyPackCatalog,
    graph: GraphDocument,
    required_kcs: Iterable[str] | None = None,
) -> list[str]:
    """Return deterministic release errors for a catalog/graph pairing.

    Schema-level review and identity invariants are enforced while parsing the
    catalog. This helper checks the external graph pin and an optional set of
    KCs that another content release requires the catalog to cover.
    """

    errors: list[str] = []
    graph_kcs = graph.node_ids()
    catalog_kcs = set(catalog.pack_by_kc)
    required = set(required_kcs or ())

    if catalog.graph_version != graph.graph_version:
        errors.append(
            "graph version mismatch: "
            f"catalog={catalog.graph_version}, graph={graph.graph_version}"
        )
    for kc_id in sorted(catalog_kcs - graph_kcs):
        errors.append(f"catalog pack KC is absent from graph: {kc_id}")
    for kc_id in sorted(required - graph_kcs):
        errors.append(f"required catalog KC is absent from graph: {kc_id}")
    for kc_id in sorted(required - catalog_kcs):
        errors.append(f"required KC has no reviewed pedagogy pack: {kc_id}")
    return errors


def reviewed_misconception_ids(
    catalog: PedagogyPackCatalog,
) -> dict[str, frozenset[str]]:
    """Return the exact reviewed misconception membership pinned by a catalog."""

    return {
        pack.kc_id: frozenset(item.id for item in pack.misconceptions)
        for pack in catalog.packs
    }
