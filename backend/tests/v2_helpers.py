"""Explicitly trusted, test-only content fixtures for session-v2 tests.

The packaged item bank is intentionally an unreleased draft. Tests that need
to exercise the runtime construct a narrow graph and mark a copied bank as
approved inside the test boundary; none of these review claims ship as
student-eligible content.
"""

from tutor.content.item_bank import load_item_bank
from tutor.schemas.assessment import ItemBankDocument
from tutor.schemas.kc import GraphDocument
from tutor.seed.load_seed import load_graph

POWER_RULE_KC = "kc.der.power_rule"


def approved_power_rule_bank() -> ItemBankDocument:
    """Return an explicit test-only release of the packaged draft inventory."""
    payload = load_item_bank().model_dump(mode="json")
    payload["bank_version"] = "test-approved-power-v2"
    payload["released_kcs"] = [POWER_RULE_KC]
    next_order: dict[str, int] = {}
    for item in payload["items"]:
        surface = item["eligible_surfaces"][0]
        next_order[surface] = next_order.get(surface, 0) + 10
        item["allocation_order"] = next_order[surface]
        item["review_status"] = "human_approved"
        item["provenance"] = {
            "source": "test-only-approved-fixture",
            "author": "test fixture",
            "reviewed_by": "automated test fixture",
            "reviewed_at": "2026-01-01T00:00:00Z",
        }
    return ItemBankDocument.model_validate(payload)


def power_rule_only_graph() -> GraphDocument:
    """Return a graph whose complete hard closure is the one fixture KC."""
    graph = load_graph()
    node = next(item for item in graph.nodes if item.id == POWER_RULE_KC)
    return GraphDocument(
        graph_version=graph.graph_version,
        nodes=[node],
        edges=[],
    )
