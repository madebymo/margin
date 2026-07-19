"""Graph service behavior on synthetic graphs and the real seed."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from tutor.graph import service
from tutor.graph.service import GraphCycleError
from tutor.schemas.kc import GraphDocument, KCEdge, KCNode
from tutor.seed.load_seed import load_graph


def _node(kc_id: str) -> KCNode:
    return KCNode(
        id=kc_id,
        name=kc_id,
        description="d",
        course_level="Algebra 1",
        canonical_examples=["example"],
    )


def _edge(from_kc: str, to_kc: str, edge_type: str = "hard") -> KCEdge:
    return KCEdge(from_kc=from_kc, to_kc=to_kc, type=edge_type, rationale="r")


@pytest.fixture(scope="module")
def seed() -> GraphDocument:
    return load_graph()


def test_planted_cycle_raises_with_path():
    ids = ["kc.alg.a", "kc.alg.b", "kc.alg.c"]
    # model_construct bypasses GraphDocument validation so the service check is exercised.
    doc = GraphDocument.model_construct(
        graph_version=1,
        nodes=[_node(i) for i in ids],
        edges=[
            _edge("kc.alg.a", "kc.alg.b"),
            _edge("kc.alg.b", "kc.alg.c"),
            _edge("kc.alg.c", "kc.alg.a"),
        ],
    )
    with pytest.raises(GraphCycleError) as excinfo:
        service.validate_acyclic(doc)
    assert set(ids) <= set(excinfo.value.cycle)


def test_graph_document_rejects_cycle_at_validation():
    node_dicts = [
        {
            "id": kc,
            "name": kc,
            "description": "d",
            "course_level": "Algebra 1",
            "canonical_examples": ["e"],
        }
        for kc in ["kc.alg.a", "kc.alg.b"]
    ]
    payload = {
        "graph_version": 1,
        "nodes": node_dicts,
        "edges": [
            {"from_kc": "kc.alg.a", "to_kc": "kc.alg.b", "type": "hard", "rationale": "r"},
            {"from_kc": "kc.alg.b", "to_kc": "kc.alg.a", "type": "hard", "rationale": "r"},
        ],
    }
    with pytest.raises(ValidationError):
        GraphDocument.model_validate(payload)


def test_usub_ancestors_reach_algebra(seed):
    sub = service.ancestor_subgraph(seed, "kc.int.u_substitution")
    ids = sub.node_ids()
    for required in [
        "kc.der.chain_rule",
        "kc.fun.composition",
        "kc.der.power_rule",
        "kc.alg.exponent_rules",
    ]:
        assert required in ids
    assert "kc.der.implicit_differentiation" not in ids
    assert "kc.int.u_sub_definite" not in ids


def test_hard_only_excludes_soft_ancestors():
    doc = GraphDocument(
        graph_version=1,
        nodes=[_node("kc.alg.a"), _node("kc.alg.b"), _node("kc.alg.c")],
        edges=[
            _edge("kc.alg.a", "kc.alg.b", "soft"),
            _edge("kc.alg.b", "kc.alg.c"),
        ],
    )
    hard = service.ancestor_subgraph(doc, "kc.alg.c", hard_only=True)
    assert hard.node_ids() == {"kc.alg.b", "kc.alg.c"}
    full = service.ancestor_subgraph(doc, "kc.alg.c")
    assert full.node_ids() == {"kc.alg.a", "kc.alg.b", "kc.alg.c"}


def test_topological_order_deterministic_and_valid(seed):
    first = service.topological_order(seed)
    second = service.topological_order(seed)
    assert first == second
    position = {kc: index for index, kc in enumerate(first)}
    for edge in seed.edges:
        assert position[edge.from_kc] < position[edge.to_kc]


def test_topological_order_subset(seed):
    subset = {"kc.int.u_substitution", "kc.der.chain_rule", "kc.fun.composition"}
    order = service.topological_order(seed, subset)
    assert order.index("kc.fun.composition") < order.index("kc.der.chain_rule")
    assert order.index("kc.der.chain_rule") < order.index("kc.int.u_substitution")


def test_descendants_of_chain_rule(seed):
    downstream = service.descendants(seed, "kc.der.chain_rule")
    assert "kc.int.u_substitution" in downstream
    assert "kc.der.chain_rule" not in downstream


@settings(deadline=None, max_examples=50)
@given(data=st.data())
def test_random_dags_validate_and_back_edges_raise(data):
    n = data.draw(st.integers(min_value=2, max_value=8))
    ids = [f"kc.alg.n{i}" for i in range(n)]
    possible = [(i, j) for i in range(n) for j in range(i + 1, n)]
    chosen = data.draw(
        st.lists(st.sampled_from(possible), unique=True, min_size=1, max_size=len(possible))
    )

    def edge_dict(i: int, j: int) -> dict:
        return {"from_kc": ids[i], "to_kc": ids[j], "type": "hard", "rationale": "r"}

    node_dicts = [
        {
            "id": kc,
            "name": kc,
            "description": "d",
            "course_level": "Algebra 1",
            "canonical_examples": ["e"],
        }
        for kc in ids
    ]
    payload = {
        "graph_version": 1,
        "nodes": node_dicts,
        "edges": [edge_dict(i, j) for i, j in chosen],
    }
    GraphDocument.model_validate(payload)  # forward-only edges: always a DAG

    i, j = chosen[0]
    cyclic = {
        **payload,
        "edges": payload["edges"] + [edge_dict(j, i)],  # back edge closes a 2-cycle
    }
    with pytest.raises(ValidationError):
        GraphDocument.model_validate(cyclic)
