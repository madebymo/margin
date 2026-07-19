"""KC graph service.

Pure functions over GraphDocument (no DB dependency) plus one persistence
function, publish_graph. Traversal outside this module is discouraged: the
service interface is the seam that would let a dedicated graph store replace
Postgres later.
"""

import heapq
from collections import defaultdict

from sqlalchemy.orm import Session

from tutor.db.models import GraphVersionRow, KCEdgeRow, KCNodeRow
from tutor.schemas.common import EdgeType
from tutor.schemas.kc import GraphDocument, find_cycle


class GraphCycleError(ValueError):
    """Raised when a graph that must be a DAG contains a cycle."""

    def __init__(self, cycle: list[str]) -> None:
        self.cycle = cycle
        super().__init__("graph contains a cycle: " + " -> ".join(cycle))


def validate_acyclic(doc: GraphDocument) -> None:
    """Raise GraphCycleError (with the offending path) if the graph has a cycle."""
    cycle = find_cycle(
        [n.id for n in doc.nodes], [(e.from_kc, e.to_kc) for e in doc.edges]
    )
    if cycle is not None:
        raise GraphCycleError(cycle)


def ancestor_subgraph(
    doc: GraphDocument, target_kc: str, hard_only: bool = False
) -> GraphDocument:
    """Return the subgraph of ``target_kc`` and all its (transitive) prerequisites.

    With ``hard_only=True``, traversal and the returned edges are restricted to
    hard prerequisite edges.
    """
    if target_kc not in doc.node_ids():
        raise KeyError(f"unknown kc: {target_kc}")

    predecessors: dict[str, list[str]] = defaultdict(list)
    for edge in doc.edges:
        if hard_only and edge.type != EdgeType.HARD:
            continue
        predecessors[edge.to_kc].append(edge.from_kc)

    keep = {target_kc}
    stack = [target_kc]
    while stack:
        node = stack.pop()
        for pred in predecessors[node]:
            if pred not in keep:
                keep.add(pred)
                stack.append(pred)

    nodes = [n for n in doc.nodes if n.id in keep]
    edges = [
        e
        for e in doc.edges
        if e.from_kc in keep
        and e.to_kc in keep
        and (not hard_only or e.type == EdgeType.HARD)
    ]
    return GraphDocument(graph_version=doc.graph_version, nodes=nodes, edges=edges)


def topological_order(doc: GraphDocument, kc_ids: set[str] | None = None) -> list[str]:
    """Deterministic topological order (lexicographic tie-break).

    If ``kc_ids`` is given, orders the induced subgraph on that subset.
    """
    subset = set(kc_ids) if kc_ids is not None else doc.node_ids()
    unknown = subset - doc.node_ids()
    if unknown:
        raise KeyError(f"unknown kcs: {sorted(unknown)}")

    indegree: dict[str, int] = {n: 0 for n in subset}
    successors: dict[str, list[str]] = defaultdict(list)
    for edge in doc.edges:
        if edge.from_kc in subset and edge.to_kc in subset:
            successors[edge.from_kc].append(edge.to_kc)
            indegree[edge.to_kc] += 1

    heap = [n for n, d in indegree.items() if d == 0]
    heapq.heapify(heap)
    order: list[str] = []
    while heap:
        node = heapq.heappop(heap)
        order.append(node)
        for succ in successors[node]:
            indegree[succ] -= 1
            if indegree[succ] == 0:
                heapq.heappush(heap, succ)

    if len(order) != len(subset):
        cycle = find_cycle(
            sorted(subset),
            [
                (e.from_kc, e.to_kc)
                for e in doc.edges
                if e.from_kc in subset and e.to_kc in subset
            ],
        )
        raise GraphCycleError(cycle or sorted(subset - set(order)))
    return order


def roots(doc: GraphDocument) -> list[str]:
    """Nodes with no prerequisites, sorted lexicographically."""
    has_incoming = {e.to_kc for e in doc.edges}
    return sorted(n.id for n in doc.nodes if n.id not in has_incoming)


def descendants(doc: GraphDocument, kc_id: str) -> set[str]:
    """All KCs that (transitively) depend on ``kc_id`` (exclusive of itself)."""
    if kc_id not in doc.node_ids():
        raise KeyError(f"unknown kc: {kc_id}")
    successors: dict[str, list[str]] = defaultdict(list)
    for edge in doc.edges:
        successors[edge.from_kc].append(edge.to_kc)
    seen: set[str] = set()
    stack = [kc_id]
    while stack:
        node = stack.pop()
        for succ in successors[node]:
            if succ not in seen:
                seen.add(succ)
                stack.append(succ)
    return seen


def publish_graph(session: Session, doc: GraphDocument) -> int:
    """Validate acyclicity, then persist the graph as a published version.

    Flushes and returns the graph_versions row id; the caller owns the commit.
    """
    validate_acyclic(doc)
    version_row = GraphVersionRow(version=doc.graph_version, status="published")
    session.add(version_row)
    session.flush()
    for node in doc.nodes:
        session.add(
            KCNodeRow(
                graph_version_id=version_row.id,
                kc_id=node.id,
                name=node.name,
                description=node.description,
                course_level=node.course_level,
                canonical_examples=node.canonical_examples,
            )
        )
    for edge in doc.edges:
        session.add(
            KCEdgeRow(
                graph_version_id=version_row.id,
                from_kc=edge.from_kc,
                to_kc=edge.to_kc,
                type=edge.type.value,
                rationale=edge.rationale,
            )
        )
    session.flush()
    return version_row.id
