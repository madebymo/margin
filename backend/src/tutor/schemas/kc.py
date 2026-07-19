"""Knowledge-component (KC) graph schemas.

Edges point prerequisite -> dependent. A GraphDocument validates that node ids
are unique, all edge endpoints exist, and the graph is acyclic.
"""

from collections import defaultdict

from pydantic import BaseModel, Field, model_validator

from tutor.schemas.common import EdgeType

KC_ID_PATTERN = r"^kc\.(alg|fun|lim|der|int)\.[a-z0-9_]+$"


def find_cycle(node_ids: list[str], edges: list[tuple[str, str]]) -> list[str] | None:
    """Return one cycle as a closed path of kc ids, or None if the graph is a DAG.

    Iterative three-color depth-first search: hitting a GRAY (in-progress) node
    means the current DFS path contains a cycle, which is returned directly.
    Edges referencing unknown nodes are ignored (endpoint existence is checked
    separately by GraphDocument validation).
    """
    successors: dict[str, list[str]] = defaultdict(list)
    for from_kc, to_kc in edges:
        successors[from_kc].append(to_kc)

    white, gray, black = 0, 1, 2
    color: dict[str, int] = {n: white for n in node_ids}

    for root in sorted(node_ids):
        if color[root] != white:
            continue
        color[root] = gray
        path: list[str] = [root]
        stack = [(root, iter(sorted(successors[root])))]
        while stack:
            node, succ_iter = stack[-1]
            advanced = False
            for succ in succ_iter:
                if color.get(succ, black) == gray:
                    cycle_start = path.index(succ)
                    return path[cycle_start:] + [succ]
                if color.get(succ, black) == white:
                    color[succ] = gray
                    path.append(succ)
                    stack.append((succ, iter(sorted(successors[succ]))))
                    advanced = True
                    break
            if not advanced:
                stack.pop()
                path.pop()
                color[node] = black
    return None


class KCNode(BaseModel):
    """A single knowledge component (skill) in the prerequisite graph."""

    id: str = Field(pattern=KC_ID_PATTERN)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    course_level: str = Field(min_length=1)
    canonical_examples: list[str] = Field(min_length=1, max_length=3)
    version: int = 1


class KCEdge(BaseModel):
    """A prerequisite relation: from_kc must be learned before to_kc."""

    from_kc: str = Field(pattern=KC_ID_PATTERN)
    to_kc: str = Field(pattern=KC_ID_PATTERN)
    type: EdgeType
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def _no_self_loop(self) -> "KCEdge":
        if self.from_kc == self.to_kc:
            raise ValueError(f"self-loop edge on {self.from_kc}")
        return self


class GraphDocument(BaseModel):
    """A complete, versioned KC graph. Validation enforces DAG structure."""

    graph_version: int = Field(ge=1)
    nodes: list[KCNode] = Field(min_length=1)
    edges: list[KCEdge]

    @model_validator(mode="after")
    def _validate_structure(self) -> "GraphDocument":
        ids = [n.id for n in self.nodes]
        id_set = set(ids)
        if len(ids) != len(id_set):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate node ids: {dupes}")
        for edge in self.edges:
            if edge.from_kc not in id_set:
                raise ValueError(f"edge references unknown node: {edge.from_kc}")
            if edge.to_kc not in id_set:
                raise ValueError(f"edge references unknown node: {edge.to_kc}")
        cycle = find_cycle(ids, [(e.from_kc, e.to_kc) for e in self.edges])
        if cycle is not None:
            raise ValueError(f"graph contains a cycle: {' -> '.join(cycle)}")
        return self

    def node_ids(self) -> set[str]:
        """Return the set of node ids in this graph."""
        return {n.id for n in self.nodes}
