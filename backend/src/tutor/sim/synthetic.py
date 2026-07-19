"""Synthetic learners for diagnostic-policy simulation.

Ground-truth mastery is generated over the KC graph. Monotone learners
respect prerequisite closure (knowledge-space style: mastery of a node
requires mastery of all hard prerequisites). Patchy learners start monotone
and then lose random nodes, producing the non-monotonic "holes" the audit
asked the policy to be tested against. Answers follow a slip/guess model.
"""

from dataclasses import dataclass, field
from random import Random

from tutor.graph import service as graph_service
from tutor.schemas.common import EdgeType
from tutor.schemas.kc import GraphDocument


@dataclass
class SyntheticLearner:
    """Ground-truth mastery plus a slip/guess answer model."""

    mastered: set[str]
    slip: float = 0.0
    guess: float = 0.0
    rng: Random = field(default_factory=Random)

    def answer_correct(self, kc: str) -> bool:
        """Answer a probe on ``kc``: mastered learners slip, unmastered guess."""
        if kc in self.mastered:
            return self.rng.random() >= self.slip
        return self.rng.random() < self.guess


def _hard_predecessors(graph: GraphDocument) -> dict[str, list[str]]:
    predecessors: dict[str, list[str]] = {node.id: [] for node in graph.nodes}
    for edge in graph.edges:
        if edge.type == EdgeType.HARD:
            predecessors[edge.to_kc].append(edge.from_kc)
    return predecessors


def make_monotone_mastery(
    graph: GraphDocument,
    rng: Random,
    root_rate: float = 0.9,
    inherit_rate: float = 0.8,
) -> set[str]:
    """Downward-closed mastery: a node is only masterable if all hard prereqs are."""
    predecessors = _hard_predecessors(graph)
    mastered: set[str] = set()
    for kc in graph_service.topological_order(graph):
        node_preds = predecessors[kc]
        if node_preds and not all(pred in mastered for pred in node_preds):
            continue
        rate = root_rate if not node_preds else inherit_rate
        if rng.random() < rate:
            mastered.add(kc)
    return mastered


def make_patchy_mastery(
    graph: GraphDocument,
    rng: Random,
    holes: int = 2,
    root_rate: float = 0.9,
    inherit_rate: float = 0.8,
) -> set[str]:
    """Monotone mastery with random holes punched in (non-monotonic knowledge)."""
    mastered = make_monotone_mastery(graph, rng, root_rate, inherit_rate)
    candidates = sorted(mastered)
    for _ in range(min(holes, len(candidates))):
        victim = rng.choice(candidates)
        mastered.discard(victim)
        candidates.remove(victim)
    return mastered


def generate_population(
    graph: GraphDocument,
    n: int,
    seed: int = 7,
    patchy_fraction: float = 0.4,
    slip: float = 0.1,
    guess: float = 0.15,
    holes: int = 2,
) -> list[SyntheticLearner]:
    """Deterministic population; the first ``patchy_fraction`` of learners get holes."""
    master_rng = Random(seed)
    learners: list[SyntheticLearner] = []
    for index in range(n):
        generation_rng = Random(master_rng.randrange(2**32))
        answer_rng = Random(master_rng.randrange(2**32))
        if index < int(patchy_fraction * n):
            mastery = make_patchy_mastery(graph, generation_rng, holes=holes)
        else:
            mastery = make_monotone_mastery(graph, generation_rng)
        learners.append(
            SyntheticLearner(mastered=mastery, slip=slip, guess=guess, rng=answer_rng)
        )
    return learners
