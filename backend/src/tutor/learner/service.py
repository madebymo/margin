"""LearnerModelService: derived mastery over an append-only evidence log.

Invariants (from the plan):
- Direct and inferred evidence are tracked separately and never merged in storage.
- Inferred evidence alone can never cross the mastery threshold (capped below it).
- A dependent miss never lowers prerequisite beliefs; only direct misses do.
- Multi-KC events inform routing only — they are recorded but skip BKT updates.
- Derived state is rebuildable by replaying the event log (see ``replay``).
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid4

from tutor.learner.params import DEFAULT_PARAMS_V1, BKTParams
from tutor.schemas.common import EdgeType
from tutor.schemas.kc import GraphDocument
from tutor.schemas.learner import DerivedLearnerState, EvidenceEvent, MasteryEstimate


@dataclass
class _KCState:
    direct: float
    inferred: float = 0.0
    observations: int = 0
    last_practiced: datetime | None = None


class LearnerModelService:
    """Holds derived mastery for one learner over one graph version."""

    def __init__(
        self,
        graph: GraphDocument,
        params: BKTParams | None = None,
        assumed_floor_levels: set[str] | None = None,
        learner_id: UUID | None = None,
    ) -> None:
        self._graph = graph
        self._params = params or DEFAULT_PARAMS_V1
        self._floor_levels = set(assumed_floor_levels or set())
        self.learner_id = learner_id or uuid4()
        self._state: dict[str, _KCState] = {}
        for node in graph.nodes:
            prior = (
                self._params.prior_assumed_floor
                if node.course_level in self._floor_levels
                else self._params.prior_default
            )
            self._state[node.id] = _KCState(direct=prior)
        self._hard_ancestors = self._compute_hard_ancestors()
        self._events: list[EvidenceEvent] = []
        self._misconceptions: list[str] = []

    def _compute_hard_ancestors(self) -> dict[str, dict[str, int]]:
        """Map each KC to its hard-edge ancestors with minimum depth (BFS)."""
        predecessors: dict[str, list[str]] = defaultdict(list)
        for edge in self._graph.edges:
            if edge.type == EdgeType.HARD:
                predecessors[edge.to_kc].append(edge.from_kc)
        result: dict[str, dict[str, int]] = {}
        for kc in self._state:
            depths: dict[str, int] = {}
            queue: list[tuple[str, int]] = [(kc, 0)]
            while queue:
                node, depth = queue.pop(0)
                for pred in predecessors[node]:
                    if pred not in depths or depths[pred] > depth + 1:
                        depths[pred] = depth + 1
                        queue.append((pred, depth + 1))
            result[kc] = depths
        return result

    # -- updates ------------------------------------------------------------

    def apply_event(self, event: EvidenceEvent) -> None:
        """Append one evidence event and update derived mastery."""
        self._events.append(event)
        if event.misconception_id and event.misconception_id not in self._misconceptions:
            self._misconceptions.append(event.misconception_id)
        if len(event.kc_ids) != 1:
            return  # multi-KC items inform routing only (no allocation rule in v1)
        kc = event.kc_ids[0]
        if kc not in self._state:
            raise KeyError(f"unknown kc: {kc}")
        state = self._state[kc]
        class_params = self._params.response_class[event.response_class]
        prior = state.direct

        if event.correct:
            numerator = prior * (1 - class_params.slip)
            denominator = numerator + (1 - prior) * class_params.guess
        else:
            numerator = prior * class_params.slip
            denominator = numerator + (1 - prior) * (1 - class_params.guess)
        posterior = numerator / denominator if denominator > 0 else prior
        posterior = posterior + (1 - posterior) * self._params.learn

        assisted = event.assisted or event.hints_used > 0
        if assisted:
            posterior = prior + self._params.assisted_credit * (posterior - prior)

        state.direct = min(1.0, max(0.0, posterior))
        state.observations += 1
        state.last_practiced = event.t

        if event.correct and not assisted:
            self._propagate_up(kc)

    def _propagate_up(self, kc: str) -> None:
        """Discounted inferred-evidence increase for hard ancestors, capped."""
        for ancestor, depth in self._hard_ancestors[kc].items():
            ancestor_state = self._state[ancestor]
            gain = (
                self._params.propagation_strength
                * (self._params.propagation_decay ** (depth - 1))
                * (1 - ancestor_state.inferred)
            )
            ancestor_state.inferred = min(
                self._params.inferred_cap, ancestor_state.inferred + gain
            )

    # -- reads ---------------------------------------------------------------

    def routing_score(self, kc: str) -> float:
        """Belief used for routing: direct once observed, else best of prior/inferred."""
        state = self._state[kc]
        if state.observations > 0:
            return state.direct
        return max(state.direct, state.inferred)

    def is_mastered(self, kc: str) -> bool:
        """Routing-level mastery check against the params threshold."""
        return self.routing_score(kc) >= self._params.mastery_threshold

    def observations(self, kc: str) -> int:
        """Number of direct observations recorded for this KC."""
        return self._state[kc].observations

    @property
    def events(self) -> tuple[EvidenceEvent, ...]:
        """The append-only evidence log (read-only view)."""
        return tuple(self._events)

    @property
    def params(self) -> BKTParams:
        """The active, versioned parameter set."""
        return self._params

    def snapshot(self) -> DerivedLearnerState:
        """Serialize derived state (rebuildable from the event log)."""
        return DerivedLearnerState(
            learner_id=self.learner_id,
            graph_version=self._graph.graph_version,
            params_version=self._params.params_version,
            mastery={
                kc: MasteryEstimate(
                    direct=s.direct,
                    inferred=s.inferred,
                    observations=s.observations,
                    last_practiced=s.last_practiced,
                )
                for kc, s in self._state.items()
            },
            misconception_flags=list(self._misconceptions),
        )

    def replay(self, events: list[EvidenceEvent] | None = None) -> "LearnerModelService":
        """Build a fresh service and re-apply events (defaults to this log)."""
        fresh = LearnerModelService(
            self._graph,
            params=self._params,
            assumed_floor_levels=self._floor_levels,
            learner_id=self.learner_id,
        )
        for event in events if events is not None else self._events:
            fresh.apply_event(event)
        return fresh
