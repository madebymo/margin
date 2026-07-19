"""Adaptive diagnosis: confidence-bounded selection of where to start teaching.

Not exact frontier reconstruction. Policy (v1.1, deterministic):
- First probe is the target concept itself.
- On a miss: probe the implicated prerequisite if the error analysis names
  one; otherwise binary-search the unresolved hard-ancestor chain of the
  missed node (midpoint by depth).
- On a hit: FIRST confirm any single-observation bad node by re-probing it
  once (slip recovery — a lone wrong answer is weak evidence), then narrow
  toward the deepest known-bad node's unresolved ancestors.
- Leftover budget is spent on verification: probe the shallowest unprobed
  node that would otherwise be taught on prior alone. Gated on at least one
  observed gap, so a passing student still short-circuits after one probe.
- Stop when nothing actionable remains or the probe budget is spent. No node
  is ever probed more than twice.

Output: a frontier (deepest observed-bad nodes with no bad hard ancestor) and
a teaching path (topological order of unmastered nodes in the hard-ancestor
subgraph, ending at the target). Unprobed low-prior nodes stay "uncertain" and
are settled during the teach loop.
"""

from collections import defaultdict

from pydantic import BaseModel

from tutor.graph import service as graph_service
from tutor.learner.service import LearnerModelService
from tutor.schemas.kc import GraphDocument


class ProbeResult(BaseModel):
    """The scored outcome of one diagnostic probe."""

    kc_id: str
    correct: bool
    implicated_prereq: str | None = None


class DiagnosisController:
    """Drives probe selection over the hard-ancestor subgraph of the target."""

    def __init__(
        self,
        graph: GraphDocument,
        target_kc: str,
        learner: LearnerModelService,
        probe_budget: int = 8,
    ) -> None:
        if target_kc not in graph.node_ids():
            raise KeyError(f"unknown kc: {target_kc}")
        self._hard = graph_service.ancestor_subgraph(graph, target_kc, hard_only=True)
        self._target = target_kc
        self._learner = learner
        self._budget = probe_budget
        self._issued = 0
        self._last: ProbeResult | None = None
        self._finished = False

        self._preds: dict[str, list[str]] = defaultdict(list)
        for edge in self._hard.edges:
            self._preds[edge.to_kc].append(edge.from_kc)
        order = graph_service.topological_order(self._hard)
        self._depth: dict[str, int] = {}
        self._ancestors: dict[str, set[str]] = {}
        for kc in order:
            self._depth[kc] = max(
                (self._depth[p] + 1 for p in self._preds[kc]), default=0
            )
            acc: set[str] = set()
            for pred in self._preds[kc]:
                acc.add(pred)
                acc |= self._ancestors[pred]
            self._ancestors[kc] = acc

    # -- probe selection ------------------------------------------------------

    @property
    def probes_issued(self) -> int:
        """Number of probes issued so far."""
        return self._issued

    @property
    def finished(self) -> bool:
        """Whether diagnosis has terminated (localized or budget spent)."""
        return self._finished

    def next_probe_kc(self) -> str | None:
        """Select the next KC to probe, or None when diagnosis is finished."""
        if self._finished:
            return None
        if self._issued >= self._budget:
            self._finished = True
            return None
        kc = self._select_probe()
        if kc is None:
            self._finished = True
            return None
        self._issued += 1
        return kc

    def record_result(self, result: ProbeResult) -> None:
        """Record the scored outcome of the most recent probe."""
        self._last = result

    def _known_bad(self) -> list[str]:
        return [
            kc
            for kc in self._hard.node_ids()
            if self._learner.observations(kc) > 0 and not self._learner.is_mastered(kc)
        ]

    def _unresolved_ancestors(self, kc: str) -> list[str]:
        """Unprobed, not-assumed-mastered hard ancestors, shallowest first."""
        candidates = [
            ancestor
            for ancestor in self._ancestors[kc]
            if self._learner.observations(ancestor) == 0
            and not self._learner.is_mastered(ancestor)
        ]
        return sorted(candidates, key=lambda k: (self._depth[k], k))

    def _select_probe(self) -> str | None:
        if self._last is None:
            return self._target
        if not self._last.correct:
            candidate = self._miss_candidate()
        else:
            candidate = self._confirmation_candidate() or self._drill_candidate()
        if candidate is None:
            candidate = self._confirmation_candidate() or self._verification_candidate()
        return candidate

    def _miss_candidate(self) -> str | None:
        """After a miss: implicated prerequisite, else midpoint of the missed chain."""
        missed = self._last.kc_id if self._last else self._target
        implicated = self._last.implicated_prereq if self._last else None
        if (
            implicated is not None
            and implicated in self._hard.node_ids()
            and self._learner.observations(implicated) == 0
            and not self._learner.is_mastered(implicated)
        ):
            return implicated
        chain = self._unresolved_ancestors(missed)
        if not chain:
            return None  # gap localized at the missed node
        return chain[len(chain) // 2]

    def _drill_candidate(self) -> str | None:
        """After a hit: narrow toward the deepest known-bad node's ancestors."""
        known_bad = self._known_bad()
        if not known_bad:
            return None  # nothing bad observed
        deepest_bad = max(known_bad, key=lambda k: (self._depth[k], k))
        chain = self._unresolved_ancestors(deepest_bad)
        if not chain:
            return None  # localization exhausted
        return chain[len(chain) // 2]

    def _confirmation_candidate(self) -> str | None:
        """Re-probe single-observation bad nodes once before trusting them.

        A lone wrong answer may be a slip; a confirmation probe either recovers
        the node (Bayes update pushes it back above threshold) or cements it.
        Shallowest first, since shallow bad nodes define the frontier.
        """
        singles = [
            kc
            for kc in self._hard.node_ids()
            if self._learner.observations(kc) == 1 and not self._learner.is_mastered(kc)
        ]
        if not singles:
            return None
        return min(singles, key=lambda k: (self._depth[k], k))

    def _verification_candidate(self) -> str | None:
        """Spend leftover budget on the shallowest unprobed would-be lesson node.

        Converts prior-only uncertainty at the front of the teaching path into
        direct evidence (better next-KC accuracy, less overteaching). Gated on
        an observed gap so passing students are not burdened.
        """
        if not self._known_bad():
            return None
        unverified = [
            kc
            for kc in self._hard.node_ids()
            if self._learner.observations(kc) == 0 and not self._learner.is_mastered(kc)
        ]
        if not unverified:
            return None
        return min(unverified, key=lambda k: (self._depth[k], k))

    # -- outputs ---------------------------------------------------------------

    def frontier(self) -> list[str]:
        """Deepest observed-bad nodes with no observed-bad hard ancestor."""
        bad = set(self._known_bad())
        return sorted(kc for kc in bad if not (self._ancestors[kc] & bad))

    def plan_path(self) -> list[str]:
        """Topological order of unmastered nodes in the hard subgraph, target last."""
        if self._learner.is_mastered(self._target):
            return []
        unmastered = {
            kc for kc in self._hard.node_ids() if not self._learner.is_mastered(kc)
        }
        return graph_service.topological_order(self._hard, unmastered)
