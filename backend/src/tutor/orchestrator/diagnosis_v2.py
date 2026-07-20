"""Resumable, confirmation-first diagnostic policy for trustworthy sessions.

The policy never routes from a single response.  Any observation that can
change the learner's route is confirmed with an unexposed item family, then
remaining probe budget is allocated by noisy-test information gain weighted
by the hard-prerequisite DAG.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Literal

from pydantic import BaseModel, Field

from tutor.graph import service as graph_service
from tutor.learner.service import LearnerModelService
from tutor.schemas.common import ResponseClass
from tutor.schemas.kc import GraphDocument

DIAGNOSIS_POLICY_VERSION = "diagnosis-v2.1"
PINNED_IMPACT_LAMBDA = 0.0
PINNED_IMPACT_DECAY = 0.25


class DiagnosticObservation(BaseModel):
    """One independently scored diagnostic response."""

    kc_id: str
    family_id: str
    correct: bool
    assisted: bool = False
    response_class: ResponseClass = ResponseClass.SYMBOLIC_ENTRY
    implicated_prereq: str | None = None


class DiagnosisState(BaseModel):
    """Serializable state required to resume policy selection exactly."""

    target_kc: str
    probe_budget: int = Field(ge=1)
    probes_issued: int = Field(ge=0, default=0)
    observations: list[DiagnosticObservation] = Field(default_factory=list)
    prior_probabilities: dict[str, float] = Field(default_factory=dict)
    issued_families: list[str] = Field(default_factory=list)
    pending_confirmation_kc: str | None = None
    suspected_prereqs: list[str] = Field(default_factory=list)
    stop_reason: str | None = None
    impact_lambda: float = Field(default=PINNED_IMPACT_LAMBDA, ge=0)
    impact_decay: float = Field(default=PINNED_IMPACT_DECAY, ge=0, le=1)


class ProbeSelection(BaseModel):
    """The next KC and the reason it has priority."""

    kc_id: str
    reason: Literal[
        "target_first",
        "independent_confirmation",
        "verify_suspected_prerequisite",
        "information_gain",
    ]


class LearningPlanStep(BaseModel):
    """An honest post-diagnosis action; uncertainty is not called a gap."""

    kind: Literal["teach_confirmed_gap", "verify_uncertain", "practice_target"]
    kc_id: str


class DiagnosisControllerV2:
    """Purely deterministic diagnosis policy with serializable state."""

    def __init__(
        self,
        graph: GraphDocument,
        target_kc: str,
        learner: LearnerModelService,
        probe_budget: int = 8,
        *,
        impact_lambda: float = PINNED_IMPACT_LAMBDA,
        impact_decay: float = PINNED_IMPACT_DECAY,
        state: DiagnosisState | None = None,
    ) -> None:
        if target_kc not in graph.node_ids():
            raise KeyError(f"unknown kc: {target_kc}")
        self._hard = graph_service.ancestor_subgraph(graph, target_kc, hard_only=True)
        self._target = target_kc
        self._learner = learner
        self.state = state or DiagnosisState(
            target_kc=target_kc,
            probe_budget=probe_budget,
            impact_lambda=impact_lambda,
            impact_decay=impact_decay,
        )
        if self.state.target_kc != target_kc:
            raise ValueError("diagnosis state target does not match controller target")
        self._impact_lambda = self.state.impact_lambda
        self._impact_decay = self.state.impact_decay

        self._preds: dict[str, list[str]] = defaultdict(list)
        self._succs: dict[str, list[str]] = defaultdict(list)
        for edge in self._hard.edges:
            self._preds[edge.to_kc].append(edge.from_kc)
            self._succs[edge.from_kc].append(edge.to_kc)
        self._order = graph_service.topological_order(self._hard)
        if not self.state.prior_probabilities:
            self.state.prior_probabilities = {
                kc: learner.routing_score(kc) for kc in self._hard.node_ids()
            }
        self._depth: dict[str, int] = {}
        for kc in self._order:
            self._depth[kc] = max(
                (self._depth[pred] + 1 for pred in self._preds[kc]), default=0
            )

    @property
    def probes_issued(self) -> int:
        return self.state.probes_issued

    @property
    def finished(self) -> bool:
        return self.state.stop_reason is not None

    def observations_for(self, kc_id: str) -> list[DiagnosticObservation]:
        """Return independent, unassisted observations for one KC."""
        seen: set[str] = set()
        result: list[DiagnosticObservation] = []
        for observation in self.state.observations:
            if (
                observation.kc_id == kc_id
                and not observation.assisted
                and observation.family_id not in seen
            ):
                seen.add(observation.family_id)
                result.append(observation)
        return result

    def attempts_for(self, kc_id: str) -> list[DiagnosticObservation]:
        """Return every distinct-family attempt, including assisted attempts."""
        seen: set[str] = set()
        result: list[DiagnosticObservation] = []
        for observation in self.state.observations:
            if observation.kc_id == kc_id and observation.family_id not in seen:
                seen.add(observation.family_id)
                result.append(observation)
        return result

    def probability(self, kc_id: str) -> float:
        """Posterior mastery probability with no practice-learning transition."""
        probability = self.state.prior_probabilities[kc_id]
        for observation in self.observations_for(kc_id):
            params = self._learner.params.response_class[observation.response_class]
            if observation.correct:
                numerator = probability * (1 - params.slip)
                denominator = numerator + (1 - probability) * params.guess
            else:
                numerator = probability * params.slip
                denominator = numerator + (1 - probability) * (1 - params.guess)
            if denominator:
                probability = numerator / denominator
        return probability

    def status(self, kc_id: str) -> Literal["confirmed_mastered", "confirmed_gap", "uncertain"]:
        observations = self.observations_for(kc_id)
        correct = sum(
            item.correct and item.response_class != ResponseClass.MULTIPLE_CHOICE
            for item in observations
        )
        wrong = sum(not item.correct for item in observations)
        probability = self.probability(kc_id)
        if correct >= 2 and probability >= 0.90:
            return "confirmed_mastered"
        if wrong >= 2 and probability <= 0.10:
            return "confirmed_gap"
        return "uncertain"

    def next_probe(self) -> ProbeSelection | None:
        """Select the next probe without choosing its assessment family."""
        if self.finished:
            return None
        if not self.state.observations:
            if self.state.probes_issued >= self.state.probe_budget:
                self.state.stop_reason = "budget_exhausted"
                return None
            return self._issue(self._target, "target_first")

        if self.status(self._target) == "confirmed_mastered":
            self.state.stop_reason = "target_confirmed_mastered"
            return None
        if (
            len(self.observations_for(self._target)) >= 3
            and self.status(self._target) == "uncertain"
        ):
            # The third independent family completes confirmation even when
            # the outcome remains uncertain. Verify with fresh content instead
            # of turning a single target slip into broad ancestor probing.
            self.state.stop_reason = "target_conflict_uncertain"
            return None
        if self.state.probes_issued >= self.state.probe_budget:
            self.state.stop_reason = "budget_exhausted"
            return None

        confirmation = self._confirmation_candidate()
        if confirmation is not None:
            self.state.pending_confirmation_kc = confirmation
            return self._issue(confirmation, "independent_confirmation")
        self.state.pending_confirmation_kc = None

        while self.state.suspected_prereqs:
            suspected = self.state.suspected_prereqs.pop(0)
            if self.status(suspected) == "uncertain" and len(
                self.attempts_for(suspected)
            ) < 3:
                return self._issue(suspected, "verify_suspected_prerequisite")

        candidate = self._information_gain_candidate()
        if candidate is None:
            self.state.stop_reason = "localized"
            return None
        return self._issue(candidate, "information_gain")

    def _issue(self, kc_id: str, reason: str) -> ProbeSelection:
        self.state.probes_issued += 1
        return ProbeSelection(kc_id=kc_id, reason=reason)

    def record_result(self, observation: DiagnosticObservation) -> None:
        """Record one result, rejecting family reuse and invalid implications."""
        if observation.kc_id not in self._hard.node_ids():
            raise ValueError("observation is outside the target hard-ancestor graph")
        if observation.family_id in self.state.issued_families:
            raise ValueError("diagnostic item family has already been used")
        self.state.issued_families.append(observation.family_id)
        self.state.observations.append(observation)
        implicated = observation.implicated_prereq
        if (
            not observation.correct
            and implicated is not None
            and implicated in self._ancestors_of(observation.kc_id)
            and implicated not in self.state.suspected_prereqs
        ):
            self.state.suspected_prereqs.append(implicated)

    def _confirmation_candidate(self) -> str | None:
        """Require two agreeing families, or a third family after a conflict."""
        candidates: list[str] = []
        for kc in self._hard.node_ids():
            observations = self.observations_for(kc)
            attempts = len(self.attempts_for(kc))
            if not observations and 0 < attempts < 3:
                candidates.append(kc)
            elif len(observations) == 1 and attempts < 3:
                candidates.append(kc)
            elif (
                len(observations) == 2
                and attempts < 3
                and observations[0].correct != observations[1].correct
            ):
                candidates.append(kc)
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda kc: (
                0 if kc == self._target else 1,
                -self._impact(kc),
                -self._depth[kc],
                kc,
            ),
        )

    def _information_gain_candidate(self) -> str | None:
        candidates = [
            kc
            for kc in self._hard.node_ids()
            if self.status(kc) == "uncertain" and len(self.attempts_for(kc)) < 3
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda kc: (
                self._mutual_information(kc) * self._impact(kc),
                self._depth[kc],
                kc,
            ),
        )

    @staticmethod
    def _entropy(probability: float) -> float:
        if probability <= 0 or probability >= 1:
            return 0.0
        return -probability * math.log2(probability) - (1 - probability) * math.log2(
            1 - probability
        )

    def _mutual_information(self, kc_id: str) -> float:
        p_mastered = self.probability(kc_id)
        params = self._learner.params.response_class[ResponseClass.SYMBOLIC_ENTRY]
        p_correct = p_mastered * (1 - params.slip) + (1 - p_mastered) * params.guess
        if p_correct <= 0 or p_correct >= 1:
            return 0.0
        posterior_correct = p_mastered * (1 - params.slip) / p_correct
        p_wrong = 1 - p_correct
        posterior_wrong = p_mastered * params.slip / p_wrong
        return self._entropy(p_mastered) - (
            p_correct * self._entropy(posterior_correct)
            + p_wrong * self._entropy(posterior_wrong)
        )

    def _impact(self, kc_id: str) -> float:
        total = 0.0
        queue: deque[tuple[str, int]] = deque([(kc_id, 0)])
        visited = {kc_id}
        while queue:
            current, distance = queue.popleft()
            for child in self._succs[current]:
                if child in visited:
                    continue
                visited.add(child)
                next_distance = distance + 1
                if self.status(child) == "uncertain":
                    total += self._impact_decay ** max(0, next_distance - 1)
                queue.append((child, next_distance))
        return 1 + self._impact_lambda * total

    def _ancestors_of(self, kc_id: str) -> set[str]:
        return (
            graph_service.ancestor_subgraph(self._hard, kc_id, hard_only=True).node_ids()
            - {kc_id}
        )

    def frontier(self) -> list[str]:
        """Confirmed gaps having no confirmed-gap hard ancestor."""
        gaps = {
            kc for kc in self._hard.node_ids() if self.status(kc) == "confirmed_gap"
        }
        return sorted(kc for kc in gaps if not (self._ancestors_of(kc) & gaps))

    def learner_summary(self) -> dict[str, list[str]]:
        """Three-way classification for honest student-facing summaries."""
        summary = {
            "confirmed_mastered": [],
            "confirmed_gaps": [],
            "uncertain": [],
        }
        for kc in sorted(self._hard.node_ids()):
            status = self.status(kc)
            summary[
                "confirmed_gaps" if status == "confirmed_gap" else status
            ].append(kc)
        return summary

    def learning_plan(self) -> list[LearningPlanStep]:
        """Teach only confirmed gaps; verify uncertainty and then practice target."""
        if self.status(self._target) == "confirmed_mastered":
            return []
        gaps = {
            kc for kc in self._hard.node_ids() if self.status(kc) == "confirmed_gap"
        }
        steps = [
            LearningPlanStep(kind="teach_confirmed_gap", kc_id=kc)
            for kc in self._order
            if kc in gaps and kc != self._target
        ]
        target_status = self.status(self._target)
        if target_status == "confirmed_gap":
            steps.append(
                LearningPlanStep(kind="teach_confirmed_gap", kc_id=self._target)
            )
        else:
            steps.append(LearningPlanStep(kind="verify_uncertain", kc_id=self._target))
        steps.append(LearningPlanStep(kind="practice_target", kc_id=self._target))
        return steps
