"""Learner model v2: observation updates separated from learning transitions."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from uuid import UUID

from tutor.learner.params import BKTParams, DEFAULT_PARAMS_V2
from tutor.learner.service import LearnerModelService
from tutor.schemas.common import ResponseClass
from tutor.schemas.kc import GraphDocument
from tutor.schemas.learner import EvidenceEvent


class LearnerModelServiceV2(LearnerModelService):
    """BKT evidence with explicit activity semantics and recency gates."""

    def __init__(
        self,
        graph: GraphDocument,
        params: BKTParams | None = None,
        assumed_floor_levels: set[str] | None = None,
        learner_id: UUID | None = None,
        *,
        as_of: datetime | None = None,
        retention_half_life_days: int = 180,
        confirmation_window_days: int = 90,
    ) -> None:
        super().__init__(
            graph,
            params=params or DEFAULT_PARAMS_V2,
            assumed_floor_levels=assumed_floor_levels,
            learner_id=learner_id,
        )
        self.as_of = as_of or datetime.now(timezone.utc)
        self.retention_half_life_days = retention_half_life_days
        self.confirmation_window_days = confirmation_window_days
        self._priors = {kc: state.direct for kc, state in self._state.items()}

    def apply_event(self, event: EvidenceEvent) -> None:
        """Append evidence, but grant learning only for declared practice."""
        self._events.append(event)
        if event.misconception_id and event.misconception_id not in self._misconceptions:
            self._misconceptions.append(event.misconception_id)
        if len(event.kc_ids) != 1:
            return
        kc = event.kc_ids[0]
        if kc not in self._state:
            raise KeyError(f"unknown kc: {kc}")
        if event.surface == "guided_widget":
            return  # trajectory is retained, but widgets do not change mastery

        state = self._state[kc]
        timestamp = self._aware(event.t)
        if state.last_practiced is not None and timestamp < self._aware(
            state.last_practiced
        ):
            raise ValueError("v2 evidence must be applied in timestamp order per KC")
        prior = self._decay(
            state.direct,
            self._priors[kc],
            self._aware(state.last_practiced) if state.last_practiced else None,
            timestamp,
        )
        if event.surface == "instructional_practice":
            # A lesson transition is logged separately from the response that
            # completed practice. It applies learn once, without treating the
            # practice response itself as assessment evidence.
            state.direct = min(1.0, prior + (1 - prior) * self._params.learn)
            state.last_practiced = timestamp
            return

        class_params = self._params.response_class[event.response_class]
        if event.correct:
            numerator = prior * (1 - class_params.slip)
            denominator = numerator + (1 - prior) * class_params.guess
        else:
            numerator = prior * class_params.slip
            denominator = numerator + (1 - prior) * (1 - class_params.guess)
        posterior = numerator / denominator if denominator else prior
        if event.surface == "legacy":
            # Legacy rows can only nudge the prior. They lack reviewed family
            # identity and therefore cannot carry v2-strength evidence.
            posterior = prior + 0.25 * (posterior - prior)

        # V2 records assistance explicitly. The first two authored hints are
        # conceptual; only a revealing hint sets ``event.assisted``.
        assisted = event.assisted
        if assisted:
            posterior = prior + self._params.assisted_credit * (posterior - prior)
        state.direct = min(1.0, max(0.0, posterior))
        state.observations += 1
        state.last_practiced = timestamp
        if event.correct and not assisted:
            self._propagate_up(kc)

    def aged_probability(self, kc: str, as_of: datetime | None = None) -> float:
        """Age direct belief toward its KC prior using a pinned half-life."""
        state = self._state[kc]
        if state.last_practiced is None:
            return state.direct
        effective_as_of = as_of or self.as_of
        return self._decay(
            state.direct,
            self._priors[kc],
            self._aware(state.last_practiced),
            self._aware(effective_as_of),
        )

    @staticmethod
    def _aware(value: datetime) -> datetime:
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)

    def _decay(
        self,
        probability: float,
        prior: float,
        from_time: datetime | None,
        to_time: datetime,
    ) -> float:
        if from_time is None:
            return probability
        elapsed_days = max(
            0.0,
            (self._aware(to_time) - self._aware(from_time)).total_seconds() / 86400,
        )
        retained = math.pow(0.5, elapsed_days / self.retention_half_life_days)
        return prior + retained * (probability - prior)

    def routing_score(self, kc: str) -> float:
        state = self._state[kc]
        if state.observations > 0:
            return self.aged_probability(kc)
        return max(state.direct, state.inferred)

    def recent_independent_counts(self, kc: str) -> tuple[int, int]:
        """Distinct-family unassisted successes and failures in the recency window."""
        cutoff = self.as_of - timedelta(days=self.confirmation_window_days)
        family_outcomes: dict[str, bool] = {}
        for event in self._events:
            timestamp = event.t
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            if (
                event.kc_ids == [kc]
                and event.family_id
                and not event.assisted
                and timestamp >= cutoff
                and event.surface in {"diagnostic", "checkin"}
            ):
                if (
                    event.correct
                    and event.response_class == ResponseClass.MULTIPLE_CHOICE
                ):
                    continue
                family_outcomes.setdefault(event.family_id, event.correct)
        successes = sum(family_outcomes.values())
        return successes, len(family_outcomes) - successes

    def mastery_status(self, kc: str) -> str:
        successes, failures = self.recent_independent_counts(kc)
        probability = self.routing_score(kc)
        if successes >= 2 and probability >= 0.90:
            return "confirmed_mastered"
        if failures >= 2 and probability <= 0.10:
            return "confirmed_gap"
        return "uncertain"

    def replay(
        self,
        events: list[EvidenceEvent] | None = None,
        *,
        as_of: datetime | None = None,
    ) -> "LearnerModelServiceV2":
        """Rebuild at a fixed session time so resume is deterministic."""
        fresh = LearnerModelServiceV2(
            self._graph,
            params=self._params,
            assumed_floor_levels=self._floor_levels,
            learner_id=self.learner_id,
            as_of=as_of or self.as_of,
            retention_half_life_days=self.retention_half_life_days,
            confirmation_window_days=self.confirmation_window_days,
        )
        source = events if events is not None else list(self._events)
        ordered = sorted(
            enumerate(source),
            key=lambda pair: (self._aware(pair[1].t), pair[0]),
        )
        for _, event in ordered:
            fresh.apply_event(event)
        return fresh
