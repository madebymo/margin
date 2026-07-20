"""Failure-isolated diagnosis-v2 shadow evaluation for legacy sessions.

The observer is deliberately off the student-serving control path. It receives
only already-scored evidence metadata, compares the next KC selected by the v2
policy with the KC selected by v1, and retains aggregate counters. Raw answers,
prompts, session identifiers, and learner identifiers never enter its metrics.

Legacy probes do not carry reviewed family metadata. The conservative shadow
policy therefore treats all v1 probes for one KC as a single family. A repeated
probe cannot be mistaken for independent confirmation.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import Any

from tutor.orchestrator.diagnosis_v2 import (
    DIAGNOSIS_POLICY_VERSION,
    DiagnosticObservation,
    DiagnosisControllerV2,
    ProbeSelection,
)
from tutor.orchestrator.machine import SessionOrchestrator
from tutor.schemas.kc import GraphDocument
from tutor.schemas.learner import EvidenceEvent

logger = logging.getLogger("tutor.api.diagnosis_shadow")

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_SHADOW_ENV = "TUTOR_ENABLE_DIAGNOSIS_V2_SHADOW"


def diagnosis_shadow_enabled_from_environment() -> bool:
    """Return the explicit opt-in state; shadowing defaults to disabled."""
    return os.environ.get(_SHADOW_ENV, "").strip().lower() in _TRUE_VALUES


@dataclass
class _ShadowEpisode:
    controller: DiagnosisControllerV2
    expected: ProbeSelection | None
    actual_probe_kc: str | None


class DiagnosisV2ShadowObserver:
    """Compare diagnosis-v2 choices with v1 without influencing v1 state."""

    def __init__(
        self,
        graph: GraphDocument,
        *,
        enabled: bool = False,
        max_episodes: int = 500,
    ) -> None:
        self.enabled = enabled
        self._graph = graph
        self._max_episodes = max_episodes
        self._episodes: OrderedDict[str, _ShadowEpisode] = OrderedDict()
        self._metrics: Counter[str] = Counter()
        self._lock = threading.RLock()

    def start(self, session_id: str, orchestrator: SessionOrchestrator) -> None:
        """Start one comparable shadow episode after v1 has issued its first probe."""
        if not self.enabled:
            return
        try:
            with self._lock:
                target_kc = str(orchestrator.summary()["target"])
                controller = DiagnosisControllerV2(
                    self._graph,
                    target_kc,
                    orchestrator.learner,
                )
                expected = controller.next_probe()
                actual = self._actual_probe(orchestrator)
                self._metrics["sessions_started"] += 1
                comparable = self._compare(expected, actual)
                if not comparable:
                    self._metrics["episodes_stopped_after_divergence"] += 1
                    return
                while len(self._episodes) >= self._max_episodes:
                    self._episodes.popitem(last=False)
                    self._metrics["episodes_evicted"] += 1
                self._episodes[session_id] = _ShadowEpisode(
                    controller=controller,
                    expected=expected,
                    actual_probe_kc=actual,
                )
        except Exception as exc:  # noqa: BLE001 - shadowing must never affect v1
            self._record_failure("start", exc)

    def observe_answer(
        self,
        session_id: str,
        event: EvidenceEvent,
        orchestrator: SessionOrchestrator,
    ) -> None:
        """Observe one already-scored v1 answer without receiving its raw text."""
        if not self.enabled:
            return
        try:
            with self._lock:
                episode = self._episodes.get(session_id)
                if episode is None:
                    return
                self._metrics["answers_observed"] += 1
                expected_kc = episode.expected.kc_id if episode.expected else None
                if (
                    expected_kc is None
                    or episode.actual_probe_kc != expected_kc
                    or len(event.kc_ids) != 1
                    or event.kc_ids[0] != expected_kc
                ):
                    self._metrics["off_policy_observations_skipped"] += 1
                    self._metrics["episodes_stopped_after_divergence"] += 1
                    self._episodes.pop(session_id, None)
                    return

                # v1 exposes no reviewed family identity. Grouping by KC is
                # conservative: a repeated legacy prompt can never become a
                # second independent confirmation in the shadow policy.
                family_id = f"legacy-v1.canonical.{expected_kc}"
                if family_id in episode.controller.state.issued_families:
                    self._metrics["family_reuse_observations_rejected"] += 1
                    self._metrics["episodes_stopped_for_untrusted_content"] += 1
                    self._episodes.pop(session_id, None)
                    return

                episode.controller.record_result(
                    DiagnosticObservation(
                        kc_id=expected_kc,
                        family_id=family_id,
                        correct=event.correct,
                        assisted=event.assisted,
                        response_class=event.response_class,
                    )
                )
                self._metrics["eligible_observations"] += 1
                if event.assisted:
                    self._metrics["assisted_observations"] += 1

                expected = episode.controller.next_probe()
                actual = self._actual_probe(orchestrator)
                comparable = self._compare(expected, actual)
                if expected is None and actual is None:
                    self._metrics["episodes_completed_comparably"] += 1
                    self._episodes.pop(session_id, None)
                    return
                if not comparable:
                    self._metrics["episodes_stopped_after_divergence"] += 1
                    self._episodes.pop(session_id, None)
                    return
                episode.expected = expected
                episode.actual_probe_kc = actual
                self._episodes.move_to_end(session_id)
        except Exception as exc:  # noqa: BLE001 - shadowing must never affect v1
            self._record_failure("answer", exc)
            with self._lock:
                self._episodes.pop(session_id, None)

    def note_boundary_failure(self, boundary: str) -> None:
        """Record a wrapper-level failure without accepting exception text."""
        if not self.enabled:
            return
        with self._lock:
            self._metrics["observer_failures"] += 1
            self._metrics[f"{boundary}_failures"] += 1

    def note_unscored_submission(self) -> None:
        """Record an invalid/ungraded v1 submission without treating it as failure."""
        if not self.enabled:
            return
        with self._lock:
            self._metrics["unscored_submissions_ignored"] += 1

    def metrics_snapshot(self) -> dict[str, Any]:
        """Return aggregate, privacy-safe operational metrics."""
        with self._lock:
            counters = dict(sorted(self._metrics.items()))
            comparisons = counters.get("boundary_comparisons", 0)
            matches = counters.get("next_probe_matches", 0)
            return {
                "enabled": self.enabled,
                "policy_version": DIAGNOSIS_POLICY_VERSION,
                "legacy_family_policy": "one_family_per_kc",
                "active_episodes": len(self._episodes),
                "counters": counters,
                "next_probe_match_rate": (
                    matches / comparisons if comparisons else None
                ),
            }

    @staticmethod
    def _actual_probe(orchestrator: SessionOrchestrator) -> str | None:
        if orchestrator.pending_kind != "probe":
            return None
        return orchestrator.pending_kc

    def _compare(
        self,
        expected: ProbeSelection | None,
        actual_kc: str | None,
    ) -> bool:
        self._metrics["boundary_comparisons"] += 1
        expected_kc = expected.kc_id if expected else None
        if expected_kc == actual_kc:
            self._metrics["next_probe_matches"] += 1
            return True
        self._metrics["next_probe_divergences"] += 1
        if expected_kc is None:
            self._metrics["v2_completed_before_v1"] += 1
        elif actual_kc is None:
            self._metrics["v1_completed_before_v2"] += 1
        else:
            self._metrics["different_next_kc"] += 1
        return False

    def _record_failure(self, boundary: str, exc: Exception) -> None:
        with self._lock:
            self._metrics["observer_failures"] += 1
            self._metrics[f"{boundary}_failures"] += 1
        # Log only the exception class. Model or transport exception messages
        # are not needed for this metrics-only observer.
        logger.warning(
            "diagnosis-v2 shadow observer failed boundary=%s error_type=%s",
            boundary,
            type(exc).__name__,
        )
