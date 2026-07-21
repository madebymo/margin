"""Atomic in-process lifecycle for v2 tutoring sessions.

Each session has its own re-entrant lock, so unrelated learners can progress
concurrently while actions for one learner are serialized.  Mutations run on
a deep copy and are swapped into the live handle only after durable commit.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import threading
from collections import Counter, OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Protocol
from uuid import UUID, uuid4

from pydantic import BaseModel

from tutor.api.v2_metrics import MetricsSink, V2MetricDimensions
from tutor.api.v2_schemas import (
    AnswerAction,
    ContentModeView,
    GoalView,
    HintAction,
    LearnerSummaryView,
    PendingHintView,
    PendingView,
    ProgressView,
    ResetResponse,
    ResetSessionV2Request,
    SessionAction,
    SessionProfile,
    SessionView,
    TextFallbackAction,
    TranscriptEntry,
    WidgetAttemptAction,
)
from tutor.schemas.assessment import PlotPromptSegment
from tutor.orchestrator.machine import Interaction

logger = logging.getLogger("tutor.api.v2.operations")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


_RESUME_WINDOW = timedelta(days=30)


class SessionConflict(RuntimeError):
    """A typed, safe conflict that includes the authoritative current view."""

    def __init__(self, code: str, message: str, view: SessionView | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.view = view


class SessionUnavailable(RuntimeError):
    """A retryable persistence or restoration failure."""


class SessionIntegrityError(RuntimeError):
    """A fail-closed commit invariant detected corrupt or incomplete state."""


class SessionRateLimited(RuntimeError):
    """A bounded episode has exhausted its mutation-storage allowance."""


class DurableReceiptReplay(RuntimeError):
    """An identical request was already committed by another process."""

    def __init__(self, view: SessionView) -> None:
        super().__init__("request already committed")
        self.view = view


class DurableResetReplay(RuntimeError):
    """An identical reset/replacement was committed by another process."""

    def __init__(self, response: ResetResponse) -> None:
        super().__init__("reset already committed")
        self.response = response


class ResumeTokenExpired(KeyError):
    """A process-local resume token was known but passed its rolling expiry."""


class V2PersistencePort(Protocol):
    """Minimal transactional persistence seam used by the in-process store."""

    def create_session(
        self,
        handle: V2SessionHandle,
        token_hash: str,
        request_id: UUID,
        *,
        replace_token_hash: str | None = None,
        replace_session_id: str | None = None,
        replace_expected_revision: int | None = None,
    ) -> dict[str, Any] | None:
        ...

    def commit_action(
        self,
        *,
        handle: V2SessionHandle,
        previous_revision: int,
        request_id: UUID,
        payload_hash: str,
        request_payload: dict[str, Any],
        response_payload: dict[str, Any],
        new_transcript: list[TranscriptEntry],
        new_events: list[Any],
        new_exposures: list[Any],
        widget_attempt: dict[str, Any] | None,
        token_hash: str,
    ) -> dict[str, Any] | None:
        ...

    def touch_resume(self, token_hash: str) -> bool:
        ...

    def replay_create(
        self, *, request_id: UUID, payload_hash: str
    ) -> dict[str, Any] | None:
        ...

    def revoke_token(self, token_hash: str) -> None:
        ...

    def commit_reset(
        self,
        *,
        handle: V2SessionHandle,
        token_hash: str,
        request_id: UUID,
        payload_hash: str,
        request_payload: dict[str, Any],
        response_payload: dict[str, Any],
        expected_revision: int,
        pending_key: str | None,
        replacement: V2SessionHandle,
        replacement_token_hash: str,
    ) -> dict[str, Any] | None:
        ...

    def replay_reset(
        self,
        *,
        token_hash: str,
        request_id: UUID,
        payload_hash: str,
    ) -> dict[str, Any] | None:
        ...

    def recover_create(self, request_id: UUID) -> str | None:
        ...

    def recover_reset_rotation(
        self, token_hash: str, request_id: UUID
    ) -> str | None:
        ...


@dataclass
class MutationReceipt:
    payload_hash: str
    response: SessionView
    request_payload: dict[str, Any]


@dataclass
class ResetReceipt:
    payload_hash: str
    response: ResetResponse
    created_at: datetime = field(default_factory=utcnow)


@dataclass
class V2SessionHandle:
    """All mutable state for one live session."""

    session_id: str
    learner_id: str
    orchestrator: Any
    goal: GoalView
    profile: SessionProfile
    context: str | None
    content_mode: ContentModeView
    durability: str
    started_at: datetime
    updated_at: datetime
    # Keep the current anonymous learner's reset chain independently of the
    # bounded live-session cache.  Otherwise FIFO eviction of an old episode
    # would also erase the in-memory rate-limit history and permit an
    # unbounded reset loop in memory-only deployments.
    episode_starts: tuple[datetime, ...] = field(default_factory=tuple)
    revision: int = 0
    transcript: list[TranscriptEntry] = field(default_factory=list)
    receipts: OrderedDict[str, MutationReceipt] = field(default_factory=OrderedDict)
    token_hashes: set[str] = field(default_factory=set)
    metric_dimensions: V2MetricDimensions | None = None
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)


class V2SessionStore:
    """FIFO-bounded store with per-session atomic action application."""

    def __init__(
        self,
        *,
        graph_nodes: dict[str, Any],
        persistence: V2PersistencePort | None = None,
        metrics_sink: MetricsSink | None = None,
        metric_dimensions: V2MetricDimensions | None = None,
        metric_dimensions_resolver: Callable[[Any], V2MetricDimensions] | None = None,
        max_sessions: int = 500,
        max_receipts_per_session: int = 256,
        max_episodes_per_learner: int = 32,
    ) -> None:
        self._sessions: OrderedDict[str, V2SessionHandle] = OrderedDict()
        self._tokens: dict[str, tuple[str, datetime]] = {}
        self._reset_receipts: dict[str, dict[str, ResetReceipt]] = {}
        self._metrics: Counter[str] = Counter()
        self._item_metrics: Counter[str] = Counter()
        self._lock = threading.RLock()
        self._graph_nodes = graph_nodes
        self._persistence = persistence
        self._metrics_sink = metrics_sink
        self._metric_dimensions = metric_dimensions or V2MetricDimensions(
            graph_version="unknown",
            item_bank_version="unknown",
            pedagogy_catalog_version="unknown",
            policy_versions=(),
            learner_parameter_version="unknown",
            capability_manifest_version="unknown",
        )
        self._metric_dimensions_resolver = metric_dimensions_resolver
        self._max_sessions = max_sessions
        self._max_receipts = max_receipts_per_session
        self._max_episodes_per_learner = max_episodes_per_learner

    def create(
        self,
        *,
        orchestrator: Any,
        goal: GoalView,
        profile: SessionProfile,
        context: str | None,
        content_mode: ContentModeView,
        interactions: list[Interaction],
        token_hash: str,
        request_id: UUID,
        request_payload: dict[str, Any],
        session_id: str | None = None,
        replace_handle: V2SessionHandle | None = None,
        replace_token_hash: str | None = None,
    ) -> V2SessionHandle:
        """Install a freshly begun session and persist its initial checkpoint."""
        replacement_lock = replace_handle.lock if replace_handle is not None else nullcontext()
        with replacement_lock:
            if replace_handle is not None:
                current = self.view(replace_handle)
                if current.phase not in ("done", "stopped"):
                    raise SessionConflict(
                        "active_session_exists",
                        "reset or finish the current session before starting another",
                        current,
                    )
                if replace_token_hash is None or not self.owns(
                    replace_handle.session_id, replace_token_hash
                ):
                    raise SessionConflict(
                        "session_revoked",
                        "the anonymous session token is no longer active",
                        current,
                    )
                self._enforce_episode_quota(replace_handle)

            now = utcnow()
            learner_id = str(getattr(orchestrator.learner, "learner_id", uuid4()))
            authoritative_session_id = session_id or uuid4().hex
            bind_episode = getattr(orchestrator, "bind_episode_id", None)
            if callable(bind_episode):
                bind_episode(authoritative_session_id)
            handle = V2SessionHandle(
                session_id=authoritative_session_id,
                learner_id=learner_id,
                orchestrator=orchestrator,
                goal=goal,
                profile=profile,
                context=context,
                content_mode=content_mode,
                durability="durable" if self._persistence is not None else "memory_only",
                started_at=now,
                updated_at=now,
                episode_starts=self._next_episode_starts(replace_handle, now),
                revision=0,
                transcript=self._interaction_entries([], interactions),
                token_hashes={token_hash},
                metric_dimensions=self._resolve_metric_dimensions(orchestrator),
            )
            self._replace_visible_snapshot(
                handle.orchestrator,
                context=handle.context,
                transcript=handle.transcript,
            )
            initial = self.view(handle)
            create_hash = self._payload_hash(request_payload)
            handle.receipts[str(request_id)] = MutationReceipt(
                payload_hash=create_hash,
                response=initial,
                request_payload=request_payload,
            )
            if self._persistence is not None:
                try:
                    persisted = self._persistence.create_session(
                        handle,
                        token_hash,
                        request_id,
                        replace_token_hash=replace_token_hash,
                        replace_session_id=(
                            replace_handle.session_id if replace_handle else None
                        ),
                        replace_expected_revision=(
                            replace_handle.revision if replace_handle else None
                        ),
                    )
                except SessionConflict:
                    raise
                except SessionRateLimited:
                    self.record_metric("episode_resets_rate_limited")
                    raise
                except Exception as exc:
                    raise SessionUnavailable("could not durably create session") from exc
                if persisted is not None:
                    raise DurableReceiptReplay(SessionView.model_validate(persisted))
            terminal_rollover = replace_handle is not None
            with self._lock:
                if replace_handle is not None and replace_token_hash is not None:
                    self._tokens.pop(replace_token_hash, None)
                    replace_handle.token_hashes.discard(replace_token_hash)
                while len(self._sessions) >= self._max_sessions:
                    evicted_id, evicted = self._sessions.popitem(last=False)
                    for old_hash in evicted.token_hashes:
                        token = self._tokens.get(old_hash)
                        if token is not None and token[0] == evicted_id:
                            del self._tokens[old_hash]
                self._sessions[handle.session_id] = handle
                self._tokens[token_hash] = (
                    handle.session_id,
                    utcnow() + _RESUME_WINDOW,
                )
            self.record_metric(
                "sessions_created",
                metric_dimensions=handle.metric_dimensions,
            )
            if terminal_rollover:
                self.record_metric(
                    "terminal_rollovers_committed",
                    metric_dimensions=handle.metric_dimensions,
                )
            return handle

    def get(self, session_id: str) -> V2SessionHandle:
        with self._lock:
            try:
                return self._sessions[session_id]
            except KeyError:
                raise KeyError(f"unknown session: {session_id}") from None

    def resolve_token(self, token_hash: str) -> V2SessionHandle:
        expired = False
        with self._lock:
            token = self._tokens.get(token_hash)
            if token is not None and token[1] <= utcnow():
                del self._tokens[token_hash]
                expired = True
                token = None
        if expired:
            raise ResumeTokenExpired("expired resume token")
        if token is None:
            raise KeyError("unknown resume token")
        return self.get(token[0])

    def repeat_create(
        self, handle: V2SessionHandle, request_id: UUID, request_payload: dict[str, Any]
    ) -> SessionView | None:
        """Return an idempotent create response, or reject payload reuse."""
        with handle.lock:
            receipt = handle.receipts.get(str(request_id))
            if receipt is None:
                return None
            if receipt.payload_hash != self._payload_hash(request_payload):
                raise SessionConflict(
                    "idempotency_conflict",
                    "request_id was already used with a different payload",
                    self.view(handle),
                )
            self.record_metric(
                "create_replays",
                metric_dimensions=handle.metric_dimensions,
            )
            return receipt.response.model_copy(deep=True)

    def repeat_create_without_token(
        self, request_id: UUID, request_payload: dict[str, Any]
    ) -> tuple[V2SessionHandle, SessionView] | None:
        """Replay creation when the first response (and cookie) was lost."""
        request_key = str(request_id)
        payload_hash = self._payload_hash(request_payload)
        with self._lock:
            handles = list(self._sessions.values())
        for handle in handles:
            with handle.lock:
                receipt = handle.receipts.get(request_key)
                if receipt is None:
                    continue
                if receipt.payload_hash != payload_hash:
                    raise SessionConflict(
                        "idempotency_conflict",
                        "request_id was already used with a different payload",
                        self.view(handle),
                    )
                if receipt.request_payload.get("type") != "create":
                    raise SessionConflict(
                        "idempotency_conflict",
                        "request_id was already used for another mutation",
                        self.view(handle),
                    )
                self.record_metric(
                    "create_replays",
                    metric_dimensions=handle.metric_dimensions,
                )
                return handle, receipt.response.model_copy(deep=True)
        return None

    def metrics_snapshot(self) -> dict[str, Any]:
        """Privacy-safe counters for rollout gates and operational alerts."""
        with self._lock:
            counters = dict(self._metrics)
            resume_attempts = counters.get("resume_eligible_attempts", 0)
            action_requests = counters.get("action_requests", 0)
            return {
                "counters": counters,
                "actions_by_item_id": dict(self._item_metrics),
                "resume_outcomes": {
                    "cookie_attempts": counters.get("resume_cookie_attempts", 0),
                    "eligible_attempts": resume_attempts,
                    "eligible_failures": counters.get(
                        "resume_eligible_failures", 0
                    ),
                    "no_cookie": counters.get("resume_no_cookie", 0),
                    "invalid": counters.get("resume_invalid", 0),
                    "expired": counters.get("resume_expired", 0),
                    "session_mismatch": counters.get(
                        "resume_session_mismatch", 0
                    ),
                    "restore_failures": counters.get("resume_restore_failures", 0),
                    "refresh_failures": counters.get("resume_refresh_failures", 0),
                    "successes": counters.get("resume_successes", 0),
                },
                "rollout_gates": {
                    "resume_success_rate": (
                        counters.get("resume_successes", 0) / resume_attempts
                        if resume_attempts
                        else None
                    ),
                    "action_5xx_rate": (
                        counters.get("action_5xx", 0) / action_requests
                        if action_requests
                        else None
                    ),
                    "duplicate_advances_detected": counters.get(
                        "duplicate_advances_detected", 0
                    ),
                    "missing_evidence_detected": counters.get(
                        "missing_evidence_detected", 0
                    ),
                    "commit_integrity_failures": counters.get(
                        "commit_integrity_failures", 0
                    ),
                },
            }

    def record_metric(
        self,
        name: str,
        amount: int = 1,
        *,
        item_id: str | None = None,
        orchestrator: Any | None = None,
        metric_dimensions: V2MetricDimensions | None = None,
    ) -> None:
        """Record locally and best-effort export one privacy-safe counter."""
        if amount < 1:
            raise ValueError("metric increments must be positive")
        with self._lock:
            self._metrics[name] += amount
            if item_id is not None:
                self._item_metrics[item_id] += amount
        if self._metrics_sink is None:
            return
        resolved_dimensions = metric_dimensions or self._metric_dimensions
        if metric_dimensions is None and orchestrator is not None:
            try:
                resolved_dimensions = V2MetricDimensions.from_orchestrator(
                    orchestrator,
                    fallback=self._metric_dimensions,
                )
            except Exception:  # noqa: BLE001 - retain safe active-release fallback
                resolved_dimensions = self._metric_dimensions
        dimensions = resolved_dimensions.as_labels()
        if item_id is not None:
            dimensions["item_id"] = item_id
        try:
            self._metrics_sink.increment(
                name,
                amount,
                dimensions=dimensions,
            )
        except Exception as exc:  # noqa: BLE001 - telemetry cannot break tutoring
            with self._lock:
                self._metrics["metrics_export_failures"] += 1
            logger.warning(
                "v2 metric export failed metric=%s error_type=%s",
                name,
                type(exc).__name__,
            )

    def _resolve_metric_dimensions(self, orchestrator: Any) -> V2MetricDimensions:
        if self._metric_dimensions_resolver is not None:
            try:
                resolved = self._metric_dimensions_resolver(orchestrator)
                if not isinstance(resolved, V2MetricDimensions):
                    raise TypeError("metric dimension resolver returned an invalid value")
                return resolved
            except Exception as exc:  # noqa: BLE001 - telemetry cannot block sessions
                logger.warning(
                    "v2 metric pin resolution failed error_type=%s",
                    type(exc).__name__,
                )
        try:
            return V2MetricDimensions.from_orchestrator(
                orchestrator,
                fallback=self._metric_dimensions,
            )
        except Exception:  # noqa: BLE001 - retain safe active-release fallback
            return self._metric_dimensions

    def restore(
        self,
        *,
        orchestrator: Any,
        checkpoint: dict[str, Any],
        receipts: list[dict[str, Any]],
        token_hash: str,
    ) -> V2SessionHandle:
        """Reinstall a durably checkpointed session after a process restart."""
        view = SessionView.model_validate(checkpoint["session_view"])
        bind_episode = getattr(orchestrator, "bind_episode_id", None)
        if callable(bind_episode):
            bind_episode(view.session_id)
        handle = V2SessionHandle(
            session_id=view.session_id,
            learner_id=str(checkpoint["learner_id"]),
            orchestrator=orchestrator,
            goal=GoalView.model_validate(checkpoint["goal"]),
            profile=SessionProfile.model_validate(checkpoint["profile"]),
            context=checkpoint.get("context"),
            content_mode=ContentModeView.model_validate(checkpoint["content_mode"]),
            durability="durable",
            started_at=view.started_at,
            updated_at=view.updated_at,
            # Durable persistence remains authoritative for the full rolling
            # count after process recovery.  Retain the restored episode here
            # so the local guard still has the strongest history available.
            episode_starts=(view.started_at,),
            revision=view.revision,
            transcript=list(view.transcript),
            token_hashes={token_hash},
            metric_dimensions=self._resolve_metric_dimensions(orchestrator),
        )
        for receipt_data in receipts:
            response = SessionView.model_validate(receipt_data["response_payload"])
            handle.receipts[str(receipt_data["request_id"])] = MutationReceipt(
                payload_hash=str(receipt_data["payload_hash"]),
                response=response,
                request_payload=dict(receipt_data["request_payload"]),
            )
        with self._lock:
            self._sessions[handle.session_id] = handle
            self._sessions.move_to_end(handle.session_id)
            self._tokens[token_hash] = (
                handle.session_id,
                utcnow() + _RESUME_WINDOW,
            )
            while len(self._sessions) > self._max_sessions:
                evicted_id, evicted = self._sessions.popitem(last=False)
                for old_hash in evicted.token_hashes:
                    token = self._tokens.get(old_hash)
                    if token is not None and token[0] == evicted_id:
                        del self._tokens[old_hash]
        return handle

    def owns(self, session_id: str, token_hash: str) -> bool:
        with self._lock:
            token = self._tokens.get(token_hash)
            return token is not None and token[0] == session_id and token[1] > utcnow()

    def forget_token(self, token_hash: str) -> None:
        """Drop only the process-local token cache after external revocation."""
        with self._lock:
            token = self._tokens.pop(token_hash, None)
            if token is None:
                return
            handle = self._sessions.get(token[0])
            if handle is not None:
                handle.token_hashes.discard(token_hash)

    def refresh_token(self, token_hash: str, *, durable: bool = True) -> bool:
        """Roll one valid token in durable storage and the process-local cache."""
        if durable and self._persistence is not None:
            try:
                active = self._persistence.touch_resume(token_hash)
            except Exception as exc:
                raise SessionUnavailable("could not refresh resume token") from exc
            if not active:
                self.forget_token(token_hash)
                return False

        with self._lock:
            token = self._tokens.get(token_hash)
            if token is None or token[1] <= utcnow():
                if token is not None:
                    del self._tokens[token_hash]
                return False
            self._tokens[token_hash] = (
                token[0],
                utcnow() + _RESUME_WINDOW,
            )
            handle = self._sessions.get(token[0])
        self.record_metric(
            "resume_refreshes",
            metric_dimensions=(handle.metric_dimensions if handle is not None else None),
        )
        return True

    def revoke(self, token_hash: str) -> bool:
        if self._persistence is not None:
            try:
                self._persistence.revoke_token(token_hash)
            except Exception as exc:
                raise SessionUnavailable("could not revoke resume token") from exc
        with self._lock:
            token = self._tokens.pop(token_hash, None)
            session_id = token[0] if token is not None else None
            handle = self._sessions.get(session_id) if session_id else None
            if handle is not None:
                handle.token_hashes.discard(token_hash)
        return session_id is not None

    def replay_reset(
        self, token_hash: str, request: ResetSessionV2Request
    ) -> ResetResponse | None:
        """Replay a committed reset even though its token is now revoked."""
        payload = {"type": "reset", **request.model_dump(mode="json")}
        payload_hash = self._payload_hash(payload)
        request_id = str(request.request_id)
        with self._lock:
            self._prune_reset_receipts_locked()
            receipt = self._reset_receipts.get(token_hash, {}).get(request_id)
        if receipt is not None:
            if receipt.payload_hash != payload_hash:
                raise SessionConflict(
                    "idempotency_conflict",
                    "request_id was already used with a different reset payload",
                )
            self.record_metric("reset_replays")
            return receipt.response.model_copy(deep=True)
        if self._persistence is None:
            return None
        try:
            persisted = self._persistence.replay_reset(
                token_hash=token_hash,
                request_id=request.request_id,
                payload_hash=payload_hash,
            )
        except SessionConflict:
            raise
        except Exception as exc:
            raise SessionUnavailable("could not replay reset receipt") from exc
        if persisted is None:
            return None
        response = ResetResponse.model_validate(persisted)
        with self._lock:
            self._reset_receipts.setdefault(token_hash, {})[request_id] = ResetReceipt(
                payload_hash=payload_hash,
                response=response,
            )
        self.record_metric("reset_replays")
        return response.model_copy(deep=True)

    def recover_create(self, request_id: UUID) -> str | None:
        """Resolve a committed create using its client-held request capability."""

        request_key = str(request_id)
        with self._lock:
            handles = list(self._sessions.values())
        for handle in handles:
            with handle.lock:
                receipt = handle.receipts.get(request_key)
                if (
                    receipt is not None
                    and receipt.request_payload.get("type") == "create"
                ):
                    return receipt.response.session_id
        if self._persistence is None:
            return None
        try:
            return self._persistence.recover_create(request_id)
        except Exception as exc:
            raise SessionUnavailable("could not recover creation receipt") from exc

    def recover_reset_rotation(
        self, token_hash: str, request_id: UUID
    ) -> str | None:
        """Resolve reset only with both the revoked cookie and request proof."""

        request_key = str(request_id)
        with self._lock:
            self._prune_reset_receipts_locked()
            receipt = self._reset_receipts.get(token_hash, {}).get(request_key)
        if receipt is not None:
            return receipt.response.session.session_id
        if self._persistence is None:
            return None
        try:
            return self._persistence.recover_reset_rotation(
                token_hash, request_id
            )
        except Exception as exc:
            raise SessionUnavailable("could not recover reset receipt") from exc

    def reset(
        self,
        handle: V2SessionHandle,
        token_hash: str,
        request: ResetSessionV2Request,
        *,
        replacement_orchestrator: Any,
        replacement_interactions: list[Interaction],
        replacement_token_hash: str,
        accept_quarantine_recovery_key: bool = False,
        replacement_goal: GoalView | None = None,
        replacement_content_mode: ContentModeView | None = None,
    ) -> ResetResponse:
        """Atomically revoke one episode and install its fresh replacement."""
        payload = {"type": "reset", **request.model_dump(mode="json")}
        payload_hash = self._payload_hash(payload)
        request_id = str(request.request_id)
        with handle.lock:
            replayed = self.replay_reset(token_hash, request)
            if replayed is not None:
                self.forget_token(token_hash)
                return replayed
            self.validate_reset_preconditions(
                handle,
                token_hash,
                request,
                accept_quarantine_recovery_key=accept_quarantine_recovery_key,
            )
            pending_key = self._pending_key(handle.orchestrator)

            now = utcnow()
            self._enforce_episode_quota(handle, now=now)
            replacement_session_id = uuid4().hex
            bind_episode = getattr(replacement_orchestrator, "bind_episode_id", None)
            if callable(bind_episode):
                bind_episode(replacement_session_id)
            replacement = V2SessionHandle(
                session_id=replacement_session_id,
                learner_id=handle.learner_id,
                orchestrator=replacement_orchestrator,
                goal=replacement_goal or handle.goal,
                profile=handle.profile,
                context=handle.context,
                content_mode=replacement_content_mode or handle.content_mode,
                durability=handle.durability,
                started_at=now,
                updated_at=now,
                episode_starts=self._next_episode_starts(handle, now),
                revision=0,
                transcript=self._interaction_entries([], replacement_interactions),
                token_hashes={replacement_token_hash},
                metric_dimensions=self._resolve_metric_dimensions(
                    replacement_orchestrator
                ),
            )
            self._replace_visible_snapshot(
                replacement.orchestrator,
                context=replacement.context,
                transcript=replacement.transcript,
            )
            response = ResetResponse(reset=True, session=self.view(replacement))
            if self._persistence is not None:
                try:
                    persisted = self._persistence.commit_reset(
                        handle=handle,
                        token_hash=token_hash,
                        request_id=request.request_id,
                        payload_hash=payload_hash,
                        request_payload=payload,
                        response_payload=response.model_dump(mode="json"),
                        expected_revision=request.expected_revision,
                        pending_key=pending_key,
                        replacement=replacement,
                        replacement_token_hash=replacement_token_hash,
                    )
                except SessionConflict:
                    raise
                except SessionRateLimited:
                    self.record_metric("episode_resets_rate_limited")
                    raise
                except Exception as exc:
                    raise SessionUnavailable("could not durably commit reset") from exc
                if persisted is not None:
                    raise DurableResetReplay(
                        ResetResponse.model_validate(persisted)
                    )

            with self._lock:
                self._reset_receipts.setdefault(token_hash, {})[request_id] = ResetReceipt(
                    payload_hash=payload_hash,
                    response=response,
                )
                token = self._tokens.pop(token_hash, None)
                session_id = token[0] if token is not None else handle.session_id
                live = self._sessions.get(session_id)
                if live is not None:
                    live.token_hashes.discard(token_hash)
                while len(self._sessions) >= self._max_sessions:
                    evicted_id, evicted = self._sessions.popitem(last=False)
                    for old_hash in evicted.token_hashes:
                        old_token = self._tokens.get(old_hash)
                        if old_token is not None and old_token[0] == evicted_id:
                            del self._tokens[old_hash]
                self._sessions[replacement.session_id] = replacement
                self._tokens[replacement_token_hash] = (
                    replacement.session_id,
                    utcnow() + _RESUME_WINDOW,
                )
            self.record_metric(
                "resets_committed",
                metric_dimensions=replacement.metric_dimensions,
            )
            self.record_metric(
                "sessions_created_by_reset",
                metric_dimensions=replacement.metric_dimensions,
            )
            return response.model_copy(deep=True)

    def validate_reset_preconditions(
        self,
        handle: V2SessionHandle,
        token_hash: str,
        request: ResetSessionV2Request,
        *,
        accept_quarantine_recovery_key: bool = False,
    ) -> SessionView:
        """Check reset concurrency/auth facts without changing session state.

        The API uses this before constructing and qualifying a replacement so
        a stale request cannot be misreported as content exhaustion. ``reset``
        repeats the same checks under its commit lock to close the race between
        preflight and the authoritative transaction.
        """

        with handle.lock:
            current = self.view(handle)
            if request.expected_revision != handle.revision:
                raise SessionConflict(
                    "stale_interaction",
                    "session revision changed; use the authoritative snapshot",
                    current,
                )
            pending_key = self._pending_key(handle.orchestrator)
            if (
                not accept_quarantine_recovery_key
                and pending_key != request.pending_key
            ):
                raise SessionConflict(
                    "stale_interaction",
                    "the pending interaction changed; use the authoritative snapshot",
                    current,
                )
            if not self.owns(handle.session_id, token_hash):
                raise SessionConflict(
                    "session_revoked",
                    "the anonymous session token is no longer active",
                    current,
                )
            return current.model_copy(deep=True)

    def _enforce_episode_quota(
        self,
        handle: V2SessionHandle,
        *,
        now: datetime | None = None,
    ) -> None:
        """Bound anonymous episode churn in the rolling resume window."""
        cutoff = (now or utcnow()) - _RESUME_WINDOW
        recent = sum(started_at >= cutoff for started_at in handle.episode_starts)
        if recent >= self._max_episodes_per_learner:
            self.record_metric("episode_resets_rate_limited")
            raise SessionRateLimited(
                "this anonymous learner reached the rolling episode limit"
            )

    @staticmethod
    def _next_episode_starts(
        previous: V2SessionHandle | None,
        now: datetime,
    ) -> tuple[datetime, ...]:
        """Carry one reset chain's recent starts across bounded-cache eviction."""
        if previous is None:
            return (now,)
        cutoff = now - _RESUME_WINDOW
        recent = tuple(
            started_at
            for started_at in previous.episode_starts
            if started_at >= cutoff
        )
        return (*recent, now)

    def _prune_reset_receipts_locked(self) -> None:
        """Discard reset replay material after its 30-day token window."""
        cutoff = utcnow() - _RESUME_WINDOW
        for token_hash, receipts in list(self._reset_receipts.items()):
            current = {
                request_id: receipt
                for request_id, receipt in receipts.items()
                if receipt.created_at >= cutoff
            }
            if current:
                self._reset_receipts[token_hash] = current
            else:
                del self._reset_receipts[token_hash]

    def replay_action(
        self,
        handle: V2SessionHandle,
        action: SessionAction,
        *,
        token_hash: str,
    ) -> SessionView | None:
        """Return only an already-committed action; never invoke the orchestrator.

        This is the safe idempotency path used while new v2 mutations are
        paused. A reused request id with a different payload remains a conflict,
        while an unseen request returns ``None`` for the API control plane to
        reject with its retryable pause response.
        """
        payload = action.model_dump(mode="json")
        payload_hash = self._payload_hash(payload)
        request_id = str(action.request_id)
        with handle.lock:
            if not self.owns(handle.session_id, token_hash):
                raise SessionConflict(
                    "session_revoked",
                    "the anonymous session token is no longer active",
                    self.view(handle),
                )
            receipt = handle.receipts.get(request_id)
            if receipt is None:
                return None
            if receipt.payload_hash != payload_hash:
                raise SessionConflict(
                    "idempotency_conflict",
                    "request_id was already used with a different payload",
                    self.view(handle),
                )
            revision_before_replay = handle.revision
            if not self.refresh_token(token_hash):
                raise SessionConflict(
                    "session_revoked",
                    "the anonymous session token is no longer active",
                    self.view(handle),
                )
            if handle.revision != revision_before_replay:
                self._integrity_failure("duplicate_advances_detected")
                raise SessionIntegrityError(
                    "idempotent replay changed the authoritative revision"
                )
            response = receipt.response.model_copy(deep=True)
        self.record_metric(
            "action_replays",
            metric_dimensions=handle.metric_dimensions,
        )
        return response

    def apply(
        self,
        session_id: str,
        action: SessionAction,
        *,
        token_hash: str | None = None,
    ) -> SessionView:
        """Apply one action exactly once under revision and pending-key guards."""
        handle = self.get(session_id)
        payload = action.model_dump(mode="json")
        payload_hash = self._payload_hash(payload)
        request_id = str(action.request_id)

        with handle.lock:
            if token_hash is not None and not self.owns(session_id, token_hash):
                raise SessionConflict(
                    "session_revoked",
                    "the anonymous session token is no longer active",
                )
            previous_receipt = handle.receipts.get(request_id)
            if previous_receipt is not None:
                if previous_receipt.payload_hash != payload_hash:
                    raise SessionConflict(
                        "idempotency_conflict",
                        "request_id was already used with a different payload",
                        self.view(handle),
                    )
                revision_before_replay = handle.revision
                if token_hash is not None and not self.refresh_token(token_hash):
                    raise SessionConflict(
                        "session_revoked",
                        "the anonymous session token is no longer active",
                        self.view(handle),
                    )
                if handle.revision != revision_before_replay:
                    self._integrity_failure("duplicate_advances_detected")
                    raise SessionIntegrityError(
                        "idempotent replay changed the authoritative revision"
                    )
                self.record_metric(
                    "action_replays",
                    metric_dimensions=handle.metric_dimensions,
                )
                return previous_receipt.response.model_copy(deep=True)

            committed_actions = sum(
                receipt.request_payload.get("type") not in {"create", "reset"}
                for receipt in handle.receipts.values()
            )
            if committed_actions >= self._max_receipts:
                self.record_metric("actions_rate_limited")
                raise SessionRateLimited(
                    "this episode reached its safe action-storage limit"
                )

            current = self.view(handle)
            if action.expected_revision != handle.revision:
                raise SessionConflict(
                    "stale_interaction",
                    "session revision changed; use the authoritative snapshot",
                    current,
                )
            pending_key = self._pending_key(handle.orchestrator)
            if pending_key != action.pending_key:
                raise SessionConflict(
                    "stale_interaction",
                    "the pending interaction changed; use the authoritative snapshot",
                    current,
                )
            if self._has_prior_advance(handle, action.pending_key):
                self._integrity_failure("duplicate_advances_detected")
                raise SessionIntegrityError(
                    "the pending interaction was already advanced by a committed action"
                )

            working = copy.deepcopy(handle.orchestrator)
            self._sync_visible_content(
                working,
                context=handle.context,
                transcript=handle.transcript,
                action=action,
            )
            pending_before = getattr(working, "pending", None)
            if callable(pending_before):
                pending_before = pending_before()
            if pending_before is None:
                pending_before = getattr(working, "_pending", None)
            event_count = len(getattr(working.learner, "events", ()))
            previous_exposures = self._exposure_records(working)
            additions, widget_attempt = self._invoke(working, action)
            new_events = list(getattr(working.learner, "events", ()))[event_count:]
            new_exposures = self._changed_exposures(
                previous_exposures, self._exposure_records(working)
            )
            new_entries = self._action_entries(handle.transcript, action, additions)
            self._sync_visible_entries(working, new_entries)
            next_transcript = [*handle.transcript, *new_entries]
            self._replace_visible_snapshot(
                working,
                context=handle.context,
                transcript=next_transcript,
            )
            next_handle = V2SessionHandle(
                session_id=handle.session_id,
                learner_id=handle.learner_id,
                orchestrator=working,
                goal=handle.goal,
                profile=handle.profile,
                context=handle.context,
                content_mode=handle.content_mode,
                durability=handle.durability,
                started_at=handle.started_at,
                updated_at=utcnow(),
                episode_starts=handle.episode_starts,
                revision=handle.revision + 1,
                transcript=next_transcript,
                receipts=handle.receipts.copy(),
                token_hashes=set(handle.token_hashes),
                metric_dimensions=handle.metric_dimensions,
                lock=handle.lock,
            )
            next_view = self.view(next_handle)
            self._validate_action_commit(
                handle=handle,
                next_handle=next_handle,
                action=action,
                pending_key_before=pending_key,
                new_entries=new_entries,
                new_events=new_events,
                widget_attempt=widget_attempt,
            )
            receipt = MutationReceipt(
                payload_hash=payload_hash,
                response=next_view,
                request_payload=payload,
            )
            next_handle.receipts[request_id] = receipt
            # Mutation receipts are correctness state, not a cache: evicting an
            # old receipt would turn a valid transport retry into stale input.
            # Durable deployments retain them in the database as well.

            if self._persistence is not None:
                try:
                    persisted = self._persistence.commit_action(
                        handle=next_handle,
                        previous_revision=handle.revision,
                        request_id=action.request_id,
                        payload_hash=payload_hash,
                        request_payload=payload,
                        response_payload=next_view.model_dump(mode="json"),
                        new_transcript=new_entries,
                        new_events=new_events,
                        new_exposures=new_exposures,
                        widget_attempt=widget_attempt,
                        token_hash=token_hash,
                    )
                except SessionConflict:
                    raise
                except Exception as exc:
                    raise SessionUnavailable("could not durably commit action") from exc
                if persisted is not None:
                    self.record_metric(
                        "action_replays",
                        metric_dimensions=next_handle.metric_dimensions,
                    )
                    raise DurableReceiptReplay(SessionView.model_validate(persisted))

            handle.orchestrator = next_handle.orchestrator
            handle.updated_at = next_handle.updated_at
            handle.revision = next_handle.revision
            handle.transcript = next_handle.transcript
            handle.receipts = next_handle.receipts
            item_id = str(getattr(pending_before, "item_id", "unknown"))
            with self._lock:
                for token_hash in handle.token_hashes:
                    if token_hash in self._tokens:
                        self._tokens[token_hash] = (
                            handle.session_id,
                            utcnow() + _RESUME_WINDOW,
                        )
            self.record_metric(
                "actions_committed",
                item_id=item_id,
                metric_dimensions=handle.metric_dimensions,
            )
            logger.info(
                json.dumps(
                    {
                        "event": "v2_action_committed",
                        "session_revision": next_view.revision,
                        "phase": next_view.phase,
                        "action_type": getattr(action, "type", "unknown"),
                        "item_id": getattr(pending_before, "item_id", None),
                        "family_id": getattr(pending_before, "family_id", None),
                        "kc_id": getattr(pending_before, "kc_id", None),
                        "item_revision": getattr(
                            pending_before, "item_revision", None
                        ),
                        "item_bank_version": self._summary(
                            handle.orchestrator
                        ).get("item_bank_version"),
                    },
                    sort_keys=True,
                )
            )
            return next_view

    def _integrity_failure(self, metric: str) -> None:
        """Record an actionable invariant failure without learner content."""
        self.record_metric(metric)
        self.record_metric("commit_integrity_failures")

    @staticmethod
    def _has_prior_advance(handle: V2SessionHandle, pending_key: str) -> bool:
        """Detect a previously receipted advance that live state somehow lost."""
        for receipt in handle.receipts.values():
            if receipt.request_payload.get("pending_key") != pending_key:
                continue
            response_pending = receipt.response.pending
            if response_pending is None or response_pending.key != pending_key:
                return True
        return False

    def _validate_action_commit(
        self,
        *,
        handle: V2SessionHandle,
        next_handle: V2SessionHandle,
        action: SessionAction,
        pending_key_before: str,
        new_entries: list[TranscriptEntry],
        new_events: list[Any],
        widget_attempt: dict[str, Any] | None,
    ) -> None:
        """Fail before persistence/live swap when atomic deltas are incoherent."""
        if next_handle.revision != handle.revision + 1:
            self._integrity_failure("duplicate_advances_detected")
            raise SessionIntegrityError(
                "a successful action must advance the revision exactly once"
            )
        if (
            next_handle.transcript[: len(handle.transcript)] != handle.transcript
            or next_handle.transcript[len(handle.transcript) :] != new_entries
            or not new_entries
        ):
            self._integrity_failure("transcript_integrity_failures")
            raise SessionIntegrityError("the transcript delta is incomplete")

        old_event_ids = {
            str(event.event_id)
            for event in getattr(handle.orchestrator.learner, "events", ())
        }
        new_event_ids = [str(event.event_id) for event in new_events]
        if (
            len(new_event_ids) != len(set(new_event_ids))
            or old_event_ids.intersection(new_event_ids)
        ):
            self._integrity_failure("duplicate_evidence_detected")
            raise SessionIntegrityError("the evidence delta contains a duplicate event")
        if any(
            getattr(event, "episode_id", None) != handle.session_id
            for event in new_events
        ):
            self._integrity_failure("evidence_provenance_failures")
            raise SessionIntegrityError("new evidence is not bound to this episode")
        pinned_catalog_version = getattr(
            next_handle.orchestrator,
            "pedagogy_catalog_version",
            None,
        )
        if new_events and (
            not isinstance(pinned_catalog_version, str)
            or not pinned_catalog_version
            or pinned_catalog_version == "legacy"
            or any(
                getattr(event, "pedagogy_catalog_version", "legacy")
                != pinned_catalog_version
                or getattr(event, "content_versions", {}).get(
                    "pedagogy_catalog"
                )
                != pinned_catalog_version
                for event in new_events
            )
        ):
            self._integrity_failure("evidence_provenance_failures")
            raise SessionIntegrityError(
                "new evidence is not bound to the pinned pedagogy catalog"
            )

        advanced = self._pending_key(next_handle.orchestrator) != pending_key_before
        evidence_required = (
            isinstance(action, WidgetAttemptAction)
            and widget_attempt is not None
            and bool(widget_attempt.get("counted"))
        ) or (isinstance(action, AnswerAction) and advanced)
        if evidence_required and not new_events:
            self._integrity_failure("missing_evidence_detected")
            raise SessionIntegrityError(
                "a scored state transition produced no evidence"
            )
        if isinstance(action, WidgetAttemptAction) and widget_attempt is None:
            self._integrity_failure("missing_widget_attempts_detected")
            raise SessionIntegrityError(
                "a widget action produced no durable attempt trajectory"
            )

    def view(self, handle: V2SessionHandle) -> SessionView:
        """Project internal state into the public safe snapshot."""
        # Mutations swap several correlated fields after the durable commit.
        # Holding the same per-session RLock prevents GET from observing a
        # hybrid (for example, a new pending item with an old revision).
        with handle.lock:
            return self._view_locked(handle)

    def _view_locked(self, handle: V2SessionHandle) -> SessionView:
        """Build a snapshot while the caller owns ``handle.lock``."""
        orchestrator = handle.orchestrator
        graph_nodes = getattr(orchestrator, "_nodes", self._graph_nodes)
        summary = self._summary(orchestrator)
        pending = self._pending(orchestrator, graph_nodes)
        current_kc = pending.kc_id if pending is not None else None
        current_name = (
            self._skill_name(current_kc, graph_nodes) if current_kc else None
        )
        strengths = self._kc_names(
            summary.get("confirmed_mastery", summary.get("mastered_in_session", [])),
            graph_nodes,
        )
        gaps = self._kc_names(summary.get("confirmed_gaps", []), graph_nodes)
        uncertain_ids = summary.get(
            "uncertain",
            summary.get("uncertain_skills", summary.get("frontier", [])),
        )
        uncertain = [
            name
            for name in self._kc_names(uncertain_ids, graph_nodes)
            if name not in set(gaps)
        ]
        phase = str(getattr(getattr(orchestrator, "phase", "unknown"), "value", None) or
                    getattr(orchestrator, "phase", "unknown"))
        return SessionView(
            session_id=handle.session_id,
            revision=handle.revision,
            phase=phase,
            durability=handle.durability,
            goal=handle.goal,
            profile=handle.profile,
            context=handle.context,
            content_mode=handle.content_mode,
            transcript=list(handle.transcript),
            pending=pending,
            progress=ProgressView(
                phase=phase,
                current_skill=current_name,
                plan_step=summary.get("plan_step"),
                diagnosis_probes_used=int(summary.get("probes_used", 0)),
                diagnosis_probe_budget=summary.get("probe_budget"),
                interactions_used=int(summary.get("interactions_used", 0)),
            ),
            learner_summary=LearnerSummaryView(
                confirmed_strengths=strengths,
                confirmed_gaps=gaps,
                uncertain_skills=uncertain,
            ),
            started_at=handle.started_at,
            updated_at=handle.updated_at,
        )

    @staticmethod
    def _payload_hash(payload: dict[str, Any]) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _pending_key(orchestrator: Any) -> str | None:
        value = getattr(orchestrator, "pending_key", None)
        if value is not None:
            return value
        pending = getattr(orchestrator, "_pending", None)
        return getattr(pending, "key", None)

    def _pending(
        self,
        orchestrator: Any,
        graph_nodes: dict[str, Any] | None = None,
    ) -> PendingView | None:
        pending = getattr(orchestrator, "pending", None)
        if callable(pending):
            pending = pending()
        if pending is None:
            pending = getattr(orchestrator, "_pending", None)
        if pending is None:
            return None
        key = getattr(pending, "key", self._pending_key(orchestrator))
        kc_id = getattr(pending, "kc_id", getattr(orchestrator, "pending_kc", None))
        kind = getattr(pending, "kind", getattr(orchestrator, "pending_kind", None))
        if key is None or kc_id is None or kind is None:
            return None
        input_mode = getattr(pending, "input_mode", "math")
        answer_spec = getattr(pending, "answer_spec", None)
        choice_options = (
            list(getattr(answer_spec, "option_ids", ()))
            if str(getattr(answer_spec, "kind", "")) == "choice"
            else []
        )
        prompt_segments = [
            (
                segment.model_dump(mode="json")
                if hasattr(segment, "model_dump")
                else dict(segment)
            )
            for segment in getattr(pending, "prompt_segments", ())
            if hasattr(segment, "model_dump") or isinstance(segment, dict)
        ]
        hints = list(getattr(pending, "hints", ()))
        revealing_hints = list(getattr(pending, "revealing_hints", ()))
        hints_given = max(0, int(getattr(pending, "hints_given", 0)))
        can_hint = bool(getattr(pending, "can_hint", False))
        hint_available = can_hint and hints_given < len(hints)
        next_reveals_answer = bool(
            hint_available
            and hints_given < len(revealing_hints)
            and revealing_hints[hints_given]
        )
        public_widget = getattr(orchestrator, "pending_widget", None)
        if callable(public_widget):
            public_widget = public_widget()
        widget = self._safe_widget(public_widget)
        raw_widget_state = getattr(pending, "widget_state", None)
        if hasattr(raw_widget_state, "model_dump"):
            raw_widget_state = raw_widget_state.model_dump(mode="json")
        return PendingView(
            key=key,
            kind=str(getattr(kind, "value", kind)),
            kc_id=kc_id,
            skill_name=self._skill_name(kc_id, graph_nodes),
            input_mode=input_mode,
            prompt=str(getattr(pending, "prompt", "")),
            prompt_segments=prompt_segments,
            choice_options=choice_options,
            widget=widget,
            widget_state=self._safe_widget_state(raw_widget_state),
            hint=PendingHintView(
                available=hint_available,
                next_index=min(hints_given, len(hints)),
                total=len(hints),
                next_reveals_answer=next_reveals_answer,
            ),
            can_hint=hint_available,
        )

    @staticmethod
    def _summary(orchestrator: Any) -> dict[str, Any]:
        summary = getattr(orchestrator, "summary", None)
        if callable(summary):
            return dict(summary())
        return {}

    @staticmethod
    def _exposure_records(orchestrator: Any) -> list[Any]:
        state = getattr(orchestrator, "exposure_state", None)
        if state is None:
            state = getattr(orchestrator, "_exposure_state", None)
        records = (
            getattr(state, "records", getattr(state, "exposures", ()))
            if state is not None
            else ()
        )
        return list(records)

    @staticmethod
    def _sync_visible_content(
        orchestrator: Any,
        *,
        context: str | None,
        transcript: list[TranscriptEntry],
        action: SessionAction,
    ) -> None:
        """Synchronize the safe UI history before a same-call allocation.

        The orchestrator normally records its own outputs. Replaying the actual
        API transcript here also covers API-authored feedback bubbles and makes
        the public snapshot—not an implicit generation path—the final authority
        over what the learner has seen.
        """
        remember = getattr(orchestrator, "remember_visible_content", None)
        if not callable(remember):
            return
        remember(context)
        for entry in transcript:
            remember(entry.text, entry.prompt_segments, entry.widget)
        if isinstance(action, AnswerAction):
            remember(action.answer)
        elif isinstance(action, WidgetAttemptAction):
            # This generic bubble is public and precedes the tutor verdict.
            remember("Submitted guided-practice response.")
            remember_private = getattr(
                orchestrator, "remember_private_visible_input", None
            )
            if callable(remember_private):
                remember_private(action.response)

    @staticmethod
    def _sync_visible_entries(
        orchestrator: Any,
        entries: list[TranscriptEntry],
    ) -> None:
        """Include every API-authored transcript field in checkpointed history."""
        remember = getattr(orchestrator, "remember_visible_content", None)
        if not callable(remember):
            return
        for entry in entries:
            remember(entry.text, entry.prompt_segments, entry.widget)

    @staticmethod
    def _replace_visible_snapshot(
        orchestrator: Any,
        *,
        context: str | None,
        transcript: list[TranscriptEntry],
    ) -> None:
        """Make public visible history exactly derivable from the safe snapshot."""
        replace = getattr(orchestrator, "replace_public_visible_content", None)
        if not callable(replace):
            return
        values: list[Any] = [context]
        for entry in transcript:
            values.extend((entry.text, entry.prompt_segments, entry.widget))
        pending = getattr(orchestrator, "pending", None)
        if callable(pending):
            pending = pending()
        if pending is None:
            pending = getattr(orchestrator, "_pending", None)
        if pending is not None:
            answer_spec = getattr(pending, "answer_spec", None)
            choice_options = (
                list(getattr(answer_spec, "option_ids", ()))
                if str(getattr(answer_spec, "kind", "")) == "choice"
                else []
            )
            values.extend(
                (
                    getattr(pending, "prompt", ""),
                    getattr(pending, "prompt_segments", ()),
                    choice_options,
                )
            )
        replace(*values)

    @staticmethod
    def _changed_exposures(before: list[Any], after: list[Any]) -> list[Any]:
        def payload(record: Any) -> dict[str, Any]:
            if hasattr(record, "model_dump"):
                return record.model_dump(mode="json")
            if isinstance(record, dict):
                return record
            return vars(record)

        previous = {
            (data.get("item_id"), data.get("revision", 1)): data
            for data in (payload(record) for record in before)
        }
        return [
            record
            for record in after
            if previous.get(
                (
                    payload(record).get("item_id"),
                    payload(record).get("revision", 1),
                )
            )
            != payload(record)
        ]

    def _skill_name(
        self,
        kc_id: str,
        graph_nodes: dict[str, Any] | None = None,
    ) -> str:
        node = (graph_nodes or self._graph_nodes).get(kc_id)
        return getattr(node, "name", kc_id)

    def _kc_names(
        self,
        ids: Any,
        graph_nodes: dict[str, Any] | None = None,
    ) -> list[str]:
        if not isinstance(ids, (list, tuple, set)):
            return []
        return [
            self._skill_name(str(kc_id), graph_nodes)
            for kc_id in ids
        ]

    @staticmethod
    def _safe_widget(widget: Any) -> dict[str, Any] | None:
        """Project a widget through an explicit learner-safe allow-list."""
        if not isinstance(widget, dict):
            return None
        widget_type = widget.get("widget_type")
        if not isinstance(widget_type, str):
            return None
        common = {
            "widget_type": widget_type,
            "learning_objective": str(widget.get("learning_objective", "")),
            "prompt": str(widget.get("prompt", "")),
            "text_fallback": str(widget.get("text_fallback", "")),
        }
        interaction_version = widget.get("interaction_version")
        if interaction_version in {"mapping_v1", "slider_v1"}:
            common["interaction_version"] = interaction_version
        if widget_type in {"slider_v1", "slider"}:
            presentation = widget.get("presentation")
            if isinstance(presentation, dict):
                required_strings = (
                    "prompt",
                    "label",
                    "help_text",
                    "value_label",
                )
                required_numbers = (
                    "minimum",
                    "maximum",
                    "step",
                    "initial_value",
                )
                if not all(
                    isinstance(presentation.get(key), str)
                    for key in required_strings
                ):
                    return None
                if not all(
                    not isinstance(presentation.get(key), bool)
                    and isinstance(presentation.get(key), (int, float))
                    and math.isfinite(float(presentation[key]))
                    for key in required_numbers
                ):
                    return None
                safe_presentation = {
                    key: presentation[key]
                    for key in (*required_strings, *required_numbers)
                }
                result_template = presentation.get("result_template")
                if result_template is not None:
                    if not isinstance(result_template, str):
                        return None
                    safe_presentation["result_template"] = result_template
                visual_summary = presentation.get("visual_summary")
                if visual_summary is not None:
                    try:
                        safe_presentation["visual_summary"] = (
                            PlotPromptSegment.model_validate(
                                visual_summary
                            ).model_dump(mode="json")
                        )
                    except (TypeError, ValueError):
                        return None
                common["presentation"] = safe_presentation
                return common
            if widget_type == "slider_v1":
                return None
            params = widget.get("params")
            if not isinstance(params, dict):
                return None
            common["params"] = {
                key: params[key]
                for key in ("min", "max", "step", "initial", "plot", "shade")
                if key in params
            }
        elif widget_type == "click_region":
            common["diagram"] = widget.get("diagram")
            regions = widget.get("regions")
            if not isinstance(regions, list):
                return None
            common["regions"] = [
                {
                    key: region[key]
                    for key in ("id", "label", "shape")
                    if isinstance(region, dict) and key in region
                }
                for region in regions
                if isinstance(region, dict)
            ]
        elif widget_type in {"mapping_v1", "mapping"}:
            presentation = widget.get("presentation")
            if isinstance(presentation, dict):
                rows = V2SessionStore._safe_mapping_entries(
                    presentation.get("rows")
                )
                options = V2SessionStore._safe_mapping_entries(
                    presentation.get("options")
                )
                prompt = presentation.get("prompt")
                if rows is None or options is None or not isinstance(prompt, str):
                    return None
                common["presentation"] = {
                    "prompt": prompt,
                    "rows": rows,
                    "options": options,
                }
                return common
            if widget_type == "mapping_v1":
                return None
            left, right = widget.get("left"), widget.get("right")
            if not isinstance(left, list) or not isinstance(right, list):
                return None
            common["left"] = list(left)
            common["right"] = list(right)
        elif widget_type == "live_input":
            common["input_kind"] = widget.get("input_kind")
            render = widget.get("render")
            if isinstance(render, dict):
                common["render"] = {
                    key: render[key] for key in ("plot", "var") if key in render
                }
        else:
            return None
        return common

    @staticmethod
    def _safe_mapping_entries(value: Any) -> list[dict[str, str]] | None:
        if not isinstance(value, list) or not 2 <= len(value) <= 12:
            return None
        result: list[dict[str, str]] = []
        for entry in value:
            if not isinstance(entry, dict):
                return None
            entry_id = entry.get("entry_id")
            label = entry.get("label")
            spoken_text = entry.get("spoken_text")
            if not all(
                isinstance(field, str) and field
                for field in (entry_id, label, spoken_text)
            ):
                return None
            result.append(
                {
                    "entry_id": entry_id,
                    "label": label,
                    "spoken_text": spoken_text,
                }
            )
        return result

    @staticmethod
    def _safe_widget_state(value: Any) -> dict[str, Any] | None:
        """Keep only resumable learner selections, never private widget truth."""

        if not isinstance(value, dict):
            return None
        if "value" in value:
            current = value.get("value")
            if (
                isinstance(current, bool)
                or not isinstance(current, (int, float))
                or not math.isfinite(float(current))
            ):
                return None
            return {"value": current}
        rows = value.get("rows")
        if not isinstance(rows, list) or not 2 <= len(rows) <= 12:
            return None
        safe_rows: list[dict[str, str]] = []
        for row in rows:
            if not isinstance(row, dict):
                return None
            row_id = row.get("id")
            selected = row.get("value", "")
            if (
                not isinstance(row_id, str)
                or not row_id
                or len(row_id) > 64
                or not isinstance(selected, str)
                or len(selected) > 64
            ):
                return None
            safe_rows.append({"id": row_id, "value": selected})
        return {"rows": safe_rows}

    @staticmethod
    def _interaction_entries(
        transcript: list[TranscriptEntry], interactions: list[Any]
    ) -> list[TranscriptEntry]:
        start = len(transcript)
        entries: list[TranscriptEntry] = []
        for offset, interaction in enumerate(interactions):
            if isinstance(interaction, BaseModel):
                data = interaction.model_dump()
            elif isinstance(interaction, dict):
                data = interaction
            else:
                continue
            entries.append(
                TranscriptEntry(
                    sequence=start + offset,
                    role="tutor",
                    kind=data.get("kind", "message"),
                    text=str(data.get("text", "")),
                    content_blocks=data.get("content_blocks") or [],
                    interaction_key=data.get("key"),
                    kc_id=data.get("kc_id"),
                    prompt_segments=data.get("prompt_segments"),
                    widget=V2SessionStore._safe_widget(data.get("widget")),
                    widget_state=V2SessionStore._safe_widget_state(
                        data.get("widget_state")
                    ),
                )
            )
        return entries

    def _action_entries(
        self,
        transcript: list[TranscriptEntry],
        action: SessionAction,
        additions: list[dict[str, Any]],
    ) -> list[TranscriptEntry]:
        entries: list[TranscriptEntry] = []
        if isinstance(action, AnswerAction):
            entries.append(
                TranscriptEntry(
                    sequence=len(transcript),
                    role="student",
                    kind="message",
                    text=action.answer,
                    interaction_key=action.pending_key,
                )
            )
        elif isinstance(action, WidgetAttemptAction):
            entries.append(
                TranscriptEntry(
                    sequence=len(transcript),
                    role="student",
                    kind="widget_attempt",
                    text="Submitted guided-practice response.",
                    interaction_key=action.pending_key,
                )
            )
        for addition in additions:
            entries.append(
                TranscriptEntry(
                    sequence=len(transcript) + len(entries),
                    role=addition.get("role", "tutor"),
                    kind=addition.get("kind", "message"),
                    text=addition.get("text", ""),
                    content_blocks=addition.get("content_blocks") or [],
                    interaction_key=(
                        addition.get("interaction_key") or addition.get("key")
                    ),
                    kc_id=addition.get("kc_id"),
                    prompt_segments=addition.get("prompt_segments"),
                    widget=self._safe_widget(addition.get("widget")),
                    widget_state=self._safe_widget_state(
                        addition.get("widget_state")
                    ),
                    widget_status=addition.get("widget_status"),
                    widget_attempt_number=addition.get("widget_attempt_number"),
                )
            )
        return entries

    def _invoke(
        self, orchestrator: Any, action: SessionAction
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        if isinstance(action, AnswerAction):
            result = orchestrator.submit(action.answer)
            return self._normalize_interactions(result), None
        if isinstance(action, HintAction):
            hint = orchestrator.hint()
            if hint is None:
                raise SessionConflict(
                    "hint_unavailable", "no further hint is available"
                )
            hint_text = str(getattr(hint, "text", hint))
            transitions = self._normalize_interactions(
                getattr(hint, "interactions", [])
            )
            return [
                {
                    "role": "tutor",
                    "kind": "hint",
                    "text": hint_text,
                    "interaction_key": action.pending_key,
                },
                *transitions,
            ], None
        if isinstance(action, WidgetAttemptAction):
            result = orchestrator.answer_widget(action.pending_key, action.response)
            if isinstance(result, tuple) and len(result) == 2:
                correct, message = result
                transitions: list[dict[str, Any]] = []
                verification_status = "solved" if correct else "attempted"
                counted = True
                attempt_number = None
            else:
                correct = bool(getattr(result, "correct", False))
                message = str(getattr(result, "message", result))
                transitions = self._normalize_interactions(
                    getattr(result, "interactions", [])
                )
                verification_status = str(
                    getattr(
                        result,
                        "status",
                        "solved"
                        if correct
                        else "remediated"
                        if transitions
                        else "attempted",
                    )
                )
                counted = bool(getattr(result, "counted", True))
                attempt_number = getattr(result, "attempt_number", None)
            widget_state = getattr(result, "widget_state", None)
            attempt = {
                "interaction_key": action.pending_key,
                "response": action.response,
                "correct": bool(correct),
                "verification_status": verification_status,
                "counted": counted,
            }
            return [
                {
                    "role": "tutor",
                    "kind": "widget_feedback",
                    "text": str(message),
                    "interaction_key": action.pending_key,
                    "widget_status": verification_status,
                    "widget_attempt_number": attempt_number,
                    "widget_state": widget_state,
                },
                *transitions,
            ], attempt
        if isinstance(action, TextFallbackAction):
            method = getattr(orchestrator, "use_text_fallback", None)
            if method is None:
                raise SessionConflict(
                    "text_fallback_unavailable",
                    "this interaction does not have a text fallback",
                )
            feedback = "The accessible text alternative is ready."
            remember = getattr(orchestrator, "remember_visible_content", None)
            if callable(remember):
                remember(feedback)
            transitions = self._normalize_interactions(method())
            return [
                {
                    "role": "tutor",
                    "kind": "widget_feedback",
                    "text": feedback,
                    "interaction_key": action.pending_key,
                    "widget_status": "text_fallback",
                },
                *transitions,
            ], None
        raise TypeError(f"unsupported action type: {type(action)!r}")

    @staticmethod
    def _normalize_interactions(result: Any) -> list[dict[str, Any]]:
        if result is None:
            return []
        interactions = result
        if not isinstance(interactions, (list, tuple)):
            interactions = getattr(result, "interactions", [])
        normalized: list[dict[str, Any]] = []
        for interaction in interactions:
            if isinstance(interaction, BaseModel):
                normalized.append(interaction.model_dump())
            elif isinstance(interaction, dict):
                normalized.append(dict(interaction))
        return normalized

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)
