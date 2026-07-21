"""Additive API v2 routes for authoritative, resumable sessions."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any
from urllib.parse import urlsplit

from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.responses import JSONResponse

from tutor.api.v2_persistence import DurableLedgerMismatch, V2PersistenceService
from tutor.api.v2_features import V2FeatureFlags
from tutor.api.v2_metrics import MetricsSink, V2MetricDimensions
from tutor.api.v2_schemas import (
    APIError,
    CatalogRolloutView,
    ContentModeView,
    CreateSessionV2Request,
    GoalCatalog,
    GoalView,
    RecoverSessionV2Request,
    RecoverSessionV2Response,
    ResetResponse,
    ResetSessionV2Request,
    SessionAction,
    SessionProfile,
    SessionView,
    WidgetCapabilityManifestView,
)
from tutor.api.v2_store import (
    DurableReceiptReplay,
    DurableResetReplay,
    SessionConflict,
    SessionIntegrityError,
    SessionRateLimited,
    SessionUnavailable,
    ResumeTokenExpired,
    V2SessionHandle,
    V2SessionStore,
)
from tutor.api.v2_controls import (
    MutationGate,
    MutationGateSnapshot,
    StaticMutationGate,
    safe_mutation_gate_snapshot,
)
from tutor.api.v2_versions import V2PolicyRegistry, V2VersionRegistry
from tutor.learner.evidence_trust import EvidenceTrustPolicy
from tutor.learner.params import DEFAULT_PARAMS_V2
from tutor.schemas.assessment import ItemBankDocument
from tutor.schemas.kc import GraphDocument
from tutor.schemas.learner import LearnerProfile
from tutor.schemas.pedagogy import PedagogyPackCatalog
from tutor.runtime_capabilities import widget_capability_manifest

_COOKIE_NAME = "tutor_resume_v2"
_COOKIE_MAX_AGE = 30 * 24 * 60 * 60
_ROLLOUT_COOKIE_NAME = "tutor_rollout_v2"
_ROLLOUT_COOKIE_MAX_AGE = 365 * 24 * 60 * 60
_RESUME_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_-]{43}$")
_PILOT_TARGETS = (
    "kc.int.u_substitution",
    "kc.der.chain_rule",
    "kc.int.ftc",
    "kc.der.product_quotient",
    "kc.alg.solve_quadratic",
)
_Factory = Callable[
    [GraphDocument, str, LearnerProfile, CreateSessionV2Request],
    tuple[Any, ContentModeView],
]


def _token_hash(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def _resume_token_well_formed(raw_token: str) -> bool:
    """Recognize the exact opaque 256-bit cookie encoding before DB lookup."""
    if _RESUME_TOKEN_PATTERN.fullmatch(raw_token) is None:
        return False
    try:
        return len(_urlsafe_decode(raw_token)) == 32
    except (ValueError, TypeError):
        return False


def _resume_secret(explicit: bytes | str | None) -> bytes:
    configured = explicit or os.environ.get("TUTOR_RESUME_TOKEN_SECRET")
    if configured is None:
        if os.environ.get("TUTOR_PILOT_PRODUCTION", "").lower() in {
            "1",
            "true",
            "yes",
        }:
            raise RuntimeError(
                "TUTOR_PILOT_PRODUCTION requires TUTOR_RESUME_TOKEN_SECRET"
            )
        return secrets.token_bytes(32)
    value = configured if isinstance(configured, bytes) else configured.encode()
    if len(value) < 32:
        raise RuntimeError("TUTOR_RESUME_TOKEN_SECRET must contain at least 32 bytes")
    return value


def _create_resume_token(secret: bytes, request_id: Any) -> str:
    digest = hmac.new(secret, str(request_id).encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _create_session_id(secret: bytes, request_id: Any) -> str:
    """Derive one retry-stable opaque episode id for session creation."""
    return hmac.new(
        secret,
        f"session:{request_id}".encode(),
        hashlib.sha256,
    ).hexdigest()[:32]


def _urlsafe_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _urlsafe_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(f"{value}{padding}")


def _create_rollout_cookie(secret: bytes, identity: bytes | None = None) -> str:
    """Create a signed anonymous 256-bit cohort identity."""
    identity = identity or secrets.token_bytes(32)
    if len(identity) != 32:
        raise ValueError("rollout identity must contain exactly 32 bytes")
    encoded_identity = _urlsafe_encode(identity)
    signature = hmac.new(
        secret,
        f"rollout-cookie-v1:{encoded_identity}".encode(),
        hashlib.sha256,
    ).digest()
    return f"{encoded_identity}.{_urlsafe_encode(signature)}"


def _rollout_identity(secret: bytes, cookie_value: str | None) -> bytes | None:
    if not cookie_value or len(cookie_value) > 128:
        return None
    try:
        encoded_identity, encoded_signature = cookie_value.split(".", 1)
        identity = _urlsafe_decode(encoded_identity)
        signature = _urlsafe_decode(encoded_signature)
    except (ValueError, TypeError):
        return None
    if len(identity) != 32 or len(signature) != hashlib.sha256().digest_size:
        return None
    expected = hmac.new(
        secret,
        f"rollout-cookie-v1:{encoded_identity}".encode(),
        hashlib.sha256,
    ).digest()
    return identity if hmac.compare_digest(signature, expected) else None


@dataclass(frozen=True)
class _RolloutAssignment:
    cookie_value: str
    bucket: int
    selected: bool


def _rollout_assignment(
    secret: bytes,
    percentage: int,
    cookie_value: str | None,
) -> _RolloutAssignment:
    identity = _rollout_identity(secret, cookie_value)
    if identity is None:
        cookie_value = _create_rollout_cookie(secret)
        identity = _rollout_identity(secret, cookie_value)
        assert identity is not None
    digest = hmac.new(
        secret,
        b"rollout-bucket-v1:" + identity,
        hashlib.sha256,
    ).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    return _RolloutAssignment(
        cookie_value=cookie_value,
        bucket=bucket,
        selected=bucket < percentage,
    )


def _replacement_resume_token(
    secret: bytes, token_hash: str, request_id: Any
) -> str:
    """Derive a retry-stable replacement token without storing raw secrets."""
    return _create_resume_token(secret, f"reset:{token_hash}:{request_id}")


def _error(
    status: int,
    code: str,
    message: str,
    session: SessionView | None = None,
    *,
    retryable: bool | None = None,
) -> JSONResponse:
    body = APIError(
        code=code,
        message=message,
        session=session,
        retryable=retryable,
    )
    return JSONResponse(
        status_code=status,
        content=body.model_dump(
            mode="json",
            exclude={"retryable"} if retryable is None else None,
        ),
    )


def _origin_allowed(request: Request) -> bool:
    fetch_site = request.headers.get("sec-fetch-site", "").lower()
    if fetch_site == "cross-site":
        return False
    source = request.headers.get("origin")
    if not source:
        referer = request.headers.get("referer")
        source = referer if referer else None
    if not source:
        local_hosts = {"127.0.0.1", "::1", "localhost", "testserver"}
        pilot_production = os.environ.get(
            "TUTOR_PILOT_PRODUCTION", ""
        ).lower() in {"1", "true", "yes"}
        explicitly_allowed = (
            not pilot_production
            and os.environ.get("TUTOR_ALLOW_MISSING_ORIGIN") == "1"
        )
        return request.url.hostname in local_hosts or explicitly_allowed
    parsed = urlsplit(source)
    actual = f"{parsed.scheme}://{parsed.netloc}"
    expected = f"{request.url.scheme}://{request.url.netloc}"
    configured = {
        value.strip().rstrip("/")
        for value in os.environ.get("TUTOR_ALLOWED_ORIGINS", "").split(",")
        if value.strip()
    }
    return actual == expected or actual in configured


def _set_resume_cookie(response: Response, raw_token: str, request: Request) -> None:
    local_hosts = {"127.0.0.1", "::1", "localhost", "testserver"}
    response.set_cookie(
        key=_COOKIE_NAME,
        value=raw_token,
        max_age=_COOKIE_MAX_AGE,
        httponly=True,
        secure=request.url.hostname not in local_hosts,
        samesite="lax",
        path="/api/v2",
    )


def _set_rollout_cookie(
    response: Response,
    cookie_value: str,
    request: Request,
) -> None:
    local_hosts = {"127.0.0.1", "::1", "localhost", "testserver"}
    response.set_cookie(
        key=_ROLLOUT_COOKIE_NAME,
        value=cookie_value,
        max_age=_ROLLOUT_COOKIE_MAX_AGE,
        httponly=True,
        secure=request.url.hostname not in local_hosts,
        samesite="lax",
        path="/api/v2",
    )


def _goals(
    graph: GraphDocument,
    available_targets: tuple[str, ...] | None = None,
    item_bank: ItemBankDocument | None = None,
    pedagogy_catalog: PedagogyPackCatalog | None = None,
) -> list[GoalView]:
    nodes = {node.id: node for node in graph.nodes}
    from tutor.content.item_bank import validate_item_bank
    from tutor.graph.service import ancestor_subgraph

    candidates = _PILOT_TARGETS if available_targets is None else available_targets
    try:
        if item_bank is None or pedagogy_catalog is None:
            return []
        bank = item_bank
        released = set(bank.released_kcs)
        eligible: list[str] = []
        for target in candidates:
            if target in nodes:
                closure = ancestor_subgraph(
                    graph, target, hard_only=True
                ).node_ids()
                if closure <= released and not validate_item_bank(
                    bank,
                    graph,
                    pedagogy_catalog,
                    released_kcs=closure,
                ):
                    eligible.append(target)
        available_targets = tuple(eligible)
    except (OSError, ValueError):
        available_targets = ()
    goals = [
        GoalView(
            goal_id=f"goal.{target.removeprefix('kc.')}",
            target_kc=target,
            title=nodes[target].name,
            description=nodes[target].description,
            course_level=nodes[target].course_level,
        )
        for target in available_targets
        if target in nodes
    ]
    return goals


def _default_factory(
    graph: GraphDocument,
    target_kc: str,
    profile: LearnerProfile,
    request: CreateSessionV2Request,
    *,
    item_bank: ItemBankDocument | None = None,
    pedagogy_catalog: PedagogyPackCatalog,
    evidence_trust_policy: EvidenceTrustPolicy,
    widget_capabilities: dict[str, Any] | None = None,
) -> tuple[Any, ContentModeView]:
    """Build the best available deterministic v2 machine.

    Curated mode is always available.  LLM coaching is explicitly reported as
    a fallback until a v2 coaching adapter can preserve atomic replay.
    """
    fallback = None
    effective = request.content_mode
    if request.content_mode == "llm_coaching":
        effective = "curated"
        fallback = "LLM coaching is unavailable in this v2 runtime; using curated content."
    from tutor.orchestrator.session_v2 import SessionOrchestratorV2

    orchestrator = SessionOrchestratorV2(
        graph,
        target_kc,
        profile,
        item_bank=item_bank,
        pedagogy_catalog=pedagogy_catalog,
        evidence_trust_policy=evidence_trust_policy,
        widget_capabilities=widget_capabilities,
    )
    return orchestrator, ContentModeView(
        requested=request.content_mode,
        effective=effective,
        fallback_reason=fallback,
    )


def install_v2_routes(
    app: FastAPI,
    graph: GraphDocument,
    *,
    persistence: V2PersistenceService | None = None,
    orchestrator_factory: _Factory | None = None,
    available_targets: tuple[str, ...] | None = None,
    item_bank: ItemBankDocument | None = None,
    pedagogy_catalog: PedagogyPackCatalog | None = None,
    resume_token_secret: bytes | str | None = None,
    version_registry: V2VersionRegistry | None = None,
    policy_registry: V2PolicyRegistry | None = None,
    feature_flags: V2FeatureFlags | None = None,
    metrics_sink: MetricsSink | None = None,
    mutation_gate: MutationGate | None = None,
    mutation_gate_max_age: timedelta | None = None,
) -> V2SessionStore:
    """Install API v2 without changing any v1 route or response model."""
    token_secret = _resume_secret(resume_token_secret)
    flags = feature_flags or V2FeatureFlags.from_environment()
    active_widget_manifest = widget_capability_manifest(
        rich_widgets=flags.rich_widgets
    )
    if mutation_gate_max_age is not None and mutation_gate_max_age <= timedelta(0):
        raise ValueError("mutation_gate_max_age must be positive")
    active_mutation_gate = (
        mutation_gate
        if mutation_gate is not None
        else StaticMutationGate(
            False,
            revision="builtin-static-v1:open",
            source="builtin_static",
        )
    )

    def observe_mutation_gate() -> MutationGateSnapshot:
        return safe_mutation_gate_snapshot(
            active_mutation_gate,
            max_age=mutation_gate_max_age,
        )

    def mutations_paused(snapshot: MutationGateSnapshot) -> bool:
        # The startup feature flag is a one-way safety ceiling: a dynamic
        # provider may pause an open deployment, but can never reopen a flag-
        # paused one.
        return flags.pause_v2_mutations or snapshot.paused

    def mutation_gate_view(snapshot: MutationGateSnapshot) -> dict[str, str]:
        return {
            "revision": snapshot.revision,
            "source": snapshot.source,
            "observed_at": snapshot.observed_at.isoformat(),
        }

    active_item_bank = item_bank
    active_pedagogy_catalog = pedagogy_catalog
    if active_item_bank is None or active_pedagogy_catalog is None:
        try:
            from tutor.content.item_bank import load_item_bank
            from tutor.packs.loader import load_pedagogy_catalog

            if active_item_bank is None:
                active_item_bank = load_item_bank()
            if active_pedagogy_catalog is None:
                active_pedagogy_catalog = load_pedagogy_catalog()
        except (OSError, ValueError):
            # The public catalog remains empty when the packaged release
            # cannot be loaded and validated.
            active_item_bank = None
            active_pedagogy_catalog = None
    registry = version_registry or V2VersionRegistry.from_environment()
    active_release = None
    if active_item_bank is not None and active_pedagogy_catalog is not None:
        active_release = registry.register(
            graph,
            active_item_bank,
            active_pedagogy_catalog,
        )
    evidence_trust_policy = registry.evidence_trust_registry
    from tutor.orchestrator.session_v2 import SessionOrchestratorV2

    active_policy_versions = SessionOrchestratorV2._policy_versions()
    policies = policy_registry or V2PolicyRegistry.from_environment()
    policies.register(
        active_policy_versions,
        SessionOrchestratorV2.restore,
    )
    eligible_goals = _goals(
        graph,
        available_targets,
        active_item_bank,
        active_pedagogy_catalog,
    )
    goals_by_id = {goal.goal_id: goal for goal in eligible_goals}
    if orchestrator_factory is None:
        def factory(
            factory_graph: GraphDocument,
            target_kc: str,
            profile: LearnerProfile,
            request: CreateSessionV2Request,
        ) -> tuple[Any, ContentModeView]:
            return _default_factory(
                factory_graph,
                target_kc,
                profile,
                request,
                item_bank=active_item_bank,
                pedagogy_catalog=active_pedagogy_catalog,
                evidence_trust_policy=evidence_trust_policy,
                widget_capabilities=active_widget_manifest,
            )
    else:
        factory = orchestrator_factory

    def apply_runtime_widget_capabilities(orchestrator: Any) -> None:
        setter = getattr(orchestrator, "set_runtime_widget_capabilities", None)
        if callable(setter):
            setter(active_widget_manifest)
    store = V2SessionStore(
        graph_nodes={node.id: node for node in graph.nodes},
        persistence=persistence,
        metrics_sink=metrics_sink,
        metric_dimensions=V2MetricDimensions(
            graph_version=str(graph.graph_version),
            item_bank_version=(
                active_item_bank.bank_version
                if active_item_bank is not None
                else "unavailable"
            ),
            pedagogy_catalog_version=(
                active_pedagogy_catalog.catalog_version
                if active_pedagogy_catalog is not None
                else "unavailable"
            ),
            policy_versions=tuple(sorted(active_policy_versions.items())),
            learner_parameter_version=(
                f"bkt-v{DEFAULT_PARAMS_V2.params_version}"
            ),
            capability_manifest_version=str(active_widget_manifest["version"]),
        ),
    )

    @app.middleware("http")
    async def v2_action_metrics(request: Request, call_next):
        """Count every action response, including unexpected framework 5xxs."""
        parts = request.url.path.strip("/").split("/")
        is_action = (
            request.method == "POST"
            and len(parts) == 5
            and parts[:3] == ["api", "v2", "sessions"]
            and parts[4] == "actions"
        )
        if is_action:
            store.record_metric("action_requests")
        try:
            result = await call_next(request)
        except Exception:
            if is_action:
                store.record_metric("action_5xx")
            raise
        if is_action and result.status_code >= 500:
            store.record_metric("action_5xx")
        return result

    router = APIRouter(prefix="/api/v2")

    def rollout_catalog(
        assignment: _RolloutAssignment,
        gate_snapshot: MutationGateSnapshot | None = None,
    ) -> GoalCatalog:
        observed_gate = gate_snapshot or observe_mutation_gate()
        percentage = flags.student_rollout_percent
        if not assignment.selected:
            reason = (
                "New pilot sessions are not currently being admitted."
                if percentage == 0
                else (
                    f"New pilot sessions are currently available to {percentage}% "
                    "of anonymous browsers; this browser is not in that cohort."
                )
            )
            return GoalCatalog(
                goals=[],
                rollout=CatalogRolloutView(
                    status="not_selected",
                    reason=reason,
                    percentage=percentage,
                ),
            )
        if not flags.student_stack_enabled or mutations_paused(observed_gate):
            return GoalCatalog(
                goals=[],
                rollout=CatalogRolloutView(
                    status="paused",
                    reason=(
                        "New pilot sessions are temporarily paused by a runtime "
                        "safety switch."
                    ),
                    percentage=percentage,
                ),
            )
        if not eligible_goals:
            return GoalCatalog(
                goals=[],
                rollout=CatalogRolloutView(
                    status="content_unavailable",
                    reason=(
                        "No pilot goal has completed the required reviewed-content "
                        "coverage."
                    ),
                    percentage=percentage,
                ),
            )
        return GoalCatalog(
            goals=eligible_goals,
            rollout=CatalogRolloutView(
                status="available",
                reason="This browser is included in the current pilot rollout.",
                percentage=percentage,
            ),
        )

    def assignment_for(request: Request) -> _RolloutAssignment:
        return _rollout_assignment(
            token_secret,
            flags.student_rollout_percent,
            request.cookies.get(_ROLLOUT_COOKIE_NAME),
        )

    def admission_error(
        assignment: _RolloutAssignment,
        request: Request,
        gate_snapshot: MutationGateSnapshot | None = None,
    ) -> JSONResponse | None:
        observed_gate = gate_snapshot or observe_mutation_gate()
        if (
            assignment.selected
            and flags.student_stack_enabled
            and not mutations_paused(observed_gate)
        ):
            return None
        if mutations_paused(observed_gate):
            result = mutation_paused_error("create")
            _set_rollout_cookie(result, assignment.cookie_value, request)
            return result
        catalog_view = rollout_catalog(assignment, observed_gate).rollout
        code = (
            "rollout_not_selected"
            if catalog_view.status == "not_selected"
            else "rollout_paused"
        )
        status = 403 if catalog_view.status == "not_selected" else 503
        result = _error(status, code, catalog_view.reason)
        store.record_metric(f"create_blocked_{catalog_view.status}")
        _set_rollout_cookie(result, assignment.cookie_value, request)
        return result

    def mutation_paused_error(
        operation: str,
        session: SessionView | None = None,
    ) -> JSONResponse:
        """Reject a new state change without disturbing committed receipts."""
        store.record_metric(f"mutation_paused_{operation}")
        return _error(
            503,
            "v2_mutations_paused",
            (
                "Session changes are temporarily paused for a safety check; "
                "retry this same request_id shortly."
            ),
            session,
            retryable=True,
        )

    def restore_durable(
        token_hash: str, bundle: dict[str, Any] | None = None
    ) -> V2SessionHandle | None:
        if persistence is None:
            return None
        try:
            bundle = bundle or persistence.resolve_resume(token_hash)
        except DurableLedgerMismatch as exc:
            store.record_metric(exc.metric)
            store.record_metric("commit_integrity_failures")
            raise SessionUnavailable("durable ledger integrity check failed") from exc
        except Exception as exc:
            raise SessionUnavailable("durable checkpoint cannot be read") from exc
        if bundle is None:
            return None
        checkpoint = bundle["checkpoint"]
        state = checkpoint.get("orchestrator")
        if state is None:
            raise SessionUnavailable("durable checkpoint cannot be restored")
        try:
            release = registry.resolve_checkpoint(state)
            policy_runtime = policies.resolve_checkpoint(state)
            orchestrator = policy_runtime.restore(
                release.graph,
                state,
                release.item_bank,
                release.pedagogy_catalog,
                evidence_trust_policy,
            )
            apply_runtime_widget_capabilities(orchestrator)
            return store.restore(
                orchestrator=orchestrator,
                checkpoint=checkpoint,
                receipts=bundle["receipts"],
                token_hash=token_hash,
            )
        except Exception as exc:
            raise SessionUnavailable("durable checkpoint cannot be restored") from exc

    def authorized_handle(
        request: Request,
        session_id: str | None = None,
        *,
        measure_resume: bool = False,
    ) -> tuple[V2SessionHandle | None, str | None, JSONResponse | None]:
        def outside_eligible_failure(outcome: str) -> None:
            if not measure_resume:
                return
            store.record_metric(f"resume_{outcome}")
            store.record_metric("resume_failures")

        def eligible_failure(*outcomes: str) -> None:
            if not measure_resume:
                return
            store.record_metric("resume_eligible_attempts")
            # Compatibility alias: unlike raw cookie attempts, this is the
            # denominator used by the rollout reliability gate.
            store.record_metric("resume_attempts")
            store.record_metric("resume_eligible_failures")
            store.record_metric("resume_failures")
            for outcome in outcomes:
                store.record_metric(outcome)

        def eligible_attempt() -> None:
            if not measure_resume:
                return
            store.record_metric("resume_eligible_attempts")
            store.record_metric("resume_attempts")

        raw = request.cookies.get(_COOKIE_NAME)
        if not raw:
            outside_eligible_failure("no_cookie")
            return None, None, _error(
                401, "resume_token_required", "no current anonymous session"
            )
        if measure_resume:
            store.record_metric("resume_cookie_attempts")
        if not _resume_token_well_formed(raw):
            outside_eligible_failure("invalid")
            return None, None, _error(
                401,
                "invalid_resume_token",
                "the anonymous session token is invalid",
            )
        hashed = _token_hash(raw)

        if persistence is None:
            try:
                handle = store.resolve_token(hashed)
            except ResumeTokenExpired:
                outside_eligible_failure("expired")
                return None, hashed, _error(
                    401,
                    "invalid_resume_token",
                    "the anonymous session token is invalid or expired",
                )
            except KeyError:
                outside_eligible_failure("invalid")
                return None, hashed, _error(
                    401,
                    "invalid_resume_token",
                    "the anonymous session token is invalid or expired",
                )
        else:
            # Durable status is authoritative over a potentially stale local
            # cache. Only a known-active token can enter the reliability gate.
            try:
                durable_status = persistence.resume_token_status(hashed)
            except Exception:  # noqa: BLE001 - authorization dependency failure
                outside_eligible_failure("status_failures")
                return None, hashed, _error(
                    503,
                    "session_restore_unavailable",
                    "the durable session could not be checked; retry shortly",
                )
            if durable_status != "active":
                store.forget_token(hashed)
                outside_eligible_failure(durable_status)
                return None, hashed, _error(
                    401,
                    "invalid_resume_token",
                    "the anonymous session token is invalid or expired",
                )

            try:
                bundle = persistence.resolve_resume(hashed)
            except DurableLedgerMismatch as exc:
                store.record_metric(exc.metric)
                store.record_metric("commit_integrity_failures")
                eligible_failure("resume_restore_failures")
                return None, hashed, _error(
                    503,
                    "session_restore_unavailable",
                    "the durable session failed its integrity check; retry shortly",
                )
            except Exception:
                eligible_failure("resume_restore_failures")
                return None, hashed, _error(
                    503,
                    "session_restore_unavailable",
                    "the durable session could not be read; retry shortly",
                )
            if bundle is None:
                store.forget_token(hashed)
                try:
                    durable_status = persistence.resume_token_status(hashed)
                except Exception:  # noqa: BLE001 - active status was already known
                    durable_status = "active"
                if durable_status == "active":
                    eligible_failure("resume_restore_failures")
                    return None, hashed, _error(
                        503,
                        "session_restore_unavailable",
                        "the durable session could not be restored; retry shortly",
                    )
                outside_eligible_failure(durable_status)
                return None, hashed, _error(
                    401,
                    "invalid_resume_token",
                    "the anonymous session token is invalid or expired",
                )

            try:
                handle = store.resolve_token(hashed)
            except (KeyError, ResumeTokenExpired):
                handle = None
            try:
                durable_view = SessionView.model_validate(
                    bundle["checkpoint"]["session_view"]
                )
                if (
                    handle is None
                    or durable_view.session_id != handle.session_id
                    or durable_view.revision != handle.revision
                ):
                    handle = restore_durable(hashed, bundle)
            except SessionUnavailable:
                eligible_failure("resume_restore_failures")
                return None, hashed, _error(
                    503,
                    "session_restore_unavailable",
                    "the durable session could not be restored; retry shortly",
                )
            except Exception:
                eligible_failure("resume_restore_failures")
                return None, hashed, _error(
                    503,
                    "session_restore_unavailable",
                    "the durable session could not be restored; retry shortly",
                )
            if handle is None:
                eligible_failure("resume_restore_failures")
                return None, hashed, _error(
                    503,
                    "session_restore_unavailable",
                    "the durable session could not be restored; retry shortly",
                )

        if session_id is not None and not hmac.compare_digest(
            handle.session_id, session_id
        ):
            outside_eligible_failure("session_mismatch")
            return None, hashed, _error(404, "session_not_found", "unknown session")
        eligible_attempt()
        return handle, hashed, None

    def refresh_resume(
        request: Request,
        response: Response,
        token_hash: str,
        *,
        measure_resume: bool = False,
    ) -> JSONResponse | None:
        """Roll durable/local expiry and refresh the browser cookie together."""
        def refresh_failure() -> None:
            if not measure_resume:
                return
            store.record_metric("resume_eligible_failures")
            store.record_metric("resume_failures")
            store.record_metric("resume_refresh_failures")

        try:
            active = store.refresh_token(token_hash)
        except SessionUnavailable:
            refresh_failure()
            return _error(
                503,
                "persistence_unavailable",
                "the anonymous session expiry could not be refreshed; retry shortly",
            )
        if not active:
            refresh_failure()
            return _error(
                401,
                "invalid_resume_token",
                "the anonymous session token is invalid or expired",
            )
        raw_token = request.cookies.get(_COOKIE_NAME)
        if raw_token is None:
            refresh_failure()
            return _error(
                401, "resume_token_required", "no current anonymous session"
            )
        _set_resume_cookie(response, raw_token, request)
        if measure_resume:
            store.record_metric("resume_successes")
        return None

    @router.get("/goals", response_model=GoalCatalog)
    def list_goals(request: Request, response: Response) -> GoalCatalog:
        assignment = assignment_for(request)
        _set_rollout_cookie(response, assignment.cookie_value, request)
        catalog = rollout_catalog(assignment)
        store.record_metric("catalog_requests")
        store.record_metric(f"catalog_{catalog.rollout.status}")
        return catalog

    @router.get("/capabilities", response_model=WidgetCapabilityManifestView)
    def capabilities() -> WidgetCapabilityManifestView:
        return WidgetCapabilityManifestView.model_validate(
            active_widget_manifest
        )

    @router.post(
        "/sessions/recover",
        response_model=RecoverSessionV2Response,
        responses={
            401: {"model": APIError},
            403: {"model": APIError},
            409: {"model": APIError},
            503: {"model": APIError},
        },
    )
    def recover_session(
        request_body: RecoverSessionV2Request,
        request: Request,
        response: Response,
    ) -> RecoverSessionV2Response | JSONResponse:
        """Recover a committed response without persisting its private payload.

        The browser keeps only the operation and cryptographically random
        request id in sessionStorage. Reset additionally requires the revoked
        HttpOnly cookie, so that cookie alone can never follow its successor.
        """

        if not _origin_allowed(request):
            return _error(403, "origin_not_allowed", "cross-origin mutation rejected")
        try:
            if request_body.operation == "create":
                replacement_session_id = store.recover_create(
                    request_body.request_id
                )
                replacement_raw_token = _create_resume_token(
                    token_secret, request_body.request_id
                )
            else:
                old_raw_token = request.cookies.get(_COOKIE_NAME)
                if old_raw_token is None:
                    return _error(
                        401,
                        "resume_token_required",
                        "the reset recovery requires its previous anonymous cookie",
                    )
                old_token_hash = _token_hash(old_raw_token)
                replacement_session_id = store.recover_reset_rotation(
                    old_token_hash, request_body.request_id
                )
                replacement_raw_token = _replacement_resume_token(
                    token_secret, old_token_hash, request_body.request_id
                )
        except SessionUnavailable:
            return _error(
                503,
                "recovery_unavailable",
                "the committed session could not be checked; retry shortly",
            )
        if replacement_session_id is None:
            return _error(
                409,
                "recovery_not_committed",
                "no active committed session matches this recovery request",
            )

        replacement_token_hash = _token_hash(replacement_raw_token)
        try:
            handle = store.resolve_token(replacement_token_hash)
        except KeyError:
            try:
                handle = restore_durable(replacement_token_hash)
            except SessionUnavailable:
                return _error(
                    503,
                    "session_restore_unavailable",
                    "the committed session could not be restored; retry shortly",
                )
        if handle is None or not hmac.compare_digest(
            handle.session_id, replacement_session_id
        ):
            return _error(
                409,
                "recovery_not_committed",
                "the committed replacement session is no longer active",
            )
        try:
            active = store.refresh_token(replacement_token_hash)
        except SessionUnavailable:
            return _error(
                503,
                "recovery_unavailable",
                "the replacement session could not be refreshed; retry shortly",
            )
        if not active:
            return _error(
                409,
                "recovery_not_committed",
                "the committed replacement session is no longer active",
            )
        _set_resume_cookie(response, replacement_raw_token, request)
        store.record_metric(f"{request_body.operation}_responses_recovered")
        return RecoverSessionV2Response(session_id=handle.session_id)

    @router.post(
        "/sessions",
        response_model=SessionView,
        responses={409: {"model": APIError}, 503: {"model": APIError}},
    )
    def create_session(
        request_body: CreateSessionV2Request, request: Request, response: Response
    ) -> SessionView | JSONResponse:
        if not _origin_allowed(request):
            return _error(403, "origin_not_allowed", "cross-origin mutation rejected")
        assignment = assignment_for(request)
        _set_rollout_cookie(response, assignment.cookie_value, request)
        request_payload = {
            "type": "create",
            **request_body.model_dump(mode="json"),
        }
        raw_token = _create_resume_token(token_secret, request_body.request_id)
        create_token_hash = _token_hash(raw_token)
        create_session_id = _create_session_id(
            token_secret, request_body.request_id
        )
        prior_learner_id = None
        prior_events: list[Any] = []
        prior_exposure_state: Any | None = None
        replace_handle: V2SessionHandle | None = None
        replace_token_hash: str | None = None

        # Creation retries must be recoverable even when the first response and
        # cookie were lost, or when an atomic terminal rollover revoked the old
        # cookie before its response reached the browser.
        try:
            local_replay = store.repeat_create_without_token(
                request_body.request_id, request_payload
            )
            if local_replay is not None:
                handle, repeated = local_replay
                if not store.owns(handle.session_id, create_token_hash):
                    return _error(
                        409,
                        "session_revoked",
                        "the original session is no longer resumable",
                    )
                if not store.refresh_token(create_token_hash):
                    return _error(
                        409,
                        "session_revoked",
                        "the original session is no longer resumable",
                    )
                _set_resume_cookie(response, raw_token, request)
                return repeated
            if persistence is not None:
                persisted = persistence.replay_create(
                    request_id=request_body.request_id,
                    payload_hash=store._payload_hash(request_payload),
                )
                if persisted is not None:
                    restored = restore_durable(create_token_hash)
                    if restored is None:
                        return _error(
                            409,
                            "session_revoked",
                            "the original session is no longer resumable",
                        )
                    if not store.refresh_token(create_token_hash):
                        return _error(
                            409,
                            "session_revoked",
                            "the original session is no longer resumable",
                        )
                    _set_resume_cookie(response, raw_token, request)
                    return SessionView.model_validate(persisted)
        except SessionConflict as exc:
            # The caller is not authenticated to the session named by a global
            # create receipt, so never disclose its transcript or context.
            return _error(409, exc.code, str(exc))
        except SessionUnavailable:
            return _error(
                503,
                "persistence_unavailable",
                "the creation receipt could not be checked; retry with the same request_id",
            )
        except Exception:
            return _error(
                503,
                "persistence_unavailable",
                "the creation receipt could not be checked; retry with the same request_id",
            )

        gate_snapshot = observe_mutation_gate()
        if mutations_paused(gate_snapshot):
            paused = mutation_paused_error("create")
            _set_rollout_cookie(paused, assignment.cookie_value, request)
            return paused

        current_raw = request.cookies.get(_COOKIE_NAME)
        if current_raw:
            current_hash = _token_hash(current_raw)
            current, _, current_error = authorized_handle(request)
            if current_error is not None:
                if current_error.status_code >= 500:
                    return current_error
                current = None
            if current is not None:
                if store.view(current).phase not in ("done", "stopped"):
                    return _error(
                        409,
                        "active_session_exists",
                        "reset or finish the current session before starting another",
                        store.view(current),
                    )
                blocked = admission_error(
                    assignment,
                    request,
                    gate_snapshot,
                )
                if blocked is not None:
                    return blocked
                prior_learner_id = getattr(
                    getattr(current.orchestrator, "learner", None),
                    "learner_id",
                    None,
                )
                prior_events = list(
                    getattr(
                        getattr(current.orchestrator, "learner", None),
                        "events",
                        (),
                    )
                )
                exposure_state = getattr(current.orchestrator, "exposure_state", None)
                if exposure_state is not None:
                    prior_exposure_state = exposure_state.model_copy(deep=True)
                replace_handle = current
                replace_token_hash = current_hash

        blocked = admission_error(assignment, request, gate_snapshot)
        if blocked is not None:
            return blocked
        goal = goals_by_id.get(request_body.goal_id)
        if goal is None:
            return _error(404, "goal_not_found", "unknown or unavailable goal")
        store.record_metric("create_admitted_by_rollout")

        profile = LearnerProfile(
            course=request_body.course, age_band=request_body.age_band
        )
        try:
            orchestrator, content_mode = factory(
                graph, goal.target_kc, profile, request_body
            )
            apply_runtime_widget_capabilities(orchestrator)
            remember_visible = getattr(
                orchestrator, "remember_visible_content", None
            )
            if callable(remember_visible):
                # Context is already learner-visible in the authoritative
                # SessionView, so it must gate the very first diagnostic item.
                remember_visible(request_body.context)
            seed_longitudinal = getattr(orchestrator, "seed_longitudinal", None)
            if prior_learner_id is not None and callable(seed_longitudinal):
                from datetime import datetime, timezone

                seed_kwargs = {
                    "as_of": datetime.now(timezone.utc),
                    "exposure_state": prior_exposure_state,
                }
                if isinstance(orchestrator, SessionOrchestratorV2):
                    seed_kwargs["evidence_trust_policy"] = (
                        evidence_trust_policy
                    )
                seed_longitudinal(
                    prior_learner_id,
                    prior_events,
                    **seed_kwargs,
                )
            interactions = orchestrator.begin()
            handle = store.create(
                orchestrator=orchestrator,
                goal=goal,
                profile=SessionProfile(
                    course=request_body.course, age_band=request_body.age_band
                ),
                context=request_body.context,
                content_mode=content_mode,
                interactions=interactions,
                token_hash=create_token_hash,
                request_id=request_body.request_id,
                request_payload=request_payload,
                session_id=create_session_id,
                replace_handle=replace_handle,
                replace_token_hash=replace_token_hash,
            )
        except DurableReceiptReplay as replay:
            try:
                restored = restore_durable(create_token_hash)
            except SessionUnavailable:
                return _error(
                    503,
                    "session_restore_unavailable",
                    "the committed session could not be restored; retry shortly",
                )
            if restored is None:
                return _error(
                    409,
                    "session_revoked",
                    "the committed session is no longer resumable",
                )
            _set_resume_cookie(response, raw_token, request)
            return replay.view
        except SessionConflict as exc:
            return _error(409, exc.code, str(exc), exc.view)
        except SessionRateLimited:
            return _error(
                429,
                "episode_limit",
                "this anonymous learner reached the rolling episode limit",
            )
        except KeyError:
            return _error(404, "goal_not_found", "unknown or unavailable goal")
        except SessionUnavailable:
            return _error(
                503,
                "persistence_unavailable",
                "the session could not be durably created; retry with the same request_id",
            )
        _set_resume_cookie(response, raw_token, request)
        return store.view(handle)

    @router.get("/sessions/current", response_model=SessionView)
    def get_current(
        request: Request, response: Response
    ) -> SessionView | JSONResponse:
        handle, token_hash, error = authorized_handle(
            request,
            measure_resume=True,
        )
        if error is not None:
            return error
        assert handle is not None
        assert token_hash is not None
        refresh_error = refresh_resume(
            request,
            response,
            token_hash,
            measure_resume=True,
        )
        if refresh_error is not None:
            return refresh_error
        return store.view(handle)

    @router.post("/sessions/current/reset", response_model=ResetResponse)
    def reset_current(
        request_body: ResetSessionV2Request,
        request: Request,
        response: Response,
    ) -> ResetResponse | JSONResponse:
        if not _origin_allowed(request):
            return _error(403, "origin_not_allowed", "cross-origin mutation rejected")
        raw_token = request.cookies.get(_COOKIE_NAME)
        if raw_token is None:
            return _error(401, "resume_token_required", "no current anonymous session")
        token_hash = _token_hash(raw_token)
        replacement_raw_token = _replacement_resume_token(
            token_secret, token_hash, request_body.request_id
        )
        replacement_token_hash = _token_hash(replacement_raw_token)
        try:
            replayed = store.replay_reset(token_hash, request_body)
        except SessionConflict as exc:
            return _error(409, exc.code, str(exc), exc.view)
        except SessionUnavailable:
            return _error(
                503,
                "persistence_unavailable",
                "the reset receipt could not be checked; retry with the same request_id",
            )
        if replayed is not None:
            try:
                if persistence is not None:
                    restore_durable(replacement_token_hash)
            except SessionUnavailable:
                return _error(
                    503,
                    "session_restore_unavailable",
                    "the replacement episode could not be restored; retry shortly",
                )
            _set_resume_cookie(response, replacement_raw_token, request)
            return replayed

        handle, token_hash, error = authorized_handle(request)
        if error is not None:
            if error.status_code >= 500:
                return error
            response.delete_cookie(_COOKIE_NAME, path="/api/v2")
            return error
        assert handle is not None
        assert token_hash is not None
        gate_snapshot = observe_mutation_gate()
        if mutations_paused(gate_snapshot):
            return mutation_paused_error("reset", store.view(handle))
        try:
            from datetime import datetime, timezone

            fresh_episode = getattr(handle.orchestrator, "fresh_episode", None)
            if not callable(fresh_episode):
                return _error(
                    409,
                    "reset_unavailable",
                    "this episode cannot be restarted safely",
                    store.view(handle),
                )
            replacement_orchestrator = fresh_episode(
                as_of=datetime.now(timezone.utc)
            )
            apply_runtime_widget_capabilities(replacement_orchestrator)
            remember_visible = getattr(
                replacement_orchestrator, "remember_visible_content", None
            )
            if callable(remember_visible):
                remember_visible(handle.context)
            replacement_interactions = replacement_orchestrator.begin()
            reset = store.reset(
                handle,
                token_hash,
                request_body,
                replacement_orchestrator=replacement_orchestrator,
                replacement_interactions=replacement_interactions,
                replacement_token_hash=replacement_token_hash,
            )
        except DurableResetReplay as replay:
            try:
                restore_durable(replacement_token_hash)
            except SessionUnavailable:
                return _error(
                    503,
                    "session_restore_unavailable",
                    "the replacement episode could not be restored; retry shortly",
                )
            reset = replay.response
        except SessionConflict as exc:
            return _error(409, exc.code, str(exc), exc.view or store.view(handle))
        except SessionRateLimited:
            return _error(
                429,
                "episode_limit",
                "this anonymous learner reached the rolling episode limit",
                store.view(handle),
            )
        except SessionUnavailable:
            return _error(
                503,
                "persistence_unavailable",
                "the reset could not be durably committed; retry with the same request_id",
            )
        _set_resume_cookie(response, replacement_raw_token, request)
        return reset

    @router.get("/sessions/{session_id}", response_model=SessionView)
    def get_session(
        session_id: str, request: Request, response: Response
    ) -> SessionView | JSONResponse:
        handle, token_hash, error = authorized_handle(
            request,
            session_id,
            measure_resume=True,
        )
        if error is not None:
            return error
        assert handle is not None
        assert token_hash is not None
        refresh_error = refresh_resume(
            request,
            response,
            token_hash,
            measure_resume=True,
        )
        if refresh_error is not None:
            return refresh_error
        return store.view(handle)

    @router.post(
        "/sessions/{session_id}/actions",
        response_model=SessionView,
        responses={
            403: {"model": APIError},
            409: {"model": APIError},
            429: {"model": APIError},
            503: {"model": APIError},
        },
    )
    def apply_action(
        session_id: str,
        action: SessionAction,
        request: Request,
        response: Response,
    ) -> SessionView | JSONResponse:
        if not _origin_allowed(request):
            return _error(403, "origin_not_allowed", "cross-origin mutation rejected")
        handle, token_hash, error = authorized_handle(request, session_id)
        if error is not None:
            return error
        assert handle is not None
        assert token_hash is not None
        try:
            # Receipt lookup always precedes the live gate so a control-plane
            # pause cannot turn a committed transport retry into a new error.
            replayed = store.replay_action(
                handle,
                action,
                token_hash=token_hash,
            )
            if replayed is not None:
                raw_token = request.cookies[_COOKIE_NAME]
                _set_resume_cookie(response, raw_token, request)
                return replayed
            gate_snapshot = observe_mutation_gate()
            if mutations_paused(gate_snapshot):
                return mutation_paused_error("action", store.view(handle))
            view = store.apply(session_id, action, token_hash=token_hash)
            raw_token = request.cookies[_COOKIE_NAME]
            _set_resume_cookie(response, raw_token, request)
            return view
        except DurableReceiptReplay as replay:
            if token_hash is not None:
                try:
                    restore_durable(token_hash)
                except SessionUnavailable:
                    return _error(
                        503,
                        "session_restore_unavailable",
                        "the committed action could not be restored; retry shortly",
                    )
            raw_token = request.cookies[_COOKIE_NAME]
            _set_resume_cookie(response, raw_token, request)
            return replay.view
        except SessionConflict as exc:
            if (
                persistence is not None
                and token_hash is not None
                and exc.view is not None
                and exc.view.revision > handle.revision
            ):
                try:
                    refreshed = restore_durable(token_hash)
                    if refreshed is not None:
                        exc.view = store.view(refreshed)
                except SessionUnavailable:
                    return _error(
                        503,
                        "session_restore_unavailable",
                        "the authoritative session could not be restored; retry shortly",
                    )
            return _error(409, exc.code, str(exc), exc.view or store.view(handle))
        except KeyError:
            return _error(404, "session_not_found", "unknown session")
        except SessionUnavailable:
            return _error(
                503,
                "persistence_unavailable",
                "the action was not committed; retry with the same request_id",
                store.view(handle),
            )
        except SessionIntegrityError:
            return _error(
                500,
                "session_integrity_failure",
                "the action was rejected because its atomic state delta was incomplete",
                store.view(handle),
            )
        except SessionRateLimited:
            return _error(
                429,
                "session_action_limit",
                "this episode reached its safe action limit; reset to begin a new episode",
                store.view(handle),
            )
        except RuntimeError as exc:
            return _error(409, "action_rejected", str(exc), store.view(handle))

    app.include_router(router)
    app.state.v2_store = store
    app.state.v2_persistence = persistence
    app.state.v2_goal_catalog = GoalCatalog(
        goals=eligible_goals,
        rollout=CatalogRolloutView(
            status="available" if eligible_goals else "content_unavailable",
            reason="Internal content-readiness view.",
            percentage=flags.student_rollout_percent,
        ),
    )
    app.state.v2_version_registry = registry
    app.state.v2_active_release = active_release
    app.state.v2_evidence_trust_registry = evidence_trust_policy
    app.state.v2_policy_registry = policies
    app.state.v2_feature_flags = flags

    def readiness_view() -> dict[str, Any]:
        """Observe the live gate without exposing provider failures or config."""

        gate_snapshot = observe_mutation_gate()
        effective_pause = mutations_paused(gate_snapshot)
        content_ready = bool(eligible_goals)
        return {
            "student_stack_enabled": flags.student_stack_enabled,
            "content_ready": content_ready,
            "mutations_paused": effective_pause,
            "accepting_mutations": (
                flags.student_stack_enabled
                and content_ready
                and not effective_pause
            ),
            "mutation_gate": mutation_gate_view(gate_snapshot),
            "durable_persistence": persistence is not None,
            "reviewed_goal_count": len(eligible_goals),
            "active_versions": {
                "graph": graph.graph_version,
                "item_bank": (
                    active_item_bank.bank_version
                    if active_item_bank is not None
                    else None
                ),
                "pedagogy_catalog": (
                    active_pedagogy_catalog.catalog_version
                    if active_pedagogy_catalog is not None
                    else None
                ),
                "policies": dict(sorted(active_policy_versions.items())),
                "learner_parameters": f"bkt-v{DEFAULT_PARAMS_V2.params_version}",
                "capability_manifest": active_widget_manifest["version"],
            },
        }

    app.state.v2_readiness_provider = readiness_view
    app.state.v2_readiness = readiness_view()
    return store
