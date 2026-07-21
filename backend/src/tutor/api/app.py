"""FastAPI session service wrapping the SessionOrchestrator.

Run locally:
    uvicorn tutor.api.app:app --reload

Expected answers never leave the server: responses carry only displayable
interactions, phase, pending-item metadata, and the session summary.
"""

import logging
import os
from datetime import timedelta
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from tutor.api.diagnosis_shadow import (
    DiagnosisV2ShadowObserver,
    diagnosis_shadow_enabled_from_environment,
)
from tutor.api.runtime_plugins import (
    RuntimePluginError,
    build_runtime_plugin_from_environment,
)
from tutor.api.store import SessionStore
from tutor.api.v2 import install_v2_routes
from tutor.api.v2_controls import MutationGate, StaticMutationGate
from tutor.api.v2_features import V2FeatureFlags
from tutor.api.v2_metrics import MetricsSink
from tutor.api.v2_persistence import V2PersistenceService
from tutor.db.persistence import PersistenceService
from tutor.llm.client import LLMError
from tutor.orchestrator.machine import Interaction, SessionOrchestrator
from tutor.schemas.kc import GraphDocument
from tutor.schemas.learner import LearnerProfile
from tutor.seed.load_seed import load_graph

load_dotenv()

logger = logging.getLogger("tutor.api")

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_DIST_DIR = _STATIC_DIR / "dist"
_V2_METRICS_SINK_FACTORY_ENV = "TUTOR_V2_METRICS_SINK_FACTORY"
_V2_MUTATION_GATE_FACTORY_ENV = "TUTOR_V2_MUTATION_GATE_FACTORY"
_DEFAULT_DYNAMIC_MUTATION_GATE_MAX_AGE = timedelta(seconds=60)


class CreateSessionRequest(BaseModel):
    """Intake: target concept plus the two-question profile."""

    target_kc: str = "kc.int.u_substitution"
    course: str = "AP Calculus AB"
    age_band: str = "16-18"
    llm: bool = False
    provider: Literal["openai", "anthropic"] = "openai"


class AnswerRequest(BaseModel):
    """A student answer to the pending item."""

    answer: str = Field(min_length=1)


class PendingInfo(BaseModel):
    """What the session is waiting on (never includes the expected answer)."""

    kind: str | None = None
    kc_id: str | None = None


class TurnResponse(BaseModel):
    """One turn of tutor output plus session state."""

    session_id: str
    phase: str
    interactions: list[Interaction]
    pending: PendingInfo
    summary: dict
    llm_enabled: bool | None = None


class HintResponse(BaseModel):
    """Next rung of the hint ladder, if any."""

    hint: str | None


class WidgetAnswerRequest(BaseModel):
    """One widget attempt: the lesson interaction key plus the raw response."""

    key: str = Field(min_length=1)
    response: dict


class WidgetAnswerResponse(BaseModel):
    """Server-side verdict for a widget attempt."""

    correct: bool
    message: str


def _turn(
    session_id: str,
    orchestrator: SessionOrchestrator,
    interactions: list[Interaction],
    llm_enabled: bool | None = None,
) -> TurnResponse:
    return TurnResponse(
        session_id=session_id,
        phase=orchestrator.phase.value,
        interactions=interactions,
        pending=PendingInfo(
            kind=orchestrator.pending_kind, kc_id=orchestrator.pending_kc
        ),
        summary=orchestrator.summary(),
        llm_enabled=llm_enabled,
    )


def create_app(
    graph: GraphDocument | None = None,
    database_url: str | None = None,
    *,
    allow_v1_session_creation: bool | None = None,
    enable_diagnosis_v2_shadow: bool | None = None,
    v2_metrics_sink: MetricsSink | None = None,
    v2_mutation_gate: MutationGate | None = None,
    v2_mutation_gate_max_age: timedelta | None = None,
    v2_active_release_bundle: str | Path | None = None,
    v2_active_release_sha256: str | None = None,
) -> FastAPI:
    """Build the API app around one graph version and an in-memory store.

    Persistence activates when ``database_url`` or the ``DATABASE_URL`` env var
    is set; otherwise sessions are memory-only (unavailability degrades with a
    warning rather than failing startup).
    """
    resolved_graph = graph or load_graph()
    pilot_production = os.environ.get("TUTOR_PILOT_PRODUCTION", "").lower() in {
        "1",
        "true",
        "yes",
    }
    legacy_creation_enabled = (
        allow_v1_session_creation
        if allow_v1_session_creation is not None
        else os.environ.get("TUTOR_ALLOW_V1_SESSION_CREATION", "").lower()
        in {"1", "true", "yes"}
    )
    if pilot_production and legacy_creation_enabled:
        raise RuntimeError(
            "TUTOR_PILOT_PRODUCTION forbids new legacy v1 session creation"
        )
    if pilot_production and os.environ.get("TUTOR_ALLOW_MISSING_ORIGIN") == "1":
        raise RuntimeError(
            "TUTOR_PILOT_PRODUCTION forbids the missing-origin escape hatch"
        )
    app = FastAPI(title="Adaptive Math Tutor", version="0.1.0")
    store = SessionStore()
    persistence: PersistenceService | None = None
    resolved_url = database_url or os.environ.get("DATABASE_URL")
    if pilot_production and (
        not resolved_url or not resolved_url.lower().startswith("postgresql")
    ):
        raise RuntimeError(
            "TUTOR_PILOT_PRODUCTION requires a PostgreSQL DATABASE_URL"
        )
    if resolved_url:
        try:
            persistence = PersistenceService(url=resolved_url)
            logger.info("persistence enabled")
        except Exception as exc:  # noqa: BLE001 — degrade to memory-only
            if pilot_production:
                raise RuntimeError(
                    "pilot production persistence could not be initialized"
                ) from exc
            logger.warning("persistence unavailable (%s); sessions are memory-only", exc)
    app.state.store = store
    app.state.graph = resolved_graph
    app.state.persistence = persistence
    v2_flags = V2FeatureFlags.from_environment()
    app.state.v2_feature_flags = v2_flags
    resolved_v2_metrics_sink = v2_metrics_sink
    if v2_flags.api_session_v2 and resolved_v2_metrics_sink is None:
        metrics_factory_configured = bool(
            os.environ.get(_V2_METRICS_SINK_FACTORY_ENV, "").strip()
        )
        try:
            resolved_v2_metrics_sink = build_runtime_plugin_from_environment(
                _V2_METRICS_SINK_FACTORY_ENV,
                contract=MetricsSink,
                contract_name="MetricsSink",
                required=pilot_production,
            )
        except RuntimePluginError as exc:
            # A configured exporter is explicit deployment intent. Refuse to
            # start instead of silently dropping fleet-level safety signals,
            # while keeping provider names, configuration, and errors private.
            logger.error(
                "v2 metrics sink plugin startup failed error_type=%s",
                type(exc).__name__,
            )
            if pilot_production and not metrics_factory_configured:
                raise RuntimeError(
                    "TUTOR_PILOT_PRODUCTION requires a v2 fleet metrics sink"
                ) from None
            raise RuntimeError(
                "configured v2 fleet metrics sink could not be initialized"
            ) from None
    resolved_v2_mutation_gate = v2_mutation_gate
    resolved_v2_mutation_gate_max_age = v2_mutation_gate_max_age
    if v2_flags.api_session_v2 and resolved_v2_mutation_gate is None:
        try:
            resolved_v2_mutation_gate = build_runtime_plugin_from_environment(
                _V2_MUTATION_GATE_FACTORY_ENV,
                contract=MutationGate,
                contract_name="MutationGate",
            )
        except RuntimePluginError as exc:
            # The plugin specification and underlying exception may contain
            # deployment details. Log only the bounded failure class and hold
            # the serving path closed until startup configuration is repaired.
            logger.error(
                "v2 mutation gate plugin failed closed error_type=%s",
                type(exc).__name__,
            )
            resolved_v2_mutation_gate = StaticMutationGate(
                True,
                revision="plugin-load-failed-v1",
                source="fail_closed",
            )
            resolved_v2_mutation_gate_max_age = None
        else:
            if (
                resolved_v2_mutation_gate is not None
                and resolved_v2_mutation_gate_max_age is None
            ):
                resolved_v2_mutation_gate_max_age = (
                    _DEFAULT_DYNAMIC_MUTATION_GATE_MAX_AGE
                )
    diagnosis_shadow = DiagnosisV2ShadowObserver(
        resolved_graph,
        enabled=(
            enable_diagnosis_v2_shadow
            if enable_diagnosis_v2_shadow is not None
            else diagnosis_shadow_enabled_from_environment()
        ),
    )
    app.state.diagnosis_v2_shadow = diagnosis_shadow

    def _observe_shadow(boundary: str, callback, *args) -> None:
        """Keep optional metrics observation outside the v1 serving path."""
        try:
            callback(*args)
        except Exception as exc:  # noqa: BLE001 - shadow failures never affect v1
            diagnosis_shadow.note_boundary_failure(boundary)
            logger.warning(
                "diagnosis-v2 shadow wrapper failed boundary=%s error_type=%s",
                boundary,
                type(exc).__name__,
            )

    def _get_session(session_id: str) -> SessionOrchestrator:
        try:
            return store.get(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="unknown session") from None

    @app.get("/healthz")
    def healthz() -> dict:
        """Liveness check."""
        readiness_provider = getattr(app.state, "v2_readiness_provider", None)
        v2_readiness = (
            readiness_provider()
            if callable(readiness_provider)
            else getattr(
                app.state,
                "v2_readiness",
                {
                    "student_stack_enabled": False,
                    "content_ready": False,
                    "fleet_metrics_configured": False,
                    "mutations_paused": True,
                    "accepting_mutations": False,
                    "durable_persistence": False,
                    "reviewed_goal_count": 0,
                    "active_versions": {
                        "graph": resolved_graph.graph_version,
                        "item_bank": None,
                        "pedagogy_catalog": None,
                        "policies": {},
                        "learner_parameters": None,
                        "capability_manifest": None,
                    },
                },
            )
        )
        app.state.v2_readiness = v2_readiness
        return {
            "status": "ok",
            "sessions": len(store),
            "persistence": persistence is not None,
            "v2_sessions": len(getattr(app.state, "v2_store", ())),
            "v2_goals": len(
                getattr(getattr(app.state, "v2_goal_catalog", None), "goals", ())
            ),
            "v1_session_creation": legacy_creation_enabled,
            "diagnosis_v2_shadow": diagnosis_shadow.metrics_snapshot(),
            "v2_features": v2_flags.as_dict(),
            "v2_readiness": v2_readiness,
            "v2_metrics": (
                app.state.v2_store.metrics_snapshot()
                if hasattr(app.state, "v2_store")
                else {
                    "counters": {},
                    "actions_by_item_id": {},
                    "resume_outcomes": {
                        "cookie_attempts": 0,
                        "eligible_attempts": 0,
                        "eligible_failures": 0,
                        "no_cookie": 0,
                        "invalid": 0,
                        "expired": 0,
                        "session_mismatch": 0,
                        "restore_failures": 0,
                        "refresh_failures": 0,
                        "successes": 0,
                    },
                    "rollout_gates": {
                        "resume_success_rate": None,
                        "action_5xx_rate": None,
                        "duplicate_advances_detected": 0,
                        "missing_evidence_detected": 0,
                        "commit_integrity_failures": 0,
                    },
                }
            ),
        }

    @app.get("/", response_class=FileResponse)
    def index() -> FileResponse:
        """Serve the production frontend entry point."""
        return FileResponse(_DIST_DIR / "index.html")

    @app.post("/sessions", response_model=TurnResponse)
    def create_session(request: CreateSessionRequest) -> TurnResponse:
        """Start a session: intake -> first diagnostic probe."""
        if not legacy_creation_enabled:
            raise HTTPException(
                status_code=410,
                detail=(
                    "new legacy sessions are disabled; use the reviewed "
                    "/api/v2 session catalog"
                ),
            )
        profile = LearnerProfile(course=request.course, age_band=request.age_band)
        diagnostician = lesson_writer = interaction_generator = evaluator = None
        llm_enabled = False
        if request.llm:
            try:
                from tutor.llm.factory import build_llm_ports

                ports = build_llm_ports(resolved_graph, profile, request.provider)
                diagnostician = ports.diagnostician
                lesson_writer = ports.lesson_writer
                interaction_generator = ports.interaction_generator
                evaluator = ports.evaluator
                llm_enabled = True
            except LLMError as exc:
                logger.warning("LLM ports unavailable (%s); using templates", exc)
        try:
            orchestrator = SessionOrchestrator(
                resolved_graph,
                request.target_kc,
                profile,
                diagnostician=diagnostician,
                lesson_writer=lesson_writer,
                interaction_generator=interaction_generator,
                evaluator=evaluator,
                persistence=persistence,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        interactions = orchestrator.begin()
        session_id = store.create(orchestrator)
        _observe_shadow("start_wrapper", diagnosis_shadow.start, session_id, orchestrator)
        return _turn(session_id, orchestrator, interactions, llm_enabled)

    @app.post("/sessions/{session_id}/answer", response_model=TurnResponse)
    def answer(session_id: str, request: AnswerRequest) -> TurnResponse:
        """Submit an answer to the pending item."""
        orchestrator = _get_session(session_id)
        event_count = len(orchestrator.learner.events)
        try:
            interactions = orchestrator.submit(request.answer)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        new_events = orchestrator.learner.events[event_count:]
        if len(new_events) == 1:
            _observe_shadow(
                "answer_wrapper",
                diagnosis_shadow.observe_answer,
                session_id,
                new_events[0],
                orchestrator,
            )
        elif not new_events:
            diagnosis_shadow.note_unscored_submission()
        elif diagnosis_shadow.enabled:
            diagnosis_shadow.note_boundary_failure("answer_event")
        return _turn(session_id, orchestrator, interactions)

    @app.post("/sessions/{session_id}/hint", response_model=HintResponse)
    def hint(session_id: str) -> HintResponse:
        """Serve the next hint (marks the eventual answer as assisted)."""
        orchestrator = _get_session(session_id)
        return HintResponse(hint=orchestrator.hint())

    @app.post("/sessions/{session_id}/widget", response_model=WidgetAnswerResponse)
    def answer_widget(session_id: str, request: WidgetAnswerRequest) -> WidgetAnswerResponse:
        """Score a widget attempt (authoritative, server-side)."""
        orchestrator = _get_session(session_id)
        try:
            correct, message = orchestrator.answer_widget(request.key, request.response)
        except KeyError:
            raise HTTPException(status_code=404, detail="unknown widget") from None
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        return WidgetAnswerResponse(correct=correct, message=message)

    @app.get("/sessions/{session_id}", response_model=TurnResponse)
    def get_session(session_id: str) -> TurnResponse:
        """Current state without advancing the session."""
        orchestrator = _get_session(session_id)
        return _turn(session_id, orchestrator, [])

    if v2_flags.api_session_v2:
        install_v2_routes(
            app,
            resolved_graph,
            persistence=(
                V2PersistenceService(persistence.engine)
                if persistence is not None
                else None
            ),
            feature_flags=v2_flags,
            metrics_sink=resolved_v2_metrics_sink,
            mutation_gate=resolved_v2_mutation_gate,
            mutation_gate_max_age=resolved_v2_mutation_gate_max_age,
            active_release_bundle=v2_active_release_bundle,
            active_release_sha256=v2_active_release_sha256,
        )

    # Keep the mount last so API routes retain precedence. ``check_dir=False``
    # lets backend modules import before the frontend's first production build.
    app.mount(
        "/static",
        StaticFiles(directory=_DIST_DIR, check_dir=False),
        name="static",
    )

    return app


app = create_app()
