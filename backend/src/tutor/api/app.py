"""FastAPI session service wrapping the SessionOrchestrator.

Run locally:
    uvicorn tutor.api.app:app --reload

Expected answers never leave the server: responses carry only displayable
interactions, phase, pending-item metadata, and the session summary.
"""

import logging
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from tutor.api.store import SessionStore
from tutor.db.persistence import PersistenceService
from tutor.llm.client import LLMError
from tutor.orchestrator.machine import Interaction, SessionOrchestrator
from tutor.schemas.kc import GraphDocument
from tutor.schemas.learner import LearnerProfile
from tutor.seed.load_seed import load_graph

logger = logging.getLogger("tutor.api")

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_DIST_DIR = _STATIC_DIR / "dist"


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
    graph: GraphDocument | None = None, database_url: str | None = None
) -> FastAPI:
    """Build the API app around one graph version and an in-memory store.

    Persistence activates when ``database_url`` or the ``DATABASE_URL`` env var
    is set; otherwise sessions are memory-only (unavailability degrades with a
    warning rather than failing startup).
    """
    resolved_graph = graph or load_graph()
    app = FastAPI(title="Adaptive Math Tutor", version="0.1.0")
    store = SessionStore()
    persistence: PersistenceService | None = None
    resolved_url = database_url or os.environ.get("DATABASE_URL")
    if resolved_url:
        try:
            persistence = PersistenceService(url=resolved_url)
            logger.info("persistence enabled")
        except Exception as exc:  # noqa: BLE001 — degrade to memory-only
            logger.warning("persistence unavailable (%s); sessions are memory-only", exc)
    app.state.store = store
    app.state.graph = resolved_graph
    app.state.persistence = persistence

    def _get_session(session_id: str) -> SessionOrchestrator:
        try:
            return store.get(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="unknown session") from None

    @app.get("/healthz")
    def healthz() -> dict:
        """Liveness check."""
        return {
            "status": "ok",
            "sessions": len(store),
            "persistence": persistence is not None,
        }

    @app.get("/", response_class=FileResponse)
    def index() -> FileResponse:
        """Serve the production frontend entry point."""
        return FileResponse(_DIST_DIR / "index.html")

    @app.post("/sessions", response_model=TurnResponse)
    def create_session(request: CreateSessionRequest) -> TurnResponse:
        """Start a session: intake -> first diagnostic probe."""
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
        return _turn(session_id, orchestrator, interactions, llm_enabled)

    @app.post("/sessions/{session_id}/answer", response_model=TurnResponse)
    def answer(session_id: str, request: AnswerRequest) -> TurnResponse:
        """Submit an answer to the pending item."""
        orchestrator = _get_session(session_id)
        try:
            interactions = orchestrator.submit(request.answer)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
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

    # Keep the mount last so API routes retain precedence. ``check_dir=False``
    # lets backend modules import before the frontend's first production build.
    app.mount(
        "/static",
        StaticFiles(directory=_DIST_DIR, check_dir=False),
        name="static",
    )

    return app


app = create_app()
