"""Test-only reviewed v2 runtime for Playwright journeys.

The packaged bank remains an unreleased draft. This module is importable only
with an explicit test guard and installs the narrow approved fixture used by
backend tests; it is never imported by ``tutor.api.app``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from tests.v2_helpers import approved_power_rule_bank, power_rule_only_graph
from tutor.api.v2 import install_v2_routes
from tutor.api.v2_features import V2FeatureFlags
from tutor.api.v2_schemas import ContentModeView, CreateSessionV2Request
from tutor.orchestrator.session_v2 import SessionOrchestratorV2
from tutor.schemas.kc import GraphDocument
from tutor.schemas.learner import LearnerProfile

if os.environ.get("TUTOR_E2E_TEST_APP") != "1":
    raise RuntimeError("browser_v2_app requires TUTOR_E2E_TEST_APP=1")

_DIST_DIR = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "tutor"
    / "api"
    / "static"
    / "dist"
)
_GRAPH = power_rule_only_graph()
_BANK = approved_power_rule_bank()


def _orchestrator_factory(
    graph: GraphDocument,
    target_kc: str,
    profile: LearnerProfile,
    request: CreateSessionV2Request,
) -> tuple[Any, ContentModeView]:
    fallback_reason = None
    if request.content_mode == "llm_coaching":
        fallback_reason = (
            "LLM coaching is unavailable in this test runtime; using curated content."
        )
    return (
        SessionOrchestratorV2(
            graph,
            target_kc,
            profile,
            item_bank=_BANK,
            probe_budget=2,
        ),
        ContentModeView(
            requested=request.content_mode,
            effective="curated",
            fallback_reason=fallback_reason,
        ),
    )


app = FastAPI(title="Adaptive Math Tutor browser-test runtime")
install_v2_routes(
    app,
    _GRAPH,
    orchestrator_factory=_orchestrator_factory,
    available_targets=("kc.der.power_rule",),
    item_bank=_BANK,
    resume_token_secret=b"browser-test-only-resume-secret-32-bytes",
    feature_flags=V2FeatureFlags(student_rollout_percent=100),
)


@app.get("/", response_class=FileResponse, include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(_DIST_DIR / "index.html")


app.mount(
    "/static",
    StaticFiles(directory=_DIST_DIR, check_dir=True),
    name="static",
)
