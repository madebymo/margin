"""Guarded v2 runtime for Playwright and the public engineering demo.

The packaged bank remains an unreleased draft. This module is importable only
with an explicit test or submission-demo guard and installs the narrow fixture
used by backend tests. The demo identifies this content as synthetic and never
changes the fail-closed behavior of ``tutor.api.app``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from tests.v2_helpers import (
    approved_power_rule_catalog,
    approved_power_rule_stress_bank,
    power_rule_only_graph,
)
from tutor.api.v2 import install_v2_routes
from tutor.api.v2_features import V2FeatureFlags
from tutor.api.http_safety import (
    HttpSecurityHeadersMiddleware,
    RequestBodyLimitMiddleware,
    trusted_hosts_from_environment,
)
from tutor.api.v2_schemas import ContentModeView, CreateSessionV2Request
from tutor.orchestrator.session_v2 import SessionOrchestratorV2
from tutor.schemas.kc import GraphDocument
from tutor.schemas.learner import LearnerProfile

_IS_TEST = os.environ.get("TUTOR_E2E_TEST_APP") == "1"
_IS_SUBMISSION_DEMO = os.environ.get("TUTOR_SUBMISSION_DEMO") == "1"
if not (_IS_TEST or _IS_SUBMISSION_DEMO):
    raise RuntimeError(
        "browser_v2_app requires TUTOR_E2E_TEST_APP=1 or "
        "TUTOR_SUBMISSION_DEMO=1"
    )
if _IS_SUBMISSION_DEMO:
    _RESUME_SECRET = os.environ.get("TUTOR_RESUME_TOKEN_SECRET", "")
    if len(_RESUME_SECRET.encode("utf-8")) < 32:
        raise RuntimeError(
            "TUTOR_SUBMISSION_DEMO requires TUTOR_RESUME_TOKEN_SECRET "
            "with at least 32 bytes"
        )
else:
    _RESUME_SECRET = "browser-test-only-resume-secret-32-bytes"

_DIST_DIR = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "tutor"
    / "api"
    / "static"
    / "dist"
)
_BASE_GRAPH = power_rule_only_graph()
_GRAPH = _BASE_GRAPH.model_copy(
    update={
        "nodes": [
            _BASE_GRAPH.nodes[0].model_copy(
                update={
                    "name": "Power rule — engineering demo",
                    "description": (
                        "Synthetic test content for demonstrating the learning "
                        "system; not a released curriculum."
                    ),
                }
            )
        ]
    }
)
_BANK = approved_power_rule_stress_bank()
_PEDAGOGY_CATALOG = approved_power_rule_catalog()


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
            pedagogy_catalog=_PEDAGOGY_CATALOG,
        ),
        ContentModeView(
            requested=request.content_mode,
            effective="curated",
            fallback_reason=fallback_reason,
        ),
    )


app = FastAPI(title="Margin engineering demo")
app.add_middleware(RequestBodyLimitMiddleware)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=list(
        trusted_hosts_from_environment(
            pilot_production=False,
            environ=os.environ,
        )
    ),
)
app.add_middleware(
    HttpSecurityHeadersMiddleware,
    secure_transport=_IS_SUBMISSION_DEMO,
)
install_v2_routes(
    app,
    _GRAPH,
    orchestrator_factory=_orchestrator_factory,
    available_targets=("kc.der.power_rule",),
    item_bank=_BANK,
    pedagogy_catalog=_PEDAGOGY_CATALOG,
    resume_token_secret=_RESUME_SECRET,
    feature_flags=V2FeatureFlags(student_rollout_percent=100),
)


@app.get("/livez", include_in_schema=False)
def livez() -> dict[str, str]:
    return {"status": "ok", "mode": "submission_demo" if _IS_SUBMISSION_DEMO else "test"}


@app.get("/", response_class=FileResponse, include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(_DIST_DIR / "index.html")


app.mount(
    "/static",
    StaticFiles(directory=_DIST_DIR, check_dir=True),
    name="static",
)
