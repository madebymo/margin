"""Durable recovery regressions for exposure-ledger ordering."""

from fastapi.testclient import TestClient

from tests.test_api_v2 import _answer, _create, _v2_app, _widget_response
from tutor.schemas.assessment import GuidedMappingSpec, GuidedSliderSpec
from tutor.api.v2_persistence import V2PersistenceService
from tutor.db.persistence import PersistenceService
from tutor.db.session import get_engine


def test_restart_after_capstone_remediation_reconciles_exposure_ledger():
    persistence = V2PersistenceService(
        PersistenceService(
            engine=get_engine("sqlite+pysqlite:///:memory:")
        ).engine
    )
    secret = b"durable-remediation-resume-secret-32-bytes"
    first = TestClient(
        _v2_app(
            persistence=persistence,
            resume_token_secret=secret,
        )
    )
    created, _ = _create(first)
    view = created.json()
    session_id = view["session_id"]

    for _ in range(2):
        response = first.post(
            f"/api/v2/sessions/{session_id}/actions",
            json=_answer(view, "0"),
        )
        assert response.status_code == 200
        view = response.json()

    assert view["phase"] == "teach"
    while view["phase"] == "teach":
        handle = first.app.state.v2_store.get(session_id)
        action = _answer(view, handle.orchestrator.pending_expected)
        if view["pending"]["input_mode"] == "widget":
            item = handle.orchestrator._item_for(
                handle.orchestrator.pending.reservation
            )
            spec = item.guided_interaction
            if isinstance(spec, GuidedSliderSpec):
                widget_response = {"value": spec.scoring.target}
            elif isinstance(spec, GuidedMappingSpec):
                widget_response = {"pairs": list(spec.scoring.correct_pairs)}
            else:  # pragma: no cover - released guided items are typed
                raise AssertionError("guided interaction has no reviewed scorer")
            action = _widget_response(view, widget_response)
        response = first.post(
            f"/api/v2/sessions/{session_id}/actions",
            json=action,
        )
        assert response.status_code == 200
        view = response.json()

    assert view["phase"] == "capstone"
    failed = first.post(
        f"/api/v2/sessions/{session_id}/actions",
        json=_answer(view, "0"),
    )
    assert failed.status_code == 200
    remediated_view = failed.json()
    assert remediated_view["phase"] == "capstone"
    raw_token = first.cookies.get("tutor_resume_v2")

    restarted = TestClient(
        _v2_app(
            persistence=persistence,
            resume_token_secret=secret,
        )
    )
    restarted.cookies.set("tutor_resume_v2", raw_token, path="/api/v2")
    restored = restarted.get("/api/v2/sessions/current")

    assert restored.status_code == 200
    assert restored.json() == remediated_view
