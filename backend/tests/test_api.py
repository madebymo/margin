"""Session API: lifecycle over HTTP against the in-process store."""

import mimetypes
import re
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient

from tutor.api.app import create_app

_DIST_INDEX = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "tutor"
    / "api"
    / "static"
    / "dist"
    / "index.html"
)
_STATIC_REFERENCE = re.compile(r"""(?:src|href)=["'](/static/[^"']+)["']""")


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app(allow_v1_session_creation=True))


def test_new_v1_sessions_are_disabled_by_default():
    guarded = TestClient(create_app())
    response = guarded.post(
        "/sessions", json={"target_kc": "kc.der.chain_rule"}
    )

    assert response.status_code == 410


def test_pilot_production_fails_closed_when_postgres_is_unavailable(monkeypatch):
    class UnavailablePersistence:
        def __init__(self, *args, **kwargs):
            raise OSError("database unavailable")

    monkeypatch.setenv("TUTOR_PILOT_PRODUCTION", "1")
    monkeypatch.setattr("tutor.api.app.PersistenceService", UnavailablePersistence)

    with pytest.raises(RuntimeError, match="persistence could not be initialized"):
        create_app(database_url="postgresql://pilot.invalid/tutor")


def test_pilot_production_forbids_legacy_session_escape_hatch(monkeypatch):
    monkeypatch.setenv("TUTOR_PILOT_PRODUCTION", "1")
    monkeypatch.setenv("TUTOR_ALLOW_V1_SESSION_CREATION", "1")

    with pytest.raises(RuntimeError, match="forbids new legacy v1"):
        create_app(database_url="postgresql://pilot.invalid/tutor")

    with pytest.raises(RuntimeError, match="forbids new legacy v1"):
        create_app(
            database_url="postgresql://pilot.invalid/tutor",
            allow_v1_session_creation=True,
        )


def test_pilot_production_forbids_missing_origin_escape_hatch(monkeypatch):
    monkeypatch.setenv("TUTOR_PILOT_PRODUCTION", "1")
    monkeypatch.setenv("TUTOR_ALLOW_MISSING_ORIGIN", "1")

    with pytest.raises(RuntimeError, match="forbids the missing-origin"):
        create_app(database_url="postgresql://pilot.invalid/tutor")


def test_api_v2_can_be_disabled_independently(monkeypatch):
    monkeypatch.setenv("TUTOR_ENABLE_API_SESSION_V2", "0")
    guarded = TestClient(create_app())

    assert guarded.get("/api/v2/goals").status_code == 404
    assert guarded.get("/healthz").json()["v2_features"]["api_session_v2"] is False


def _orchestrator(client: TestClient, session_id: str):
    return client.app.state.store.get(session_id)


def test_healthz(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["v2_sessions"] == 0
    assert response.json()["v2_goals"] >= 0


def test_index_serves_production_bundle_and_all_referenced_assets(client):
    assert _DIST_INDEX.is_file(), "run `npm --prefix frontend run build`"

    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "margin" in response.text
    static_references = sorted(set(_STATIC_REFERENCE.findall(response.text)))
    assert static_references

    for reference in static_references:
        assert re.search(r"/assets/.+-[A-Za-z0-9_-]{8}\.(?:css|js)$", reference)
        asset = client.get(reference)
        assert asset.status_code == 200, reference
        expected_type = mimetypes.guess_type(urlsplit(reference).path)[0]
        assert expected_type is not None, reference
        actual_type = asset.headers["content-type"].split(";", 1)[0]
        assert actual_type == expected_type, reference
        assert asset.content, reference


def test_create_session_starts_diagnosis(client):
    response = client.post("/sessions", json={"target_kc": "kc.der.chain_rule"})
    assert response.status_code == 200
    data = response.json()
    assert data["phase"] == "diagnose"
    assert data["llm_enabled"] is None or data["llm_enabled"] is False
    kinds = [interaction["kind"] for interaction in data["interactions"]]
    assert "probe" in kinds
    assert data["pending"]["kind"] == "probe"
    # expected answers never leave the server
    assert "expected" not in data["interactions"][-1]


def test_unknown_target_is_404(client):
    response = client.post("/sessions", json={"target_kc": "kc.der.not_a_node"})
    assert response.status_code == 404


def test_unknown_session_is_404(client):
    response = client.post("/sessions/deadbeef/answer", json={"answer": "x"})
    assert response.status_code == 404


def test_hint_marks_next_answer_assisted(client):
    created = client.post("/sessions", json={}).json()
    session_id = created["session_id"]
    hint = client.post(f"/sessions/{session_id}/hint")
    assert hint.status_code == 200
    assert hint.json()["hint"]
    orchestrator = _orchestrator(client, session_id)
    response = client.post(
        f"/sessions/{session_id}/answer", json={"answer": orchestrator.pending_expected}
    )
    assert response.status_code == 200
    assert orchestrator.learner.events[-1].assisted is True


def test_full_session_to_done_over_http(client):
    created = client.post("/sessions", json={"target_kc": "kc.int.u_substitution"}).json()
    session_id = created["session_id"]
    orchestrator = _orchestrator(client, session_id)

    data = created
    guard = 0
    while data["phase"] not in ("done", "stopped"):
        guard += 1
        assert guard < 100, "session did not terminate"
        response = client.post(
            f"/sessions/{session_id}/answer",
            json={"answer": orchestrator.pending_expected},
        )
        assert response.status_code == 200
        data = response.json()

    assert data["phase"] == "done"
    assert data["summary"]["probes_used"] == 1  # all-correct student short-circuits
    assert data["pending"]["kind"] is None

    # the session is over: further answers conflict
    response = client.post(f"/sessions/{session_id}/answer", json={"answer": "x"})
    assert response.status_code == 409

    # state endpoint still readable
    state = client.get(f"/sessions/{session_id}")
    assert state.status_code == 200
    assert state.json()["phase"] == "done"
