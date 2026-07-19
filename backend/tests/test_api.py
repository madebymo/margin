"""Session API: lifecycle over HTTP against the in-process store."""

import pytest
from fastapi.testclient import TestClient

from tutor.api.app import create_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


def _orchestrator(client: TestClient, session_id: str):
    return client.app.state.store.get(session_id)


def test_healthz(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_index_serves_chat_ui(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Adaptive Math Tutor" in response.text


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
