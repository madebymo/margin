"""Legacy diagnosis-v2 shadowing is observational, private, and fail-safe."""

import json

from fastapi.testclient import TestClient

from tutor.api.app import create_app
from tutor.orchestrator.diagnosis_v2 import DiagnosisControllerV2


def _without_session_id(payload: dict) -> dict:
    return {key: value for key, value in payload.items() if key != "session_id"}


def test_shadow_is_opt_in_from_environment(monkeypatch):
    monkeypatch.setenv("TUTOR_ENABLE_DIAGNOSIS_V2_SHADOW", "1")

    client = TestClient(create_app(allow_v1_session_creation=True))

    shadow = client.get("/healthz").json()["diagnosis_v2_shadow"]
    assert shadow["enabled"] is True
    assert shadow["counters"] == {}


def test_shadow_never_changes_legacy_routing_and_keeps_metrics_private():
    baseline = TestClient(
        create_app(
            allow_v1_session_creation=True,
            enable_diagnosis_v2_shadow=False,
        )
    )
    shadowed = TestClient(
        create_app(
            allow_v1_session_creation=True,
            enable_diagnosis_v2_shadow=True,
        )
    )
    target = "kc.der.chain_rule"
    baseline_view = baseline.post("/sessions", json={"target_kc": target}).json()
    shadowed_view = shadowed.post("/sessions", json={"target_kc": target}).json()
    assert _without_session_id(shadowed_view) == _without_session_id(baseline_view)

    baseline_id = baseline_view["session_id"]
    shadowed_id = shadowed_view["session_id"]
    private_wrong_answer = "student-private-shadow-answer"
    guard = 0
    while baseline_view["phase"] not in {"done", "stopped"}:
        guard += 1
        assert guard < 100
        baseline_machine = baseline.app.state.store.get(baseline_id)
        shadowed_machine = shadowed.app.state.store.get(shadowed_id)
        assert shadowed_machine.pending_expected == baseline_machine.pending_expected
        answer = private_wrong_answer if guard == 1 else baseline_machine.pending_expected

        baseline_response = baseline.post(
            f"/sessions/{baseline_id}/answer",
            json={"answer": answer},
        )
        shadowed_response = shadowed.post(
            f"/sessions/{shadowed_id}/answer",
            json={"answer": answer},
        )
        assert baseline_response.status_code == 200
        assert shadowed_response.status_code == 200
        baseline_view = baseline_response.json()
        shadowed_view = shadowed_response.json()
        assert _without_session_id(shadowed_view) == _without_session_id(baseline_view)

    metrics = shadowed.get("/healthz").json()["diagnosis_v2_shadow"]
    assert metrics["counters"]["sessions_started"] == 1
    assert metrics["counters"]["next_probe_matches"] >= 1
    assert metrics["counters"]["next_probe_divergences"] >= 1
    serialized_metrics = json.dumps(metrics, sort_keys=True)
    assert private_wrong_answer not in serialized_metrics
    assert baseline_id not in serialized_metrics
    assert shadowed_id not in serialized_metrics


def test_shadow_start_failure_never_breaks_session(monkeypatch):
    def fail_next_probe(self):
        raise RuntimeError("shadow-only failure")

    monkeypatch.setattr(DiagnosisControllerV2, "next_probe", fail_next_probe)
    client = TestClient(
        create_app(
            allow_v1_session_creation=True,
            enable_diagnosis_v2_shadow=True,
        )
    )

    response = client.post("/sessions", json={"target_kc": "kc.der.chain_rule"})

    assert response.status_code == 200
    assert response.json()["phase"] == "diagnose"
    counters = client.get("/healthz").json()["diagnosis_v2_shadow"]["counters"]
    assert counters["observer_failures"] == 1
    assert counters["start_failures"] == 1


def test_shadow_answer_failure_never_breaks_session(monkeypatch):
    client = TestClient(
        create_app(
            allow_v1_session_creation=True,
            enable_diagnosis_v2_shadow=True,
        )
    )
    created = client.post(
        "/sessions",
        json={"target_kc": "kc.der.chain_rule"},
    ).json()
    session_id = created["session_id"]
    orchestrator = client.app.state.store.get(session_id)

    def fail_record_result(self, observation):
        raise RuntimeError("shadow-only failure")

    monkeypatch.setattr(DiagnosisControllerV2, "record_result", fail_record_result)
    response = client.post(
        f"/sessions/{session_id}/answer",
        json={"answer": orchestrator.pending_expected},
    )

    assert response.status_code == 200
    counters = client.get("/healthz").json()["diagnosis_v2_shadow"]["counters"]
    assert counters["observer_failures"] == 1
    assert counters["answer_failures"] == 1
