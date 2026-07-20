"""Version-registry and pinned durable-restore regression tests."""

from __future__ import annotations

import json
import sys
from types import ModuleType

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tutor.api.v2 import install_v2_routes
from tutor.api.v2_persistence import V2PersistenceService
from tutor.api.v2_versions import (
    V2PolicyRegistry,
    V2VersionConflict,
    V2VersionRegistry,
    V2VersionUnavailable,
)
from tutor.db.persistence import PersistenceService
from tutor.db.session import get_engine
from tutor.schemas.assessment import ItemBankDocument
from tutor.schemas.kc import GraphDocument

from tests.v2_helpers import approved_power_rule_bank, power_rule_only_graph


def _release(
    graph_version: int,
    bank_version: str,
    *,
    node_name: str | None = None,
) -> tuple[GraphDocument, ItemBankDocument]:
    graph_payload = power_rule_only_graph().model_dump(mode="json")
    graph_payload["graph_version"] = graph_version
    if node_name is not None:
        graph_payload["nodes"][0]["name"] = node_name
    graph = GraphDocument.model_validate(graph_payload)
    bank_payload = approved_power_rule_bank().model_dump(mode="json")
    bank_payload["graph_version"] = graph_version
    bank_payload["bank_version"] = bank_version
    bank = ItemBankDocument.model_validate(bank_payload)
    return graph, bank


def _app(
    graph: GraphDocument,
    bank: ItemBankDocument,
    persistence: V2PersistenceService,
    registry: V2VersionRegistry | None,
) -> FastAPI:
    app = FastAPI()
    app.state.persistence = persistence
    install_v2_routes(
        app,
        graph,
        persistence=persistence,
        available_targets=("kc.der.power_rule",),
        item_bank=bank,
        version_registry=registry,
    )
    return app


def test_registry_rejects_version_aliasing_and_incompatible_pairs():
    graph_v1, bank_v1 = _release(1, "power-bank-v1")
    registry = V2VersionRegistry([(graph_v1, bank_v1)])

    changed_graph_payload = graph_v1.model_dump(mode="json")
    changed_graph_payload["nodes"][0]["name"] = "Changed under the same version"
    changed_graph = GraphDocument.model_validate(changed_graph_payload)
    with pytest.raises(V2VersionConflict):
        registry.register(changed_graph, bank_v1)

    changed_bank_payload = bank_v1.model_dump(mode="json")
    changed_bank_payload["items"][0]["difficulty"] = (
        "stretch"
        if changed_bank_payload["items"][0]["difficulty"] != "stretch"
        else "foundation"
    )
    changed_bank = ItemBankDocument.model_validate(changed_bank_payload)
    with pytest.raises(V2VersionConflict):
        registry.register(graph_v1, changed_bank)

    graph_v2, _ = _release(2, "unused-bank-v2")
    with pytest.raises(V2VersionConflict):
        registry.register(graph_v2, bank_v1)

    with pytest.raises(V2VersionUnavailable):
        registry.resolve(1, "not-retained")


def test_policy_registry_dispatches_exact_pins_and_rejects_aliasing():
    registry = V2PolicyRegistry()
    versions = {
        "diagnosis": "diagnosis-v2.1",
        "lesson": "lesson-v2.0",
    }

    def restore_old(graph, checkpoint, item_bank):
        return graph, checkpoint, item_bank

    runtime = registry.register(versions, restore_old)
    assert registry.resolve_checkpoint({"policy_versions": versions}) == runtime
    assert registry.register(versions, restore_old) == runtime

    with pytest.raises(V2VersionConflict):
        registry.register(versions, lambda graph, checkpoint, item_bank: None)
    with pytest.raises(V2VersionUnavailable):
        registry.resolve_checkpoint(
            {"policy_versions": {**versions, "lesson": "lesson-v3"}}
        )


def test_policy_registry_loads_retained_runtime_modules_from_environment(monkeypatch):
    module_name = "tests._retained_v2_policy_fixture"
    module = ModuleType(module_name)
    versions = {"diagnosis": "diagnosis-v1-retained", "lesson": "lesson-v1"}

    def restore_old(graph, checkpoint, item_bank):
        return graph, checkpoint, item_bank

    def register_v2_policy_runtimes(registry):
        registry.register(versions, restore_old)

    module.register_v2_policy_runtimes = register_v2_policy_runtimes
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setenv("TUTOR_V2_POLICY_RUNTIME_MODULES", module_name)

    registry = V2PolicyRegistry.from_environment()

    assert registry.resolve_checkpoint({"policy_versions": versions}).restore is restore_old


def test_policy_registry_fails_startup_for_invalid_retained_module(monkeypatch):
    module_name = "tests._invalid_retained_v2_policy_fixture"
    monkeypatch.setitem(sys.modules, module_name, ModuleType(module_name))
    monkeypatch.setenv("TUTOR_V2_POLICY_RUNTIME_MODULES", module_name)

    with pytest.raises(RuntimeError, match="register_v2_policy_runtimes"):
        V2PolicyRegistry.from_environment()


def test_new_deployment_restores_checkpoint_with_older_pinned_release(
    tmp_path,
    monkeypatch,
):
    graph_v1, bank_v1 = _release(1, "power-bank-v1")
    graph_v2, bank_v2 = _release(
        2,
        "power-bank-v2",
        node_name="Renamed power rule in the active graph",
    )
    registry = V2VersionRegistry([(graph_v1, bank_v1)])
    persistence = V2PersistenceService(
        PersistenceService(
            engine=get_engine("sqlite+pysqlite:///:memory:")
        ).engine
    )

    old_client = TestClient(_app(graph_v1, bank_v1, persistence, registry))
    created = old_client.post(
        "/api/v2/sessions",
        json={
            "request_id": "00000000-0000-4000-8000-000000000001",
            "goal_id": "goal.der.power_rule",
        },
    )
    assert created.status_code == 200
    original = created.json()
    token = old_client.cookies.get("tutor_resume_v2")
    assert token

    # A fresh deployment loads retained immutable releases from configuration,
    # then registers its new active pair.
    (tmp_path / "power-v1.json").write_text(
        json.dumps(
            {
                "graph": graph_v1.model_dump(mode="json"),
                "item_bank": bank_v1.model_dump(mode="json"),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TUTOR_V2_RELEASE_REGISTRY_DIR", str(tmp_path))
    new_app = _app(graph_v2, bank_v2, persistence, None)
    new_client = TestClient(new_app)
    new_client.cookies.set("tutor_resume_v2", token, path="/api/v2")

    restored = new_client.get("/api/v2/sessions/current")
    assert restored.status_code == 200
    assert restored.json() == original
    restored_orchestrator = new_app.state.v2_store.get(
        original["session_id"]
    ).orchestrator
    checkpoint = restored_orchestrator.export_checkpoint()
    assert checkpoint["graph_version"] == graph_v1.graph_version
    assert checkpoint["item_bank_version"] == bank_v1.bank_version
    deployed_registry = new_app.state.v2_version_registry
    assert deployed_registry.graph_versions == (1, 2)
    assert deployed_registry.item_bank_versions == (
        "power-bank-v1",
        "power-bank-v2",
    )

    continued = new_client.post(
        f"/api/v2/sessions/{original['session_id']}/actions",
        json={
            "type": "request_hint",
            "request_id": "00000000-0000-4000-8000-000000000002",
            "expected_revision": original["revision"],
            "pending_key": original["pending"]["key"],
        },
    )
    assert continued.status_code == 200
    assert continued.json()["pending"]["skill_name"] == original["pending"]["skill_name"]
    assert continued.json()["pending"]["skill_name"] != graph_v2.nodes[0].name
