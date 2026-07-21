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
from tutor.learner.evidence_trust import ReviewedEvidenceTrustRegistry
from tutor.schemas.assessment import ItemBankDocument
from tutor.schemas.kc import GraphDocument
from tutor.schemas.pedagogy import PedagogyPackCatalog

from tests.v2_helpers import (
    approved_power_rule_episode_bank,
    approved_power_rule_catalog,
    power_rule_only_graph,
)


def _release(
    graph_version: int,
    bank_version: str,
    *,
    catalog_version: str | None = None,
    node_name: str | None = None,
) -> tuple[GraphDocument, ItemBankDocument, PedagogyPackCatalog]:
    graph_payload = power_rule_only_graph().model_dump(mode="json")
    graph_payload["graph_version"] = graph_version
    if node_name is not None:
        graph_payload["nodes"][0]["name"] = node_name
    graph = GraphDocument.model_validate(graph_payload)
    bank_payload = approved_power_rule_episode_bank().model_dump(mode="json")
    bank_payload["graph_version"] = graph_version
    bank_payload["bank_version"] = bank_version
    bank = ItemBankDocument.model_validate(bank_payload)
    catalog = approved_power_rule_catalog(
        graph_version=graph_version,
        catalog_version=catalog_version or f"power-pedagogy-v{graph_version}",
    )
    return graph, bank, catalog


def _app(
    graph: GraphDocument,
    bank: ItemBankDocument,
    catalog: PedagogyPackCatalog,
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
        pedagogy_catalog=catalog,
        version_registry=registry,
    )
    return app


def test_registry_rejects_version_aliasing_and_incompatible_pairs():
    graph_v1, bank_v1, catalog_v1 = _release(1, "power-bank-v1")
    registry = V2VersionRegistry([(graph_v1, bank_v1, catalog_v1)])

    changed_graph_payload = graph_v1.model_dump(mode="json")
    changed_graph_payload["nodes"][0]["name"] = "Changed under the same version"
    changed_graph = GraphDocument.model_validate(changed_graph_payload)
    with pytest.raises(V2VersionConflict):
        registry.register(changed_graph, bank_v1, catalog_v1)

    changed_bank_payload = bank_v1.model_dump(mode="json")
    changed_bank_payload["items"][0]["difficulty"] = (
        "stretch"
        if changed_bank_payload["items"][0]["difficulty"] != "stretch"
        else "foundation"
    )
    changed_bank = ItemBankDocument.model_validate(changed_bank_payload)
    with pytest.raises(V2VersionConflict):
        registry.register(graph_v1, changed_bank, catalog_v1)

    changed_catalog_payload = catalog_v1.model_dump(mode="json")
    changed_catalog_payload["published_by"] = "another release manager"
    changed_catalog = PedagogyPackCatalog.model_validate(changed_catalog_payload)
    with pytest.raises(V2VersionConflict):
        registry.register(graph_v1, bank_v1, changed_catalog)

    graph_v2, _, _ = _release(2, "unused-bank-v2")
    with pytest.raises(V2VersionConflict):
        registry.register(graph_v2, bank_v1, catalog_v1)

    with pytest.raises(V2VersionUnavailable):
        registry.resolve(1, "not-retained", catalog_v1.catalog_version)


def test_registry_never_assembles_an_unregistered_cross_product_release():
    graph, bank_v1, catalog_v1 = _release(
        1,
        "power-bank-v1",
        catalog_version="power-pedagogy-v1",
    )
    _, bank_v2, catalog_v2 = _release(
        1,
        "power-bank-v2",
        catalog_version="power-pedagogy-v2",
    )
    registry = V2VersionRegistry(
        [
            (graph, bank_v1, catalog_v1),
            (graph, bank_v2, catalog_v2),
        ]
    )

    with pytest.raises(V2VersionUnavailable, match="never registered"):
        registry.resolve(
            graph.graph_version,
            bank_v1.bank_version,
            catalog_v2.catalog_version,
        )

    with pytest.raises(V2VersionUnavailable, match="pedagogy-catalog"):
        registry.resolve_checkpoint(
            {
                "graph_version": graph.graph_version,
                "item_bank_version": bank_v1.bank_version,
            }
        )


def test_registry_snapshots_cannot_be_mutated_through_callers_or_resolved_views():
    graph, bank, catalog = _release(1, "immutable-power-bank-v1")
    original_name = graph.nodes[0].name
    original_item_count = len(bank.items)
    original_sources = list(catalog.packs[0].sources)
    registry = V2VersionRegistry([(graph, bank, catalog)])

    graph.nodes[0].name = "caller-mutated graph"
    bank.items.clear()
    catalog.packs[0].sources.append("caller-mutated source")

    first = registry.resolve(
        1,
        "immutable-power-bank-v1",
        catalog.catalog_version,
    )
    assert first.graph.nodes[0].name == original_name
    assert len(first.item_bank.items) == original_item_count
    assert first.pedagogy_catalog.packs[0].sources == original_sources

    first.graph.nodes[0].name = "resolved-view mutation"
    first.item_bank.items.clear()
    first.pedagogy_catalog.packs[0].sources.append("resolved mutation")
    second = registry.resolve(
        1,
        "immutable-power-bank-v1",
        catalog.catalog_version,
    )
    assert second.graph.nodes[0].name == original_name
    assert len(second.item_bank.items) == original_item_count
    assert second.pedagogy_catalog.packs[0].sources == original_sources


def test_policy_registry_dispatches_exact_pins_and_rejects_aliasing():
    registry = V2PolicyRegistry()
    trust = ReviewedEvidenceTrustRegistry()
    versions = {
        "diagnosis": "diagnosis-v2.1",
        "lesson": "lesson-v2.0",
    }

    def restore_old(
        graph,
        checkpoint,
        item_bank,
        pedagogy_catalog,
        evidence_trust_policy,
    ):
        return (
            graph,
            checkpoint,
            item_bank,
            pedagogy_catalog,
            evidence_trust_policy,
        )

    runtime = registry.register(
        versions,
        restore_old,
        checkpoint_schema_versions=(3,),
    )
    assert registry.resolve_checkpoint({"policy_versions": versions}) == runtime
    assert (
        registry.register(
            versions,
            restore_old,
            checkpoint_schema_versions=(3,),
        )
        == runtime
    )
    assert (
        registry.resolve_restoration_checkpoint(
            {"schema_version": 3, "policy_versions": versions}
        )
        == runtime
    )
    assert runtime.restore(None, {}, None, None, trust)[-1] is trust

    with pytest.raises(V2VersionUnavailable, match="schema"):
        registry.resolve_restoration_checkpoint(
            {"schema_version": 4, "policy_versions": versions}
        )

    with pytest.raises(V2VersionConflict):
        registry.register(
            versions,
            lambda graph, checkpoint, item_bank, pedagogy_catalog, trust: None,
        )
    with pytest.raises(V2VersionUnavailable):
        registry.resolve_checkpoint(
            {"policy_versions": {**versions, "lesson": "lesson-v3"}}
        )


def test_policy_registry_loads_retained_runtime_modules_from_environment(monkeypatch):
    module_name = "tests._retained_v2_policy_fixture"
    module = ModuleType(module_name)
    versions = {"diagnosis": "diagnosis-v1-retained", "lesson": "lesson-v1"}

    def restore_old(
        graph,
        checkpoint,
        item_bank,
        pedagogy_catalog,
        evidence_trust_policy,
    ):
        return (
            graph,
            checkpoint,
            item_bank,
            pedagogy_catalog,
            evidence_trust_policy,
        )

    def register_v2_policy_runtimes(registry):
        registry.register(
            versions,
            restore_old,
            checkpoint_schema_versions=(3,),
        )

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


def test_release_directory_rejects_legacy_two_component_bundles(tmp_path):
    graph, bank, _ = _release(1, "power-bank-v1")
    (tmp_path / "legacy-v1.json").write_text(
        json.dumps(
            {
                "graph": graph.model_dump(mode="json"),
                "item_bank": bank.model_dump(mode="json"),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema version 2"):
        V2VersionRegistry.from_release_directory(tmp_path)


def test_new_deployment_restores_checkpoint_with_older_pinned_release(
    tmp_path,
    monkeypatch,
):
    graph_v1, bank_v1, catalog_v1 = _release(1, "power-bank-v1")
    graph_v2, bank_v2, catalog_v2 = _release(
        2,
        "power-bank-v2",
        node_name="Renamed power rule in the active graph",
    )
    registry = V2VersionRegistry([(graph_v1, bank_v1, catalog_v1)])
    persistence = V2PersistenceService(
        PersistenceService(
            engine=get_engine("sqlite+pysqlite:///:memory:")
        ).engine
    )

    old_client = TestClient(
        _app(graph_v1, bank_v1, catalog_v1, persistence, registry)
    )
    created = old_client.post(
        "/api/v2/sessions",
        json={
            "request_id": "00000000-0000-4000-8000-000000000001",
            "goal_id": "goal.der.power_rule",
        },
    )
    assert created.status_code == 200
    initial = created.json()
    old_handle = old_client.app.state.v2_store.get(initial["session_id"])
    answered = old_client.post(
        f"/api/v2/sessions/{initial['session_id']}/actions",
        json={
            "type": "answer",
            "request_id": "00000000-0000-4000-8000-000000000002",
            "expected_revision": initial["revision"],
            "pending_key": initial["pending"]["key"],
            "answer": old_handle.orchestrator.pending_expected,
        },
    )
    assert answered.status_code == 200
    original = answered.json()
    token = old_client.cookies.get("tutor_resume_v2")
    assert token

    # A fresh deployment loads retained immutable releases from configuration,
    # then registers its new active pair.
    (tmp_path / "power-v1.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "graph": graph_v1.model_dump(mode="json"),
                "item_bank": bank_v1.model_dump(mode="json"),
                "pedagogy_catalog": catalog_v1.model_dump(mode="json"),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TUTOR_V2_RELEASE_REGISTRY_DIR", str(tmp_path))
    new_app = _app(graph_v2, bank_v2, catalog_v2, persistence, None)
    new_client = TestClient(new_app)
    new_client.cookies.set("tutor_resume_v2", token, path="/api/v2")

    restored = new_client.get("/api/v2/sessions/current")
    assert restored.status_code == 200
    assert restored.json() == original
    restored_orchestrator = new_app.state.v2_store.get(
        original["session_id"]
    ).orchestrator
    trust = new_app.state.v2_evidence_trust_registry
    assert trust.release_versions == (
        (1, "power-bank-v1", "power-pedagogy-v1"),
        (2, "power-bank-v2", "power-pedagogy-v2"),
    )
    assert restored_orchestrator.evidence_trust_policy is trust
    assert restored_orchestrator.learner.recent_independent_counts(
        "kc.der.power_rule"
    ) == (1, 0)
    checkpoint = restored_orchestrator.export_checkpoint()
    assert checkpoint["graph_version"] == graph_v1.graph_version
    assert checkpoint["item_bank_version"] == bank_v1.bank_version
    assert (
        checkpoint["pedagogy_catalog_version"]
        == catalog_v1.catalog_version
    )
    deployed_registry = new_app.state.v2_version_registry
    assert deployed_registry.graph_versions == (1, 2)
    assert deployed_registry.item_bank_versions == (
        "power-bank-v1",
        "power-bank-v2",
    )
    assert deployed_registry.pedagogy_catalog_versions == (
        "power-pedagogy-v1",
        "power-pedagogy-v2",
    )
    assert deployed_registry.release_versions == (
        (1, "power-bank-v1", "power-pedagogy-v1"),
        (2, "power-bank-v2", "power-pedagogy-v2"),
    )

    continued = new_client.post(
        f"/api/v2/sessions/{original['session_id']}/actions",
        json={
            "type": "request_hint",
            "request_id": "00000000-0000-4000-8000-000000000003",
            "expected_revision": original["revision"],
            "pending_key": original["pending"]["key"],
        },
    )
    assert continued.status_code == 200
    assert continued.json()["pending"]["skill_name"] == original["pending"]["skill_name"]
    assert continued.json()["pending"]["skill_name"] != graph_v2.nodes[0].name
