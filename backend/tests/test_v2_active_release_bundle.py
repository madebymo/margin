"""Startup selection for one immutable, reviewed v2 content release."""

from __future__ import annotations

import hashlib
import json
import logging
import traceback
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tutor.api.app import create_app
from tutor.api.v2 import install_v2_routes
from tutor.api.v2_features import V2FeatureFlags
from tutor.api.v2_versions import (
    V2_ACTIVE_RELEASE_BUNDLE_ENV,
    V2_ACTIVE_RELEASE_SHA256_ENV,
    V2VersionConflict,
)
from tutor.schemas.assessment import ItemBankDocument
from tutor.schemas.kc import GraphDocument
from tutor.schemas.pedagogy import PedagogyPackCatalog

from tests.v2_helpers import (
    approved_power_rule_episode_bank,
    approved_power_rule_catalog,
    power_rule_only_graph,
)

_TARGET_KC = "kc.der.product_quotient"
_GOAL_ID = "goal.der.product_quotient"


def _release(
    graph_version: int,
    bank_version: str,
    catalog_version: str,
) -> tuple[GraphDocument, ItemBankDocument, PedagogyPackCatalog]:
    """Build narrow, explicitly test-trusted content for a public pilot goal."""

    graph_payload = power_rule_only_graph().model_dump(mode="json")
    graph_payload["graph_version"] = graph_version
    graph_payload["nodes"][0].update(
        {
            "id": _TARGET_KC,
            "name": "Product and quotient rules",
            "description": "Differentiate products and quotients of functions.",
        }
    )
    graph = GraphDocument.model_validate(graph_payload)

    bank_payload = approved_power_rule_episode_bank().model_dump(mode="json")
    bank_payload["graph_version"] = graph_version
    bank_payload["bank_version"] = bank_version
    bank_payload["released_kcs"] = [_TARGET_KC]
    for item in bank_payload["items"]:
        item["kc_id"] = _TARGET_KC
    bank = ItemBankDocument.model_validate(bank_payload)

    catalog_payload = approved_power_rule_catalog(
        graph_version=graph_version,
        catalog_version=catalog_version,
    ).model_dump(mode="json")
    catalog_payload["packs"][0]["kc_id"] = _TARGET_KC
    catalog = PedagogyPackCatalog.model_validate(catalog_payload)
    return graph, bank, catalog


def _write_bundle(
    path: Path,
    release: tuple[GraphDocument, ItemBankDocument, PedagogyPackCatalog],
) -> Path:
    graph, bank, catalog = release
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "graph": graph.model_dump(mode="json"),
                "item_bank": bank.model_dump(mode="json"),
                "pedagogy_catalog": catalog.model_dump(mode="json"),
            }
        ),
        encoding="utf-8",
    )
    return path


def test_environment_bundle_is_active_exposes_only_valid_goal_and_pins_versions(
    tmp_path,
    monkeypatch,
):
    release = _release(41, "active-bank-v41", "active-pedagogy-v41")
    bundle = _write_bundle(tmp_path / "active.json", release)
    monkeypatch.setenv(V2_ACTIVE_RELEASE_BUNDLE_ENV, str(bundle))
    monkeypatch.setenv(
        V2_ACTIVE_RELEASE_SHA256_ENV,
        hashlib.sha256(bundle.read_bytes()).hexdigest(),
    )

    app = create_app()
    client = TestClient(app)

    catalog = client.get("/api/v2/goals")
    assert catalog.status_code == 200
    assert [goal["goal_id"] for goal in catalog.json()["goals"]] == [_GOAL_ID]

    created = client.post(
        "/api/v2/sessions",
        json={"request_id": str(uuid4()), "goal_id": _GOAL_ID},
    )
    assert created.status_code == 200
    handle = app.state.v2_store.get(created.json()["session_id"])
    checkpoint = handle.orchestrator.export_checkpoint()
    assert checkpoint["graph_version"] == 41
    assert checkpoint["item_bank_version"] == "active-bank-v41"
    assert checkpoint["pedagogy_catalog_version"] == "active-pedagogy-v41"
    active_versions = app.state.v2_readiness["active_versions"]
    assert active_versions["graph"] == 41
    assert active_versions["item_bank"] == "active-bank-v41"
    assert active_versions["pedagogy_catalog"] == "active-pedagogy-v41"


def test_explicit_bundle_argument_wins_over_environment(tmp_path, monkeypatch):
    explicit = _write_bundle(
        tmp_path / "explicit.json",
        _release(51, "explicit-bank-v51", "explicit-pedagogy-v51"),
    )
    monkeypatch.setenv(
        V2_ACTIVE_RELEASE_BUNDLE_ENV,
        str(tmp_path / "environment-bundle-does-not-exist.json"),
    )

    app = create_app(v2_active_release_bundle=explicit)

    assert app.state.v2_active_release.graph.graph_version == 51
    assert app.state.v2_active_release.item_bank.bank_version == "explicit-bank-v51"


def test_exact_sha256_pin_is_verified_before_bundle_activation(tmp_path):
    bundle = _write_bundle(
        tmp_path / "sha-pinned.json",
        _release(52, "sha-bank-v52", "sha-pedagogy-v52"),
    )
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()

    app = create_app(
        v2_active_release_bundle=bundle,
        v2_active_release_sha256=digest,
    )
    assert app.state.v2_active_release.graph.graph_version == 52

    with pytest.raises(V2VersionConflict) as exc_info:
        create_app(
            v2_active_release_bundle=bundle,
            v2_active_release_sha256="0" * 64,
        )
    assert str(bundle) not in str(exc_info.value)


def test_pilot_production_requires_sha_pin_for_selected_bundle(
    tmp_path,
    monkeypatch,
):
    bundle = _write_bundle(
        tmp_path / "pilot.json",
        _release(53, "pilot-bank-v53", "pilot-pedagogy-v53"),
    )
    monkeypatch.setenv("TUTOR_PILOT_PRODUCTION", "1")
    monkeypatch.delenv(V2_ACTIVE_RELEASE_SHA256_ENV, raising=False)

    with pytest.raises(RuntimeError, match="SHA-256 pin"):
        install_v2_routes(
            FastAPI(),
            power_rule_only_graph(),
            active_release_bundle=bundle,
            resume_token_secret=b"test-resume-token-secret-32-bytes",
            feature_flags=V2FeatureFlags(),
        )


@pytest.mark.parametrize("failure", ["malformed", "cross_version", "unreleased"])
def test_configured_bundle_failures_abort_startup(tmp_path, failure):
    path = tmp_path / f"{failure}.json"
    if failure == "malformed":
        path.write_text('{"schema_version": 2, "graph":', encoding="utf-8")
    else:
        graph, bank, catalog = _release(
            61,
            f"{failure}-bank-v61",
            f"{failure}-pedagogy-v61",
        )
        bank_payload = bank.model_dump(mode="json")
        if failure == "cross_version":
            bank_payload["graph_version"] = 62
        else:
            bank_payload["released_kcs"] = []
        path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "graph": graph.model_dump(mode="json"),
                    "item_bank": bank_payload,
                    "pedagogy_catalog": catalog.model_dump(mode="json"),
                }
            ),
            encoding="utf-8",
        )

    with pytest.raises((ValueError, V2VersionConflict)):
        create_app(v2_active_release_bundle=path)


def test_active_bundle_merges_with_retained_history_and_evidence_trust(
    tmp_path,
    monkeypatch,
):
    retained_dir = tmp_path / "retained"
    retained_dir.mkdir()
    old_release = _release(71, "old-bank-v71", "old-pedagogy-v71")
    _write_bundle(retained_dir / "old.json", old_release)
    active = _write_bundle(
        tmp_path / "active.json",
        _release(72, "active-bank-v72", "active-pedagogy-v72"),
    )
    monkeypatch.setenv("TUTOR_V2_RELEASE_REGISTRY_DIR", str(retained_dir))

    app = create_app(v2_active_release_bundle=active)
    registry = app.state.v2_version_registry

    restored = registry.resolve(71, "old-bank-v71", "old-pedagogy-v71")
    assert restored.graph.graph_version == 71
    expected_releases = (
        (71, "old-bank-v71", "old-pedagogy-v71"),
        (72, "active-bank-v72", "active-pedagogy-v72"),
    )
    assert registry.release_versions == expected_releases
    assert app.state.v2_evidence_trust_registry.release_versions == expected_releases


def test_active_bundle_cannot_cross_product_retained_components(
    tmp_path,
    monkeypatch,
):
    retained_dir = tmp_path / "retained"
    retained_dir.mkdir()
    graph, bank_a, catalog_a = _release(81, "bank-a-v81", "catalog-a-v81")
    _, bank_b, catalog_b = _release(81, "bank-b-v81", "catalog-b-v81")
    _write_bundle(retained_dir / "a.json", (graph, bank_a, catalog_a))
    _write_bundle(retained_dir / "b.json", (graph, bank_b, catalog_b))
    cross_product = _write_bundle(
        tmp_path / "cross-product.json",
        (graph, bank_a, catalog_b),
    )
    monkeypatch.setenv("TUTOR_V2_RELEASE_REGISTRY_DIR", str(retained_dir))

    with pytest.raises(V2VersionConflict, match="cross-product"):
        create_app(v2_active_release_bundle=cross_product)


def test_invalid_bundle_payload_is_not_written_to_logs(tmp_path, caplog):
    sentinel = "RAW-EXPECTED-ANSWER-MUST-NOT-APPEAR"
    bundle = tmp_path / "invalid.json"
    bundle.write_text(
        json.dumps({"schema_version": 2, "private_content": sentinel}),
        encoding="utf-8",
    )

    with caplog.at_level(logging.DEBUG), pytest.raises(ValueError):
        create_app(v2_active_release_bundle=bundle)

    assert sentinel not in caplog.text


def test_invalid_nested_content_is_suppressed_from_startup_traceback(tmp_path):
    sentinel = "RAW-EXPECTED-ANSWER-MUST-NOT-APPEAR"
    graph, bank, catalog = _release(
        91,
        "invalid-secret-bank-v91",
        "invalid-secret-pedagogy-v91",
    )
    payload = {
        "schema_version": 2,
        "graph": graph.model_dump(mode="json"),
        "item_bank": bank.model_dump(mode="json"),
        "pedagogy_catalog": catalog.model_dump(mode="json"),
    }
    payload["item_bank"]["items"][0]["answer"]["expected"] = {
        "secret": sentinel,
    }
    bundle = tmp_path / "invalid-nested-content.json"
    bundle.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        create_app(v2_active_release_bundle=bundle)

    rendered = "".join(
        traceback.format_exception(
            exc_info.type,
            exc_info.value,
            exc_info.tb,
        )
    )
    assert sentinel not in rendered
