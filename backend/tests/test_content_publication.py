"""Accessible review packets and attestation-gated atomic publication."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from tutor.api.app import create_app
from tutor.api.v2 import _token_hash, install_v2_routes
from tutor.api.v2_persistence import V2PersistenceService
from tutor.api.v2_versions import V2VersionConflict
from tutor.content.item_bank import load_item_bank
from tutor.content.publication import (
    ReleasePublicationError,
    prepare_release_candidate,
    publish_release,
)
from tutor.content.review_artifacts import (
    canonical_digest,
    canonical_json_bytes,
    compiled_family_digest,
    family_attestation_set_digest,
    kc_attestation_set_digest,
)
from tutor.content.reviewer_packet import (
    build_reviewer_packet,
    render_reviewer_html,
    write_reviewer_packet,
)
from tutor.db import models as m
from tutor.db.persistence import PersistenceService
from tutor.db.session import get_engine
from tutor.schemas.assessment import (
    ItemBankDocument,
    PlotPromptSegment,
    StaticPlotPoint,
    StaticPlotSeries,
    TablePromptSegment,
)
from tutor.schemas.pedagogy import Metaphor, Misconception, PedagogyPackCatalog
from tutor.schemas.release_authoring import (
    FamilyApprovalAttestation,
    KCApprovalAttestation,
    ReleaseApprovalAttestation,
    ReleasePublicationMetadata,
    ReleaseReviewManifest,
    PublishedReleaseManifest,
)

from tests.v2_helpers import (
    approved_power_rule_bank,
    approved_power_rule_catalog,
    power_rule_only_graph,
)

_REVIEWED_AT = "2026-07-20T12:00:00Z"
_PUBLISHED_AT = "2026-07-21T12:00:00Z"


def _modern_bank() -> ItemBankDocument:
    payload = approved_power_rule_bank().model_dump(mode="json")
    packaged_item_ids = {item.item_id for item in load_item_bank().items}
    payload["items"] = [
        item for item in payload["items"] if item["item_id"] in packaged_item_ids
    ]
    payload["schema_version"] = 3
    payload["bank_version"] = "test-publish-power-v3"
    additions = []
    for source_index, surface, suffix, given, expected, order in (
        (0, "diagnostic", "ninth", "x^9", "9*x^8", 40),
        (3, "checkin", "scaled-cube", "6*x^3", "18*x^2", 50),
    ):
        item = deepcopy(payload["items"][source_index])
        item.update(
            {
                "item_id": f"item.power.{surface}.{suffix}",
                "family_id": f"family.power.{surface}.{suffix}",
                "eligible_surfaces": [surface],
                "allocation_order": order,
            }
        )
        math_segment = next(
            segment for segment in item["prompt"] if segment["kind"] == "math"
        )
        math_segment["expression"] = given
        item["answer"]["expected"] = expected
        item["hints"][2]["text"] = f"The derivative is {expected}."
        additions.append(item)
    payload["items"].extend(additions)

    for item in payload["items"]:
        for segment in item["prompt"]:
            if segment["kind"] == "math":
                segment["spoken_text"] = f"Math expression: {segment['expression']}"
        if item["eligible_surfaces"] == ["guided_widget"]:
            item["guided_interaction"] = {
                "kind": "slider_v1",
                "presentation": {
                    "prompt": "Choose the new exponent after differentiating.",
                    "label": "New exponent",
                    "help_text": "Use the arrow keys or slider control.",
                    "minimum": 0,
                    "maximum": 5,
                    "step": 1,
                    "initial_value": 0,
                    "value_label": "Selected exponent",
                    "result_template": "The new exponent is {value}.",
                },
                "scoring": {"target": 3, "tolerance": 0},
            }
        family_digest = hashlib.sha256(item["family_id"].encode("utf-8")).hexdigest()
        item["provenance"].update(
            {
                "source_id": item["family_id"].replace("family.", "source."),
                "source_revision": 1,
                "source_digest": family_digest,
                "compiler_version": "test-task-compiler-v1",
            }
        )
    return ItemBankDocument.model_validate(payload)


def _modern_catalog() -> PedagogyPackCatalog:
    payload = approved_power_rule_catalog(
        catalog_version="test-publish-pedagogy-v2"
    ).model_dump(mode="json")
    payload["schema_version"] = 2
    pack = payload["packs"][0]
    pack["lesson_narrative"] = [
        {
            "kind": "text",
            "role": "instruction",
            "text": "Use the old exponent as a multiplier, then reduce its power by one.",
        }
    ]
    pack["remediation"] = [
        {
            "kind": "text",
            "role": "worked_step",
            "text": "Separate the multiplier step from the exponent-reduction step.",
        }
    ]
    pack["misconceptions"] = [
        Misconception(
            id=f"m.power.test_{index}",
            description=f"Reviewed test misconception {index}.",
            error_signature=f"Reviewed test wrong form {index}.",
            remediation_hint=f"Reviewed test nudge {index}.",
        ).model_dump(mode="json")
        for index in range(1, 4)
    ]
    pack["metaphors"] = [
        Metaphor(
            id="met.power.test_machine",
            description="A reviewed two-step exponent machine.",
            widget_affinity=["mapping"],
        ).model_dump(mode="json")
    ]
    pack["error_patterns"] = [f"Reviewed error pattern {index}." for index in range(1, 4)]
    pack["sources"] = [
        "Test-only authoritative calculus source, section one.",
        "Test-only independent curriculum source, section two.",
    ]
    return PedagogyPackCatalog.model_validate(payload)


def _review_manifest(graph, bank, catalog) -> ReleaseReviewManifest:
    by_family = {
        family_id: [item for item in bank.items if item.family_id == family_id]
        for family_id in sorted({item.family_id for item in bank.items})
    }
    families = []
    for family_id, items in by_family.items():
        provenance = items[0].provenance
        families.append(
            FamilyApprovalAttestation(
                attestation_id=family_id.replace("family.", "attestation.family."),
                family_id=family_id,
                source_id=provenance.source_id,
                source_revision=provenance.source_revision,
                source_digest=provenance.source_digest,
                compiled_artifact_digest=compiled_family_digest(items),
                compiler_version=provenance.compiler_version,
                graph_version=graph.graph_version,
                author=provenance.author,
                reviewed_by=provenance.reviewed_by,
                reviewed_at=provenance.reviewed_at,
                mathematical_correctness=True,
                accessibility=True,
                instructional_clarity=True,
            )
        )
    kc = KCApprovalAttestation(
        attestation_id="attestation.kc.power-rule-v1",
        kc_id="kc.der.power_rule",
        family_ids=tuple(by_family),
        family_attestation_digest=family_attestation_set_digest(families),
        prepared_by="Test curriculum lead",
        reviewed_by="Independent KC reviewer",
        reviewed_at=_REVIEWED_AT,
        construct_coverage=True,
        family_independence=True,
        difficulty_progression=True,
        first_two_paths_reviewed=True,
    )
    candidate = prepare_release_candidate(graph, bank, catalog)
    release = ReleaseApprovalAttestation(
        attestation_id="attestation.release.power-rule-v1",
        release_id="release.power-rule-v1",
        graph_version=graph.graph_version,
        graph_digest=candidate.graph_digest,
        bank_version=bank.bank_version,
        bank_digest=candidate.bank_digest,
        catalog_version=catalog.catalog_version,
        catalog_digest=candidate.catalog_digest,
        released_kcs=tuple(sorted(bank.released_kcs)),
        kc_attestation_digest=kc_attestation_set_digest([kc]),
        bundle_sha256=candidate.bundle_sha256,
        prepared_by="Test release preparer",
        reviewed_by="Independent release reviewer",
        reviewed_at=_REVIEWED_AT,
        cross_component_compatibility=True,
        complete_hard_closure=True,
        exact_bytes_reviewed=True,
    )
    return ReleaseReviewManifest(
        family_attestations=families,
        kc_attestations=[kc],
        release_attestation=release,
    )


@pytest.fixture(scope="module")
def modern_release():
    graph = power_rule_only_graph()
    bank = _modern_bank()
    catalog = _modern_catalog()
    reviews = _review_manifest(graph, bank, catalog)
    publication = ReleasePublicationMetadata(
        published_by="Test release manager",
        published_at=_PUBLISHED_AT,
    )
    return graph, bank, catalog, reviews, publication


def test_accessible_table_plot_and_schema_v3_math_contracts():
    table = TablePromptSegment(
        caption="Values of the function",
        column_headers=["x", "f(x)"],
        rows=[["0", "1"], ["1", "3"]],
        spoken_text="The value rises from one to three as x moves from zero to one.",
    )
    plot = PlotPromptSegment(
        title="Two sampled points",
        x_label="x",
        y_label="f of x",
        series=[
            StaticPlotSeries(
                label="f",
                points=[StaticPlotPoint(x="0", y="1"), StaticPlotPoint(x="1", y="3")],
            )
        ],
        spoken_text="The plotted line rises through the two sampled function values.",
        equivalent_table=table,
    )

    assert plot.equivalent_table == table
    with pytest.raises(ValueError, match="column-header width"):
        TablePromptSegment(
            caption="Bad table",
            column_headers=["x", "y"],
            rows=[["0"]],
            spoken_text="One incomplete row.",
        )

    payload = approved_power_rule_bank().model_dump(mode="json")
    payload["schema_version"] = 3
    first_math = next(
        segment
        for item in payload["items"]
        for segment in item["prompt"]
        if segment["kind"] == "math"
    )
    first_math["spoken_text"] = None
    with pytest.raises(ValueError, match="math without spoken_text"):
        ItemBankDocument.model_validate(payload)


def test_reviewer_packet_is_deterministic_complete_and_offline(
    tmp_path,
    modern_release,
):
    graph, bank, catalog, _reviews, _publication = modern_release
    first = build_reviewer_packet(graph, bank, catalog)
    second = build_reviewer_packet(graph, bank, catalog)

    assert first == second
    assert first["schema_version"] == 2
    assert first["warnings"] == ["construct ids were not supplied for 13 families"]
    assert len(first["families"]) == 13
    assert first["families"][0]["items"][0]["answer_spec"]["expected"]
    assert first["families"][0]["items"][0]["rendering"]["production_text_fallback"]
    html_output = render_reviewer_html(first)
    assert "OFFLINE REVIEW ARTIFACT" in html_output
    assert "Expected answer contract" in html_output

    output = tmp_path / "review"
    write_reviewer_packet(output, first)
    assert json.loads((output / "review-packet.json").read_text()) == first
    assert (output / "review-packet.html").read_text().endswith("</html>\n")


def test_atomic_publication_emits_exact_bundle_manifest_and_sha(
    tmp_path,
    modern_release,
):
    graph, bank, catalog, reviews, publication = modern_release
    output = tmp_path / "release"

    manifest = publish_release(output, graph, bank, catalog, reviews, publication)

    bundle = (output / "bundle.json").read_bytes()
    assert hashlib.sha256(bundle).hexdigest() == manifest.bundle_sha256
    assert (output / "bundle.sha256").read_text() == (
        f"{manifest.bundle_sha256}  bundle.json\n"
    )
    reviews_bytes = (output / "release-reviews.json").read_bytes()
    assert reviews_bytes == canonical_json_bytes(reviews, trailing_newline=True)
    assert hashlib.sha256(reviews_bytes).hexdigest() == manifest.reviews_sha256
    assert set(json.loads(bundle)) == {
        "schema_version",
        "graph",
        "item_bank",
        "pedagogy_catalog",
    }
    app = FastAPI()
    install_v2_routes(
        app,
        graph,
        available_targets=("kc.der.power_rule",),
        active_release_bundle=output,
        active_release_sha256=manifest.bundle_sha256,
        resume_token_secret=b"published-release-test-secret-32-bytes",
    )
    client = TestClient(app)
    created = client.post(
        "/api/v2/sessions",
        json={
            "request_id": str(uuid4()),
            "goal_id": "goal.der.power_rule",
        },
    )
    assert created.status_code == 200
    view = created.json()
    assert view["release_id"] == manifest.release_id
    assert view["release_digest"] == manifest.bundle_sha256
    checkpoint = app.state.v2_store.get(
        view["session_id"]
    ).orchestrator.export_checkpoint()
    assert checkpoint["schema_version"] == 4
    assert checkpoint["release_id"] == manifest.release_id
    assert checkpoint["release_digest"] == manifest.bundle_sha256
    assert app.state.v2_active_release.published is True
    with pytest.raises(ReleasePublicationError, match="already exists"):
        publish_release(output, graph, bank, catalog, reviews, publication)


@pytest.mark.parametrize(
    ("artifact", "replacement", "match"),
    [
        ("bundle.sha256", b"0" * 64 + b"  bundle.json\n", "exact bundle"),
        ("release-reviews.json", b"{}\n", "invalid published"),
    ],
)
def test_runtime_rejects_changed_publication_artifacts(
    tmp_path,
    modern_release,
    artifact,
    replacement,
    match,
):
    graph, bank, catalog, reviews, publication = modern_release
    output = tmp_path / "release"
    manifest = publish_release(output, graph, bank, catalog, reviews, publication)
    (output / artifact).write_bytes(replacement)

    with pytest.raises((ValueError, V2VersionConflict), match=match):
        create_app(
            v2_active_release_bundle=output,
            v2_active_release_sha256=manifest.bundle_sha256,
        )


def test_runtime_rejects_partial_publication_directory(
    tmp_path,
    modern_release,
):
    graph, bank, catalog, reviews, publication = modern_release
    output = tmp_path / "release"
    manifest = publish_release(output, graph, bank, catalog, reviews, publication)
    (output / "release-manifest.json").unlink()

    with pytest.raises(ValueError, match="incomplete published"):
        create_app(
            v2_active_release_bundle=output,
            v2_active_release_sha256=manifest.bundle_sha256,
        )


def test_runtime_revalidates_attestations_instead_of_trusting_manifest_ids(
    tmp_path,
    modern_release,
):
    graph, bank, catalog, reviews, publication = modern_release
    output = tmp_path / "release"
    manifest = publish_release(output, graph, bank, catalog, reviews, publication)
    review_payload = reviews.model_dump(mode="json")
    review_payload["release_attestation"]["kc_attestation_digest"] = "0" * 64
    forged_reviews = ReleaseReviewManifest.model_validate(review_payload)
    forged_review_bytes = canonical_json_bytes(
        forged_reviews,
        trailing_newline=True,
    )
    manifest_payload = manifest.model_dump(mode="json")
    manifest_payload["reviews_sha256"] = hashlib.sha256(
        forged_review_bytes
    ).hexdigest()
    manifest_payload["release_attestation_digest"] = canonical_digest(
        forged_reviews.release_attestation
    )
    forged_manifest = PublishedReleaseManifest.model_validate(manifest_payload)
    (output / "release-reviews.json").write_bytes(forged_review_bytes)
    (output / "release-manifest.json").write_bytes(
        canonical_json_bytes(forged_manifest, trailing_newline=True)
    )

    with pytest.raises(V2VersionConflict, match="valid exact attestations"):
        create_app(
            v2_active_release_bundle=output,
            v2_active_release_sha256=manifest.bundle_sha256,
        )


def test_durable_schema_three_session_upgrades_on_published_release(
    tmp_path,
    modern_release,
):
    graph, bank, catalog, reviews, publication = modern_release
    output = tmp_path / "release"
    manifest = publish_release(output, graph, bank, catalog, reviews, publication)
    legacy = PersistenceService(
        engine=get_engine("sqlite+pysqlite:///:memory:")
    )
    persistence = V2PersistenceService(legacy.engine)
    secret = b"published-schema-three-resume-secret"

    def app_for_release() -> FastAPI:
        app = FastAPI()
        install_v2_routes(
            app,
            graph,
            persistence=persistence,
            available_targets=("kc.der.power_rule",),
            active_release_bundle=output,
            active_release_sha256=manifest.bundle_sha256,
            resume_token_secret=secret,
        )
        return app

    first = TestClient(app_for_release())
    created = first.post(
        "/api/v2/sessions",
        json={
            "request_id": str(uuid4()),
            "goal_id": "goal.der.power_rule",
        },
    )
    assert created.status_code == 200
    session_id = created.json()["session_id"]
    token = first.cookies.get("tutor_resume_v2")
    assert token

    with Session(legacy.engine) as database:
        checkpoint_row = database.get(m.SessionCheckpointRow, session_id)
        assert checkpoint_row is not None
        checkpoint = deepcopy(checkpoint_row.checkpoint)
        checkpoint["schema_version"] = 3
        checkpoint["orchestrator"]["schema_version"] = 3
        for key in ("release_id", "release_digest"):
            checkpoint["content_release"].pop(key)
            checkpoint["orchestrator"].pop(key)
            checkpoint["session_view"].pop(key)
        checkpoint["session_view"]["schema_version"] = 2
        checkpoint_row.checkpoint = checkpoint
        receipts = database.scalars(
            select(m.SessionMutationReceiptRow).where(
                m.SessionMutationReceiptRow.session_id == session_id
            )
        ).all()
        for receipt in receipts:
            response = deepcopy(receipt.response_payload)
            response["schema_version"] = 2
            response.pop("release_id")
            response.pop("release_digest")
            receipt.response_payload = response
        database.commit()

    resumed = TestClient(app_for_release())
    resumed.cookies.set("tutor_resume_v2", token, path="/api/v2")
    durable_bundle = persistence.resolve_resume(_token_hash(token))
    assert durable_bundle is not None
    state = durable_bundle["checkpoint"]["orchestrator"]
    release = resumed.app.state.v2_version_registry.resolve_checkpoint(state)
    policy = resumed.app.state.v2_policy_registry.resolve_checkpoint(state)
    restored_runtime = policy.restore(
        release.graph,
        state,
        release.item_bank,
        release.pedagogy_catalog,
        resumed.app.state.v2_evidence_trust_registry,
    )
    restored_runtime.bind_release_identity(
        release.release_id,
        release.release_digest,
    )
    restored = resumed.get("/api/v2/sessions/current")

    assert restored.status_code == 200, restored.text
    view = restored.json()
    assert view["schema_version"] == 3
    assert view["release_id"] == manifest.release_id
    assert view["release_digest"] == manifest.bundle_sha256


@pytest.mark.parametrize(
    "release_id",
    [
        "nonproduction.legacy-unpinned",
        "nonproduction.fixture.0123456789abcdef",
    ],
)
def test_release_attestations_reject_reserved_runtime_identities(
    modern_release,
    release_id,
):
    reviews = modern_release[3]
    payload = reviews.release_attestation.model_dump(mode="json")
    payload["release_id"] = release_id

    with pytest.raises(ValueError, match="reserved non-production"):
        ReleaseApprovalAttestation.model_validate(payload)


def test_publication_rejects_tampered_family_and_legacy_contracts(
    tmp_path,
    modern_release,
):
    graph, bank, catalog, reviews, publication = modern_release
    payload = reviews.model_dump(mode="json")
    payload["family_attestations"][0]["compiled_artifact_digest"] = "0" * 64
    tampered = ReleaseReviewManifest.model_validate(payload)

    with pytest.raises(ReleasePublicationError, match="compiled artifact digest"):
        publish_release(
            tmp_path / "tampered",
            graph,
            bank,
            catalog,
            tampered,
            publication,
        )
    assert not (tmp_path / "tampered").exists()

    legacy_payload = approved_power_rule_bank().model_dump(mode="json")
    legacy_payload["schema_version"] = 2
    legacy = ItemBankDocument.model_validate(legacy_payload)
    with pytest.raises(ReleasePublicationError, match="schema-v3"):
        publish_release(
            tmp_path / "legacy",
            graph,
            legacy,
            catalog,
            reviews,
            publication,
        )


def test_attestations_require_independent_review():
    with pytest.raises(ValueError, match="cannot approve"):
        FamilyApprovalAttestation(
            attestation_id="attestation.family.bad",
            family_id="family.bad",
            source_id="source.bad",
            source_revision=1,
            source_digest="a" * 64,
            compiled_artifact_digest="b" * 64,
            compiler_version="compiler-v1",
            graph_version=1,
            author="Same Person",
            reviewed_by=" same person ",
            reviewed_at=_REVIEWED_AT,
            mathematical_correctness=True,
            accessibility=True,
            instructional_clarity=True,
        )
