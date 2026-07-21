"""Activity separation and reviewed-evidence trust gates."""

from datetime import datetime, timedelta, timezone
from functools import lru_cache
from uuid import uuid4

import pytest

from tutor.learner.evidence_trust import (
    EvidenceTrustPolicy,
    ReviewedEvidenceTrustRegistry,
)
from tutor.learner.service_v2 import LearnerModelServiceV2
from tutor.schemas.assessment import AssessmentItem, AssessmentSurface
from tutor.schemas.common import ResponseClass
from tutor.schemas.learner import EvidenceEvent
from tutor.seed.load_seed import load_graph

from tests.v2_helpers import (
    POWER_RULE_KC,
    approved_power_rule_bank,
    approved_power_rule_catalog,
)

KC = POWER_RULE_KC


@lru_cache(maxsize=1)
def _trust_registry() -> ReviewedEvidenceTrustRegistry:
    return ReviewedEvidenceTrustRegistry.from_release(
        load_graph(),
        approved_power_rule_bank(),
        approved_power_rule_catalog(),
    )


def _learner(
    *,
    as_of: datetime | None = None,
    trust_policy: EvidenceTrustPolicy | None = None,
) -> LearnerModelServiceV2:
    return LearnerModelServiceV2(
        load_graph(),
        as_of=as_of,
        evidence_trust_policy=trust_policy or _trust_registry(),
    )


def _item(surface: str, item_number: int) -> AssessmentItem:
    authored_surface = (
        AssessmentSurface.GUIDED_WIDGET
        if surface in {"guided_widget", "instructional_practice"}
        else AssessmentSurface.CHECKIN
        if surface == "checkin"
        else AssessmentSurface.DIAGNOSTIC
    )
    items = [
        item
        for item in approved_power_rule_bank().items
        if authored_surface in item.eligible_surfaces
    ]
    return items[item_number % len(items)]


def _event(
    learner: LearnerModelServiceV2,
    *,
    item_number: int,
    correct: bool = True,
    surface: str = "diagnostic",
    learning: bool = False,
    t: datetime | None = None,
    response_class: ResponseClass | None = None,
    pedagogy_catalog_version: str = "test-approved-pedagogy-v1",
    misconception_id: str | None = None,
    family_id: str | None = None,
) -> EvidenceEvent:
    item = _item(surface, item_number)
    catalog = approved_power_rule_catalog()
    versions = (
        {
            "graph": str(load_graph().graph_version),
            "item_bank": approved_power_rule_bank().bank_version,
            "pedagogy_catalog": pedagogy_catalog_version,
            "pedagogy_pack": str(catalog.pack_by_kc[KC].version),
        }
        if pedagogy_catalog_version != "legacy"
        else {}
    )
    return EvidenceEvent(
        event_id=uuid4(),
        learner_id=learner.learner_id,
        t=t or learner.as_of,
        item_id=(
            f"lesson-transition.{item.item_id}"
            if surface == "instructional_practice"
            else item.item_id
        ),
        family_id=family_id or item.family_id,
        kc_ids=[item.kc_id],
        correct=correct,
        response_class=response_class or (
            ResponseClass.WIDGET
            if surface == "instructional_practice"
            else ResponseClass.SYMBOLIC_ENTRY
        ),
        surface=surface,
        item_revision=item.revision,
        misconception_id=misconception_id,
        content_versions=versions,
        pedagogy_catalog_version=pedagogy_catalog_version,
        policy_version="v2",
        learner_params_version="bkt-v2",
        content_provenance=item.provenance.source,
        learning_opportunity=learning,
    )


def test_guided_widget_is_recorded_but_does_not_change_probability():
    learner = _learner()
    before = learner.routing_score(KC)
    event = _event(learner, item_number=0, surface="guided_widget")
    assert learner.evidence_trust_policy.trusts(event)
    learner.apply_event(event)

    assert len(learner.events) == 1
    assert learner.routing_score(KC) == before
    assert learner.mastery_status(KC) == "uncertain"


def test_learning_transition_only_applies_to_declared_reviewed_practice():
    observed = _learner()
    practiced = _learner()
    observed.apply_event(_event(observed, item_number=0))
    practiced.apply_event(_event(practiced, item_number=0))
    transition = _event(
        practiced,
        item_number=0,
        surface="instructional_practice",
        learning=True,
    )
    assert practiced.evidence_trust_policy.trusts(transition)
    practiced.apply_event(transition)

    assert practiced.routing_score(KC) > observed.routing_score(KC)


def test_checkin_observation_cannot_smuggle_a_learning_transition():
    plain = _learner()
    flagged = _learner()
    plain.apply_event(_event(plain, item_number=0, surface="checkin"))
    smuggled = _event(
        flagged,
        item_number=0,
        surface="checkin",
        learning=True,
    )
    assert not flagged.evidence_trust_policy.trusts(smuggled)
    flagged.apply_event(smuggled)

    assert flagged.routing_score(KC) < plain.routing_score(KC)


def test_mastery_requires_two_distinct_recent_reviewed_families():
    learner = _learner()
    learner.apply_event(_event(learner, item_number=0))
    learner.apply_event(_event(learner, item_number=0))
    assert learner.mastery_status(KC) == "uncertain"

    learner.apply_event(_event(learner, item_number=1))
    assert learner.mastery_status(KC) == "confirmed_mastered"


class _TrustEveryTestEvent:
    def trusts(self, event: EvidenceEvent) -> bool:
        del event
        return True


def test_recognition_items_cannot_satisfy_production_confirmation():
    learner = _learner(trust_policy=_TrustEveryTestEvent())
    learner.apply_event(
        _event(
            learner,
            item_number=0,
            response_class=ResponseClass.MULTIPLE_CHOICE,
        )
    )
    learner.apply_event(
        _event(
            learner,
            item_number=1,
            response_class=ResponseClass.MULTIPLE_CHOICE,
        )
    )

    assert learner.routing_score(KC) >= 0.9
    assert learner.recent_independent_counts(KC) == (0, 0)
    assert learner.mastery_status(KC) == "uncertain"


def test_old_families_do_not_satisfy_confirmation_window():
    learner = _learner()
    old = learner.as_of - timedelta(days=91)
    learner.apply_event(_event(learner, item_number=0, t=old))
    learner.apply_event(_event(learner, item_number=1, t=old))

    assert learner.recent_independent_counts(KC) == (0, 0)
    assert learner.mastery_status(KC) == "uncertain"


def test_historical_evidence_decays_between_observations_before_replay():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with_old_success = _learner(as_of=start)
    recent_only = _learner(as_of=start)
    old = start - timedelta(days=360)

    with_old_success.apply_event(
        _event(with_old_success, item_number=0, correct=True, t=old)
    )
    with_old_success.apply_event(
        _event(with_old_success, item_number=1, correct=False, t=start)
    )
    recent_only.apply_event(
        _event(recent_only, item_number=1, correct=False, t=start)
    )

    prior = _learner(as_of=start).routing_score(KC)
    assert abs(with_old_success.routing_score(KC) - prior) < abs(0.9 - prior)
    assert with_old_success.routing_score(KC) > recent_only.routing_score(KC)


def test_legacy_evidence_only_weakly_nudges_the_prior():
    learner = _learner()
    full = _learner()
    learner.apply_event(_event(learner, item_number=0, surface="legacy"))
    full.apply_event(_event(full, item_number=0))

    assert learner.routing_score(KC) < full.routing_score(KC)
    assert learner.recent_independent_counts(KC) == (0, 0)


def test_unpinned_legacy_evidence_cannot_confirm_or_add_misconception_flags():
    learner = _learner()
    learner.apply_event(
        _event(
            learner,
            item_number=0,
            pedagogy_catalog_version="legacy",
            misconception_id="m.unreviewed.claim",
        )
    )
    learner.apply_event(
        _event(
            learner,
            item_number=1,
            pedagogy_catalog_version="legacy",
        )
    )

    assert learner.recent_independent_counts(KC) == (0, 0)
    assert learner.mastery_status(KC) == "uncertain"
    assert learner.snapshot().misconception_flags == []


def test_default_learner_policy_fails_closed_for_release_shaped_events():
    learner = LearnerModelServiceV2(load_graph())
    learner.apply_event(_event(learner, item_number=0))
    learner.apply_event(_event(learner, item_number=1))

    assert learner.recent_independent_counts(KC) == (0, 0)
    assert learner.mastery_status(KC) == "uncertain"


def test_two_fabricated_nonlegacy_families_cannot_confirm_mastery():
    learner = _learner()
    learner.apply_event(
        _event(
            learner,
            item_number=0,
            family_id="fabricated.family.a",
        )
    )
    learner.apply_event(
        _event(
            learner,
            item_number=1,
            family_id="fabricated.family.b",
        )
    )

    assert learner.recent_independent_counts(KC) == (0, 0)
    assert learner.mastery_status(KC) == "uncertain"


def test_two_valid_families_under_a_fabricated_nonlegacy_catalog_cannot_confirm():
    learner = _learner()
    learner.apply_event(
        _event(
            learner,
            item_number=0,
            pedagogy_catalog_version="fabricated-pedagogy-v99",
        )
    )
    learner.apply_event(
        _event(
            learner,
            item_number=1,
            pedagogy_catalog_version="fabricated-pedagogy-v99",
        )
    )

    assert learner.recent_independent_counts(KC) == (0, 0)
    assert learner.mastery_status(KC) == "uncertain"


@pytest.mark.parametrize(
    "updates",
    [
        {"item_id": "item.fabricated"},
        {"item_revision": 99},
        {"family_id": "family.fabricated"},
        {"kc_ids": ["kc.der.chain_rule"]},
        {"surface": "capstone"},
        {"content_provenance": "fabricated-source"},
        {"pedagogy_catalog_version": "fabricated-pedagogy-v99"},
        {"misconception_id": "m.fabricated", "correct": False},
    ],
)
def test_registry_rejects_tampered_reviewed_item_identity(updates):
    learner = _learner()
    event = _event(learner, item_number=0).model_copy(update=updates)

    assert not learner.evidence_trust_policy.trusts(event)


@pytest.mark.parametrize(
    ("version_name", "value"),
    [
        ("graph", "999"),
        ("item_bank", "fabricated-bank-v99"),
        ("pedagogy_catalog", "fabricated-pedagogy-v99"),
        ("pedagogy_pack", "999"),
    ],
)
def test_registry_requires_every_content_version_pin(version_name, value):
    learner = _learner()
    event = _event(learner, item_number=0)
    versions = {**event.content_versions, version_name: value}

    assert not learner.evidence_trust_policy.trusts(
        event.model_copy(update={"content_versions": versions})
    )


def test_registry_rejects_extra_content_version_claims():
    learner = _learner()
    event = _event(learner, item_number=0)
    versions = {**event.content_versions, "ambient": "unreviewed"}

    assert not learner.evidence_trust_policy.trusts(
        event.model_copy(update={"content_versions": versions})
    )


def test_replay_uses_fixed_as_of_and_preserves_the_trust_policy():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    learner = _learner(as_of=start)
    learner.apply_event(
        _event(
            learner,
            item_number=0,
            t=start - timedelta(days=180),
        )
    )
    replayed = learner.replay()

    assert replayed.as_of == start
    assert replayed.evidence_trust_policy is learner.evidence_trust_policy
    assert replayed.routing_score(KC) == learner.routing_score(KC)
