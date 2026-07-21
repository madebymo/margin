"""Explicitly trusted, test-only content fixtures for session-v2 tests.

The packaged item bank is intentionally an unreleased draft. Tests that need
to exercise the runtime construct a narrow graph and mark a copied bank as
approved inside the test boundary; none of these review claims ship as
student-eligible content.
"""

from tutor.content.item_bank import load_item_bank
from tutor.schemas.assessment import (
    AssessmentHint,
    AssessmentItem,
    AssessmentProvenance,
    AssessmentSurface,
    BlankPromptSegment,
    ErrorSignature,
    ItemBankDocument,
    MathPromptSegment,
    NumericAnswerSpec,
    PromptSemanticRole,
    SymbolicAnswerSpec,
    TextPromptSegment,
)
from tutor.schemas.common import ReviewStatus, WidgetType
from tutor.schemas.kc import GraphDocument
from tutor.schemas.pedagogy import (
    Metaphor,
    Misconception,
    PedagogyPack,
    PedagogyPackCatalog,
    PedagogyPackProvenance,
)
from tutor.packs.loader import load_pedagogy_catalog
from tutor.seed.load_seed import load_graph

POWER_RULE_KC = "kc.der.power_rule"


def approved_power_rule_bank() -> ItemBankDocument:
    """Return an explicit test-only release of the packaged draft inventory."""
    payload = load_item_bank().model_dump(mode="json")
    payload["bank_version"] = "test-approved-power-v2"
    payload["released_kcs"] = [POWER_RULE_KC]
    next_order: dict[str, int] = {}
    for item in payload["items"]:
        surface = item["eligible_surfaces"][0]
        next_order[surface] = next_order.get(surface, 0) + 10
        # Keep this test release independent from draft-source recompiles. The
        # revision-lineage tests deliberately add revision 2 themselves.
        item["revision"] = 1
        item["allocation_order"] = next_order[surface]
        item["review_status"] = "human_approved"
        item["provenance"] = {
            "source": "test-only-approved-fixture",
            "author": "test fixture",
            "reviewed_by": "automated test fixture",
            "reviewed_at": "2026-01-01T00:00:00Z",
        }
    return ItemBankDocument.model_validate(payload)


def approved_power_rule_stress_bank() -> ItemBankDocument:
    """Return extra inventory for lifecycle tests that span several episodes.

    The canonical fixture above intentionally preserves the packaged 11-family
    release shape.  API reset, rollover, and quota tests need several unused
    diagnostic families after an episode has already displayed content, so
    they opt into this distinctly versioned synthetic bank instead of changing
    the fixture used by release-validation tests.
    """

    payload = approved_power_rule_bank().model_dump(mode="json")
    payload["bank_version"] = "test-approved-power-stress-v2"
    by_id = {item["item_id"]: item for item in payload["items"]}

    def add_power_item(
        template_id: str,
        *,
        item_id: str,
        family_id: str,
        allocation_order: int,
        given: str,
        expected: str,
    ) -> None:
        item = dict(by_id[template_id])
        item.update(
            {
                "item_id": item_id,
                "family_id": family_id,
                "allocation_order": allocation_order,
                "prompt": [
                    TextPromptSegment(text="Differentiate:").model_dump(mode="json"),
                    MathPromptSegment(
                        role=PromptSemanticRole.GIVEN,
                        expression=given,
                        spoken_text=f"the expression {given}",
                    ).model_dump(mode="json"),
                    BlankPromptSegment(label="answer:").model_dump(mode="json"),
                ],
                "hints": [
                    AssessmentHint(
                        text="Use the power rule on each term."
                    ).model_dump(mode="json"),
                    AssessmentHint(
                        text="Multiply by the old exponent, then subtract one."
                    ).model_dump(mode="json"),
                    AssessmentHint(
                        text=f"A correct completed form is {expected}.",
                        revealing=True,
                    ).model_dump(mode="json"),
                ],
                "answer": SymbolicAnswerSpec(
                    expected=expected,
                    variables=["x"],
                ).model_dump(mode="json"),
                "error_signatures": [],
            }
        )
        payload["items"].append(item)

    # These families exist only to exercise several reset/terminal-rollover
    # attempts. They are not part of the canonical reviewed fixture.
    add_power_item(
        "item.power.diagnostic.cube",
        item_id="item.power.diagnostic.ninth",
        family_id="family.power.diagnostic.ninth",
        allocation_order=40,
        given="x^9",
        expected="9*x^8",
    )
    add_power_item(
        "item.power.diagnostic.cube",
        item_id="item.power.diagnostic.eleventh",
        family_id="family.power.diagnostic.eleventh",
        allocation_order=50,
        given="x^11",
        expected="11*x^10",
    )
    add_power_item(
        "item.power.diagnostic.cube",
        item_id="item.power.diagnostic.thirteenth",
        family_id="family.power.diagnostic.thirteenth",
        allocation_order=60,
        given="x^13",
        expected="13*x^12",
    )
    add_power_item(
        "item.power.checkin.scaled-quintic",
        item_id="item.power.checkin.scaled-sixth",
        family_id="family.power.checkin.scaled-sixth",
        allocation_order=50,
        given="3*x^6",
        expected="18*x^5",
    )
    add_power_item(
        "item.power.capstone.polynomial-a",
        item_id="item.power.capstone.polynomial-c",
        family_id="family.power.capstone.polynomial-c",
        allocation_order=30,
        given="x^8 + 2*x^3",
        expected="8*x^7 + 6*x^2",
    )
    return ItemBankDocument.model_validate(payload)


def approved_power_rule_episode_bank() -> ItemBankDocument:
    """Return exactly enough synthetic inventory for one bounded v2 episode."""

    stress = approved_power_rule_stress_bank()
    lifecycle_only = {
        "item.power.diagnostic.ninth",
        "item.power.diagnostic.eleventh",
        "item.power.diagnostic.thirteenth",
        "item.power.capstone.polynomial-c",
    }
    return stress.model_copy(
        update={
            "bank_version": "test-approved-power-episode-v2",
            "items": [
                item
                for item in stress.items
                if item.item_id not in lifecycle_only
            ],
        }
    )


def approved_power_rule_catalog(
    *,
    graph_version: int | None = None,
    catalog_version: str = "test-approved-pedagogy-v1",
) -> PedagogyPackCatalog:
    """Return a reviewed test-only pack catalog covering the power-rule KC."""

    return PedagogyPackCatalog(
        catalog_version=catalog_version,
        graph_version=graph_version or load_item_bank().graph_version,
        published_by="test release manager",
        published_at="2026-01-01T00:00:00Z",
        packs=[
            PedagogyPack(
                kc_id=POWER_RULE_KC,
                sources=["test-only reviewed pedagogy fixture"],
                review_status=ReviewStatus.HUMAN_APPROVED,
                provenance=PedagogyPackProvenance(
                    author="test curriculum author",
                    reviewed_by="independent test reviewer",
                    reviewed_at="2026-01-01T00:00:00Z",
                ),
            )
        ],
    )


def approved_power_rule_catalog_v2(
    *, graph_version: int | None = None
) -> PedagogyPackCatalog:
    """Reviewed structured pedagogy fixture used by the v2 lesson runtime."""

    pack = PedagogyPack(
        kc_id=POWER_RULE_KC,
        misconceptions=[
            Misconception(
                id=f"m.kc.der.power_rule.test_{index}",
                description=f"Test misconception {index}.",
                error_signature=f"test signature {index}",
                remediation_hint=f"Test remediation hint {index}.",
            )
            for index in range(1, 4)
        ],
        metaphors=[
            Metaphor(
                id="met.power_rule.test_ladder",
                description="A reviewed test-only exponent ladder.",
                widget_affinity=[WidgetType.MAPPING],
            )
        ],
        error_patterns=[
            "test error pattern one",
            "test error pattern two",
            "test error pattern three",
        ],
        sources=[
            "https://example.edu/test-calculus-source-a",
            "https://example.edu/test-calculus-source-b",
        ],
        lesson_narrative=(
            TextPromptSegment(
                text="Reviewed narrative marker: connect the exponent to its multiplier."
            ),
        ),
        remediation=(
            TextPromptSegment(
                text="Reviewed remediation marker: check the multiplier, then the new exponent."
            ),
        ),
        review_status=ReviewStatus.HUMAN_APPROVED,
        provenance=PedagogyPackProvenance(
            author="test curriculum author",
            reviewed_by="independent test reviewer",
            reviewed_at="2026-01-01T00:00:00Z",
        ),
    )
    return PedagogyPackCatalog(
        schema_version=2,
        catalog_version="test-approved-pedagogy-v2",
        graph_version=graph_version or load_item_bank().graph_version,
        published_by="test release manager",
        published_at="2026-01-01T00:00:00Z",
        packs=[pack],
    )


def empty_pedagogy_catalog() -> PedagogyPackCatalog:
    """Return the exact packaged catalog used with unreleased draft banks."""

    return load_pedagogy_catalog()


def power_rule_only_graph() -> GraphDocument:
    """Return a graph whose complete hard closure is the one fixture KC."""
    graph = load_graph()
    node = next(item for item in graph.nodes if item.id == POWER_RULE_KC)
    return GraphDocument(
        graph_version=load_item_bank().graph_version,
        nodes=[node],
        edges=[],
    )


def approved_power_rule_prerequisite_release() -> tuple[
    GraphDocument,
    ItemBankDocument,
    PedagogyPackCatalog,
]:
    """Two-KC test release for bounded post-check prerequisite detours."""

    prerequisite = "kc.alg.exponent_rules"
    graph = load_graph()
    nodes = [
        node
        for node in graph.nodes
        if node.id in {prerequisite, POWER_RULE_KC}
    ]
    edge = next(
        edge
        for edge in graph.edges
        if edge.from_kc == prerequisite and edge.to_kc == POWER_RULE_KC
    )
    target_payload = approved_power_rule_stress_bank().model_dump(mode="json")
    target_payload["schema_version"] = 2
    target_payload["bank_version"] = "test-approved-power-prereq-v2"
    target_payload["released_kcs"] = [prerequisite, POWER_RULE_KC]
    narrow_graph = GraphDocument(
        graph_version=target_payload["graph_version"],
        nodes=nodes,
        edges=[edge],
    )
    for item in target_payload["items"]:
        if "checkin" in item["eligible_surfaces"]:
            item["error_signatures"] = [
                ErrorSignature(
                    expected_wrong="0",
                    implicated_prereq=prerequisite,
                ).model_dump(mode="json")
            ]

    provenance = AssessmentProvenance(
        source="test-only-prerequisite-fixture",
        author="test curriculum author",
        reviewed_by="independent test reviewer",
        reviewed_at="2026-01-01T00:00:00Z",
    )
    surface_counts = {
        AssessmentSurface.DIAGNOSTIC: 4,
        AssessmentSurface.CHECKIN: 5,
        AssessmentSurface.GUIDED_WIDGET: 1,
        AssessmentSurface.CAPSTONE: 2,
        AssessmentSurface.WORKED_EXAMPLE: 1,
    }
    next_truth = 101
    for surface, count in surface_counts.items():
        for index in range(1, count + 1):
            expected = str(next_truth)
            next_truth += 1
            if surface == AssessmentSurface.WORKED_EXAMPLE:
                prompt = [
                    TextPromptSegment(
                        text=(
                            "Worked prerequisite example: the reviewed result "
                            f"is {expected}."
                        )
                    )
                ]
            else:
                prompt = [
                    TextPromptSegment(
                        text=f"Complete prerequisite test task {surface.value} {index}."
                    ),
                    BlankPromptSegment(label="answer:"),
                ]
            target_payload["items"].append(
                AssessmentItem(
                    item_id=f"item.test.exponent.{surface.value}.{index}",
                    family_id=f"family.test.exponent.{surface.value}.{index}",
                    kc_id=prerequisite,
                    eligible_surfaces=[surface],
                    allocation_order=index * 10,
                    prompt=prompt,
                    hints=[
                        AssessmentHint(text="Use the reviewed exponent rule."),
                        AssessmentHint(text="Check the operation one step at a time."),
                        AssessmentHint(
                            text=f"A correct completed form is {expected}.",
                            revealing=True,
                        ),
                    ],
                    answer=NumericAnswerSpec(expected=expected, tolerance=0),
                    review_status=ReviewStatus.HUMAN_APPROVED,
                    provenance=provenance,
                ).model_dump(mode="json")
            )
    bank = ItemBankDocument.model_validate(target_payload)

    target_catalog = approved_power_rule_catalog()
    prerequisite_pack = PedagogyPack(
        kc_id=prerequisite,
        sources=["test-only reviewed prerequisite fixture"],
        review_status=ReviewStatus.HUMAN_APPROVED,
        provenance=PedagogyPackProvenance(
            author="test prerequisite author",
            reviewed_by="independent prerequisite reviewer",
            reviewed_at="2026-01-01T00:00:00Z",
        ),
    )
    catalog = target_catalog.model_copy(
        update={"packs": (*target_catalog.packs, prerequisite_pack)}
    )
    return narrow_graph, bank, catalog
