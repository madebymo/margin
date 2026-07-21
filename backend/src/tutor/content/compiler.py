"""Deterministically compile reviewed family blueprints into assessment items.

The compiler owns the mathematical construction for its named prototypes. It
does not evaluate author-supplied code or use symbolic algebra to invent the
truth. The verifier is used only as a final consistency check on the answer
contract the prototype produced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path

from tutor.content.item_bank import validate_item_bank
from tutor.schemas.assessment import (
    AssessmentHint,
    AssessmentItem,
    AssessmentProvenance,
    BlankPromptSegment,
    ItemBankDocument,
    MathPromptSegment,
    PromptSemanticRole,
    SymbolicAnswerSpec,
    TextPromptSegment,
)
from tutor.schemas.common import ReviewStatus
from tutor.schemas.content_authoring import (
    ContentReviewEntry,
    ContentReviewManifest,
    ItemBlueprintDocument,
    ItemFamilyBlueprint,
    PowerOfPowerBlueprint,
    ProductSameBaseBlueprint,
    QuotientSameBaseBlueprint,
    ReviewDecision,
)
from tutor.schemas.kc import GraphDocument
from tutor.verify.checker import VerificationStatus, verify_answer

COMPILER_VERSION = "content-compiler-v1"
SEED_DIR = Path(__file__).resolve().parents[1] / "seed"
DEFAULT_BLUEPRINT_PATH = SEED_DIR / "item_family_blueprints_v1.json"
DEFAULT_REVIEW_MANIFEST_PATH = SEED_DIR / "item_review_manifest_v1.json"
DEFAULT_GRAPH_PATH = SEED_DIR / "kc_graph_calc1.json"


class CompilationError(ValueError):
    """A blueprint set cannot be compiled into trusted deterministic output."""


def load_blueprints(path: Path | None = None) -> ItemBlueprintDocument:
    """Parse the packaged authoring source or an explicit source document."""
    source = path or DEFAULT_BLUEPRINT_PATH
    return ItemBlueprintDocument.model_validate_json(source.read_text(encoding="utf-8"))


def load_review_manifest(path: Path | None = None) -> ContentReviewManifest:
    """Parse the packaged review manifest or an explicit manifest."""
    source = path or DEFAULT_REVIEW_MANIFEST_PATH
    return ContentReviewManifest.model_validate_json(source.read_text(encoding="utf-8"))


def blueprint_digest(
    source: ItemBlueprintDocument,
    blueprint: ItemFamilyBlueprint,
) -> str:
    """Return the exact document-bound SHA-256 review identity.

    Review approval covers more than the local parameter template. It also
    binds the compiler, graph, output-bank version, and release declaration,
    so changing a promotion switch or build coordinate invalidates the prior
    review rather than silently reusing it.
    """
    canonical = json.dumps(
        {
            "blueprint": blueprint.model_dump(mode="json"),
            "blueprint_version": source.blueprint_version,
            "compiler_version": COMPILER_VERSION,
            "graph_version": source.graph_version,
            "output_bank_version": source.output_bank_version,
            "released_kcs": sorted(source.released_kcs),
            "schema_version": source.schema_version,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _item_id(blueprint: ItemFamilyBlueprint, case_id: str) -> str:
    source_slug = blueprint.blueprint_id.removeprefix("blueprint.")
    item_id = f"item.{source_slug}.{case_id}"
    if len(item_id) > 128:
        raise CompilationError(
            f"compiled item id exceeds 128 characters: {item_id!r}"
        )
    return item_id


def _review_fields(
    blueprint: ItemFamilyBlueprint,
    review: ContentReviewEntry,
) -> tuple[ReviewStatus, AssessmentProvenance]:
    if review.decision == ReviewDecision.REJECTED:
        raise CompilationError(
            f"blueprint {blueprint.blueprint_id}@{blueprint.revision} was rejected"
        )
    approved = review.decision == ReviewDecision.APPROVED
    if approved:
        if review.reviewed_by is None or review.reviewed_at is None:
            raise CompilationError("approved review entry lacks reviewer provenance")
        if review.reviewed_by.strip().casefold() == blueprint.author.strip().casefold():
            raise CompilationError(
                f"blueprint {blueprint.blueprint_id}@{blueprint.revision} "
                "cannot be approved by its author"
            )
    return (
        ReviewStatus.HUMAN_APPROVED if approved else ReviewStatus.DRAFT,
        AssessmentProvenance(
            source=blueprint.source,
            author=blueprint.author,
            reviewed_by=review.reviewed_by if approved else None,
            reviewed_at=review.reviewed_at if approved else None,
            source_id=blueprint.blueprint_id,
            source_revision=blueprint.revision,
            source_digest=review.source_digest,
            compiler_version=COMPILER_VERSION,
        ),
    )


def _build_symbolic_item(
    blueprint: ItemFamilyBlueprint,
    review: ContentReviewEntry,
    *,
    case_id: str,
    instruction: str,
    given: str,
    expected: str,
    conceptual_hint: str,
    operation_hint: str,
) -> AssessmentItem:
    review_status, provenance = _review_fields(blueprint, review)
    return AssessmentItem(
        item_id=_item_id(blueprint, case_id),
        revision=blueprint.revision,
        family_id=blueprint.family_id,
        kc_id=blueprint.kc_id,
        difficulty=blueprint.difficulty,
        task_kind=blueprint.task_kind,
        eligible_surfaces=[blueprint.surface],
        allocation_order=blueprint.allocation_order,
        prompt=[
            TextPromptSegment(
                role=PromptSemanticRole.INSTRUCTION,
                text=instruction,
            ),
            MathPromptSegment(
                role=PromptSemanticRole.GIVEN,
                expression=given,
            ),
            BlankPromptSegment(
                role=PromptSemanticRole.RESPONSE,
                label="Simplified expression:",
            ),
        ],
        hints=[
            AssessmentHint(text=conceptual_hint),
            AssessmentHint(text=operation_hint),
            AssessmentHint(
                text=f"The simplified expression is {expected}.",
                revealing=True,
            ),
        ],
        answer=SymbolicAnswerSpec(expected=expected, variables=["x"]),
        review_status=review_status,
        provenance=provenance,
    )


def _compile_product_same_base(
    blueprint: ItemFamilyBlueprint,
    review: ContentReviewEntry,
) -> list[AssessmentItem]:
    if not isinstance(blueprint, ProductSameBaseBlueprint):
        raise CompilationError("product compiler received the wrong blueprint type")
    items: list[AssessmentItem] = []
    for case in sorted(blueprint.cases, key=lambda entry: entry.case_id):
        result_exponent = case.left_exponent + case.right_exponent
        items.append(
            _build_symbolic_item(
                blueprint,
                review,
                case_id=case.case_id,
                instruction="Simplify using the product rule for equal bases.",
                given=f"x^{case.left_exponent} * x^{case.right_exponent}",
                expected=f"x^{result_exponent}",
                conceptual_hint="Keep the common base and combine its exponents.",
                operation_hint=(
                    f"Add the exponents {case.left_exponent} and "
                    f"{case.right_exponent}."
                ),
            )
        )
    return items


def _compile_quotient_same_base(
    blueprint: ItemFamilyBlueprint,
    review: ContentReviewEntry,
) -> list[AssessmentItem]:
    if not isinstance(blueprint, QuotientSameBaseBlueprint):
        raise CompilationError("quotient compiler received the wrong blueprint type")
    items: list[AssessmentItem] = []
    for case in sorted(blueprint.cases, key=lambda entry: entry.case_id):
        result_exponent = case.numerator_exponent - case.denominator_exponent
        items.append(
            _build_symbolic_item(
                blueprint,
                review,
                case_id=case.case_id,
                instruction="Simplify using the quotient rule for equal bases.",
                given=(
                    f"x^{case.numerator_exponent} / "
                    f"x^{case.denominator_exponent}"
                ),
                expected=f"x^{result_exponent}",
                conceptual_hint="Keep the common base and combine its exponents.",
                operation_hint=(
                    f"Subtract {case.denominator_exponent} from "
                    f"{case.numerator_exponent}."
                ),
            )
        )
    return items


def _compile_power_of_power(
    blueprint: ItemFamilyBlueprint,
    review: ContentReviewEntry,
) -> list[AssessmentItem]:
    if not isinstance(blueprint, PowerOfPowerBlueprint):
        raise CompilationError("power-of-power compiler received the wrong blueprint type")
    items: list[AssessmentItem] = []
    for case in sorted(blueprint.cases, key=lambda entry: entry.case_id):
        result_exponent = case.inner_exponent * case.outer_exponent
        items.append(
            _build_symbolic_item(
                blueprint,
                review,
                case_id=case.case_id,
                instruction="Simplify using the power-of-a-power rule.",
                given=f"(x^{case.inner_exponent})^{case.outer_exponent}",
                expected=f"x^{result_exponent}",
                conceptual_hint="Keep the base and combine the nested exponents.",
                operation_hint=(
                    f"Multiply the exponents {case.inner_exponent} and "
                    f"{case.outer_exponent}."
                ),
            )
        )
    return items


_PrototypeCompiler = Callable[
    [ItemFamilyBlueprint, ContentReviewEntry],
    list[AssessmentItem],
]
_PROTOTYPE_COMPILERS: dict[str, _PrototypeCompiler] = {
    "exponent.product_same_base": _compile_product_same_base,
    "exponent.quotient_same_base": _compile_quotient_same_base,
    "exponent.power_of_power": _compile_power_of_power,
}


def compile_blueprints(
    source: ItemBlueprintDocument,
    manifest: ContentReviewManifest,
    graph: GraphDocument,
) -> ItemBankDocument:
    """Compile exact, manifest-matched blueprints into one deterministic bank."""
    if source.graph_version != graph.graph_version:
        raise CompilationError(
            "blueprint/graph version mismatch: "
            f"source={source.graph_version}, graph={graph.graph_version}"
        )
    if manifest.graph_version != graph.graph_version:
        raise CompilationError(
            "review-manifest/graph version mismatch: "
            f"manifest={manifest.graph_version}, graph={graph.graph_version}"
        )
    if manifest.compiler_version != COMPILER_VERSION:
        raise CompilationError(
            "review manifest pins unsupported compiler version "
            f"{manifest.compiler_version!r}; expected {COMPILER_VERSION!r}"
        )

    graph_kcs = graph.node_ids()
    unknown_releases = set(source.released_kcs) - graph_kcs
    if unknown_releases:
        raise CompilationError(
            f"blueprint document releases unknown KCs: {sorted(unknown_releases)}"
        )

    reviews = {
        (entry.blueprint_id, entry.revision): entry
        for entry in manifest.entries
    }
    source_identities = {
        (blueprint.blueprint_id, blueprint.revision)
        for blueprint in source.family_blueprints
    }
    review_identities = set(reviews)
    missing_reviews = source_identities - review_identities
    extra_reviews = review_identities - source_identities
    if missing_reviews:
        raise CompilationError(f"missing review entries: {sorted(missing_reviews)}")
    if extra_reviews:
        raise CompilationError(f"review entries have no source blueprint: {sorted(extra_reviews)}")

    items: list[AssessmentItem] = []
    ordered_blueprints = sorted(
        source.family_blueprints,
        key=lambda blueprint: (
            blueprint.kc_id,
            blueprint.surface.value,
            blueprint.allocation_order,
            blueprint.blueprint_id,
            blueprint.revision,
        ),
    )
    for blueprint in ordered_blueprints:
        identity = (blueprint.blueprint_id, blueprint.revision)
        review = reviews[identity]
        if blueprint.kc_id not in graph_kcs:
            raise CompilationError(
                f"blueprint {blueprint.blueprint_id} names unknown KC {blueprint.kc_id}"
            )
        digest = blueprint_digest(source, blueprint)
        if review.source_digest != digest:
            raise CompilationError(
                f"review digest mismatch for {blueprint.blueprint_id}@{blueprint.revision}"
            )
        try:
            compiler = _PROTOTYPE_COMPILERS[blueprint.prototype_id]
        except KeyError as exc:  # pragma: no cover - discriminated schema prevents this
            raise CompilationError(
                f"unsupported prototype {blueprint.prototype_id!r}"
            ) from exc
        compiled = compiler(blueprint, review)
        for item in compiled:
            answer = item.answer
            if not isinstance(answer, SymbolicAnswerSpec):
                raise CompilationError(
                    f"prototype {blueprint.prototype_id} produced a non-symbolic answer"
                )
            verdict = verify_answer(answer, answer.expected, supervised=True)
            if verdict.status != VerificationStatus.CORRECT:
                raise CompilationError(
                    f"compiled answer for {item.item_id} failed verification ({verdict.code})"
                )
        items.extend(compiled)

    return ItemBankDocument(
        bank_version=source.output_bank_version,
        graph_version=source.graph_version,
        released_kcs=source.released_kcs,
        items=items,
    )


def compile_default_blueprints(graph: GraphDocument) -> ItemBankDocument:
    """Compile the packaged authoring source against its packaged manifest."""
    return compile_blueprints(load_blueprints(), load_review_manifest(), graph)


def main(argv: list[str] | None = None) -> int:
    """Compile or check an authoring document from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_BLUEPRINT_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_REVIEW_MANIFEST_PATH)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH_PATH)
    parser.add_argument(
        "--pedagogy-catalog",
        type=Path,
        default=None,
        help="exact reviewed pedagogy catalog (defaults to the packaged release)",
    )
    parser.add_argument("--check", action="store_true", help="validate compiled output")
    parser.add_argument("--out", type=Path, default=None, help="write compiled bank JSON")
    args = parser.parse_args(argv)
    if not args.check and args.out is None:
        parser.error("nothing to do: pass --check and/or --out PATH")

    try:
        source = load_blueprints(args.source)
        manifest = load_review_manifest(args.manifest)
        graph = GraphDocument.model_validate_json(args.graph.read_text(encoding="utf-8"))
        bank = compile_blueprints(source, manifest, graph)
        if args.check:
            from tutor.packs.loader import load_pedagogy_catalog

            pedagogy_catalog = load_pedagogy_catalog(args.pedagogy_catalog)
            errors = validate_item_bank(bank, graph, pedagogy_catalog)
        else:
            errors = []
    except Exception as exc:  # noqa: BLE001 - CLI boundary reports a safe failure
        print(f"content compilation INVALID: {exc}", file=sys.stderr)
        return 1
    if errors:
        for error in errors:
            print(f"compiled item bank INVALID: {error}", file=sys.stderr)
        return 1

    if args.out is not None:
        args.out.write_text(bank.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        f"content compilation OK: {source.blueprint_version}, "
        f"{len(source.family_blueprints)} families, {len(bank.items)} items, "
        f"released KCs={len(bank.released_kcs)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
