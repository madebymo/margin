"""Build the private reviewer packet for the pending Solve Quadratics wave.

The command compiles the exact 52-family assessment source and four pedagogy
sources, then writes truth-bearing JSON/HTML for independent human review. It
does not alter review decisions, approve KCs, or produce a release bundle.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tutor.content.pending_reviewer_packet import build_pending_reviewer_packet
from tutor.content.reviewer_packet import write_reviewer_packet
from tutor.content.solve_quadratics_release import (
    DEFAULT_GRAPH_PATH,
    DEFAULT_MANIFEST_PATH as DEFAULT_ASSESSMENT_REVIEW_PATH,
    DEFAULT_PEDAGOGY_MANIFEST_PATH as DEFAULT_PEDAGOGY_REVIEW_PATH,
    DEFAULT_PEDAGOGY_SOURCE_PATH,
    DEFAULT_SOURCE_PATH as DEFAULT_ASSESSMENT_SOURCE_PATH,
    compile_release_inventory,
    family_digest,
    load_manifest,
    load_source,
)
from tutor.packs.review_compiler import (
    load_review_manifest as load_pedagogy_reviews,
    load_source_document as load_pedagogy_source,
)
from tutor.schemas.content_authoring import ContentReviewManifest
from tutor.schemas.kc import GraphDocument
from tutor.schemas.pedagogy_authoring import (
    PedagogyReviewManifest,
    PedagogySourceDocument,
)
from tutor.schemas.solve_quadratics_authoring import (
    SolveQuadraticsBlueprintDocument,
)


def build_pending_solve_quadratics_packet(
    graph: GraphDocument,
    assessment_source: SolveQuadraticsBlueprintDocument,
    assessment_reviews: ContentReviewManifest,
    pedagogy_source: PedagogySourceDocument,
    pedagogy_reviews: PedagogyReviewManifest,
) -> dict[str, object]:
    """Compile exact pending Solve Quadratics inputs for human review."""

    return build_pending_reviewer_packet(
        graph,
        assessment_source,
        assessment_reviews,
        pedagogy_source,
        pedagogy_reviews,
        artifact_kind="pending_solve_quadratics_review",
        compile_inventory=compile_release_inventory,
        family_digest=family_digest,
    )


def main(argv: list[str] | None = None) -> int:
    """Validate or write the exact pending Solve Quadratics review packet."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assessment-source",
        type=Path,
        default=DEFAULT_ASSESSMENT_SOURCE_PATH,
    )
    parser.add_argument(
        "--assessment-reviews",
        type=Path,
        default=DEFAULT_ASSESSMENT_REVIEW_PATH,
    )
    parser.add_argument(
        "--pedagogy-source",
        type=Path,
        default=DEFAULT_PEDAGOGY_SOURCE_PATH,
    )
    parser.add_argument(
        "--pedagogy-reviews",
        type=Path,
        default=DEFAULT_PEDAGOGY_REVIEW_PATH,
    )
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH_PATH)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    if not args.check and args.out_dir is None:
        parser.error("nothing to do: pass --check and/or --out-dir PATH")

    try:
        graph = GraphDocument.model_validate_json(
            args.graph.read_text(encoding="utf-8")
        )
        packet = build_pending_solve_quadratics_packet(
            graph,
            load_source(args.assessment_source),
            load_manifest(args.assessment_reviews),
            load_pedagogy_source(args.pedagogy_source),
            load_pedagogy_reviews(args.pedagogy_reviews),
        )
        if args.out_dir is not None:
            write_reviewer_packet(args.out_dir, packet)
    except Exception as exc:  # noqa: BLE001 - offline CLI fail-closed boundary
        print(f"Solve Quadratics review packet INVALID: {exc}", file=sys.stderr)
        return 1

    if args.check:
        print(
            "Solve Quadratics review packet OK: "
            f"{len(packet['families'])} draft families, "
            f"{len(packet['pedagogy_packs'])} draft packs, "
            f"digest={packet['packet_digest']}"
        )
    if args.out_dir is not None:
        print(f"wrote private review packet to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
