"""Validate (and optionally publish) the Calc-1 seed graph and coverage matrix.

Usage:
    python -m tutor.seed.load_seed --validate
    python -m tutor.seed.load_seed --validate --db postgresql+psycopg://tutor:tutor@localhost/tutor
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from tutor.graph import service
from tutor.schemas.common import WidgetType
from tutor.schemas.kc import GraphDocument

SEED_DIR = Path(__file__).resolve().parent
GRAPH_PATH = SEED_DIR / "kc_graph_calc1.json"
COVERAGE_PATH = SEED_DIR / "coverage_matrix.json"

VALID_MEASURES = {"production", "recognition", "reasoning"}


def load_graph() -> GraphDocument:
    """Parse and fully validate the seed graph (runs all GraphDocument validators)."""
    return GraphDocument.model_validate(json.loads(GRAPH_PATH.read_text()))


def load_coverage() -> dict[str, Any]:
    """Load the raw coverage matrix JSON."""
    return json.loads(COVERAGE_PATH.read_text())


def validate_coverage(doc: GraphDocument, coverage: dict[str, Any]) -> list[str]:
    """Return a list of coverage-matrix problems (empty when valid).

    The matrix must cover exactly the node set, use valid widget types, mandate
    a text fallback everywhere, and declare what each KC's interaction measures.
    """
    errors: list[str] = []
    node_ids = doc.node_ids()
    matrix_ids = set(coverage)
    for missing in sorted(node_ids - matrix_ids):
        errors.append(f"missing coverage entry: {missing}")
    for extra in sorted(matrix_ids - node_ids):
        errors.append(f"coverage entry for unknown kc: {extra}")

    valid_types = {w.value for w in WidgetType}
    for kc_id, entry in sorted(coverage.items()):
        widget_types = entry.get("widget_types", [])
        if not widget_types:
            errors.append(f"{kc_id}: no widget types")
        for widget_type in widget_types:
            if widget_type not in valid_types:
                errors.append(f"{kc_id}: invalid widget type {widget_type!r}")
        if entry.get("text_fallback") is not True:
            errors.append(f"{kc_id}: text_fallback must be true")
        if entry.get("measures") not in VALID_MEASURES:
            errors.append(f"{kc_id}: invalid measures {entry.get('measures')!r}")
    return errors


def topo_depth(doc: GraphDocument) -> int:
    """Length of the longest prerequisite chain in the DAG."""
    order = service.topological_order(doc)
    predecessors: dict[str, list[str]] = defaultdict(list)
    for edge in doc.edges:
        predecessors[edge.to_kc].append(edge.from_kc)
    depth: dict[str, int] = {}
    for kc in order:
        depth[kc] = max((depth[p] + 1 for p in predecessors[kc]), default=0)
    return max(depth.values())


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate", action="store_true", help="validate the seed graph and coverage matrix"
    )
    parser.add_argument(
        "--db", metavar="URL", default=None, help="publish the graph to this database URL"
    )
    args = parser.parse_args(argv)
    if not args.validate and not args.db:
        parser.error("nothing to do: pass --validate and/or --db URL")

    try:
        doc = load_graph()
    except Exception as exc:  # noqa: BLE001 — CLI boundary, report and exit
        print(f"seed graph INVALID: {exc}", file=sys.stderr)
        return 1

    coverage = load_coverage()
    errors = validate_coverage(doc, coverage)
    if errors:
        for error in errors:
            print(f"coverage INVALID: {error}", file=sys.stderr)
        return 1

    print(
        f"seed OK: {len(doc.nodes)} nodes, {len(doc.edges)} edges, "
        f"{len(service.roots(doc))} roots, max prerequisite depth {topo_depth(doc)}"
    )

    if args.db:
        from sqlalchemy.orm import Session

        from tutor.db.session import create_all, get_engine

        engine = get_engine(args.db)
        create_all(engine)
        with Session(engine) as session:
            row_id = service.publish_graph(session, doc)
            session.commit()
        print(f"published graph version {doc.graph_version} (graph_versions.id={row_id})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
