"""Export JSON Schemas from the Pydantic models into schemas/json/.

These schemas are the contract shared with the frontend widget runtime and
with LLM structured-output calls in later phases.
"""

import argparse
import json
from pathlib import Path

from tutor.schemas.assessment import (
    ItemBankDocument,
    answer_spec_adapter,
    display_prompt_segment_adapter,
    prompt_segment_adapter,
)
from tutor.schemas.content_authoring import ContentReviewManifest, ItemBlueprintDocument
from tutor.schemas.kc import GraphDocument
from tutor.schemas.learner import EvidenceEvent
from tutor.schemas.lesson import MiniLessonPackage
from tutor.schemas.pedagogy import PedagogyPack, PedagogyPackCatalog
from tutor.schemas.pedagogy_authoring import (
    PedagogyReviewManifest,
    PedagogySourceDocument,
)
from tutor.schemas.probe import DiagnosticProbe
from tutor.schemas.product_quotient_authoring import ProductQuotientBlueprintDocument
from tutor.schemas.release_authoring import (
    PublishedReleaseManifest,
    ReleaseReviewManifest,
)
from tutor.schemas.widgets import widget_config_adapter

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "schemas" / "json"


def export_schemas(out_dir: Path) -> list[Path]:
    """Write one pretty-printed JSON Schema per model; return the written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    schemas = {
        "widget_config": widget_config_adapter.json_schema(),
        "mini_lesson_package": MiniLessonPackage.model_json_schema(),
        "diagnostic_probe": DiagnosticProbe.model_json_schema(),
        "pedagogy_pack": PedagogyPack.model_json_schema(),
        "pedagogy_pack_catalog": PedagogyPackCatalog.model_json_schema(),
        "graph_document": GraphDocument.model_json_schema(),
        "evidence_event": EvidenceEvent.model_json_schema(),
        "answer_spec": answer_spec_adapter.json_schema(),
        "display_prompt_segment": display_prompt_segment_adapter.json_schema(),
        "prompt_segment": prompt_segment_adapter.json_schema(),
        "item_bank_document": ItemBankDocument.model_json_schema(),
        "item_blueprint_document": ItemBlueprintDocument.model_json_schema(),
        "product_quotient_blueprint_document": (
            ProductQuotientBlueprintDocument.model_json_schema()
        ),
        "content_review_manifest": ContentReviewManifest.model_json_schema(),
        "pedagogy_source_document": PedagogySourceDocument.model_json_schema(),
        "pedagogy_review_manifest": PedagogyReviewManifest.model_json_schema(),
        "release_review_manifest": ReleaseReviewManifest.model_json_schema(),
        "published_release_manifest": PublishedReleaseManifest.model_json_schema(),
    }
    written: list[Path] = []
    for name, schema in schemas.items():
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    args = parser.parse_args(argv)
    for path in export_schemas(args.out):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
