"""Generate deterministic offline JSON/HTML packets for independent review.

The output includes expected answers and therefore must never be mounted by
the learner-facing application. It is an offline authoring artifact only.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

from tutor.content.item_bank import render_prompt, render_prompt_segments
from tutor.content.publication import prepare_release_candidate
from tutor.content.review_artifacts import (
    canonical_digest,
    canonical_json_bytes,
    compiled_family_digest,
)
from tutor.schemas.assessment import AssessmentItem, ItemBankDocument
from tutor.schemas.kc import GraphDocument
from tutor.schemas.pedagogy import PedagogyPackCatalog


class ReviewerPacketError(ValueError):
    """Review packet inputs are incomplete or internally inconsistent."""


def _parameter_shape(item: AssessmentItem) -> str:
    """Return a conservative shape for near-isomorphic family clustering."""
    payload = {
        "task_kind": item.task_kind.value,
        "surface": [surface.value for surface in item.eligible_surfaces],
        "prompt": [segment.model_dump(mode="json") for segment in item.prompt],
        "hints": [hint.model_dump(mode="json") for hint in item.hints],
        "answer_kind": item.answer.kind,
        "guided_interaction": (
            item.guided_interaction.model_dump(mode="json")
            if item.guided_interaction is not None
            else None
        ),
    }
    text = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    text = re.sub(r"(?<![A-Za-z_])[+-]?\d+(?:\.\d+)?(?:/\d+)?", "<n>", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def build_reviewer_packet(
    graph: GraphDocument,
    item_bank: ItemBankDocument,
    pedagogy_catalog: PedagogyPackCatalog,
    *,
    construct_ids: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Build exact review data from the same prompt serializer used at runtime."""
    if item_bank.graph_version != graph.graph_version:
        raise ReviewerPacketError("item bank and graph versions differ")
    if pedagogy_catalog.graph_version != graph.graph_version:
        raise ReviewerPacketError("pedagogy catalog and graph versions differ")
    construct_ids = dict(construct_ids or {})
    candidate = prepare_release_candidate(graph, item_bank, pedagogy_catalog)
    items_by_family: dict[str, list[AssessmentItem]] = defaultdict(list)
    for item in item_bank.items:
        items_by_family[item.family_id].append(item)

    family_entries: list[dict[str, object]] = []
    shapes: dict[str, list[str]] = defaultdict(list)
    for family_id in sorted(items_by_family):
        items = sorted(
            items_by_family[family_id],
            key=lambda item: (item.item_id, item.revision),
        )
        first = items[0]
        source_ids = {item.provenance.source_id for item in items}
        source_digests = {item.provenance.source_digest for item in items}
        compiler_versions = {item.provenance.compiler_version for item in items}
        if len(source_ids) != 1 or len(source_digests) != 1 or len(compiler_versions) != 1:
            raise ReviewerPacketError(
                f"family {family_id!r} has inconsistent source bindings"
            )
        item_entries = [
            {
                "item_id": item.item_id,
                "revision": item.revision,
                "prompt_text": render_prompt(item),
                "prompt_segments": [
                    segment.model_dump(mode="json") for segment in item.prompt
                ],
                "answer_spec": item.answer.model_dump(mode="json"),
                "hints": [
                    {
                        "index": index,
                        **hint.model_dump(mode="json"),
                    }
                    for index, hint in enumerate(item.hints, start=1)
                ],
                "error_signatures": [
                    signature.model_dump(mode="json")
                    for signature in item.error_signatures
                ],
                # This packet is offline and explicitly truth-bearing: reviewers
                # must see both the learner presentation and private scorer bytes.
                "guided_interaction": (
                    item.guided_interaction.model_dump(mode="json")
                    if item.guided_interaction is not None
                    else None
                ),
            }
            for item in items
        ]
        artifact_digest = compiled_family_digest(items)
        family_entries.append(
            {
                "family_id": family_id,
                "kc_id": first.kc_id,
                "surface": first.eligible_surfaces[0].value,
                "allocation_order": first.allocation_order,
                "difficulty": first.difficulty,
                "task_kind": first.task_kind.value,
                "construct_id": construct_ids.get(family_id),
                "review_status": first.review_status.value,
                "author": first.provenance.author,
                "reviewed_by": first.provenance.reviewed_by,
                "reviewed_at": (
                    first.provenance.reviewed_at.isoformat()
                    if first.provenance.reviewed_at is not None
                    else None
                ),
                "source_id": next(iter(source_ids)),
                "source_revision": first.provenance.source_revision,
                "source_digest": next(iter(source_digests)),
                "compiler_version": next(iter(compiler_versions)),
                "compiled_artifact_digest": artifact_digest,
                "items": item_entries,
            }
        )
        shapes[_parameter_shape(first)].append(family_id)

    first_paths: list[dict[str, object]] = []
    for kc_id in sorted({item.kc_id for item in item_bank.items}):
        for surface in sorted(
            {item.eligible_surfaces[0] for item in item_bank.items if item.kc_id == kc_id},
            key=lambda value: value.value,
        ):
            ordered = sorted(
                {
                    (item.allocation_order, item.family_id)
                    for item in item_bank.items
                    if item.kc_id == kc_id and item.eligible_surfaces == [surface]
                },
                key=lambda value: (
                    value[0] is None,
                    value[0] if value[0] is not None else 0,
                    value[1],
                ),
            )
            first_paths.append(
                {
                    "kc_id": kc_id,
                    "surface": surface.value,
                    "first_two_family_ids": [family_id for _order, family_id in ordered[:2]],
                    "full_allocation_order": [family_id for _order, family_id in ordered],
                }
            )

    packs = [
        {
            "kc_id": pack.kc_id,
            "version": pack.version,
            "review_status": pack.review_status.value,
            "lesson_narrative": [
                segment.model_dump(mode="json") for segment in pack.lesson_narrative
            ],
            "lesson_narrative_text": render_prompt_segments(pack.lesson_narrative),
            "remediation": [
                segment.model_dump(mode="json") for segment in pack.remediation
            ],
            "remediation_text": render_prompt_segments(pack.remediation),
            "misconceptions": [item.model_dump(mode="json") for item in pack.misconceptions],
            "metaphors": [item.model_dump(mode="json") for item in pack.metaphors],
            "error_patterns": list(pack.error_patterns),
            "citations": list(pack.sources),
            "provenance": (
                pack.provenance.model_dump(mode="json")
                if pack.provenance is not None
                else None
            ),
        }
        for pack in sorted(pedagogy_catalog.packs, key=lambda pack: pack.kc_id)
    ]
    warnings: list[str] = []
    if item_bank.schema_version < 3:
        warnings.append("legacy item bank: reviewed spoken math is not enforced")
    if pedagogy_catalog.schema_version < 2:
        warnings.append("legacy pedagogy catalog: narrative/remediation is not enforced")
    missing_constructs = sorted(
        entry["family_id"]
        for entry in family_entries
        if entry["construct_id"] is None
    )
    if missing_constructs:
        warnings.append(
            f"construct ids were not supplied for {len(missing_constructs)} families"
        )

    packet: dict[str, object] = {
        "schema_version": 1,
        "warning": (
            "OFFLINE REVIEW ARTIFACT: contains expected answers; never serve to learners."
        ),
        "graph_version": graph.graph_version,
        "graph_digest": candidate.graph_digest,
        "bank_version": item_bank.bank_version,
        "bank_digest": candidate.bank_digest,
        "catalog_version": pedagogy_catalog.catalog_version,
        "catalog_digest": candidate.catalog_digest,
        "candidate_bundle_sha256": candidate.bundle_sha256,
        "released_kcs": sorted(item_bank.released_kcs),
        "warnings": warnings,
        "first_two_paths": first_paths,
        "near_isomorphic_clusters": [
            {
                "shape_digest": canonical_digest(shape),
                "family_ids": sorted(family_ids),
            }
            for shape, family_ids in sorted(shapes.items())
            if len(family_ids) > 1
        ],
        "families": family_entries,
        "pedagogy_packs": packs,
    }
    packet["packet_digest"] = canonical_digest(packet)
    return packet


def render_reviewer_html(packet: Mapping[str, object]) -> str:
    """Render a deterministic human-readable view of a packet."""
    def escape(value: object) -> str:
        return html.escape(str(value), quote=True)

    sections = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        "<title>Adaptive Math Tutor content review</title>",
        "<style>body{font:16px/1.5 system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem}"
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #999;padding:.4rem;"
        "text-align:left;vertical-align:top}pre{white-space:pre-wrap;background:#f4f4f4;padding:1rem}"
        ".warning{border:3px solid #900;padding:1rem;color:#700}</style></head><body>",
        f'<p class="warning">{escape(packet["warning"])}</p>',
        "<h1>Content review packet</h1>",
        "<dl>",
    ]
    for field in (
        "graph_version",
        "graph_digest",
        "bank_version",
        "bank_digest",
        "catalog_version",
        "catalog_digest",
        "candidate_bundle_sha256",
        "packet_digest",
    ):
        sections.append(f"<dt>{escape(field)}</dt><dd><code>{escape(packet[field])}</code></dd>")
    sections.append("</dl>")
    warnings = packet.get("warnings", [])
    if isinstance(warnings, list) and warnings:
        sections.append("<h2>Blocking warnings</h2><ul>")
        sections.extend(f"<li>{escape(item)}</li>" for item in warnings)
        sections.append("</ul>")

    families = packet.get("families", [])
    if not isinstance(families, list):
        raise ReviewerPacketError("packet families must be a list")
    sections.append("<h2>Families</h2>")
    for family in families:
        if not isinstance(family, dict):
            raise ReviewerPacketError("packet family must be an object")
        sections.append(f"<section><h3>{escape(family['family_id'])}</h3>")
        sections.append(
            "<p>"
            f"KC: {escape(family['kc_id'])}; surface: {escape(family['surface'])}; "
            f"construct: {escape(family.get('construct_id'))}; order: "
            f"{escape(family['allocation_order'])}; compiled digest: "
            f"<code>{escape(family['compiled_artifact_digest'])}</code></p>"
        )
        items = family.get("items", [])
        if not isinstance(items, list):
            raise ReviewerPacketError("packet family items must be a list")
        for item in items:
            if not isinstance(item, dict):
                raise ReviewerPacketError("packet item must be an object")
            sections.extend(
                [
                    f"<h4>{escape(item['item_id'])}</h4>",
                    f"<h5>Exact prompt</h5><pre>{escape(item['prompt_text'])}</pre>",
                    "<h5>Structured segments</h5><pre>"
                    + escape(json.dumps(item["prompt_segments"], indent=2, sort_keys=True))
                    + "</pre>",
                    "<h5>Expected answer contract</h5><pre>"
                    + escape(json.dumps(item["answer_spec"], indent=2, sort_keys=True))
                    + "</pre>",
                    "<h5>Ordered hints</h5><pre>"
                    + escape(json.dumps(item["hints"], indent=2, sort_keys=True))
                    + "</pre>",
                ]
            )
        sections.append("</section>")

    for title, field in (
        ("First-two and allocation paths", "first_two_paths"),
        ("Near-isomorphic clusters", "near_isomorphic_clusters"),
        ("Pedagogy packs", "pedagogy_packs"),
    ):
        sections.append(f"<h2>{escape(title)}</h2><pre>")
        sections.append(escape(json.dumps(packet[field], indent=2, sort_keys=True)))
        sections.append("</pre>")
    sections.append("</body></html>\n")
    return "".join(sections)


def write_reviewer_packet(destination: Path, packet: Mapping[str, object]) -> None:
    """Atomically expose a new offline packet directory."""
    destination = Path(destination)
    if destination.exists():
        raise ReviewerPacketError("review packet destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(dir=destination.parent, prefix=f".{destination.name}.")
    )
    try:
        payloads = {
            "review-packet.json": canonical_json_bytes(packet, trailing_newline=True),
            "review-packet.html": render_reviewer_html(packet).encode("utf-8"),
        }
        for filename, payload in payloads.items():
            with (staging / filename).open("xb") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
        descriptor = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    """Build one deterministic offline content-review packet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--item-bank", type=Path, required=True)
    parser.add_argument("--pedagogy-catalog", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        graph = GraphDocument.model_validate_json(args.graph.read_text(encoding="utf-8"))
        bank = ItemBankDocument.model_validate_json(
            args.item_bank.read_text(encoding="utf-8")
        )
        catalog = PedagogyPackCatalog.model_validate_json(
            args.pedagogy_catalog.read_text(encoding="utf-8")
        )
        packet = build_reviewer_packet(graph, bank, catalog)
        write_reviewer_packet(args.out_dir, packet)
    except Exception as exc:  # noqa: BLE001 - offline CLI boundary
        print(f"review packet INVALID: {exc}", file=sys.stderr)
        return 1
    print(
        f"review packet OK: {len(packet['families'])} families, "
        f"digest={packet['packet_digest']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
