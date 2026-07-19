"""CSV import surface for pedagogy packs.

The spreadsheet/CSV is a validated import surface only — the database remains
the authoritative store. Multi-value cells use '|' as a separator. Rows are
grouped by kc_id into one PedagogyPack per KC, imported as review_status=draft.
"""

import argparse
import csv
import sys
from pathlib import Path

from pydantic import ValidationError

from tutor.schemas.pedagogy import Metaphor, Misconception, PedagogyPack

EXPECTED_COLUMNS = [
    "kc_id",
    "misconception_id",
    "misconception_description",
    "error_signature",
    "remediation_hint",
    "metaphor_id",
    "metaphor_description",
    "widget_affinity",
    "error_patterns",
    "sources",
]


class PackImportError(ValueError):
    """Raised when the CSV is malformed; message includes the offending row."""


def _split_multi(cell: str | None) -> list[str]:
    """Split a '|'-separated multi-value cell into trimmed, non-empty parts."""
    if not cell:
        return []
    return [part.strip() for part in cell.split("|") if part.strip()]


def parse_pack_csv(path: Path | str) -> list[PedagogyPack]:
    """Parse a pack CSV into validated PedagogyPack models (review_status=draft)."""
    path = Path(path)
    misconceptions: dict[str, dict[str, Misconception]] = {}
    metaphors: dict[str, dict[str, Metaphor]] = {}
    error_patterns: dict[str, list[str]] = {}
    sources: dict[str, list[str]] = {}

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing_cols = set(EXPECTED_COLUMNS) - set(reader.fieldnames or [])
        if missing_cols:
            raise PackImportError(f"missing columns: {sorted(missing_cols)}")

        for row_num, row in enumerate(reader, start=2):
            kc_id = (row.get("kc_id") or "").strip()
            if not kc_id:
                raise PackImportError(f"row {row_num}: empty kc_id")
            misconceptions.setdefault(kc_id, {})
            metaphors.setdefault(kc_id, {})
            error_patterns.setdefault(kc_id, [])
            sources.setdefault(kc_id, [])

            try:
                if (row.get("misconception_id") or "").strip():
                    misconception = Misconception(
                        id=row["misconception_id"].strip(),
                        description=(row.get("misconception_description") or "").strip(),
                        error_signature=(row.get("error_signature") or "").strip(),
                        remediation_hint=(row.get("remediation_hint") or "").strip(),
                    )
                    misconceptions[kc_id].setdefault(misconception.id, misconception)
                if (row.get("metaphor_id") or "").strip():
                    metaphor = Metaphor(
                        id=row["metaphor_id"].strip(),
                        description=(row.get("metaphor_description") or "").strip(),
                        widget_affinity=_split_multi(row.get("widget_affinity")),
                    )
                    metaphors[kc_id].setdefault(metaphor.id, metaphor)
            except ValidationError as exc:
                raise PackImportError(f"row {row_num}: {exc}") from exc

            for pattern in _split_multi(row.get("error_patterns")):
                if pattern not in error_patterns[kc_id]:
                    error_patterns[kc_id].append(pattern)
            for source in _split_multi(row.get("sources")):
                if source not in sources[kc_id]:
                    sources[kc_id].append(source)

    packs: list[PedagogyPack] = []
    for kc_id in sorted(misconceptions):
        try:
            packs.append(
                PedagogyPack(
                    kc_id=kc_id,
                    misconceptions=list(misconceptions[kc_id].values()),
                    metaphors=list(metaphors[kc_id].values()),
                    error_patterns=error_patterns[kc_id],
                    sources=sources[kc_id],
                )
            )
        except ValidationError as exc:
            raise PackImportError(f"pack for {kc_id}: {exc}") from exc
    return packs


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: validate a CSV and optionally write one JSON per pack."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", metavar="CSV", required=True, help="pack CSV to validate")
    parser.add_argument("--out", metavar="DIR", default=None, help="write pack JSON files here")
    args = parser.parse_args(argv)

    try:
        packs = parse_pack_csv(args.validate)
    except PackImportError as exc:
        print(f"pack CSV INVALID: {exc}", file=sys.stderr)
        return 1

    print(f"pack CSV OK: {len(packs)} pack(s) for {[p.kc_id for p in packs]}")
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        for pack in packs:
            out_path = out_dir / f"{pack.kc_id.replace('.', '_')}.json"
            out_path.write_text(pack.model_dump_json(indent=2) + "\n")
            print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
