"""Pack loading: merge human-authored CSV packs with generated JSON drafts.

Precedence: human-authored template packs always win over LLM-generated
drafts for the same KC. Invalid generated files are skipped with a warning —
a bad draft can never break port construction.
"""

import logging
from pathlib import Path

from pydantic import ValidationError

import tutor.packs
from tutor.packs.import_csv import parse_pack_csv
from tutor.schemas.pedagogy import PedagogyPack

logger = logging.getLogger("tutor.packs")

PACKAGE_DIR = Path(tutor.packs.__file__).resolve().parent
TEMPLATE_CSV = PACKAGE_DIR / "template.csv"
GENERATED_DIR = PACKAGE_DIR / "generated"


def load_template_packs() -> dict[str, PedagogyPack]:
    """Load the bundled human-authored packs keyed by kc id."""
    if not TEMPLATE_CSV.is_file():
        return {}
    return {pack.kc_id: pack for pack in parse_pack_csv(TEMPLATE_CSV)}


def load_generated_packs(
    directory: Path | str | None = None,
) -> dict[str, PedagogyPack]:
    """Load LLM-generated draft packs from a directory of JSON files."""
    resolved = Path(directory) if directory is not None else GENERATED_DIR
    packs: dict[str, PedagogyPack] = {}
    if not resolved.is_dir():
        return packs
    for path in sorted(resolved.glob("*.json")):
        try:
            pack = PedagogyPack.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError as exc:
            logger.warning("skipping invalid pack file %s: %s", path.name, exc)
            continue
        packs[pack.kc_id] = pack
    return packs


def load_packs(generated_dir: Path | str | None = None) -> dict[str, PedagogyPack]:
    """All available packs: generated drafts, overridden by human template packs."""
    packs = load_generated_packs(generated_dir)
    packs.update(load_template_packs())
    return packs
