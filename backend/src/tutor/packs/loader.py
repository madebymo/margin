"""Pedagogy authoring inputs and immutable catalog release loading.

``load_packs`` is the legacy-v1/offline authoring merge: human-authored CSV
drafts win over LLM-generated drafts. It must not be used as a v2 release
gate. ``load_pedagogy_catalog`` instead parses one exact, versioned document;
it never merges or promotes ambient draft content.
"""

import logging
from pathlib import Path

from pydantic import ValidationError

import tutor.packs
from tutor.packs.import_csv import parse_pack_csv
from tutor.schemas.pedagogy import PedagogyPack, PedagogyPackCatalog

logger = logging.getLogger("tutor.packs")

PACKAGE_DIR = Path(tutor.packs.__file__).resolve().parent
TEMPLATE_CSV = PACKAGE_DIR / "template.csv"
GENERATED_DIR = PACKAGE_DIR / "generated"
DEFAULT_PEDAGOGY_CATALOG_PATH = (
    PACKAGE_DIR.parent / "seed" / "pedagogy_catalog_v2.json"
)


def load_pedagogy_catalog(
    path: Path | str | None = None,
) -> PedagogyPackCatalog:
    """Parse one exact reviewed catalog document or raise on any defect."""

    source = Path(path) if path is not None else DEFAULT_PEDAGOGY_CATALOG_PATH
    return PedagogyPackCatalog.model_validate_json(
        source.read_text(encoding="utf-8")
    )


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
    """Legacy/offline drafts, with generated content overridden by the CSV.

    This merge is intentionally not a reviewed catalog and must not be used to
    admit v2 content or restore a catalog-pinned session.
    """
    packs = load_generated_packs(generated_dir)
    packs.update(load_template_packs())
    return packs
