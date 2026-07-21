"""Canonical identities for exact content-and-policy releases.

Published bundles carry a human-approved release id and an SHA-256 over their
exact bytes. Directly constructed releases are useful in tests and local
fixtures only; they receive an explicit non-production id and a digest over
the same canonical bundle representation used by publication.
"""

from __future__ import annotations

import hashlib

from tutor.content.review_artifacts import canonical_json_bytes
from tutor.schemas.assessment import ItemBankDocument
from tutor.schemas.kc import GraphDocument
from tutor.schemas.pedagogy import PedagogyPackCatalog

NONPRODUCTION_RELEASE_PREFIX = "nonproduction.fixture."
LEGACY_RELEASE_ID = "nonproduction.legacy-unpinned"
LEGACY_RELEASE_DIGEST = hashlib.sha256(LEGACY_RELEASE_ID.encode("ascii")).hexdigest()


def canonical_bundle_sha256(
    graph: GraphDocument,
    item_bank: ItemBankDocument,
    pedagogy_catalog: PedagogyPackCatalog,
) -> str:
    """Hash deterministic schema-v2 bundle bytes for local/test releases."""

    payload = {
        "schema_version": 2,
        "graph": graph.model_dump(mode="json"),
        "item_bank": item_bank.model_dump(mode="json"),
        "pedagogy_catalog": pedagogy_catalog.model_dump(mode="json"),
    }
    return hashlib.sha256(
        canonical_json_bytes(payload, trailing_newline=True)
    ).hexdigest()


def fixture_release_id(bundle_sha256: str) -> str:
    """Return a conspicuous, deterministic identity for an unpublished fixture."""

    _validate_sha256("bundle_sha256", bundle_sha256)
    return f"{NONPRODUCTION_RELEASE_PREFIX}{bundle_sha256[:24]}"


def _validate_sha256(name: str, value: object) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
