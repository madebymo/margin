"""Canonical identities shared by review packets and release publication."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel

from tutor.schemas.assessment import AssessmentItem
from tutor.schemas.release_authoring import (
    FamilyApprovalAttestation,
    KCApprovalAttestation,
)


def _json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def canonical_json_bytes(value: Any, *, trailing_newline: bool = False) -> bytes:
    """Serialize exact deterministic JSON for hashing and immutable artifacts."""
    payload = json.dumps(
        _json_value(value),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return payload + (b"\n" if trailing_newline else b"")


def canonical_digest(value: Any) -> str:
    """Return a SHA-256 digest of canonical JSON without presentation bytes."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def compiled_family_artifact(items: Iterable[AssessmentItem]) -> dict[str, object]:
    """Return reviewable compiler output with circular approval facts removed."""
    ordered = sorted(items, key=lambda item: (item.item_id, item.revision))
    if not ordered:
        raise ValueError("a compiled family artifact requires at least one item")
    family_ids = {item.family_id for item in ordered}
    if len(family_ids) != 1:
        raise ValueError("a compiled family artifact cannot span family ids")
    artifacts: list[dict[str, object]] = []
    for item in ordered:
        artifact = item.model_dump(mode="json")
        artifact.pop("review_status", None)
        provenance = artifact.get("provenance")
        if not isinstance(provenance, dict):
            raise TypeError("compiled provenance must serialize as an object")
        provenance.pop("reviewed_by", None)
        provenance.pop("reviewed_at", None)
        artifacts.append(artifact)
    return {
        "family_id": ordered[0].family_id,
        "items": artifacts,
    }


def compiled_family_digest(items: Iterable[AssessmentItem]) -> str:
    """Bind approval to exact learner-visible content and server-side truth."""
    return canonical_digest(compiled_family_artifact(items))


def family_attestation_set_digest(
    attestations: Iterable[FamilyApprovalAttestation],
) -> str:
    """Bind a KC review to the exact approved family attestations."""
    ordered = sorted(attestations, key=lambda item: (item.family_id, item.attestation_id))
    return canonical_digest([item.model_dump(mode="json") for item in ordered])


def kc_attestation_set_digest(
    attestations: Iterable[KCApprovalAttestation],
) -> str:
    """Bind final release review to exact KC independence attestations."""
    ordered = sorted(attestations, key=lambda item: (item.kc_id, item.attestation_id))
    payloads: list[dict[str, object]] = []
    for item in ordered:
        payload = item.model_dump(mode="json")
        # Preserve the exact digest of retained schema-v1 attestations, whose
        # bytes predate mastery-claim and constructor-coverage fields. New
        # schema-v2 records always supply both and therefore bind both.
        if item.mastery_claim is None:
            payload.pop("mastery_claim", None)
        if not item.construct_ids:
            payload.pop("construct_ids", None)
        payloads.append(payload)
    return canonical_digest(payloads)
