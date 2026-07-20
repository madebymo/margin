"""Canonical learner-visible text extraction for allocation and replay checks."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


def visible_fragments(value: Any) -> list[str]:
    """Flatten public structured content while retaining math line boundaries."""
    if value is None:
        return []
    if isinstance(value, BaseModel):
        return visible_fragments(value.model_dump(mode="json"))
    if isinstance(value, str):
        normalized = "\n".join(
            line
            for line in (
                " ".join(raw_line.split()) for raw_line in value.splitlines()
            )
            if line
        )
        return [normalized] if normalized else []
    if isinstance(value, bool):
        return ["true" if value else "false"]
    if isinstance(value, (int, float)):
        return [str(value)]
    if isinstance(value, dict):
        return [
            fragment
            for item in value.values()
            for fragment in visible_fragments(item)
        ]
    if isinstance(value, (list, tuple)):
        return [
            fragment
            for item in value
            for fragment in visible_fragments(item)
        ]
    if isinstance(value, set):
        return [
            fragment
            for item in sorted(value, key=str)
            for fragment in visible_fragments(item)
        ]
    return []


def extend_visible_texts(target: list[str], *values: Any) -> None:
    """Append canonical fragments once, preserving their first display order."""
    for value in values:
        for fragment in visible_fragments(value):
            if fragment not in target:
                target.append(fragment)


def canonical_visible_texts(*values: Any) -> list[str]:
    """Build the normalized, de-duplicated ledger for structured values."""
    result: list[str] = []
    extend_visible_texts(result, *values)
    return result
