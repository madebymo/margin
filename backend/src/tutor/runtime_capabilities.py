"""One versioned widget capability contract shared through API v2."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

WIDGET_CAPABILITY_VERSION = "web-widget-capabilities-v2.2"
_KNOWN_WIDGET_TYPES = frozenset(
    {"mapping_v1", "slider_v1", "live_input", "click_region"}
)


def widget_capability_manifest(*, rich_widgets: bool = True) -> dict[str, Any]:
    """Return a fresh learner-safe manifest for generation and rendering."""
    supported = {
        "mapping_v1": {
            "keyboard_equivalent": True,
            "live_visual": False,
        },
    }
    disabled = {
        "click_region": (
            "Diagram hit targets remain disabled until true geometry and "
            "equivalent keyboard controls are released."
        ),
        "live_input": (
            "Live-input visuals remain disabled until reviewed render semantics "
            "are implemented end to end."
        ),
    }
    if rich_widgets:
        supported["slider_v1"] = {
            "keyboard_equivalent": True,
            "live_visual": False,
        }
    else:
        disabled["slider_v1"] = "Rich widget rollout is disabled."
    return {
        "version": WIDGET_CAPABILITY_VERSION,
        "supported": supported,
        "disabled": disabled,
    }


def normalize_widget_capability_manifest(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and copy a capability manifest at the control-plane boundary."""
    if candidate.get("version") != WIDGET_CAPABILITY_VERSION:
        raise ValueError("widget capability manifest version is unavailable")
    supported = candidate.get("supported")
    disabled = candidate.get("disabled")
    if not isinstance(supported, Mapping) or not isinstance(disabled, Mapping):
        raise ValueError("widget capability manifest must define supported and disabled maps")
    if not set(supported).issubset(_KNOWN_WIDGET_TYPES):
        raise ValueError("widget capability manifest names an unknown supported type")
    if not set(disabled).issubset(_KNOWN_WIDGET_TYPES):
        raise ValueError("widget capability manifest names an unknown disabled type")
    if set(supported) & set(disabled):
        raise ValueError("widget capability manifest cannot support and disable one type")

    copied_supported: dict[str, dict[str, bool]] = {}
    for widget_type, raw_capability in supported.items():
        if not isinstance(raw_capability, Mapping):
            raise ValueError("supported widget capabilities must be objects")
        keyboard_equivalent = raw_capability.get("keyboard_equivalent")
        live_visual = raw_capability.get("live_visual")
        if not isinstance(keyboard_equivalent, bool) or not isinstance(live_visual, bool):
            raise ValueError("widget capability flags must be booleans")
        if not keyboard_equivalent:
            raise ValueError("released widgets must provide a keyboard equivalent")
        copied_supported[str(widget_type)] = {
            "keyboard_equivalent": keyboard_equivalent,
            "live_visual": live_visual,
        }
    copied_disabled = {
        str(widget_type): str(reason)
        for widget_type, reason in disabled.items()
        if isinstance(reason, str) and reason.strip()
    }
    if set(copied_disabled) != set(disabled):
        raise ValueError("disabled widget capabilities require a non-empty reason")
    return {
        "version": WIDGET_CAPABILITY_VERSION,
        "supported": copied_supported,
        "disabled": copied_disabled,
    }


def effective_widget_capability_manifest(
    pinned: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Intersect episode pins with runtime switches so emergency disables win."""
    pinned_copy = normalize_widget_capability_manifest(pinned)
    runtime_copy = normalize_widget_capability_manifest(runtime)
    supported = {
        widget_type: capability
        for widget_type, capability in pinned_copy["supported"].items()
        if widget_type in runtime_copy["supported"]
    }
    disabled: dict[str, str] = {}
    for widget_type in _KNOWN_WIDGET_TYPES - set(supported):
        disabled[widget_type] = runtime_copy["disabled"].get(
            widget_type,
            pinned_copy["disabled"].get(
                widget_type, "This widget is not enabled for this episode."
            ),
        )
    return {
        "version": WIDGET_CAPABILITY_VERSION,
        "supported": supported,
        "disabled": disabled,
    }


def widget_supported(
    widget_type: str,
    manifest: Mapping[str, Any] | None = None,
) -> bool:
    """Whether the supplied learner/client contract implements this widget."""
    active = normalize_widget_capability_manifest(
        manifest or widget_capability_manifest()
    )
    return widget_type in active["supported"]
