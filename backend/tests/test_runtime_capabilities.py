"""Fail-closed checks for the shared widget capability contract."""

import pytest

from tutor.runtime_capabilities import (
    WIDGET_CAPABILITY_VERSION,
    effective_widget_capability_manifest,
    normalize_widget_capability_manifest,
    widget_capability_manifest,
    widget_supported,
)


def test_rich_manifest_does_not_release_live_input_without_render_semantics():
    manifest = widget_capability_manifest(rich_widgets=True)

    assert set(manifest["supported"]) == {"mapping_v1", "slider_v1"}
    assert manifest["supported"]["slider_v1"]["live_visual"] is False
    assert "live_input" in manifest["disabled"]
    assert "render semantics" in manifest["disabled"]["live_input"]
    assert widget_supported("live_input", manifest) is False


def test_unversioned_widget_alias_cannot_enter_a_v2_episode_pin():
    unversioned = {
        "version": WIDGET_CAPABILITY_VERSION,
        "supported": {
            "mapping": {
                "keyboard_equivalent": True,
                "live_visual": False,
            },
        },
        "disabled": {
            "slider_v1": "Not pinned for this episode.",
            "live_input": "Not pinned for this episode.",
            "click_region": "Not pinned for this episode.",
        },
    }

    with pytest.raises(ValueError, match="unknown supported type"):
        normalize_widget_capability_manifest(unversioned)


def test_runtime_manifest_can_disable_a_pinned_reviewed_slider():
    effective = effective_widget_capability_manifest(
        widget_capability_manifest(rich_widgets=True),
        widget_capability_manifest(rich_widgets=False),
    )

    assert set(effective["supported"]) == {"mapping_v1"}
    assert "Rich widget rollout" in effective["disabled"]["slider_v1"]


def test_disabling_rich_widgets_also_disables_slider_but_keeps_mapping():
    manifest = widget_capability_manifest(rich_widgets=False)

    assert set(manifest["supported"]) == {"mapping_v1"}
    assert {"slider_v1", "live_input", "click_region"} <= set(
        manifest["disabled"]
    )
