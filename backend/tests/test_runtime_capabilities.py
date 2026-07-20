"""Fail-closed checks for the shared widget capability contract."""

from tutor.runtime_capabilities import (
    WIDGET_CAPABILITY_VERSION,
    effective_widget_capability_manifest,
    widget_capability_manifest,
    widget_supported,
)


def test_rich_manifest_does_not_release_live_input_without_render_semantics():
    manifest = widget_capability_manifest(rich_widgets=True)

    assert set(manifest["supported"]) == {"mapping", "slider"}
    assert "live_input" in manifest["disabled"]
    assert "render semantics" in manifest["disabled"]["live_input"]
    assert widget_supported("live_input", manifest) is False


def test_runtime_manifest_disables_live_input_from_an_older_episode_pin():
    older_pin = {
        "version": WIDGET_CAPABILITY_VERSION,
        "supported": {
            "mapping": {
                "keyboard_equivalent": True,
                "live_visual": False,
            },
            "live_input": {
                "keyboard_equivalent": True,
                "live_visual": True,
            },
        },
        "disabled": {
            "slider": "Not pinned for this episode.",
            "click_region": "Not pinned for this episode.",
        },
    }

    effective = effective_widget_capability_manifest(
        older_pin,
        widget_capability_manifest(rich_widgets=True),
    )

    assert "live_input" not in effective["supported"]
    assert "render semantics" in effective["disabled"]["live_input"]


def test_disabling_rich_widgets_also_disables_slider_but_keeps_mapping():
    manifest = widget_capability_manifest(rich_widgets=False)

    assert set(manifest["supported"]) == {"mapping"}
    assert {"slider", "live_input", "click_region"} <= set(manifest["disabled"])
