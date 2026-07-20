import { afterEach, describe, expect, it, vi } from "vitest";

import { recipeFor, recipes } from "../src/widgets/recipes.js";
import {
  installMinimalWidgetCapabilities,
  installWidgetCapabilities,
  minimalWidgetCapabilities,
  widgetCapabilities,
  widgetCapability,
} from "../src/widgets/capabilities.js";

afterEach(() => {
  installMinimalWidgetCapabilities();
  vi.restoreAllMocks();
});

describe("widget recipes", () => {
  it("advertises only widget semantics the frontend fully implements", () => {
    installWidgetCapabilities(widgetCapabilities);
    expect(Object.keys(recipes).sort()).toEqual(["mapping", "slider"]);
    for (const recipe of Object.values(recipes)) {
      expect(Object.keys(recipe).sort()).toEqual([
        "control",
        "init",
        "normalize",
        "responseFrom",
      ]);
    }
    expect(Object.keys(widgetCapabilities.supported).sort()).toEqual([
      "mapping",
      "slider",
    ]);
    expect(recipeFor("click_region")).toBeNull();
    expect(recipeFor("live_input")).toBeNull();
    expect(recipeFor("unknown")).toBeNull();
    expect(widgetCapability("live_input")).toMatchObject({
      supported: false,
      reason: expect.stringMatching(/render semantics/i),
    });
    expect(widgetCapability("click_region")).toMatchObject({
      supported: false,
    });
  });

  it("fails closed to mapping-only capabilities until the server confirms rich UI", () => {
    installWidgetCapabilities(widgetCapabilities);
    expect(recipeFor("live_input")).toBeNull();
    expect(recipeFor("slider")).toBe(recipes.slider);

    // This is the bootstrap error path after a prior successful manifest.
    installMinimalWidgetCapabilities();

    expect(Object.keys(minimalWidgetCapabilities.supported)).toEqual(["mapping"]);
    expect(recipeFor("mapping")).toBe(recipes.mapping);
    expect(recipeFor("live_input")).toBeNull();
    expect(recipeFor("slider")).toBeNull();
    expect(widgetCapability("live_input")).toMatchObject({
      supported: false,
    });
  });

  it("rejects an incompatible or inaccessible rich-capability contract", () => {
    expect(() =>
      installWidgetCapabilities({
        ...widgetCapabilities,
        version: "future-widget-contract",
      }),
    ).toThrow(/invalid widget capability manifest/i);
    expect(() =>
      installWidgetCapabilities({
        ...widgetCapabilities,
        supported: {
          live_input: { keyboard_equivalent: true, live_visual: true },
        },
        disabled: {
          click_region: widgetCapabilities.disabled.click_region,
        },
      }),
    ).toThrow(/invalid widget capability manifest/i);
  });

  it("submits only the slider value", () => {
    const state = recipes.slider.init({
      params: { min: -2, max: 2, step: 0.1 },
    });
    state.value = 1.25;

    expect(recipes.slider.responseFrom(state)).toEqual({ value: 1.25 });
  });

  it("shuffles mapping options once during initialization and preserves left order", () => {
    const random = vi.spyOn(Math, "random").mockReturnValue(0);
    const config = {
      left: ["first", "second", "third"],
      right: ["one", "two", "three"],
    };
    const state = recipes.mapping.init(config);

    expect(random).toHaveBeenCalledTimes(2);
    expect(state.rightOptions).not.toEqual(config.right);
    state.rows[0].value = "three";
    state.rows[2].value = "one";
    expect(recipes.mapping.responseFrom(state)).toEqual({
      pairs: [
        ["first", "three"],
        ["third", "one"],
      ],
    });
    expect(random).toHaveBeenCalledTimes(2);
  });
});
