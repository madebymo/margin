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
      "mapping_v1",
      "slider_v1",
    ]);
    expect(widgetCapabilities.supported.slider_v1.live_visual).toBe(false);
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
    expect(recipeFor("slider_v1")).toBe(recipes.slider);
    expect(recipeFor("slider")).toBe(recipes.slider);

    // This is the bootstrap error path after a prior successful manifest.
    installMinimalWidgetCapabilities();

    expect(Object.keys(minimalWidgetCapabilities.supported)).toEqual([
      "mapping_v1",
    ]);
    expect(recipeFor("mapping_v1")).toBe(recipes.mapping);
    expect(recipeFor("mapping")).toBe(recipes.mapping);
    expect(recipeFor("live_input")).toBeNull();
    expect(recipeFor("slider_v1")).toBeNull();
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

  it("restores slider-v1 state without requiring a legacy plot recipe", () => {
    const config = {
      presentation: {
        prompt: "Choose the new exponent.",
        label: "Exponent",
        help_text: "Use arrow keys or the slider.",
        minimum: -2,
        maximum: 7,
        step: 1,
        initial_value: 0,
        value_label: "Selected exponent",
        result_template: "The exponent becomes {value}.",
      },
    };

    expect(recipes.slider.init(config, { value: 3 })).toEqual({ value: 3 });
    expect(recipes.slider.normalize(config, { value: 3 })).toBeNull();
    expect(recipes.slider.responseFrom({ value: 3 })).toEqual({ value: 3 });
  });

  it.each([
    null,
    { minimum: 2, maximum: 2, step: 1, initial_value: 2 },
    { minimum: 0, maximum: 2, step: 0, initial_value: 1 },
    { minimum: 0, maximum: 2, step: 1, initial_value: 3 },
  ])("rejects a malformed slider presentation instead of emitting NaN", (presentation) => {
    expect(() => recipes.slider.init({ presentation })).toThrow(
      /invalid slider presentation/i,
    );
  });

  it("preserves reviewed mapping order and restores a valid draft", () => {
    const config = {
      left: ["first", "second", "third"],
      right: ["one", "two", "three"],
    };
    const state = recipes.mapping.init(config, {
      rows: [
        { left: "first", value: "three" },
        { left: "second", value: "not-an-option" },
      ],
    });

    expect(state.rightOptions.map((option) => option.label)).toEqual(config.right);
    expect(state.rows[0].value).toBe("three");
    expect(state.rows[1].value).toBe("");
    state.rows[2].value = "one";
    expect(recipes.mapping.responseFrom(state)).toEqual({
      pairs: [
        ["first", "three"],
        ["third", "one"],
      ],
    });
  });

  it("uses stable ids for a mapping-v1 presentation", () => {
    const state = recipes.mapping.init({
      presentation: {
        rows: [
          { entry_id: "row.square", label: "x^2", spoken_text: "x squared" },
          { entry_id: "row.cube", label: "x^3", spoken_text: "x cubed" },
        ],
        options: [
          { entry_id: "option.2x", label: "2*x", spoken_text: "two x" },
          { entry_id: "option.3x2", label: "3*x^2", spoken_text: "three x squared" },
        ],
      },
    });
    state.rows[0].value = "option.2x";

    expect(recipes.mapping.responseFrom(state)).toEqual({
      pairs: [["row.square", "option.2x"]],
    });
  });
});
