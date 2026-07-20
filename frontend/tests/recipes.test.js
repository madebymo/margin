import { afterEach, describe, expect, it, vi } from "vitest";

import { recipeFor, recipes } from "../src/widgets/recipes.js";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("widget recipes", () => {
  it("exposes the explicit recipe contract for exactly four widget types", () => {
    expect(Object.keys(recipes).sort()).toEqual([
      "click_region",
      "live_input",
      "mapping",
      "slider",
    ]);
    for (const recipe of Object.values(recipes)) {
      expect(Object.keys(recipe).sort()).toEqual([
        "control",
        "init",
        "normalize",
        "responseFrom",
      ]);
    }
    expect(recipeFor("unknown")).toBeNull();
  });

  it("submits only the slider value", () => {
    const state = recipes.slider.init({
      params: { min: -2, max: 2, step: 0.1 },
    });
    state.value = 1.25;

    expect(recipes.slider.responseFrom(state)).toEqual({ value: 1.25 });
  });

  it("submits only selected click-region ids in stable control order", () => {
    const state = recipes.click_region.init({
      regions: [{ id: "r1" }, { id: "r2" }],
    });
    state.selected = ["r2", "r1"];

    expect(recipes.click_region.responseFrom(state)).toEqual({
      selected: ["r1", "r2"],
    });
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

  it("submits only live-input text", () => {
    const state = recipes.live_input.init();
    state.text = "4*x^3";

    expect(recipes.live_input.responseFrom(state)).toEqual({ text: "4*x^3" });
  });
});
