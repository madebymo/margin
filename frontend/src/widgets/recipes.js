import ClickRegionControl from "./controls/ClickRegionControl.svelte";
import LiveInputControl from "./controls/LiveInputControl.svelte";
import MappingControl from "./controls/MappingControl.svelte";
import SliderControl from "./controls/SliderControl.svelte";
import { normalizeSlider } from "../scene/normalize.js";

function shuffled(values) {
  const result = [...values];
  for (let index = result.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(Math.random() * (index + 1));
    [result[index], result[swapIndex]] = [result[swapIndex], result[index]];
  }
  return result;
}

export const recipes = Object.freeze({
  slider: Object.freeze({
    init(config) {
      return { value: Number(config.params.min) };
    },
    normalize(config, state) {
      if (!config.params.plot) {
        return null;
      }
      return normalizeSlider(config, state);
    },
    responseFrom(state) {
      return { value: Number(state.value) };
    },
    control: SliderControl,
  }),
  click_region: Object.freeze({
    init(config) {
      return {
        selected: [],
        regionOrder: config.regions.map((region) => region.id),
      };
    },
    normalize() {
      return null;
    },
    responseFrom(state) {
      return {
        selected: state.regionOrder.filter((id) => state.selected.includes(id)),
      };
    },
    control: ClickRegionControl,
  }),
  mapping: Object.freeze({
    init(config) {
      return {
        rows: config.left.map((left) => ({ left, value: "" })),
        rightOptions: shuffled(config.right),
      };
    },
    normalize() {
      return null;
    },
    responseFrom(state) {
      return {
        pairs: state.rows
          .filter((row) => row.value)
          .map((row) => [row.left, row.value]),
      };
    },
    control: MappingControl,
  }),
  live_input: Object.freeze({
    init() {
      return { text: "" };
    },
    normalize() {
      return null;
    },
    responseFrom(state) {
      return { text: state.text };
    },
    control: LiveInputControl,
  }),
});

export function recipeFor(widgetType) {
  if (!Object.prototype.hasOwnProperty.call(recipes, widgetType)) {
    return null;
  }
  return recipes[widgetType];
}
