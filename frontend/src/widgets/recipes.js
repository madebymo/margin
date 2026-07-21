import MappingControl from "./controls/MappingControl.svelte";
import SliderControl from "./controls/SliderControl.svelte";
import { normalizeSlider } from "../scene/normalize.js";
import { widgetCapability } from "./capabilities.js";

function sliderParams(config) {
  const value = config.params ?? config.presentation ?? config;
  return {
    min: Number(value.min ?? value.minimum),
    max: Number(value.max ?? value.maximum),
    step: Number(value.step),
    initial: Number(value.initial_value ?? value.min ?? value.minimum),
  };
}

function mappingParts(config) {
  const presentation = config.presentation ?? config;
  if (Array.isArray(presentation.rows) && Array.isArray(presentation.options)) {
    return {
      left: presentation.rows.map((row) => ({
        id: String(row.entry_id),
        label: String(row.label),
        spokenText: String(row.spoken_text ?? row.label),
      })),
      right: presentation.options.map((option) => ({
        id: String(option.entry_id),
        label: String(option.label),
        spokenText: String(option.spoken_text ?? option.label),
      })),
    };
  }
  return {
    left: (config.left ?? []).map((label) => ({
      id: String(label),
      label: String(label),
      spokenText: String(label),
    })),
    right: (config.right ?? []).map((label) => ({
      id: String(label),
      label: String(label),
      spokenText: String(label),
    })),
  };
}

export const recipes = Object.freeze({
  slider: Object.freeze({
    init(config, restored) {
      const params = sliderParams(config);
      const restoredValue = Number(restored?.value);
      const value = Number.isFinite(restoredValue)
        && restoredValue >= params.min
        && restoredValue <= params.max
        ? restoredValue
        : params.initial;
      return { value };
    },
    normalize(config, state) {
      // Reviewed slider-v1 interactions deliberately expose bounded state and
      // an optional static summary, not the legacy live-input plot recipe.
      // The native range control and live text remain the complete interaction.
      if (!config.params?.plot) {
        return null;
      }
      return normalizeSlider(config, state);
    },
    responseFrom(state) {
      return { value: Number(state.value) };
    },
    control: SliderControl,
  }),
  mapping: Object.freeze({
    init(config, restored) {
      const { left, right } = mappingParts(config);
      const restoredById = new Map(
        (restored?.rows ?? []).map((row) => [
          String(row.id ?? row.left),
          String(row.value ?? ""),
        ]),
      );
      const validOptions = new Set(right.map((option) => option.id));
      return {
        rows: left.map((row) => {
          const restoredValue = restoredById.get(row.id) ?? "";
          return {
            ...row,
            value: validOptions.has(restoredValue) ? restoredValue : "",
          };
        }),
        // Review and replay bind this exact order. Never randomize it in-browser.
        rightOptions: right,
      };
    },
    normalize() {
      return null;
    },
    responseFrom(state) {
      return {
        pairs: state.rows
          .filter((row) => row.value)
          .map((row) => [row.id, row.value]),
      };
    },
    control: MappingControl,
  }),
});

const recipeNameByWidgetType = Object.freeze({
  mapping_v1: "mapping",
  slider_v1: "slider",
  // Read-only compatibility for legacy sessions while their resume window drains.
  mapping: "mapping",
  slider: "slider",
});

export function recipeFor(widgetType) {
  if (!widgetCapability(widgetType).supported) {
    return null;
  }
  const recipeName = recipeNameByWidgetType[widgetType];
  if (!recipeName || !Object.prototype.hasOwnProperty.call(recipes, recipeName)) {
    return null;
  }
  return recipes[recipeName];
}
