export const widgetCapabilities = Object.freeze({
  version: "web-widget-capabilities-v2.1",
  supported: Object.freeze({
    slider: Object.freeze({
      keyboard_equivalent: true,
      live_visual: true,
    }),
    mapping: Object.freeze({
      keyboard_equivalent: true,
      live_visual: false,
    }),
  }),
  disabled: Object.freeze({
    live_input:
      "Live-input visuals are unavailable until reviewed render semantics are implemented.",
    click_region: "Diagram hit targets are not implemented.",
  }),
});

export const minimalWidgetCapabilities = Object.freeze({
  version: "web-widget-capabilities-v2.1",
  supported: Object.freeze({
    mapping: Object.freeze({
      keyboard_equivalent: true,
      live_visual: false,
    }),
  }),
  disabled: Object.freeze({
    slider: "Rich widget capabilities have not been confirmed by the server.",
    live_input: "Rich widget capabilities have not been confirmed by the server.",
    click_region: "Diagram hit targets are not implemented.",
  }),
});

let activeCapabilities = minimalWidgetCapabilities;
const knownWidgetTypes = new Set([
  "mapping",
  "slider",
  "live_input",
  "click_region",
]);

export function installWidgetCapabilities(candidate) {
  if (
    !candidate ||
    candidate.version !== widgetCapabilities.version ||
    !candidate.supported ||
    typeof candidate.supported !== "object" ||
    !candidate.disabled ||
    typeof candidate.disabled !== "object"
  ) {
    throw new Error("The server returned an invalid widget capability manifest.");
  }
  const supportedEntries = Object.entries(candidate.supported);
  const disabledEntries = Object.entries(candidate.disabled);
  const validSupported = supportedEntries.every(
    ([widgetType, capability]) =>
      knownWidgetTypes.has(widgetType) &&
      Object.prototype.hasOwnProperty.call(
        widgetCapabilities.supported,
        widgetType,
      ) &&
      capability?.keyboard_equivalent === true &&
      capability?.live_visual ===
        widgetCapabilities.supported[widgetType].live_visual,
  );
  const validDisabled = disabledEntries.every(
    ([widgetType, reason]) =>
      knownWidgetTypes.has(widgetType) &&
      typeof reason === "string" &&
      reason.trim(),
  );
  const overlap = supportedEntries.some(([widgetType]) =>
    Object.prototype.hasOwnProperty.call(candidate.disabled, widgetType),
  );
  if (!validSupported || !validDisabled || overlap) {
    throw new Error("The server returned an invalid widget capability manifest.");
  }
  activeCapabilities = Object.freeze({
    version: candidate.version,
    supported: Object.freeze({ ...candidate.supported }),
    disabled: Object.freeze({ ...candidate.disabled }),
  });
  return activeCapabilities;
}

export function installMinimalWidgetCapabilities() {
  return installWidgetCapabilities(minimalWidgetCapabilities);
}

export function widgetCapability(widgetType) {
  if (activeCapabilities.supported[widgetType]) {
    return {
      supported: true,
      ...activeCapabilities.supported[widgetType],
    };
  }
  return {
    supported: false,
    reason:
      activeCapabilities.disabled[widgetType] ??
      "This interaction type is not supported by this tutor.",
  };
}
