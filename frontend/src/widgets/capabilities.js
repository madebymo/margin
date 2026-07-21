export const widgetCapabilities = Object.freeze({
  version: "web-widget-capabilities-v2.2",
  supported: Object.freeze({
    slider_v1: Object.freeze({
      keyboard_equivalent: true,
      live_visual: false,
    }),
    mapping_v1: Object.freeze({
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
  version: "web-widget-capabilities-v2.2",
  supported: Object.freeze({
    mapping_v1: Object.freeze({
      keyboard_equivalent: true,
      live_visual: false,
    }),
  }),
  disabled: Object.freeze({
    slider_v1: "Rich widget capabilities have not been confirmed by the server.",
    live_input: "Rich widget capabilities have not been confirmed by the server.",
    click_region: "Diagram hit targets are not implemented.",
  }),
});

let activeCapabilities = minimalWidgetCapabilities;
const knownWidgetTypes = new Set([
  "mapping_v1",
  "slider_v1",
  "live_input",
  "click_region",
]);
const legacyWidgetAliases = Object.freeze({
  mapping: "mapping_v1",
  slider: "slider_v1",
});

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
  // Legacy v1 transcripts used unversioned widget names. They may finish via
  // the same reviewed frontend implementation, but API v2 never advertises
  // those aliases in its capability manifest.
  const canonicalType = legacyWidgetAliases[widgetType] ?? widgetType;
  if (activeCapabilities.supported[canonicalType]) {
    return {
      supported: true,
      ...activeCapabilities.supported[canonicalType],
    };
  }
  return {
    supported: false,
    reason:
      activeCapabilities.disabled[canonicalType] ??
      "This interaction type is not supported by this tutor.",
  };
}
