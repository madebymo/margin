import { describe, expect, it } from "vitest";

import {
  clearExpressionCacheForTests,
  expressionCacheSizeForTests,
} from "../src/scene/expression.js";
import { normalizeSlider, parseShade, VIEWPORT } from "../src/scene/normalize.js";

function slider(plot, shade = null, minimum = -2, maximum = 4) {
  return { params: { min: minimum, max: maximum, step: 0.1, plot, shade } };
}

describe("slider normalization", () => {
  it("normalizes a responsive plot and exact point marker", () => {
    const result = normalizeSlider(
      slider("y = m*x", "point(pi/2, sqrt(2))"),
      { value: 1.5 },
    );
    expect(result.status).toBe("");
    expect(result.scene.kind).toBe("plot");
    expect(result.scene.viewport).toEqual(VIEWPORT);
    expect(result.scene.curveSegments.length).toBeGreaterThan(0);
    expect(result.scene.marker.x).toBeCloseTo(Math.PI / 2);
    expect(result.scene.marker.y).toBeCloseTo(Math.sqrt(2));
  });

  it("changes curve geometry when the slider value changes", () => {
    const config = slider("y = m*x");
    const first = normalizeSlider(config, { value: 1 }).scene.curveSegments;
    const second = normalizeSlider(config, { value: 2 }).scene.curveSegments;
    expect(second).not.toEqual(first);
  });

  it("reuses the sole parse cache across repeated slider updates", () => {
    for (const config of [
      slider("y = m*x", "point(pi/2, sqrt(2))"),
      slider("y = m*x", "0 <= x <= sqrt(2)"),
    ]) {
      clearExpressionCacheForTests();
      expect(normalizeSlider(config, { value: 1 }).scene).not.toBeNull();
      const parsedOnce = expressionCacheSizeForTests();
      expect(parsedOnce).toBeGreaterThan(1);
      expect(normalizeSlider(config, { value: 1.5 }).scene).not.toBeNull();
      expect(normalizeSlider(config, { value: 2 }).scene).not.toBeNull();
      expect(expressionCacheSizeForTests()).toBe(parsedOnce);
    }
  });

  it("normalizes one-sided, bounded, reversed, and exact interval bounds", () => {
    expect(parseShade("x >= 0")).toMatchObject({ kind: "region", xMin: 0, xMax: 5 });
    expect(parseShade("0 <= x <= 2")).toMatchObject({
      kind: "region",
      xMin: 0,
      xMax: 2,
    });
    expect(parseShade("2 >= x >= sqrt(2)")).toMatchObject({
      kind: "region",
      xMin: Math.sqrt(2),
      xMax: 2,
    });
    expect(parseShade("x < pi/2")).toMatchObject({
      kind: "region",
      xMin: -5,
      xMax: Math.PI / 2,
    });
  });

  it("builds region polygons between the curve and x-axis", () => {
    const result = normalizeSlider(slider("y = a*x", "0 <= x <= 2"), { value: 2 });
    expect(result.status).toBe("");
    expect(result.scene.shadeSegments.length).toBeGreaterThan(0);
    for (const polygon of result.scene.shadeSegments) {
      expect(Math.min(...polygon.map((point) => point.x))).toBeGreaterThanOrEqual(0);
      expect(Math.max(...polygon.map((point) => point.x))).toBeLessThanOrEqual(2);
      expect(polygon.some((point) => point.y === 0)).toBe(true);
    }
  });

  it("keeps a valid region scene when the current curve has zero area", () => {
    const result = normalizeSlider(
      slider("y = a*x", "0 <= x <= 2", 0, 4),
      { value: 0 },
    );
    expect(result.status).toMatch(/zero/);
    expect(result.scene.curveSegments.length).toBeGreaterThan(0);
    expect(result.scene.shadeSegments).toEqual([]);
  });

  it("keeps drawable finite pieces of ln(x)", () => {
    const result = normalizeSlider(slider("y = a*ln(x)"), { value: 1 });
    expect(result.status).toBe("");
    expect(result.scene.curveSegments.length).toBeGreaterThan(0);
    expect(
      result.scene.curveSegments.every((segment) =>
        segment.every((point) => Number.isFinite(point.x) && Number.isFinite(point.y)),
      ),
    ).toBe(true);
  });

  it("splits tan(x) instead of joining opposite viewport bounds across poles", () => {
    const result = normalizeSlider(slider("y = a*tan(x)"), { value: 1 });
    expect(result.status).toBe("");
    expect(result.scene.curveSegments.length).toBeGreaterThan(1);
    for (const segment of result.scene.curveSegments) {
      for (let index = 1; index < segment.length; index += 1) {
        const previous = segment[index - 1];
        const point = segment[index];
        expect(
          (previous.y === VIEWPORT.yMax && point.y === VIEWPORT.yMin) ||
            (previous.y === VIEWPORT.yMin && point.y === VIEWPORT.yMax),
        ).toBe(false);
      }
    }
  });

  it("adaptively finds a pole between finite in-bounds base samples", () => {
    const result = normalizeSlider(
      slider("y = a/(x-0.0125)", null, 0.001, 0.01),
      { value: 0.001 },
    );

    expect(result.scene).not.toBeNull();
    const crossesPole = result.scene.curveSegments.some((segment) => {
      const xs = segment.map((point) => point.x);
      return Math.min(...xs) < 0.0125 && Math.max(...xs) > 0.0125;
    });
    expect(crossesPole).toBe(false);
  });

  it("does not bridge an off-midpoint pole", () => {
    const result = normalizeSlider(
      slider("y = a/(x-0.001)", null, 0.001, 0.01),
      { value: 0.001 },
    );

    expect(result.scene).not.toBeNull();
    const crossesPole = result.scene.curveSegments.some((segment) => {
      const xs = segment.map((point) => point.x);
      return Math.min(...xs) < 0.001 && Math.max(...xs) > 0.001;
    });
    expect(crossesPole).toBe(false);
  });

  it("falls back explicitly for unsupported, non-responsive, or undrawable scenes", () => {
    const cases = [
      slider("y = m*x + b"),
      slider("y = m-m+x"),
      slider("y = m*x", "circle(0, 0)"),
      slider("y = m*sqrt(-1)*x"),
    ];
    for (const config of cases) {
      const result = normalizeSlider(config, { value: 1 });
      expect(result.scene).toBeNull();
      expect(result.status).toContain("slider");
    }
  });

  it("preserves a drawable main curve when a valid shaded interval has no finite span", () => {
    const result = normalizeSlider(slider("y = m*ln(x)", "x <= -1"), { value: 1 });
    expect(result.scene).not.toBeNull();
    expect(result.scene.curveSegments.length).toBeGreaterThan(0);
    expect(result.scene.shadeSegments).toEqual([]);
    expect(result.status).toMatch(/No drawable curve/);
  });
});
