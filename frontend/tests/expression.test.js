import { beforeEach, describe, expect, it } from "vitest";

import {
  clearExpressionCacheForTests,
  compileExpression,
  compilePlotEquation,
  expressionCacheSizeForTests,
} from "../src/scene/expression.js";

beforeEach(() => clearExpressionCacheForTests());

describe("expression compiler", () => {
  it("supports the p5 arithmetic, constants, functions, and precedence", () => {
    const expression = compileExpression(
      "-2^2 + 2^-2 + sin(pi/2) + cos(0) + tan(0) + sec(0) + exp(0) + log(1) + ln(1) + sqrt(4)",
    );
    expect(expression.evaluate()).toBeCloseTo(2.25);
    expect(compileExpression("2^3^2").evaluate()).toBe(512);
    expect(compileExpression("2**3").evaluate()).toBe(8);
  });

  it("allows only unambiguous implicit multiplication", () => {
    expect(compileExpression("2x + 3(x + 1)").evaluate({ x: 2 })).toBe(13);
    expect(compileExpression("(x + 1)(x - 1)").evaluate({ x: 3 })).toBe(8);
    expect(() => compileExpression("m(x + 1)")).toThrow(/unknown function/);
    expect(() => compileExpression("unknown(x)")).toThrow(/unknown function/);
  });

  it("returns NaN, rather than throwing, for non-real or non-finite evaluations", () => {
    expect(compileExpression("ln(x)").evaluate({ x: -1 })).toBeNaN();
    expect(compileExpression("1/x").evaluate({ x: 0 })).toBeNaN();
    expect(compileExpression("sqrt(x)").evaluate({ x: -1 })).toBeNaN();
  });

  it("rejects syntax outside the small grammar", () => {
    expect(() => compileExpression("x[0]")).toThrow();
    expect(() => compileExpression("abs(x)")).toThrow();
    expect(() => compileExpression("sin x")).toThrow();
    expect(() => compileExpression("1e999")).toThrow(/finite/);
    expect(() => compileExpression("sin(x, 2)")).toThrow();
  });

  it("requires y = expression with exactly x and one inferred parameter", () => {
    const plot = compilePlotEquation(" y = a * sin(x) ");
    expect(plot.parameter).toBe("a");
    expect(plot.compiled.evaluate({ x: Math.PI / 2, a: 3 })).toBeCloseTo(3);

    expect(() => compilePlotEquation("x = m*x")).toThrow();
    expect(() => compilePlotEquation("y = x^2")).toThrow();
    expect(() => compilePlotEquation("y = m")).toThrow();
    expect(() => compilePlotEquation("y = x*y")).toThrow();
    expect(() => compilePlotEquation("y = m*x + b")).toThrow();
    expect(compilePlotEquation("y = speed*x").parameter).toBe("speed");
    expect(compilePlotEquation("y = a1*x").parameter).toBe("a1");
    expect(compilePlotEquation("y = rate*x + 1e-3").parameter).toBe("rate");
  });

  it("uses one raw-expression cache for successes and failures", () => {
    const first = compileExpression("sqrt(2)");
    const second = compileExpression("sqrt(2)");
    expect(second).toBe(first);
    expect(expressionCacheSizeForTests()).toBe(1);

    expect(() => compileExpression("bad(x)")).toThrow();
    expect(() => compileExpression("bad(x)")).toThrow();
    expect(expressionCacheSizeForTests()).toBe(2);
  });
});
