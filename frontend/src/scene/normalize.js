import { compileExpression, compilePlotEquation, ExpressionError } from "./expression.js";
import { curveSegmentsFromSamples, shadeSegmentsFromSamples } from "./segments.js";

export const VIEWPORT = Object.freeze({
  xMin: -5,
  xMax: 5,
  yMin: -5,
  yMax: 5,
});

const SAMPLE_INTERVALS = 400;
const BASE_SUBDIVISION_DEPTH = 2;
const MAX_ADAPTIVE_DEPTH = 6;
const FALLBACK_MESSAGE = "Rich visualization unavailable; use the slider control.";

function fallback() {
  return {
    scene: null,
    status: FALLBACK_MESSAGE,
  };
}

function constantValue(raw) {
  const expression = compileExpression(raw.trim());
  if (expression.symbols.length) {
    throw new ExpressionError("overlay coordinates and bounds must be constant");
  }
  const value = expression.evaluate();
  if (!Number.isFinite(value)) throw new ExpressionError("overlay value is not finite");
  return value;
}

function splitPointArguments(body) {
  let depth = 0;
  let comma = -1;
  for (let index = 0; index < body.length; index += 1) {
    const character = body[index];
    if (character === "(") depth += 1;
    else if (character === ")") depth -= 1;
    else if (character === "," && depth === 0) {
      if (comma !== -1) throw new ExpressionError("point accepts exactly two coordinates");
      comma = index;
    }
    if (depth < 0) throw new ExpressionError("invalid point parentheses");
  }
  if (depth !== 0 || comma === -1) {
    throw new ExpressionError("point accepts exactly two coordinates");
  }
  return [body.slice(0, comma), body.slice(comma + 1)];
}

function topLevelComparisons(raw) {
  const parts = [];
  const operators = [];
  let depth = 0;
  let start = 0;
  for (let index = 0; index < raw.length; index += 1) {
    const character = raw[index];
    if (character === "(") depth += 1;
    else if (character === ")") depth -= 1;
    if (depth < 0) throw new ExpressionError("invalid region parentheses");
    if (depth === 0 && (character === "<" || character === ">")) {
      parts.push(raw.slice(start, index).trim());
      const inclusive = raw[index + 1] === "=";
      operators.push(character + (inclusive ? "=" : ""));
      if (inclusive) index += 1;
      start = index + 1;
    } else if (depth === 0 && character === "=") {
      throw new ExpressionError("region uses an unsupported comparison");
    }
  }
  if (depth !== 0) throw new ExpressionError("invalid region parentheses");
  parts.push(raw.slice(start).trim());
  if (parts.some((part) => !part)) throw new ExpressionError("region comparison is incomplete");
  return { parts, operators };
}

function boundFromComparison(left, operator, right) {
  if (left === "x" && right !== "x") {
    const value = constantValue(right);
    if (operator.startsWith("<")) {
      return { side: "upper", value, inclusive: operator === "<=" };
    }
    return { side: "lower", value, inclusive: operator === ">=" };
  }
  if (right === "x" && left !== "x") {
    const value = constantValue(left);
    if (operator.startsWith("<")) {
      return { side: "lower", value, inclusive: operator === "<=" };
    }
    return { side: "upper", value, inclusive: operator === ">=" };
  }
  throw new ExpressionError("each region comparison must contain one bare x");
}

export function parseShade(raw) {
  if (typeof raw !== "string" || !raw.trim()) {
    throw new ExpressionError("overlay is empty");
  }
  const trimmed = raw.trim();
  if (trimmed.startsWith("point")) {
    const match = /^point\s*\((.*)\)$/.exec(trimmed);
    if (!match) throw new ExpressionError("invalid point overlay");
    const [xRaw, yRaw] = splitPointArguments(match[1]);
    const marker = { x: constantValue(xRaw), y: constantValue(yRaw) };
    if (
      marker.x < VIEWPORT.xMin ||
      marker.x > VIEWPORT.xMax ||
      marker.y < VIEWPORT.yMin ||
      marker.y > VIEWPORT.yMax
    ) {
      throw new ExpressionError("point is outside the viewport");
    }
    return { kind: "point", marker };
  }

  const { parts, operators } = topLevelComparisons(trimmed);
  if (operators.length < 1 || operators.length > 2) {
    throw new ExpressionError("region must contain one or two comparisons");
  }

  const bounds = [];
  if (operators.length === 1) {
    bounds.push(boundFromComparison(parts[0], operators[0], parts[1]));
  } else {
    if (parts[1] !== "x") {
      throw new ExpressionError("bounded region must have x in the middle");
    }
    bounds.push(boundFromComparison(parts[0], operators[0], parts[1]));
    bounds.push(boundFromComparison(parts[1], operators[1], parts[2]));
    if (bounds[0].side === bounds[1].side) {
      throw new ExpressionError("bounded region needs one lower and one upper bound");
    }
  }

  const lower = bounds.find((bound) => bound.side === "lower");
  const upper = bounds.find((bound) => bound.side === "upper");
  const xMin = Math.max(VIEWPORT.xMin, lower?.value ?? VIEWPORT.xMin);
  const xMax = Math.min(VIEWPORT.xMax, upper?.value ?? VIEWPORT.xMax);
  if (xMin >= xMax) throw new ExpressionError("region is empty in the viewport");

  return {
    kind: "region",
    xMin,
    xMax,
    minInclusive: lower?.value === xMin ? lower.inclusive : true,
    maxInclusive: upper?.value === xMax ? upper.inclusive : true,
  };
}

function evaluatedPoint(compiled, parameter, value, x) {
  const y = compiled.evaluate({ x, [parameter]: value });
  return Number.isFinite(y) ? { x, y } : null;
}

function crossesLikelyPole(left, middle, right) {
  if (!left || !middle || !right) return true;
  const signsChange =
    Math.sign(left.y) !== 0 &&
    Math.sign(right.y) !== 0 &&
    Math.sign(left.y) !== Math.sign(right.y);
  if (!signsChange) return false;
  const lower = Math.min(left.y, right.y);
  const upper = Math.max(left.y, right.y);
  return middle.y < lower || middle.y > upper;
}

function hasLargeMidpointSpike(left, middle, right) {
  if (!left || !middle || !right) return true;
  const endpointMagnitude = Math.max(
    1e-12,
    Math.abs(left.y),
    Math.abs(right.y),
  );
  return Math.abs(middle.y) > endpointMagnitude * 4;
}

function sampleInterval(
  compiled,
  parameter,
  value,
  left,
  right,
  depth,
  points,
) {
  const middle = evaluatedPoint(
    compiled,
    parameter,
    value,
    (left.x + right.x) / 2,
  );
  if (!middle) {
    points.push(null, right);
    return;
  }

  if (crossesLikelyPole(left, middle, right)) {
    points.push(null, right);
    return;
  }

  const suspicious = hasLargeMidpointSpike(left, middle, right);
  if (depth < BASE_SUBDIVISION_DEPTH || (suspicious && depth < MAX_ADAPTIVE_DEPTH)) {
    sampleInterval(
      compiled,
      parameter,
      value,
      left,
      middle,
      depth + 1,
      points,
    );
    sampleInterval(
      compiled,
      parameter,
      value,
      middle,
      right,
      depth + 1,
      points,
    );
    return;
  }
  if (suspicious) points.push(null);
  points.push(right);
}

function sample(compiled, parameter, value, xMin, xMax) {
  const fraction = (xMax - xMin) / (VIEWPORT.xMax - VIEWPORT.xMin);
  const intervals = Math.max(1, Math.ceil(SAMPLE_INTERVALS * fraction));
  const first = evaluatedPoint(compiled, parameter, value, xMin);
  const points = [first];
  let left = first;
  for (let index = 1; index <= intervals; index += 1) {
    const x = xMin + ((xMax - xMin) * index) / intervals;
    const right = evaluatedPoint(compiled, parameter, value, x);
    if (!left || !right) {
      points.push(null, right);
    } else {
      sampleInterval(compiled, parameter, value, left, right, 0, points);
    }
    left = right;
  }
  return points;
}

function isResponsive(compiled, parameter, minimum, maximum) {
  const values = [
    minimum,
    maximum,
    (minimum + maximum) / 2,
    minimum + (maximum - minimum) * 0.3819660112501051,
  ];
  const xs = [-4.1, -2.3, -0.7, 0.6, 1.9, 3.7];
  for (const x of xs) {
    const results = values
      .map((value) => compiled.evaluate({ x, [parameter]: value }))
      .filter(Number.isFinite);
    for (let index = 1; index < results.length; index += 1) {
      const scale = Math.max(1, Math.abs(results[0]), Math.abs(results[index]));
      if (Math.abs(results[index] - results[0]) > 1e-9 * scale) return true;
    }
  }
  return false;
}

export function normalizeSlider(config, state) {
  try {
    const params = config?.params;
    const minimum = params?.min;
    const maximum = params?.max;
    const value = state?.value;
    if (
      typeof minimum !== "number" ||
      typeof maximum !== "number" ||
      !Number.isFinite(minimum) ||
      !Number.isFinite(maximum) ||
      maximum <= minimum ||
      typeof value !== "number" ||
      !Number.isFinite(value)
    ) {
      return fallback();
    }

    const { compiled, parameter } = compilePlotEquation(params.plot);
    if (!isResponsive(compiled, parameter, minimum, maximum)) return fallback();

    const curveSamples = sample(
      compiled,
      parameter,
      value,
      VIEWPORT.xMin,
      VIEWPORT.xMax,
    );
    const curveSegments = curveSegmentsFromSamples(curveSamples, VIEWPORT);
    if (!curveSegments.length) return fallback();

    let marker = null;
    let shadeSegments = [];
    let status = "";
    if (params.shade != null) {
      const overlay = parseShade(params.shade);
      if (overlay.kind === "point") {
        marker = overlay.marker;
      } else {
        const shadeSamples = sample(
          compiled,
          parameter,
          value,
          overlay.xMin,
          overlay.xMax,
        );
        shadeSegments = shadeSegmentsFromSamples(shadeSamples, VIEWPORT);
        if (!shadeSegments.length) {
          status = curveSegmentsFromSamples(shadeSamples, VIEWPORT).length
            ? "The shaded area is zero at this slider value."
            : "No drawable curve lies in the shaded interval at this slider value.";
        }
      }
    }

    return {
      scene: {
        kind: "plot",
        viewport: VIEWPORT,
        curveSegments,
        shadeSegments,
        marker,
      },
      status,
    };
  } catch {
    return fallback();
  }
}
