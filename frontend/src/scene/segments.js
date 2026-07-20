function isPoint(point) {
  return point && Number.isFinite(point.x) && Number.isFinite(point.y);
}

function crossesOppositeBounds(left, right, viewport) {
  return (
    (left.y > viewport.yMax && right.y < viewport.yMin) ||
    (left.y < viewport.yMin && right.y > viewport.yMax)
  );
}

export function splitSampleRuns(samples, viewport) {
  const runs = [];
  let run = [];
  const finish = () => {
    if (run.length >= 2) runs.push(run);
    run = [];
  };

  for (const point of samples) {
    if (!isPoint(point)) {
      finish();
      continue;
    }
    const previous = run.at(-1);
    if (previous && crossesOppositeBounds(previous, point, viewport)) {
      finish();
    }
    run.push(point);
  }
  finish();
  return runs;
}

function clipLine(start, end, viewport) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const p = [-dx, dx, -dy, dy];
  const q = [
    start.x - viewport.xMin,
    viewport.xMax - start.x,
    start.y - viewport.yMin,
    viewport.yMax - start.y,
  ];
  let first = 0;
  let last = 1;

  for (let index = 0; index < 4; index += 1) {
    if (p[index] === 0) {
      if (q[index] < 0) return null;
      continue;
    }
    const ratio = q[index] / p[index];
    if (p[index] < 0) first = Math.max(first, ratio);
    else last = Math.min(last, ratio);
    if (first > last) return null;
  }

  return [
    { x: start.x + first * dx, y: start.y + first * dy },
    { x: start.x + last * dx, y: start.y + last * dy },
  ];
}

function samePoint(left, right) {
  return Math.abs(left.x - right.x) < 1e-12 && Math.abs(left.y - right.y) < 1e-12;
}

function hasLength(points) {
  return points.some((point, index) => index > 0 && !samePoint(points[index - 1], point));
}

export function curveSegmentsFromSamples(samples, viewport) {
  const clippedRuns = [];
  for (const run of splitSampleRuns(samples, viewport)) {
    let clippedRun = [];
    const finish = () => {
      if (clippedRun.length >= 2 && hasLength(clippedRun)) clippedRuns.push(clippedRun);
      clippedRun = [];
    };

    for (let index = 1; index < run.length; index += 1) {
      const clipped = clipLine(run[index - 1], run[index], viewport);
      if (!clipped) {
        finish();
      } else if (!clippedRun.length) {
        clippedRun.push(...clipped);
      } else if (samePoint(clippedRun.at(-1), clipped[0])) {
        clippedRun.push(clipped[1]);
      } else {
        finish();
        clippedRun.push(...clipped);
      }
    }
    finish();
  }
  return clippedRuns;
}

function intersectVertical(start, end, x) {
  const ratio = (x - start.x) / (end.x - start.x);
  return { x, y: start.y + ratio * (end.y - start.y) };
}

function intersectHorizontal(start, end, y) {
  const ratio = (y - start.y) / (end.y - start.y);
  return { x: start.x + ratio * (end.x - start.x), y };
}

function clipAgainst(points, inside, intersection) {
  if (!points.length) return [];
  const output = [];
  let previous = points.at(-1);
  let previousInside = inside(previous);
  for (const point of points) {
    const pointInside = inside(point);
    if (pointInside !== previousInside) {
      output.push(intersection(previous, point));
    }
    if (pointInside) output.push(point);
    previous = point;
    previousInside = pointInside;
  }
  return output;
}

export function clipPolygon(points, viewport) {
  let result = points;
  result = clipAgainst(
    result,
    (point) => point.x >= viewport.xMin,
    (start, end) => intersectVertical(start, end, viewport.xMin),
  );
  result = clipAgainst(
    result,
    (point) => point.x <= viewport.xMax,
    (start, end) => intersectVertical(start, end, viewport.xMax),
  );
  result = clipAgainst(
    result,
    (point) => point.y >= viewport.yMin,
    (start, end) => intersectHorizontal(start, end, viewport.yMin),
  );
  return clipAgainst(
    result,
    (point) => point.y <= viewport.yMax,
    (start, end) => intersectHorizontal(start, end, viewport.yMax),
  );
}

function polygonArea(points) {
  let twiceArea = 0;
  for (let index = 0; index < points.length; index += 1) {
    const point = points[index];
    const next = points[(index + 1) % points.length];
    twiceArea += point.x * next.y - next.x * point.y;
  }
  return Math.abs(twiceArea) / 2;
}

export function shadeSegmentsFromSamples(samples, viewport) {
  const polygons = [];
  for (const run of splitSampleRuns(samples, viewport)) {
    const polygon = [
      { x: run[0].x, y: 0 },
      ...run,
      { x: run.at(-1).x, y: 0 },
    ];
    const clipped = clipPolygon(polygon, viewport);
    if (clipped.length >= 3 && polygonArea(clipped) > 1e-10) {
      polygons.push(clipped);
    }
  }
  return polygons;
}
