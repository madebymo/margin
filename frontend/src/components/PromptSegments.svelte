<script>
  export let segments = [];

  const width = 360;
  const height = 210;
  const inset = 30;

  function kindOf(segment) {
    return segment?.kind ?? segment?.type ?? "text";
  }

  function spokenText(segment) {
    return segment?.spoken_text ?? segment?.description ?? segment?.alt_text ?? "";
  }

  function tableHeaders(segment) {
    return segment?.column_headers ?? segment?.headers ?? [];
  }

  function plotModel(segment) {
    const series = (segment?.series ?? []).map((entry) => ({
      label: String(entry?.label ?? "series"),
      points: (entry?.points ?? [])
        .map((point) => ({ x: Number(point?.x), y: Number(point?.y) }))
        .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y)),
    }));
    const points = series.flatMap((entry) => entry.points);
    if (points.length < 2) return null;
    const xs = points.map((point) => point.x);
    const ys = points.map((point) => point.y);
    let minX = Math.min(...xs);
    let maxX = Math.max(...xs);
    let minY = Math.min(...ys);
    let maxY = Math.max(...ys);
    if (minX === maxX) [minX, maxX] = [minX - 1, maxX + 1];
    if (minY === maxY) [minY, maxY] = [minY - 1, maxY + 1];
    const project = (point) => ({
      x: inset + ((point.x - minX) / (maxX - minX)) * (width - 2 * inset),
      y: height - inset - ((point.y - minY) / (maxY - minY)) * (height - 2 * inset),
    });
    return {
      minX,
      maxX,
      minY,
      maxY,
      series: series.map((entry) => ({
        ...entry,
        projected: entry.points.map(project),
      })),
    };
  }

  function pointList(points) {
    return points.map((point) => `${point.x},${point.y}`).join(" ");
  }
</script>

<div class="structured-prompt">
  {#each segments as segment}
    {@const kind = kindOf(segment)}
    {#if kind === "math"}
      <span
        class="math-segment"
        aria-label={spokenText(segment) || undefined}
      >{segment.expression ?? segment.latex ?? segment.math}</span>
    {:else if kind === "blank"}
      <span class="blank-segment">{segment.label ?? "___"}</span>
    {:else if kind === "table"}
      <div class="segment-table-wrap">
        <table>
          <caption>{segment.caption ?? "Data table"}</caption>
          {#if tableHeaders(segment).length}
            <thead>
              <tr>
                {#each tableHeaders(segment) as header}<th scope="col">{header}</th>{/each}
              </tr>
            </thead>
          {/if}
          <tbody>
            {#each segment.rows ?? [] as row}
              <tr>{#each row as cell}<td>{cell}</td>{/each}</tr>
            {/each}
          </tbody>
        </table>
        {#if spokenText(segment)}<p class="sr-only">{spokenText(segment)}</p>{/if}
      </div>
    {:else if kind === "plot"}
      {@const plot = plotModel(segment)}
      <figure class="static-plot" aria-label={spokenText(segment)}>
        <figcaption>{segment.title ?? "Static plot"}</figcaption>
        {#if plot}
          <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-hidden="true">
            <line x1={inset} y1={height - inset} x2={width - inset} y2={height - inset} />
            <line x1={inset} y1={inset} x2={inset} y2={height - inset} />
            <text x={width / 2} y={height - 5}>{segment.x_label}</text>
            <text x="8" y={height / 2}>{segment.y_label}</text>
            {#each plot.series as entry, seriesIndex}
              <polyline
                class={`plot-series series-${seriesIndex % 4}`}
                points={pointList(entry.projected)}
              />
              {#each entry.projected as point}
                <circle
                  class={`plot-series series-${seriesIndex % 4}`}
                  cx={point.x}
                  cy={point.y}
                  r="4"
                />
              {/each}
            {/each}
          </svg>
        {/if}
        <p class="plot-description">{spokenText(segment) || "Static plot"}</p>
        {#if segment.equivalent_table}
          <div class="segment-table-wrap plot-data">
            <table>
              <caption>{segment.equivalent_table.caption ?? "Plot data"}</caption>
              <thead>
                <tr>
                  {#each tableHeaders(segment.equivalent_table) as header}
                    <th scope="col">{header}</th>
                  {/each}
                </tr>
              </thead>
              <tbody>
                {#each segment.equivalent_table.rows ?? [] as row}
                  <tr>{#each row as cell}<td>{cell}</td>{/each}</tr>
                {/each}
              </tbody>
            </table>
          </div>
        {/if}
      </figure>
    {:else}
      <span>{segment.text ?? ""}</span>
    {/if}
  {/each}
</div>
