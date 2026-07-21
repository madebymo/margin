<script>
  export let segments = [];

  function kindOf(segment) {
    return segment?.kind ?? segment?.type ?? "text";
  }

  function spokenText(segment) {
    return segment?.spoken_text ?? segment?.description ?? segment?.alt_text ?? "";
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
      <span class="segment-table-wrap">
        {#if segment.caption}<span class="segment-caption">{segment.caption}</span>{/if}
        <table>
          {#if segment.headers?.length}
            <thead><tr>{#each segment.headers as header}<th scope="col">{header}</th>{/each}</tr></thead>
          {/if}
          <tbody>
            {#each segment.rows ?? [] as row}
              <tr>{#each row as cell}<td>{cell}</td>{/each}</tr>
            {/each}
          </tbody>
        </table>
        {#if spokenText(segment)}<span class="sr-only">{spokenText(segment)}</span>{/if}
      </span>
    {:else if kind === "plot"}
      <span class="static-plot-description" role="img" aria-label={spokenText(segment)}>
        {spokenText(segment) || "Static plot"}
      </span>
    {:else}
      <span>{segment.text ?? ""}</span>
    {/if}
  {/each}
</div>
