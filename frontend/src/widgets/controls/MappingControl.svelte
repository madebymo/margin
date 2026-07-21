<script>
  export let config;
  export let state;
  export let disabled = false;
  export let onChange = () => {};

  function selectRight(index, value) {
    const rows = state.rows.map((row, rowIndex) =>
      rowIndex === index ? { ...row, value } : row,
    );
    onChange({ ...state, rows });
  }
</script>

<div
  class="widget-controls mapping-controls"
  role="group"
  aria-label={config.prompt || "Matching interaction"}
>
  {#each state.rows as row, index}
    <div class="maprow">
      <span aria-label={row.spokenText}>{row.label}</span>
      <select
        value={row.value}
        aria-label={`Match ${row.spokenText}`}
        {disabled}
        on:change={(event) => selectRight(index, event.currentTarget.value)}
      >
        <option value="">choose…</option>
        {#each state.rightOptions as right}
          <option value={right.id}>{right.label}</option>
        {/each}
      </select>
    </div>
  {/each}
</div>
