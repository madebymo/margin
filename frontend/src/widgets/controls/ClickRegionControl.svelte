<script>
  export let config;
  export let state;
  export let disabled = false;
  export let onChange = () => {};

  function toggle(regionId) {
    const selected = state.selected.includes(regionId)
      ? state.selected.filter((id) => id !== regionId)
      : [...state.selected, regionId];
    onChange({ ...state, selected });
  }
</script>

<div class="widget-controls">
  {#each config.regions as region}
    <button
      type="button"
      class="region"
      class:selected={state.selected.includes(region.id)}
      data-id={region.id}
      aria-pressed={state.selected.includes(region.id)}
      {disabled}
      on:click={() => toggle(region.id)}
    >{region.label || region.id}</button>
  {/each}
</div>
