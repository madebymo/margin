<script>
  const componentId = $props.id();
  const gradientId = `${componentId}-margin-gradient`;
</script>

<svg class="margin-mark" viewBox="0 0 100 100" fill="none" aria-hidden="true">
  <defs>
    <linearGradient id={gradientId} x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#f5822a" />
      <stop offset="1" stop-color="#1a6fe0" />
    </linearGradient>
  </defs>
  <path
    d="M18 82 V22 L50 56 L82 22 V82"
    stroke={`url(#${gradientId})`}
    stroke-width="16"
    stroke-linecap="square"
    stroke-linejoin="miter"
  />
</svg>
