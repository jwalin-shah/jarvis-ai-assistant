<script lang="ts">
  import type { IconProps } from './types';

  let { size = 24, class: className = '', strokeWidth = 2 }: IconProps = $props();
</script>

<svg
  viewBox="0 0 24 24"
  fill="none"
  stroke="currentColor"
  stroke-width={strokeWidth}
  width={size}
  height={size}
  class={className}
  aria-hidden="true"
>
  <circle cx="11" cy="11" r="8" />
  <path d="m21 21-4.35-4.35" />
</svg>
