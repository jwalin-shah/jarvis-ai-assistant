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
  <path
    d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"
  />
</svg>
