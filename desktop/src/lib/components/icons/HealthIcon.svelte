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
  <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
</svg>
