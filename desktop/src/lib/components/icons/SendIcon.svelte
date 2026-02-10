<script lang="ts">
  import type { IconProps } from './types';

  let { size = 24, class: className = '' }: IconProps = $props();
</script>

<svg
  viewBox="0 0 24 24"
  fill="currentColor"
  width={size}
  height={size}
  class={className}
  aria-hidden="true"
>
  <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
</svg>
