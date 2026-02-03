<script lang="ts">
  interface Props {
    width?: string;
    height?: string;
    borderRadius?: string;
  }

  let { width = "100%", height = "16px", borderRadius = "4px" }: Props = $props();
</script>

<div
  class="skeleton"
  style="width: {width}; height: {height}; border-radius: {borderRadius};"
></div>

<style>
  .skeleton {
    background: linear-gradient(
      90deg,
      var(--bg-hover, #3a3a3c) 0%,
      var(--bg-active, #48484a) 50%,
      var(--bg-hover, #3a3a3c) 100%
    );
    background-size: 200% 100%;
    animation: shimmer 1.5s ease-in-out infinite;
  }

  @keyframes shimmer {
    0% {
      background-position: 200% 0;
    }
    100% {
      background-position: -200% 0;
    }
  }
</style>
