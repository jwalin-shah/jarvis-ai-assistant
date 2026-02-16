<script lang="ts">
  interface Props {
    width?: string;
    height?: string;
    borderRadius?: string;
    animated?: boolean;
  }

  let { width = '100%', height = '16px', borderRadius = '4px', animated = true }: Props = $props();
</script>

<div
  class="skeleton"
  class:animated
  style="width: {width}; height: {height}; border-radius: {borderRadius};"
  aria-hidden="true"
></div>

<style>
  .skeleton {
    background: var(--bg-surface);
    position: relative;
    overflow: hidden;
  }

  .skeleton.animated::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(
      90deg,
      transparent 0%,
      rgba(255, 255, 255, 0.05) 20%,
      rgba(255, 255, 255, 0.1) 50%,
      rgba(255, 255, 255, 0.05) 80%,
      transparent 100%
    );
    background-size: 200% 100%;
    animation: shimmer 1.5s ease-in-out infinite;
  }

  :global(:root.light) .skeleton.animated::after {
    background: linear-gradient(
      90deg,
      transparent 0%,
      rgba(0, 0, 0, 0.03) 20%,
      rgba(0, 0, 0, 0.06) 50%,
      rgba(0, 0, 0, 0.03) 80%,
      transparent 100%
    );
    background-size: 200% 100%;
  }

  @keyframes shimmer {
    0% {
      background-position: 200% 0;
    }
    100% {
      background-position: -200% 0;
    }
  }

  /* Reduced motion */
  @media (prefers-reduced-motion: reduce) {
    .skeleton.animated::after {
      animation: none;
      background: rgba(255, 255, 255, 0.05);
    }
  }

  :global(.reduce-motion) .skeleton.animated::after {
    animation: none;
    background: rgba(255, 255, 255, 0.05);
  }
</style>
