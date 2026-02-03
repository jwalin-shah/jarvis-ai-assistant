<script lang="ts">
  /**
   * Transition wrapper component for animated page/view transitions
   */

  import { onMount } from "svelte";

  type TransitionType = "fade" | "slide-up" | "slide-down" | "slide-left" | "slide-right" | "scale" | "none";

  interface Props {
    type?: TransitionType;
    duration?: number;
    delay?: number;
    children?: import("svelte").Snippet;
  }

  let { type = "fade", duration = 200, delay = 0, children }: Props = $props();

  let visible = $state(false);

  onMount(() => {
    // Trigger animation after mount
    const timer = setTimeout(() => {
      visible = true;
    }, delay);

    return () => clearTimeout(timer);
  });
</script>

<div
  class="transition-wrapper transition-{type}"
  class:visible
  style="--duration: {duration}ms; --delay: {delay}ms"
>
  {#if children}
    {@render children()}
  {/if}
</div>

<style>
  .transition-wrapper {
    opacity: 0;
    transition: opacity var(--duration) cubic-bezier(0.33, 1, 0.68, 1),
                transform var(--duration) cubic-bezier(0.33, 1, 0.68, 1);
  }

  .transition-wrapper.visible {
    opacity: 1;
  }

  /* Fade */
  .transition-fade {
    /* Just opacity, handled by base styles */
  }

  /* Slide up */
  .transition-slide-up {
    transform: translateY(20px);
  }

  .transition-slide-up.visible {
    transform: translateY(0);
  }

  /* Slide down */
  .transition-slide-down {
    transform: translateY(-20px);
  }

  .transition-slide-down.visible {
    transform: translateY(0);
  }

  /* Slide left */
  .transition-slide-left {
    transform: translateX(20px);
  }

  .transition-slide-left.visible {
    transform: translateX(0);
  }

  /* Slide right */
  .transition-slide-right {
    transform: translateX(-20px);
  }

  .transition-slide-right.visible {
    transform: translateX(0);
  }

  /* Scale */
  .transition-scale {
    transform: scale(0.95);
  }

  .transition-scale.visible {
    transform: scale(1);
  }

  /* None - instant visibility */
  .transition-none {
    opacity: 1;
    transform: none;
    transition: none;
  }

  /* Reduced motion */
  @media (prefers-reduced-motion: reduce) {
    .transition-wrapper {
      opacity: 1;
      transform: none;
      transition: none;
    }
  }

  :global(.reduce-motion) .transition-wrapper {
    opacity: 1;
    transform: none;
    transition: none;
  }
</style>
