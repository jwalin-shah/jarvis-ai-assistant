<script lang="ts">
  import Skeleton from "./Skeleton.svelte";

  // Variable widths for natural look
  const messageWidths = ["75%", "60%", "80%", "50%", "70%", "65%"];
  const alignments = [false, true, false, true, false, true]; // alternating for visual variety
</script>

<div class="message-skeleton-container">
  {#each messageWidths as width, i}
    <div class="message-skeleton" class:from-me={alignments[i]}>
      <div class="bubble-skeleton">
        <Skeleton width={width} height="16px" borderRadius="4px" />
        {#if i % 2 === 0}
          <Skeleton width="40%" height="14px" borderRadius="4px" />
        {/if}
      </div>
    </div>
  {/each}
</div>

<style>
  .message-skeleton-container {
    display: flex;
    flex-direction: column;
    gap: 12px;
    padding: 16px;
  }

  .message-skeleton {
    display: flex;
    flex-direction: column;
    max-width: 70%;
  }

  .message-skeleton.from-me {
    align-self: flex-end;
  }

  .bubble-skeleton {
    padding: 12px 16px;
    border-radius: 18px;
    background: var(--bg-secondary, #2c2c2e);
    display: flex;
    flex-direction: column;
    gap: 8px;
    min-width: 120px;
  }

  .message-skeleton.from-me .bubble-skeleton {
    background: rgba(11, 147, 246, 0.3);
  }
</style>
