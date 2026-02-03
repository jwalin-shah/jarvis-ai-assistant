<script lang="ts">
  import { toast } from "../stores/toast";

  interface Props {
    messageId: number;
    onReact: (reaction: string) => void;
    visible: boolean;
  }

  let { messageId, onReact, visible }: Props = $props();

  const reactions = ["❤️", "👍", "👎", "😂", "😮", "😢"];

  function handleReaction(emoji: string) {
    onReact(emoji);
    toast.success(`Reacted with ${emoji}`);
  }
</script>

{#if visible}
  <div class="reactions-picker" role="menu" aria-label="React to message">
    {#each reactions as emoji}
      <button
        class="reaction-btn"
        onclick={() => handleReaction(emoji)}
        title="React with {emoji}"
      >
        {emoji}
      </button>
    {/each}
  </div>
{/if}

<style>
  .reactions-picker {
    display: flex;
    gap: 4px;
    padding: 8px 12px;
    background: var(--bg-elevated);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-xl);
    box-shadow: var(--shadow-lg);
    animation: scaleIn 0.15s cubic-bezier(0.16, 1, 0.3, 1);
  }

  @keyframes scaleIn {
    from {
      opacity: 0;
      transform: scale(0.9);
    }
    to {
      opacity: 1;
      transform: scale(1);
    }
  }

  .reaction-btn {
    width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: transparent;
    border: none;
    border-radius: var(--radius-md);
    font-size: 20px;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .reaction-btn:hover {
    background: var(--bg-hover);
    transform: scale(1.2);
  }

  .reaction-btn:active {
    transform: scale(0.95);
  }

  /* Reduced motion */
  @media (prefers-reduced-motion: reduce) {
    .reactions-picker {
      animation: none;
    }
    .reaction-btn:hover {
      transform: none;
    }
  }
</style>
