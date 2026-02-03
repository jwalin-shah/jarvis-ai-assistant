<script lang="ts">
  interface Props {
    sender?: string;
  }

  let { sender }: Props = $props();
</script>

<div class="typing-indicator">
  <div class="typing-bubble">
    {#if sender}
      <span class="typing-sender">{sender}</span>
    {/if}
    <div class="typing-dots">
      <span class="dot"></span>
      <span class="dot"></span>
      <span class="dot"></span>
    </div>
  </div>
</div>

<style>
  .typing-indicator {
    display: flex;
    align-items: flex-start;
    padding: 4px 0;
    animation: fadeIn 0.2s ease;
  }

  @keyframes fadeIn {
    from {
      opacity: 0;
      transform: translateY(8px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  .typing-bubble {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 12px 16px;
    background: var(--bg-bubble-other);
    border-radius: var(--radius-xl);
    border-bottom-left-radius: var(--radius-sm);
  }

  .typing-sender {
    font-size: 11px;
    font-weight: 500;
    color: var(--color-primary);
  }

  .typing-dots {
    display: flex;
    align-items: center;
    gap: 4px;
    height: 20px;
  }

  .dot {
    width: 8px;
    height: 8px;
    background: var(--text-tertiary);
    border-radius: 50%;
    animation: typingBounce 1.4s ease-in-out infinite;
  }

  .dot:nth-child(1) {
    animation-delay: 0ms;
  }

  .dot:nth-child(2) {
    animation-delay: 160ms;
  }

  .dot:nth-child(3) {
    animation-delay: 320ms;
  }

  @keyframes typingBounce {
    0%, 60%, 100% {
      transform: translateY(0);
      opacity: 0.4;
    }
    30% {
      transform: translateY(-8px);
      opacity: 1;
    }
  }

  /* Reduced motion */
  @media (prefers-reduced-motion: reduce) {
    .typing-indicator {
      animation: none;
    }
    .dot {
      animation: typingPulse 1.4s ease-in-out infinite;
    }
    @keyframes typingPulse {
      0%, 100% { opacity: 0.4; }
      50% { opacity: 1; }
    }
  }
</style>
