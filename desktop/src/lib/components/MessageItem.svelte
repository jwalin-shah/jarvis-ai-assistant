<script lang="ts">
  import type { Message } from "../api/types";
  import { formatTime } from "../utils/date";

  interface Props {
    message: Message;
    isNew?: boolean;
    isHighlighted?: boolean;
    isKeyboardFocused?: boolean;
    isOptimistic?: boolean;
    optimisticStatus?: "sending" | "sent" | "failed" | null;
    optimisticId?: string;
    showSenderName?: boolean;
    onRetry?: (id: string) => void;
    onDismiss?: (id: string) => void;
  }

  let {
    message,
    isNew = false,
    isHighlighted = false,
    isKeyboardFocused = false,
    isOptimistic = false,
    optimisticStatus = null,
    optimisticId,
    showSenderName = false,
    onRetry,
    onDismiss
  } = $props<Props>();

</script>

<div
  class="message"
  class:from-me={message.is_from_me}
  class:new-message={isNew}
  class:highlighted={isHighlighted}
  class:keyboard-focused={isKeyboardFocused}
  class:optimistic={isOptimistic}
  class:optimistic-sending={optimisticStatus === "sending"}
  class:optimistic-failed={optimisticStatus === "failed"}
  data-message-id={message.id}
  tabindex="-1"
  role="article"
  aria-label={`Message from ${message.is_from_me ? "you" : message.sender_name || message.sender}`}
>
  <div class="bubble" class:from-me={message.is_from_me}>
    {#if !message.is_from_me && showSenderName}
      <span class="sender">{message.sender_name || message.sender}</span>
    {/if}
    <p>{message.text}</p>
    {#if message.attachments && message.attachments.length > 0}
      <div class="attachments">
        {#each message.attachments as attachment}
          <div class="attachment">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/>
            </svg>
            <span>{attachment.filename}</span>
          </div>
        {/each}
      </div>
    {/if}
    {#if optimisticStatus === "sending"}
      <span class="optimistic-status sending">
        <span class="sending-dot"></span>
        Sending...
      </span>
    {:else if optimisticStatus === "failed"}
      <span class="optimistic-status failed">
        Failed to send
      </span>
    {:else}
      <span class="time">{formatTime(message.date)}</span>
    {/if}
  </div>
  {#if optimisticStatus === "failed" && optimisticId}
    <div class="optimistic-actions">
      <button
        class="retry-btn"
        onclick={() => onRetry?.(optimisticId)}
        title="Retry sending"
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="23 4 23 10 17 10"></polyline>
          <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
        </svg>
        Retry
      </button>
      <button
        class="dismiss-btn"
        onclick={() => onDismiss?.(optimisticId)}
        title="Dismiss"
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="18" y1="6" x2="6" y2="18"></line>
          <line x1="6" y1="6" x2="18" y2="18"></line>
        </svg>
      </button>
    </div>
  {/if}
  {#if message.reactions && message.reactions.length > 0}
    <div class="reactions">
      {#each message.reactions as reaction}
        <span class="reaction" title={reaction.sender_name || reaction.sender}>
          {reaction.type}
        </span>
      {/each}
    </div>
  {/if}
</div>

<style>
  .message {
    display: flex;
    flex-direction: column;
    max-width: 70%;
    animation: none;
  }

  .message.new-message {
    animation: fadeIn 0.15s ease-out, highlight 1s ease-out;
  }

  @keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
  }

  @keyframes highlight {
    0% { background: rgba(0, 122, 255, 0.2); }
    100% { background: transparent; }
  }

  .message.from-me {
    align-self: flex-end;
  }

  .message.highlighted {
    animation: highlightPulse 3s ease-out;
  }

  .message.keyboard-focused {
    outline: 2px solid var(--color-primary, #007aff);
    outline-offset: 2px;
    border-radius: var(--radius-lg, 12px);
  }

  @keyframes highlightPulse {
    0% {
      background: rgba(251, 191, 36, 0.4);
      border-radius: 12px;
    }
    50% {
      background: rgba(251, 191, 36, 0.2);
    }
    100% {
      background: transparent;
    }
  }

  .bubble {
    padding: var(--space-3) var(--space-4);
    border-radius: var(--radius-xl);
    border-bottom-left-radius: var(--radius-sm);
    background: var(--bg-bubble-other);
    box-shadow: var(--shadow-sm);
  }

  .bubble.from-me {
    background: linear-gradient(135deg, var(--color-primary) 0%, #0056b3 100%);
    border-bottom-left-radius: var(--radius-xl);
    border-bottom-right-radius: var(--radius-sm);
  }

  .bubble .sender {
    font-size: 12px;
    font-weight: 500;
    color: var(--accent-color);
    display: block;
    margin-bottom: 4px;
  }

  .bubble p {
    font-size: 15px;
    line-height: 1.4;
    word-wrap: break-word;
    word-break: break-word;
    overflow-wrap: break-word;
    white-space: pre-wrap;
  }

  .bubble .time {
    font-size: 11px;
    color: var(--text-secondary);
    opacity: 0.7;
    display: block;
    text-align: right;
    margin-top: 4px;
  }

  .from-me .bubble .time {
    color: rgba(255, 255, 255, 0.7);
  }

  .attachments {
    margin-top: 8px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .attachment {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 13px;
    color: var(--text-secondary);
  }

  .attachment svg {
    width: 14px;
    height: 14px;
  }

  .reactions {
    display: flex;
    gap: 4px;
    margin-top: 4px;
  }

  .reaction {
    font-size: 14px;
    font-family: "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji", sans-serif;
    background: var(--bg-secondary);
    padding: 2px 6px;
    border-radius: 10px;
    cursor: default;
  }

  /* Optimistic message styles */
  .message.optimistic {
    opacity: 0.9;
  }

  .message.optimistic-sending .bubble {
    background: rgba(11, 147, 246, 0.7);
  }

  .message.optimistic-failed .bubble {
    background: rgba(255, 59, 48, 0.2);
    border: 1px solid rgba(255, 59, 48, 0.4);
  }

  .optimistic-status {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    margin-top: 4px;
  }

  .optimistic-status.sending {
    color: rgba(255, 255, 255, 0.7);
  }

  .optimistic-status.failed {
    color: #ff3b30;
  }

  .sending-dot {
    width: 6px;
    height: 6px;
    background: currentColor;
    border-radius: 50%;
    animation: pulse 1s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 0.4; }
    50% { opacity: 1; }
  }

  .optimistic-actions {
    display: flex;
    gap: 8px;
    margin-top: 6px;
    justify-content: flex-end;
  }

  .retry-btn,
  .dismiss-btn {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .retry-btn {
    background: var(--accent-color);
    border: none;
    color: white;
  }

  .retry-btn:hover {
    background: #0a82e0;
  }

  .retry-btn svg {
    width: 12px;
    height: 12px;
  }

  .dismiss-btn {
    background: transparent;
    border: 1px solid var(--border-color);
    color: var(--text-secondary);
    padding: 4px 8px;
  }

  .dismiss-btn:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }

  .dismiss-btn svg {
    width: 14px;
    height: 14px;
  }
</style>
