<script lang="ts">
  import type { Message } from '../../types';
  import { isOptimisticMessage, getOptimisticStatus, getOptimisticId } from '../../types';
  import { formatRelativeTime, formatFullTimestamp } from '../../utils/date';
  import LinkPreview from './LinkPreview.svelte';

  interface Props {
    message: Message;
    isGroup: boolean;
    isHighlighted: boolean;
    isKeyboardFocused: boolean;
    isNew: boolean;
    onRetry?: (optimisticId: string) => void;
    onDismiss?: (optimisticId: string) => void;
    onHeightChange?: (messageId: number, height: number) => void;
  }

  let {
    message,
    isGroup,
    isHighlighted,
    isKeyboardFocused,
    isNew,
    onRetry,
    onDismiss,
    onHeightChange,
  }: Props = $props();

  // ResizeObserver for efficient height tracking
  let messageElement: HTMLElement | null = $state(null);
  let resizeObserver: ResizeObserver | null = null;

  $effect(() => {
    if (!messageElement || !onHeightChange) return;

    // Initial measurement
    const measure = () => {
      const style = window.getComputedStyle(messageElement!);
      const marginTop = parseFloat(style.marginTop) || 0;
      const marginBottom = parseFloat(style.marginBottom) || 0;
      const height = messageElement!.offsetHeight + marginTop + marginBottom;
      onHeightChange!(message.id, height);
    };

    // Use ResizeObserver for efficient height tracking
    resizeObserver = new ResizeObserver(() => {
      measure();
    });
    resizeObserver.observe(messageElement);

    // Initial measurement
    measure();

    return () => {
      resizeObserver?.disconnect();
      resizeObserver = null;
    };
  });

  // Derived values using strict type guards
  const optimistic = $derived(isOptimisticMessage(message));
  const optimisticStatus = $derived(getOptimisticStatus(message));
  const optimisticId = $derived(getOptimisticId(message));

  // Format time consistently
  function formatTime(dateStr: string): string {
    return new Date(dateStr).toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit',
    });
  }

  // Detect URLs in message text
  function extractUrls(text: string): string[] {
    if (!text) return [];
    const urlRegex = /https?:\/\/[^\s<>"{}|\\^`[\]]+/g;
    return text.match(urlRegex) || [];
  }

  // Group reactions by type with counts
  interface GroupedReaction {
    type: string;
    count: number;
    senders: string[];
  }

  function groupReactions(reactions: typeof message.reactions): GroupedReaction[] {
    const groups = new Map<string, GroupedReaction>();
    for (const r of reactions) {
      const existing = groups.get(r.type);
      if (existing) {
        existing.count++;
        existing.senders.push(r.sender_name || r.sender);
      } else {
        groups.set(r.type, {
          type: r.type,
          count: 1,
          senders: [r.sender_name || r.sender],
        });
      }
    }
    return Array.from(groups.values());
  }

  let urls = $derived(extractUrls(message.text));
  let groupedReactions = $derived(groupReactions(message.reactions));

  function handleRetry() {
    if (optimisticId && onRetry) {
      onRetry(optimisticId);
    }
  }

  function handleDismiss() {
    if (optimisticId && onDismiss) {
      onDismiss(optimisticId);
    }
  }
</script>

{#if message.is_system_message}
  <div class="system-message" data-message-id={message.id} bind:this={messageElement}>
    {message.text}
  </div>
{:else}
  <div
    class="message"
    bind:this={messageElement}
    class:from-me={message.is_from_me}
    class:new-message={isNew}
    class:highlighted={isHighlighted}
    class:keyboard-focused={isKeyboardFocused}
    class:optimistic
    class:optimistic-sending={optimisticStatus === 'sending'}
    class:optimistic-failed={optimisticStatus === 'failed'}
    data-message-id={message.id}
    tabindex="-1"
    role="article"
    aria-label={`Message from ${message.is_from_me ? 'you' : message.sender_name || message.sender}`}
  >
    <div class="bubble" class:from-me={message.is_from_me}>
      {#if !message.is_from_me && isGroup}
        <span class="sender">{message.sender_name || message.sender}</span>
      {/if}
      <p>{message.text}</p>
      {#if message.attachments.length > 0}
        <div class="attachments">
          {#each message.attachments as attachment}
            <div class="attachment">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path
                  d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"
                />
              </svg>
              <span>{attachment.filename}</span>
            </div>
          {/each}
        </div>
      {/if}
      {#if urls.length > 0}
        <div class="link-previews">
          {#each urls.slice(0, 3) as url}
            <LinkPreview {url} />
          {/each}
        </div>
      {/if}
      {#if optimisticStatus === 'sending'}
        <span class="optimistic-status sending">
          <span class="sending-dot"></span>
          Sending...
        </span>
      {:else if optimisticStatus === 'failed'}
        <span class="optimistic-status failed">Failed to send</span>
      {:else}
        <span class="time" title={formatFullTimestamp(message.date)}>{formatRelativeTime(message.date)}</span>
      {/if}
    </div>
    {#if optimisticStatus === 'failed' && optimisticId}
      <div class="optimistic-actions">
        <button class="retry-btn" onclick={handleRetry} title="Retry sending">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="23 4 23 10 17 10"></polyline>
            <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
          </svg>
          Retry
        </button>
        <button class="dismiss-btn" onclick={handleDismiss} title="Dismiss">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="18" y1="6" x2="6" y2="18"></line>
            <line x1="6" y1="6" x2="18" y2="18"></line>
          </svg>
        </button>
      </div>
    {/if}
    {#if message.reactions.length > 0}
      <div class="reactions">
        {#each groupedReactions as reaction}
          <span class="reaction" title={reaction.senders.join(', ')}>
            {reaction.type}{#if reaction.count > 1} <span class="reaction-count">{reaction.count}</span>{/if}
          </span>
        {/each}
      </div>
    {/if}
  </div>
{/if}

<style>
  .message {
    display: flex;
    flex-direction: column;
    max-width: 70%;
    animation: none;
  }

  .message.new-message {
    animation: fadeIn var(--duration-fast) var(--ease-out),
      highlight var(--duration-slow) var(--ease-out);
  }

  @keyframes fadeIn {
    from {
      opacity: 0;
    }
    to {
      opacity: 1;
    }
  }

  @keyframes highlight {
    0% {
      background: rgba(0, 122, 255, 0.2);
    }
    100% {
      background: transparent;
    }
  }

  .message.from-me {
    align-self: flex-end;
  }

  .message.highlighted {
    animation: highlightPulse 3s var(--ease-out);
  }

  .message.keyboard-focused {
    outline: 2px solid var(--color-primary);
    outline-offset: 2px;
    border-radius: var(--radius-lg);
  }

  @keyframes highlightPulse {
    0% {
      background: rgba(251, 191, 36, 0.4);
      border-radius: var(--radius-lg);
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
    background: var(--bubble-other);
    box-shadow: var(--shadow-sm);
  }

  .bubble.from-me {
    background: var(--bubble-me-gradient);
    border-bottom-left-radius: var(--radius-xl);
    border-bottom-right-radius: var(--radius-sm);
  }

  .bubble .sender {
    font-size: var(--text-sm);
    font-weight: var(--font-weight-medium);
    color: var(--color-primary);
    display: block;
    margin-bottom: var(--space-1);
  }

  .bubble p {
    font-size: var(--text-base);
    line-height: var(--line-height-normal);
    word-wrap: break-word;
    word-break: break-word;
    overflow-wrap: break-word;
    white-space: pre-wrap;
    margin: 0;
  }

  .bubble .time {
    font-size: var(--text-xs);
    color: var(--text-secondary);
    opacity: 0.5;
    display: block;
    text-align: right;
    margin-top: var(--space-1);
    cursor: default;
    transition: opacity var(--duration-fast) var(--ease-out);
  }

  .bubble:hover .time {
    opacity: 1;
  }

  .from-me .bubble .time {
    color: rgba(255, 255, 255, 0.7);
  }

  .attachments {
    margin-top: var(--space-2);
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
  }

  .attachment {
    display: flex;
    align-items: center;
    gap: var(--space-1);
    font-size: var(--text-sm);
    color: var(--text-secondary);
  }

  .attachment svg {
    width: 14px;
    height: 14px;
    flex-shrink: 0;
  }

  .reactions {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-1);
    margin-top: calc(var(--space-1) * -1);
    position: relative;
    top: -4px;
  }

  .reaction {
    display: inline-flex;
    align-items: center;
    gap: 2px;
    font-size: 13px;
    font-family: 'Apple Color Emoji', 'Segoe UI Emoji', 'Noto Color Emoji', sans-serif;
    background: var(--surface-elevated);
    padding: 2px 8px;
    border-radius: var(--radius-full);
    cursor: default;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border-subtle);
  }

  .reaction-count {
    font-family: var(--font-family-sans);
    font-size: var(--text-xs);
    font-weight: var(--font-weight-medium);
    color: var(--text-secondary);
  }

  .link-previews {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
    margin-top: var(--space-2);
  }

  .system-message {
    text-align: center;
    font-size: var(--text-sm);
    color: var(--text-secondary);
    font-style: italic;
    padding: var(--space-2) 0;
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
    gap: var(--space-1);
    font-size: var(--text-xs);
    margin-top: var(--space-1);
  }

  .optimistic-status.sending {
    color: rgba(255, 255, 255, 0.7);
  }

  .optimistic-status.failed {
    color: var(--color-error);
  }

  .sending-dot {
    width: 6px;
    height: 6px;
    background: currentColor;
    border-radius: 50%;
    animation: pulse 1s var(--ease-in-out) infinite;
  }

  @keyframes pulse {
    0%,
    100% {
      opacity: 0.4;
    }
    50% {
      opacity: 1;
    }
  }

  .optimistic-actions {
    display: flex;
    gap: var(--space-2);
    margin-top: var(--space-1);
    justify-content: flex-end;
  }

  .retry-btn,
  .dismiss-btn {
    display: flex;
    align-items: center;
    gap: var(--space-1);
    padding: 4px 10px;
    border-radius: var(--radius-sm);
    font-size: var(--text-xs);
    cursor: pointer;
    transition: all var(--duration-fast) var(--ease-out);
    border: none;
  }

  .retry-btn {
    background: var(--color-primary);
    color: white;
  }

  .retry-btn:hover {
    background: var(--color-primary-hover);
  }

  .retry-btn svg {
    width: 12px;
    height: 12px;
  }

  .dismiss-btn {
    background: transparent;
    border: 1px solid var(--border-default);
    color: var(--text-secondary);
    padding: 4px 8px;
  }

  .dismiss-btn:hover {
    background: var(--surface-hover);
    color: var(--text-primary);
  }

  .dismiss-btn svg {
    width: 14px;
    height: 14px;
  }

  /* Reduced motion */
  :global(:root.reduce-motion) .message.new-message,
  :global(:root.reduce-motion) .message.highlighted {
    animation: none;
  }

  :global(:root.reduce-motion) .sending-dot {
    animation: none;
    opacity: 0.5;
  }
</style>
