<script lang="ts">
  import type { Message } from "../api/types";

  export let message: Message;

  function formatTime(dateStr: string): string {
    return new Date(dateStr).toLocaleTimeString([], {
      hour: "numeric",
      minute: "2-digit",
    });
  }

  function formatDate(dateStr: string): string {
    return new Date(dateStr).toLocaleDateString([], {
      weekday: "short",
      month: "short",
      day: "numeric",
    });
  }
</script>

<div class="message" class:from-me={message.is_from_me}>
  <div class="bubble">
    <div class="text">{message.text}</div>

    {#if message.attachments.length > 0}
      <div class="attachments">
        {#each message.attachments as attachment}
          <div class="attachment">
            <span class="attachment-icon">&#128206;</span>
            <span class="attachment-name">{attachment.filename}</span>
          </div>
        {/each}
      </div>
    {/if}

    {#if message.reactions.length > 0}
      <div class="reactions">
        {#each message.reactions as reaction}
          <span class="reaction" title={reaction.sender_name || reaction.sender}>
            {#if reaction.type === "love"}
              &#10084;
            {:else if reaction.type === "like"}
              &#128077;
            {:else if reaction.type === "dislike"}
              &#128078;
            {:else if reaction.type === "laugh"}
              &#128514;
            {:else if reaction.type === "emphasize"}
              &#10071;
            {:else if reaction.type === "question"}
              &#10067;
            {:else}
              {reaction.type}
            {/if}
          </span>
        {/each}
      </div>
    {/if}
  </div>

  <div class="meta">
    {#if !message.is_from_me}
      <span class="sender">{message.sender_name || message.sender}</span>
    {/if}
    <span class="time" title={formatDate(message.date)}>{formatTime(message.date)}</span>
    {#if message.is_from_me}
      {#if message.date_read}
        <span class="status read" title="Read {formatTime(message.date_read)}">Read</span>
      {:else if message.date_delivered}
        <span class="status delivered" title="Delivered {formatTime(message.date_delivered)}">Delivered</span>
      {/if}
    {/if}
  </div>
</div>

<style>
  .message {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    max-width: 70%;
    margin-bottom: 8px;
  }

  .message.from-me {
    align-items: flex-end;
    align-self: flex-end;
  }

  .bubble {
    padding: 10px 14px;
    border-radius: 18px;
    background: var(--bg-bubble-other);
    color: var(--text-primary);
    word-wrap: break-word;
    position: relative;
  }

  .message.from-me .bubble {
    background: var(--bg-bubble-me);
    color: white;
  }

  .text {
    font-size: 14px;
    line-height: 1.4;
    white-space: pre-wrap;
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
    font-size: 12px;
    opacity: 0.8;
  }

  .attachment-icon {
    font-size: 14px;
  }

  .attachment-name {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .reactions {
    position: absolute;
    bottom: -10px;
    right: 10px;
    display: flex;
    gap: 2px;
    background: var(--bg-secondary);
    padding: 2px 6px;
    border-radius: 10px;
    font-size: 12px;
  }

  .message.from-me .reactions {
    left: 10px;
    right: auto;
  }

  .reaction {
    cursor: default;
  }

  .meta {
    display: flex;
    gap: 8px;
    margin-top: 4px;
    padding: 0 4px;
    font-size: 11px;
    color: var(--text-secondary);
  }

  .sender {
    font-weight: 500;
  }

  .status {
    font-size: 10px;
    font-style: italic;
  }

  .status.delivered {
    color: var(--text-secondary);
  }

  .status.read {
    color: #3b82f6;
  }
</style>
