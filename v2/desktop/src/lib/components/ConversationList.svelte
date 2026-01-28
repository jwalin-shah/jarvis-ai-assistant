<script lang="ts">
  import type { Conversation } from "../api/types";

  export let conversations: Conversation[] = [];
  export let selectedChatId: string | null = null;
  export let onSelect: (chatId: string) => void = () => {};

  function getInitials(conv: Conversation): string {
    const name = conv.display_name || conv.participants[0] || "?";
    if (name.startsWith("+")) {
      return name.slice(-2);
    }
    const parts = name.split(" ");
    if (parts.length >= 2) {
      return `${parts[0][0]}${parts[1][0]}`.toUpperCase();
    }
    return name[0]?.toUpperCase() || "?";
  }

  function getDisplayName(conv: Conversation): string {
    return conv.display_name || conv.participants[0] || "Unknown";
  }

  function formatTime(dateStr: string | null): string {
    if (!dateStr) return "";
    const date = new Date(dateStr);
    const now = new Date();
    const diffDays = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60 * 24));

    if (diffDays === 0) {
      return date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
    } else if (diffDays < 7) {
      return date.toLocaleDateString([], { weekday: "short" });
    } else {
      return date.toLocaleDateString([], { month: "short", day: "numeric" });
    }
  }

  function truncate(text: string | null, maxLen: number = 40): string {
    if (!text) return "";
    return text.length > maxLen ? text.slice(0, maxLen) + "..." : text;
  }
</script>

<div class="conversation-list">
  {#each conversations as conv (conv.chat_id)}
    <button
      class="conversation-item"
      class:selected={conv.chat_id === selectedChatId}
      on:click={() => onSelect(conv.chat_id)}
    >
      <div class="avatar">
        {getInitials(conv)}
      </div>
      <div class="conversation-info">
        <div class="conversation-name">{getDisplayName(conv)}</div>
        <div class="conversation-preview">{truncate(conv.last_message_text)}</div>
      </div>
      <div class="conversation-time">{formatTime(conv.last_message_date)}</div>
    </button>
  {/each}
</div>

<style>
  .conversation-list {
    flex: 1;
    overflow-y: auto;
  }

  .conversation-item {
    display: flex;
    align-items: center;
    padding: 12px 16px;
    cursor: pointer;
    border: none;
    border-bottom: 1px solid var(--border-color);
    background: transparent;
    width: 100%;
    text-align: left;
    transition: background 0.15s;
  }

  .conversation-item:hover {
    background: var(--hover-color);
  }

  .conversation-item.selected {
    background: var(--bg-tertiary);
  }

  .avatar {
    width: 44px;
    height: 44px;
    border-radius: 50%;
    background: var(--accent-blue);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    font-size: 16px;
    margin-right: 12px;
    flex-shrink: 0;
    color: white;
  }

  .conversation-info {
    flex: 1;
    min-width: 0;
  }

  .conversation-name {
    font-weight: 600;
    margin-bottom: 2px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    color: var(--text-primary);
  }

  .conversation-preview {
    color: var(--text-secondary);
    font-size: 13px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .conversation-time {
    color: var(--text-secondary);
    font-size: 12px;
    margin-left: 8px;
    flex-shrink: 0;
  }
</style>
