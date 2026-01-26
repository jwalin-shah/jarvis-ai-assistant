<script lang="ts">
  import { onMount } from "svelte";
  import {
    conversations,
    selectedConversation,
    loadingConversations,
    fetchConversations,
    selectConversation,
  } from "../stores/conversations";
  import LoadingSpinner from "./LoadingSpinner.svelte";

  let searchQuery = "";

  onMount(() => {
    fetchConversations();
  });

  $: filteredConversations = searchQuery
    ? $conversations.filter((c) => {
        const name = c.display_name || c.participants.join(", ");
        return name.toLowerCase().includes(searchQuery.toLowerCase());
      })
    : $conversations;

  function formatDate(dateString: string): string {
    const date = new Date(dateString);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));

    if (days === 0) {
      return date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
    } else if (days === 1) {
      return "Yesterday";
    } else if (days < 7) {
      return date.toLocaleDateString([], { weekday: "short" });
    } else {
      return date.toLocaleDateString([], { month: "short", day: "numeric" });
    }
  }

  function getDisplayName(conv: typeof $conversations[0]): string {
    return conv.display_name || conv.participants.join(", ");
  }
</script>

<div class="conversation-list">
  <div class="header">
    <h2>Messages</h2>
    <input
      type="search"
      placeholder="Search conversations..."
      bind:value={searchQuery}
      class="search-input"
    />
  </div>

  <div class="list-container">
    {#if $loadingConversations}
      <div class="loading">
        <LoadingSpinner size="medium" />
        <span>Loading conversations...</span>
      </div>
    {:else if filteredConversations.length === 0}
      <div class="empty">
        {#if searchQuery}
          <p>No conversations match "{searchQuery}"</p>
        {:else}
          <p>No conversations found</p>
        {/if}
      </div>
    {:else}
      {#each filteredConversations as conversation (conversation.chat_id)}
        <button
          class="conversation-item"
          class:selected={$selectedConversation === conversation.chat_id}
          class:is-group={conversation.is_group}
          on:click={() => selectConversation(conversation.chat_id)}
        >
          <div class="avatar" class:group={conversation.is_group}>
            {#if conversation.is_group}
              <span class="avatar-icon">👥</span>
            {:else}
              <span class="avatar-letter">
                {getDisplayName(conversation).charAt(0).toUpperCase()}
              </span>
            {/if}
          </div>

          <div class="content">
            <div class="top-row">
              <span class="name">{getDisplayName(conversation)}</span>
              <span class="date">{formatDate(conversation.last_message_date)}</span>
            </div>
            <div class="preview">
              {conversation.last_message_text || "No messages"}
            </div>
          </div>
        </button>
      {/each}
    {/if}
  </div>
</div>

<style>
  .conversation-list {
    width: 300px;
    min-width: 300px;
    background: var(--bg-primary);
    display: flex;
    flex-direction: column;
    border-right: 1px solid var(--border-color);
  }

  .header {
    padding: 16px;
    border-bottom: 1px solid var(--border-color);
  }

  h2 {
    font-size: 20px;
    font-weight: 600;
    margin: 0 0 12px 0;
    color: var(--text-primary);
  }

  .search-input {
    width: 100%;
    padding: 8px 12px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    color: var(--text-primary);
    font-size: 14px;
  }

  .search-input::placeholder {
    color: var(--text-secondary);
  }

  .search-input:focus {
    outline: none;
    border-color: var(--accent-color);
  }

  .list-container {
    flex: 1;
    overflow-y: auto;
  }

  .loading,
  .empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    padding: 32px;
    color: var(--text-secondary);
  }

  .conversation-item {
    display: flex;
    align-items: center;
    gap: 12px;
    width: 100%;
    padding: 12px 16px;
    background: transparent;
    border: none;
    cursor: pointer;
    text-align: left;
    transition: background 0.15s ease;
  }

  .conversation-item:hover {
    background: var(--bg-hover);
  }

  .conversation-item.selected {
    background: var(--bg-active);
  }

  .avatar {
    width: 44px;
    height: 44px;
    border-radius: 50%;
    background: var(--bg-secondary);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
  }

  .avatar.group {
    background: var(--group-color);
  }

  .avatar-letter {
    font-size: 18px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .avatar-icon {
    font-size: 20px;
  }

  .content {
    flex: 1;
    min-width: 0;
  }

  .top-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 8px;
  }

  .name {
    font-size: 15px;
    font-weight: 500;
    color: var(--text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .date {
    font-size: 12px;
    color: var(--text-secondary);
    flex-shrink: 0;
  }

  .preview {
    font-size: 13px;
    color: var(--text-secondary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-top: 2px;
  }
</style>
