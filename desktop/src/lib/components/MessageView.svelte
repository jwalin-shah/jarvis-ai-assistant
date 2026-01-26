<script lang="ts">
  import { onMount } from "svelte";
  import {
    messages,
    selectedConversation,
    selectedConversationDetails,
    loadingMessages,
  } from "../stores/conversations";
  import { modelStatus, preloadModel } from "../stores/health";
  import LoadingSpinner from "./LoadingSpinner.svelte";

  let messageContainer: HTMLDivElement;
  let isGenerating = false;
  let draftReply = "";

  // Auto-scroll to bottom when messages change
  $: if ($messages && messageContainer) {
    setTimeout(() => {
      messageContainer.scrollTop = messageContainer.scrollHeight;
    }, 0);
  }

  function formatTime(dateString: string): string {
    const date = new Date(dateString);
    return date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  }

  function formatDateHeader(dateString: string): string {
    const date = new Date(dateString);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));

    if (days === 0) {
      return "Today";
    } else if (days === 1) {
      return "Yesterday";
    } else {
      return date.toLocaleDateString([], {
        weekday: "long",
        month: "long",
        day: "numeric",
      });
    }
  }

  // Group messages by date
  function groupMessagesByDate(msgs: typeof $messages) {
    const groups: { date: string; messages: typeof $messages }[] = [];
    let currentDate = "";

    for (const msg of msgs) {
      const msgDate = new Date(msg.date).toDateString();
      if (msgDate !== currentDate) {
        currentDate = msgDate;
        groups.push({ date: msg.date, messages: [] });
      }
      groups[groups.length - 1].messages.push(msg);
    }

    return groups;
  }

  $: messageGroups = groupMessagesByDate($messages);

  async function handleGenerateDraft() {
    if (!$selectedConversation || isGenerating) return;

    // Check if model is loaded
    if ($modelStatus.state !== "loaded") {
      // Trigger model loading
      await preloadModel();
      return;
    }

    isGenerating = true;
    try {
      // Get suggestions from API
      const lastMessage = $messages.length > 0 ? $messages[$messages.length - 1] : null;
      if (!lastMessage || lastMessage.is_from_me) {
        draftReply = "No recent message to reply to";
        return;
      }

      const response = await fetch("http://localhost:8742/suggestions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          last_message: lastMessage.text,
          num_suggestions: 1,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        if (data.suggestions && data.suggestions.length > 0) {
          draftReply = data.suggestions[0].text;
        }
      }
    } finally {
      isGenerating = false;
    }
  }
</script>

<div class="message-view">
  {#if !$selectedConversation}
    <div class="empty-state">
      <div class="empty-icon">💬</div>
      <h3>Select a conversation</h3>
      <p>Choose a conversation from the list to view messages</p>
    </div>
  {:else}
    <div class="header">
      {#if $selectedConversationDetails}
        <div class="contact-info">
          <div class="avatar" class:group={$selectedConversationDetails.is_group}>
            {#if $selectedConversationDetails.is_group}
              <span>👥</span>
            {:else}
              <span>
                {($selectedConversationDetails.display_name ||
                  $selectedConversationDetails.participants[0] ||
                  "?")
                  .charAt(0)
                  .toUpperCase()}
              </span>
            {/if}
          </div>
          <div class="details">
            <h3>
              {$selectedConversationDetails.display_name ||
                $selectedConversationDetails.participants.join(", ")}
            </h3>
            {#if $selectedConversationDetails.is_group}
              <span class="participant-count">
                {$selectedConversationDetails.participants.length} participants
              </span>
            {/if}
          </div>
        </div>
      {/if}
    </div>

    <div class="messages-container" bind:this={messageContainer}>
      {#if $loadingMessages}
        <div class="loading">
          <LoadingSpinner size="medium" />
          <span>Loading messages...</span>
        </div>
      {:else if $messages.length === 0}
        <div class="empty-state">
          <p>No messages in this conversation</p>
        </div>
      {:else}
        {#each messageGroups as group}
          <div class="date-header">
            <span>{formatDateHeader(group.date)}</span>
          </div>
          {#each group.messages as message (message.id)}
            <div
              class="message"
              class:from-me={message.is_from_me}
              class:system={message.is_system_message}
            >
              {#if message.is_system_message}
                <div class="system-message">{message.text}</div>
              {:else}
                <div class="bubble">
                  {#if !message.is_from_me && $selectedConversationDetails?.is_group}
                    <div class="sender-name">
                      {message.sender_name || message.sender}
                    </div>
                  {/if}
                  <div class="text">{message.text}</div>
                  {#if message.attachments.length > 0}
                    <div class="attachments">
                      {#each message.attachments as attachment}
                        <div class="attachment">
                          📎 {attachment.filename}
                        </div>
                      {/each}
                    </div>
                  {/if}
                  <div class="time">{formatTime(message.date)}</div>
                </div>
                {#if message.reactions.length > 0}
                  <div class="reactions">
                    {#each message.reactions as reaction}
                      <span class="reaction" title={reaction.sender_name || reaction.sender}>
                        {reaction.type}
                      </span>
                    {/each}
                  </div>
                {/if}
              {/if}
            </div>
          {/each}
        {/each}
      {/if}
    </div>

    <div class="composer">
      {#if isGenerating}
        <div class="generating-indicator">
          <LoadingSpinner size="small" />
          <span>Generating reply...</span>
        </div>
      {/if}

      <div class="input-row">
        <button
          class="ai-button"
          on:click={handleGenerateDraft}
          disabled={isGenerating || $modelStatus.state === "loading"}
          title={$modelStatus.state !== "loaded" ? "Load model first" : "Generate AI draft"}
        >
          {#if $modelStatus.state === "loading"}
            <LoadingSpinner size="small" />
          {:else}
            ✨
          {/if}
        </button>
        <input
          type="text"
          placeholder={draftReply || "Type a message..."}
          class="message-input"
          bind:value={draftReply}
        />
        <button class="send-button" disabled={!draftReply}>
          Send
        </button>
      </div>
    </div>
  {/if}
</div>

<style>
  .message-view {
    flex: 1;
    display: flex;
    flex-direction: column;
    background: var(--bg-primary);
  }

  .empty-state {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: var(--text-secondary);
    gap: 8px;
  }

  .empty-icon {
    font-size: 48px;
    margin-bottom: 8px;
  }

  .empty-state h3 {
    font-size: 18px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0;
  }

  .empty-state p {
    margin: 0;
  }

  .header {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border-color);
    background: var(--bg-secondary);
  }

  .contact-info {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background: var(--bg-active);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    font-weight: 600;
  }

  .avatar.group {
    background: var(--group-color);
  }

  .details h3 {
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0;
  }

  .participant-count {
    font-size: 12px;
    color: var(--text-secondary);
  }

  .messages-container {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .loading {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    color: var(--text-secondary);
  }

  .date-header {
    display: flex;
    justify-content: center;
    margin: 16px 0 8px;
  }

  .date-header span {
    font-size: 12px;
    color: var(--text-secondary);
    background: var(--bg-secondary);
    padding: 4px 12px;
    border-radius: 10px;
  }

  .message {
    display: flex;
    flex-direction: column;
    max-width: 70%;
    margin-bottom: 2px;
  }

  .message.from-me {
    align-self: flex-end;
    align-items: flex-end;
  }

  .message.system {
    align-self: center;
    max-width: 100%;
  }

  .system-message {
    font-size: 12px;
    color: var(--text-secondary);
    font-style: italic;
    padding: 4px 12px;
  }

  .bubble {
    background: var(--bg-bubble-other);
    padding: 8px 12px;
    border-radius: 18px;
    position: relative;
  }

  .message.from-me .bubble {
    background: var(--bg-bubble-me);
  }

  .sender-name {
    font-size: 11px;
    color: var(--accent-color);
    font-weight: 600;
    margin-bottom: 2px;
  }

  .text {
    font-size: 15px;
    color: var(--text-primary);
    line-height: 1.4;
    white-space: pre-wrap;
    word-wrap: break-word;
  }

  .attachments {
    margin-top: 6px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .attachment {
    font-size: 12px;
    color: var(--text-secondary);
    background: rgba(0, 0, 0, 0.2);
    padding: 4px 8px;
    border-radius: 6px;
  }

  .time {
    font-size: 10px;
    color: rgba(255, 255, 255, 0.6);
    margin-top: 4px;
    text-align: right;
  }

  .reactions {
    display: flex;
    gap: 4px;
    margin-top: 4px;
  }

  .reaction {
    font-size: 14px;
    background: var(--bg-secondary);
    padding: 2px 6px;
    border-radius: 10px;
    cursor: help;
  }

  .composer {
    padding: 12px 16px;
    border-top: 1px solid var(--border-color);
    background: var(--bg-secondary);
  }

  .generating-indicator {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    margin-bottom: 8px;
    background: var(--bg-primary);
    border-radius: 8px;
    font-size: 13px;
    color: var(--text-secondary);
  }

  .input-row {
    display: flex;
    gap: 8px;
    align-items: center;
  }

  .ai-button {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: var(--bg-active);
    border: none;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    transition: background 0.15s ease;
  }

  .ai-button:hover:not(:disabled) {
    background: var(--accent-color);
  }

  .ai-button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .message-input {
    flex: 1;
    padding: 10px 14px;
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 20px;
    color: var(--text-primary);
    font-size: 14px;
  }

  .message-input::placeholder {
    color: var(--text-secondary);
  }

  .message-input:focus {
    outline: none;
    border-color: var(--accent-color);
  }

  .send-button {
    padding: 10px 16px;
    background: var(--accent-color);
    border: none;
    border-radius: 20px;
    color: white;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: opacity 0.15s ease;
  }

  .send-button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .send-button:hover:not(:disabled) {
    opacity: 0.9;
  }
</style>
