<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { apiClient, APIError } from "../api/client";
  import type {
    PriorityInboxResponse,
    PriorityMessage,
    PriorityLevel,
  } from "../api/types";

  interface Props {
    onSelectConversation: (chatId: string) => void;
    onClose: () => void;
  }

  let { onSelectConversation, onClose }: Props = $props();

  // State
  type PanelState = "idle" | "loading" | "results" | "error";
  let panelState: PanelState = $state("idle");
  let messages: PriorityMessage[] = $state([]);
  let totalCount = $state(0);
  let unhandledCount = $state(0);
  let needsResponseCount = $state(0);
  let criticalCount = $state(0);
  let highCount = $state(0);
  let errorMessage = $state("");
  let includeHandled = $state(false);
  let minLevel: PriorityLevel | undefined = $state(undefined);

  // AbortController for cancelling in-flight requests
  let abortController: AbortController | null = null;

  // Auto-refresh interval
  let refreshInterval: ReturnType<typeof setInterval> | null = null;

  onMount(() => {
    loadPriorityInbox();
    // Auto-refresh every 60 seconds
    refreshInterval = setInterval(loadPriorityInbox, 60000);
  });

  onDestroy(() => {
    abortController?.abort();
    if (refreshInterval) {
      clearInterval(refreshInterval);
    }
  });

  async function loadPriorityInbox() {
    abortController?.abort();
    abortController = new AbortController();

    panelState = "loading";
    errorMessage = "";

    try {
      const response = await apiClient.getPriorityInbox(
        50,
        includeHandled,
        minLevel,
        abortController.signal
      );
      messages = response.messages;
      totalCount = response.total_count;
      unhandledCount = response.unhandled_count;
      needsResponseCount = response.needs_response_count;
      criticalCount = response.critical_count;
      highCount = response.high_count;
      panelState = "results";
    } catch (e) {
      if (e instanceof Error && e.name === "AbortError") {
        return;
      }
      panelState = "error";
      if (e instanceof APIError) {
        errorMessage = e.detail || e.message;
      } else if (e instanceof Error) {
        errorMessage = e.message;
      } else {
        errorMessage = "An unknown error occurred";
      }
    }
  }

  async function handleMarkHandled(message: PriorityMessage) {
    try {
      await apiClient.markMessageHandled(message.chat_id, message.message_id);
      // Update local state
      messages = messages.map((m) =>
        m.message_id === message.message_id && m.chat_id === message.chat_id
          ? { ...m, handled: true }
          : m
      );
      // If not showing handled, remove from list
      if (!includeHandled) {
        messages = messages.filter(
          (m) =>
            !(m.message_id === message.message_id && m.chat_id === message.chat_id)
        );
        unhandledCount = Math.max(0, unhandledCount - 1);
      }
    } catch (e) {
      console.error("Failed to mark as handled:", e);
    }
  }

  async function handleUnmarkHandled(message: PriorityMessage) {
    try {
      await apiClient.unmarkMessageHandled(message.chat_id, message.message_id);
      // Update local state
      messages = messages.map((m) =>
        m.message_id === message.message_id && m.chat_id === message.chat_id
          ? { ...m, handled: false }
          : m
      );
      unhandledCount = unhandledCount + 1;
    } catch (e) {
      console.error("Failed to unmark as handled:", e);
    }
  }

  function handleViewConversation(chatId: string) {
    onSelectConversation(chatId);
    onClose();
  }

  function handleKeyDown(event: KeyboardEvent) {
    if (event.key === "Escape") {
      onClose();
    }
  }

  function formatDate(dateStr: string): string {
    const date = new Date(dateStr);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffHours = diffMs / (1000 * 60 * 60);

    if (diffHours < 1) {
      const diffMins = Math.floor(diffMs / (1000 * 60));
      return `${diffMins}m ago`;
    } else if (diffHours < 24) {
      return `${Math.floor(diffHours)}h ago`;
    } else {
      return date.toLocaleDateString();
    }
  }

  function getPriorityColor(level: PriorityLevel): string {
    switch (level) {
      case "critical":
        return "var(--error-color)";
      case "high":
        return "#f59e0b";
      case "medium":
        return "var(--accent-color)";
      case "low":
        return "var(--text-secondary)";
    }
  }

  function getReasonLabel(reason: string): string {
    const labels: Record<string, string> = {
      contains_question: "Question",
      action_requested: "Action",
      time_sensitive: "Urgent",
      important_contact: "VIP",
      frequent_contact: "Frequent",
      awaiting_response: "Waiting",
      multiple_messages: "Multiple",
      contains_urgency: "Urgent",
      normal: "Normal",
    };
    return labels[reason] || reason;
  }

  function handleFilterChange() {
    loadPriorityInbox();
  }
</script>

<svelte:window onkeydown={handleKeyDown} />

<!-- svelte-ignore a11y_click_events_have_key_events -->
<div class="panel-overlay" onclick={onClose} role="presentation">
  <!-- svelte-ignore a11y_interactive_supports_focus -->
  <div
    class="panel"
    onclick={(e) => e.stopPropagation()}
    role="dialog"
    aria-label="Priority Inbox"
  >
    <header class="panel-header">
      <div class="panel-title">
        <span class="priority-icon">!</span>
        <h2>Priority Inbox</h2>
      </div>
      <button class="close-btn" onclick={onClose} aria-label="Close">
        ×
      </button>
    </header>

    <div class="panel-stats">
      <div class="stat">
        <span class="stat-value critical">{criticalCount}</span>
        <span class="stat-label">Critical</span>
      </div>
      <div class="stat">
        <span class="stat-value high">{highCount}</span>
        <span class="stat-label">High</span>
      </div>
      <div class="stat">
        <span class="stat-value">{needsResponseCount}</span>
        <span class="stat-label">Need Reply</span>
      </div>
      <div class="stat">
        <span class="stat-value">{unhandledCount}</span>
        <span class="stat-label">Unhandled</span>
      </div>
    </div>

    <div class="panel-filters">
      <label class="filter-checkbox">
        <input
          type="checkbox"
          bind:checked={includeHandled}
          onchange={handleFilterChange}
        />
        Show handled
      </label>
      <select
        class="filter-select"
        bind:value={minLevel}
        onchange={handleFilterChange}
      >
        <option value={undefined}>All priorities</option>
        <option value="critical">Critical only</option>
        <option value="high">High and above</option>
        <option value="medium">Medium and above</option>
      </select>
      <button class="refresh-btn" onclick={loadPriorityInbox} aria-label="Refresh">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M23 4v6h-6"></path>
          <path d="M1 20v-6h6"></path>
          <path
            d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"
          ></path>
        </svg>
      </button>
    </div>

    <div class="panel-content">
      {#if panelState === "loading" && messages.length === 0}
        <div class="loading-section">
          <div class="loading-indicator">
            <span class="spinner large"></span>
            <p>Analyzing messages...</p>
          </div>
        </div>
      {:else if panelState === "error"}
        <div class="error-section">
          <div class="error-icon">!</div>
          <p class="error-message">{errorMessage}</p>
          <button class="retry-btn" onclick={loadPriorityInbox}>
            Try Again
          </button>
        </div>
      {:else if messages.length === 0}
        <div class="empty-section">
          <div class="empty-icon">!</div>
          <p>No priority messages</p>
          <span class="empty-subtitle">You're all caught up!</span>
        </div>
      {:else}
        <div class="message-list">
          {#each messages as message (message.message_id + message.chat_id)}
            <div
              class="message-item"
              class:handled={message.handled}
              class:critical={message.priority_level === "critical"}
              class:high={message.priority_level === "high"}
            >
              <div class="message-header">
                <div class="sender-info">
                  <span class="sender-name">
                    {message.sender_name || message.sender}
                  </span>
                  {#if message.conversation_name && message.conversation_name !== message.sender_name}
                    <span class="conversation-name">
                      in {message.conversation_name}
                    </span>
                  {/if}
                </div>
                <div class="message-meta">
                  <span
                    class="priority-badge"
                    style="background-color: {getPriorityColor(message.priority_level)}"
                  >
                    {message.priority_level}
                  </span>
                  <span class="message-time">{formatDate(message.date)}</span>
                </div>
              </div>

              <div class="message-text">{message.text}</div>

              <div class="message-footer">
                <div class="reason-tags">
                  {#if message.needs_response}
                    <span class="reason-tag needs-response">Needs reply</span>
                  {/if}
                  {#each message.reasons.filter((r) => r !== "normal") as reason}
                    <span class="reason-tag">{getReasonLabel(reason)}</span>
                  {/each}
                </div>

                <div class="message-actions">
                  <button
                    class="action-btn view"
                    onclick={() => handleViewConversation(message.chat_id)}
                    title="View conversation"
                  >
                    <svg
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      stroke-width="2"
                    >
                      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
                      <circle cx="12" cy="12" r="3"></circle>
                    </svg>
                  </button>
                  {#if message.handled}
                    <button
                      class="action-btn restore"
                      onclick={() => handleUnmarkHandled(message)}
                      title="Restore to inbox"
                    >
                      <svg
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        stroke-width="2"
                      >
                        <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"
                        ></path>
                        <path d="M3 3v5h5"></path>
                      </svg>
                    </button>
                  {:else}
                    <button
                      class="action-btn done"
                      onclick={() => handleMarkHandled(message)}
                      title="Mark as handled"
                    >
                      <svg
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        stroke-width="2"
                      >
                        <polyline points="20 6 9 17 4 12"></polyline>
                      </svg>
                    </button>
                  {/if}
                </div>
              </div>
            </div>
          {/each}
        </div>
      {/if}
    </div>
  </div>
</div>

<style>
  .panel-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
    animation: fadeIn 0.15s ease;
  }

  @keyframes fadeIn {
    from {
      opacity: 0;
    }
    to {
      opacity: 1;
    }
  }

  .panel {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    width: 90%;
    max-width: 600px;
    max-height: 85vh;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    animation: slideUp 0.2s ease;
  }

  @keyframes slideUp {
    from {
      transform: translateY(20px);
      opacity: 0;
    }
    to {
      transform: translateY(0);
      opacity: 1;
    }
  }

  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 20px;
    border-bottom: 1px solid var(--border-color);
  }

  .panel-title {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .priority-icon {
    width: 28px;
    height: 28px;
    background: var(--error-color);
    color: white;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    font-size: 16px;
  }

  .panel-title h2 {
    font-size: 18px;
    font-weight: 600;
  }

  .close-btn {
    background: none;
    border: none;
    font-size: 24px;
    color: var(--text-secondary);
    cursor: pointer;
    width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 6px;
    transition: all 0.15s ease;
  }

  .close-btn:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }

  .panel-stats {
    display: flex;
    gap: 16px;
    padding: 12px 20px;
    border-bottom: 1px solid var(--border-color);
    background: var(--bg-primary);
  }

  .stat {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2px;
  }

  .stat-value {
    font-size: 20px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .stat-value.critical {
    color: var(--error-color);
  }

  .stat-value.high {
    color: #f59e0b;
  }

  .stat-label {
    font-size: 11px;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .panel-filters {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 20px;
    border-bottom: 1px solid var(--border-color);
  }

  .filter-checkbox {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 13px;
    color: var(--text-secondary);
    cursor: pointer;
  }

  .filter-checkbox input {
    accent-color: var(--accent-color);
  }

  .filter-select {
    padding: 6px 12px;
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 13px;
    cursor: pointer;
  }

  .filter-select:focus {
    outline: none;
    border-color: var(--accent-color);
  }

  .refresh-btn {
    margin-left: auto;
    background: none;
    border: none;
    color: var(--text-secondary);
    cursor: pointer;
    padding: 6px;
    border-radius: 6px;
    transition: all 0.15s ease;
  }

  .refresh-btn:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }

  .refresh-btn svg {
    width: 18px;
    height: 18px;
  }

  .panel-content {
    flex: 1;
    overflow-y: auto;
    padding: 12px;
  }

  .loading-section {
    padding: 60px 20px;
  }

  .loading-indicator {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 16px;
    color: var(--text-secondary);
  }

  .spinner {
    width: 16px;
    height: 16px;
    border: 2px solid rgba(255, 255, 255, 0.3);
    border-top-color: white;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  .spinner.large {
    width: 32px;
    height: 32px;
    border-width: 3px;
    border-color: rgba(255, 255, 255, 0.2);
    border-top-color: var(--accent-color);
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  .error-section,
  .empty-section {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 60px 20px;
    text-align: center;
  }

  .error-icon,
  .empty-icon {
    width: 48px;
    height: 48px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    font-weight: bold;
    margin-bottom: 12px;
  }

  .error-icon {
    background: var(--error-color);
    color: white;
  }

  .empty-icon {
    background: var(--bg-hover);
    color: var(--text-secondary);
  }

  .error-message {
    color: var(--error-color);
    margin-bottom: 16px;
  }

  .empty-subtitle {
    color: var(--text-secondary);
    font-size: 14px;
  }

  .retry-btn {
    padding: 10px 24px;
    background: var(--error-color);
    border: none;
    border-radius: 8px;
    color: white;
    font-size: 14px;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .retry-btn:hover {
    opacity: 0.9;
  }

  .message-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .message-item {
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 14px 16px;
    transition: all 0.15s ease;
  }

  .message-item:hover {
    border-color: var(--bg-hover);
    background: var(--bg-hover);
  }

  .message-item.handled {
    opacity: 0.6;
  }

  .message-item.critical {
    border-left: 3px solid var(--error-color);
  }

  .message-item.high {
    border-left: 3px solid #f59e0b;
  }

  .message-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 8px;
  }

  .sender-info {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .sender-name {
    font-weight: 600;
    font-size: 14px;
  }

  .conversation-name {
    font-size: 12px;
    color: var(--text-secondary);
  }

  .message-meta {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .priority-badge {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    padding: 2px 6px;
    border-radius: 4px;
    color: white;
  }

  .message-time {
    font-size: 12px;
    color: var(--text-secondary);
  }

  .message-text {
    font-size: 14px;
    line-height: 1.4;
    margin-bottom: 10px;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }

  .message-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .reason-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
  }

  .reason-tag {
    font-size: 10px;
    padding: 2px 6px;
    background: var(--bg-secondary);
    border-radius: 4px;
    color: var(--text-secondary);
  }

  .reason-tag.needs-response {
    background: rgba(239, 68, 68, 0.2);
    color: var(--error-color);
    font-weight: 500;
  }

  .message-actions {
    display: flex;
    gap: 4px;
  }

  .action-btn {
    background: none;
    border: none;
    color: var(--text-secondary);
    cursor: pointer;
    padding: 6px;
    border-radius: 6px;
    transition: all 0.15s ease;
  }

  .action-btn:hover {
    background: var(--bg-secondary);
  }

  .action-btn.view:hover {
    color: var(--accent-color);
  }

  .action-btn.done:hover {
    color: #10b981;
  }

  .action-btn.restore:hover {
    color: #f59e0b;
  }

  .action-btn svg {
    width: 16px;
    height: 16px;
  }
</style>
