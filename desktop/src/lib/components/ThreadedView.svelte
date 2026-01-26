<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { api } from "../api/client";
  import type { ThreadedViewResponse, ThreadResponse, ThreadedMessage } from "../api/types";

  // Props
  export let chatId: string;
  export let onClose: () => void;
  export let onMessageClick: ((messageId: number) => void) | undefined = undefined;

  // State
  let loading = $state(true);
  let error: string | null = $state(null);
  let threadedView: ThreadedViewResponse | null = $state(null);
  let expandedThreads: Set<string> = $state(new Set());
  let timeGapMinutes = $state(30);
  let useSemantic = $state(true);

  // Computed: messages grouped by thread
  function getMessagesByThread(): Map<string, ThreadedMessage[]> {
    const map = new Map<string, ThreadedMessage[]>();
    if (!threadedView) return map;

    for (const msg of threadedView.messages) {
      const existing = map.get(msg.thread_id) || [];
      existing.push(msg);
      map.set(msg.thread_id, existing);
    }
    return map;
  }

  async function fetchThreadedView() {
    loading = true;
    error = null;
    try {
      threadedView = await api.getThreadedView(
        chatId,
        200,
        undefined,
        timeGapMinutes,
        useSemantic
      );
      // Auto-expand first few threads
      if (threadedView && threadedView.threads.length > 0) {
        const initialExpanded = new Set<string>();
        threadedView.threads.slice(0, 3).forEach((t) => {
          initialExpanded.add(t.thread_id);
        });
        expandedThreads = initialExpanded;
      }
    } catch (e) {
      if (e instanceof Error) {
        error = `Failed to load threaded view: ${e.message}`;
      } else {
        error = "Failed to load threaded view. Please try again.";
      }
    } finally {
      loading = false;
    }
  }

  function toggleThread(threadId: string) {
    const newExpanded = new Set(expandedThreads);
    if (newExpanded.has(threadId)) {
      newExpanded.delete(threadId);
    } else {
      newExpanded.add(threadId);
    }
    expandedThreads = newExpanded;
  }

  function expandAll() {
    if (!threadedView) return;
    const all = new Set<string>();
    threadedView.threads.forEach((t) => all.add(t.thread_id));
    expandedThreads = all;
  }

  function collapseAll() {
    expandedThreads = new Set();
  }

  function handleMessageClick(messageId: number) {
    if (onMessageClick) {
      onMessageClick(messageId);
      onClose();
    }
  }

  function formatTime(dateStr: string): string {
    return new Date(dateStr).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    });
  }

  function formatDate(dateStr: string): string {
    const date = new Date(dateStr);
    const today = new Date();
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);

    if (date.toDateString() === today.toDateString()) {
      return "Today";
    } else if (date.toDateString() === yesterday.toDateString()) {
      return "Yesterday";
    } else {
      return date.toLocaleDateString([], {
        month: "short",
        day: "numeric",
      });
    }
  }

  function formatThreadDuration(thread: ThreadResponse): string {
    if (!thread.start_time || !thread.end_time) return "";
    const start = new Date(thread.start_time);
    const end = new Date(thread.end_time);
    const diffMs = end.getTime() - start.getTime();
    const diffMins = Math.round(diffMs / 60000);

    if (diffMins < 60) {
      return `${diffMins}m`;
    } else if (diffMins < 1440) {
      const hours = Math.floor(diffMins / 60);
      return `${hours}h`;
    } else {
      const days = Math.floor(diffMins / 1440);
      return `${days}d`;
    }
  }

  function getThreadColor(index: number): string {
    const colors = [
      "var(--thread-blue)",
      "var(--thread-green)",
      "var(--thread-purple)",
      "var(--thread-orange)",
      "var(--thread-pink)",
      "var(--thread-teal)",
    ];
    return colors[index % colors.length];
  }

  function handleBackdropClick(event: MouseEvent) {
    if (event.target === event.currentTarget) {
      onClose();
    }
  }

  function handleKeydown(event: KeyboardEvent) {
    if (event.key === "Escape") {
      onClose();
    }
  }

  onMount(() => {
    fetchThreadedView();
    window.addEventListener("keydown", handleKeydown);
    return () => {
      window.removeEventListener("keydown", handleKeydown);
    };
  });
</script>

<!-- svelte-ignore a11y_click_events_have_key_events -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<div class="modal-overlay" onclick={handleBackdropClick}>
  <div class="modal" role="dialog" aria-modal="true" aria-labelledby="modal-title">
    <div class="modal-header">
      <h2 id="modal-title">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="header-icon">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
          <line x1="9" y1="10" x2="15" y2="10"></line>
        </svg>
        Threaded View
      </h2>
      <div class="header-actions">
        <button class="icon-btn" onclick={expandAll} title="Expand all threads" aria-label="Expand all">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="6 9 12 15 18 9"></polyline>
          </svg>
        </button>
        <button class="icon-btn" onclick={collapseAll} title="Collapse all threads" aria-label="Collapse all">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="18 15 12 9 6 15"></polyline>
          </svg>
        </button>
        <button class="close-btn" onclick={onClose} aria-label="Close">
          x
        </button>
      </div>
    </div>

    <div class="settings-bar">
      <label class="setting">
        <span>Time gap:</span>
        <select bind:value={timeGapMinutes} onchange={fetchThreadedView}>
          <option value={15}>15 min</option>
          <option value={30}>30 min</option>
          <option value={60}>1 hour</option>
          <option value={120}>2 hours</option>
          <option value={360}>6 hours</option>
        </select>
      </label>
      <label class="setting checkbox">
        <input type="checkbox" bind:checked={useSemantic} onchange={fetchThreadedView} />
        <span>Topic detection</span>
      </label>
      {#if threadedView}
        <span class="stats">
          {threadedView.total_threads} threads / {threadedView.total_messages} messages
        </span>
      {/if}
    </div>

    <div class="modal-content">
      {#if loading}
        <div class="loading-state">
          <div class="loading-spinner"></div>
          <p>Analyzing conversation threads...</p>
        </div>
      {:else if error}
        <div class="error-state">
          <p class="error-message">{error}</p>
          <button class="retry-btn" onclick={fetchThreadedView}>
            Try Again
          </button>
        </div>
      {:else if threadedView && threadedView.threads.length > 0}
        <div class="threads-list">
          {#each threadedView.threads as thread, index (thread.thread_id)}
            {@const isExpanded = expandedThreads.has(thread.thread_id)}
            {@const messages = getMessagesByThread().get(thread.thread_id) || []}
            <div
              class="thread-group"
              style="--thread-color: {getThreadColor(index)}"
            >
              <button
                class="thread-header"
                class:expanded={isExpanded}
                onclick={() => toggleThread(thread.thread_id)}
                aria-expanded={isExpanded}
              >
                <div class="thread-indicator"></div>
                <div class="thread-info">
                  <span class="topic-label">{thread.topic_label || "Thread"}</span>
                  <span class="thread-meta">
                    {thread.message_count} messages
                    {#if thread.start_time}
                      <span class="dot">.</span>
                      {formatDate(thread.start_time)}
                      {#if formatThreadDuration(thread)}
                        <span class="dot">.</span>
                        {formatThreadDuration(thread)}
                      {/if}
                    {/if}
                  </span>
                </div>
                <svg
                  class="chevron"
                  class:rotated={isExpanded}
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  stroke-width="2"
                >
                  <polyline points="6 9 12 15 18 9"></polyline>
                </svg>
              </button>

              {#if isExpanded}
                <div class="thread-messages">
                  {#each messages as msg (msg.id)}
                    <button
                      class="message-item"
                      class:from-me={msg.is_from_me}
                      onclick={() => handleMessageClick(msg.id)}
                      type="button"
                    >
                      <div class="message-content">
                        {#if !msg.is_from_me}
                          <span class="sender">{msg.sender_name || msg.sender}</span>
                        {/if}
                        <p class="text">{msg.text}</p>
                      </div>
                      <span class="time">{formatTime(msg.date)}</span>
                    </button>
                  {/each}
                </div>
              {/if}
            </div>
          {/each}
        </div>
      {:else}
        <div class="empty-state">
          <p>No threads detected in this conversation.</p>
        </div>
      {/if}
    </div>

    <div class="modal-footer">
      <button class="btn secondary" onclick={fetchThreadedView} disabled={loading}>
        Refresh
      </button>
      <button class="btn primary" onclick={onClose}>
        Done
      </button>
    </div>
  </div>
</div>

<style>
  /* Thread colors */
  :global(:root) {
    --thread-blue: #3b82f6;
    --thread-green: #22c55e;
    --thread-purple: #a855f7;
    --thread-orange: #f97316;
    --thread-pink: #ec4899;
    --thread-teal: #14b8a6;
  }

  .modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.6);
    backdrop-filter: blur(4px);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
  }

  .modal {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    width: 90%;
    max-width: 640px;
    max-height: 85vh;
    display: flex;
    flex-direction: column;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
  }

  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 20px;
    border-bottom: 1px solid var(--border-color);
  }

  .modal-header h2 {
    margin: 0;
    font-size: 17px;
    font-weight: 600;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .header-icon {
    width: 20px;
    height: 20px;
  }

  .header-actions {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .icon-btn {
    background: none;
    border: none;
    color: var(--text-secondary);
    padding: 6px;
    border-radius: 6px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.15s;
  }

  .icon-btn:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }

  .icon-btn svg {
    width: 18px;
    height: 18px;
  }

  .close-btn {
    background: none;
    border: none;
    color: var(--text-secondary);
    font-size: 20px;
    cursor: pointer;
    padding: 4px 8px;
    border-radius: 6px;
    transition: background-color 0.15s;
    margin-left: 4px;
  }

  .close-btn:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }

  .settings-bar {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 12px 20px;
    background: var(--bg-primary);
    border-bottom: 1px solid var(--border-color);
    flex-wrap: wrap;
  }

  .setting {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 13px;
    color: var(--text-secondary);
  }

  .setting select {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 13px;
    color: var(--text-primary);
    cursor: pointer;
  }

  .setting.checkbox {
    cursor: pointer;
  }

  .setting.checkbox input {
    cursor: pointer;
  }

  .stats {
    margin-left: auto;
    font-size: 12px;
    color: var(--text-secondary);
  }

  .modal-content {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
    min-height: 300px;
  }

  .loading-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 200px;
    gap: 16px;
    color: var(--text-secondary);
  }

  .loading-spinner {
    width: 32px;
    height: 32px;
    border: 3px solid var(--border-color);
    border-top-color: var(--accent-color);
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  .error-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 200px;
    gap: 16px;
    text-align: center;
  }

  .error-message {
    color: var(--error-color);
    font-size: 14px;
  }

  .retry-btn {
    background: var(--bg-hover);
    border: 1px solid var(--border-color);
    color: var(--text-primary);
    padding: 8px 16px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 14px;
    transition: background-color 0.15s;
  }

  .retry-btn:hover {
    background: var(--bg-active);
  }

  .empty-state {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 200px;
    color: var(--text-secondary);
    font-size: 14px;
  }

  .threads-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .thread-group {
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    overflow: hidden;
  }

  .thread-header {
    width: 100%;
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    background: none;
    border: none;
    cursor: pointer;
    text-align: left;
    transition: background-color 0.15s;
  }

  .thread-header:hover {
    background: var(--bg-hover);
  }

  .thread-indicator {
    width: 4px;
    height: 36px;
    background: var(--thread-color);
    border-radius: 2px;
    flex-shrink: 0;
  }

  .thread-info {
    flex: 1;
    min-width: 0;
  }

  .topic-label {
    display: block;
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .thread-meta {
    display: block;
    font-size: 12px;
    color: var(--text-secondary);
    margin-top: 2px;
  }

  .dot {
    margin: 0 4px;
  }

  .chevron {
    width: 18px;
    height: 18px;
    color: var(--text-secondary);
    flex-shrink: 0;
    transition: transform 0.2s;
  }

  .chevron.rotated {
    transform: rotate(180deg);
  }

  .thread-messages {
    border-top: 1px solid var(--border-color);
    padding: 8px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .message-item {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 12px;
    padding: 10px 12px;
    background: var(--bg-secondary);
    border: 1px solid transparent;
    border-radius: 8px;
    cursor: pointer;
    text-align: left;
    transition: all 0.15s;
    width: 100%;
  }

  .message-item:hover {
    background: var(--bg-hover);
    border-color: var(--border-color);
  }

  .message-item.from-me {
    background: var(--bg-bubble-me-light, rgba(0, 122, 255, 0.1));
  }

  .message-item.from-me:hover {
    background: var(--bg-bubble-me-light-hover, rgba(0, 122, 255, 0.15));
  }

  .message-content {
    flex: 1;
    min-width: 0;
  }

  .sender {
    display: block;
    font-size: 12px;
    font-weight: 600;
    color: var(--accent-color);
    margin-bottom: 2px;
  }

  .text {
    margin: 0;
    font-size: 13px;
    line-height: 1.4;
    color: var(--text-primary);
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }

  .time {
    font-size: 11px;
    color: var(--text-secondary);
    white-space: nowrap;
    flex-shrink: 0;
  }

  .modal-footer {
    display: flex;
    gap: 10px;
    padding: 16px 20px;
    border-top: 1px solid var(--border-color);
    justify-content: flex-end;
  }

  .btn {
    padding: 8px 16px;
    border-radius: 6px;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s;
  }

  .btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .btn.primary {
    background: var(--accent-color);
    color: white;
    border: none;
  }

  .btn.primary:hover:not(:disabled) {
    background: #0a82e0;
  }

  .btn.secondary {
    background: var(--bg-hover);
    color: var(--text-primary);
    border: 1px solid var(--border-color);
  }

  .btn.secondary:hover:not(:disabled) {
    background: var(--bg-active);
  }
</style>
