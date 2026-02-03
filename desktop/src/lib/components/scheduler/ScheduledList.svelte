<script lang="ts">
  import { apiClient, APIError } from "../../api/client";
  import type { ScheduledItem, ScheduledStatus } from "../../api/types";

  interface Props {
    contactId?: number;
    limit?: number;
  }

  let { contactId, limit = 50 }: Props = $props();

  // State
  let items: ScheduledItem[] = $state([]);
  let totalCount = $state(0);
  let pendingCount = $state(0);
  let sentCount = $state(0);
  let failedCount = $state(0);
  let isLoading = $state(false);
  let error = $state("");
  let filterStatus: ScheduledStatus | "" = $state("");
  let expandedItemId: string | null = $state(null);
  let editingItemId: string | null = $state(null);
  let editText = $state("");
  let rescheduleItemId: string | null = $state(null);
  let rescheduleDate = $state("");
  let rescheduleTime = $state("");

  // Load items on mount and when filters change
  $effect(() => {
    loadItems();
  });

  async function loadItems() {
    isLoading = true;
    error = "";

    try {
      const status = filterStatus || undefined;
      const response = await apiClient.getScheduledItems(contactId, status, limit);
      items = response.items;
      totalCount = response.total;
      pendingCount = response.pending;
      sentCount = response.sent;
      failedCount = response.failed;
    } catch (e) {
      if (e instanceof APIError) {
        error = e.detail || e.message;
      } else if (e instanceof Error) {
        error = e.message;
      }
    } finally {
      isLoading = false;
    }
  }

  async function cancelItem(itemId: string) {
    try {
      await apiClient.cancelScheduledItem(itemId);
      await loadItems();
    } catch (e) {
      if (e instanceof APIError) {
        error = e.detail || e.message;
      }
    }
  }

  async function saveEdit(itemId: string) {
    if (!editText.trim()) return;

    try {
      await apiClient.updateScheduledMessage(itemId, editText.trim());
      editingItemId = null;
      editText = "";
      await loadItems();
    } catch (e) {
      if (e instanceof APIError) {
        error = e.detail || e.message;
      }
    }
  }

  async function saveReschedule(itemId: string) {
    if (!rescheduleDate || !rescheduleTime) return;

    try {
      const sendAt = new Date(`${rescheduleDate}T${rescheduleTime}`).toISOString();
      await apiClient.rescheduleItem(itemId, sendAt);
      rescheduleItemId = null;
      rescheduleDate = "";
      rescheduleTime = "";
      await loadItems();
    } catch (e) {
      if (e instanceof APIError) {
        error = e.detail || e.message;
      }
    }
  }

  function startEdit(item: ScheduledItem) {
    editingItemId = item.id;
    editText = item.message_text;
    rescheduleItemId = null;
  }

  function startReschedule(item: ScheduledItem) {
    rescheduleItemId = item.id;
    const date = new Date(item.send_at);
    rescheduleDate = date.toISOString().split("T")[0];
    rescheduleTime = date.toTimeString().slice(0, 5);
    editingItemId = null;
  }

  function toggleExpand(itemId: string) {
    expandedItemId = expandedItemId === itemId ? null : itemId;
  }

  function formatDate(isoString: string): string {
    return new Date(isoString).toLocaleString(undefined, {
      weekday: "short",
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
    });
  }

  function getStatusColor(status: ScheduledStatus): string {
    switch (status) {
      case "pending":
        return "#ff9500";
      case "queued":
        return "#ffcc00";
      case "sending":
        return "#007aff";
      case "sent":
        return "#34c759";
      case "failed":
        return "#ff3b30";
      case "cancelled":
        return "#8e8e93";
      case "expired":
        return "#8e8e93";
      default:
        return "#8e8e93";
    }
  }

  function getPriorityColor(priority: string): string {
    switch (priority) {
      case "urgent":
        return "#ff3b30";
      case "normal":
        return "#007aff";
      case "low":
        return "#8e8e93";
      default:
        return "#8e8e93";
    }
  }

  function canEdit(item: ScheduledItem): boolean {
    return item.status === "pending" || item.status === "queued";
  }
</script>

<div class="scheduled-list">
  <div class="list-header">
    <h3>Scheduled Messages</h3>
    <div class="header-stats">
      <span class="stat pending">{pendingCount} pending</span>
      <span class="stat sent">{sentCount} sent</span>
      <span class="stat failed">{failedCount} failed</span>
    </div>
  </div>

  <!-- Filter Bar -->
  <div class="filter-bar">
    <select bind:value={filterStatus} onchange={loadItems}>
      <option value="">All Status</option>
      <option value="pending">Pending</option>
      <option value="sent">Sent</option>
      <option value="failed">Failed</option>
      <option value="cancelled">Cancelled</option>
    </select>
    <button class="refresh-btn" onclick={loadItems} disabled={isLoading}>
      {isLoading ? "Loading..." : "Refresh"}
    </button>
  </div>

  {#if error}
    <div class="error-message">{error}</div>
  {/if}

  {#if isLoading && items.length === 0}
    <div class="loading-state">Loading scheduled messages...</div>
  {:else if items.length === 0}
    <div class="empty-state">
      <p>No scheduled messages found.</p>
    </div>
  {:else}
    <div class="items-list">
      {#each items as item (item.id)}
        <div class="scheduled-item" class:expanded={expandedItemId === item.id}>
          <!-- Item Header -->
          <button class="item-header" onclick={() => toggleExpand(item.id)}>
            <div class="item-info">
              <span
                class="status-badge"
                style="background-color: {getStatusColor(item.status)}"
              >
                {item.status}
              </span>
              <span
                class="priority-badge"
                style="color: {getPriorityColor(item.priority)}"
              >
                {item.priority}
              </span>
              <span class="send-time">{formatDate(item.send_at)}</span>
            </div>
            <div class="item-preview">
              {item.message_text.slice(0, 50)}
              {#if item.message_text.length > 50}...{/if}
            </div>
            <span class="expand-icon">{expandedItemId === item.id ? "−" : "+"}</span>
          </button>

          <!-- Expanded Content -->
          {#if expandedItemId === item.id}
            <div class="item-details">
              <!-- Message Content -->
              {#if editingItemId === item.id}
                <div class="edit-section">
                  <textarea bind:value={editText} rows="3"></textarea>
                  <div class="edit-actions">
                    <button onclick={() => (editingItemId = null)}>Cancel</button>
                    <button class="primary" onclick={() => saveEdit(item.id)}>
                      Save
                    </button>
                  </div>
                </div>
              {:else}
                <div class="message-content">
                  <h4>Message</h4>
                  <p>{item.message_text}</p>
                </div>
              {/if}

              <!-- Reschedule Section -->
              {#if rescheduleItemId === item.id}
                <div class="reschedule-section">
                  <h4>Reschedule</h4>
                  <div class="datetime-inputs">
                    <input type="date" bind:value={rescheduleDate} />
                    <input type="time" bind:value={rescheduleTime} />
                  </div>
                  <div class="reschedule-actions">
                    <button onclick={() => (rescheduleItemId = null)}>Cancel</button>
                    <button class="primary" onclick={() => saveReschedule(item.id)}>
                      Save
                    </button>
                  </div>
                </div>
              {:else}
                <!-- Meta Information -->
                <div class="item-meta">
                  <div class="meta-row">
                    <span class="meta-label">Scheduled:</span>
                    <span>{formatDate(item.send_at)}</span>
                  </div>
                  <div class="meta-row">
                    <span class="meta-label">Created:</span>
                    <span>{formatDate(item.created_at)}</span>
                  </div>
                  {#if item.result}
                    <div class="meta-row">
                      <span class="meta-label">Result:</span>
                      <span class:success={item.result.success} class:error={!item.result.success}>
                        {item.result.success ? "Sent" : item.result.error || "Failed"}
                      </span>
                    </div>
                  {/if}
                  {#if item.retry_count > 0}
                    <div class="meta-row">
                      <span class="meta-label">Retries:</span>
                      <span>{item.retry_count} / {item.max_retries}</span>
                    </div>
                  {/if}
                </div>
              {/if}

              <!-- Actions -->
              {#if canEdit(item) && rescheduleItemId !== item.id && editingItemId !== item.id}
                <div class="item-actions">
                  <button onclick={() => startEdit(item)}>Edit</button>
                  <button onclick={() => startReschedule(item)}>Reschedule</button>
                  <button class="danger" onclick={() => cancelItem(item.id)}>
                    Cancel
                  </button>
                </div>
              {/if}
            </div>
          {/if}
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .scheduled-list {
    background: var(--bg-primary, #1e1e1e);
    border-radius: 12px;
    padding: 1rem;
  }

  .list-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
  }

  .list-header h3 {
    margin: 0;
    font-size: 1.125rem;
    color: var(--text-primary, #fff);
  }

  .header-stats {
    display: flex;
    gap: 0.75rem;
    font-size: 0.75rem;
  }

  .stat {
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
  }

  .stat.pending {
    background: rgba(255, 149, 0, 0.2);
    color: #ff9500;
  }

  .stat.sent {
    background: rgba(52, 199, 89, 0.2);
    color: #34c759;
  }

  .stat.failed {
    background: rgba(255, 59, 48, 0.2);
    color: #ff3b30;
  }

  .filter-bar {
    display: flex;
    gap: 0.75rem;
    margin-bottom: 1rem;
  }

  .filter-bar select {
    flex: 1;
    background: var(--bg-secondary, #2a2a2a);
    border: 1px solid var(--border-color, #3a3a3a);
    border-radius: 6px;
    padding: 0.5rem;
    color: var(--text-primary, #fff);
    font-size: 0.875rem;
  }

  .refresh-btn {
    background: var(--bg-secondary, #2a2a2a);
    border: 1px solid var(--border-color, #3a3a3a);
    border-radius: 6px;
    padding: 0.5rem 1rem;
    color: var(--text-primary, #fff);
    font-size: 0.875rem;
    cursor: pointer;
  }

  .refresh-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .error-message {
    background: rgba(255, 59, 48, 0.1);
    color: #ff3b30;
    padding: 0.75rem;
    border-radius: 8px;
    margin-bottom: 1rem;
    font-size: 0.875rem;
  }

  .loading-state,
  .empty-state {
    text-align: center;
    padding: 2rem;
    color: var(--text-secondary, #999);
  }

  .items-list {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .scheduled-item {
    background: var(--bg-secondary, #2a2a2a);
    border: 1px solid var(--border-color, #3a3a3a);
    border-radius: 8px;
    overflow: hidden;
  }

  .item-header {
    width: 100%;
    display: flex;
    flex-direction: column;
    gap: 0.375rem;
    padding: 0.75rem;
    background: none;
    border: none;
    cursor: pointer;
    text-align: left;
    position: relative;
  }

  .item-info {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
  }

  .status-badge {
    font-size: 0.625rem;
    padding: 0.125rem 0.375rem;
    border-radius: 4px;
    color: #fff;
    text-transform: uppercase;
    font-weight: 600;
  }

  .priority-badge {
    font-size: 0.75rem;
    font-weight: 500;
  }

  .send-time {
    font-size: 0.75rem;
    color: var(--text-secondary, #999);
    margin-left: auto;
  }

  .item-preview {
    font-size: 0.875rem;
    color: var(--text-primary, #fff);
    line-height: 1.4;
  }

  .expand-icon {
    position: absolute;
    top: 0.75rem;
    right: 0.75rem;
    font-size: 1.25rem;
    color: var(--text-secondary, #999);
  }

  .item-details {
    padding: 0.75rem;
    border-top: 1px solid var(--border-color, #3a3a3a);
    background: var(--bg-tertiary, #252525);
  }

  .message-content,
  .edit-section,
  .reschedule-section {
    margin-bottom: 1rem;
  }

  .message-content h4,
  .reschedule-section h4 {
    font-size: 0.75rem;
    color: var(--text-secondary, #999);
    margin: 0 0 0.5rem 0;
    text-transform: uppercase;
  }

  .message-content p {
    margin: 0;
    font-size: 0.875rem;
    color: var(--text-primary, #fff);
    white-space: pre-wrap;
  }

  .edit-section textarea {
    width: 100%;
    background: var(--bg-secondary, #2a2a2a);
    border: 1px solid var(--border-color, #3a3a3a);
    border-radius: 6px;
    padding: 0.5rem;
    color: var(--text-primary, #fff);
    font-size: 0.875rem;
    resize: vertical;
    font-family: inherit;
  }

  .edit-actions,
  .reschedule-actions {
    display: flex;
    justify-content: flex-end;
    gap: 0.5rem;
    margin-top: 0.5rem;
  }

  .datetime-inputs {
    display: flex;
    gap: 0.5rem;
  }

  .datetime-inputs input {
    flex: 1;
    background: var(--bg-secondary, #2a2a2a);
    border: 1px solid var(--border-color, #3a3a3a);
    border-radius: 6px;
    padding: 0.5rem;
    color: var(--text-primary, #fff);
    font-size: 0.875rem;
  }

  .item-meta {
    display: flex;
    flex-direction: column;
    gap: 0.375rem;
    margin-bottom: 1rem;
  }

  .meta-row {
    display: flex;
    font-size: 0.75rem;
  }

  .meta-label {
    color: var(--text-secondary, #999);
    width: 80px;
  }

  .meta-row .success {
    color: #34c759;
  }

  .meta-row .error {
    color: #ff3b30;
  }

  .item-actions {
    display: flex;
    gap: 0.5rem;
  }

  .item-actions button,
  .edit-actions button,
  .reschedule-actions button {
    padding: 0.375rem 0.75rem;
    border-radius: 6px;
    font-size: 0.75rem;
    cursor: pointer;
    background: var(--bg-secondary, #2a2a2a);
    border: 1px solid var(--border-color, #3a3a3a);
    color: var(--text-primary, #fff);
  }

  .item-actions button:hover,
  .edit-actions button:hover,
  .reschedule-actions button:hover {
    background: var(--bg-tertiary, #3a3a3a);
  }

  .item-actions button.danger {
    border-color: #ff3b30;
    color: #ff3b30;
  }

  .item-actions button.danger:hover {
    background: rgba(255, 59, 48, 0.1);
  }

  button.primary {
    background: var(--accent-color, #007aff);
    border-color: var(--accent-color, #007aff);
    color: #fff;
  }

  button.primary:hover {
    background: #0066d6;
  }
</style>
