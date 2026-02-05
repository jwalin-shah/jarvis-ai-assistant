<script lang="ts">
  import { onMount } from "svelte";
  import { api, APIError } from "../api/client";
  import type { Calendar, DetectedEvent } from "../api/types";

  export let chatId: string;
  export let messageLimit: number = 50;

  let detectedEvents: DetectedEvent[] = [];
  let calendars: Calendar[] = [];
  let selectedCalendarId: string = "";
  let loading = false;
  let calendarsLoading = false;
  let error: string | null = null;
  let successMessage: string | null = null;
  let addingEventId: string | null = null;

  // Format date for display
  function formatDateTime(dateStr: string, allDay: boolean): string {
    const date = new Date(dateStr);
    if (allDay) {
      return date.toLocaleDateString([], {
        weekday: "short",
        month: "short",
        day: "numeric",
      });
    }
    return date.toLocaleString([], {
      weekday: "short",
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
    });
  }

  // Get confidence level color
  function getConfidenceColor(confidence: number): string {
    if (confidence >= 0.8) return "var(--color-success)";
    if (confidence >= 0.6) return "var(--color-warning)";
    return "var(--color-muted)";
  }

  // Get confidence label
  function getConfidenceLabel(confidence: number): string {
    if (confidence >= 0.8) return "High";
    if (confidence >= 0.6) return "Medium";
    return "Low";
  }

  // Generate unique ID for event (for tracking adding state)
  function getEventKey(event: DetectedEvent): string {
    return `${event.title}-${event.start}-${event.message_id || "no-msg"}`;
  }

  async function loadCalendars() {
    try {
      calendarsLoading = true;
      calendars = await api.getCalendars();
      // Select first editable calendar by default
      const editableCalendar = calendars.find((c) => c.is_editable);
      if (editableCalendar) {
        selectedCalendarId = editableCalendar.id;
      }
    } catch (err) {
      console.error("Failed to load calendars:", err);
      // Calendars will be empty, but events can still be detected
    } finally {
      calendarsLoading = false;
    }
  }

  async function detectEvents() {
    if (!chatId) return;

    try {
      loading = true;
      error = null;
      successMessage = null;
      detectedEvents = await api.detectEventsInMessages(chatId, messageLimit);
    } catch (err) {
      if (err instanceof APIError) {
        error = err.detail || err.message;
      } else {
        error = "Failed to detect events";
      }
      console.error("Event detection error:", err);
    } finally {
      loading = false;
    }
  }

  async function addToCalendar(event: DetectedEvent) {
    if (!selectedCalendarId) {
      error = "Please select a calendar first";
      return;
    }

    const eventKey = getEventKey(event);
    try {
      addingEventId = eventKey;
      error = null;
      successMessage = null;

      const result = await api.createEventFromDetected(selectedCalendarId, event);

      if (result.success) {
        successMessage = `Added "${event.title}" to calendar`;
        // Remove the event from the list after successful add
        detectedEvents = detectedEvents.filter((e) => getEventKey(e) !== eventKey);
      } else {
        error = result.error || "Failed to create event";
      }
    } catch (err) {
      if (err instanceof APIError) {
        error = err.detail || err.message;
      } else {
        error = "Failed to add event to calendar";
      }
      console.error("Add to calendar error:", err);
    } finally {
      addingEventId = null;
    }
  }

  onMount(() => {
    loadCalendars();
  });

  // Re-detect when chatId changes
  $: if (chatId) {
    detectEvents();
  }
</script>

<div class="event-detection">
  <div class="header">
    <h3>Detected Events</h3>
    <button
      class="refresh-btn"
      on:click={detectEvents}
      disabled={loading}
      title="Refresh detected events"
    >
      {#if loading}
        <span class="spinner"></span>
      {:else}
        &#8635;
      {/if}
    </button>
  </div>

  {#if error}
    <div class="error-message">{error}</div>
  {/if}

  {#if successMessage}
    <div class="success-message">{successMessage}</div>
  {/if}

  {#if calendars.length > 0}
    <div class="calendar-selector">
      <label for="calendar-select">Add to:</label>
      <select id="calendar-select" bind:value={selectedCalendarId}>
        {#each calendars as calendar}
          <option value={calendar.id} disabled={!calendar.is_editable}>
            {calendar.name}
            {calendar.is_editable ? "" : "(read-only)"}
          </option>
        {/each}
      </select>
    </div>
  {/if}

  {#if loading}
    <div class="loading">Scanning messages for events...</div>
  {:else if detectedEvents.length === 0}
    <div class="empty">No events detected in recent messages</div>
  {:else}
    <div class="events-list">
      {#each detectedEvents as event (getEventKey(event))}
        <div class="event-card">
          <div class="event-header">
            <span class="event-title">{event.title}</span>
            <span
              class="confidence-badge"
              style="background-color: {getConfidenceColor(event.confidence)}"
              title="Detection confidence: {Math.round(event.confidence * 100)}%"
            >
              {getConfidenceLabel(event.confidence)}
            </span>
          </div>

          <div class="event-details">
            <div class="event-time">
              <span class="icon">&#128197;</span>
              <span>{formatDateTime(event.start, event.all_day)}</span>
              {#if event.all_day}
                <span class="all-day-badge">All Day</span>
              {/if}
            </div>

            {#if event.location}
              <div class="event-location">
                <span class="icon">&#128205;</span>
                <span>{event.location}</span>
              </div>
            {/if}

            <div class="event-source">
              <span class="source-text" title={event.source_text}>
                "{event.source_text.length > 80
                  ? event.source_text.slice(0, 80) + "..."
                  : event.source_text}"
              </span>
            </div>
          </div>

          <div class="event-actions">
            <button
              class="add-btn"
              on:click={() => addToCalendar(event)}
              disabled={addingEventId === getEventKey(event) ||
                !selectedCalendarId ||
                calendars.length === 0}
            >
              {#if addingEventId === getEventKey(event)}
                <span class="spinner small"></span>
                Adding...
              {:else}
                <span class="icon">&#128197;</span>
                Add to Calendar
              {/if}
            </button>
          </div>
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .event-detection {
    padding: 16px;
    background: var(--bg-secondary, #f5f5f5);
    border-radius: 12px;
  }

  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
  }

  .header h3 {
    margin: 0;
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary, #1a1a1a);
  }

  .refresh-btn {
    width: 32px;
    height: 32px;
    border: none;
    border-radius: 8px;
    background: var(--bg-tertiary, #e5e5e5);
    color: var(--text-primary, #1a1a1a);
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    transition: background 0.2s;
  }

  .refresh-btn:hover:not(:disabled) {
    background: var(--bg-hover, #d5d5d5);
  }

  .refresh-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .calendar-selector {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
    font-size: 14px;
  }

  .calendar-selector label {
    color: var(--text-secondary, #666);
  }

  .calendar-selector select {
    flex: 1;
    padding: 6px 10px;
    border: 1px solid var(--border-color, #ddd);
    border-radius: 6px;
    background: var(--bg-primary, #fff);
    font-size: 14px;
    color: var(--text-primary, #1a1a1a);
  }

  .error-message {
    padding: 10px 12px;
    margin-bottom: 12px;
    background: #fee2e2;
    border: 1px solid #fecaca;
    border-radius: 8px;
    color: #dc2626;
    font-size: 13px;
  }

  .success-message {
    padding: 10px 12px;
    margin-bottom: 12px;
    background: #dcfce7;
    border: 1px solid #bbf7d0;
    border-radius: 8px;
    color: #16a34a;
    font-size: 13px;
  }

  .loading,
  .empty {
    padding: 24px;
    text-align: center;
    color: var(--text-secondary, #666);
    font-size: 14px;
  }

  .events-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .event-card {
    padding: 14px;
    background: var(--bg-primary, #fff);
    border-radius: 10px;
    border: 1px solid var(--border-color, #e5e5e5);
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
  }

  .event-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 10px;
  }

  .event-title {
    font-weight: 600;
    font-size: 15px;
    color: var(--text-primary, #1a1a1a);
  }

  .confidence-badge {
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 11px;
    font-weight: 500;
    color: white;
    text-transform: uppercase;
  }

  .event-details {
    display: flex;
    flex-direction: column;
    gap: 6px;
    font-size: 13px;
    color: var(--text-secondary, #666);
  }

  .event-time,
  .event-location {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .icon {
    font-size: 14px;
    width: 20px;
    text-align: center;
  }

  .all-day-badge {
    padding: 1px 6px;
    background: var(--bg-tertiary, #e5e5e5);
    border-radius: 4px;
    font-size: 11px;
    color: var(--text-secondary, #666);
  }

  .event-source {
    margin-top: 4px;
    padding-top: 8px;
    border-top: 1px solid var(--border-color, #e5e5e5);
  }

  .source-text {
    font-size: 12px;
    color: var(--text-muted, #999);
    font-style: italic;
    line-height: 1.4;
  }

  .event-actions {
    margin-top: 12px;
    display: flex;
    justify-content: flex-end;
  }

  .add-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 14px;
    border: none;
    border-radius: 8px;
    background: var(--color-primary, #3b82f6);
    color: white;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.2s;
  }

  .add-btn:hover:not(:disabled) {
    background: var(--color-primary-hover, #2563eb);
  }

  .add-btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .spinner {
    width: 18px;
    height: 18px;
    border: 2px solid transparent;
    border-top-color: currentColor;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  .spinner.small {
    width: 14px;
    height: 14px;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  /* Dark mode support */
  :global(.dark) .event-detection {
    background: var(--bg-secondary-dark, #2a2a2a);
  }

  :global(.dark) .event-card {
    background: var(--bg-primary-dark, #1a1a1a);
    border-color: var(--border-color-dark, #333);
  }

  :global(.dark) .error-message {
    background: #450a0a;
    border-color: #7f1d1d;
    color: #fecaca;
  }

  :global(.dark) .success-message {
    background: #052e16;
    border-color: #166534;
    color: #bbf7d0;
  }
</style>
