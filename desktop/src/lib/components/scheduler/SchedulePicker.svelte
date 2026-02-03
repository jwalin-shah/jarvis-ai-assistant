<script lang="ts">
  import { apiClient, APIError } from "../../api/client";
  import type { TimingSuggestion, ScheduledPriority } from "../../api/types";

  interface Props {
    contactId: number;
    chatId: string;
    draftId: string;
    messageText: string;
    onSchedule: (sendAt: Date, priority: ScheduledPriority) => void;
    onCancel: () => void;
  }

  let { contactId, chatId, draftId, messageText, onSchedule, onCancel }: Props = $props();

  // State
  let selectedDate = $state("");
  let selectedTime = $state("");
  let priority: ScheduledPriority = $state("normal");
  let suggestions: TimingSuggestion[] = $state([]);
  let isLoading = $state(false);
  let error = $state("");
  let useSmartTiming = $state(true);

  // Get minimum date (now)
  const now = new Date();
  const minDate = now.toISOString().split("T")[0];
  const minTime = now.toTimeString().slice(0, 5);

  // Load timing suggestions on mount
  $effect(() => {
    loadSuggestions();
  });

  async function loadSuggestions() {
    isLoading = true;
    error = "";

    try {
      const response = await apiClient.getTimingSuggestions(
        contactId,
        undefined,
        undefined,
        3
      );
      suggestions = response.suggestions;

      // Pre-select the optimal suggestion
      if (suggestions.length > 0 && useSmartTiming) {
        const optimal = suggestions.find((s) => s.is_optimal) || suggestions[0];
        const suggestedDate = new Date(optimal.suggested_time);
        selectedDate = suggestedDate.toISOString().split("T")[0];
        selectedTime = suggestedDate.toTimeString().slice(0, 5);
      }
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

  function selectSuggestion(suggestion: TimingSuggestion) {
    const date = new Date(suggestion.suggested_time);
    selectedDate = date.toISOString().split("T")[0];
    selectedTime = date.toTimeString().slice(0, 5);
  }

  function handleSchedule() {
    if (!selectedDate || !selectedTime) {
      error = "Please select a date and time";
      return;
    }

    const sendAt = new Date(`${selectedDate}T${selectedTime}`);
    if (sendAt <= new Date()) {
      error = "Please select a future date and time";
      return;
    }

    onSchedule(sendAt, priority);
  }

  function formatSuggestionTime(isoString: string): string {
    const date = new Date(isoString);
    return date.toLocaleString(undefined, {
      weekday: "short",
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
    });
  }

  function getConfidenceLabel(confidence: number): string {
    if (confidence >= 0.8) return "Excellent";
    if (confidence >= 0.6) return "Good";
    if (confidence >= 0.4) return "Fair";
    return "Low";
  }
</script>

<div class="schedule-picker">
  <div class="picker-header">
    <h3>Schedule Message</h3>
    <button class="close-btn" onclick={onCancel} aria-label="Close">
      &times;
    </button>
  </div>

  {#if error}
    <div class="error-message">{error}</div>
  {/if}

  <!-- Smart Timing Suggestions -->
  {#if suggestions.length > 0}
    <div class="suggestions-section">
      <h4>Suggested Times</h4>
      <div class="suggestions-list">
        {#each suggestions as suggestion}
          <button
            class="suggestion-card"
            class:optimal={suggestion.is_optimal}
            onclick={() => selectSuggestion(suggestion)}
          >
            <div class="suggestion-time">
              {formatSuggestionTime(suggestion.suggested_time)}
            </div>
            <div class="suggestion-meta">
              <span class="confidence">
                {getConfidenceLabel(suggestion.confidence)}
              </span>
              {#if suggestion.reason}
                <span class="reason">{suggestion.reason}</span>
              {/if}
            </div>
            {#if suggestion.is_optimal}
              <span class="optimal-badge">Best</span>
            {/if}
          </button>
        {/each}
      </div>
    </div>
  {/if}

  <!-- Manual Date/Time Selection -->
  <div class="manual-section">
    <h4>Or choose manually</h4>
    <div class="datetime-inputs">
      <div class="input-group">
        <label for="schedule-date">Date</label>
        <input
          type="date"
          id="schedule-date"
          bind:value={selectedDate}
          min={minDate}
        />
      </div>
      <div class="input-group">
        <label for="schedule-time">Time</label>
        <input
          type="time"
          id="schedule-time"
          bind:value={selectedTime}
        />
      </div>
    </div>
  </div>

  <!-- Priority Selection -->
  <div class="priority-section">
    <h4>Priority</h4>
    <div class="priority-options">
      <label class="priority-option">
        <input
          type="radio"
          name="priority"
          value="urgent"
          bind:group={priority}
        />
        <span class="priority-label urgent">Urgent</span>
      </label>
      <label class="priority-option">
        <input
          type="radio"
          name="priority"
          value="normal"
          bind:group={priority}
        />
        <span class="priority-label normal">Normal</span>
      </label>
      <label class="priority-option">
        <input
          type="radio"
          name="priority"
          value="low"
          bind:group={priority}
        />
        <span class="priority-label low">Low</span>
      </label>
    </div>
  </div>

  <!-- Message Preview -->
  <div class="message-preview">
    <h4>Message</h4>
    <p class="preview-text">{messageText}</p>
  </div>

  <!-- Actions -->
  <div class="picker-actions">
    <button class="btn-cancel" onclick={onCancel}>Cancel</button>
    <button
      class="btn-schedule"
      onclick={handleSchedule}
      disabled={!selectedDate || !selectedTime || isLoading}
    >
      {isLoading ? "Loading..." : "Schedule"}
    </button>
  </div>
</div>

<style>
  .schedule-picker {
    background: var(--bg-primary, #1e1e1e);
    border-radius: 12px;
    padding: 1.5rem;
    max-width: 480px;
    width: 100%;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
  }

  .picker-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
  }

  .picker-header h3 {
    margin: 0;
    font-size: 1.25rem;
    color: var(--text-primary, #fff);
  }

  .close-btn {
    background: none;
    border: none;
    font-size: 1.5rem;
    color: var(--text-secondary, #999);
    cursor: pointer;
    padding: 0.25rem;
    line-height: 1;
  }

  .close-btn:hover {
    color: var(--text-primary, #fff);
  }

  .error-message {
    background: rgba(255, 59, 48, 0.1);
    color: #ff3b30;
    padding: 0.75rem;
    border-radius: 8px;
    margin-bottom: 1rem;
    font-size: 0.875rem;
  }

  .suggestions-section,
  .manual-section,
  .priority-section,
  .message-preview {
    margin-bottom: 1.5rem;
  }

  h4 {
    font-size: 0.875rem;
    color: var(--text-secondary, #999);
    margin: 0 0 0.75rem 0;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .suggestions-list {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .suggestion-card {
    background: var(--bg-secondary, #2a2a2a);
    border: 1px solid var(--border-color, #3a3a3a);
    border-radius: 8px;
    padding: 0.75rem;
    cursor: pointer;
    text-align: left;
    position: relative;
    transition: all 0.2s;
  }

  .suggestion-card:hover {
    border-color: var(--accent-color, #007aff);
  }

  .suggestion-card.optimal {
    border-color: var(--success-color, #34c759);
  }

  .suggestion-time {
    font-weight: 500;
    color: var(--text-primary, #fff);
    margin-bottom: 0.25rem;
  }

  .suggestion-meta {
    font-size: 0.75rem;
    color: var(--text-secondary, #999);
    display: flex;
    gap: 0.5rem;
  }

  .confidence {
    background: var(--bg-tertiary, #3a3a3a);
    padding: 0.125rem 0.5rem;
    border-radius: 4px;
  }

  .optimal-badge {
    position: absolute;
    top: 0.5rem;
    right: 0.5rem;
    background: var(--success-color, #34c759);
    color: #fff;
    font-size: 0.625rem;
    padding: 0.125rem 0.375rem;
    border-radius: 4px;
    text-transform: uppercase;
    font-weight: 600;
  }

  .datetime-inputs {
    display: flex;
    gap: 1rem;
  }

  .input-group {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 0.375rem;
  }

  .input-group label {
    font-size: 0.75rem;
    color: var(--text-secondary, #999);
  }

  .input-group input {
    background: var(--bg-secondary, #2a2a2a);
    border: 1px solid var(--border-color, #3a3a3a);
    border-radius: 6px;
    padding: 0.5rem;
    color: var(--text-primary, #fff);
    font-size: 0.875rem;
  }

  .input-group input:focus {
    outline: none;
    border-color: var(--accent-color, #007aff);
  }

  .priority-options {
    display: flex;
    gap: 1rem;
  }

  .priority-option {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    cursor: pointer;
  }

  .priority-option input {
    display: none;
  }

  .priority-label {
    padding: 0.5rem 1rem;
    border-radius: 6px;
    font-size: 0.875rem;
    font-weight: 500;
    border: 1px solid var(--border-color, #3a3a3a);
    transition: all 0.2s;
  }

  .priority-option input:checked + .priority-label.urgent {
    background: rgba(255, 59, 48, 0.2);
    border-color: #ff3b30;
    color: #ff3b30;
  }

  .priority-option input:checked + .priority-label.normal {
    background: rgba(0, 122, 255, 0.2);
    border-color: #007aff;
    color: #007aff;
  }

  .priority-option input:checked + .priority-label.low {
    background: rgba(142, 142, 147, 0.2);
    border-color: #8e8e93;
    color: #8e8e93;
  }

  .preview-text {
    background: var(--bg-secondary, #2a2a2a);
    padding: 0.75rem;
    border-radius: 8px;
    font-size: 0.875rem;
    color: var(--text-primary, #fff);
    margin: 0;
    max-height: 80px;
    overflow-y: auto;
  }

  .picker-actions {
    display: flex;
    justify-content: flex-end;
    gap: 0.75rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border-color, #3a3a3a);
  }

  .btn-cancel,
  .btn-schedule {
    padding: 0.625rem 1.25rem;
    border-radius: 8px;
    font-size: 0.875rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
  }

  .btn-cancel {
    background: var(--bg-secondary, #2a2a2a);
    border: 1px solid var(--border-color, #3a3a3a);
    color: var(--text-primary, #fff);
  }

  .btn-cancel:hover {
    background: var(--bg-tertiary, #3a3a3a);
  }

  .btn-schedule {
    background: var(--accent-color, #007aff);
    border: none;
    color: #fff;
  }

  .btn-schedule:hover:not(:disabled) {
    background: #0066d6;
  }

  .btn-schedule:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
</style>
