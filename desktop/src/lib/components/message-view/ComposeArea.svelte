<script lang="ts">
  import { tick } from 'svelte';
  import { SendIcon } from '../icons';
  import Button from '../ui/Button.svelte';

  interface Props {
    value?: string;
    disabled?: boolean;
    sending?: boolean;
    placeholder?: string;
    onSend?: (text: string) => void;
    onAIDraft?: () => void;
    onFocus?: () => void;
  }

  let {
    value = $bindable(''),
    disabled = false,
    sending = false,
    placeholder = 'iMessage',
    onSend,
    onAIDraft,
    onFocus,
  }: Props = $props();

  let textareaRef = $state<HTMLTextAreaElement | null>(null);

  // Auto-resize textarea as user types
  function autoResize(event: Event) {
    const textarea = event.target as HTMLTextAreaElement;
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
  }

  // Handle Enter key (send on Enter, new line on Shift+Enter)
  function handleKeydown(event: KeyboardEvent) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  }

  function handleSend() {
    const trimmed = value.trim();
    if (!trimmed || disabled || sending) return;

    onSend?.(trimmed);

    // Reset input
    value = '';
    if (textareaRef) {
      textareaRef.style.height = 'auto';
    }
  }

  // Public method to focus the textarea
  export function focus() {
    tick().then(() => {
      textareaRef?.focus();
    });
  }

  // Public method to set value and resize
  export function setValue(text: string) {
    value = text;
    tick().then(() => {
      if (textareaRef) {
        textareaRef.style.height = 'auto';
        textareaRef.style.height = Math.min(textareaRef.scrollHeight, 120) + 'px';
        textareaRef.focus();
      }
    });
  }
</script>

<div class="compose-area">
  <div class="compose-input-wrapper">
    <textarea
      bind:this={textareaRef}
      class="compose-input"
      bind:value
      {placeholder}
      rows="1"
      aria-label="Compose message"
      onkeydown={handleKeydown}
      oninput={autoResize}
      onfocus={onFocus}
      disabled={disabled || sending}
    ></textarea>
    <div class="compose-actions">
      {#if onAIDraft}
        <button
          type="button"
          class="action-btn ai-draft-btn"
          onclick={onAIDraft}
          disabled={disabled}
          title="Generate AI Draft"
          aria-label="Generate AI Draft"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18">
            <path d="M12 3l1.912 5.886L20 10.8l-5.886 1.912L12 18.6l-1.912-5.886L4.2 10.8l5.886-1.912L12 3z" fill="currentColor" opacity="0.2" />
            <path d="M5 3L6 5L9 6L6 7L5 9L4 7L1 6L4 5L5 3z" fill="currentColor" />
            <path d="M19 15L20 17L23 18L20 19L19 21L18 19L15 18L18 17L19 15z" fill="currentColor" />
          </svg>
        </button>
      {/if}
      <Button
        variant="primary"
        size="sm"
        onclick={handleSend}
        disabled={!value.trim() || disabled}
        loading={sending}
        title="Send message (Enter)"
        aria-label="Send message"
      >
        <SendIcon size={18} />
      </Button>
    </div>
  </div>
</div>

<style>
  .compose-area {
    padding: var(--space-2) var(--space-4) var(--space-4);
    border-top: 1px solid var(--border-default);
    background: var(--surface-elevated);
  }

  .compose-input-wrapper {
    display: flex;
    align-items: flex-end;
    gap: var(--space-2);
    background: var(--surface-base);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-2xl);
    padding: var(--space-2) var(--space-2) var(--space-2) var(--space-4);
    transition: border-color var(--duration-fast) var(--ease-out);
  }

  .compose-input-wrapper:focus-within {
    border-color: var(--color-primary);
  }

  .compose-actions {
    display: flex;
    align-items: center;
    gap: var(--space-1);
    padding-bottom: var(--space-1);
  }

  .action-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    background: none;
    border: none;
    border-radius: var(--radius-md);
    color: var(--text-tertiary);
    cursor: pointer;
    padding: var(--space-1);
    transition: all var(--duration-fast) var(--ease-out);
  }

  .action-btn:hover:not(:disabled) {
    background: var(--surface-base-hover);
    color: var(--color-primary);
  }

  .ai-draft-btn {
    color: var(--color-primary);
    opacity: 0.8;
  }

  .ai-draft-btn:hover {
    opacity: 1;
    transform: scale(1.05);
  }

  .compose-input {
    flex: 1;
    background: transparent;
    border: none;
    color: var(--text-primary);
    font-size: var(--text-base);
    line-height: var(--line-height-normal);
    resize: none;
    outline: none;
    max-height: 120px;
    min-height: 24px;
    padding: var(--space-1) 0;
    font-family: var(--font-family-sans);
  }

  .compose-input::placeholder {
    color: var(--text-tertiary);
  }

  .compose-input:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
</style>
