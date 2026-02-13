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
    onFocus?: () => void;
  }

  let {
    value = $bindable(''),
    disabled = false,
    sending = false,
    placeholder = 'iMessage',
    onSend,
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
    <Button
      variant="primary"
      size="sm"
      onclick={handleSend}
      disabled={!value.trim() || disabled}
      loading={sending}
      title="Send message (Enter)"
    >
      <SendIcon size={18} />
    </Button>
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
