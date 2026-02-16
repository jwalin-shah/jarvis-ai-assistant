<script lang="ts">
  import Icon from "./Icon.svelte";

  interface Props {
    onClose: () => void;
  }

  let { onClose }: Props = $props();

  const shortcuts = [
    {
      category: "Navigation",
      items: [
        { keys: ["⌘", "⇧", "P"], description: "Command Palette" },
        { keys: ["⌘", "K"], description: "Open search" },
        { keys: ["⌘", "1"], description: "Go to Dashboard" },
        { keys: ["⌘", "2"], description: "Go to Messages" },
        { keys: ["⌘", "3"], description: "Go to Templates" },
        { keys: ["⌘", ","], description: "Open Settings" },
        { keys: ["⌘", "⇧", "]"], description: "Next unread conversation" },
      ],
    },
    {
      category: "List Navigation",
      items: [
        { keys: ["j", "/", "↓"], description: "Next item" },
        { keys: ["k", "/", "↑"], description: "Previous item" },
        { keys: ["g"], description: "Go to first item" },
        { keys: ["G"], description: "Go to last item" },
        { keys: ["↵"], description: "Select item" },
        { keys: ["h", "/", "←"], description: "Go back" },
      ],
    },
    {
      category: "Messages",
      items: [
        { keys: ["r"], description: "Reply to message" },
        { keys: ["⌘", "D"], description: "Open AI Draft" },
        { keys: ["⌘", "S"], description: "Show Summary" },
        { keys: ["⌘", "E"], description: "Export to PDF" },
        { keys: ["↵"], description: "Send message" },
        { keys: ["⇧", "↵"], description: "New line" },
      ],
    },
    {
      category: "General",
      items: [
        { keys: ["⌘", "/"], description: "Show shortcuts" },
        { keys: ["⌘", "R"], description: "Refresh" },
        { keys: ["Esc"], description: "Close / Clear selection" },
      ],
    },
  ];

  function handleKeydown(event: KeyboardEvent) {
    if (event.key === "Escape") {
      onClose();
    }
  }

  function handleOverlayClick(event: MouseEvent) {
    if (event.target === event.currentTarget) {
      onClose();
    }
  }

  function handleOverlayKeydown(event: KeyboardEvent) {
    if (event.key === "Escape" || event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onClose();
    }
  }
</script>

<svelte:window onkeydown={handleKeydown} />

<div
  class="shortcuts-overlay"
  onclick={handleOverlayClick}
  onkeydown={handleOverlayKeydown}
  role="button"
  aria-label="Close keyboard shortcuts"
  tabindex="0"
>
  <div class="shortcuts-panel glass" role="dialog" aria-label="Keyboard Shortcuts" aria-modal="true" tabindex="-1">
    <header class="shortcuts-header">
      <div class="shortcuts-title">
        <Icon name="settings" size={20} />
        <h2>Keyboard Shortcuts</h2>
      </div>
      <button class="close-btn" onclick={onClose} aria-label="Close">
        <Icon name="x-circle" size={20} />
      </button>
    </header>

    <div class="shortcuts-content">
      {#each shortcuts as group}
        <div class="shortcut-group">
          <h3 class="group-title">{group.category}</h3>
          <div class="shortcut-list">
            {#each group.items as shortcut}
              <div class="shortcut-item">
                <span class="shortcut-description">{shortcut.description}</span>
                <div class="shortcut-keys">
                  {#each shortcut.keys as key, i}
                    {#if i > 0}
                      <span class="key-separator">+</span>
                    {/if}
                    <kbd class="key">{key}</kbd>
                  {/each}
                </div>
              </div>
            {/each}
          </div>
        </div>
      {/each}
    </div>

    <footer class="shortcuts-footer">
      <span class="footer-hint">Press <kbd>Esc</kbd> to close</span>
    </footer>
  </div>
</div>

<style>
  .shortcuts-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.6);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: var(--z-modal);
    animation: fadeIn 0.15s ease;
    backdrop-filter: blur(4px);
    -webkit-backdrop-filter: blur(4px);
  }

  @keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
  }

  .shortcuts-panel {
    width: 90%;
    max-width: 600px;
    max-height: 80vh;
    border-radius: var(--radius-xl);
    overflow: hidden;
    display: flex;
    flex-direction: column;
    animation: scaleIn 0.2s cubic-bezier(0.16, 1, 0.3, 1);
  }

  @keyframes scaleIn {
    from {
      opacity: 0;
      transform: scale(0.95);
    }
    to {
      opacity: 1;
      transform: scale(1);
    }
  }

  .shortcuts-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px 24px;
    border-bottom: 1px solid var(--border-subtle);
  }

  .shortcuts-title {
    display: flex;
    align-items: center;
    gap: 12px;
    color: var(--text-primary);
  }

  .shortcuts-title h2 {
    font-size: var(--text-lg);
    font-weight: 600;
  }

  .close-btn {
    padding: 8px;
    background: transparent;
    border: none;
    border-radius: var(--radius-md);
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .close-btn:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }

  .shortcuts-content {
    flex: 1;
    overflow-y: auto;
    padding: 16px 24px;
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 24px;
  }

  .shortcut-group {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .group-title {
    font-size: var(--text-xs);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-secondary);
  }

  .shortcut-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .shortcut-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 12px;
    background: var(--bg-surface);
    border-radius: var(--radius-md);
    transition: background 0.15s ease;
  }

  .shortcut-item:hover {
    background: var(--bg-hover);
  }

  .shortcut-description {
    font-size: var(--text-sm);
    color: var(--text-primary);
  }

  .shortcut-keys {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .key {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 24px;
    height: 24px;
    padding: 0 8px;
    background: var(--bg-elevated);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-sm);
    font-family: inherit;
    font-size: var(--text-xs);
    font-weight: 500;
    color: var(--text-secondary);
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
  }

  .key-separator {
    font-size: var(--text-xs);
    color: var(--text-tertiary);
  }

  .shortcuts-footer {
    padding: 16px 24px;
    border-top: 1px solid var(--border-subtle);
    text-align: center;
  }

  .footer-hint {
    font-size: var(--text-xs);
    color: var(--text-tertiary);
  }

  .footer-hint kbd {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 20px;
    height: 20px;
    padding: 0 6px;
    background: var(--bg-surface);
    border: 1px solid var(--border-default);
    border-radius: 4px;
    font-family: inherit;
    font-size: 10px;
    margin: 0 4px;
  }

  /* Responsive */
  @media (max-width: 600px) {
    .shortcuts-content {
      grid-template-columns: 1fr;
    }
  }

  /* Reduced motion */
  @media (prefers-reduced-motion: reduce) {
    .shortcuts-overlay,
    .shortcuts-panel {
      animation: none;
    }
  }
</style>
