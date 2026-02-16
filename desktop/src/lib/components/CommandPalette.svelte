<script lang="ts">
  import { tick, onMount } from 'svelte';
  import Icon, { type IconName } from './Icon.svelte';
  import { conversationsStore, selectConversation } from '../stores/conversations.svelte';

  interface Command {
    id: string;
    label: string;
    description?: string;
    shortcut?: string[];
    icon?: IconName;
    category: string;
    action: () => void;
  }

  interface Props {
    onClose: () => void;
    onNavigate: (
      view: 'messages' | 'dashboard' | 'health' | 'settings' | 'templates' | 'network' | 'chat'
    ) => void;
    onOpenSearch: () => void;
    onOpenShortcuts: () => void;
  }

  let { onClose, onNavigate, onOpenSearch, onOpenShortcuts }: Props = $props();

  let searchQuery = $state('');
  let selectedIndex = $state(0);
  let inputRef = $state<HTMLInputElement | null>(null);

  // Define all available commands
  const commands: Command[] = [
    // Navigation
    {
      id: 'nav-messages',
      label: 'Go to Messages',
      description: 'View your conversations',
      shortcut: ['⌘', '2'],
      icon: 'message-circle',
      category: 'Navigation',
      action: () => {
        onNavigate('messages');
        onClose();
      },
    },
    {
      id: 'nav-dashboard',
      label: 'Go to Dashboard',
      description: 'View analytics and insights',
      shortcut: ['⌘', '1'],
      icon: 'bar-chart-2',
      category: 'Navigation',
      action: () => {
        onNavigate('dashboard');
        onClose();
      },
    },
    {
      id: 'nav-chat',
      label: 'Go to Chat',
      description: 'Chat directly with the local AI',
      shortcut: ['⌘', '5'],
      icon: 'message-circle',
      category: 'Navigation',
      action: () => {
        onNavigate('chat');
        onClose();
      },
    },
    {
      id: 'nav-templates',
      label: 'Go to Templates',
      description: 'Manage reply templates',
      shortcut: ['⌘', '3'],
      icon: 'copy',
      category: 'Navigation',
      action: () => {
        onNavigate('templates');
        onClose();
      },
    },
    {
      id: 'nav-settings',
      label: 'Go to Settings',
      description: 'Configure preferences',
      shortcut: ['⌘', ','],
      icon: 'settings',
      category: 'Navigation',
      action: () => {
        onNavigate('settings');
        onClose();
      },
    },
    {
      id: 'nav-health',
      label: 'Go to Health Status',
      description: 'Check system health',
      icon: 'alert-circle',
      category: 'Navigation',
      action: () => {
        onNavigate('health');
        onClose();
      },
    },
    // Actions
    {
      id: 'action-search',
      label: 'Search Messages',
      description: 'Find messages across all conversations',
      shortcut: ['⌘', 'K'],
      icon: 'search',
      category: 'Actions',
      action: () => {
        onClose();
        onOpenSearch();
      },
    },
    {
      id: 'action-shortcuts',
      label: 'Keyboard Shortcuts',
      description: 'View all keyboard shortcuts',
      shortcut: ['⌘', '/'],
      icon: 'settings',
      category: 'Actions',
      action: () => {
        onClose();
        onOpenShortcuts();
      },
    },
    {
      id: 'action-next-unread',
      label: 'Next Unread Conversation',
      description: 'Jump to the next conversation with unread messages',
      shortcut: ['⌘', '⇧', ']'],
      icon: 'message-circle',
      category: 'Actions',
      action: () => {
        for (const [chatId] of conversationsStore.unreadCounts) {
          if (chatId !== conversationsStore.selectedChatId) {
            selectConversation(chatId);
            onClose();
            return;
          }
        }
        onClose();
      },
    },
    {
      id: 'action-refresh',
      label: 'Refresh',
      description: 'Reload current view',
      shortcut: ['⌘', 'R'],
      icon: 'refresh-cw',
      category: 'Actions',
      action: () => {
        location.reload();
      },
    },
    // Theme
    {
      id: 'theme-dark',
      label: 'Switch to Dark Theme',
      category: 'Theme',
      action: () => {
        import('../stores/theme').then(({ setTheme }) => setTheme('dark'));
        onClose();
      },
    },
    {
      id: 'theme-light',
      label: 'Switch to Light Theme',
      category: 'Theme',
      action: () => {
        import('../stores/theme').then(({ setTheme }) => setTheme('light'));
        onClose();
      },
    },
    {
      id: 'theme-system',
      label: 'Use System Theme',
      category: 'Theme',
      action: () => {
        import('../stores/theme').then(({ setTheme }) => setTheme('system'));
        onClose();
      },
    },
  ];

  // Filter commands based on search query
  let filteredCommands = $derived.by(() => {
    if (!searchQuery.trim()) return commands;

    const query = searchQuery.toLowerCase();
    return commands.filter(
      (cmd) =>
        cmd.label.toLowerCase().includes(query) ||
        cmd.description?.toLowerCase().includes(query) ||
        cmd.category.toLowerCase().includes(query)
    );
  });

  // Group commands by category
  let groupedCommands = $derived.by(() => {
    const groups: Record<string, Command[]> = {};
    for (const cmd of filteredCommands) {
      if (!groups[cmd.category]) {
        groups[cmd.category] = [];
      }
      groups[cmd.category]!.push(cmd);
    }
    return groups;
  });

  // Reset selection when search changes
  $effect(() => {
    searchQuery;
    selectedIndex = 0;
  });

  // Focus input on mount
  onMount(() => {
    tick().then(() => {
      inputRef?.focus();
    });
  });

  function handleKeydown(event: KeyboardEvent) {
    const cmds = filteredCommands;

    switch (event.key) {
      case 'ArrowDown':
        event.preventDefault();
        selectedIndex = Math.min(selectedIndex + 1, cmds.length - 1);
        scrollToSelected();
        break;
      case 'ArrowUp':
        event.preventDefault();
        selectedIndex = Math.max(selectedIndex - 1, 0);
        scrollToSelected();
        break;
      case 'Enter':
        event.preventDefault();
        if (cmds[selectedIndex]) {
          cmds[selectedIndex]!.action();
        }
        break;
      case 'Escape':
        event.preventDefault();
        onClose();
        break;
      case 'Tab':
        event.preventDefault();
        if (event.shiftKey) {
          selectedIndex = Math.max(selectedIndex - 1, 0);
        } else {
          selectedIndex = Math.min(selectedIndex + 1, cmds.length - 1);
        }
        scrollToSelected();
        break;
    }
  }

  let paletteContentRef = $state<HTMLDivElement | null>(null);

  function scrollToSelected() {
    tick().then(() => {
      const selected = document.querySelector('.command-item.selected') as HTMLElement | null;
      if (!selected || !paletteContentRef) return;
      const containerRect = paletteContentRef.getBoundingClientRect();
      const itemRect = selected.getBoundingClientRect();
      if (itemRect.bottom > containerRect.bottom) {
        paletteContentRef.scrollTop += itemRect.bottom - containerRect.bottom;
      } else if (itemRect.top < containerRect.top) {
        paletteContentRef.scrollTop -= containerRect.top - itemRect.top;
      }
    });
  }

  function getCommandIndex(cmd: Command): number {
    return filteredCommands.findIndex((c) => c.id === cmd.id);
  }

  function handleOverlayClick(event: MouseEvent) {
    if (event.target === event.currentTarget) {
      onClose();
    }
  }

  function handleOverlayKeydown(event: KeyboardEvent) {
    // Don't close if typing in the input field
    if (event.target === inputRef) return;

    if (event.key === 'Escape' || event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      onClose();
    }
  }
</script>

<svelte:window onkeydown={handleKeydown} />

<div
  class="palette-overlay"
  onclick={handleOverlayClick}
  onkeydown={handleOverlayKeydown}
  role="button"
  aria-label="Close command palette"
  tabindex="0"
>
  <div
    class="palette glass"
    role="dialog"
    aria-label="Command Palette"
    aria-modal="true"
    tabindex="-1"
  >
    <div class="palette-header">
      <Icon name="search" size={18} />
      <input
        bind:this={inputRef}
        type="text"
        bind:value={searchQuery}
        placeholder="Type a command or search..."
        class="palette-input"
        aria-label="Search commands"
        autocomplete="off"
        spellcheck="false"
      />
      <kbd class="esc-hint">esc</kbd>
    </div>

    <div class="palette-content" bind:this={paletteContentRef}>
      {#if filteredCommands.length === 0}
        <div class="no-results">
          <p>No commands found for "{searchQuery}"</p>
        </div>
      {:else}
        {#each Object.entries(groupedCommands) as [category, cmds]}
          <div class="command-group">
            <div class="group-label">{category}</div>
            {#each cmds as cmd}
              {@const index = getCommandIndex(cmd)}
              <button
                class="command-item"
                class:selected={index === selectedIndex}
                onclick={() => cmd.action()}
                onmouseenter={() => (selectedIndex = index)}
                role="option"
                aria-selected={index === selectedIndex}
              >
                {#if cmd.icon}
                  <span class="command-icon">
                    <Icon name={cmd.icon} size={16} />
                  </span>
                {/if}
                <div class="command-info">
                  <span class="command-label">{cmd.label}</span>
                  {#if cmd.description}
                    <span class="command-description">{cmd.description}</span>
                  {/if}
                </div>
                {#if cmd.shortcut}
                  <div class="command-shortcut">
                    {#each cmd.shortcut as key}
                      <kbd>{key}</kbd>
                    {/each}
                  </div>
                {/if}
              </button>
            {/each}
          </div>
        {/each}
      {/if}
    </div>

    <div class="palette-footer">
      <span class="hint"><kbd>↑</kbd><kbd>↓</kbd> to navigate</span>
      <span class="hint"><kbd>↵</kbd> to select</span>
      <span class="hint"><kbd>esc</kbd> to close</span>
    </div>
  </div>
</div>

<style>
  .palette-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.5);
    display: flex;
    align-items: flex-start;
    justify-content: center;
    padding-top: 15vh;
    z-index: var(--z-modal);
    animation: fadeIn 0.1s ease;
  }

  @keyframes fadeIn {
    from {
      opacity: 0;
    }
    to {
      opacity: 1;
    }
  }

  .palette {
    width: 90%;
    max-width: 560px;
    max-height: 60vh;
    border-radius: var(--radius-xl);
    overflow: hidden;
    display: flex;
    flex-direction: column;
    animation: scaleIn 0.15s cubic-bezier(0.16, 1, 0.3, 1);
  }

  @keyframes scaleIn {
    from {
      opacity: 0;
      transform: scale(0.96) translateY(-8px);
    }
    to {
      opacity: 1;
      transform: scale(1) translateY(0);
    }
  }

  .palette-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 16px;
    border-bottom: 1px solid var(--border-subtle);
    color: var(--text-secondary);
  }

  .palette-input {
    flex: 1;
    background: transparent;
    border: none;
    outline: none;
    font-size: var(--text-base);
    color: var(--text-primary);
  }

  .palette-input::placeholder {
    color: var(--text-tertiary);
  }

  .esc-hint {
    padding: 4px 8px;
    background: var(--bg-surface);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-sm);
    font-size: var(--text-xs);
    color: var(--text-tertiary);
  }

  .palette-content {
    flex: 1;
    overflow-y: auto;
    padding: 8px;
  }

  .no-results {
    padding: 32px;
    text-align: center;
    color: var(--text-secondary);
  }

  .command-group {
    margin-bottom: 8px;
  }

  .group-label {
    padding: 8px 12px 4px;
    font-size: var(--text-xs);
    font-weight: 600;
    color: var(--text-tertiary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .command-item {
    width: 100%;
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 12px;
    background: transparent;
    border: none;
    border-radius: var(--radius-md);
    cursor: pointer;
    text-align: left;
    transition: background 0.1s ease;
  }

  .command-item:hover,
  .command-item.selected {
    background: var(--bg-hover);
  }

  .command-item.selected {
    background: rgba(var(--color-primary-rgb, 0, 122, 255), 0.15);
  }

  .command-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    background: var(--bg-surface);
    border-radius: var(--radius-sm);
    color: var(--text-secondary);
  }

  .command-item.selected .command-icon {
    background: var(--color-primary);
    color: white;
  }

  .command-info {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .command-label {
    font-size: var(--text-sm);
    font-weight: 500;
    color: var(--text-primary);
  }

  .command-description {
    font-size: var(--text-xs);
    color: var(--text-secondary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .command-shortcut {
    display: flex;
    gap: 4px;
  }

  .command-shortcut kbd {
    min-width: 22px;
    height: 22px;
    padding: 0 6px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: var(--bg-surface);
    border: 1px solid var(--border-default);
    border-radius: 4px;
    font-family: inherit;
    font-size: 11px;
    color: var(--text-secondary);
  }

  .palette-footer {
    display: flex;
    gap: 16px;
    padding: 10px 16px;
    border-top: 1px solid var(--border-subtle);
    background: var(--bg-surface);
  }

  .hint {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: var(--text-xs);
    color: var(--text-tertiary);
  }

  .hint kbd {
    min-width: 18px;
    height: 18px;
    padding: 0 4px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: var(--bg-elevated);
    border: 1px solid var(--border-default);
    border-radius: 3px;
    font-size: 10px;
  }

  /* Reduced motion */
  @media (prefers-reduced-motion: reduce) {
    .palette-overlay,
    .palette {
      animation: none;
    }
  }
</style>
