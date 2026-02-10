<script lang="ts">
  import { onMount } from 'svelte';
  import Sidebar from './lib/components/Sidebar.svelte';
  import ConversationList from './lib/components/ConversationList.svelte';
  import MessageView from './lib/components/MessageView.svelte';
  import GlobalSearch from './lib/components/GlobalSearch.svelte';
  import ErrorBoundary from './lib/components/ErrorBoundary.svelte';
  import KeyboardShortcuts from './lib/components/KeyboardShortcuts.svelte';
  import CommandPalette from './lib/components/CommandPalette.svelte';
  import Toast from './lib/components/Toast.svelte';
  import { checkApiConnection } from './lib/stores/health';
  import { clearSelection } from './lib/stores/conversations';
  import { initializeTheme } from './lib/stores/theme';
  import { initAnnouncer } from './lib/stores/keyboard';

  // Import design tokens CSS
  import './lib/styles/tokens.css';

  // Check if running in Tauri context
  const isTauri = typeof window !== 'undefined' && '__TAURI__' in window;

  // View state
  let currentView = $state<'messages' | 'dashboard' | 'health' | 'settings' | 'templates' | 'network'>('messages');
  let showSearch = $state(false);
  let showShortcuts = $state(false);
  let showCommandPalette = $state(false);
  let sidebarCollapsed = $state(false);

  // Lazy loaded components
  const Dashboard = $derived(
    currentView === 'dashboard'
      ? import('./lib/components/Dashboard.svelte').then((m) => m.default)
      : null
  );

  const HealthStatus = $derived(
    currentView === 'health'
      ? import('./lib/components/HealthStatus.svelte').then((m) => m.default)
      : null
  );

  const Settings = $derived(
    currentView === 'settings'
      ? import('./lib/components/Settings.svelte').then((m) => m.default)
      : null
  );

  const TemplateBuilder = $derived(
    currentView === 'templates'
      ? import('./lib/components/TemplateBuilder.svelte').then((m) => m.default)
      : null
  );

  const RelationshipGraph = $derived(
    currentView === 'network'
      ? import('./lib/components/graph/RelationshipGraph.svelte').then((m) => m.default)
      : null
  );

  function handleKeydown(event: KeyboardEvent) {
    const isMod = event.metaKey || event.ctrlKey;

    // Cmd+Shift+P to open command palette
    if (isMod && event.shiftKey && event.key.toLowerCase() === 'p' && !showCommandPalette) {
      event.preventDefault();
      showCommandPalette = true;
      return;
    }

    // Cmd+K or Cmd+F to open search
    if (isMod && (event.key === 'k' || event.key === 'f') && !showSearch) {
      event.preventDefault();
      showSearch = true;
      return;
    }

    // Cmd+/ to show keyboard shortcuts
    if (isMod && event.key === '/' && !showShortcuts) {
      event.preventDefault();
      showShortcuts = true;
      return;
    }

    // Cmd+, to open settings
    if (isMod && event.key === ',') {
      event.preventDefault();
      currentView = 'settings';
      return;
    }

    // Cmd+1/2/3/4 for navigation
    if (isMod) {
      switch (event.key) {
        case '1':
          event.preventDefault();
          currentView = 'dashboard';
          return;
        case '2':
          event.preventDefault();
          currentView = 'messages';
          return;
        case '3':
          event.preventDefault();
          currentView = 'templates';
          return;
        case '4':
          event.preventDefault();
          currentView = 'network';
          return;
      }
    }
  }

  onMount(() => {
    // Initialize theme system
    const cleanupTheme = initializeTheme();

    // Initialize ARIA announcer
    initAnnouncer();

    // Check API connection on start
    void checkApiConnection();

    let unlisten: (() => void) | null = null;

    // Listen for navigation events from tray menu (Tauri only)
    if (isTauri) {
      void installFrontendLogBridge();
      void (async () => {
        try {
          const { listen } = await import('@tauri-apps/api/event');
          unlisten = await listen<string>('navigate', (event) => {
            const view = event.payload;
            if (
              view === 'health' ||
              view === 'dashboard' ||
              view === 'messages' ||
              view === 'settings' ||
              view === 'templates' ||
              view === 'network'
            ) {
              currentView = view;
              if (view !== 'messages') {
                clearSelection();
              }
            }
          });
        } catch (error) {
          console.warn('Failed to set up Tauri event listener:', error);
        }
      })();
    }

    window.addEventListener('keydown', handleKeydown);

    return () => {
      if (unlisten) unlisten();
      window.removeEventListener('keydown', handleKeydown);
      cleanupTheme();
    };
  });

  function serializeLogArg(arg: unknown): string {
    if (arg instanceof Error) {
      return `${arg.name}: ${arg.message}${arg.stack ? `\n${arg.stack}` : ''}`;
    }
    if (typeof arg === 'string') return arg;
    try {
      return JSON.stringify(arg);
    } catch {
      return String(arg);
    }
  }

  async function installFrontendLogBridge(): Promise<void> {
    if (!isTauri) return;

    const windowObj = window as unknown as Record<string, unknown>;
    if (windowObj.__JARVIS_CONSOLE_BRIDGED__) return;
    windowObj.__JARVIS_CONSOLE_BRIDGED__ = true;

    try {
      const { invoke } = await import('@tauri-apps/api/core');
      const levels = ['log', 'info', 'warn', 'error', 'debug'] as const;

      const forward = async (level: (typeof levels)[number], args: unknown[]) => {
        const message = args.map(serializeLogArg).join(' ');
        try {
          await invoke('frontend_log', { level, message });
        } catch {
          // Avoid recursive logging
        }
      };

      for (const level of levels) {
        const original = console[level].bind(console);
        console[level] = (...args: unknown[]) => {
          original(...args);
          void forward(level, args);
        };
      }

      window.addEventListener('error', (event) => {
        void forward('error', [
          'window.error',
          event.message,
          event.filename,
          `line:${event.lineno}`,
          `col:${event.colno}`,
        ]);
      });

      window.addEventListener('unhandledrejection', (event) => {
        void forward('error', ['unhandledrejection', event.reason]);
      });
    } catch (error) {
      console.warn('Failed to install frontend log bridge:', error);
    }
  }
</script>

<main class="app">
  <Sidebar bind:currentView bind:collapsed={sidebarCollapsed} />

  <ErrorBoundary>
    {#if currentView === 'dashboard'}
      {#await Dashboard then Component}
        <Component onNavigate={(view) => (currentView = view)} />
      {:catch}
        <div class="error-fallback">Failed to load Dashboard</div>
      {/await}
    {:else if currentView === 'health'}
      {#await HealthStatus then Component}
        <Component />
      {:catch}
        <div class="error-fallback">Failed to load Health Status</div>
      {/await}
    {:else if currentView === 'settings'}
      {#await Settings then Component}
        <Component />
      {:catch}
        <div class="error-fallback">Failed to load Settings</div>
      {/await}
    {:else if currentView === 'templates'}
      {#await TemplateBuilder then Component}
        <Component />
      {:catch}
        <div class="error-fallback">Failed to load Template Builder</div>
      {/await}
    {:else if currentView === 'network'}
      <div class="network-container">
        {#await RelationshipGraph then Component}
          <Component />
        {:catch}
          <div class="error-fallback">Failed to load Network Graph</div>
        {/await}
      </div>
    {:else}
      <div class="messages-container">
        <div class="search-bar">
          <button class="search-button" onclick={() => (showSearch = true)} title="Search messages (Cmd+K)">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="11" cy="11" r="8"></circle>
              <path d="m21 21-4.35-4.35"></path>
            </svg>
            <span>Search messages...</span>
            <kbd>⌘K</kbd>
          </button>
        </div>
        <div class="messages-content">
          <ConversationList />
          <MessageView />
        </div>
      </div>
    {/if}
  </ErrorBoundary>
</main>

{#if showSearch}
  <GlobalSearch onClose={() => (showSearch = false)} />
{/if}

{#if showShortcuts}
  <KeyboardShortcuts onClose={() => (showShortcuts = false)} />
{/if}

{#if showCommandPalette}
  <CommandPalette
    onClose={() => (showCommandPalette = false)}
    onNavigate={(view) => (currentView = view)}
    onOpenSearch={() => (showSearch = true)}
    onOpenShortcuts={() => (showShortcuts = true)}
  />
{/if}

<Toast />

<style>
  :global(*) {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }

  :global(body) {
    font-family: var(--font-family-sans);
    background: var(--surface-base);
    color: var(--text-primary);
    overflow: hidden;
    font-size: var(--text-base);
    line-height: var(--line-height-normal);
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }

  /* Typography - headings with tighter tracking */
  :global(h1, h2, h3, h4, h5, h6) {
    letter-spacing: var(--letter-spacing-tight);
    font-weight: var(--font-weight-semibold);
  }

  /* Smooth scrollbars */
  :global(::-webkit-scrollbar) {
    width: 8px;
    height: 8px;
  }

  :global(::-webkit-scrollbar-track) {
    background: transparent;
  }

  :global(::-webkit-scrollbar-thumb) {
    background: var(--border-default);
    border-radius: var(--radius-full);
  }

  :global(::-webkit-scrollbar-thumb:hover) {
    background: var(--text-tertiary);
  }

  .app {
    display: flex;
    height: 100vh;
    width: 100vw;
  }

  .messages-container {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .search-bar {
    padding: var(--space-2) var(--space-3);
    background: var(--surface-elevated);
    border-bottom: 1px solid var(--border-default);
  }

  .search-button {
    width: 100%;
    display: flex;
    align-items: center;
    gap: var(--space-2);
    padding: var(--space-2) var(--space-3);
    background: var(--surface-base);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-md);
    color: var(--text-secondary);
    font-size: var(--text-base);
    cursor: pointer;
    transition: all var(--duration-fast) var(--ease-out);
  }

  .search-button:hover {
    background: var(--surface-hover);
    border-color: var(--color-primary);
    color: var(--text-primary);
  }

  .search-button svg {
    width: 16px;
    height: 16px;
    flex-shrink: 0;
  }

  .search-button span {
    flex: 1;
    text-align: left;
  }

  .search-button kbd {
    padding: 2px 6px;
    background: var(--surface-elevated);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-sm);
    font-family: inherit;
    font-size: var(--text-xs);
  }

  .messages-content {
    flex: 1;
    display: flex;
    overflow: hidden;
  }

  .network-container {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    padding: var(--space-4);
  }

  .error-fallback {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--color-error);
    font-size: var(--text-lg);
  }
</style>
