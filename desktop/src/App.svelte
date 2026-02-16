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
  import { clearSelection, conversationsStore, selectConversation, handleUserActivity } from './lib/stores/conversations.svelte';
  import { initializeTheme } from './lib/stores/theme';
  import { initAnnouncer } from './lib/stores/keyboard';

  // Import design tokens CSS
  import './lib/styles/tokens.css';

  // Check if running in Tauri context
  const isTauri = typeof window !== 'undefined' && '__TAURI__' in window;

  // View state
  let currentView = $state<'messages' | 'dashboard' | 'health' | 'settings' | 'templates' | 'network' | 'chat'>('messages');
  let showSearch = $state(false);
  let showShortcuts = $state(false);
  let showCommandPalette = $state(false);
  let sidebarCollapsed = $state(false);
  let showConversationList = $state(true);
  let isNarrow = $state(false);

  $effect(() => {
    if (isNarrow && conversationsStore.selectedChatId) {
      showConversationList = false;
    }
  });

  // Lazy loaded components with error handling
  type LazyComponent = typeof import('./lib/components/Dashboard.svelte').default;
  
  interface LoadState {
    component: LazyComponent | null;
    loading: boolean;
    error: Error | null;
  }
  
  function createLazyLoader(importFn: () => Promise<{ default: LazyComponent }>) {
    let state: LoadState = $state({ component: null, loading: false, error: null });
    let loaded = false;
    
    return {
      get state() { return state; },
      load: async () => {
        if (loaded) return;
        state.loading = true;
        state.error = null;
        try {
          const module = await importFn();
          state.component = module.default;
          loaded = true;
        } catch (err) {
          state.error = err instanceof Error ? err : new Error(String(err));
          console.error('Failed to load component:', err);
        } finally {
          state.loading = false;
        }
      },
      reset: () => {
        state = { component: null, loading: false, error: null };
        loaded = false;
      }
    };
  }
  
  const lazyLoaders = {
    dashboard: createLazyLoader(() => import('./lib/components/Dashboard.svelte')),
    health: createLazyLoader(() => import('./lib/components/HealthStatus.svelte')),
    settings: createLazyLoader(() => import('./lib/components/Settings.svelte')),
    templates: createLazyLoader(() => import('./lib/components/TemplateBuilder.svelte')),
    network: createLazyLoader(() => import('./lib/components/graph/RelationshipGraph.svelte')),
    chat: createLazyLoader(() => import('./lib/components/ChatView.svelte')),
  };
  
  // Trigger load when view changes
  $effect(() => {
    if (currentView === 'dashboard') lazyLoaders.dashboard.load();
    else if (currentView === 'health') lazyLoaders.health.load();
    else if (currentView === 'settings') lazyLoaders.settings.load();
    else if (currentView === 'templates') lazyLoaders.templates.load();
    else if (currentView === 'network') lazyLoaders.network.load();
    else if (currentView === 'chat') lazyLoaders.chat.load();
  });
  
  // Derived states for templates
  const Dashboard = $derived(lazyLoaders.dashboard.state.component);
  const dashboardLoading = $derived(lazyLoaders.dashboard.state.loading);
  const dashboardError = $derived(lazyLoaders.dashboard.state.error);
  
  const HealthStatus = $derived(lazyLoaders.health.state.component);
  const healthLoading = $derived(lazyLoaders.health.state.loading);
  const healthError = $derived(lazyLoaders.health.state.error);
  
  const Settings = $derived(lazyLoaders.settings.state.component);
  const settingsLoading = $derived(lazyLoaders.settings.state.loading);
  const settingsError = $derived(lazyLoaders.settings.state.error);
  
  const TemplateBuilder = $derived(lazyLoaders.templates.state.component);
  const templatesLoading = $derived(lazyLoaders.templates.state.loading);
  const templatesError = $derived(lazyLoaders.templates.state.error);
  
  const RelationshipGraph = $derived(lazyLoaders.network.state.component);
  const networkLoading = $derived(lazyLoaders.network.state.loading);
  const networkError = $derived(lazyLoaders.network.state.error);
  
  const ChatView = $derived(lazyLoaders.chat.state.component);
  const chatLoading = $derived(lazyLoaders.chat.state.loading);
  const chatError = $derived(lazyLoaders.chat.state.error);

  function goToNextUnread() {
    const unread = conversationsStore.unreadCounts;
    if (unread.size === 0) return;
    // Get first unread chat that's not currently selected
    for (const [chatId] of unread) {
      if (chatId !== conversationsStore.selectedChatId) {
        selectConversation(chatId);
        return;
      }
    }
    // If all unread are the current chat, select the first one
    const firstUnread = unread.keys().next().value;
    if (firstUnread) selectConversation(firstUnread);
  }

  function handleKeydown(event: KeyboardEvent) {
    const isMod = event.metaKey || event.ctrlKey;

    // Cmd+Shift+P to open command palette
    if (isMod && event.shiftKey && event.key.toLowerCase() === 'p' && !showCommandPalette) {
      event.preventDefault();
      showCommandPalette = true;
      return;
    }

    // Cmd+Shift+] to go to next unread
    if (isMod && event.shiftKey && event.key === ']') {
      event.preventDefault();
      goToNextUnread();
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
        case '5':
          event.preventDefault();
          currentView = 'chat';
          return;
      }
    }
  }

  onMount(() => {
    // Initialize theme system
    const cleanupTheme = initializeTheme();

    // Initialize ARIA announcer
    initAnnouncer();

    // Responsive layout detection
    const mediaQuery = window.matchMedia('(max-width: 768px)');
    const handleResize = (e: MediaQueryListEvent | MediaQueryList) => {
      isNarrow = e.matches;
      if (isNarrow) {
        sidebarCollapsed = true;
      }
    };
    handleResize(mediaQuery);
    mediaQuery.addEventListener('change', handleResize);

    // Show window after first paint to avoid white flash
    if (isTauri) {
      requestAnimationFrame(() => {
        void import('@tauri-apps/api/webviewWindow')
          .then(({ getCurrentWebviewWindow }) => getCurrentWebviewWindow().show())
          .catch((e) => console.warn('Failed to show window:', e));
      });
    }

    // Defer socket connection check - not needed for initial conversation list
    // (direct SQLite handles that). Socket is only for real-time push + AI features.
    const apiCheckTimer = setTimeout(() => void checkApiConnection(), 3000);

    let unlisten: (() => void) | null = null;
    let unlistenBackend: (() => void) | null = null;

    // Listen for navigation events from tray menu (Tauri only)
    if (isTauri) {
      void installFrontendLogBridge();

      // Listen for backend ready event to trigger immediate connection
      void (async () => {
        try {
          const { listen } = await import('@tauri-apps/api/event');
          unlistenBackend = await listen('jarvis:backend_ready', () => {
            console.info('[App] Backend ready, connecting...');
            void checkApiConnection();
          });
        } catch (error) {
          console.warn('Failed to listen for backend_ready:', error);
        }
      })();

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
              view === 'network' ||
              view === 'chat'
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

    // Throttled user activity tracking for adaptive polling backoff
    let activityThrottled = false;
    const onUserActivity = () => {
      if (activityThrottled) return;
      activityThrottled = true;
      handleUserActivity();
      setTimeout(() => { activityThrottled = false; }, 5000);
    };
    window.addEventListener('mousemove', onUserActivity);
    window.addEventListener('keydown', onUserActivity);
    window.addEventListener('click', onUserActivity);

    return () => {
      clearTimeout(apiCheckTimer);
      if (unlisten) unlisten();
      if (unlistenBackend) unlistenBackend();
      window.removeEventListener('keydown', handleKeydown);
      window.removeEventListener('mousemove', onUserActivity);
      window.removeEventListener('keydown', onUserActivity);
      window.removeEventListener('click', onUserActivity);
      mediaQuery.removeEventListener('change', handleResize as (e: MediaQueryListEvent) => void);
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
      {#if dashboardLoading}
        <div class="loading-fallback">
          <div class="spinner"></div>
          <span>Loading Dashboard...</span>
        </div>
      {:else if dashboardError}
        <ErrorBoundary context="dashboard" onReset={() => lazyLoaders.dashboard.reset()}>
          <div class="error-fallback">Failed to load Dashboard</div>
        </ErrorBoundary>
      {:else if Dashboard}
        <Dashboard onNavigate={(view) => (currentView = view)} />
      {/if}
    {:else if currentView === 'health'}
      {#if healthLoading}
        <div class="loading-fallback">
          <div class="spinner"></div>
          <span>Loading Health Status...</span>
        </div>
      {:else if healthError}
        <ErrorBoundary context="health" onReset={() => lazyLoaders.health.reset()}>
          <div class="error-fallback">Failed to load Health Status</div>
        </ErrorBoundary>
      {:else if HealthStatus}
        <HealthStatus />
      {/if}
    {:else if currentView === 'settings'}
      {#if settingsLoading}
        <div class="loading-fallback">
          <div class="spinner"></div>
          <span>Loading Settings...</span>
        </div>
      {:else if settingsError}
        <ErrorBoundary context="settings" onReset={() => lazyLoaders.settings.reset()}>
          <div class="error-fallback">Failed to load Settings</div>
        </ErrorBoundary>
      {:else if Settings}
        <Settings />
      {/if}
    {:else if currentView === 'templates'}
      {#if templatesLoading}
        <div class="loading-fallback">
          <div class="spinner"></div>
          <span>Loading Templates...</span>
        </div>
      {:else if templatesError}
        <ErrorBoundary context="templates" onReset={() => lazyLoaders.templates.reset()}>
          <div class="error-fallback">Failed to load Template Builder</div>
        </ErrorBoundary>
      {:else if TemplateBuilder}
        <TemplateBuilder />
      {/if}
    {:else if currentView === 'network'}
      <div class="network-container">
        {#if networkLoading}
          <div class="loading-fallback">
            <div class="spinner"></div>
            <span>Loading Network Graph...</span>
          </div>
        {:else if networkError}
          <ErrorBoundary context="network" onReset={() => lazyLoaders.network.reset()}>
            <div class="error-fallback">Failed to load Network Graph</div>
          </ErrorBoundary>
        {:else if RelationshipGraph}
          <RelationshipGraph />
        {/if}
      </div>
    {:else if currentView === 'chat'}
      {#if chatLoading}
        <div class="loading-fallback">
          <div class="spinner"></div>
          <span>Loading Chat...</span>
        </div>
      {:else if chatError}
        <ErrorBoundary context="chat" onReset={() => lazyLoaders.chat.reset()}>
          <div class="error-fallback">Failed to load Chat</div>
        </ErrorBoundary>
      {:else if ChatView}
        <ChatView />
      {/if}
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
          {#if !isNarrow || showConversationList}
            <div class="conversation-list-wrapper" class:overlay={isNarrow}>
              <ConversationList />
            </div>
          {/if}
          <div class="message-view-wrapper">
            {#if isNarrow && !showConversationList}
              <button class="back-button" onclick={() => showConversationList = true}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="20" height="20">
                  <polyline points="15 18 9 12 15 6"></polyline>
                </svg>
                Back
              </button>
            {/if}
            <MessageView />
          </div>
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

  .conversation-list-wrapper {
    display: contents;
  }

  .conversation-list-wrapper.overlay {
    display: block;
    position: absolute;
    top: 0;
    left: 0;
    bottom: 0;
    z-index: var(--z-dropdown);
    width: 300px;
    background: var(--surface-elevated);
    box-shadow: var(--shadow-lg);
    animation: slideInLeft var(--duration-fast) var(--ease-out);
  }

  .message-view-wrapper {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-width: 0;
    overflow: hidden;
  }

  .back-button {
    display: flex;
    align-items: center;
    gap: var(--space-1);
    padding: var(--space-2) var(--space-3);
    background: var(--surface-elevated);
    border: none;
    border-bottom: 1px solid var(--border-default);
    color: var(--color-primary);
    font-size: var(--text-sm);
    font-weight: var(--font-weight-medium);
    cursor: pointer;
  }

  .back-button:hover {
    background: var(--surface-hover);
  }

  @keyframes slideInLeft {
    from {
      transform: translateX(-100%);
    }
    to {
      transform: translateX(0);
    }
  }

  .loading-fallback {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: var(--space-4);
    color: var(--text-secondary);
    font-size: var(--text-base);
  }

  .loading-fallback .spinner {
    width: 32px;
    height: 32px;
    border: 3px solid var(--border-default);
    border-top-color: var(--color-primary);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  @media (max-width: 768px) {
    .messages-content {
      position: relative;
    }
  }
</style>
