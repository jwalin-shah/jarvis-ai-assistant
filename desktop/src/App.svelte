<script lang="ts">
  import { onMount } from "svelte";
  import Sidebar from "./lib/components/Sidebar.svelte";
  import ConversationList from "./lib/components/ConversationList.svelte";
  import MessageView from "./lib/components/MessageView.svelte";
  import Dashboard from "./lib/components/Dashboard.svelte";
  import HealthStatus from "./lib/components/HealthStatus.svelte";
  import Settings from "./lib/components/Settings.svelte";
  import TemplateBuilder from "./lib/components/TemplateBuilder.svelte";
  import RelationshipGraph from "./lib/components/graph/RelationshipGraph.svelte";
  import GlobalSearch from "./lib/components/GlobalSearch.svelte";
  import ErrorBoundary from "./lib/components/ErrorBoundary.svelte";
  import KeyboardShortcuts from "./lib/components/KeyboardShortcuts.svelte";
  import CommandPalette from "./lib/components/CommandPalette.svelte";
  import Toast from "./lib/components/Toast.svelte";
  import { checkApiConnection } from "./lib/stores/health";
  import { clearSelection } from "./lib/stores/conversations";
  import { initializeTheme } from "./lib/stores/theme";
  import { initAnnouncer } from "./lib/stores/keyboard";

  // Check if running in Tauri context
  const isTauri = typeof window !== "undefined" && "__TAURI__" in window;

  let currentView = $state<"messages" | "dashboard" | "health" | "settings" | "templates" | "network">("messages");
  let showSearch = $state(false);
  let showShortcuts = $state(false);
  let showCommandPalette = $state(false);
  let sidebarCollapsed = $state(false);

  function handleKeydown(event: KeyboardEvent) {
    const isMod = event.metaKey || event.ctrlKey;

    // Cmd+Shift+P to open command palette
    if (isMod && event.shiftKey && event.key.toLowerCase() === "p" && !showCommandPalette) {
      event.preventDefault();
      showCommandPalette = true;
      return;
    }

    // Cmd+K or Cmd+F to open search (when search is not open)
    if (isMod && (event.key === "k" || event.key === "f") && !showSearch) {
      event.preventDefault();
      showSearch = true;
      return;
    }

    // Cmd+/ to show keyboard shortcuts
    if (isMod && event.key === "/" && !showShortcuts) {
      event.preventDefault();
      showShortcuts = true;
      return;
    }

    // Cmd+, to open settings
    if (isMod && event.key === ",") {
      event.preventDefault();
      currentView = "settings";
      return;
    }

    // Cmd+1/2/3 for navigation
    if (isMod && event.key === "1") {
      event.preventDefault();
      currentView = "dashboard";
      return;
    }
    if (isMod && event.key === "2") {
      event.preventDefault();
      currentView = "messages";
      return;
    }
    if (isMod && event.key === "3") {
      event.preventDefault();
      currentView = "templates";
      return;
    }
    if (isMod && event.key === "4") {
      event.preventDefault();
      currentView = "network";
      return;
    }
  }

  onMount(() => {
    // Initialize theme system
    const cleanupTheme = initializeTheme();

    // Initialize ARIA announcer for keyboard navigation
    initAnnouncer();

    // Check API connection on start
    void checkApiConnection();

    let unlisten: (() => void) | null = null;

    // Listen for navigation events from tray menu (only in Tauri context)
    if (isTauri) {
      void installFrontendLogBridge();
      void (async () => {
        try {
          const { listen } = await import("@tauri-apps/api/event");
          unlisten = await listen<string>("navigate", (event) => {
            if (
              event.payload === "health" ||
              event.payload === "dashboard" ||
              event.payload === "messages" ||
              event.payload === "settings" ||
              event.payload === "templates" ||
              event.payload === "network"
            ) {
              currentView = event.payload;
              if (event.payload !== "messages") {
                clearSelection();
              }
            }
          });
        } catch (error) {
          console.warn("Failed to set up Tauri event listener:", error);
        }
      })();
    }

    // Add keyboard listener for search
    window.addEventListener("keydown", handleKeydown);

    // Cleanup on unmount - consolidate all cleanup in one place
    return () => {
      if (unlisten) unlisten();
      window.removeEventListener("keydown", handleKeydown);
      cleanupTheme();
    };
  });

  function openSearch() {
    showSearch = true;
  }

  function closeSearch() {
    showSearch = false;
  }

  function closeShortcuts() {
    showShortcuts = false;
  }

  function openCommandPalette() {
    showCommandPalette = true;
  }

  function closeCommandPalette() {
    showCommandPalette = false;
  }

  function serializeLogArg(arg: unknown): string {
    if (arg instanceof Error) {
      return `${arg.name}: ${arg.message}${arg.stack ? `\n${arg.stack}` : ""}`;
    }
    if (typeof arg === "string") return arg;
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
      const { invoke } = await import("@tauri-apps/api/core");
      const levels = ["log", "info", "warn", "error", "debug"] as const;

      const forward = async (level: (typeof levels)[number], args: unknown[]) => {
        const message = args.map(serializeLogArg).join(" ");
        try {
          await invoke("frontend_log", { level, message });
        } catch {
          // Avoid recursive logging if invoke fails.
        }
      };

      for (const level of levels) {
        const original = console[level].bind(console);
        console[level] = (...args: unknown[]) => {
          original(...args);
          void forward(level, args);
        };
      }

      window.addEventListener("error", (event) => {
        void forward("error", [
          "window.error",
          event.message,
          event.filename,
          `line:${event.lineno}`,
          `col:${event.colno}`,
        ]);
      });

      window.addEventListener("unhandledrejection", (event) => {
        void forward("error", ["unhandledrejection", event.reason]);
      });
    } catch (error) {
      console.warn("Failed to install frontend log bridge:", error);
    }
  }
</script>

<main class="app">
  <Sidebar bind:currentView bind:collapsed={sidebarCollapsed} />

  <ErrorBoundary>
    {#if currentView === "dashboard"}
      <Dashboard onNavigate={(view) => (currentView = view)} />
    {:else if currentView === "health"}
      <HealthStatus />
    {:else if currentView === "settings"}
      <Settings />
    {:else if currentView === "templates"}
      <TemplateBuilder />
    {:else if currentView === "network"}
      <div class="network-container">
        <RelationshipGraph />
      </div>
    {:else}
      <div class="messages-container">
        <div class="search-bar">
          <button class="search-button" onclick={openSearch} title="Search messages (Cmd+K)">
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
  <GlobalSearch onClose={closeSearch} />
{/if}

{#if showShortcuts}
  <KeyboardShortcuts onClose={closeShortcuts} />
{/if}

{#if showCommandPalette}
  <CommandPalette
    onClose={closeCommandPalette}
    onNavigate={(view) => currentView = view}
    onOpenSearch={openSearch}
    onOpenShortcuts={() => showShortcuts = true}
  />
{/if}

<!-- Global toast notifications -->
<Toast />

<style>
  :global(*) {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }

  :global(body) {
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Segoe UI', Roboto,
      Helvetica, Arial, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    overflow: hidden;
    font-size: var(--text-base);
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }

  /* Typography - headings with tighter tracking */
  :global(h1, h2, h3, h4, h5, h6) {
    letter-spacing: -0.02em;
    font-weight: 600;
  }

  /* Glassmorphism utility class */
  :global(.glass) {
    background: rgba(28, 28, 30, 0.8);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--border-subtle);
  }

  :global(:root.light .glass) {
    background: rgba(255, 255, 255, 0.8);
  }

  /* Focus ring for accessibility */
  :global(*:focus-visible) {
    outline: none;
    box-shadow: 0 0 0 3px rgba(0, 122, 255, 0.3);
  }

  /* Button micro-interactions */
  :global(button) {
    transition: transform var(--duration-fast) var(--ease-out),
                box-shadow var(--duration-fast) var(--ease-out),
                background var(--duration-fast) var(--ease-out),
                border-color var(--duration-fast) var(--ease-out);
  }

  :global(button:not(:disabled):hover) {
    transform: translateY(-1px);
  }

  :global(button:not(:disabled):active) {
    transform: scale(0.98);
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

  /* Screen reader only - visually hidden but accessible */
  :global(.sr-only) {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
  }

  :global(:root) {
    /* Colors */
    --color-primary: #007AFF;
    --color-success: #34C759;
    --color-warning: #FF9500;
    --color-error: #FF3B30;

    /* Spacing - 4px base scale */
    --space-1: 4px;
    --space-2: 8px;
    --space-3: 12px;
    --space-4: 16px;
    --space-5: 20px;
    --space-6: 24px;
    --space-7: 28px;
    --space-8: 32px;
    --space-9: 36px;
    --space-10: 40px;

    /* Typography */
    --text-xs: 11px;
    --text-sm: 13px;
    --text-base: 15px;
    --text-lg: 17px;
    --text-xl: 20px;
    --text-2xl: 24px;

    /* Border Radius */
    --radius-sm: 6px;
    --radius-md: 8px;
    --radius-lg: 12px;
    --radius-xl: 16px;
    --radius-2xl: 20px;
    --radius-full: 9999px;

    /* Animation */
    --duration-fast: 150ms;
    --duration-normal: 200ms;
    --duration-slow: 300ms;
    --ease-out: cubic-bezier(0.33, 1, 0.68, 1);
    --ease-in-out: cubic-bezier(0.65, 0, 0.35, 1);

    /* Shadows */
    --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.1);
    --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.15);
    --shadow-lg: 0 8px 24px rgba(0, 0, 0, 0.2);

    /* Dark theme (default) */
    --bg-base: #0A0A0A;
    --bg-elevated: #141414;
    --bg-surface: #1C1C1E;
    --text-primary: #FFFFFF;
    --text-secondary: rgba(255, 255, 255, 0.6);
    --text-tertiary: rgba(255, 255, 255, 0.4);
    --border-subtle: rgba(255, 255, 255, 0.08);
    --border-default: rgba(255, 255, 255, 0.12);
    --bubble-me: var(--color-primary);
    --bubble-other: var(--bg-surface);

    /* Legacy aliases for compatibility */
    --bg-primary: var(--bg-base);
    --bg-secondary: var(--bg-elevated);
    --bg-hover: #3a3a3c;
    --bg-active: #48484a;
    --bg-bubble-me: var(--bubble-me);
    --bg-bubble-other: var(--bubble-other);
    --border-color: var(--border-default);
    --accent-color: var(--color-primary);
    --group-color: #5856d6;
    --error-color: var(--color-error);
  }

  /* Light theme */
  :global(:root.light) {
    --bg-base: #FFFFFF;
    --bg-elevated: #F5F5F7;
    --bg-surface: #E5E5EA;
    --text-primary: #1D1D1F;
    --text-secondary: rgba(0, 0, 0, 0.55);
    --text-tertiary: rgba(0, 0, 0, 0.35);
    --border-subtle: rgba(0, 0, 0, 0.06);
    --border-default: rgba(0, 0, 0, 0.1);
    --bubble-me: var(--color-primary);
    --bubble-other: var(--bg-surface);

    /* Legacy aliases for compatibility */
    --bg-primary: var(--bg-base);
    --bg-secondary: var(--bg-elevated);
    --bg-hover: #E5E5EA;
    --bg-active: #D1D1D6;
    --bg-bubble-me: var(--bubble-me);
    --bg-bubble-other: var(--bubble-other);
    --border-color: var(--border-default);
  }

  /* Reduced motion - disable animations and transitions */
  :global(:root.reduce-motion),
  :global(:root.reduce-motion *) {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }

  @media (prefers-reduced-motion: reduce) {
    :global(*) {
      animation-duration: 0.01ms !important;
      animation-iteration-count: 1 !important;
      transition-duration: 0.01ms !important;
      scroll-behavior: auto !important;
    }
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
    padding: 8px 12px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-color);
  }

  .search-button {
    width: 100%;
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    color: var(--text-secondary);
    font-size: 14px;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .search-button:hover {
    background: var(--bg-hover);
    border-color: var(--accent-color);
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
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    font-family: inherit;
    font-size: 11px;
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
    padding: 16px;
  }
</style>
