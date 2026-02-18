<script lang="ts">
  import { onMount, tick } from 'svelte';
  import { slide } from 'svelte/transition';
  import {
    conversationsStore,
    selectConversation,
    initializePolling,
    togglePinChat,
    toggleArchiveChat,
  } from '../stores/conversations.svelte';
  import type { Conversation } from '../types';
  import ConversationSkeleton from './ConversationSkeleton.svelte';
  import {
    activeZone,
    setActiveZone,
    conversationIndex,
    setConversationIndex,
    announce,
  } from '../stores/keyboard';
  import { getApiBaseUrl } from '../config/runtime';
  import { formatConversationDate } from '../utils/date';
  import { getNavAction, isTypingInInput } from '../utils/keyboard-nav';
  import { LRUCache } from '../utils/lru-cache';
  import { jarvis } from '../socket';
  import type { ConnectionInfo } from '../socket';
  import { formatParticipant } from '../db';

  // Track focused conversation for keyboard navigation
  let focusedIndex = $state(-1);
  let itemRefs = $state<HTMLButtonElement[]>([]);

  // Sync focusedIndex with store
  $effect(() => {
    focusedIndex = $conversationIndex;
  });

  // Context menu state
  let contextMenu = $state<{ x: number; y: number; chatId: string } | null>(null);

  // Connection status (UX-01)
  let connectionInfo = $state<ConnectionInfo>(jarvis.getConnectionInfo());
  let unsubConnectionInfo: (() => void) | null = null;

  // Search filter state
  let searchQuery = $state('');

  // Show archived section
  let showArchived = $state(false);

  // Virtual scrolling configuration for conversation list
  const ESTIMATED_CONVERSATION_HEIGHT = 72; // Height of each conversation item
  const CONVERSATION_BUFFER_SIZE = 5; // Extra items to render above/below viewport
  const MIN_VISIBLE_CONVERSATIONS = 10; // Minimum items to render

  // Virtual scrolling state
  let visibleStartIndex = $state(0);
  let visibleEndIndex = $state(MIN_VISIBLE_CONVERSATIONS);
  let virtualTopPadding = $state(0);
  let virtualBottomPadding = $state(0);
  let listContainerRef = $state<HTMLDivElement | null>(null);
  let rafPending = $state(false);

  // Filter helper: matches conversation against search query
  function matchesSearch(conv: Conversation, query: string): boolean {
    const q = query.toLowerCase();
    const name = getDisplayName(conv).toLowerCase();
    if (name.includes(q)) return true;
    const preview = (conv.last_message_text || '').toLowerCase();
    if (preview.includes(q)) return true;
    return conv.participants.some(p => p.toLowerCase().includes(q));
  }

  // Sort conversations: pinned first, then by last_message_date, archived filtered
  let sortedConversations = $derived.by(() => {
    let convs = conversationsStore.conversations.filter(
      c => !conversationsStore.archivedChats.has(c.chat_id)
    );
    if (searchQuery.trim()) {
      convs = convs.filter(c => matchesSearch(c, searchQuery.trim()));
    }
    return convs.sort((a, b) => {
      const aPinned = conversationsStore.pinnedChats.has(a.chat_id);
      const bPinned = conversationsStore.pinnedChats.has(b.chat_id);
      if (aPinned && !bPinned) return -1;
      if (!aPinned && bPinned) return 1;
      return new Date(b.last_message_date).getTime() - new Date(a.last_message_date).getTime();
    });
  });

  let archivedConversations = $derived(
    conversationsStore.conversations.filter(c => conversationsStore.archivedChats.has(c.chat_id))
  );

  // Avatar state with LRU cache to prevent unbounded memory growth
  // Max 50 avatars (~5-10MB) - enough for visible + scroll buffer
  const MAX_CACHED_AVATARS = 50;
  let avatarStates = $state<Map<string, 'loading' | 'loaded' | 'error'>>(new Map());
  let avatarUrls = $state<LRUCache<string, string>>(new LRUCache(MAX_CACHED_AVATARS, (_key, url) => {
    if (url.startsWith('blob:')) {
      URL.revokeObjectURL(url);
    }
  }));

  // Intersection Observer for lazy loading avatars
  let observer = $state<IntersectionObserver | null>(null);
  let observedElements = $state<Map<string, HTMLElement>>(new Map());

  const API_BASE = getApiBaseUrl();
  let cleanup: (() => void) | null = null;

  onMount(() => {
    initializePolling().then((fn) => { cleanup = fn; });
    window.addEventListener('keydown', handleKeydown);

    // Subscribe to connection info changes (UX-01)
    unsubConnectionInfo = jarvis.on<ConnectionInfo>('connection_info_changed', (info) => {
      connectionInfo = info;
    });

    observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const identifier = entry.target.getAttribute('data-identifier');
            if (identifier && !avatarStates.has(identifier)) {
              void loadAvatar(identifier);
            }
          }
        });
      },
      {
        root: null,
        rootMargin: '50px',
        threshold: 0.1,
      }
    );

    return () => {
      cleanup?.();
      unsubConnectionInfo?.();
      window.removeEventListener('keydown', handleKeydown);
      observer?.disconnect();
      // LRU cache clear() now handles blob URL revocation via onEvict callback
      avatarUrls.clear();
    };
  });

  // Avatar loading with concurrency limit and rate limiting
  const MAX_CONCURRENT_AVATARS = 3;
  const AVATAR_LOAD_RATE_LIMIT = 10; // Max 10 avatars per second
  let activeAvatarLoads = 0;
  let avatarQueue: string[] = [];
  let avatarQueueSet = new Set<string>();
  let avatarLoadTimestamps: number[] = []; // Track load times for rate limiting
  
  // Avatar cache metrics
  let avatarCacheMetrics = $state({
    hits: 0,
    misses: 0,
    errors: 0,
    lastReset: Date.now(),
  });

  function checkRateLimit(): boolean {
    const now = Date.now();
    // Remove timestamps older than 1 second
    avatarLoadTimestamps = avatarLoadTimestamps.filter(ts => now - ts < 1000);
    return avatarLoadTimestamps.length < AVATAR_LOAD_RATE_LIMIT;
  }

  async function loadAvatar(identifier: string) {
    if (avatarStates.get(identifier) === 'loading') return;

    // If already cached, count as hit and return
    if (avatarUrls.has(identifier)) {
      avatarStates.set(identifier, 'loaded');
      avatarStates = avatarStates;
      // Touch the cache to update LRU order
      const url = avatarUrls.get(identifier);
      if (url) avatarUrls.set(identifier, url);
      // Update metrics
      avatarCacheMetrics.hits++;
      return;
    }

    // Check rate limit
    if (!checkRateLimit()) {
      // Delay and retry
      setTimeout(() => loadAvatar(identifier), 100);
      return;
    }

    if (activeAvatarLoads >= MAX_CONCURRENT_AVATARS) {
      if (!avatarQueueSet.has(identifier)) {
        avatarQueue.push(identifier);
        avatarQueueSet.add(identifier);
      }
      return;
    }

    activeAvatarLoads++;
    avatarStates.set(identifier, 'loading');
    avatarStates = avatarStates;
    avatarCacheMetrics.misses++;
    avatarLoadTimestamps.push(Date.now());

    try {
      const response = await fetch(
        `${API_BASE}/contacts/${encodeURIComponent(identifier)}/avatar?size=88&format=png`
      );

      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const blob = await response.blob();
      const ALLOWED_AVATAR_TYPES = new Set(['image/jpeg', 'image/png']);
      if (!ALLOWED_AVATAR_TYPES.has(blob.type)) {
        throw new Error(`Disallowed avatar MIME type: ${blob.type}`);
      }
      const url = URL.createObjectURL(blob);

      // LRU cache handles eviction + blob URL revocation via onEvict callback
      avatarUrls.set(identifier, url);

      avatarStates.set(identifier, 'loaded');
      avatarStates = avatarStates;
      // Trigger reactivity for avatarUrls
      avatarUrls = avatarUrls;
    } catch (error) {
      avatarCacheMetrics.errors++;
      avatarUrls.delete(identifier);
      avatarStates.set(identifier, 'error');
      avatarStates = avatarStates;
      // Trigger reactivity
      avatarUrls = avatarUrls;
    } finally {
      activeAvatarLoads--;
      if (avatarQueue.length > 0) {
        const next = avatarQueue.shift();
        if (next) {
          avatarQueueSet.delete(next);
          void loadAvatar(next);
        }
      }
    }
  }

  function observeAvatar(node: HTMLElement, identifier: string) {
    if (observer && identifier) {
      node.setAttribute('data-identifier', identifier);
      observer.observe(node);
      observedElements.set(identifier, node);
    }

    return {
      destroy() {
        if (observer && node) {
          observer.unobserve(node);
        }
        observedElements.delete(identifier);
      },
    };
  }

  function handleKeydown(event: KeyboardEvent) {
    if ($activeZone !== 'conversations' && $activeZone !== null) return;
    if (isTypingInInput(event)) return;

    const conversations = sortedConversations;
    if (conversations.length === 0) return;

    const maxIndex = conversations.length - 1;

    // Enter/Space: select conversation (component-specific)
    if ((event.key === 'Enter' || event.key === ' ') && focusedIndex >= 0 && focusedIndex <= maxIndex) {
      event.preventDefault();
      const conv = conversations[focusedIndex]!;
      selectConversation(conv.chat_id);
      setActiveZone('messages');
      announce(`Opened conversation with ${getDisplayName(conv)}`);
      return;
    }

    const action = getNavAction(event.key, event.shiftKey, focusedIndex, maxIndex);
    if (!action) return;

    event.preventDefault();

    if (action.type === 'escape') {
      setConversationIndex(-1);
      focusedIndex = -1;
      setActiveZone(null);
      return;
    }

    setActiveZone('conversations');
    const newIndex = action.type === 'first' ? 0 : action.index;
    setConversationIndex(newIndex);
    focusedIndex = newIndex;
    scrollToItem(newIndex);
    announce(`${getDisplayName(conversations[newIndex]!)}, ${newIndex + 1} of ${conversations.length}`);
  }

  function scrollToItem(index: number) {
    tick().then(() => {
      // For virtual scrolling, calculate position based on estimated height
      if (!listContainerRef) return;
      
      const itemTop = index * ESTIMATED_CONVERSATION_HEIGHT;
      const containerHeight = listContainerRef.clientHeight;
      
      // Update virtual scroll to include this item
      visibleStartIndex = Math.max(0, index - CONVERSATION_BUFFER_SIZE);
      visibleEndIndex = Math.min(
        sortedConversations.length,
        index + MIN_VISIBLE_CONVERSATIONS + CONVERSATION_BUFFER_SIZE
      );
      updateVirtualPadding();
      
      // Scroll to position
      listContainerRef.scrollTo({
        top: itemTop - containerHeight / 2 + ESTIMATED_CONVERSATION_HEIGHT / 2,
        behavior: 'smooth'
      });
    });
  }

  function updateVirtualPadding() {
    virtualTopPadding = visibleStartIndex * ESTIMATED_CONVERSATION_HEIGHT;
    const totalHeight = sortedConversations.length * ESTIMATED_CONVERSATION_HEIGHT;
    const visibleHeight = (visibleEndIndex - visibleStartIndex) * ESTIMATED_CONVERSATION_HEIGHT;
    virtualBottomPadding = Math.max(0, totalHeight - virtualTopPadding - visibleHeight);
  }

  function calculateVisibleRange() {
    if (!listContainerRef) return;
    
    const { scrollTop, clientHeight } = listContainerRef;
    const totalConversations = sortedConversations.length;
    
    if (totalConversations === 0) {
      visibleStartIndex = 0;
      visibleEndIndex = 0;
      virtualTopPadding = 0;
      virtualBottomPadding = 0;
      return;
    }

    // Calculate which items should be visible
    const startIdx = Math.max(0, Math.floor(scrollTop / ESTIMATED_CONVERSATION_HEIGHT) - CONVERSATION_BUFFER_SIZE);
    const visibleCount = Math.ceil(clientHeight / ESTIMATED_CONVERSATION_HEIGHT);
    const endIdx = Math.min(
      totalConversations,
      startIdx + visibleCount + CONVERSATION_BUFFER_SIZE * 2
    );

    visibleStartIndex = startIdx;
    visibleEndIndex = Math.max(endIdx, startIdx + MIN_VISIBLE_CONVERSATIONS);
    updateVirtualPadding();
  }

  function handleScroll() {
    if (rafPending) return;
    rafPending = true;
    requestAnimationFrame(() => {
      rafPending = false;
      calculateVisibleRange();
    });
  }

  // Get visible conversations slice
  let visibleConversations = $derived(
    sortedConversations.slice(visibleStartIndex, visibleEndIndex)
  );

  // Reset virtual scroll when conversations change significantly
  $effect(() => {
    // Access sortedConversations to track changes
    const count = sortedConversations.length;
    if (count > 0 && visibleEndIndex > count) {
      visibleEndIndex = Math.min(MIN_VISIBLE_CONVERSATIONS, count);
      updateVirtualPadding();
    }
  });

  function getDisplayName(conv: Conversation): string {
    if (conv.display_name) return conv.display_name;
    if (conv.participants.length === 1) {
      return formatParticipant(conv.participants[0]!);
    }
    // For group chats, format participant names
    const formatted = conv.participants.slice(0, 2).map(p => formatParticipant(p));
    return (
      formatted.join(', ') +
      (conv.participants.length > 2 ? ` +${conv.participants.length - 2}` : '')
    );
  }

  function hasNewMessages(chatId: string): boolean {
    return conversationsStore.hasNewMessages(chatId);
  }

  function getUnreadCount(chatId: string): number {
    return conversationsStore.getUnreadCount(chatId);
  }

  function getPrimaryIdentifier(conv: Conversation): string | null {
    if (conv.is_group) return null;
    return conv.participants[0] || null;
  }

  function handleContextMenu(event: MouseEvent, chatId: string) {
    event.preventDefault();
    contextMenu = { x: event.clientX, y: event.clientY, chatId };
  }

  function closeContextMenu() {
    contextMenu = null;
  }

  function handlePin(chatId: string) {
    togglePinChat(chatId);
    closeContextMenu();
  }

  function handleArchive(chatId: string) {
    toggleArchiveChat(chatId);
    closeContextMenu();
  }

  function getInitials(name: string): string {
    const parts = name.trim().split(/\s+/);
    const first = parts[0];
    const last = parts[parts.length - 1];
    if (parts.length >= 2 && first && first[0] && last && last[0]) {
      return `${first[0]}${last[0]}`.toUpperCase();
    }
    return parts[0]?.[0]?.toUpperCase() || '?';
  }
</script>

<div class="conversation-list">
  <div class="header">
    <h2>Messages</h2>
    {#if connectionInfo.state !== 'connected'}
      <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
      <div
        class="connection-badge disconnected"
        role="status"
        aria-live="polite"
        tabindex="0"
        title="Disconnected from server"
      >
        <span class="connection-dot"></span>
        <span class="sr-only">Disconnected from server</span>
      </div>
    {:else if connectionInfo.isFallback}
      <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
      <div
        class="connection-badge fallback"
        role="status"
        aria-live="polite"
        tabindex="0"
        title="Using WebSocket fallback (Unix socket unavailable)"
      >
        <span class="connection-dot"></span>
        <span class="sr-only">Using WebSocket fallback</span>
      </div>
    {/if}
  </div>

  <div class="search" role="search">
    <label for="conversation-search" class="sr-only">Search conversations</label>
    <input
      id="conversation-search"
      type="text"
      placeholder="Search conversations..."
      bind:value={searchQuery}
      aria-controls="conversation-list"
    />
  </div>

  {#if conversationsStore.loading && conversationsStore.isInitialLoad}
    <ConversationSkeleton />
  {:else if conversationsStore.error}
    <div class="error">{conversationsStore.error}</div>
  {:else if conversationsStore.conversations.length === 0}
    <div class="empty">No conversations found</div>
  {:else}
    <div
      id="conversation-list"
      class="list"
      role="listbox"
      aria-label="Conversations"
      bind:this={listContainerRef}
      onscroll={handleScroll}
    >
      <!-- Virtual scroll spacer for conversations above viewport -->
      <div class="virtual-spacer-top" style="height: {virtualTopPadding}px;"></div>
      
      {#each visibleConversations as conv, visibleIdx (conv.chat_id)}
        {@const index = visibleStartIndex + visibleIdx}
        {@const identifier = getPrimaryIdentifier(conv)}
        {@const avatarUrl = identifier ? avatarUrls.get(identifier) : null}
        {@const avatarState = identifier ? avatarStates.get(identifier) : null}
        {@const isFocused = focusedIndex === index}
        <button
          bind:this={itemRefs[index]}
          class="conversation"
          class:active={conversationsStore.selectedChatId === conv.chat_id}
          class:focused={isFocused}
          class:group={conv.is_group}
          class:has-new={hasNewMessages(conv.chat_id)}
          class:pinned={conversationsStore.pinnedChats.has(conv.chat_id)}
          onclick={() => {
            selectConversation(conv.chat_id);
            setConversationIndex(index);
          }}
          oncontextmenu={(e) => handleContextMenu(e, conv.chat_id)}
          role="option"
          aria-selected={conversationsStore.selectedChatId === conv.chat_id}
          tabindex={isFocused ? 0 : -1}
        >
          <div class="avatar-container">
            <div
              class="avatar"
              class:group={conv.is_group}
              class:has-image={avatarState === 'loaded' && avatarUrl}
              use:observeAvatar={identifier || ''}
            >
              {#if conv.is_group}
                <svg viewBox="0 0 24 24" fill="currentColor">
                  <path
                    d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm-6 8v-2c0-2.67 5.33-4 6-4s6 1.33 6 4v2H6z"
                  />
                </svg>
              {:else if avatarState === 'loaded' && avatarUrl}
                <img src={avatarUrl} alt="" class="avatar-image" />
              {:else if avatarState === 'loading'}
                <span class="avatar-loading">{getInitials(getDisplayName(conv))}</span>
              {:else}
                {getInitials(getDisplayName(conv))}
              {/if}
            </div>
            {#if getUnreadCount(conv.chat_id) > 0}
              <span class="unread-badge" aria-label="{getUnreadCount(conv.chat_id)} unread messages">
                {getUnreadCount(conv.chat_id) > 99 ? '99+' : getUnreadCount(conv.chat_id)}
              </span>
            {/if}
          </div>
          <div class="info">
            <div class="name-row">
              <span class="name" class:has-new={hasNewMessages(conv.chat_id)}>
                {#if conversationsStore.pinnedChats.has(conv.chat_id)}
                  <svg class="pin-icon" viewBox="0 0 24 24" fill="currentColor" width="12" height="12">
                    <path d="M16 12V4h1V2H7v2h1v8l-2 2v2h5.2v6h1.6v-6H18v-2l-2-2z" />
                  </svg>
                {/if}
                {getDisplayName(conv)}
              </span>
              <span class="date" class:has-new={hasNewMessages(conv.chat_id)}>
                {formatConversationDate(conv.last_message_date)}
              </span>
            </div>
            <div class="preview" class:has-new={hasNewMessages(conv.chat_id)}>
              {conv.last_message_text || 'No messages'}
            </div>
          </div>
        </button>
      {/each}
      
      <!-- Virtual scroll spacer for conversations below viewport -->
      <div class="virtual-spacer-bottom" style="height: {virtualBottomPadding}px;"></div>

      {#if archivedConversations.length > 0}
        <button
          class="archived-toggle"
          onclick={() => (showArchived = !showArchived)}
          aria-expanded={showArchived}
          aria-controls="archived-list"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14">
            <polyline points={showArchived ? "18 15 12 9 6 15" : "6 9 12 15 18 9"}></polyline>
          </svg>
          Archived ({archivedConversations.length})
        </button>
        {#if showArchived}
          <div id="archived-list" transition:slide={{ duration: 200 }}>
            {#each archivedConversations as conv (conv.chat_id)}
              {@const identifier = getPrimaryIdentifier(conv)}
              {@const avatarUrl = identifier ? avatarUrls.get(identifier) : null}
              {@const avatarState = identifier ? avatarStates.get(identifier) : null}
              <button
                class="conversation archived"
                class:active={conversationsStore.selectedChatId === conv.chat_id}
                onclick={() => selectConversation(conv.chat_id)}
                oncontextmenu={(e) => handleContextMenu(e, conv.chat_id)}
              >
                <div class="avatar-container">
                  <div
                    class="avatar"
                    class:group={conv.is_group}
                    class:has-image={avatarState === 'loaded' && avatarUrl}
                    use:observeAvatar={identifier || ''}
                  >
                    {#if conv.is_group}
                      <svg viewBox="0 0 24 24" fill="currentColor"
                        ><path
                          d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm-6 8v-2c0-2.67 5.33-4 6-4s6 1.33 6 4v2H6z"
                        /></svg
                      >
                    {:else if avatarState === 'loaded' && avatarUrl}
                      <img src={avatarUrl} alt="" class="avatar-image" />
                    {:else}
                      {getInitials(getDisplayName(conv))}
                    {/if}
                  </div>
                </div>
                <div class="info">
                  <div class="name-row">
                    <span class="name">{getDisplayName(conv)}</span>
                    <span class="date">{formatConversationDate(conv.last_message_date)}</span>
                  </div>
                  <div class="preview">{conv.last_message_text || 'No messages'}</div>
                </div>
              </button>
            {/each}
          </div>
        {/if}
      {/if}
    </div>
  {/if}

  {#if contextMenu}
    <!-- svelte-ignore a11y_click_events_have_key_events -->
    <div class="context-menu-overlay" onclick={closeContextMenu} role="presentation">
      <div class="context-menu" style="left: {contextMenu.x}px; top: {contextMenu.y}px;">
        <button class="context-menu-item" onclick={() => handlePin(contextMenu!.chatId)}>
          {#if conversationsStore.pinnedChats.has(contextMenu.chatId)}
            Unpin
          {:else}
            Pin to Top
          {/if}
        </button>
        <button class="context-menu-item" onclick={() => handleArchive(contextMenu!.chatId)}>
          {#if conversationsStore.archivedChats.has(contextMenu.chatId)}
            Unarchive
          {:else}
            Archive
          {/if}
        </button>
      </div>
    </div>
  {/if}
</div>

<style>
  .conversation-list {
    width: 300px;
    min-width: 300px;
    background: var(--surface-elevated);
    border-right: 1px solid var(--border-default);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .header {
    padding: var(--space-4);
    border-bottom: 1px solid var(--border-default);
    display: flex;
    align-items: center;
    gap: var(--space-2);
  }

  .header h2 {
    font-size: var(--text-xl);
    font-weight: var(--font-weight-semibold);
    letter-spacing: var(--letter-spacing-tight);
    margin: 0;
  }

  .connection-badge {
    display: flex;
    align-items: center;
    margin-left: auto;
    padding: 2px;
    border-radius: var(--radius-sm);
  }

  .connection-badge:focus-visible {
    outline: 2px solid var(--border-focus);
    outline-offset: 2px;
  }

  .connection-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
  }

  .connection-badge.disconnected .connection-dot {
    background: var(--color-error);
  }

  .connection-badge.fallback .connection-dot {
    background: var(--color-warning, #ff9500);
  }

  .search {
    padding: var(--space-3) var(--space-4);
    border-bottom: 1px solid var(--border-default);
  }

  .search input {
    width: 100%;
    padding: var(--space-2) var(--space-3);
    background: var(--surface-base);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-md);
    color: var(--text-primary);
    font-size: var(--text-base);
    font-family: var(--font-family-sans);
  }

  .search input::placeholder {
    color: var(--text-tertiary);
  }

  .list {
    flex: 1;
    overflow-y: auto;
  }

  .virtual-spacer-top,
  .virtual-spacer-bottom {
    flex-shrink: 0;
  }

  .conversation {
    width: 100%;
    display: flex;
    align-items: center;
    gap: var(--space-3);
    padding: var(--space-3) var(--space-4);
    background: transparent;
    border: none;
    cursor: pointer;
    text-align: left;
    transition: background var(--duration-fast) var(--ease-out);
  }

  .conversation:hover {
    background: var(--surface-hover);
  }

  .conversation.active {
    background: var(--surface-active);
  }

  .conversation.focused {
    outline: 2px solid var(--color-primary);
    outline-offset: -2px;
  }

  .conversation.focused:not(.active) {
    background: var(--surface-hover);
  }

  .avatar-container {
    position: relative;
    flex-shrink: 0;
  }

  .avatar {
    width: 44px;
    height: 44px;
    border-radius: 50%;
    background: var(--color-primary);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: var(--font-weight-semibold);
    font-size: 18px;
    color: white;
    overflow: hidden;
    position: relative;
  }

  .avatar.group {
    background: var(--group-color);
  }

  .avatar.has-image {
    background: transparent;
  }

  .avatar svg {
    width: 24px;
    height: 24px;
  }

  .avatar-image {
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: 50%;
  }

  .avatar-loading {
    opacity: 0.7;
    animation: avatarPulse 1.5s var(--ease-in-out) infinite;
  }

  @keyframes avatarPulse {
    0%,
    100% {
      opacity: 0.7;
    }
    50% {
      opacity: 0.4;
    }
  }


  .info {
    flex: 1;
    min-width: 0;
  }

  .name-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: var(--space-1);
  }

  .name {
    font-weight: var(--font-weight-medium);
    color: var(--text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .name.has-new {
    font-weight: var(--font-weight-bold);
  }

  .date {
    font-size: var(--text-xs);
    color: var(--text-secondary);
    flex-shrink: 0;
    margin-left: var(--space-2);
  }

  .date.has-new {
    color: var(--color-primary);
    font-weight: var(--font-weight-semibold);
  }

  .preview {
    font-size: var(--text-sm);
    color: var(--text-secondary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .preview.has-new {
    color: var(--text-primary);
    font-weight: var(--font-weight-medium);
  }

  .error,
  .empty {
    padding: var(--space-6) var(--space-4);
    text-align: center;
    color: var(--text-secondary);
  }

  .error {
    color: var(--color-error);
  }

  .unread-badge {
    position: absolute;
    top: -2px;
    right: -2px;
    min-width: 18px;
    height: 18px;
    padding: 0 5px;
    background: var(--color-error);
    color: white;
    font-size: 10px;
    font-weight: var(--font-weight-bold);
    border-radius: var(--radius-full);
    display: flex;
    align-items: center;
    justify-content: center;
    line-height: 1;
    border: 2px solid var(--surface-elevated);
  }

  .pin-icon {
    flex-shrink: 0;
    color: var(--text-tertiary);
    margin-right: 2px;
  }

  .conversation.pinned {
    border-left: 2px solid var(--color-primary);
  }

  .conversation.archived {
    opacity: 0.6;
  }

  .archived-toggle {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    width: 100%;
    padding: var(--space-2) var(--space-4);
    background: transparent;
    border: none;
    border-top: 1px solid var(--border-subtle);
    color: var(--text-tertiary);
    font-size: var(--text-xs);
    cursor: pointer;
    transition: color var(--duration-fast) var(--ease-out);
  }

  .archived-toggle:hover {
    color: var(--text-secondary);
  }

  .context-menu-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: var(--z-popover);
  }

  .context-menu {
    position: fixed;
    background: var(--surface-elevated);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-md);
    box-shadow: var(--shadow-lg);
    padding: var(--space-1);
    min-width: 160px;
    z-index: var(--z-popover);
    animation: fadeIn 0.1s ease;
  }

  @keyframes fadeIn {
    from { opacity: 0; transform: scale(0.95); }
    to { opacity: 1; transform: scale(1); }
  }

  .context-menu-item {
    display: block;
    width: 100%;
    padding: var(--space-2) var(--space-3);
    background: transparent;
    border: none;
    border-radius: var(--radius-sm);
    color: var(--text-primary);
    font-size: var(--text-sm);
    text-align: left;
    cursor: pointer;
    transition: background var(--duration-fast) var(--ease-out);
  }

  .context-menu-item:hover {
    background: var(--surface-hover);
  }
</style>
