<script lang="ts">
  import { onMount, tick } from 'svelte';
  import {
    conversationsStore,
    selectConversation,
    initializePolling,
    togglePinChat,
    toggleArchiveChat,
  } from '../stores/conversations.svelte';
  import { api } from '../api/client';
  import type { Topic, Conversation } from '../types';
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

  // Track focused conversation for keyboard navigation
  let focusedIndex = $state(-1);
  // @ts-expect-error - used in bind:this
  let _listRef = $state<HTMLElement | null>(null);
  let itemRefs = $state<HTMLButtonElement[]>([]);

  // Sync focusedIndex with store
  $effect(() => {
    focusedIndex = $conversationIndex;
  });

  // Topics state
  let topicsMap = $state<Map<string, Topic[]>>(new Map());
  let allTopicsMap = $state<Map<string, Topic[]>>(new Map());
  let loadingTopics = $state<Set<string>>(new Set());
  let topicFetchControllers = $state<Map<string, AbortController>>(new Map());

  // Context menu state
  let contextMenu = $state<{ x: number; y: number; chatId: string } | null>(null);

  // Show archived section
  let showArchived = $state(false);

  // Sort conversations: pinned first, then by last_message_date, archived filtered
  let sortedConversations = $derived.by(() => {
    const convs = conversationsStore.conversations.filter(
      c => !conversationsStore.archivedChats.has(c.chat_id)
    );
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

  // Intersection Observer for lazy loading avatars and topics
  let observer = $state<IntersectionObserver | null>(null);
  let topicObserver = $state<IntersectionObserver | null>(null);
  let observedElements = $state<Map<string, HTMLElement>>(new Map());
  let observedTopicElements = $state<Map<string, HTMLElement>>(new Map());

  const API_BASE = getApiBaseUrl();
  let cleanup: (() => void) | null = null;

  // Use $derived for conversation fingerprint to avoid manual tracking
  let conversationFingerprint = $derived(
    conversationsStore.conversations
      .slice(0, 20)
      .map((c) => c.chat_id)
      .join(',')
  );

  // Defer topic fetching - topics are cosmetic tags, not critical for initial render
  // Only fetch for first 5 conversations to avoid N+1 burst
  $effect(() => {
    const fingerprint = conversationFingerprint;
    if (!fingerprint) return;
    const timer = setTimeout(() => void fetchTopicsForConversations(), 2000);
    return () => clearTimeout(timer);
  });

  onMount(() => {
    initializePolling().then((fn) => { cleanup = fn; });
    window.addEventListener('keydown', handleKeydown);

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

    topicObserver = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const chatId = entry.target.getAttribute('data-chat-id');
            if (chatId && !topicsMap.has(chatId) && !loadingTopics.has(chatId)) {
              void fetchTopicsForChat(chatId);
            }
          }
        });
      },
      {
        root: null,
        rootMargin: '100px',
        threshold: 0.1,
      }
    );

    return () => {
      cleanup?.();
      window.removeEventListener('keydown', handleKeydown);
      observer?.disconnect();
      topicObserver?.disconnect();
      // LRU cache clear() now handles blob URL revocation via onEvict callback
      avatarUrls.clear();
      topicFetchControllers.forEach((controller) => controller.abort());
    };
  });

  async function fetchTopicsForConversations() {
    // Only fetch topics for first 5 conversations to avoid N+1 burst
    // Topics for other conversations will be fetched on-demand when scrolled into view
    const visibleConvs = conversationsStore.conversations.slice(0, 5);
    const toFetch = visibleConvs.filter(
      (conv) => !topicsMap.has(conv.chat_id) && !loadingTopics.has(conv.chat_id)
    );

    // Limit concurrent topic requests to avoid server overload
    const BATCH_SIZE = 3;
    for (let i = 0; i < toFetch.length; i += BATCH_SIZE) {
      const batch = toFetch.slice(i, i + BATCH_SIZE);
      await Promise.all(batch.map((conv) => fetchTopicsForChat(conv.chat_id)));
    }
  }

  async function fetchTopicsForChat(chatId: string) {
    const existingController = topicFetchControllers.get(chatId);
    if (existingController) {
      existingController.abort();
    }

    const controller = new AbortController();
    topicFetchControllers.set(chatId, controller);

    loadingTopics.add(chatId);
    loadingTopics = loadingTopics;

    try {
      const response = await api.getTopics(chatId);
      if (controller.signal.aborted) return;

      topicsMap.set(chatId, response.topics);
      allTopicsMap.set(chatId, response.all_topics);
      topicsMap = topicsMap;
      allTopicsMap = allTopicsMap;
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') return;
      console.debug('Failed to fetch topics for', chatId, error);
    } finally {
      loadingTopics.delete(chatId);
      loadingTopics = loadingTopics;
      topicFetchControllers.delete(chatId);
    }
  }

  // Avatar loading with concurrency limit
  const MAX_CONCURRENT_AVATARS = 3;
  let activeAvatarLoads = 0;
  let avatarQueue: string[] = [];
  let avatarQueueSet = new Set<string>();

  async function loadAvatar(identifier: string) {
    if (avatarStates.get(identifier) === 'loading') return;

    // If already cached, just mark as loaded (LRU will update recency)
    if (avatarUrls.has(identifier)) {
      avatarStates.set(identifier, 'loaded');
      avatarStates = avatarStates;
      // Touch the cache to update LRU order
      const url = avatarUrls.get(identifier);
      if (url) avatarUrls.set(identifier, url);
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

    try {
      const response = await fetch(
        `${API_BASE}/contacts/${encodeURIComponent(identifier)}/avatar?size=88&format=png`
      );

      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);

      // LRU cache handles eviction + blob URL revocation via onEvict callback
      avatarUrls.set(identifier, url);

      avatarStates.set(identifier, 'loaded');
      avatarStates = avatarStates;
      // Trigger reactivity for avatarUrls
      avatarUrls = avatarUrls;
    } catch (error) {
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

  function observeTopics(node: HTMLElement, chatId: string) {
    if (topicObserver && chatId) {
      node.setAttribute('data-chat-id', chatId);
      topicObserver.observe(node);
      observedTopicElements.set(chatId, node);
    }

    return {
      destroy() {
        if (topicObserver && node) {
          topicObserver.unobserve(node);
        }
        observedTopicElements.delete(chatId);
      },
    };
  }

  function handleKeydown(event: KeyboardEvent) {
    if ($activeZone !== 'conversations' && $activeZone !== null) return;
    if (isTypingInInput(event)) return;

    const conversations = conversationsStore.conversations;
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
      const item = itemRefs[index];
      if (item) {
        item.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      }
    });
  }

  function formatParticipant(p: string): string {
    const atParts = p.split('@');
    if (p.includes('@') && atParts[0]) return atParts[0];
    if (/^\+?\d{10,}$/.test(p.replace(/[\s\-()]/g, ''))) {
      return '...' + p.replace(/\D/g, '').slice(-4);
    }
    return p;
  }

  function getDisplayName(conv: Conversation): string {
    if (conv.display_name) return conv.display_name;
    if (conv.participants.length === 1) {
      return formatParticipant(conv.participants[0]!);
    }
    const formatted = conv.participants.slice(0, 2).map(formatParticipant);
    return (
      formatted.join(', ') +
      (conv.participants.length > 2 ? ` +${conv.participants.length - 2}` : '')
    );
  }

  function getTopicColorClass(color: string): string {
    const colorMap: Record<string, string> = {
      blue: 'topic-blue',
      green: 'topic-green',
      purple: 'topic-purple',
      pink: 'topic-pink',
      orange: 'topic-orange',
      gray: 'topic-gray',
      indigo: 'topic-indigo',
      amber: 'topic-amber',
      cyan: 'topic-cyan',
      rose: 'topic-rose',
    };
    return colorMap[color] || 'topic-gray';
  }

  function getAllTopicsTooltip(chatId: string): string {
    const allTopics = allTopicsMap.get(chatId) || [];
    if (allTopics.length <= 2) return '';
    return allTopics.map((t) => `${t.display_name} (${Math.round(t.confidence * 100)}%)`).join('\n');
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
  </div>

  <div class="search">
    <input type="text" placeholder="Search conversations..." />
  </div>

  {#if conversationsStore.loading}
    <ConversationSkeleton />
  {:else if conversationsStore.error}
    <div class="error">{conversationsStore.error}</div>
  {:else if conversationsStore.conversations.length === 0}
    <div class="empty">No conversations found</div>
  {:else}
    <div class="list" bind:this={_listRef} role="listbox" aria-label="Conversations">
      {#each sortedConversations as conv, index (conv.chat_id)}
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
            <div class="topics-row" use:observeTopics={conv.chat_id}>
              {#if topicsMap.has(conv.chat_id)}
                {#each topicsMap.get(conv.chat_id) || [] as topic}
                  <span
                    class="topic-tag {getTopicColorClass(topic.color)}"
                    title={getAllTopicsTooltip(conv.chat_id) || topic.display_name}
                  >
                    {topic.display_name}
                  </span>
                {/each}
                {#if (allTopicsMap.get(conv.chat_id)?.length || 0) > 2}
                  <span class="topic-more" title={getAllTopicsTooltip(conv.chat_id)}>
                    +{(allTopicsMap.get(conv.chat_id)?.length || 0) - 2}
                  </span>
                {/if}
              {/if}
            </div>
            <div class="preview" class:has-new={hasNewMessages(conv.chat_id)}>
              {conv.last_message_text || 'No messages'}
            </div>
          </div>
        </button>
      {/each}

      {#if archivedConversations.length > 0}
        <button class="archived-toggle" onclick={() => showArchived = !showArchived}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14">
            <polyline points={showArchived ? "18 15 12 9 6 15" : "6 9 12 15 18 9"}></polyline>
          </svg>
          Archived ({archivedConversations.length})
        </button>
        {#if showArchived}
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
                <div class="avatar" class:group={conv.is_group} class:has-image={avatarState === 'loaded' && avatarUrl} use:observeAvatar={identifier || ''}>
                  {#if conv.is_group}
                    <svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm-6 8v-2c0-2.67 5.33-4 6-4s6 1.33 6 4v2H6z" /></svg>
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
  @import '../styles/tokens.css';

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
  }

  .header h2 {
    font-size: var(--text-xl);
    font-weight: var(--font-weight-semibold);
    letter-spacing: var(--letter-spacing-tight);
    margin: 0;
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

  .topics-row {
    display: flex;
    gap: var(--space-1);
    margin-bottom: var(--space-1);
    flex-wrap: wrap;
  }

  .topic-tag {
    font-size: 10px;
    padding: 2px 6px;
    border-radius: var(--radius-full);
    font-weight: var(--font-weight-medium);
    white-space: nowrap;
  }

  .topic-more {
    font-size: 10px;
    padding: 2px 6px;
    border-radius: var(--radius-full);
    background: var(--surface-base);
    color: var(--text-secondary);
    cursor: help;
  }

  /* Topic color variants */
  .topic-blue {
    background: rgba(0, 122, 255, 0.15);
    color: #007aff;
  }
  .topic-green {
    background: rgba(52, 199, 89, 0.15);
    color: #34c759;
  }
  .topic-purple {
    background: rgba(88, 86, 214, 0.15);
    color: #5856d6;
  }
  .topic-pink {
    background: rgba(255, 45, 85, 0.15);
    color: #ff2d55;
  }
  .topic-orange {
    background: rgba(255, 149, 0, 0.15);
    color: #ff9500;
  }
  .topic-gray {
    background: rgba(142, 142, 147, 0.15);
    color: #8e8e93;
  }
  .topic-indigo {
    background: rgba(94, 92, 230, 0.15);
    color: #5e5ce6;
  }
  .topic-amber {
    background: rgba(255, 204, 0, 0.15);
    color: #ffcc00;
  }
  .topic-cyan {
    background: rgba(90, 200, 250, 0.15);
    color: #5ac8fa;
  }
  .topic-rose {
    background: rgba(255, 59, 48, 0.15);
    color: #ff3b30;
  }

  .preview.has-new {
    color: var(--text-primary);
    font-weight: var(--font-weight-medium);
  }

  .loading,
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
