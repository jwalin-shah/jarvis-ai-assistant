<script lang="ts">
  import { onMount, onDestroy, tick } from "svelte";
  import {
    conversationsStore,
    selectConversation,
    initializePolling,
  } from "../stores/conversations";
  import { api } from "../api/client";
  import type { Topic } from "../api/types";
  import ConversationSkeleton from "./ConversationSkeleton.svelte";
  import {
    activeZone,
    setActiveZone,
    conversationIndex,
    setConversationIndex,
    announce,
  } from "../stores/keyboard";
  import { getApiBaseUrl } from "../config/runtime";

  // Track focused conversation for keyboard navigation
  let focusedIndex = $state(-1);
  let listRef = $state<HTMLElement | null>(null);
  let itemRefs = $state<HTMLButtonElement[]>([]);

  // Sync focusedIndex with store
  $effect(() => {
    focusedIndex = $conversationIndex;
  });

  // Handle keyboard navigation
  function handleKeydown(event: KeyboardEvent) {
    // Only handle if this zone is active or no zone is active
    if ($activeZone !== "conversations" && $activeZone !== null) return;

    // Ignore if typing in an input
    if (
      event.target instanceof HTMLInputElement ||
      event.target instanceof HTMLTextAreaElement
    ) {
      return;
    }

    const conversations = $conversationsStore.conversations;
    if (conversations.length === 0) return;

    const maxIndex = conversations.length - 1;

    switch (event.key) {
      case "j":
      case "ArrowDown":
        event.preventDefault();
        setActiveZone("conversations");
        if (focusedIndex < maxIndex) {
          const newIndex = focusedIndex + 1;
          setConversationIndex(newIndex);
          focusedIndex = newIndex;
          scrollToItem(newIndex);
          announce(`${getDisplayName(conversations[newIndex])}, ${newIndex + 1} of ${conversations.length}`);
        }
        break;

      case "k":
      case "ArrowUp":
        event.preventDefault();
        setActiveZone("conversations");
        if (focusedIndex > 0) {
          const newIndex = focusedIndex - 1;
          setConversationIndex(newIndex);
          focusedIndex = newIndex;
          scrollToItem(newIndex);
          announce(`${getDisplayName(conversations[newIndex])}, ${newIndex + 1} of ${conversations.length}`);
        } else if (focusedIndex === -1 && conversations.length > 0) {
          // Start at first item if nothing selected
          setConversationIndex(0);
          focusedIndex = 0;
          scrollToItem(0);
          announce(`${getDisplayName(conversations[0])}, 1 of ${conversations.length}`);
        }
        break;

      case "Enter":
      case " ":
        if (focusedIndex >= 0 && focusedIndex <= maxIndex) {
          event.preventDefault();
          const conv = conversations[focusedIndex];
          selectConversation(conv.chat_id);
          setActiveZone("messages");
          announce(`Opened conversation with ${getDisplayName(conv)}`);
        }
        break;

      case "g":
        // Go to first conversation
        if (!event.shiftKey && conversations.length > 0) {
          event.preventDefault();
          setActiveZone("conversations");
          setConversationIndex(0);
          focusedIndex = 0;
          scrollToItem(0);
          announce(`${getDisplayName(conversations[0])}, 1 of ${conversations.length}`);
        }
        break;

      case "G":
        // Go to last conversation
        if (event.shiftKey && conversations.length > 0) {
          event.preventDefault();
          setActiveZone("conversations");
          setConversationIndex(maxIndex);
          focusedIndex = maxIndex;
          scrollToItem(maxIndex);
          announce(`${getDisplayName(conversations[maxIndex])}, ${maxIndex + 1} of ${conversations.length}`);
        }
        break;

      case "Escape":
        // Clear selection
        setConversationIndex(-1);
        focusedIndex = -1;
        setActiveZone(null);
        break;
    }
  }

  function scrollToItem(index: number) {
    tick().then(() => {
      const item = itemRefs[index];
      if (item) {
        item.scrollIntoView({ block: "nearest", behavior: "smooth" });
      }
    });
  }

  function setItemRef(el: HTMLButtonElement | null, index: number) {
    if (el) {
      itemRefs[index] = el;
    }
  }

  // Store for conversation topics
  let topicsMap: Map<string, Topic[]> = $state(new Map());
  let allTopicsMap: Map<string, Topic[]> = $state(new Map());
  let loadingTopics: Set<string> = $state(new Set());
  let topicFetchControllers: Map<string, AbortController> = $state(new Map());

  let cleanup: (() => void) | null = null;

  // API base URL for avatar endpoint
  const API_BASE = getApiBaseUrl();

  // Track loaded avatars and their states
  let avatarStates: Map<string, "loading" | "loaded" | "error"> = $state(new Map());
  let avatarUrls: Map<string, string> = $state(new Map());

  // Intersection Observer for lazy loading
  let observer: IntersectionObserver | null = null;
  let observedElements: Map<string, HTMLElement> = $state(new Map());

  onMount(() => {
    cleanup = initializePolling();

    // Add keyboard listener
    window.addEventListener("keydown", handleKeydown);

    // Create intersection observer for lazy loading avatars
    observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const identifier = entry.target.getAttribute("data-identifier");
            if (identifier && !avatarStates.has(identifier)) {
              loadAvatar(identifier);
            }
          }
        });
      },
      {
        root: null,
        rootMargin: "50px", // Load slightly before visible
        threshold: 0.1,
      }
    );
  });

  onDestroy(() => {
    if (cleanup) {
      cleanup();
    }
    if (observer) {
      observer.disconnect();
    }
    window.removeEventListener("keydown", handleKeydown);
    // Revoke any object URLs to prevent memory leaks
    avatarUrls.forEach((url) => {
      if (url.startsWith("blob:")) {
        URL.revokeObjectURL(url);
      }
    });
    // Abort all pending topic fetches
    topicFetchControllers.forEach((controller) => {
      controller.abort();
    });
  });

  // Track conversation count and IDs to avoid re-fetching topics on every store update
  let prevConversationCount = 0;
  let prevConversationIds = "";

  // Fetch topics when conversations are loaded (only when actual data changes)
  $effect(() => {
    const count = $conversationsStore.conversations.length;
    // Create a stable string of chat_ids to detect actual changes (not just reference updates)
    const currentIds = $conversationsStore.conversations.slice(0, 20).map(c => c.chat_id).join(",");

    if (count > 0 && (count !== prevConversationCount || currentIds !== prevConversationIds)) {
      prevConversationCount = count;
      prevConversationIds = currentIds;
      fetchTopicsForConversations();
    }
  });

  async function fetchTopicsForConversations() {
    // Fetch topics for visible conversations (first 20) in parallel
    const visibleConvs = $conversationsStore.conversations.slice(0, 20);
    const toFetch = visibleConvs.filter(
      conv => !topicsMap.has(conv.chat_id) && !loadingTopics.has(conv.chat_id)
    );

    // Batch fetch in parallel (Promise.all waits for all to complete)
    await Promise.all(toFetch.map(conv => fetchTopicsForChat(conv.chat_id)));
  }

  async function fetchTopicsForChat(chatId: string) {
    // Cancel any existing fetch for this chat
    const existingController = topicFetchControllers.get(chatId);
    if (existingController) {
      existingController.abort();
    }

    const controller = new AbortController();
    topicFetchControllers.set(chatId, controller);
    topicFetchControllers = topicFetchControllers;

    loadingTopics.add(chatId);
    loadingTopics = loadingTopics;
    try {
      const response = await api.getTopics(chatId);

      // Check if request was aborted
      if (controller.signal.aborted) return;

      topicsMap.set(chatId, response.topics);
      allTopicsMap.set(chatId, response.all_topics);
      topicsMap = topicsMap;
      allTopicsMap = allTopicsMap;
    } catch (error) {
      // Ignore abort errors
      if (error instanceof Error && error.name === "AbortError") return;
      // Silently fail - topics are optional
      console.debug("Failed to fetch topics for", chatId, error);
    } finally {
      loadingTopics.delete(chatId);
      loadingTopics = loadingTopics;
      topicFetchControllers.delete(chatId);
      topicFetchControllers = topicFetchControllers;
    }
  }

  // Register an element for observation
  function observeAvatar(node: HTMLElement, identifier: string) {
    if (observer && identifier) {
      node.setAttribute("data-identifier", identifier);
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

  // Semaphore for limiting concurrent avatar loads
  const MAX_CONCURRENT_AVATARS = 3;
  let activeAvatarLoads = 0;
  let avatarQueue: string[] = [];

  // Load avatar for a given identifier
  async function loadAvatar(identifier: string) {
    if (avatarStates.get(identifier) === "loading") return;

    // If at capacity, queue the request
    if (activeAvatarLoads >= MAX_CONCURRENT_AVATARS) {
      if (!avatarQueue.includes(identifier)) {
        avatarQueue.push(identifier);
      }
      return;
    }

    activeAvatarLoads++;
    avatarStates.set(identifier, "loading");
    avatarStates = avatarStates; // Trigger reactivity

    try {
      const response = await fetch(
        `${API_BASE}/contacts/${encodeURIComponent(identifier)}/avatar?size=88&format=png`
      );

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);

      // Revoke any existing blob URL before setting the new one
      const existingUrl = avatarUrls.get(identifier);
      if (existingUrl && existingUrl.startsWith("blob:")) {
        URL.revokeObjectURL(existingUrl);
      }

      avatarUrls.set(identifier, url);
      avatarStates.set(identifier, "loaded");
      avatarUrls = avatarUrls; // Trigger reactivity
      avatarStates = avatarStates;
    } catch (error) {
      console.error(`Failed to load avatar for ${identifier}:`, error);
      // Revoke any existing URL on error and clear it
      const existingUrl = avatarUrls.get(identifier);
      if (existingUrl && existingUrl.startsWith("blob:")) {
        URL.revokeObjectURL(existingUrl);
      }
      avatarUrls.delete(identifier);
      avatarUrls = avatarUrls;
      avatarStates.set(identifier, "error");
      avatarStates = avatarStates;
    } finally {
      activeAvatarLoads--;
      // Process next item in queue
      if (avatarQueue.length > 0) {
        const next = avatarQueue.shift();
        if (next) {
          loadAvatar(next);
        }
      }
    }
  }

  function formatDate(dateStr: string): string {
    const date = new Date(dateStr);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));

    if (days === 0) {
      return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    } else if (days === 1) {
      return "Yesterday";
    } else if (days < 7) {
      return date.toLocaleDateString([], { weekday: "short" });
    } else {
      return date.toLocaleDateString([], { month: "short", day: "numeric" });
    }
  }

  function formatParticipant(p: string): string {
    // If it looks like an email, show just the username
    if (p.includes("@")) {
      return p.split("@")[0];
    }
    // For phone numbers, show last 4 digits if long
    if (/^\+?\d{10,}$/.test(p.replace(/[\s\-()]/g, ""))) {
      const digits = p.replace(/\D/g, "");
      return "..." + digits.slice(-4);
    }
    return p;
  }

  function getDisplayName(conv: typeof $conversationsStore.conversations[0]): string {
    if (conv.display_name) return conv.display_name;
    if (conv.participants.length === 1) {
      return formatParticipant(conv.participants[0]);
    }
    // For group chats without a name, show formatted participants
    if (conv.is_group && conv.participants.length > 1) {
      const formatted = conv.participants.slice(0, 2).map(formatParticipant);
      return formatted.join(", ") +
        (conv.participants.length > 2 ? ` +${conv.participants.length - 2}` : "");
    }
    return conv.participants.slice(0, 2).map(formatParticipant).join(", ") +
      (conv.participants.length > 2 ? ` +${conv.participants.length - 2}` : "");
  }

  function getTopicColorClass(color: string): string {
    const colorMap: Record<string, string> = {
      blue: "topic-blue",
      green: "topic-green",
      purple: "topic-purple",
      pink: "topic-pink",
      orange: "topic-orange",
      gray: "topic-gray",
      indigo: "topic-indigo",
      amber: "topic-amber",
      cyan: "topic-cyan",
      rose: "topic-rose",
    };
    return colorMap[color] || "topic-gray";
  }

  function getAllTopicsTooltip(chatId: string): string {
    const allTopics = allTopicsMap.get(chatId) || [];
    if (allTopics.length <= 2) return "";
    return allTopics.map(t => `${t.display_name} (${Math.round(t.confidence * 100)}%)`).join("\n");
  }

  function hasNewMessages(chatId: string): boolean {
    return $conversationsStore.conversationsWithNewMessages.has(chatId);
  }

  // Get the primary identifier for a conversation (for avatar lookup)
  function getPrimaryIdentifier(conv: typeof $conversationsStore.conversations[0]): string | null {
    // For group chats, return null (use group icon)
    if (conv.is_group) return null;
    // For individual chats, use the first participant
    return conv.participants[0] || null;
  }

  // Get initials for fallback avatar
  function getInitials(name: string): string {
    const parts = name.trim().split(/\s+/);
    if (parts.length >= 2) {
      return `${parts[0][0]}${parts[parts.length - 1][0]}`.toUpperCase();
    }
    return parts[0]?.[0]?.toUpperCase() || "?";
  }
</script>

<div class="conversation-list">
  <div class="header">
    <h2>Messages</h2>
  </div>

  <div class="search">
    <input type="text" placeholder="Search conversations..." />
  </div>

  {#if $conversationsStore.loading}
    <ConversationSkeleton />
  {:else if $conversationsStore.error}
    <div class="error">{$conversationsStore.error}</div>
  {:else if $conversationsStore.conversations.length === 0}
    <div class="empty">No conversations found</div>
  {:else}
    <div class="list" bind:this={listRef} role="listbox" aria-label="Conversations">
      {#each $conversationsStore.conversations as conv, index (conv.chat_id)}
        {@const identifier = getPrimaryIdentifier(conv)}
        {@const avatarUrl = identifier ? avatarUrls.get(identifier) : null}
        {@const avatarState = identifier ? avatarStates.get(identifier) : null}
        {@const isFocused = focusedIndex === index}
        <button
          bind:this={itemRefs[index]}
          class="conversation"
          class:active={$conversationsStore.selectedChatId === conv.chat_id}
          class:focused={isFocused}
          class:group={conv.is_group}
          class:has-new={hasNewMessages(conv.chat_id)}
          onclick={() => { selectConversation(conv.chat_id); setConversationIndex(index); }}
          role="option"
          aria-selected={$conversationsStore.selectedChatId === conv.chat_id}
          tabindex={isFocused ? 0 : -1}
        >
          <div class="avatar-container">
            <div
              class="avatar"
              class:group={conv.is_group}
              class:has-image={avatarState === "loaded" && avatarUrl}
              use:observeAvatar={identifier || ""}
            >
              {#if conv.is_group}
                <svg viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm-6 8v-2c0-2.67 5.33-4 6-4s6 1.33 6 4v2H6zm10-8c1.93 0 3.5-1.57 3.5-3.5S17.93 5 16 5c-.54 0-1.04.13-1.5.35.63.89 1 1.98 1 3.15s-.37 2.26-1 3.15c.46.22.96.35 1.5.35z"/>
                </svg>
              {:else if avatarState === "loaded" && avatarUrl}
                <img src={avatarUrl} alt="" class="avatar-image" />
              {:else if avatarState === "loading"}
                <span class="avatar-loading">{getInitials(getDisplayName(conv))}</span>
              {:else}
                {getInitials(getDisplayName(conv))}
              {/if}
            </div>
            {#if hasNewMessages(conv.chat_id)}
              <span class="new-indicator" aria-label="New messages"></span>
            {/if}
          </div>
          <div class="info">
            <div class="name-row">
              <span class="name" class:has-new={hasNewMessages(conv.chat_id)}>
                {getDisplayName(conv)}
              </span>
              <span class="date" class:has-new={hasNewMessages(conv.chat_id)}>
                {formatDate(conv.last_message_date)}
              </span>
            </div>
            <div class="topics-row">
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
                  <span
                    class="topic-more"
                    title={getAllTopicsTooltip(conv.chat_id)}
                  >
                    +{(allTopicsMap.get(conv.chat_id)?.length || 0) - 2}
                  </span>
                {/if}
              {/if}
            </div>
            <div class="preview" class:has-new={hasNewMessages(conv.chat_id)}>
              {conv.last_message_text || "No messages"}
            </div>
          </div>
        </button>
      {/each}
    </div>
  {/if}

</div>

<style>
  .conversation-list {
    width: 300px;
    min-width: 300px;
    background: var(--bg-secondary);
    border-right: 1px solid var(--border-color);
    display: flex;
    flex-direction: column;
  }

  .header {
    padding: 16px;
    border-bottom: 1px solid var(--border-color);
  }

  .header h2 {
    font-size: 20px;
    font-weight: 600;
  }

  .search {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border-color);
  }

  .search input {
    width: 100%;
    padding: 8px 12px;
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    color: var(--text-primary);
    font-size: 14px;
  }

  .search input::placeholder {
    color: var(--text-secondary);
  }

  .list {
    flex: 1;
    overflow-y: auto;
  }

  .conversation {
    width: 100%;
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    background: transparent;
    border: none;
    cursor: pointer;
    text-align: left;
    transition: background 0.15s ease;
  }

  .conversation:hover {
    background: var(--bg-hover);
  }

  .conversation.active {
    background: var(--bg-active);
  }

  .conversation.focused {
    outline: 2px solid var(--color-primary, #007aff);
    outline-offset: -2px;
  }

  .conversation.focused:not(.active) {
    background: var(--bg-hover);
  }

  .avatar-container {
    position: relative;
    flex-shrink: 0;
  }

  .avatar {
    width: 44px;
    height: 44px;
    border-radius: 50%;
    background: var(--accent-color);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
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
    animation: avatarPulse 1.5s ease-in-out infinite;
  }

  @keyframes avatarPulse {
    0%, 100% {
      opacity: 0.7;
    }
    50% {
      opacity: 0.4;
    }
  }

  .new-indicator {
    position: absolute;
    top: 0;
    right: 0;
    width: 12px;
    height: 12px;
    background: #007aff;
    border-radius: 50%;
    border: 2px solid var(--bg-secondary);
    animation: newPulse 2s ease-in-out infinite;
  }

  @keyframes newPulse {
    0%, 100% {
      opacity: 1;
      transform: scale(1);
    }
    50% {
      opacity: 0.8;
      transform: scale(1.1);
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
    margin-bottom: 4px;
  }

  .name {
    font-weight: 500;
    color: var(--text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .name.has-new {
    font-weight: 700;
  }

  .date {
    font-size: 12px;
    color: var(--text-secondary);
    flex-shrink: 0;
    margin-left: 8px;
  }

  .date.has-new {
    color: #007aff;
    font-weight: 600;
  }

  .preview {
    font-size: 13px;
    color: var(--text-secondary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .topics-row {
    display: flex;
    gap: 4px;
    margin-bottom: 4px;
    flex-wrap: wrap;
  }

  .topic-tag {
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 8px;
    font-weight: 500;
    white-space: nowrap;
  }

  .topic-more {
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 8px;
    background: var(--bg-tertiary, #e5e5e5);
    color: var(--text-secondary);
    cursor: help;
  }

  /* Topic color variants */
  .topic-blue {
    background: #dbeafe;
    color: #1d4ed8;
  }

  .topic-green {
    background: #dcfce7;
    color: #15803d;
  }

  .topic-purple {
    background: #f3e8ff;
    color: #7c3aed;
  }

  .topic-pink {
    background: #fce7f3;
    color: #be185d;
  }

  .topic-orange {
    background: #ffedd5;
    color: #c2410c;
  }

  .topic-gray {
    background: #f3f4f6;
    color: #4b5563;
  }

  .topic-indigo {
    background: #e0e7ff;
    color: #4338ca;
  }

  .topic-amber {
    background: #fef3c7;
    color: #b45309;
  }

  .topic-cyan {
    background: #cffafe;
    color: #0e7490;
  }

  .topic-rose {
    background: #ffe4e6;
    color: #be123c;
  }

  .preview.has-new {
    color: var(--text-primary);
    font-weight: 500;
  }

  .loading,
  .error,
  .empty {
    padding: 24px 16px;
    text-align: center;
    color: var(--text-secondary);
  }

  .error {
    color: var(--error-color);
  }

</style>
