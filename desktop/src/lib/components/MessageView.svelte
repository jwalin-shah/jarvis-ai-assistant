<script lang="ts">
  import { onMount, onDestroy, tick } from 'svelte';
  import {
    conversationsStore,
    loadMoreMessages,
    stopMessagePolling,
    highlightedMessageId,
    scrollToMessageId,
    clearScrollTarget,
    addOptimisticMessage,
    updateOptimisticMessage,
    removeOptimisticMessage,
    clearOptimisticMessages,
    clearPrefetchedDraft,
    pollMessages,
    updateConversationAfterLocalSend,
  } from '../stores/conversations.svelte';
  import type { DraftSuggestion, Message } from '../types';
  import {
    activeZone,
    setActiveZone,
    messageIndex,
    setMessageIndex,
    announce,
  } from '../stores/keyboard';
  import { WS_HTTP_BASE } from '../api/websocket';
  import SuggestionBar from './SuggestionBar.svelte';
  import MessageSkeleton from './MessageSkeleton.svelte';
  import ContactHoverCard from './ContactHoverCard.svelte';
  import { MessageItem, DateHeader, ComposeArea } from './message-view';
  import { EmptyState } from './ui';
  import { MessageIcon } from './icons';
  import { formatDate, getMessageDateString } from '../utils/date';
  import { getNavAction, isTypingInInput } from '../utils/keyboard-nav';
  import { formatParticipant } from '../db';

  // Panel visibility state
  let showDraftPanel = $state(false);
  let prefetchedSuggestions = $state<DraftSuggestion[] | undefined>(undefined);

  // Contact hover state
  let hoverCardVisible = $state(false);
  let hoverCardX = $state(0);
  let hoverCardY = $state(0);
  let hoverTimeout: ReturnType<typeof setTimeout> | null = null;

  function handleAvatarMouseEnter(event: MouseEvent) {
    const rect = (event.currentTarget as HTMLElement).getBoundingClientRect();
    hoverCardX = rect.left;
    hoverCardY = rect.bottom + 8;
    
    if (hoverTimeout) clearTimeout(hoverTimeout);
    hoverTimeout = setTimeout(() => {
      hoverCardVisible = true;
    }, 400);
  }

  function handleAvatarMouseLeave() {
    if (hoverTimeout) clearTimeout(hoverTimeout);
    hoverCardVisible = false;
  }

  // Keyboard navigation state

  let focusedMessageIndex = $state(-1);

  // Sync with keyboard store
  $effect(() => {
    focusedMessageIndex = $messageIndex;
  });

  // Compose state
  let sendingMessage = $state(false);
  let composeAreaRef = $state<InstanceType<typeof ComposeArea> | null>(null);

  // Virtual scrolling configuration
  const ESTIMATED_MESSAGE_HEIGHT = 80;
  const BUFFER_SIZE = 10;
  const MIN_VISIBLE_MESSAGES = 20;
  const VIRTUALIZATION_THRESHOLD = 120;

  // Virtual scrolling state
  let messageHeights = $state<Map<number, number>>(new Map());
  let visibleStartIndex = $state(0);
  let visibleEndIndex = $state(MIN_VISIBLE_MESSAGES);
  let virtualTopPadding = $state(0);
  let virtualBottomPadding = $state(0);
  let suppressScrollRecalc = $state(false);

  // Scroll tracking
  let messagesContainer = $state<HTMLDivElement | null>(null);
  let loadEarlierSectionEl = $state<HTMLDivElement | null>(null);
  let isAtBottom = $state(true);
  let hasNewMessagesBelow = $state(false);
  let newMessageIds = $state<Set<number>>(new Set());

  // Message change tracking
  let previousMessageCount = $state(0);
  let lastMessageCount = $state(0);
  let lastLoadingState = $state(false);
  let needsScrollToBottom = $state(false);
  let prevSelectedChatId = $state<string | null>(null);

  // Batched height updates via ResizeObserver - collects changes in a rAF to avoid
  // cloning the Map on every individual resize callback
  let pendingHeightUpdates = new Map<number, number>();
  let heightUpdateRafId: number | null = null;

  function getScrollOffsets() {
    const style = messagesContainer ? getComputedStyle(messagesContainer) : null;
    const paddingTop = style ? parseFloat(style.paddingTop) || 0 : 0;
    const paddingBottom = style ? parseFloat(style.paddingBottom) || 0 : 0;
    const loadEarlierHeight = loadEarlierSectionEl?.offsetHeight ?? 0;
    return { paddingTop, paddingBottom, loadEarlierHeight };
  }

  function handleMessageHeightChange(messageId: number, height: number) {
    if (!messageId || height <= 0 || messageHeights.get(messageId) === height) return;

    pendingHeightUpdates.set(messageId, height);

    if (heightUpdateRafId === null) {
      heightUpdateRafId = requestAnimationFrame(() => {
        heightUpdateRafId = null;
        if (pendingHeightUpdates.size === 0) return;

        const newHeights = new Map(messageHeights);
        for (const [id, h] of pendingHeightUpdates) {
          newHeights.set(id, h);
        }
        pendingHeightUpdates.clear();
        messageHeights = newHeights;
      });
    }
  }

  function getMessageHeight(messageId: number): number {
    return messageHeights.get(messageId) ?? ESTIMATED_MESSAGE_HEIGHT;
  }

  // Cached cumulative heights array - recomputed only when messages or heights change
  // cumulativeHeights[i] = sum of heights for messages[0..i-1], cumulativeHeights[0] = 0
  let cumulativeHeights = $derived.by(() => {
    const messages = conversationsStore.messages;
    // Access messageHeights to track it as a dependency
    const _heights = messageHeights;
    const cumulative = new Float64Array(messages.length + 1);
    for (let i = 0; i < messages.length; i++) {
      cumulative[i + 1] = cumulative[i]! + (_heights.get(messages[i]!.id) ?? ESTIMATED_MESSAGE_HEIGHT);
    }
    return cumulative;
  });

  function calculateVisibleRange(scrollTop: number, containerHeight: number) {
    const messages = conversationsStore.messages;
    const cumHeights = cumulativeHeights;
    if (messages.length === 0) {
      visibleStartIndex = 0;
      visibleEndIndex = 0;
      virtualTopPadding = 0;
      virtualBottomPadding = 0;
      return;
    }
    if (messages.length <= VIRTUALIZATION_THRESHOLD) {
      visibleStartIndex = 0;
      visibleEndIndex = messages.length;
      virtualTopPadding = 0;
      virtualBottomPadding = 0;
      return;
    }

    const { paddingTop, paddingBottom, loadEarlierHeight } = getScrollOffsets();
    const adjustedScrollTop = Math.max(0, scrollTop - loadEarlierHeight - paddingTop);
    const visibleHeight = Math.max(0, containerHeight - paddingTop - paddingBottom);
    const totalHeight = cumHeights[messages.length]!;

    // Clamp to full-tail rendering near bottom to avoid oscillating bottom spacer.
    if (adjustedScrollTop + visibleHeight >= totalHeight - 2 * ESTIMATED_MESSAGE_HEIGHT) {
      visibleEndIndex = messages.length;
      visibleStartIndex = Math.max(0, messages.length - MIN_VISIBLE_MESSAGES - BUFFER_SIZE);
      virtualTopPadding = cumHeights[visibleStartIndex]!;
      virtualBottomPadding = 0;
      return;
    }

    // Binary search to find first message whose bottom edge >= adjustedScrollTop
    let lo = 0;
    let hi = messages.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (cumHeights[mid + 1]! < adjustedScrollTop) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    let startIdx = Math.max(0, lo - BUFFER_SIZE);

    const topPadding = cumHeights[startIdx]!;

    // Binary search to find first message whose top edge > adjustedScrollTop + containerHeight
    const scrollBottom =
      adjustedScrollTop + visibleHeight + BUFFER_SIZE * ESTIMATED_MESSAGE_HEIGHT;
    lo = startIdx;
    hi = messages.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (cumHeights[mid]! < scrollBottom) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    let endIdx = Math.min(messages.length, Math.max(lo, startIdx + MIN_VISIBLE_MESSAGES));

    const bottomPadding = totalHeight - cumHeights[endIdx]!;

    visibleStartIndex = startIdx;
    visibleEndIndex = endIdx;
    virtualTopPadding = topPadding;
    virtualBottomPadding = bottomPadding;
  }

  function getVisibleMessages(): Message[] {
    const allMessages = conversationsStore.messagesWithOptimistic;
    // Extend end index to include optimistic messages if we're showing the tail
    const effectiveEnd = visibleEndIndex >= conversationsStore.messages.length
      ? allMessages.length
      : visibleEndIndex;
    return allMessages.slice(visibleStartIndex, effectiveEnd);
  }

  let rafPending = false;
  function updateVirtualScroll() {
    if (!messagesContainer || rafPending || suppressScrollRecalc) return;
    rafPending = true;
    requestAnimationFrame(() => {
      rafPending = false;
      if (!messagesContainer) return;
      const { scrollTop, clientHeight } = messagesContainer;
      calculateVisibleRange(scrollTop, clientHeight);
    });
  }

  async function scrollToMessage(messageId: number) {
    const messages = conversationsStore.messages;
    const msgIndex = messages.findIndex((m) => m.id === messageId);
    if (msgIndex === -1) return;

    const { paddingTop, paddingBottom, loadEarlierHeight } = getScrollOffsets();
    const scrollPosition = cumulativeHeights[msgIndex]! + loadEarlierHeight + paddingTop;

    visibleStartIndex = Math.max(0, msgIndex - BUFFER_SIZE);
    visibleEndIndex = Math.min(messages.length, msgIndex + BUFFER_SIZE + MIN_VISIBLE_MESSAGES);

    virtualTopPadding = cumulativeHeights[visibleStartIndex]!;
    virtualBottomPadding = cumulativeHeights[messages.length]! - cumulativeHeights[visibleEndIndex]!;

    await tick();

    if (messagesContainer) {
      const containerHeight = messagesContainer.clientHeight;
      const visibleHeight = Math.max(0, containerHeight - paddingTop - paddingBottom);
      const targetScroll = scrollPosition - visibleHeight / 2 + getMessageHeight(messageId) / 2;
      messagesContainer.scrollTo({ top: Math.max(0, targetScroll), behavior: 'smooth' });
    }

    await tick();
  }

  // Send message
  async function handleSendMessage(text: string, retryId?: string) {
    if (!conversationsStore.selectedConversation) return;
    sendingMessage = true;

    let optimisticId: string;
    if (retryId) {
      optimisticId = retryId;
      updateOptimisticMessage(optimisticId, { status: 'sending' });
    } else {
      optimisticId = addOptimisticMessage(text);
      // Ensure optimistic message is visible - it gets appended to messagesWithOptimistic
      // So we need to include it in the visible range
      isAtBottom = true;
      const totalWithOptimistic = conversationsStore.messages.length + conversationsStore.optimisticMessages.length;
      visibleEndIndex = totalWithOptimistic;
      visibleStartIndex = Math.max(0, totalWithOptimistic - MIN_VISIBLE_MESSAGES - BUFFER_SIZE);
      virtualTopPadding = Math.max(0, (visibleStartIndex - conversationsStore.messages.length) * ESTIMATED_MESSAGE_HEIGHT);
      virtualBottomPadding = 0;
      scrollToBottom();
    }

    try {
      const chatId = conversationsStore.selectedConversation.chat_id;
      const isGroup = conversationsStore.selectedConversation.is_group;
      const recipient =
        !isGroup && conversationsStore.selectedConversation.participants?.length > 0
          ? conversationsStore.selectedConversation.participants[0]
          : undefined;

      const response = await fetch(
        `${WS_HTTP_BASE}/conversations/${encodeURIComponent(chatId)}/send`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text,
            recipient,
            is_group: isGroup,
          }),
        }
      );

      if (!response.ok) {
        await response.text();
        updateOptimisticMessage(optimisticId, {
          status: 'failed',
          error: `Send failed: ${response.status}`,
        });
        return;
      }

      const result = await response.json();

      if (result.success) {
        updateOptimisticMessage(optimisticId, { status: 'sent' });
        updateConversationAfterLocalSend(chatId, text);
        // Watcher push (handleNewMessagePush) clears optimistic when real message arrives.
        // Proactively poll after 1.5s to catch the chat.db update from AppleScript
        setTimeout(() => pollMessages(), 1500);
        // Safety timeout: auto-clear if watcher push doesn't arrive within 10s.
        setTimeout(() => removeOptimisticMessage(optimisticId), 10000);
      } else {
        updateOptimisticMessage(optimisticId, {
          status: 'failed',
          error: result.error || 'Failed to send message',
        });
      }
    } catch (err) {
      updateOptimisticMessage(optimisticId, {
        status: 'failed',
        error: 'Failed to send. Check Messages app permissions.',
      });
      console.error('Send error:', err);
    } finally {
      sendingMessage = false;
    }
  }

  function handleRetry(optimisticId: string) {
    const msg = conversationsStore.optimisticMessages.find((m) => m.id === optimisticId);
    if (msg) {
      handleSendMessage(msg.text, optimisticId);
    }
  }

  function handleDismissFailedMessage(optimisticId: string) {
    removeOptimisticMessage(optimisticId);
  }

  let previousLastMessageId = $state<number | null>(null);

  const SCROLL_THRESHOLD = 200;

  async function handleScroll(event: Event) {
    const container = event.target as HTMLDivElement;
    if (!container) return;

    if (!suppressScrollRecalc) {
      updateVirtualScroll();
    }

    // Load more when near top
    if (
      container.scrollTop < SCROLL_THRESHOLD &&
      conversationsStore.hasMore &&
      !conversationsStore.loadingMore &&
      conversationsStore.messages.length > 0
    ) {
      await loadOlderAndMaintainScroll();
    }

    // Check if at bottom
    const { scrollTop, scrollHeight, clientHeight } = container;
    const threshold = 150;
    isAtBottom = scrollHeight - scrollTop - clientHeight < threshold;

    if (isAtBottom) {
      hasNewMessagesBelow = false;
    }
  }

  /**
   * Load older messages and maintain scroll position.
   * Uses scrollHeight diff to compensate for added content above the viewport,
   * avoiding reliance on estimated heights which cause "yeeting".
   */
  async function loadOlderAndMaintainScroll() {
    if (!messagesContainer) return;

    suppressScrollRecalc = true;
    const prevCount = conversationsStore.messages.length;
    const prevScrollTop = messagesContainer.scrollTop;
    const prevScrollHeight = messagesContainer.scrollHeight;

    await loadMoreMessages();

    const addedCount = conversationsStore.messages.length - prevCount;
    if (addedCount > 0 && messagesContainer) {
      if (conversationsStore.messages.length > VIRTUALIZATION_THRESHOLD) {
        // Shift visible range to keep the same messages rendered
        visibleStartIndex += addedCount;
        visibleEndIndex += addedCount;

        // Update padding using cumulative heights
        virtualTopPadding = cumulativeHeights[visibleStartIndex]!;
        const totalLen = conversationsStore.messages.length;
        virtualBottomPadding = cumulativeHeights[totalLen]! - cumulativeHeights[visibleEndIndex]!;
      }

      await tick();

      // Compensate based on actual rendered delta to avoid jumpy scroll behavior.
      const newScrollHeight = messagesContainer.scrollHeight;
      const addedHeight = Math.max(0, newScrollHeight - prevScrollHeight);
      messagesContainer.scrollTop = prevScrollTop + addedHeight;
    }

    requestAnimationFrame(() => {
      suppressScrollRecalc = false;
      updateVirtualScroll();
    });
  }

  async function handleLoadEarlier() {
    await loadOlderAndMaintainScroll();
  }

  // Effects
  $effect(() => {
    const currentCount = conversationsStore.messages.length;
    const currentLastId = conversationsStore.messages[currentCount - 1]?.id ?? null;

    if (currentCount > previousMessageCount && previousMessageCount > 0) {
      // Only treat as "new messages" if last message changed (appended at end).
      // When loading older messages (prepended), last message stays the same.
      if (currentLastId !== previousLastMessageId) {
        const newMessages = conversationsStore.messages.slice(previousMessageCount);
        const newIds = new Set(newMessageIds);
        newMessages.forEach((m) => newIds.add(m.id));
        newMessageIds = newIds;

        if (isAtBottom) {
          // Include optimistic messages in the total count for proper visibility
          const optimisticCount = conversationsStore.optimisticMessages.length;
          const msgLen = conversationsStore.messages.length + optimisticCount;
          visibleEndIndex = msgLen;
          visibleStartIndex = Math.max(0, msgLen - MIN_VISIBLE_MESSAGES - BUFFER_SIZE);
          // Calculate top padding based on real messages only (optimistic don't have heights yet)
          virtualTopPadding = visibleStartIndex <= conversationsStore.messages.length
            ? (cumulativeHeights[visibleStartIndex] ?? visibleStartIndex * ESTIMATED_MESSAGE_HEIGHT)
            : conversationsStore.messages.length * ESTIMATED_MESSAGE_HEIGHT + (visibleStartIndex - conversationsStore.messages.length) * ESTIMATED_MESSAGE_HEIGHT;
          virtualBottomPadding = 0;

          suppressScrollRecalc = true;
          scrollToBottom().then(() => {
            requestAnimationFrame(() => {
              suppressScrollRecalc = false;
            });
          });
        } else {
          hasNewMessagesBelow = true;
        }

        setTimeout(() => {
          newMessageIds = new Set();
        }, 1000);
      }
    }
    previousMessageCount = currentCount;
    previousLastMessageId = currentLastId;
  });

  $effect(() => {
    const msgCount = conversationsStore.messages.length;
    const isLoading = conversationsStore.loadingMessages;

    const countChanged = msgCount !== lastMessageCount;
    const loadingChanged = isLoading !== lastLoadingState;
    lastLoadingState = isLoading;
    if (!countChanged && !loadingChanged) return;
    if (!countChanged) return;
    lastMessageCount = msgCount;

    if (!isLoading && msgCount === 0 && conversationsStore.error && needsScrollToBottom) {
      needsScrollToBottom = false;
      return;
    }

    if (messagesContainer && msgCount > 0 && !isLoading) {
      if (needsScrollToBottom) {
        visibleEndIndex = msgCount;
        visibleStartIndex = Math.max(0, msgCount - MIN_VISIBLE_MESSAGES - BUFFER_SIZE);
        virtualTopPadding = cumulativeHeights[visibleStartIndex] ?? visibleStartIndex * ESTIMATED_MESSAGE_HEIGHT;
        virtualBottomPadding = 0;

        needsScrollToBottom = false;

        suppressScrollRecalc = true;
        tick().then(async () => {
          await scrollToBottom(true);
          await tick();
          await scrollToBottom(true);
          requestAnimationFrame(() => {
            suppressScrollRecalc = false;
          });
        });
      } else {
        updateVirtualScroll();
      }
    }
  });

  $effect(() => {
    const chatId = conversationsStore.selectedChatId;
    if (chatId && chatId !== prevSelectedChatId) {
      const oldChatId = prevSelectedChatId;
      prevSelectedChatId = chatId;
      previousMessageCount = 0;
      newMessageIds = new Set();
      hasNewMessagesBelow = false;
      isAtBottom = true;
      needsScrollToBottom = true;
      lastMessageCount = 0;
      messageHeights = new Map();
      visibleStartIndex = 0;
      visibleEndIndex = MIN_VISIBLE_MESSAGES;
      virtualTopPadding = 0;
      virtualBottomPadding = 0;
      // Auto-open AI drafts on chat focus so generation starts immediately.
      prefetchedSuggestions = undefined;
      showDraftPanel = true;
      if (oldChatId) {
        clearOptimisticMessages(oldChatId);
      }
    } else if (!chatId) {
      prevSelectedChatId = null;
    }
  });

  $effect(() => {
    const draft = conversationsStore.prefetchedDraft;
    const chatId = conversationsStore.selectedChatId;
    if (draft && chatId && draft.chatId === chatId && !showDraftPanel) {
      prefetchedSuggestions = draft.suggestions;
      showDraftPanel = true;
      clearPrefetchedDraft();
    }
  });

  async function scrollToBottom(instant = false) {
    await tick();
    await tick();
    if (!messagesContainer) return;

    messagesContainer.scrollTo({
      top: messagesContainer.scrollHeight,
      behavior: instant ? 'auto' : 'smooth',
    });

    // One more frame helps after late height updates (images, font/layout, observers).
    requestAnimationFrame(() => {
      if (!messagesContainer) return;
      messagesContainer.scrollTop = messagesContainer.scrollHeight;
      hasNewMessagesBelow = false;
      updateVirtualScroll();
    });
  }

  function handleNewMessagesClick() {

    const messages = conversationsStore.messages;
    visibleEndIndex = messages.length;
    visibleStartIndex = Math.max(0, messages.length - MIN_VISIBLE_MESSAGES - BUFFER_SIZE);

    virtualTopPadding = cumulativeHeights[visibleStartIndex]!;
    virtualBottomPadding = 0;

    scrollToBottom();
  }

  function handleKeydown(event: KeyboardEvent) {
    const isMod = event.metaKey || event.ctrlKey;

    if ($activeZone !== 'messages' && $activeZone !== null) return;
    if (isTypingInInput(event)) {
      if (event.key === 'Escape') {
        event.preventDefault();
        (event.target as HTMLElement).blur();
        setActiveZone('messages');
      }
      return;
    }
    if (!conversationsStore.selectedConversation) return;

    // Component-specific shortcuts
    if (isMod && event.key === 'd') {
      event.preventDefault();
      prefetchedSuggestions = undefined;
      showDraftPanel = true;
      return;
    }

    if (event.key === 'r') {
      event.preventDefault();
      setActiveZone('compose');
      document.querySelector<HTMLTextAreaElement>('.compose-input')?.focus();
      announce('Composing reply');
      return;
    }

    if (event.key === 'ArrowLeft' || event.key === 'h') {
      event.preventDefault();
      setActiveZone('conversations');
      setMessageIndex(-1);
      focusedMessageIndex = -1;
      announce('Returned to conversations list');
      return;
    }

    const messages = conversationsStore.messages;
    if (messages.length === 0) return;

    const maxIndex = messages.length - 1;
    const action = getNavAction(event.key, event.shiftKey, focusedMessageIndex, maxIndex);
    if (!action) return;

    event.preventDefault();

    if (action.type === 'escape') {
      setMessageIndex(-1);
      focusedMessageIndex = -1;
      setActiveZone(null);
      return;
    }

    setActiveZone('messages');
    const newIndex = action.type === 'first' ? 0 : action.index;
    setMessageIndex(newIndex);
    focusedMessageIndex = newIndex;
    scrollToMessageByIndex(newIndex);
    announceMessage(messages[newIndex]!);
  }

  function scrollToMessageByIndex(index: number) {
    const messages = conversationsStore.messages;
    if (index < 0 || index >= messages.length) return;

    const messageId = messages[index]!.id;
    // Reuse the same scrollToMessage logic which uses container-scoped scrollTo
    scrollToMessage(messageId);
  }

  function announceMessage(message: Message) {
    const sender = message.is_from_me
      ? 'You'
      : message.sender_name || message.sender || 'Contact';
    const time = formatDate(message.date);
    const text = message.text || 'Attachment';
    announce(`${sender} at ${time}: ${text.slice(0, 100)}${text.length > 100 ? '...' : ''}`);
  }

  function handleDraftSelect(text: string) {
    let normalized = text.replace(/\r/g, '').trim();
    normalized = normalized.replace(/^["'`]+|["'`]+$/g, '').trim();
    normalized = normalized.replace(/^(?:\(?\d{1,2}\)?[.):\-]\s*|[-*\u2022]\s+)/, '').trim();
    normalized = normalized.replace(/:\s*$/, '').trim();

    if (normalized && composeAreaRef) {
      composeAreaRef.setValue(normalized);
    }
    showDraftPanel = false;
    prefetchedSuggestions = undefined;
  }

  // Date header logic
  function shouldShowDateHeader(visibleIndex: number): boolean {
    const actualIndex = visibleStartIndex + visibleIndex;
    const messages = conversationsStore.messages;

    if (actualIndex === 0) return true;
    if (actualIndex >= messages.length) return false;

    const curr = getMessageDateString(messages[actualIndex]!.id, messages[actualIndex]!.date);
    const prev = getMessageDateString(messages[actualIndex - 1]!.id, messages[actualIndex - 1]!.date);
    return curr !== prev;
  }

  function isNewMessage(messageId: number): boolean {
    return newMessageIds.has(messageId);
  }

  let visibleMessages = $derived(getVisibleMessages());
  let unsubscribeScroll: (() => void) | null = null;

  onMount(() => {
    window.addEventListener('keydown', handleKeydown);

    unsubscribeScroll = scrollToMessageId.subscribe(async (messageId) => {
      if (messageId !== null) {
        await scrollToMessage(messageId);
        clearScrollTarget();
      }
    });
  });

  onDestroy(() => {
    window.removeEventListener('keydown', handleKeydown);
    stopMessagePolling();
    unsubscribeScroll?.();
  });
</script>

<div class="message-view" tabindex="-1">
  {#if !conversationsStore.selectedConversation}
    <EmptyState
      title="Select a conversation"
      description="Choose a conversation from the list to view messages"
    >
      {#snippet icon()}
        <MessageIcon />
      {/snippet}
    </EmptyState>
  {:else}
    <div class="header">
      <div 
        class="avatar" 
        class:group={conversationsStore.selectedConversation.is_group}
        onmouseenter={handleAvatarMouseEnter}
        onmouseleave={handleAvatarMouseLeave}
      >
        {#if conversationsStore.selectedConversation.is_group}
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path
              d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm-6 8v-2c0-2.67 5.33-4 6-4s6 1.33 6 4v2H6z"
            />
          </svg>
        {:else}
          {(
            conversationsStore.selectedConversation.display_name ||
            formatParticipant(conversationsStore.selectedConversation.participants[0] ?? '') ||
            '?'
          )
            .charAt(0)
            .toUpperCase()}
        {/if}
      </div>
      <div class="info">
        <h2>
          {conversationsStore.selectedConversation.display_name || conversationsStore.selectedConversation.participants.map(p => formatParticipant(p)).join(', ')}
        </h2>
      </div>
      <div class="header-actions">

        <button
          class="action-btn primary"
          onclick={() => {
            prefetchedSuggestions = undefined;
            showDraftPanel = true;
          }}
          title="Generate AI reply (Cmd+D)"
          aria-label="Generate AI reply"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M12 20h9"></path>
            <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"></path>
          </svg>
          <span>AI Draft</span>
        </button>
      </div>
    </div>

    <div class="messages" bind:this={messagesContainer} onscroll={handleScroll}>
      {#if conversationsStore.loadingMessages}
        <MessageSkeleton />
      {:else if conversationsStore.messages.length === 0}
        <div class="empty">No messages in this conversation</div>
      {:else}
        <!-- Load earlier messages section -->
        <div class="load-earlier-section" bind:this={loadEarlierSectionEl}>
          {#if conversationsStore.loadingMore}
            <div class="loading-more">
              <div class="spinner"></div>
              <span>Loading earlier messages...</span>
            </div>
          {:else if conversationsStore.hasMore}
            <button
              class="load-earlier-btn"
              onclick={handleLoadEarlier}
              aria-label="Load earlier messages"
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="17 11 12 6 7 11"></polyline>
                <line x1="12" y1="6" x2="12" y2="18"></line>
              </svg>
              Load earlier messages
            </button>
          {:else}
            <div class="end-of-history">
              <span>Beginning of conversation</span>
            </div>
          {/if}
        </div>

        <!-- Virtual scroll spacer for messages above visible range -->
        <div class="virtual-spacer-top" style="height: {virtualTopPadding}px;"></div>

        <!-- Only render visible messages -->
        <div class="virtual-content">
          {#each visibleMessages as message, visibleIndex (message.id)}
            {#if shouldShowDateHeader(visibleIndex)}
              <DateHeader date={message.date} />
            {/if}

            <MessageItem
              {message}
              isGroup={conversationsStore.selectedConversation.is_group}
              isHighlighted={$highlightedMessageId === message.id}
              isKeyboardFocused={focusedMessageIndex === visibleStartIndex + visibleIndex}
              isNew={isNewMessage(message.id)}
              onRetry={handleRetry}
              onDismiss={handleDismissFailedMessage}
              onHeightChange={handleMessageHeightChange}
            />
          {/each}
        </div>

        <!-- Virtual scroll spacer for messages below visible range -->
        <div class="virtual-spacer-bottom" style="height: {virtualBottomPadding}px;"></div>
      {/if}
    </div>

    {#if !isAtBottom || hasNewMessagesBelow}
      <button
        class="scroll-to-bottom-fab"
        onclick={handleNewMessagesClick}
        onkeydown={(e: KeyboardEvent) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            handleNewMessagesClick();
          }
        }}
        tabindex="0"
        aria-label="Scroll to bottom"
      >
        {#if newMessageIds.size > 0}
          <span class="fab-badge">{newMessageIds.size > 99 ? '99+' : newMessageIds.size}</span>
        {/if}
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="6 9 12 15 18 9"></polyline>
        </svg>
      </button>
    {/if}

    {#if showDraftPanel && conversationsStore.selectedConversation}
      {#key conversationsStore.selectedConversation.chat_id}
        <SuggestionBar
          chatId={conversationsStore.selectedConversation.chat_id}
          onSelect={handleDraftSelect}
          onClose={() => {
            showDraftPanel = false;
            prefetchedSuggestions = undefined;
          }}
          {...(prefetchedSuggestions && { initialSuggestions: prefetchedSuggestions })}
        />
      {/key}
    {/if}

    {#key conversationsStore.selectedConversation?.chat_id || 'none'}
      <ComposeArea
        bind:this={composeAreaRef}
        onSend={(text) => handleSendMessage(text)}
        disabled={!conversationsStore.selectedConversation}
        sending={sendingMessage}
      />
    {/key}

    {#if conversationsStore.selectedConversation && !conversationsStore.selectedConversation.is_group}
      <ContactHoverCard
        identifier={conversationsStore.selectedConversation.participants[0] || ''}
        bind:visible={hoverCardVisible}
        x={hoverCardX}
        y={hoverCardY}
      />
    {/if}
  {/if}
</div>


<style>
  .message-view {
    flex: 1;
    display: flex;
    flex-direction: column;
    background: var(--surface-base);
    position: relative;
    min-width: 0;
    overflow: hidden;
  }

  .header {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    padding: var(--space-4);
    border-bottom: 1px solid var(--border-default);
    background: var(--surface-elevated);
  }

  .avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background: var(--color-primary);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: var(--font-weight-semibold);
    color: white;
    flex-shrink: 0;
  }

  .avatar.group {
    background: var(--group-color);
  }

  .avatar svg {
    width: 20px;
    height: 20px;
  }

  .info {
    min-width: 0;
    flex: 1;
  }

  .info h2 {
    font-size: var(--text-base);
    font-weight: var(--font-weight-semibold);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    margin: 0;
  }

  .info p {
    font-size: var(--text-xs);
    color: var(--text-secondary);
    margin: 0;
  }

  .header-actions {
    display: flex;
    gap: var(--space-2);
    margin-left: auto;
  }

  .action-btn {
    display: flex;
    align-items: center;
    gap: var(--space-1);
    padding: var(--space-2) var(--space-3);
    background: var(--surface-base);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-md);
    color: var(--text-secondary);
    cursor: pointer;
    transition: all var(--duration-fast) var(--ease-out);
    font-size: var(--text-sm);
  }

  .action-btn:hover {
    background: var(--surface-hover);
    color: var(--text-primary);
    border-color: var(--color-primary);
  }

  .action-btn svg {
    width: 16px;
    height: 16px;
  }

  .action-btn.primary {
    background: var(--color-primary);
    border-color: var(--color-primary);
    color: white;
  }

  .action-btn.primary:hover {
    background: var(--color-primary-hover);
  }

  .messages {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    padding: var(--space-4);
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    overflow-anchor: none;
    overscroll-behavior-y: contain;
  }

  .virtual-spacer-top,
  .virtual-spacer-bottom {
    flex-shrink: 0;
    overflow-anchor: none;
  }

  .virtual-content {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }

  .empty {
    text-align: center;
    color: var(--text-secondary);
    padding: var(--space-6);
  }

  /* Load earlier messages section */
  .load-earlier-section {
    display: flex;
    justify-content: center;
    padding: var(--space-4) 0;
    min-height: 48px;
    flex-shrink: 0;
  }

  .loading-more {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    color: var(--text-secondary);
    font-size: var(--text-sm);
  }

  .spinner {
    width: 16px;
    height: 16px;
    border: 2px solid var(--border-default);
    border-top-color: var(--color-primary);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  .load-earlier-btn {
    display: flex;
    align-items: center;
    gap: var(--space-1);
    padding: var(--space-2) var(--space-4);
    background: var(--surface-elevated);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-full);
    color: var(--text-secondary);
    font-size: var(--text-sm);
    cursor: pointer;
    transition: all var(--duration-fast) var(--ease-out);
  }

  .load-earlier-btn:hover {
    background: var(--surface-hover);
    color: var(--text-primary);
    border-color: var(--color-primary);
  }

  .load-earlier-btn svg {
    width: 14px;
    height: 14px;
  }

  .end-of-history {
    text-align: center;
    font-size: var(--text-xs);
    color: var(--text-secondary);
    opacity: 0.7;
  }

  .end-of-history span {
    background: var(--surface-elevated);
    padding: var(--space-1) var(--space-3);
    border-radius: var(--radius-full);
  }

  .scroll-to-bottom-fab {
    position: absolute;
    bottom: 80px;
    right: var(--space-4);
    display: flex;
    align-items: center;
    justify-content: center;
    width: 40px;
    height: 40px;
    background: var(--surface-elevated);
    color: var(--text-secondary);
    border: 1px solid var(--border-default);
    border-radius: 50%;
    cursor: pointer;
    box-shadow: var(--shadow-md);
    transition: all var(--duration-fast) var(--ease-out);
    animation: fadeInUp var(--duration-fast) var(--ease-out);
    z-index: var(--z-sticky);
  }

  .scroll-to-bottom-fab:hover {
    background: var(--color-primary);
    color: white;
    border-color: var(--color-primary);
    transform: scale(1.1);
  }

  .scroll-to-bottom-fab svg {
    width: 18px;
    height: 18px;
  }

  .fab-badge {
    position: absolute;
    top: -6px;
    right: -6px;
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
  }

  @keyframes fadeInUp {
    from {
      opacity: 0;
      transform: translateY(10px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
</style>
