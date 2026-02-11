<script lang="ts">
  import { onMount, onDestroy, tick } from 'svelte';
  import {
    conversationsStore,
    loadMoreMessages,
    pollMessages,
    stopMessagePolling,
    highlightedMessageId,
    scrollToMessageId,
    clearScrollTarget,
    addOptimisticMessage,
    updateOptimisticMessage,
    removeOptimisticMessage,
    clearOptimisticMessages,
    clearPrefetchedDraft,
  } from '../stores/conversations';
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
  import { MessageItem, DateHeader, ComposeArea } from './message-view';
  import { EmptyState } from './ui';
  import { MessageIcon } from './icons';
  import { formatDate, getMessageDateString } from '../utils/date';

  // Panel visibility state
  let showDraftPanel = $state(false);
  let prefetchedSuggestions = $state<DraftSuggestion[] | undefined>(undefined);

  // Keyboard navigation state
  let focusedMessageIndex = $state(-1);

  // Sync with keyboard store
  $effect(() => {
    focusedMessageIndex = $messageIndex;
  });

  // Compose state
  let sendingMessage = $state(false);

  // Virtual scrolling configuration
  const ESTIMATED_MESSAGE_HEIGHT = 80;
  const BUFFER_SIZE = 10;
  const MIN_VISIBLE_MESSAGES = 20;

  // Virtual scrolling state
  let messageHeights = $state<Map<number, number>>(new Map());
  let visibleStartIndex = $state(0);
  let visibleEndIndex = $state(MIN_VISIBLE_MESSAGES);
  let virtualTopPadding = $state(0);
  let virtualBottomPadding = $state(0);
  let suppressScrollRecalc = $state(false);

  // Scroll tracking
  let messagesContainer = $state<HTMLDivElement | null>(null);
  let isAtBottom = $state(true);
  let hasNewMessagesBelow = $state(false);
  let newMessageIds = $state<Set<number>>(new Set());

  // Message change tracking
  let previousMessageCount = $state(0);
  let lastMessageCount = $state(0);
  let lastLoadingState = $state(false);
  let needsScrollToBottom = $state(false);
  let prevSelectedChatId = $state<string | null>(null);

  // Debounce utility
  function debounce<T extends (...args: unknown[]) => void>(
    fn: T,
    delay: number
  ): (...args: Parameters<T>) => void {
    let timeoutId: ReturnType<typeof setTimeout> | null = null;
    return (...args: Parameters<T>) => {
      if (timeoutId) clearTimeout(timeoutId);
      timeoutId = setTimeout(() => fn(...args), delay);
    };
  }

  const debouncedMeasureHeights = debounce(() => {
    measureVisibleMessages();
  }, 150);

  function getMessageHeight(messageId: number): number {
    return messageHeights.get(messageId) ?? ESTIMATED_MESSAGE_HEIGHT;
  }

  function measureVisibleMessages() {
    if (!messagesContainer) return;

    const messageElements = messagesContainer.querySelectorAll('[data-message-id]');
    let needsUpdate = false;
    const newHeights = new Map(messageHeights);

    messageElements.forEach((el) => {
      const messageId = parseInt(el.getAttribute('data-message-id') || '0', 10);
      const rect = el.getBoundingClientRect();
      const style = window.getComputedStyle(el);
      const marginTop = parseFloat(style.marginTop) || 0;
      const marginBottom = parseFloat(style.marginBottom) || 0;
      const height = rect.height + marginTop + marginBottom;

      if (messageId && height > 0 && newHeights.get(messageId) !== height) {
        newHeights.set(messageId, height);
        needsUpdate = true;
      }
    });

    if (needsUpdate) {
      messageHeights = newHeights;
    }
  }

  function calculateVisibleRange(scrollTop: number, containerHeight: number) {
    const messages = conversationsStore.messages;
    if (messages.length === 0) {
      visibleStartIndex = 0;
      visibleEndIndex = 0;
      virtualTopPadding = 0;
      virtualBottomPadding = 0;
      return;
    }

    const loadSectionHeight = 48;
    const adjustedScrollTop = Math.max(0, scrollTop - loadSectionHeight);

    let accumulatedHeight = 0;
    let startIdx = 0;

    for (let i = 0; i < messages.length; i++) {
      const height = getMessageHeight(messages[i]!.id);
      if (accumulatedHeight + height >= adjustedScrollTop) {
        startIdx = i;
        break;
      }
      accumulatedHeight += height;
    }

    startIdx = Math.max(0, startIdx - BUFFER_SIZE);

    let topPadding = 0;
    for (let i = 0; i < startIdx; i++) {
      topPadding += getMessageHeight(messages[i]!.id);
    }

    let visibleHeight = 0;
    let endIdx = startIdx;

    for (let i = startIdx; i < messages.length; i++) {
      endIdx = i + 1;
      visibleHeight += getMessageHeight(messages[i]!.id);
      if (visibleHeight >= containerHeight + BUFFER_SIZE * ESTIMATED_MESSAGE_HEIGHT) {
        break;
      }
    }

    endIdx = Math.min(messages.length, Math.max(endIdx, startIdx + MIN_VISIBLE_MESSAGES));

    let bottomPadding = 0;
    for (let i = endIdx; i < messages.length; i++) {
      bottomPadding += getMessageHeight(messages[i]!.id);
    }

    visibleStartIndex = startIdx;
    visibleEndIndex = endIdx;
    virtualTopPadding = topPadding;
    virtualBottomPadding = bottomPadding;
  }

  function getVisibleMessages(): Message[] {
    return conversationsStore.messagesWithOptimistic.slice(visibleStartIndex, visibleEndIndex);
  }

  let rafPending = false;
  function updateVirtualScroll() {
    if (!messagesContainer || rafPending) return;
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

    let scrollPosition = 48;
    for (let i = 0; i < msgIndex; i++) {
      scrollPosition += getMessageHeight(messages[i]!.id);
    }

    visibleStartIndex = Math.max(0, msgIndex - BUFFER_SIZE);
    visibleEndIndex = Math.min(messages.length, msgIndex + BUFFER_SIZE + MIN_VISIBLE_MESSAGES);

    let topPadding = 0;
    for (let i = 0; i < visibleStartIndex; i++) {
      topPadding += getMessageHeight(messages[i]!.id);
    }
    virtualTopPadding = topPadding;

    let bottomPadding = 0;
    for (let i = visibleEndIndex; i < messages.length; i++) {
      bottomPadding += getMessageHeight(messages[i]!.id);
    }
    virtualBottomPadding = bottomPadding;

    await tick();

    if (messagesContainer) {
      const containerHeight = messagesContainer.clientHeight;
      const targetScroll = scrollPosition - containerHeight / 2 + getMessageHeight(messageId) / 2;
      messagesContainer.scrollTo({ top: Math.max(0, targetScroll), behavior: 'smooth' });
    }

    await tick();
    const element = document.querySelector(`[data-message-id="${messageId}"]`);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }

  // Send message
  async function handleSendMessage(text: string, retryId?: string) {
    if (!conversationsStore.selectedConversation) return;

    let optimisticId: string;
    if (retryId) {
      optimisticId = retryId;
      updateOptimisticMessage(optimisticId, { status: 'sending' });
    } else {
      optimisticId = addOptimisticMessage(text);
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
        pollMessages()
          .then(() => {
            removeOptimisticMessage(optimisticId);
            scrollToBottom();
          })
          .catch((err) => console.error('Poll error:', err));
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

  let previousFirstMessageId = $state<number | null>(null);

  const SCROLL_THRESHOLD = 200;

  async function handleScroll(event: Event) {
    const container = event.target as HTMLDivElement;
    if (!container) return;

    if (!suppressScrollRecalc) {
      updateVirtualScroll();
    }

    debouncedMeasureHeights();

    // Load more when near top
    if (
      container.scrollTop < SCROLL_THRESHOLD &&
      conversationsStore.hasMore &&
      !conversationsStore.loadingMore &&
      conversationsStore.messages.length > 0
    ) {
      const firstVisibleMessage = conversationsStore.messages[visibleStartIndex];
      previousFirstMessageId = firstVisibleMessage?.id ?? null;

      await loadMoreMessages();

      await tick();
      if (messagesContainer && previousFirstMessageId !== null) {
        const messages = conversationsStore.messages;
        const prevIndex = messages.findIndex((m) => m.id === previousFirstMessageId);
        if (prevIndex > 0) {
          let newScrollTop = 48;
          for (let i = 0; i < prevIndex; i++) {
            newScrollTop += getMessageHeight(messages[i]!.id);
          }
          messagesContainer.scrollTop = newScrollTop;
        }
        previousFirstMessageId = null;
      }
    }

    // Check if at bottom
    const { scrollTop, scrollHeight, clientHeight } = container;
    const threshold = 50;
    isAtBottom = scrollHeight - scrollTop - clientHeight < threshold;

    if (isAtBottom) {
      hasNewMessagesBelow = false;
    }
  }

  async function handleLoadEarlier() {
    if (messagesContainer) {
      const firstVisibleMessage = conversationsStore.messages[visibleStartIndex];
      previousFirstMessageId = firstVisibleMessage?.id ?? null;
    }
    await loadMoreMessages();
  }

  // Effects
  $effect(() => {
    const currentCount = conversationsStore.messages.length;
    if (currentCount > previousMessageCount && previousMessageCount > 0) {
      const newMessages = conversationsStore.messages.slice(previousMessageCount);
      const newIds = new Set(newMessageIds);
      newMessages.forEach((m) => newIds.add(m.id));
      newMessageIds = newIds;

      if (isAtBottom) {
        visibleEndIndex = conversationsStore.messages.length;
        scrollToBottom();
      } else {
        hasNewMessagesBelow = true;
      }

      setTimeout(() => {
        newMessageIds = new Set();
      }, 1000);
    }
    previousMessageCount = currentCount;
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
        virtualTopPadding = visibleStartIndex * ESTIMATED_MESSAGE_HEIGHT;
        virtualBottomPadding = 0;

        needsScrollToBottom = false;

        suppressScrollRecalc = true;
        tick().then(async () => {
          await scrollToBottom(true);
          measureVisibleMessages();
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
    if (messagesContainer) {
      messagesContainer.scrollTo({
        top: messagesContainer.scrollHeight,
        behavior: instant ? 'instant' : 'smooth',
      });
      hasNewMessagesBelow = false;
    }
  }

  function handleNewMessagesClick() {
    const messages = conversationsStore.messages;
    visibleEndIndex = messages.length;
    visibleStartIndex = Math.max(0, messages.length - MIN_VISIBLE_MESSAGES - BUFFER_SIZE);

    let topPadding = 0;
    for (let i = 0; i < visibleStartIndex; i++) {
      topPadding += getMessageHeight(messages[i]!.id);
    }
    virtualTopPadding = topPadding;
    virtualBottomPadding = 0;

    scrollToBottom();
  }

  function handleKeydown(event: KeyboardEvent) {
    const isMod = event.metaKey || event.ctrlKey;
    const isTyping =
      event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement;

    if ($activeZone !== 'messages' && $activeZone !== null) return;
    if (isTyping) {
      if (event.key === 'Escape') {
        event.preventDefault();
        (event.target as HTMLElement).blur();
        setActiveZone('messages');
      }
      return;
    }
    if (!conversationsStore.selectedConversation) return;

    if (isMod && event.key === 'd') {
      event.preventDefault();
      prefetchedSuggestions = undefined;
      showDraftPanel = true;
      return;
    }

    const messages = conversationsStore.messages;
    if (messages.length === 0) return;

    const maxIndex = messages.length - 1;

    switch (event.key) {
      case 'j':
      case 'ArrowDown':
        event.preventDefault();
        setActiveZone('messages');
        if (focusedMessageIndex < maxIndex) {
          const newIndex = focusedMessageIndex + 1;
          setMessageIndex(newIndex);
          focusedMessageIndex = newIndex;
          scrollToMessageByIndex(newIndex);
          announceMessage(messages[newIndex]!);
        }
        break;

      case 'k':
      case 'ArrowUp':
        event.preventDefault();
        setActiveZone('messages');
        if (focusedMessageIndex > 0) {
          const newIndex = focusedMessageIndex - 1;
          setMessageIndex(newIndex);
          focusedMessageIndex = newIndex;
          scrollToMessageByIndex(newIndex);
          announceMessage(messages[newIndex]!);
        } else if (focusedMessageIndex === -1 && messages.length > 0) {
          const lastIndex = maxIndex;
          setMessageIndex(lastIndex);
          focusedMessageIndex = lastIndex;
          scrollToMessageByIndex(lastIndex);
          announceMessage(messages[lastIndex]!);
        }
        break;

      case 'r':
        event.preventDefault();
        setActiveZone('compose');
        document.querySelector<HTMLTextAreaElement>('.compose-input')?.focus();
        announce('Composing reply');
        break;

      case 'g':
        if (!event.shiftKey && messages.length > 0) {
          event.preventDefault();
          setActiveZone('messages');
          setMessageIndex(0);
          focusedMessageIndex = 0;
          scrollToMessageByIndex(0);
          announceMessage(messages[0]!);
        }
        break;

      case 'G':
        if (event.shiftKey && messages.length > 0) {
          event.preventDefault();
          setActiveZone('messages');
          setMessageIndex(maxIndex);
          focusedMessageIndex = maxIndex;
          scrollToMessageByIndex(maxIndex);
          announceMessage(messages[maxIndex]!);
        }
        break;

      case 'ArrowLeft':
      case 'h':
        event.preventDefault();
        setActiveZone('conversations');
        setMessageIndex(-1);
        focusedMessageIndex = -1;
        announce('Returned to conversations list');
        break;

      case 'Escape':
        setMessageIndex(-1);
        focusedMessageIndex = -1;
        setActiveZone(null);
        break;
    }
  }

  function scrollToMessageByIndex(index: number) {
    const messages = conversationsStore.messages;
    if (index < 0 || index >= messages.length) return;

    const messageId = messages[index]!.id;
    tick().then(() => {
      const element = document.querySelector(`[data-message-id="${messageId}"]`);
      if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    });
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
    const textarea = document.querySelector<HTMLTextAreaElement>('.compose-input');
    if (textarea) {
      textarea.value = text;
      textarea.style.height = 'auto';
      textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
      textarea.focus();
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
      <div class="avatar" class:group={conversationsStore.selectedConversation.is_group}>
        {#if conversationsStore.selectedConversation.is_group}
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path
              d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm-6 8v-2c0-2.67 5.33-4 6-4s6 1.33 6 4v2H6z"
            />
          </svg>
        {:else}
          {(conversationsStore.selectedConversation.display_name || conversationsStore.selectedConversation.participants[0] || '?').charAt(0).toUpperCase()}
        {/if}
      </div>
      <div class="info">
        <h2>
          {conversationsStore.selectedConversation.display_name || conversationsStore.selectedConversation.participants.join(', ')}
        </h2>
        <p>{conversationsStore.selectedConversation.message_count} messages</p>
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
        <div class="load-earlier-section">
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
            />
          {/each}
        </div>

        <!-- Virtual scroll spacer for messages below visible range -->
        <div class="virtual-spacer-bottom" style="height: {virtualBottomPadding}px;"></div>
      {/if}
    </div>

    {#if hasNewMessagesBelow}
      <button class="new-messages-button" onclick={handleNewMessagesClick}>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="6 9 12 15 18 9"></polyline>
        </svg>
        New messages below
      </button>
    {/if}

    {#if showDraftPanel && conversationsStore.selectedConversation}
      <SuggestionBar
        chatId={conversationsStore.selectedConversation.chat_id}
        onSelect={handleDraftSelect}
        onClose={() => {
          showDraftPanel = false;
          prefetchedSuggestions = undefined;
        }}
        {...(prefetchedSuggestions && { initialSuggestions: prefetchedSuggestions })}
      />
    {/if}

    <ComposeArea
      onSend={(text) => handleSendMessage(text)}
      disabled={!conversationsStore.selectedConversation}
      sending={sendingMessage}
    />
  {/if}
</div>

<style>
  @import '../styles/tokens.css';

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
    overflow-y: auto;
    padding: var(--space-4);
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }

  .virtual-spacer-top,
  .virtual-spacer-bottom {
    flex-shrink: 0;
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

  .new-messages-button {
    position: absolute;
    bottom: 80px;
    left: 50%;
    transform: translateX(-50%);
    display: flex;
    align-items: center;
    gap: var(--space-1);
    padding: var(--space-2) var(--space-4);
    background: var(--color-primary);
    color: white;
    border: none;
    border-radius: var(--radius-full);
    font-size: var(--text-sm);
    font-weight: var(--font-weight-medium);
    cursor: pointer;
    box-shadow: var(--shadow-md);
    transition: all var(--duration-fast) var(--ease-out);
    animation: fadeInUp var(--duration-fast) var(--ease-out);
    z-index: var(--z-sticky);
  }

  .new-messages-button:hover {
    background: var(--color-primary-hover);
    transform: translateX(-50%) scale(1.05);
  }

  .new-messages-button svg {
    width: 16px;
    height: 16px;
  }

  @keyframes fadeInUp {
    from {
      opacity: 0;
      transform: translateX(-50%) translateY(10px);
    }
    to {
      opacity: 1;
      transform: translateX(-50%) translateY(0);
    }
  }
</style>
