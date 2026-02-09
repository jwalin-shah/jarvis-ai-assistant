/**
 * Conversations store for managing chat state with real-time updates and pagination
 *
 * Uses direct SQLite reads for ~20-100x faster message loading with HTTP API fallback.
 * Supports both polling mode (legacy) and push mode (via socket server).
 */

import { writable, derived, get } from "svelte/store";
import type { Conversation, Message, SearchFilters } from "../api/types";
import { api } from "../api/client";
import {
  initDatabases,
  isDirectAccessAvailable,
  getConversations as getConversationsDirect,
  getMessages as getMessagesDirect,
  getMessage,
  getLastMessageRowid,
  getNewMessagesSince,
  populateContactsCache,
  isContactsCacheLoaded,
} from "../db";
import { jarvis, type ConnectionState as SocketConnectionState } from "../socket";
import type { DraftSuggestion } from "../api/types";

/** Check if running in Tauri context */
const isTauri = typeof window !== "undefined" && "__TAURI__" in window;

/** Default number of messages to fetch per page */
const PAGE_SIZE = 20;

/** Pagination state for a conversation */
export interface PaginationState {
  hasMore: boolean;
  loadingMore: boolean;
}

/** Cache entry for a conversation's messages */
interface MessageCacheEntry {
  messages: Message[];
  pagination: PaginationState;
}

export type ConnectionStatus = "connected" | "disconnected" | "connecting";

/** Optimistic message for immediate UI feedback during send */
export interface OptimisticMessage {
  id: string; // Temporary ID (e.g., "optimistic-{timestamp}")
  text: string;
  status: "sending" | "sent" | "failed";
  error?: string;
  timestamp: number;
}

// Store for message highlighting (when navigating from search results)
export const highlightedMessageId = writable<number | null>(null);

// Store for triggering scroll to a specific message
export const scrollToMessageId = writable<number | null>(null);

export interface ConversationsState {
  conversations: Conversation[];
  selectedChatId: string | null;
  messages: Message[];
  loading: boolean;
  loadingMessages: boolean;
  loadingMore: boolean;
  hasMore: boolean;
  error: string | null;
  connectionStatus: ConnectionStatus;
  conversationsWithNewMessages: Set<string>;
  lastKnownMessageDates: Map<string, string>;
  isWindowFocused: boolean;
  optimisticMessages: OptimisticMessage[];
  prefetchedDraft: { chatId: string; suggestions: DraftSuggestion[] } | null;
}

/** Message cache keyed by chat_id to avoid re-fetching */
const messageCache = new Map<string, MessageCacheEntry>();

/** Track the last known global message ROWID for delta detection */
let lastKnownGlobalRowid = 0;

const initialState: ConversationsState = {
  conversations: [],
  selectedChatId: null,
  messages: [],
  loading: false,
  loadingMessages: false,
  loadingMore: false,
  hasMore: false,
  error: null,
  connectionStatus: "disconnected",
  conversationsWithNewMessages: new Set(),
  lastKnownMessageDates: new Map(),
  isWindowFocused: true,
  optimisticMessages: [],
  prefetchedDraft: null,
};

export const conversationsStore = writable<ConversationsState>(initialState);

/**
 * Derived store for selected conversation.
 *
 * Memoized to avoid re-emitting when the conversation data hasn't actually changed.
 * Compares by chat_id to prevent infinite loops in $effect blocks.
 */
let prevSelectedConv: { chat_id: string; data: Conversation | null } | null = null;
export const selectedConversation = derived(
  conversationsStore,
  ($state) => {
    const found = $state.conversations.find((c) => c.chat_id === $state.selectedChatId) || null;
    const currentChatId = found?.chat_id ?? null;

    // Return cached value if chat_id hasn't changed (same conversation, even if object reference differs)
    if (prevSelectedConv && prevSelectedConv.chat_id === currentChatId) {
      return prevSelectedConv.data;
    }

    // Update cache with new conversation
    prevSelectedConv = { chat_id: currentChatId!, data: found };
    return found;
  }
);

// Derived store for connection status
export const connectionStatus = derived(
  conversationsStore,
  ($state) => $state.connectionStatus
);

// Derived store for checking if a conversation has new messages
export const hasNewMessages = derived(
  conversationsStore,
  ($state) => (chatId: string) => $state.conversationsWithNewMessages.has(chatId)
);

// Derived store for messages with optimistic messages appended
export const messagesWithOptimistic = derived(
  conversationsStore,
  ($state) => {
    // Short-circuit: return messages directly when no optimistic messages pending (99% of the time)
    if ($state.optimisticMessages.length === 0) return $state.messages;

    // Convert optimistic messages to Message-like objects for display
    const optimisticAsMessages = $state.optimisticMessages.map((opt) => ({
      id: -Date.now() - Math.random(), // Negative ID to avoid conflicts
      text: opt.text,
      date: new Date(opt.timestamp).toISOString(),
      sender: "me",
      sender_name: null,
      is_from_me: true,
      is_system_message: false,
      attachments: [],
      reactions: [],
      // Custom fields for optimistic state
      _optimistic: true,
      _optimisticId: opt.id,
      _optimisticStatus: opt.status,
      _optimisticError: opt.error,
    }));
    return [...$state.messages, ...optimisticAsMessages];
  }
);

// Polling intervals (in milliseconds)
// When socket is connected, we use longer intervals as fallback
// When socket is disconnected, we poll more frequently
const CONVERSATION_POLL_INTERVAL_CONNECTED = 60000; // 1 minute when socket connected
const CONVERSATION_POLL_INTERVAL_DISCONNECTED = 30000; // 30 seconds when disconnected
const MESSAGE_POLL_INTERVAL_CONNECTED = 30000; // 30 seconds when socket connected
const MESSAGE_POLL_INTERVAL_DISCONNECTED = 10000; // 10 seconds when disconnected

// Polling interval handles
let conversationPollInterval: ReturnType<typeof setInterval> | null = null;
let messagePollInterval: ReturnType<typeof setInterval> | null = null;

// Track if socket push is active
let socketPushActive = false;

// AbortControllers for request deduplication
let conversationFetchController: AbortController | null = null;
let messageFetchController: AbortController | null = null;
let loadMoreController: AbortController | null = null;
let pollMessagesController: AbortController | null = null;

/**
 * Update connection status
 */
function setConnectionStatus(status: ConnectionStatus): void {
  conversationsStore.update((state) => {
    if (state.connectionStatus === status) return state; // Early return if unchanged
    return { ...state, connectionStatus: status };
  });
}

/**
 * Mark a conversation as having new messages
 */
export function markConversationAsNew(chatId: string): void {
  conversationsStore.update((state) => {
    if (state.conversationsWithNewMessages.has(chatId)) return state; // Already marked
    const newSet = new Set(state.conversationsWithNewMessages);
    newSet.add(chatId);
    return { ...state, conversationsWithNewMessages: newSet };
  });
}

/**
 * Clear new message indicator for a conversation
 */
export function clearNewMessageIndicator(chatId: string): void {
  conversationsStore.update((state) => {
    if (!state.conversationsWithNewMessages.has(chatId)) return state; // Not present
    const newSet = new Set(state.conversationsWithNewMessages);
    newSet.delete(chatId);
    return { ...state, conversationsWithNewMessages: newSet };
  });
}

/**
 * Add an optimistic message for immediate UI feedback
 * Returns the optimistic message ID for tracking
 */
export function addOptimisticMessage(text: string): string {
  const id = `optimistic-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  const optimisticMsg: OptimisticMessage = {
    id,
    text,
    status: "sending",
    timestamp: Date.now(),
  };

  conversationsStore.update((state) => ({
    ...state,
    optimisticMessages: [...state.optimisticMessages, optimisticMsg],
  }));

  return id;
}

/**
 * Update an optimistic message status
 */
export function updateOptimisticMessage(
  id: string,
  updates: Partial<Pick<OptimisticMessage, "status" | "error">>
): void {
  conversationsStore.update((state) => ({
    ...state,
    optimisticMessages: state.optimisticMessages.map((msg) =>
      msg.id === id ? { ...msg, ...updates } : msg
    ),
  }));
}

/**
 * Remove an optimistic message (e.g., after successful send and real message appears)
 */
export function removeOptimisticMessage(id: string): void {
  conversationsStore.update((state) => ({
    ...state,
    optimisticMessages: state.optimisticMessages.filter((msg) => msg.id !== id),
  }));
}

/**
 * Clear all optimistic messages for a specific conversation (e.g., when switching conversations)
 * If no chatId is provided, clears all optimistic messages.
 */
export function clearOptimisticMessages(chatId?: string): void {
  conversationsStore.update((state) => {
    if (!chatId) {
      // Clear all optimistic messages
      return {
        ...state,
        optimisticMessages: [],
      };
    }
    // Only clear if we're leaving this conversation (i.e., it's not currently selected)
    if (state.selectedChatId !== chatId) {
      return {
        ...state,
        optimisticMessages: [],
      };
    }
    return state;
  });
}

/**
 * Clear the prefetched draft (e.g., after SuggestionBar uses it)
 */
export function clearPrefetchedDraft(): void {
  conversationsStore.update((state) => {
    if (!state.prefetchedDraft) return state;
    return { ...state, prefetchedDraft: null };
  });
}

/**
 * Fetch conversations and detect new messages
 * Uses direct SQLite read with HTTP API fallback
 */
export async function fetchConversations(isPolling = false): Promise<void> {
  // Cancel any in-flight conversation fetch
  if (conversationFetchController) {
    conversationFetchController.abort();
  }
  conversationFetchController = new AbortController();
  const signal = conversationFetchController.signal;

  if (!isPolling) {
    conversationsStore.update((state) => ({ ...state, loading: true, error: null }));
  }

  setConnectionStatus("connecting");

  try {
    let conversations: Conversation[];
    const fetchStartTime = performance.now();
    let fetchMethod = "unknown";

    // 1. Try socket first (fastest in Tauri context)
    if (isTauri) {
      try {
        const socketStartTime = performance.now();
        const result = await jarvis.call<{ conversations: Conversation[] }>(
          "list_conversations",
          { limit: 50 }
        );
        const socketMs = performance.now() - socketStartTime;
        conversations = result.conversations;
        fetchMethod = "socket";
        console.log(
          `[Conversations] Socket fetch took ${socketMs.toFixed(1)}ms ` +
          `(${conversations.length} conversations)`
        );
      } catch (socketError) {
        console.warn("[Conversations] Socket call failed, trying direct SQLite:", socketError);

        // 2. Fall back to direct SQLite
        if (isDirectAccessAvailable()) {
          try {
            const directStartTime = performance.now();
            conversations = await getConversationsDirect(50);
            const directMs = performance.now() - directStartTime;
            fetchMethod = "directSQL";
            console.log(
              `[Conversations] Direct SQLite fetch took ${directMs.toFixed(1)}ms ` +
              `(${conversations.length} conversations)`
            );
          } catch (directError) {
            console.warn("[Conversations] Direct read failed, falling back to HTTP:", directError);
            const httpStartTime = performance.now();
            conversations = await api.getConversations();
            const httpMs = performance.now() - httpStartTime;
            fetchMethod = "http";
            console.log(
              `[Conversations] HTTP fetch took ${httpMs.toFixed(1)}ms ` +
              `(${conversations.length} conversations)`
            );
          }
        } else {
          const httpStartTime = performance.now();
          conversations = await api.getConversations();
          const httpMs = performance.now() - httpStartTime;
          fetchMethod = "http";
          console.log(
            `[Conversations] HTTP fetch took ${httpMs.toFixed(1)}ms ` +
            `(${conversations.length} conversations)`
          );
        }
      }
    } else {
      // Browser mode: try direct SQLite then HTTP (no socket in browser)
      if (isDirectAccessAvailable()) {
        try {
          const directStartTime = performance.now();
          conversations = await getConversationsDirect(50);
          const directMs = performance.now() - directStartTime;
          fetchMethod = "directSQL";
          console.log(
            `[Conversations] Direct SQLite fetch took ${directMs.toFixed(1)}ms ` +
            `(${conversations.length} conversations)`
          );
        } catch (directError) {
          console.warn("[Conversations] Direct read failed, falling back to HTTP:", directError);
          const httpStartTime = performance.now();
          conversations = await api.getConversations();
          const httpMs = performance.now() - httpStartTime;
          fetchMethod = "http";
          console.log(
            `[Conversations] HTTP fetch took ${httpMs.toFixed(1)}ms ` +
            `(${conversations.length} conversations)`
          );
        }
      } else {
        const httpStartTime = performance.now();
        conversations = await api.getConversations();
        const httpMs = performance.now() - httpStartTime;
        fetchMethod = "http";
        console.log(
          `[Conversations] HTTP fetch took ${httpMs.toFixed(1)}ms ` +
          `(${conversations.length} conversations)`
        );
      }
    }

    const totalFetchMs = performance.now() - fetchStartTime;
    console.log(
      `[Conversations] Total fetch (${fetchMethod}) took ${totalFetchMs.toFixed(1)}ms`
    );
    if (totalFetchMs > 100) {
      console.warn(
        `[LATENCY WARNING] fetchConversations took ${totalFetchMs.toFixed(1)}ms (threshold: 100ms)`
      );
    }

    // Check if request was aborted
    if (signal.aborted) return;

    conversationsStore.update((state) => {
      const newConversationsWithNewMessages = new Set(state.conversationsWithNewMessages);
      const newLastKnownDates = new Map(state.lastKnownMessageDates);

      // Check for new messages by comparing last_message_date
      for (const conv of conversations) {
        const lastKnown = state.lastKnownMessageDates.get(conv.chat_id);

        if (lastKnown && conv.last_message_date > lastKnown) {
          // New message detected - only mark if not currently viewing this conversation
          if (conv.chat_id !== state.selectedChatId) {
            newConversationsWithNewMessages.add(conv.chat_id);
          }
        }

        // Update last known date
        newLastKnownDates.set(conv.chat_id, conv.last_message_date);
      }

      return {
        ...state,
        conversations,
        loading: false,
        connectionStatus: "connected",
        conversationsWithNewMessages: newConversationsWithNewMessages,
        lastKnownMessageDates: newLastKnownDates,
      };
    });
  } catch (error) {
    // Ignore abort errors
    if (error instanceof Error && error.name === "AbortError") return;
    const message = error instanceof Error ? error.message : "Failed to fetch conversations";
    conversationsStore.update((state) => ({
      ...state,
      loading: false,
      error: isPolling ? state.error : message, // Don't overwrite error if polling
      connectionStatus: "disconnected",
    }));
  }
}

/**
 * Fetch messages for a conversation
 * Uses direct SQLite read with HTTP API fallback
 * Returns the new messages if any (for polling use)
 */
export async function fetchMessages(chatId: string): Promise<Message[]> {
  const startTime = performance.now();
  try {
    let messages: Message[];

    // Try direct SQLite read first (much faster)
    if (isDirectAccessAvailable()) {
      try {
        messages = await getMessagesDirect(chatId, PAGE_SIZE);
      } catch (directError) {
        console.warn("[Messages] Direct read failed, falling back to HTTP:", directError);
        messages = await api.getMessages(chatId, PAGE_SIZE);
      }
    } else {
      messages = await api.getMessages(chatId, PAGE_SIZE);
    }

    const elapsed = performance.now() - startTime;
    console.log(`[LATENCY] fetchMessages took ${elapsed.toFixed(1)}ms for ${messages.length} messages`);
    if (elapsed > 100) {
      console.warn(`[LATENCY WARNING] fetchMessages took ${elapsed.toFixed(1)}ms (threshold: 100ms)`);
    }

    // Reverse messages so oldest is at top, newest at bottom (API returns newest first)
    return [...messages].reverse();
  } catch (error) {
    const elapsed = performance.now() - startTime;
    console.error(`Failed to fetch messages after ${elapsed.toFixed(1)}ms:`, error);
    return [];
  }
}

/**
 * Poll for new messages in the currently selected conversation
 * Returns newly added messages (if any) for animation purposes
 *
 * OPTIMIZATION: Uses delta detection - checks if global ROWID changed before
 * fetching all messages. This avoids unnecessary DB reads when nothing changed.
 */
export async function pollMessages(): Promise<Message[]> {
  const startTime = performance.now();
  const state = get(conversationsStore);

  if (!state.selectedChatId || !state.isWindowFocused) {
    return [];
  }

  // Cancel any in-flight poll
  if (pollMessagesController) {
    pollMessagesController.abort();
  }
  pollMessagesController = new AbortController();
  const signal = pollMessagesController.signal;

  try {
    // DELTA OPTIMIZATION: Check if any new messages exist globally before fetching
    // This avoids expensive full message fetches when nothing has changed
    if (isDirectAccessAvailable()) {
      const currentGlobalRowid = await getLastMessageRowid();
      if (currentGlobalRowid > 0 && currentGlobalRowid === lastKnownGlobalRowid) {
        // No new messages globally, skip the full fetch
        return [];
      }
      // Update tracking for next poll
      lastKnownGlobalRowid = currentGlobalRowid;
    }

    const chatIdBeforeFetch = state.selectedChatId;
    const freshMessages = await fetchMessages(chatIdBeforeFetch);

    // Check if request was aborted
    if (signal.aborted) return [];

    // Re-read store state after async gap - conversation may have changed
    const freshState = get(conversationsStore);
    if (freshState.selectedChatId !== chatIdBeforeFetch) {
      // User switched conversations during fetch, discard results
      return [];
    }
    const currentMessages = freshState.messages;

    // Find messages that are new (not in current list)
    const currentIds = new Set(currentMessages.map((m) => m.id));
    const newMessages = freshMessages.filter((m) => !currentIds.has(m.id));

    if (newMessages.length > 0) {
      // Update store with fresh messages
      conversationsStore.update((s) => ({
        ...s,
        messages: freshMessages,
      }));

      // Update cache with new messages
      const cached = messageCache.get(chatIdBeforeFetch);
      if (cached) {
        messageCache.set(chatIdBeforeFetch, {
          ...cached,
          messages: freshMessages,
        });
      }
    }

    const elapsed = performance.now() - startTime;
    if (newMessages.length > 0) {
      console.log(`[LATENCY] pollMessages found ${newMessages.length} new messages in ${elapsed.toFixed(1)}ms`);
    }
    return newMessages;
  } catch (error) {
    // Ignore abort errors
    if (error instanceof Error && error.name === "AbortError") return [];
    const elapsed = performance.now() - startTime;
    console.error(`Error polling messages after ${elapsed.toFixed(1)}ms:`, error);
    return [];
  }
}

/**
 * Select a conversation and load its messages
 */
export async function selectConversation(chatId: string): Promise<void> {
  // Skip if re-selecting the same conversation (preserves prefetched draft)
  const currentState = get(conversationsStore);
  if (currentState.selectedChatId === chatId) {
    return;
  }

  // Cancel any in-flight message requests when switching conversations
  if (messageFetchController) {
    messageFetchController.abort();
    messageFetchController = null;
  }
  if (loadMoreController) {
    loadMoreController.abort();
    loadMoreController = null;
  }
  if (pollMessagesController) {
    pollMessagesController.abort();
    pollMessagesController = null;
  }

  // Clear new message indicator when selecting
  clearNewMessageIndicator(chatId);

  // Trigger prefetch and capture response for auto-display
  jarvis
    .call<{ prefetched?: boolean; draft?: { suggestions?: DraftSuggestion[] } }>(
      "prefetch_focus",
      { chat_id: chatId }
    )
    .then((result) => {
      if (result?.prefetched && result?.draft?.suggestions?.length) {
        conversationsStore.update((state) => ({
          ...state,
          prefetchedDraft: { chatId, suggestions: result.draft!.suggestions! },
        }));
      }
    })
    .catch((err) => {
      console.debug("[Prefetch] Background prefetch failed:", err);
    });

  // Check if we have cached messages for this conversation
  const cached = messageCache.get(chatId);
  if (cached) {
    conversationsStore.update((state) => ({
      ...state,
      selectedChatId: chatId,
      messages: cached.messages,
      hasMore: cached.pagination.hasMore,
      loadingMore: false,
      loadingMessages: false,
      error: null,
      prefetchedDraft: null, // Clear draft when displaying cached messages
    }));
    // Start message polling for this conversation
    startMessagePolling();
    return;
  }

  conversationsStore.update((state) => ({
    ...state,
    selectedChatId: chatId,
    messages: [],
    loadingMessages: true,
    loadingMore: false,
    hasMore: true,
    error: null,
  }));

  try {
    let messages: Message[];

    // Try direct SQLite read first (much faster)
    if (isDirectAccessAvailable()) {
      try {
        console.log("[SelectConversation] Using direct SQLite for:", chatId);
        messages = await getMessagesDirect(chatId, PAGE_SIZE);
      } catch (directError) {
        console.warn("[SelectConversation] Direct read failed, falling back to HTTP:", directError);
        messages = await api.getMessages(chatId, PAGE_SIZE);
      }
    } else {
      console.log("[SelectConversation] Using HTTP API for:", chatId);
      messages = await api.getMessages(chatId, PAGE_SIZE);
      console.log("[SelectConversation] Got messages:", messages.length);
    }

    // Check if conversation changed during async fetch
    const freshState = get(conversationsStore);
    if (freshState.selectedChatId !== chatId) {
      console.log("[SelectConversation] Conversation changed during fetch, discarding results");
      return;
    }

    // Reverse messages so oldest is at top, newest at bottom (API returns newest first)
    const chronologicalMessages = [...messages].reverse();
    // If we got fewer messages than requested, we've reached the end
    const hasMore = messages.length >= PAGE_SIZE;

    // Cache the messages
    messageCache.set(chatId, {
      messages: chronologicalMessages,
      pagination: { hasMore, loadingMore: false },
    });

    conversationsStore.update((state) => ({
      ...state,
      messages: chronologicalMessages,
      loadingMessages: false,
      hasMore,
      prefetchedDraft: null, // Clear draft when displaying fresh messages
    }));

    // Start message polling for this conversation
    startMessagePolling();
  } catch (error) {
    // Check if conversation changed during error handling
    const freshState = get(conversationsStore);
    if (freshState.selectedChatId !== chatId) {
      return;
    }

    console.error("[SelectConversation] Error fetching messages:", error);
    const message = error instanceof Error ? error.message : "Failed to fetch messages";
    conversationsStore.update((state) => ({
      ...state,
      loadingMessages: false,
      error: message,
    }));
  }
}

/**
 * Load more (older) messages for the currently selected conversation.
 * Uses the 'before' parameter to fetch messages older than the oldest loaded message.
 * Returns true if more messages were loaded, false if at the end of history.
 */
export async function loadMoreMessages(): Promise<boolean> {
  const { selectedChatId, messages, loadingMore, hasMore } = get(conversationsStore);

  // Guard: don't fetch if no conversation selected, already loading, or no more messages
  if (!selectedChatId || loadingMore || !hasMore || messages.length === 0) {
    return false;
  }

  // Get the oldest message's date (first in array since messages are chronological)
  const oldestMessage = messages[0];
  const beforeDate = oldestMessage.date;

  // Cancel any in-flight loadMore request
  if (loadMoreController) {
    loadMoreController.abort();
  }
  loadMoreController = new AbortController();
  const signal = loadMoreController.signal;

  conversationsStore.update((state) => ({
    ...state,
    loadingMore: true,
  }));

  try {
    const olderMessages = await api.getMessages(selectedChatId, PAGE_SIZE, beforeDate);

    // Check if request was aborted
    if (signal.aborted) return false;

    // If we got fewer messages than requested, we've reached the end
    const newHasMore = olderMessages.length >= PAGE_SIZE;

    // Reverse to get chronological order and prepend to existing messages
    const chronologicalOlder = [...olderMessages].reverse();

    conversationsStore.update((state) => {
      const newMessages = [...chronologicalOlder, ...state.messages];

      // Update cache
      messageCache.set(selectedChatId, {
        messages: newMessages,
        pagination: { hasMore: newHasMore, loadingMore: false },
      });

      return {
        ...state,
        messages: newMessages,
        loadingMore: false,
        hasMore: newHasMore,
      };
    });

    return olderMessages.length > 0;
  } catch (error) {
    // Ignore abort errors
    if (error instanceof Error && error.name === "AbortError") return false;
    const message = error instanceof Error ? error.message : "Failed to load more messages";
    conversationsStore.update((state) => ({
      ...state,
      loadingMore: false,
      error: message,
    }));
    return false;
  }
}

/**
 * Clear the current conversation selection
 */
export function clearSelection(): void {
  // Cancel any in-flight message requests
  if (messageFetchController) {
    messageFetchController.abort();
    messageFetchController = null;
  }
  if (loadMoreController) {
    loadMoreController.abort();
    loadMoreController = null;
  }
  stopMessagePolling();
  conversationsStore.update((state) => ({
    ...state,
    selectedChatId: null,
    messages: [],
    hasMore: true,
    loadingMore: false,
  }));
}

/**
 * Invalidate the message cache for a specific conversation or all conversations.
 * Useful when messages might have changed (e.g., new message received).
 */
export function invalidateMessageCache(chatId?: string): void {
  if (chatId) {
    messageCache.delete(chatId);
  } else {
    messageCache.clear();
  }
}

/**
 * Start polling for conversations
 */
export function startConversationPolling(): void {
  // Clear existing interval if any
  if (conversationPollInterval) {
    clearInterval(conversationPollInterval);
  }

  // Initial fetch
  fetchConversations();

  // Use longer interval when socket push is active
  const interval = socketPushActive
    ? CONVERSATION_POLL_INTERVAL_CONNECTED
    : CONVERSATION_POLL_INTERVAL_DISCONNECTED;

  // Set up polling interval
  conversationPollInterval = setInterval(() => {
    const state = get(conversationsStore);
    if (state.isWindowFocused) {
      fetchConversations(true);
    }
  }, interval);
}

/**
 * Stop polling for conversations
 */
export function stopConversationPolling(): void {
  if (conversationPollInterval) {
    clearInterval(conversationPollInterval);
    conversationPollInterval = null;
  }
  if (conversationFetchController) {
    conversationFetchController.abort();
    conversationFetchController = null;
  }
}

/**
 * Start polling for messages in the selected conversation
 */
export function startMessagePolling(): void {
  // Clear existing interval if any
  if (messagePollInterval) {
    clearInterval(messagePollInterval);
  }

  // Use longer interval when socket push is active
  const interval = socketPushActive
    ? MESSAGE_POLL_INTERVAL_CONNECTED
    : MESSAGE_POLL_INTERVAL_DISCONNECTED;

  // Set up polling interval
  messagePollInterval = setInterval(() => {
    const state = get(conversationsStore);
    if (state.isWindowFocused && state.selectedChatId) {
      pollMessages();
    }
  }, interval);
}

/**
 * Stop polling for messages
 */
export function stopMessagePolling(): void {
  if (messagePollInterval) {
    clearInterval(messagePollInterval);
    messagePollInterval = null;
  }
  if (messageFetchController) {
    messageFetchController.abort();
    messageFetchController = null;
  }
  if (pollMessagesController) {
    pollMessagesController.abort();
    pollMessagesController = null;
  }
}

/**
 * Handle window focus change
 */
export function setWindowFocused(focused: boolean): void {
  conversationsStore.update((state) => {
    if (state.isWindowFocused === focused) return state; // Early return if unchanged
    return { ...state, isWindowFocused: focused };
  });

  if (focused) {
    // Resume polling when window gains focus
    fetchConversations(true);
    startConversationPolling();
    const state = get(conversationsStore);
    if (state.selectedChatId) {
      pollMessages();
      startMessagePolling();
    }
  } else {
    // Stop polling when window loses focus to save CPU/battery
    stopConversationPolling();
    stopMessagePolling();
  }
}

/**
 * Initialize direct database access
 * Call this early in app lifecycle
 */
export async function initializeDirectAccess(): Promise<boolean> {
  try {
    await initDatabases();
    console.log("[Conversations] Direct database access initialized");

    // Initialize the global ROWID tracker for delta polling
    if (isDirectAccessAvailable()) {
      lastKnownGlobalRowid = await getLastMessageRowid();
      console.log(`[Conversations] Initial global ROWID: ${lastKnownGlobalRowid}`);
    }

    return true;
  } catch (error) {
    console.warn("[Conversations] Direct database access unavailable, using HTTP fallback:", error);
    return false;
  }
}

/**
 * Initialize socket connection for push notifications
 */
// Socket event handlers (stored for cleanup)
let socketListenersRegistered = false;

function handleSocketNewMessage(data: {
  message_id: number;
  chat_id: string;
  sender: string;
  text: string;
  date: string;
  is_from_me: boolean;
}): void {
  handleNewMessagePush(data);
}

function handleSocketDisconnected(): void {
  console.log("[Conversations] Socket disconnected, switching to polling mode");
  socketPushActive = false;
  adjustPollingIntervals();
}

function handleSocketConnected(): void {
  console.log("[Conversations] Socket connected, reducing poll frequency");
  socketPushActive = true;
  adjustPollingIntervals();
}

/**
 * Remove all socket event listeners
 */
function cleanupSocketListeners(): void {
  if (socketListenersRegistered) {
    jarvis.off("new_message", handleSocketNewMessage);
    jarvis.off("disconnected", handleSocketDisconnected);
    jarvis.off("connected", handleSocketConnected);
    socketListenersRegistered = false;
    console.log("[Conversations] Socket listeners cleaned up");
  }
}

async function initializeSocketPush(): Promise<boolean> {
  try {
    const connected = await jarvis.connect();
    if (!connected) {
      console.warn("[Conversations] Socket connection failed, using polling fallback");
      return false;
    }

    // Avoid duplicate listener registration
    if (socketListenersRegistered) {
      return true;
    }

    // Register for new message notifications
    jarvis.on<{
      message_id: number;
      chat_id: string;
      sender: string;
      text: string;
      date: string;
      is_from_me: boolean;
    }>("new_message", handleSocketNewMessage);

    // Handle socket connection state changes
    jarvis.on("disconnected", handleSocketDisconnected);
    jarvis.on("connected", handleSocketConnected);

    socketListenersRegistered = true;
    socketPushActive = true;
    console.log("[Conversations] Socket push notifications enabled");
    return true;
  } catch (error) {
    console.warn("[Conversations] Socket push unavailable:", error);
    return false;
  }
}

/**
 * Handle a new message push notification
 */
function handleNewMessagePush(data: {
  message_id: number;
  chat_id: string;
  sender: string;
  text: string;
  date: string;
  is_from_me: boolean;
}): void {
  const state = get(conversationsStore);

  // Mark conversation as having new message if not currently viewing
  if (data.chat_id !== state.selectedChatId) {
    markConversationAsNew(data.chat_id);
  }

  // Update last known date
  conversationsStore.update((s) => {
    const newLastKnownDates = new Map(s.lastKnownMessageDates);
    newLastKnownDates.set(data.chat_id, data.date);
    return { ...s, lastKnownMessageDates: newLastKnownDates };
  });

  // If this is the selected conversation, add the message
  if (data.chat_id === state.selectedChatId) {
    // OPTIMIZATION: Fetch only the new message instead of the whole list
    if (isDirectAccessAvailable()) {
      getMessage(data.chat_id, data.message_id).then((newMessage) => {
        if (!newMessage) return;

        conversationsStore.update((s) => {
          // Avoid duplicates
          if (s.messages.some((m) => m.id === newMessage.id)) return s;
          const messages = [...s.messages, newMessage];
          return { ...s, messages };
        });

        // Update cache
        const cached = messageCache.get(data.chat_id);
        if (cached) {
          const messages = [...cached.messages, newMessage];
          messageCache.set(data.chat_id, { ...cached, messages });
        }
      });
    } else {
      // Fallback to full fetch if direct access not available
      fetchMessages(data.chat_id).then((messages) => {
        conversationsStore.update((s) => ({
          ...s,
          messages,
        }));

        // Update cache
        const cached = messageCache.get(data.chat_id);
        if (cached) {
          messageCache.set(data.chat_id, {
            ...cached,
            messages,
          });
        }
      });
    }
  }

  // Refresh conversation list to update order and last message preview
  fetchConversations(true);
}

/**
 * Adjust polling intervals based on socket connection state
 */
function adjustPollingIntervals(): void {
  // Restart polling with appropriate intervals
  stopConversationPolling();
  stopMessagePolling();
  startConversationPolling();

  const state = get(conversationsStore);
  if (state.selectedChatId) {
    startMessagePolling();
  }
}

/**
 * Resolve contact names from the backend and populate the frontend cache.
 * Collects unique participant identifiers from current conversations and
 * calls the backend resolve_contacts RPC to get display names.
 */
async function resolveContactNames(): Promise<void> {
  if (isContactsCacheLoaded()) return;

  try {
    const state = get(conversationsStore);
    const identifiers = new Set<string>();
    for (const conv of state.conversations) {
      for (const p of conv.participants) {
        if (p && p !== "me") identifiers.add(p);
      }
    }

    if (identifiers.size === 0) return;

    const resolved = await jarvis.call<Record<string, string | null>>(
      "resolve_contacts",
      { identifiers: [...identifiers] }
    );

    if (resolved && typeof resolved === "object") {
      populateContactsCache(resolved);
      // Re-fetch conversations to apply resolved names
      await fetchConversations(true);
      console.log(`[Contacts] Resolved ${Object.values(resolved).filter(Boolean).length} names`);
    }
  } catch (err) {
    console.debug("[Contacts] Backend contact resolution failed:", err);
  }
}

/**
 * Initialize polling and window focus listeners
 * Call this on app mount
 */
export function initializePolling(): () => void {
  // Initialize DB and socket in parallel for fastest startup
  const initStartTime = performance.now();
  console.log("[Conversations] Starting initialization...");
  const dbReady = initializeDirectAccess();

  initializeSocketPush().then(() => {
    setTimeout(resolveContactNames, 2000);
  });

  // Wait for DB to be ready before starting polling (prevents wasteful HTTP fallback)
  // First fetch will use socket → direct SQLite → HTTP fallback
  dbReady.then((available) => {
    const dbInitMs = performance.now() - initStartTime;
    console.log(
      `[Conversations] DB ready (available: ${available}) after ${dbInitMs.toFixed(1)}ms, ` +
      `starting polling`
    );
    startConversationPolling();
  });

  // Handle window focus/blur
  const handleFocus = () => setWindowFocused(true);
  const handleBlur = () => setWindowFocused(false);

  window.addEventListener("focus", handleFocus);
  window.addEventListener("blur", handleBlur);

  // Cleanup function
  return () => {
    stopConversationPolling();
    stopMessagePolling();
    cleanupSocketListeners();
    jarvis.disconnect();
    window.removeEventListener("focus", handleFocus);
    window.removeEventListener("blur", handleBlur);
  };
}

/**
 * Navigate to a specific message from search results
 * Selects the conversation and highlights the target message
 */
export async function navigateToMessage(chatId: string, messageId: number): Promise<void> {
  // First select the conversation
  await selectConversation(chatId);

  // Then trigger scroll and highlight
  scrollToMessageId.set(messageId);
  highlightedMessageId.set(messageId);

  // Clear highlight after 3 seconds
  setTimeout(() => {
    highlightedMessageId.set(null);
  }, 3000);
}

/**
 * Clear the scroll target after scrolling is complete
 */
export function clearScrollTarget(): void {
  scrollToMessageId.set(null);
}
