/**
 * Conversations store for managing chat state using Svelte 5 Runes
 */

import type {
  Conversation,
  Message,
  PaginationState,
  OptimisticMessage,
  ConnectionStatus,
  DraftSuggestion
} from "../api/types";
import { api } from "../api/client";
import {
  initDatabases,
  isDirectAccessAvailable,
  getConversations as getConversationsDirect,
  getMessages as getMessagesDirect,
  getLastMessageRowid,
} from "../db";
import { jarvis } from "../socket";

/** Check if running in Tauri context */
const isTauri = typeof window !== "undefined" && "__TAURI__" in window;

/** Default number of messages to fetch per page */
const PAGE_SIZE = 20;

/** Cache entry for a conversation's messages */
interface MessageCacheEntry {
  messages: Message[];
  pagination: PaginationState;
}

// Global state using runes
class ConversationsState {
  conversations = $state<Conversation[]>([]);
  selectedChatId = $state<string | null>(null);
  messages = $state<Message[]>([]);
  loading = $state(false);
  loadingMessages = $state(false);
  loadingMore = $state(false);
  hasMore = $state(false);
  error = $state<string | null>(null);
  connectionStatus = $state<ConnectionStatus>("disconnected");
  conversationsWithNewMessages = $state(new Set<string>());
  lastKnownMessageDates = $state(new Map<string, string>());
  isWindowFocused = $state(true);
  optimisticMessages = $state<OptimisticMessage[]>([]);
  prefetchedDraft = $state<{ chatId: string; suggestions: DraftSuggestion[] } | null>(null);
  
  // Internal cache
  messageCache = new Map<string, MessageCacheEntry>();
  lastKnownGlobalRowid = 0;
  lastSyncTimestamp = 0;
  SYNC_DEBOUNCE_MS = 120000; // 2 minutes

  // Derived properties
  selectedConversation = $derived(
    this.conversations.find((c) => c.chat_id === this.selectedChatId) || null
  );

  messagesWithOptimistic = $derived.by(() => {
    if (this.optimisticMessages.length === 0) return this.messages;

    const optimisticAsMessages: Message[] = this.optimisticMessages.map((opt) => ({
      id: opt.stableId,
      chat_id: this.selectedChatId || "",
      text: opt.text,
      date: new Date(opt.timestamp).toISOString(),
      sender: "me",
      sender_name: null,
      is_from_me: true,
      is_system_message: false,
      attachments: [],
      reactions: [],
      reply_to_id: null,
      date_delivered: null,
      date_read: null,
      _optimistic: true,
      _optimisticId: opt.id,
      _optimisticStatus: opt.status,
      ...(opt.error !== undefined && { _optimisticError: opt.error }),
    }));
    return [...this.messages, ...optimisticAsMessages];
  });

  hasNewMessages(chatId: string) {
    return this.conversationsWithNewMessages.has(chatId);
  }
}

export const conversationsStore = new ConversationsState();

// Helper stores for UI components (these use traditional Svelte stores, not runes)
// These can be accessed with $ prefix in components
import { writable } from "svelte/store";
export const highlightedMessageId = writable<number | null>(null);
export const scrollToMessageId = writable<number | null>(null);

// Actions
export function markConversationAsNew(chatId: string) {
  conversationsStore.conversationsWithNewMessages.add(chatId);
}

export function clearNewMessageIndicator(chatId: string) {
  conversationsStore.conversationsWithNewMessages.delete(chatId);
}

export function addOptimisticMessage(text: string): string {
  const id = `optimistic-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  const stableId = -Date.now() - Math.random(); // Negative ID assigned once
  conversationsStore.optimisticMessages.push({
    id,
    text,
    status: "sending",
    timestamp: Date.now(),
    stableId,
  });
  return id;
}

export function updateOptimisticMessage(id: string, updates: Partial<Pick<OptimisticMessage, "status" | "error">>) {
  const msg = conversationsStore.optimisticMessages.find(m => m.id === id);
  if (msg) {
    Object.assign(msg, updates);
  }
}

export function removeOptimisticMessage(id: string) {
  conversationsStore.optimisticMessages = conversationsStore.optimisticMessages.filter(m => m.id !== id);
}

export function clearOptimisticMessages(chatId?: string) {
  if (!chatId || conversationsStore.selectedChatId !== chatId) {
    conversationsStore.optimisticMessages = [];
  }
}

export function clearPrefetchedDraft() {
  conversationsStore.prefetchedDraft = null;
}

export async function fetchConversations(isPolling = false) {
  if (!isPolling) {
    conversationsStore.loading = true;
    conversationsStore.error = null;
  }
  conversationsStore.connectionStatus = "connecting";

  try {
    let conversations: Conversation[] = [];
    
    if (isTauri) {
      try {
        const result = await jarvis.call<{ conversations: Conversation[] }>("list_conversations", { limit: 50 });
        conversations = result.conversations;
      } catch (e) {
        if (isDirectAccessAvailable()) {
          conversations = await getConversationsDirect(50);
        } else {
          conversations = await api.getConversations();
        }
      }
    } else {
      conversations = await api.getConversations();
    }

    // Detect new messages
    for (const conv of conversations) {
      const lastKnown = conversationsStore.lastKnownMessageDates.get(conv.chat_id);
      if (lastKnown && conv.last_message_date > lastKnown && conv.chat_id !== conversationsStore.selectedChatId) {
        markConversationAsNew(conv.chat_id);
      }
      conversationsStore.lastKnownMessageDates.set(conv.chat_id, conv.last_message_date);
    }

    conversationsStore.conversations = conversations;
    conversationsStore.loading = false;
    conversationsStore.connectionStatus = "connected";
  } catch (e) {
    conversationsStore.loading = false;
    conversationsStore.connectionStatus = "disconnected";
    if (!isPolling) conversationsStore.error = e instanceof Error ? e.message : "Failed to fetch conversations";
  }
}

export async function fetchMessages(chatId: string): Promise<Message[]> {
  try {
    let messages: Message[];
    if (isDirectAccessAvailable()) {
      messages = await getMessagesDirect(chatId, PAGE_SIZE);
    } else {
      messages = await api.getMessages(chatId, PAGE_SIZE);
    }
    return [...messages].reverse();
  } catch (e) {
    console.error("fetchMessages failed", e);
    return [];
  }
}

export async function pollMessages(): Promise<Message[]> {
  if (!conversationsStore.selectedChatId || !conversationsStore.isWindowFocused) return [];

  const now = Date.now();
  const timeSinceLastSync = now - conversationsStore.lastSyncTimestamp;

  try {
    if (isDirectAccessAvailable()) {
      const currentGlobalRowid = await getLastMessageRowid();
      // Skip if nothing changed globally AND we synchronized recently
      if (
        currentGlobalRowid > 0 && 
        currentGlobalRowid === conversationsStore.lastKnownGlobalRowid &&
        timeSinceLastSync < conversationsStore.SYNC_DEBOUNCE_MS
      ) {
        return [];
      }
      conversationsStore.lastKnownGlobalRowid = currentGlobalRowid;
    }

    const chatId = conversationsStore.selectedChatId;
    const freshMessages = await fetchMessages(chatId);
    
    if (conversationsStore.selectedChatId !== chatId) return [];

    const currentIds = new Set(conversationsStore.messages.map(m => m.id));
    const newMessages = freshMessages.filter(m => !currentIds.has(m.id));

    if (newMessages.length > 0) {
      conversationsStore.messages = freshMessages;
      const cached = conversationsStore.messageCache.get(chatId);
      if (cached) {
        cached.messages = freshMessages;
      }
      conversationsStore.lastSyncTimestamp = now;
    }
    return newMessages;
  } catch (e) {
    return [];
  }
}

export async function selectConversation(chatId: string) {
  if (conversationsStore.selectedChatId === chatId) return;

  clearNewMessageIndicator(chatId);

  // Background prefetch
  jarvis.call("prefetch_focus", { chat_id: chatId }).then((result: any) => {
    if (result?.prefetched && result?.draft?.suggestions?.length) {
      conversationsStore.prefetchedDraft = { chatId, suggestions: result.draft.suggestions };
    }
  }).catch(() => {});

  const cached = conversationsStore.messageCache.get(chatId);
  if (cached) {
    conversationsStore.selectedChatId = chatId;
    conversationsStore.messages = cached.messages;
    conversationsStore.hasMore = cached.pagination.hasMore;
    conversationsStore.loadingMore = false;
    conversationsStore.loadingMessages = false;
    conversationsStore.prefetchedDraft = null;
    startMessagePolling();
    return;
  }

  conversationsStore.selectedChatId = chatId;
  conversationsStore.messages = [];
  conversationsStore.loadingMessages = true;
  conversationsStore.hasMore = true;

  try {
    const messages = await fetchMessages(chatId);
    if (conversationsStore.selectedChatId !== chatId) return;

    const hasMore = messages.length >= PAGE_SIZE;
    conversationsStore.messageCache.set(chatId, {
      messages,
      pagination: { hasMore, loadingMore: false }
    });

    conversationsStore.messages = messages;
    conversationsStore.loadingMessages = false;
    conversationsStore.hasMore = hasMore;
    startMessagePolling();
  } catch (e) {
    if (conversationsStore.selectedChatId !== chatId) return;
    conversationsStore.loadingMessages = false;
    conversationsStore.error = e instanceof Error ? e.message : "Failed to load messages";
  }
}

export async function loadMoreMessages(): Promise<boolean> {
  const { selectedChatId, messages, loadingMore, hasMore } = conversationsStore;
  if (!selectedChatId || loadingMore || !hasMore || messages.length === 0) return false;

  const firstMessage = messages[0];
  if (!firstMessage) return false;

  const beforeDate = firstMessage.date;
  conversationsStore.loadingMore = true;

  try {
    const olderMessages = await api.getMessages(selectedChatId, PAGE_SIZE, beforeDate);
    const newHasMore = olderMessages.length >= PAGE_SIZE;
    const chronologicalOlder = [...olderMessages].reverse();

    const newMessages = [...chronologicalOlder, ...conversationsStore.messages];
    conversationsStore.messageCache.set(selectedChatId, {
      messages: newMessages,
      pagination: { hasMore: newHasMore, loadingMore: false }
    });

    conversationsStore.messages = newMessages;
    conversationsStore.loadingMore = false;
    conversationsStore.hasMore = newHasMore;
    return olderMessages.length > 0;
  } catch (e) {
    conversationsStore.loadingMore = false;
    return false;
  }
}

// Polling Logic
let conversationPollInterval: any = null;
let messagePollInterval: any = null;

export function startConversationPolling() {
  if (conversationPollInterval) clearInterval(conversationPollInterval);
  fetchConversations();
  conversationPollInterval = setInterval(() => {
    if (conversationsStore.isWindowFocused) fetchConversations(true);
  }, 30000);
}

export function stopConversationPolling() {
  if (conversationPollInterval) clearInterval(conversationPollInterval);
  conversationPollInterval = null;
}

export function startMessagePolling() {
  if (messagePollInterval) clearInterval(messagePollInterval);
  messagePollInterval = setInterval(() => {
    if (conversationsStore.isWindowFocused && conversationsStore.selectedChatId) pollMessages();
  }, 10000);
}

export function stopMessagePolling() {
  if (messagePollInterval) clearInterval(messagePollInterval);
  messagePollInterval = null;
}

export async function initializePolling(): Promise<() => void> {
  await initDatabases();
  if (isDirectAccessAvailable()) {
    conversationsStore.lastKnownGlobalRowid = await getLastMessageRowid();
  }

  startConversationPolling();
  
  const handleFocus = () => {
    conversationsStore.isWindowFocused = true;
    fetchConversations(true);
    if (conversationsStore.selectedChatId) pollMessages();
  };
  const handleBlur = () => {
    conversationsStore.isWindowFocused = false;
  };

  window.addEventListener("focus", handleFocus);
  window.addEventListener("blur", handleBlur);

  return () => {
    stopConversationPolling();
    stopMessagePolling();
    window.removeEventListener("focus", handleFocus);
    window.removeEventListener("blur", handleBlur);
  };
}

export async function navigateToMessage(chatId: string, messageId: number) {
  await selectConversation(chatId);
  scrollToMessageId.set(messageId);
  highlightedMessageId.set(messageId);
  setTimeout(() => highlightedMessageId.set(null), 3000);
}

export function clearScrollTarget() {
  scrollToMessageId.set(null);
}

export function clearSelection() {
  conversationsStore.selectedChatId = null;
  conversationsStore.messages = [];
  stopMessagePolling();
}