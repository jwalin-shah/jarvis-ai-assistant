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
  getMessage as getMessageDirect,
  getMessagesBatch as getMessagesBatchDirect,
  getLastMessageRowid,
  getNewMessagesSince,
  populateContactsCache,
  loadContactsFromAddressBook,
  isContactsCacheLoaded,
  resolveContactName,
} from "../db";
import { jarvis } from "../socket";
import type { NewMessageEvent } from "../socket/client";

/** Check if running in Tauri context */
const isTauri = typeof window !== "undefined" && "__TAURI__" in window;

/** Number of messages to fetch per page */
const PAGE_SIZE = 40;

/** Number of conversations to fetch for the list */
const CONVERSATIONS_PAGE_SIZE = 50;

/** Max conversations to keep in message cache (LRU eviction) */
const MAX_CACHE_SIZE = 50;

/** Cache entry for a conversation's messages */
interface MessageCacheEntry {
  messages: Message[];
  pagination: PaginationState;
}

/**
 * Set a message cache entry with LRU eviction.
 * JS Maps iterate in insertion order, so the first key is the oldest.
 */
function setCacheEntry(
  cache: Map<string, MessageCacheEntry>,
  key: string,
  value: MessageCacheEntry,
): void {
  cache.delete(key);
  cache.set(key, value);
  while (cache.size > MAX_CACHE_SIZE) {
    const oldest = cache.keys().next().value;
    if (oldest !== undefined) {
      cache.delete(oldest);
    } else {
      break;
    }
  }
}

/**
 * Get a message cache entry, refreshing its LRU position.
 */
function getCacheEntry(
  cache: Map<string, MessageCacheEntry>,
  key: string,
): MessageCacheEntry | undefined {
  const entry = cache.get(key);
  if (entry !== undefined) {
    cache.delete(key);
    cache.set(key, entry);
  }
  return entry;
}

// Global state using runes
class ConversationsState {
  conversations = $state<Conversation[]>([]);
  selectedChatId = $state<string | null>(null);
  messages = $state<Message[]>([]);
  loading = $state(false);
  /** True only during the first conversation list fetch (UX-04) */
  isInitialLoad = $state(true);
  /** True when polling for conversation updates in the background (UX-04) */
  isPolling = $state(false);
  loadingMessages = $state(false);
  loadingMore = $state(false);
  hasMore = $state(false);
  error = $state<string | null>(null);
  connectionStatus = $state<ConnectionStatus>("disconnected");
  unreadCounts = $state(new Map<string, number>());
  pinnedChats = $state(new Set<string>());
  archivedChats = $state(new Set<string>());
  lastKnownMessageDates = $state(new Map<string, string>());
  isWindowFocused = $state(true);
  optimisticMessages = $state<OptimisticMessage[]>([]);
  prefetchedDraft = $state<{ chatId: string; suggestions: DraftSuggestion[] } | null>(null);
  
  // Internal cache
  messageCache = new Map<string, MessageCacheEntry>();
  lastKnownGlobalRowid = 0;
  lastSyncTimestamp = 0;
  lastConversationFetchTime = 0;
  SYNC_DEBOUNCE_MS = 120000; // 2 minutes
  _messagePolling = false;

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
    return (this.unreadCounts.get(chatId) ?? 0) > 0;
  }

  getUnreadCount(chatId: string): number {
    return this.unreadCounts.get(chatId) ?? 0;
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
  const current = conversationsStore.unreadCounts.get(chatId) ?? 0;
  conversationsStore.unreadCounts.set(chatId, current + 1);
  conversationsStore.unreadCounts = new Map(conversationsStore.unreadCounts);
}

export function clearNewMessageIndicator(chatId: string) {
  conversationsStore.unreadCounts.delete(chatId);
  conversationsStore.unreadCounts = new Map(conversationsStore.unreadCounts);
}

// Pinning & Archiving
const PINNED_STORAGE_KEY = 'jarvis-pinned-chats';
const ARCHIVED_STORAGE_KEY = 'jarvis-archived-chats';

function loadPinnedChats(): Set<string> {
  try {
    const stored = localStorage.getItem(PINNED_STORAGE_KEY);
    return stored ? new Set(JSON.parse(stored)) : new Set();
  } catch { return new Set(); }
}

function loadArchivedChats(): Set<string> {
  try {
    const stored = localStorage.getItem(ARCHIVED_STORAGE_KEY);
    return stored ? new Set(JSON.parse(stored)) : new Set();
  } catch { return new Set(); }
}

function savePinnedChats() {
  localStorage.setItem(PINNED_STORAGE_KEY, JSON.stringify([...conversationsStore.pinnedChats]));
}

function saveArchivedChats() {
  localStorage.setItem(ARCHIVED_STORAGE_KEY, JSON.stringify([...conversationsStore.archivedChats]));
}

export function togglePinChat(chatId: string) {
  if (conversationsStore.pinnedChats.has(chatId)) {
    conversationsStore.pinnedChats.delete(chatId);
  } else {
    conversationsStore.pinnedChats.add(chatId);
  }
  conversationsStore.pinnedChats = new Set(conversationsStore.pinnedChats);
  savePinnedChats();
}

export function toggleArchiveChat(chatId: string) {
  if (conversationsStore.archivedChats.has(chatId)) {
    conversationsStore.archivedChats.delete(chatId);
  } else {
    conversationsStore.archivedChats.add(chatId);
  }
  conversationsStore.archivedChats = new Set(conversationsStore.archivedChats);
  saveArchivedChats();
}

export function isPinned(chatId: string): boolean {
  return conversationsStore.pinnedChats.has(chatId);
}

export function isArchived(chatId: string): boolean {
  return conversationsStore.archivedChats.has(chatId);
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

/**
 * Reconcile real messages with optimistic ones.
 * Appends new messages and removes corresponding optimistic placeholders.
 */
export function reconcileMessages(chatId: string, newMessages: Message[]) {
  if (newMessages.length === 0) return;

  const currentMessages = conversationsStore.messages;
  const existingIds = new Set(currentMessages.map((m) => m.id));
  
  // Only add truly new messages
  const filteredNew = newMessages.filter((m) => !existingIds.has(m.id));
  if (filteredNew.length === 0) return;

  // Append to state
  conversationsStore.messages = [...currentMessages, ...filteredNew];

  // Sync cache
  const cached = conversationsStore.messageCache.get(chatId);
  if (cached) {
    cached.messages = conversationsStore.messages;
  }

  // Clear matching optimistic messages (outgoing only)
  const outgoingNew = filteredNew.filter((m) => m.is_from_me);
  if (outgoingNew.length > 0 && conversationsStore.optimisticMessages.length > 0) {
    for (const realMsg of outgoingNew) {
      // Try exact text match
      const matching = conversationsStore.optimisticMessages.find(
        (opt) => opt.text.trim() === realMsg.text?.trim()
      );
      if (matching) {
        removeOptimisticMessage(matching.id);
      } else {
        // Fallback: clear oldest 'sent' optimistic message
        const oldestSent = conversationsStore.optimisticMessages.find(
          (opt) => opt.status === "sent"
        );
        if (oldestSent) removeOptimisticMessage(oldestSent.id);
      }
    }
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
  // Track initial load vs polling for UI distinction (UX-04)
  conversationsStore.isPolling = isPolling;
  conversationsStore.connectionStatus = "connecting";

  try {
    let conversations: Conversation[] = [];
    
    if (isTauri) {
      if (isDirectAccessAvailable()) {
        // Direct SQLite is ~50ms vs 1-3s through socket RPC
        conversations = await getConversationsDirect(CONVERSATIONS_PAGE_SIZE);
      } else {
        try {
          const result = await jarvis.call<{ conversations: Conversation[] }>("list_conversations", {
            limit: CONVERSATIONS_PAGE_SIZE,
          });
          conversations = result.conversations;
        } catch (e) {
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
    conversationsStore.isInitialLoad = false;
    conversationsStore.isPolling = false;
    conversationsStore.connectionStatus = "connected";
    conversationsStore.lastConversationFetchTime = Date.now();
  } catch (e) {
    conversationsStore.loading = false;
    conversationsStore.isInitialLoad = false;
    conversationsStore.isPolling = false;
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
    return messages.toReversed();
  } catch (e) {
    console.error("fetchMessages failed", e);
    return [];
  }
}

export async function pollMessages(force = false): Promise<Message[]> {
  if (!conversationsStore.selectedChatId) return [];
  if (!force && !conversationsStore.isWindowFocused) return [];
  if (conversationsStore._messagePolling) return []; // Prevent concurrent polls

  conversationsStore._messagePolling = true;
  const now = Date.now();
  const timeSinceLastSync = now - conversationsStore.lastSyncTimestamp;

  try {
    const chatId = conversationsStore.selectedChatId;

    if (isDirectAccessAvailable()) {
      const currentGlobalRowid = await getLastMessageRowid();
      // Skip if nothing changed globally AND we synchronized recently
      if (
        !force &&
        currentGlobalRowid > 0 &&
        currentGlobalRowid === conversationsStore.lastKnownGlobalRowid &&
        timeSinceLastSync < conversationsStore.SYNC_DEBOUNCE_MS
      ) {
        return [];
      }

      // Incremental fetch: only get messages newer than last known ROWID
      const newEntries = await getNewMessagesSince(conversationsStore.lastKnownGlobalRowid);
      conversationsStore.lastKnownGlobalRowid = currentGlobalRowid;

      // Filter to messages for the current chat
      const relevantEntries = newEntries.filter(e => e.chatId === chatId);
      if (relevantEntries.length === 0) return [];

      // Batch fetch all new messages in a single query (avoids N+1)
      const existingIds = new Set(conversationsStore.messages.map(m => m.id));
      const idsToFetch = relevantEntries
        .map(e => e.messageId)
        .filter(id => !existingIds.has(id));
      if (idsToFetch.length === 0) return [];

      const fetchedMessages = await getMessagesBatchDirect(chatId, idsToFetch);
      
      if (conversationsStore.selectedChatId !== chatId) return [];

      reconcileMessages(chatId, fetchedMessages);
      conversationsStore.lastSyncTimestamp = now;
      return fetchedMessages;
    }

    // Fallback: full re-fetch for non-direct access
    const freshMessages = await fetchMessages(chatId);
    if (conversationsStore.selectedChatId !== chatId) return [];

    reconcileMessages(chatId, freshMessages);
    conversationsStore.lastSyncTimestamp = now;
    return freshMessages;
  } catch (e) {
    return [];
  } finally {
    conversationsStore._messagePolling = false;
  }
}

// Track pending prefetch to avoid race conditions (FE-04)
let pendingPrefetchChatId: string | null = null;

export async function selectConversation(chatId: string) {
  if (conversationsStore.selectedChatId === chatId) return;

  clearNewMessageIndicator(chatId);

  // Track which chat the prefetch is for so we can discard stale results (FE-04)
  pendingPrefetchChatId = chatId;

  // Update focused chat synchronously BEFORE triggering any generation
  // This ensures stale generation gets cancelled immediately
  jarvis.call("prefetch_focus", { chat_id: chatId }).catch(() => {});

  // Background prefetch - fire and forget, but guard against stale results
  void jarvis.call<{
    status: string;
    prefetched?: boolean;
    draft?: { suggestions: DraftSuggestion[] };
    error?: string;
  }>("prefetch_focus", { chat_id: chatId }).then((result) => {
    // Only apply if the user is still on the same chat AND this is still the
    // active prefetch (not superseded by a newer selectConversation call) (FE-04)
    if (
      result?.prefetched &&
      result?.draft?.suggestions?.length &&
      conversationsStore.selectedChatId === chatId &&
      pendingPrefetchChatId === chatId
    ) {
      conversationsStore.prefetchedDraft = { chatId, suggestions: result.draft.suggestions };
    }
  }).catch(() => {});

  const cached = getCacheEntry(conversationsStore.messageCache, chatId);
  if (cached) {
    conversationsStore.selectedChatId = chatId;
    conversationsStore.messages = cached.messages;
    conversationsStore.hasMore = cached.pagination.hasMore;
    conversationsStore.loadingMore = false;
    conversationsStore.loadingMessages = false;
    conversationsStore.prefetchedDraft = null;

    // Start polling immediately - don't block on prefetch
    startMessagePolling();
    return;
  }

  conversationsStore.selectedChatId = chatId;
  conversationsStore.messages = [];
  conversationsStore.loadingMessages = true;
  conversationsStore.hasMore = true;
  conversationsStore.prefetchedDraft = null;

  try {
    const messages = await fetchMessages(chatId);
    if (conversationsStore.selectedChatId !== chatId) return;

    const hasMore = messages.length >= PAGE_SIZE;
    setCacheEntry(conversationsStore.messageCache, chatId, {
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
    let olderMessages: Message[];
    if (isDirectAccessAvailable()) {
      olderMessages = await getMessagesDirect(selectedChatId, PAGE_SIZE, new Date(beforeDate));
    } else {
      olderMessages = await api.getMessages(selectedChatId, PAGE_SIZE, beforeDate);
    }
    const newHasMore = olderMessages.length >= PAGE_SIZE;
    const chronologicalOlder = [...olderMessages].reverse();

    const newMessages = [...chronologicalOlder, ...conversationsStore.messages];
    setCacheEntry(conversationsStore.messageCache, selectedChatId, {
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

// Socket push handler for real-time new messages
export async function handleNewMessagePush(data: NewMessageEvent) {
  const { chat_id, message_id, text_preview } = data;

  if (chat_id !== conversationsStore.selectedChatId) {
    markConversationAsNew(chat_id);
  } else {
    // Append to current conversation
    if (isDirectAccessAvailable()) {
      const msg = await getMessageDirect(chat_id, message_id);
      if (msg) {
        reconcileMessages(chat_id, [msg]);
      }
    } else {
      const freshMessages = await fetchMessages(chat_id);
      if (conversationsStore.selectedChatId === chat_id) {
        reconcileMessages(chat_id, freshMessages);
      }
    }
  }

  // Update conversation list in-place instead of re-fetching
  updateConversationInPlace(chat_id, text_preview);
}

/** Update a conversation's preview and reorder to top without a network request. */
function updateConversationInPlace(chatId: string, textPreview: string | null) {
  const convos = conversationsStore.conversations;
  const idx = convos.findIndex((c) => c.chat_id === chatId);
  if (idx === -1) {
    // Unknown chat - need a full fetch to pick it up
    fetchConversations(true);
    return;
  }

  const now = new Date().toISOString();
  const conv = convos[idx]!;
  const updated: Conversation = { ...conv, last_message_date: now };
  if (textPreview !== null) {
    updated.last_message_text = textPreview;
  }

  // Move to top: remove from current position and prepend
  const next: Conversation[] = [updated, ...convos.slice(0, idx), ...convos.slice(idx + 1)];
  conversationsStore.conversations = next;

  // Update last known date to prevent false "new message" detection on next full fetch
  conversationsStore.lastKnownMessageDates.set(chatId, now);
}

/**
 * Update selected conversation preview/date immediately after a local send.
 * Keeps the conversation list in sync before watcher/poll catches up.
 */
export function updateConversationAfterLocalSend(chatId: string, textPreview: string) {
  updateConversationInPlace(chatId, textPreview);
  clearNewMessageIndicator(chatId);
}

// User activity tracking for adaptive polling
let lastUserActivity = Date.now();

export function handleUserActivity() {
  lastUserActivity = Date.now();
}

// Polling Logic
let conversationPollInterval: ReturnType<typeof setInterval> | null = null;
let messagePollTimeout: ReturnType<typeof setTimeout> | null = null;

export function startConversationPolling() {
  if (conversationPollInterval) clearInterval(conversationPollInterval);
  fetchConversations();
  conversationPollInterval = setInterval(() => {
    if (conversationsStore.isWindowFocused) {
      fetchConversations(true);
    }
  }, 30_000);
}

export function stopConversationPolling() {
  if (conversationPollInterval) clearInterval(conversationPollInterval);
  conversationPollInterval = null;
}

export function startMessagePolling() {
  if (messagePollTimeout) clearTimeout(messagePollTimeout);
  scheduleNextPoll();
}

function scheduleNextPoll() {
  const idle = Date.now() - lastUserActivity;
  const interval = idle < 60_000 ? 3_000 : idle < 300_000 ? 10_000 : 30_000;
  messagePollTimeout = setTimeout(() => {
    if (conversationsStore.isWindowFocused && conversationsStore.selectedChatId) {
      pollMessages().then(scheduleNextPoll);
    } else {
      scheduleNextPoll();
    }
  }, interval);
}

export function stopMessagePolling() {
  if (messagePollTimeout) clearTimeout(messagePollTimeout);
  messagePollTimeout = null;
}

/**
 * Re-resolve display names on already-rendered conversations using the contact cache.
 * Called after contacts load asynchronously so names replace phone numbers.
 */
function refreshConversationNames() {
  const convos = conversationsStore.conversations;
  if (convos.length === 0) return;

  let changed = false;
  const updated = convos.map((conv) => {
    // Only patch 1:1 chats that still show a phone number / no display name
    if (conv.is_group || conv.display_name) return conv;
    if (conv.participants.length !== 1) return conv;

    const resolved = resolveContactName(conv.participants[0]!);
    if (resolved && resolved !== conv.display_name) {
      changed = true;
      return { ...conv, display_name: resolved };
    }
    return conv;
  });

  if (changed) {
    conversationsStore.conversations = updated;
  }
}

export async function initializePolling(): Promise<() => void> {
  await initDatabases();
  // Load pinned/archived state from localStorage
  conversationsStore.pinnedChats = loadPinnedChats();
  conversationsStore.archivedChats = loadArchivedChats();
  if (isDirectAccessAvailable()) {
    conversationsStore.lastKnownGlobalRowid = await getLastMessageRowid();

    // Fire contact loading without blocking conversation render.
    // Conversations show phone numbers initially, then names patch in.
    const contactsP = loadContactsFromAddressBook().catch(() => {});

    // Start conversation polling immediately (renders with phone numbers)
    startConversationPolling();

    // When contacts arrive, patch display names on already-rendered conversations
    await contactsP;
    if (isContactsCacheLoaded()) {
      refreshConversationNames();
    }

    // Socket fallback if AddressBook failed (only if still no contacts)
    if (!isContactsCacheLoaded()) {
      try {
        const contacts = await jarvis.call<Record<string, string | null>>("get_contacts", {});
        if (contacts && typeof contacts === "object") {
          populateContactsCache(contacts);
          refreshConversationNames();
        }
      } catch {
        // Contact resolution not critical - will fall back to phone numbers
      }
    }
  } else {
    startConversationPolling();
  }
  
  const FOCUS_STALE_MS = 60000; // Only re-fetch if >60s since last fetch
  const handleFocus = () => {
    conversationsStore.isWindowFocused = true;
    const elapsed = Date.now() - conversationsStore.lastConversationFetchTime;
    if (elapsed > FOCUS_STALE_MS) {
      fetchConversations(true);
    }
    if (conversationsStore.selectedChatId) pollMessages();
  };
  const handleBlur = () => {
    conversationsStore.isWindowFocused = false;
  };

  window.addEventListener("focus", handleFocus);
  window.addEventListener("blur", handleBlur);

  const unlistenNewMsg = jarvis.on<NewMessageEvent>("new_message", handleNewMessagePush);

  return () => {
    stopConversationPolling();
    stopMessagePolling();
    window.removeEventListener("focus", handleFocus);
    window.removeEventListener("blur", handleBlur);
    unlistenNewMsg();
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
  conversationsStore.hasMore = false;
  conversationsStore.loadingMore = false;
  stopMessagePolling();
}

export function invalidateMessageCache(chatId?: string) {
  if (chatId) {
    conversationsStore.messageCache.delete(chatId);
  } else {
    conversationsStore.messageCache.clear();
  }
}
