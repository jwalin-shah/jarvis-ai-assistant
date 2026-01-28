/**
 * JARVIS v2 App Store
 *
 * Centralized state management using Svelte stores.
 */

import { writable, derived, get } from "svelte/store";
import type { Conversation, Message, GeneratedReply } from "../api/types";
import { api } from "../api/client";

// Connection status
export type ConnectionStatus = "connected" | "disconnected" | "connecting";

// App state
export interface AppState {
  conversations: Conversation[];
  selectedChatId: string | null;
  messages: Message[];
  replies: GeneratedReply[];
  loading: boolean;
  loadingMessages: boolean;
  loadingReplies: boolean;
  error: string | null;
  connectionStatus: ConnectionStatus;
  // Track last seen date per conversation for unread indicators
  lastSeenDate: Record<string, string>;
}

const initialState: AppState = {
  conversations: [],
  selectedChatId: null,
  messages: [],
  replies: [],
  loading: false,
  loadingMessages: false,
  loadingReplies: false,
  error: null,
  connectionStatus: "disconnected",
  lastSeenDate: {},
};

// Main store
export const appStore = writable<AppState>(initialState);

// Derived stores
export const selectedConversation = derived(appStore, ($state) =>
  $state.conversations.find((c) => c.chat_id === $state.selectedChatId) || null
);

export const connectionStatus = derived(appStore, ($state) => $state.connectionStatus);

// Get unread status for a conversation
export const unreadChats = derived(appStore, ($state) => {
  const unread = new Set<string>();
  for (const conv of $state.conversations) {
    // If I sent the last message, it's not unread
    if (conv.last_message_is_from_me) {
      continue;
    }

    // Check if we've seen this conversation before
    const lastSeen = $state.lastSeenDate[conv.chat_id];
    if (!lastSeen && conv.last_message_date) {
      // Never opened this conversation and someone else sent a message
      unread.add(conv.chat_id);
    } else if (lastSeen && conv.last_message_date && conv.last_message_date > lastSeen) {
      // New message since we last opened it
      unread.add(conv.chat_id);
    }
  }
  return unread;
});

// Actions
export async function fetchConversations(): Promise<void> {
  appStore.update((s) => ({ ...s, loading: true, error: null, connectionStatus: "connecting" }));

  try {
    const conversations = await api.getConversations();
    appStore.update((s) => ({
      ...s,
      conversations,
      loading: false,
      connectionStatus: "connected",
    }));
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to fetch conversations";
    appStore.update((s) => ({
      ...s,
      loading: false,
      error: message,
      connectionStatus: "disconnected",
    }));
  }
}

export async function selectConversation(chatId: string): Promise<void> {
  appStore.update((s) => ({
    ...s,
    selectedChatId: chatId,
    messages: [],
    replies: [],
    loadingMessages: true,
    error: null,
  }));

  try {
    const messages = await api.getMessages(chatId);
    // Reverse so oldest is first (API returns newest first)
    const chronological = [...messages].reverse();

    // Mark as read - record the last message date
    const lastMsgDate =
      chronological.length > 0 ? chronological[chronological.length - 1].timestamp : null;

    appStore.update((s) => ({
      ...s,
      messages: chronological,
      loadingMessages: false,
      lastSeenDate: lastMsgDate
        ? { ...s.lastSeenDate, [chatId]: lastMsgDate }
        : s.lastSeenDate,
    }));
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to fetch messages";
    appStore.update((s) => ({
      ...s,
      loadingMessages: false,
      error: message,
    }));
  }
}

export async function generateReplies(): Promise<void> {
  const state = get(appStore);
  if (!state.selectedChatId) return;

  appStore.update((s) => ({ ...s, loadingReplies: true, replies: [] }));

  try {
    const response = await api.generateReplies(state.selectedChatId);
    appStore.update((s) => ({
      ...s,
      replies: response.replies,
      loadingReplies: false,
    }));
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to generate replies";
    appStore.update((s) => ({
      ...s,
      loadingReplies: false,
      error: message,
    }));
  }
}

export function clearSelection(): void {
  appStore.update((s) => ({
    ...s,
    selectedChatId: null,
    messages: [],
    replies: [],
  }));
}

export function clearError(): void {
  appStore.update((s) => ({ ...s, error: null }));
}

export async function sendMessage(text: string): Promise<boolean> {
  const state = get(appStore);
  if (!state.selectedChatId || !text.trim()) return false;

  try {
    const result = await api.sendMessage(state.selectedChatId, text);
    if (result.success) {
      // Refresh messages to show the sent message
      setTimeout(() => {
        const currentState = get(appStore);
        if (currentState.selectedChatId) {
          selectConversation(currentState.selectedChatId);
        }
      }, 500);
      return true;
    } else {
      appStore.update((s) => ({
        ...s,
        error: result.error || "Failed to send message",
      }));
      return false;
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to send message";
    appStore.update((s) => ({ ...s, error: message }));
    return false;
  }
}

// Polling
let pollInterval: ReturnType<typeof setInterval> | null = null;

export function startPolling(intervalMs: number = 30000): void {
  stopPolling();
  fetchConversations();
  pollInterval = setInterval(() => {
    fetchConversations();
  }, intervalMs);
}

export function stopPolling(): void {
  if (pollInterval) {
    clearInterval(pollInterval);
    pollInterval = null;
  }
}
