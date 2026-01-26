import { writable, derived } from "svelte/store";

const API_URL = "http://localhost:8742";

export interface Attachment {
  filename: string;
  file_path: string | null;
  mime_type: string | null;
  file_size: number | null;
}

export interface Reaction {
  type: string;
  sender: string;
  sender_name: string | null;
  date: string;
}

export interface Message {
  id: number;
  chat_id: string;
  sender: string;
  sender_name: string | null;
  text: string;
  date: string;
  is_from_me: boolean;
  attachments: Attachment[];
  reply_to_id: number | null;
  reactions: Reaction[];
  date_delivered: string | null;
  date_read: string | null;
  is_system_message: boolean;
}

export interface Conversation {
  chat_id: string;
  participants: string[];
  display_name: string | null;
  last_message_date: string;
  message_count: number;
  is_group: boolean;
  last_message_text: string | null;
}

// Stores
export const conversations = writable<Conversation[]>([]);
export const selectedConversation = writable<string | null>(null);
export const messages = writable<Message[]>([]);
export const loadingConversations = writable<boolean>(false);
export const loadingMessages = writable<boolean>(false);

// Derived store for selected conversation details
export const selectedConversationDetails = derived(
  [conversations, selectedConversation],
  ([$conversations, $selected]) => {
    if (!$selected) return null;
    return $conversations.find((c) => c.chat_id === $selected) || null;
  }
);

// Clear selection
export function clearSelection(): void {
  selectedConversation.set(null);
  messages.set([]);
}

// Select a conversation and load its messages
export async function selectConversation(chatId: string): Promise<void> {
  selectedConversation.set(chatId);
  await fetchMessages(chatId);
}

// Fetch all conversations
export async function fetchConversations(): Promise<Conversation[]> {
  loadingConversations.set(true);
  try {
    const response = await fetch(`${API_URL}/conversations`);
    if (response.ok) {
      const data = await response.json();
      conversations.set(data);
      return data;
    }
    return [];
  } catch {
    return [];
  } finally {
    loadingConversations.set(false);
  }
}

// Fetch messages for a conversation
export async function fetchMessages(
  chatId: string,
  limit: number = 100
): Promise<Message[]> {
  loadingMessages.set(true);
  try {
    const response = await fetch(
      `${API_URL}/conversations/${encodeURIComponent(chatId)}/messages?limit=${limit}`
    );
    if (response.ok) {
      const data = await response.json();
      messages.set(data);
      return data;
    }
    return [];
  } catch {
    return [];
  } finally {
    loadingMessages.set(false);
  }
}

// Search messages
export async function searchMessages(
  query: string,
  limit: number = 50
): Promise<Message[]> {
  try {
    const response = await fetch(
      `${API_URL}/messages/search?query=${encodeURIComponent(query)}&limit=${limit}`
    );
    if (response.ok) {
      return await response.json();
    }
    return [];
  } catch {
    return [];
  }
}
