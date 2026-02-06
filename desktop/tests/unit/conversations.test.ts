/**
 * Unit tests for the conversations store
 *
 * Tests cover: initial state, derived stores, optimistic messages,
 * fetch/select/load/poll lifecycle, caching, polling intervals,
 * window focus, navigation, socket push handling, and prefetch drafts.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { get } from "svelte/store";
import type { Conversation, Message } from "../../src/lib/api/types";

// ---------------------------------------------------------------------------
// Mocks - declared before any module that imports them
// ---------------------------------------------------------------------------

vi.mock("../../src/lib/api/client", () => ({
  api: {
    getConversations: vi.fn(),
    getMessages: vi.fn(),
  },
}));

vi.mock("../../src/lib/db", () => ({
  initDatabases: vi.fn(),
  isDirectAccessAvailable: vi.fn(() => false),
  getConversations: vi.fn(),
  getMessages: vi.fn(),
  getMessage: vi.fn(),
  getLastMessageRowid: vi.fn(() => 0),
  getNewMessagesSince: vi.fn(),
  populateContactsCache: vi.fn(),
  isContactsCacheLoaded: vi.fn(() => false),
}));

vi.mock("../../src/lib/socket", () => ({
  jarvis: {
    connect: vi.fn(() => Promise.resolve(false)),
    disconnect: vi.fn(),
    on: vi.fn(),
    off: vi.fn(),
    call: vi.fn(() => Promise.resolve(null)),
  },
}));

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeConversation(overrides: Partial<Conversation> = {}): Conversation {
  return {
    chat_id: "chat-1",
    participants: ["alice"],
    display_name: "Alice",
    last_message_date: "2024-01-01T00:00:00Z",
    message_count: 10,
    is_group: false,
    last_message_text: "Hello",
    ...overrides,
  };
}

function makeMessage(overrides: Partial<Message> = {}): Message {
  return {
    id: 1,
    chat_id: "chat-1",
    sender: "alice",
    sender_name: "Alice",
    text: "Hello",
    date: "2024-01-01T00:00:00Z",
    is_from_me: false,
    attachments: [],
    reply_to_id: null,
    reactions: [],
    date_delivered: null,
    date_read: null,
    is_system_message: false,
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// Module-level references re-imported in each beforeEach
// ---------------------------------------------------------------------------

type ConversationsModule = typeof import("../../src/lib/stores/conversations");

let mod: ConversationsModule;
let conversationsStore: ConversationsModule["conversationsStore"];
let selectedConversation: ConversationsModule["selectedConversation"];
let hasNewMessages: ConversationsModule["hasNewMessages"];
let messagesWithOptimistic: ConversationsModule["messagesWithOptimistic"];
let highlightedMessageId: ConversationsModule["highlightedMessageId"];
let scrollToMessageId: ConversationsModule["scrollToMessageId"];

let markConversationAsNew: ConversationsModule["markConversationAsNew"];
let clearNewMessageIndicator: ConversationsModule["clearNewMessageIndicator"];
let addOptimisticMessage: ConversationsModule["addOptimisticMessage"];
let updateOptimisticMessage: ConversationsModule["updateOptimisticMessage"];
let removeOptimisticMessage: ConversationsModule["removeOptimisticMessage"];
let clearOptimisticMessages: ConversationsModule["clearOptimisticMessages"];
let clearPrefetchedDraft: ConversationsModule["clearPrefetchedDraft"];
let fetchConversations: ConversationsModule["fetchConversations"];
let fetchMessages: ConversationsModule["fetchMessages"];
let selectConversation: ConversationsModule["selectConversation"];
let loadMoreMessages: ConversationsModule["loadMoreMessages"];
let pollMessages: ConversationsModule["pollMessages"];
let clearSelection: ConversationsModule["clearSelection"];
let invalidateMessageCache: ConversationsModule["invalidateMessageCache"];
let startConversationPolling: ConversationsModule["startConversationPolling"];
let stopConversationPolling: ConversationsModule["stopConversationPolling"];
let startMessagePolling: ConversationsModule["startMessagePolling"];
let stopMessagePolling: ConversationsModule["stopMessagePolling"];
let setWindowFocused: ConversationsModule["setWindowFocused"];
let navigateToMessage: ConversationsModule["navigateToMessage"];
let initializePolling: ConversationsModule["initializePolling"];

// Mock references
let mockApi: {
  getConversations: ReturnType<typeof vi.fn>;
  getMessages: ReturnType<typeof vi.fn>;
};
let mockDb: Record<string, ReturnType<typeof vi.fn>>;
let mockJarvis: Record<string, ReturnType<typeof vi.fn>>;

// ==========================================================================
// Fresh module import per test
// ==========================================================================

beforeEach(async () => {
  vi.useFakeTimers();
  vi.resetModules();

  // Re-import mocked dependencies
  const apiMod = await import("../../src/lib/api/client");
  mockApi = apiMod.api as unknown as typeof mockApi;

  const dbMod = await import("../../src/lib/db");
  mockDb = dbMod as unknown as Record<string, ReturnType<typeof vi.fn>>;

  const socketMod = await import("../../src/lib/socket");
  mockJarvis = socketMod.jarvis as unknown as Record<string, ReturnType<typeof vi.fn>>;

  // Reset all mock call counts and implementations to fresh state
  mockApi.getConversations.mockReset();
  mockApi.getMessages.mockReset();
  Object.values(mockDb).forEach((fn) => {
    if (typeof fn === "function" && "mockReset" in fn) fn.mockReset();
  });
  Object.values(mockJarvis).forEach((fn) => {
    if (typeof fn === "function" && "mockReset" in fn) fn.mockReset();
  });

  // Restore default mock implementations after reset
  mockDb.isDirectAccessAvailable.mockReturnValue(false);
  mockDb.getLastMessageRowid.mockResolvedValue(0);
  mockDb.isContactsCacheLoaded.mockReturnValue(false);
  mockJarvis.connect.mockResolvedValue(false);
  mockJarvis.call.mockResolvedValue(null);

  // Import the module under test (fresh copy)
  mod = await import("../../src/lib/stores/conversations");
  conversationsStore = mod.conversationsStore;
  selectedConversation = mod.selectedConversation;
  hasNewMessages = mod.hasNewMessages;
  messagesWithOptimistic = mod.messagesWithOptimistic;
  highlightedMessageId = mod.highlightedMessageId;
  scrollToMessageId = mod.scrollToMessageId;
  markConversationAsNew = mod.markConversationAsNew;
  clearNewMessageIndicator = mod.clearNewMessageIndicator;
  addOptimisticMessage = mod.addOptimisticMessage;
  updateOptimisticMessage = mod.updateOptimisticMessage;
  removeOptimisticMessage = mod.removeOptimisticMessage;
  clearOptimisticMessages = mod.clearOptimisticMessages;
  clearPrefetchedDraft = mod.clearPrefetchedDraft;
  fetchConversations = mod.fetchConversations;
  fetchMessages = mod.fetchMessages;
  selectConversation = mod.selectConversation;
  loadMoreMessages = mod.loadMoreMessages;
  pollMessages = mod.pollMessages;
  clearSelection = mod.clearSelection;
  invalidateMessageCache = mod.invalidateMessageCache;
  startConversationPolling = mod.startConversationPolling;
  stopConversationPolling = mod.stopConversationPolling;
  startMessagePolling = mod.startMessagePolling;
  stopMessagePolling = mod.stopMessagePolling;
  setWindowFocused = mod.setWindowFocused;
  navigateToMessage = mod.navigateToMessage;
  initializePolling = mod.initializePolling;
});

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

// ==========================================================================
// 1. Initial state
// ==========================================================================

describe("initial state", () => {
  it("starts with correct default values", () => {
    const state = get(conversationsStore);
    expect(state.conversations).toEqual([]);
    expect(state.selectedChatId).toBeNull();
    expect(state.messages).toEqual([]);
    expect(state.loading).toBe(false);
    expect(state.loadingMessages).toBe(false);
    expect(state.loadingMore).toBe(false);
    expect(state.hasMore).toBe(false);
    expect(state.error).toBeNull();
    expect(state.connectionStatus).toBe("disconnected");
    expect(state.conversationsWithNewMessages.size).toBe(0);
    expect(state.isWindowFocused).toBe(true);
    expect(state.optimisticMessages).toEqual([]);
    expect(state.prefetchedDraft).toBeNull();
  });

  it("selectedConversation derived store returns null initially", () => {
    expect(get(selectedConversation)).toBeNull();
  });

  it("hasNewMessages derived store returns false for any chatId", () => {
    const checker = get(hasNewMessages);
    expect(checker("chat-1")).toBe(false);
    expect(checker("nonexistent")).toBe(false);
  });
});

// ==========================================================================
// 2. markConversationAsNew / clearNewMessageIndicator
// ==========================================================================

describe("markConversationAsNew / clearNewMessageIndicator", () => {
  it("marks a conversation as having new messages", () => {
    markConversationAsNew("chat-1");
    const checker = get(hasNewMessages);
    expect(checker("chat-1")).toBe(true);
  });

  it("marking the same chatId twice is idempotent (returns same state ref)", () => {
    markConversationAsNew("chat-1");
    const stateAfterFirst = get(conversationsStore);
    markConversationAsNew("chat-1");
    const stateAfterSecond = get(conversationsStore);
    // The early-return path should yield the same reference
    expect(stateAfterFirst).toBe(stateAfterSecond);
  });

  it("clearing removes from the set", () => {
    markConversationAsNew("chat-1");
    clearNewMessageIndicator("chat-1");
    const checker = get(hasNewMessages);
    expect(checker("chat-1")).toBe(false);
  });

  it("clearing a non-existent chatId is idempotent", () => {
    const stateBefore = get(conversationsStore);
    clearNewMessageIndicator("nonexistent");
    const stateAfter = get(conversationsStore);
    expect(stateBefore).toBe(stateAfter);
  });

  it("multiple conversations can be marked independently", () => {
    markConversationAsNew("chat-1");
    markConversationAsNew("chat-2");
    const checker = get(hasNewMessages);
    expect(checker("chat-1")).toBe(true);
    expect(checker("chat-2")).toBe(true);

    clearNewMessageIndicator("chat-1");
    const checker2 = get(hasNewMessages);
    expect(checker2("chat-1")).toBe(false);
    expect(checker2("chat-2")).toBe(true);
  });
});

// ==========================================================================
// 3. Optimistic message lifecycle
// ==========================================================================

describe("optimistic message lifecycle", () => {
  it("addOptimisticMessage creates message with 'sending' status and returns unique ID", () => {
    const id = addOptimisticMessage("Hello");
    expect(id).toMatch(/^optimistic-/);

    const state = get(conversationsStore);
    expect(state.optimisticMessages).toHaveLength(1);
    expect(state.optimisticMessages[0].text).toBe("Hello");
    expect(state.optimisticMessages[0].status).toBe("sending");
    expect(state.optimisticMessages[0].id).toBe(id);
  });

  it("multiple adds create unique IDs", () => {
    const id1 = addOptimisticMessage("First");
    const id2 = addOptimisticMessage("Second");
    expect(id1).not.toBe(id2);
    expect(get(conversationsStore).optimisticMessages).toHaveLength(2);
  });

  it("updateOptimisticMessage changes status to 'sent'", () => {
    const id = addOptimisticMessage("Hello");
    updateOptimisticMessage(id, { status: "sent" });

    const msg = get(conversationsStore).optimisticMessages[0];
    expect(msg.status).toBe("sent");
  });

  it("updateOptimisticMessage changes status to 'failed' with error message", () => {
    const id = addOptimisticMessage("Hello");
    updateOptimisticMessage(id, { status: "failed", error: "Network error" });

    const msg = get(conversationsStore).optimisticMessages[0];
    expect(msg.status).toBe("failed");
    expect(msg.error).toBe("Network error");
  });

  it("updateOptimisticMessage for non-existent ID is a no-op", () => {
    addOptimisticMessage("Hello");
    // Should not throw
    updateOptimisticMessage("nonexistent-id", { status: "sent" });
    const msgs = get(conversationsStore).optimisticMessages;
    expect(msgs).toHaveLength(1);
    expect(msgs[0].status).toBe("sending"); // unchanged
  });

  it("removeOptimisticMessage removes the correct message", () => {
    const id1 = addOptimisticMessage("First");
    const id2 = addOptimisticMessage("Second");
    removeOptimisticMessage(id1);

    const msgs = get(conversationsStore).optimisticMessages;
    expect(msgs).toHaveLength(1);
    expect(msgs[0].id).toBe(id2);
  });

  it("removeOptimisticMessage for non-existent ID does not crash", () => {
    addOptimisticMessage("Hello");
    removeOptimisticMessage("nonexistent");
    expect(get(conversationsStore).optimisticMessages).toHaveLength(1);
  });

  it("clearOptimisticMessages clears all", () => {
    addOptimisticMessage("First");
    addOptimisticMessage("Second");
    clearOptimisticMessages();
    expect(get(conversationsStore).optimisticMessages).toEqual([]);
  });
});

// ==========================================================================
// 4. messagesWithOptimistic derived store
// ==========================================================================

describe("messagesWithOptimistic derived store", () => {
  it("returns messages directly when no optimistic messages (short-circuit)", () => {
    const msgs = [makeMessage({ id: 1 }), makeMessage({ id: 2 })];
    conversationsStore.update((s) => ({ ...s, messages: msgs }));

    const result = get(messagesWithOptimistic);
    // Should be the exact same reference (short-circuit path)
    expect(result).toBe(msgs);
  });

  it("appends optimistic messages with negative IDs and correct fields", () => {
    const msgs = [makeMessage({ id: 1 })];
    conversationsStore.update((s) => ({ ...s, messages: msgs }));
    addOptimisticMessage("Optimistic hello");

    const result = get(messagesWithOptimistic);
    expect(result).toHaveLength(2);
    // Real message first
    expect(result[0].id).toBe(1);
    // Optimistic message last with negative ID
    expect(result[1].id).toBeLessThan(0);
    expect(result[1].text).toBe("Optimistic hello");
    expect(result[1].is_from_me).toBe(true);
  });

  it("optimistic messages have _optimistic: true and correct status", () => {
    conversationsStore.update((s) => ({ ...s, messages: [] }));
    const optId = addOptimisticMessage("Test");
    updateOptimisticMessage(optId, { status: "failed", error: "Oops" });

    const result = get(messagesWithOptimistic);
    expect(result).toHaveLength(1);
    const optMsg = result[0] as Message & {
      _optimistic: boolean;
      _optimisticId: string;
      _optimisticStatus: string;
      _optimisticError: string;
    };
    expect(optMsg._optimistic).toBe(true);
    expect(optMsg._optimisticId).toBe(optId);
    expect(optMsg._optimisticStatus).toBe("failed");
    expect(optMsg._optimisticError).toBe("Oops");
  });
});

// ==========================================================================
// 5. fetchConversations
// ==========================================================================

describe("fetchConversations", () => {
  it("sets loading=true and connectionStatus='connecting' before fetch", async () => {
    let capturedLoading = false;
    let capturedStatus = "";

    // Resolve slowly so we can inspect intermediate state
    mockApi.getConversations.mockImplementation(
      () =>
        new Promise((resolve) => {
          capturedLoading = get(conversationsStore).loading;
          capturedStatus = get(conversationsStore).connectionStatus;
          resolve([]);
        })
    );

    await fetchConversations();

    expect(capturedLoading).toBe(true);
    expect(capturedStatus).toBe("connecting");
  });

  it("on success: sets conversations, connectionStatus='connected', loading=false", async () => {
    const convos = [makeConversation({ chat_id: "c1" })];
    mockApi.getConversations.mockResolvedValueOnce(convos);

    await fetchConversations();

    const state = get(conversationsStore);
    expect(state.conversations).toEqual(convos);
    expect(state.connectionStatus).toBe("connected");
    expect(state.loading).toBe(false);
    expect(state.error).toBeNull();
  });

  it("on failure: sets error message and connectionStatus='disconnected'", async () => {
    mockApi.getConversations.mockRejectedValueOnce(new Error("Network fail"));

    await fetchConversations();

    const state = get(conversationsStore);
    expect(state.loading).toBe(false);
    expect(state.error).toBe("Network fail");
    expect(state.connectionStatus).toBe("disconnected");
  });

  it("abort error is silently ignored (no error state set)", async () => {
    const abortError = new DOMException("Aborted", "AbortError");
    mockApi.getConversations.mockRejectedValueOnce(abortError);

    await fetchConversations();

    const state = get(conversationsStore);
    // Error should not be set
    expect(state.error).toBeNull();
  });

  it("polling mode (isPolling=true) does not overwrite existing error", async () => {
    // Set an existing error
    conversationsStore.update((s) => ({ ...s, error: "Previous error" }));

    mockApi.getConversations.mockRejectedValueOnce(new Error("Poll fail"));

    await fetchConversations(true);

    const state = get(conversationsStore);
    // Should keep the previous error, not overwrite with poll failure
    expect(state.error).toBe("Previous error");
  });

  it("new message detection: marks conversation as new when last_message_date increases", async () => {
    // First fetch to establish known dates
    const convos1 = [makeConversation({ chat_id: "c1", last_message_date: "2024-01-01T00:00:00Z" })];
    mockApi.getConversations.mockResolvedValueOnce(convos1);
    await fetchConversations();

    // Second fetch with updated date
    const convos2 = [makeConversation({ chat_id: "c1", last_message_date: "2024-01-02T00:00:00Z" })];
    mockApi.getConversations.mockResolvedValueOnce(convos2);
    await fetchConversations();

    const checker = get(hasNewMessages);
    expect(checker("c1")).toBe(true);
  });

  it("new message detection: does NOT mark the currently selected conversation", async () => {
    // Set c1 as selected
    conversationsStore.update((s) => ({ ...s, selectedChatId: "c1" }));

    // First fetch to establish known dates
    const convos1 = [makeConversation({ chat_id: "c1", last_message_date: "2024-01-01T00:00:00Z" })];
    mockApi.getConversations.mockResolvedValueOnce(convos1);
    await fetchConversations();

    // Second fetch with updated date
    const convos2 = [makeConversation({ chat_id: "c1", last_message_date: "2024-01-02T00:00:00Z" })];
    mockApi.getConversations.mockResolvedValueOnce(convos2);
    await fetchConversations();

    const checker = get(hasNewMessages);
    expect(checker("c1")).toBe(false);
  });

  it("cancels previous in-flight fetch on new call", async () => {
    // First call: slow, should be aborted
    mockApi.getConversations.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          setTimeout(() => {
            resolve([makeConversation({ chat_id: "stale" })]);
          }, 5000);
        })
    );

    // Second call: fast
    mockApi.getConversations.mockResolvedValueOnce([
      makeConversation({ chat_id: "fresh" }),
    ]);

    const firstPromise = fetchConversations();
    const secondPromise = fetchConversations();

    // Advance timers to let the first call's timeout fire
    vi.advanceTimersByTime(5000);
    await firstPromise;
    await secondPromise;

    const state = get(conversationsStore);
    // Only the second call's data should be present
    expect(state.conversations).toHaveLength(1);
    expect(state.conversations[0].chat_id).toBe("fresh");
  });
});

// ==========================================================================
// 6. fetchMessages
// ==========================================================================

describe("fetchMessages", () => {
  it("returns messages in reverse chronological order (reversed)", async () => {
    const msgs = [
      makeMessage({ id: 3, date: "2024-01-03" }),
      makeMessage({ id: 2, date: "2024-01-02" }),
      makeMessage({ id: 1, date: "2024-01-01" }),
    ];
    mockApi.getMessages.mockResolvedValueOnce(msgs);

    const result = await fetchMessages("chat-1");

    // Should be reversed: oldest first
    expect(result[0].id).toBe(1);
    expect(result[1].id).toBe(2);
    expect(result[2].id).toBe(3);
  });

  it("falls back to HTTP when direct DB not available", async () => {
    mockDb.isDirectAccessAvailable.mockReturnValue(false);
    const msgs = [makeMessage({ id: 1 })];
    mockApi.getMessages.mockResolvedValueOnce(msgs);

    const result = await fetchMessages("chat-1");
    expect(mockApi.getMessages).toHaveBeenCalledWith("chat-1", 20);
    expect(result).toHaveLength(1);
  });

  it("returns empty array on error", async () => {
    mockApi.getMessages.mockRejectedValueOnce(new Error("fail"));

    const result = await fetchMessages("chat-1");
    expect(result).toEqual([]);
  });

  it("direct DB failure falls back to HTTP", async () => {
    mockDb.isDirectAccessAvailable.mockReturnValue(true);
    mockDb.getMessages.mockRejectedValueOnce(new Error("DB fail"));

    const httpMsgs = [makeMessage({ id: 99 })];
    mockApi.getMessages.mockResolvedValueOnce(httpMsgs);

    const result = await fetchMessages("chat-1");
    expect(result[0].id).toBe(99);
  });
});

// ==========================================================================
// 7. selectConversation
// ==========================================================================

describe("selectConversation", () => {
  it("sets selectedChatId and loadingMessages=true", async () => {
    let capturedState: { selectedChatId: string | null; loadingMessages: boolean } | null = null;

    // Use a single mockImplementationOnce so selectConversation uses HTTP path
    mockApi.getMessages.mockImplementationOnce(() => {
      capturedState = {
        selectedChatId: get(conversationsStore).selectedChatId,
        loadingMessages: get(conversationsStore).loadingMessages,
      };
      return Promise.resolve([]);
    });

    await selectConversation("chat-1");

    expect(capturedState).not.toBeNull();
    expect(capturedState!.selectedChatId).toBe("chat-1");
    expect(capturedState!.loadingMessages).toBe(true);
  });

  it("loads messages and caches them", async () => {
    const msgs = [makeMessage({ id: 1 }), makeMessage({ id: 2 })];
    mockApi.getMessages.mockResolvedValueOnce(msgs);

    await selectConversation("chat-1");

    const state = get(conversationsStore);
    expect(state.messages).toHaveLength(2);
    expect(state.loadingMessages).toBe(false);
  });

  it("uses cached messages on second select (cache hit)", async () => {
    const msgs = [makeMessage({ id: 1 })];
    mockApi.getMessages.mockResolvedValueOnce(msgs);

    // First select: fetches
    await selectConversation("chat-1");
    expect(mockApi.getMessages).toHaveBeenCalledTimes(1);

    // Second select: should use cache
    await selectConversation("chat-1");
    expect(mockApi.getMessages).toHaveBeenCalledTimes(1); // Not called again
  });

  it("clears new message indicator", async () => {
    markConversationAsNew("chat-1");
    mockApi.getMessages.mockResolvedValueOnce([]);

    await selectConversation("chat-1");

    const checker = get(hasNewMessages);
    expect(checker("chat-1")).toBe(false);
  });

  it("clears prefetched draft", async () => {
    conversationsStore.update((s) => ({
      ...s,
      prefetchedDraft: { chatId: "chat-1", suggestions: [{ text: "Hi", confidence: 0.9 }] },
    }));
    mockApi.getMessages.mockResolvedValueOnce([]);

    await selectConversation("chat-1");

    expect(get(conversationsStore).prefetchedDraft).toBeNull();
  });

  it("sets hasMore=false when fewer than PAGE_SIZE messages returned", async () => {
    // PAGE_SIZE is 20, return fewer
    const msgs = Array.from({ length: 5 }, (_, i) => makeMessage({ id: i + 1 }));
    mockApi.getMessages.mockResolvedValueOnce(msgs);

    await selectConversation("chat-1");

    expect(get(conversationsStore).hasMore).toBe(false);
  });

  it("sets hasMore=true when PAGE_SIZE messages returned", async () => {
    // PAGE_SIZE is 20, return exactly 20
    const msgs = Array.from({ length: 20 }, (_, i) => makeMessage({ id: i + 1 }));
    mockApi.getMessages.mockResolvedValueOnce(msgs);

    await selectConversation("chat-1");

    expect(get(conversationsStore).hasMore).toBe(true);
  });

  it("error during load sets error state", async () => {
    mockApi.getMessages.mockRejectedValueOnce(new Error("Load failed"));

    await selectConversation("chat-1");

    const state = get(conversationsStore);
    expect(state.loadingMessages).toBe(false);
    expect(state.error).toBe("Load failed");
  });
});

// ==========================================================================
// 8. loadMoreMessages
// ==========================================================================

describe("loadMoreMessages", () => {
  it("returns false when no conversation selected", async () => {
    const result = await loadMoreMessages();
    expect(result).toBe(false);
  });

  it("returns false when already loading", async () => {
    conversationsStore.update((s) => ({
      ...s,
      selectedChatId: "chat-1",
      messages: [makeMessage()],
      loadingMore: true,
      hasMore: true,
    }));

    const result = await loadMoreMessages();
    expect(result).toBe(false);
  });

  it("returns false when hasMore is false", async () => {
    conversationsStore.update((s) => ({
      ...s,
      selectedChatId: "chat-1",
      messages: [makeMessage()],
      loadingMore: false,
      hasMore: false,
    }));

    const result = await loadMoreMessages();
    expect(result).toBe(false);
  });

  it("returns false when messages array is empty", async () => {
    conversationsStore.update((s) => ({
      ...s,
      selectedChatId: "chat-1",
      messages: [],
      loadingMore: false,
      hasMore: true,
    }));

    const result = await loadMoreMessages();
    expect(result).toBe(false);
  });

  it("prepends older messages to existing messages", async () => {
    const existing = [makeMessage({ id: 10, date: "2024-01-10" })];
    conversationsStore.update((s) => ({
      ...s,
      selectedChatId: "chat-1",
      messages: existing,
      loadingMore: false,
      hasMore: true,
    }));

    // Older messages returned by API (newest first, will be reversed)
    const olderMsgs = [
      makeMessage({ id: 2, date: "2024-01-02" }),
      makeMessage({ id: 1, date: "2024-01-01" }),
    ];
    mockApi.getMessages.mockResolvedValueOnce(olderMsgs);

    const result = await loadMoreMessages();
    expect(result).toBe(true);

    const state = get(conversationsStore);
    // Reversed older messages prepended: [1, 2, 10]
    expect(state.messages).toHaveLength(3);
    expect(state.messages[0].id).toBe(1);
    expect(state.messages[1].id).toBe(2);
    expect(state.messages[2].id).toBe(10);
    expect(state.loadingMore).toBe(false);
  });

  it("sets hasMore=false when fewer than PAGE_SIZE returned", async () => {
    const existing = [makeMessage({ id: 10, date: "2024-01-10" })];
    conversationsStore.update((s) => ({
      ...s,
      selectedChatId: "chat-1",
      messages: existing,
      loadingMore: false,
      hasMore: true,
    }));

    // Return fewer than PAGE_SIZE (20)
    const olderMsgs = [makeMessage({ id: 1 })];
    mockApi.getMessages.mockResolvedValueOnce(olderMsgs);

    await loadMoreMessages();

    expect(get(conversationsStore).hasMore).toBe(false);
  });
});

// ==========================================================================
// 9. pollMessages - delta detection
// ==========================================================================

describe("pollMessages - delta detection", () => {
  it("returns empty when no conversation selected", async () => {
    const result = await pollMessages();
    expect(result).toEqual([]);
  });

  it("returns empty when window not focused", async () => {
    conversationsStore.update((s) => ({
      ...s,
      selectedChatId: "chat-1",
      isWindowFocused: false,
    }));

    const result = await pollMessages();
    expect(result).toEqual([]);
  });

  it("skips full fetch when global ROWID unchanged (delta optimization)", async () => {
    mockDb.isDirectAccessAvailable.mockReturnValue(true);
    // Set up: lastKnownGlobalRowid starts at 0, getLastMessageRowid returns 100
    mockDb.getLastMessageRowid.mockResolvedValueOnce(100);

    conversationsStore.update((s) => ({
      ...s,
      selectedChatId: "chat-1",
      isWindowFocused: true,
      messages: [makeMessage({ id: 1 })],
    }));
    mockApi.getMessages.mockResolvedValueOnce([makeMessage({ id: 1 })]);

    // First poll: ROWID changed (0 -> 100), should fetch
    await pollMessages();

    // Second poll: ROWID unchanged (100 -> 100), should skip
    mockDb.getLastMessageRowid.mockResolvedValueOnce(100);
    mockApi.getMessages.mockClear();

    const result = await pollMessages();

    // Should not have called getMessages again
    expect(mockApi.getMessages).not.toHaveBeenCalled();
    expect(result).toEqual([]);
  });

  it("returns new messages that were not in previous set", async () => {
    mockDb.isDirectAccessAvailable.mockReturnValue(false);

    const existingMsg = makeMessage({ id: 1, text: "Old" });
    const newMsg = makeMessage({ id: 2, text: "New" });

    conversationsStore.update((s) => ({
      ...s,
      selectedChatId: "chat-1",
      isWindowFocused: true,
      messages: [existingMsg],
    }));

    // fetchMessages returns both messages (newest first, gets reversed)
    mockApi.getMessages.mockResolvedValueOnce([newMsg, existingMsg]);

    const result = await pollMessages();

    expect(result).toHaveLength(1);
    expect(result[0].id).toBe(2);
  });

  it("discards results if user switched conversations during async fetch", async () => {
    mockDb.isDirectAccessAvailable.mockReturnValue(false);

    conversationsStore.update((s) => ({
      ...s,
      selectedChatId: "chat-1",
      isWindowFocused: true,
      messages: [makeMessage({ id: 1 })],
    }));

    // During the fetch, the user switches conversations
    mockApi.getMessages.mockImplementationOnce(async () => {
      // Simulate switching conversation during async gap
      conversationsStore.update((s) => ({ ...s, selectedChatId: "chat-2" }));
      return [makeMessage({ id: 1 }), makeMessage({ id: 2 })];
    });

    const result = await pollMessages();

    // Results should be discarded
    expect(result).toEqual([]);
  });
});

// ==========================================================================
// 10. clearSelection
// ==========================================================================

describe("clearSelection", () => {
  it("resets selectedChatId, messages, etc.", () => {
    conversationsStore.update((s) => ({
      ...s,
      selectedChatId: "chat-1",
      messages: [makeMessage()],
      hasMore: false,
      loadingMore: true,
    }));

    clearSelection();

    const state = get(conversationsStore);
    expect(state.selectedChatId).toBeNull();
    expect(state.messages).toEqual([]);
    expect(state.hasMore).toBe(true);
    expect(state.loadingMore).toBe(false);
  });

  it("stops message polling", () => {
    // Start message polling first
    conversationsStore.update((s) => ({
      ...s,
      selectedChatId: "chat-1",
      isWindowFocused: true,
    }));
    startMessagePolling();

    // Verify interval was set (we can check by clearing and checking no poll fires)
    clearSelection();

    // Advance time past the polling interval
    vi.advanceTimersByTime(60000);

    // No new messages should have been fetched after clearSelection
    expect(mockApi.getMessages).not.toHaveBeenCalled();
  });
});

// ==========================================================================
// 11. invalidateMessageCache
// ==========================================================================

describe("invalidateMessageCache", () => {
  it("invalidates specific chatId from cache", async () => {
    const msgs = [makeMessage({ id: 1 })];
    mockApi.getMessages.mockResolvedValue(msgs);

    // Populate cache
    await selectConversation("chat-1");
    expect(mockApi.getMessages).toHaveBeenCalledTimes(1);

    // Invalidate
    invalidateMessageCache("chat-1");

    // Next select should re-fetch
    await selectConversation("chat-1");
    expect(mockApi.getMessages).toHaveBeenCalledTimes(2);
  });

  it("invalidates all when no chatId given", async () => {
    const msgs = [makeMessage({ id: 1 })];
    mockApi.getMessages.mockResolvedValue(msgs);

    // Populate caches
    await selectConversation("chat-1");
    // Reset to populate another
    mockApi.getMessages.mockResolvedValue([makeMessage({ id: 2 })]);
    await selectConversation("chat-2");

    expect(mockApi.getMessages).toHaveBeenCalledTimes(2);

    // Invalidate all
    invalidateMessageCache();

    // Both should re-fetch
    mockApi.getMessages.mockResolvedValue([makeMessage({ id: 10 })]);
    await selectConversation("chat-1");
    expect(mockApi.getMessages).toHaveBeenCalledTimes(3);

    await selectConversation("chat-2");
    expect(mockApi.getMessages).toHaveBeenCalledTimes(4);
  });

  it("after invalidation, selectConversation fetches fresh data", async () => {
    const staleMsg = makeMessage({ id: 1, text: "stale" });
    mockApi.getMessages.mockResolvedValueOnce([staleMsg]);
    await selectConversation("chat-1");

    invalidateMessageCache("chat-1");

    const freshMsg = makeMessage({ id: 1, text: "fresh" });
    mockApi.getMessages.mockResolvedValueOnce([freshMsg]);
    await selectConversation("chat-1");

    const state = get(conversationsStore);
    expect(state.messages[0].text).toBe("fresh");
  });
});

// ==========================================================================
// 12. Polling intervals
// ==========================================================================

describe("polling intervals", () => {
  it("startConversationPolling sets up interval", () => {
    mockApi.getConversations.mockResolvedValue([]);

    startConversationPolling();

    // Initial fetch should have been triggered
    expect(mockApi.getConversations).toHaveBeenCalledTimes(1);

    // Advance past the disconnected interval (30 seconds)
    vi.advanceTimersByTime(30000);
    expect(mockApi.getConversations).toHaveBeenCalledTimes(2);
  });

  it("stopConversationPolling clears interval", () => {
    mockApi.getConversations.mockResolvedValue([]);

    startConversationPolling();
    expect(mockApi.getConversations).toHaveBeenCalledTimes(1);

    stopConversationPolling();

    // Advance time: no additional fetches should happen
    vi.advanceTimersByTime(120000);
    expect(mockApi.getConversations).toHaveBeenCalledTimes(1);
  });

  it("polls only when window is focused", () => {
    mockApi.getConversations.mockResolvedValue([]);

    startConversationPolling();
    expect(mockApi.getConversations).toHaveBeenCalledTimes(1);

    // Unfocus window
    conversationsStore.update((s) => ({ ...s, isWindowFocused: false }));

    // Advance past polling interval
    vi.advanceTimersByTime(30000);
    // Should not have polled because unfocused
    expect(mockApi.getConversations).toHaveBeenCalledTimes(1);

    // Refocus
    conversationsStore.update((s) => ({ ...s, isWindowFocused: true }));

    vi.advanceTimersByTime(30000);
    // Should have polled now
    expect(mockApi.getConversations).toHaveBeenCalledTimes(2);
  });

  it("uses shorter interval when socket disconnected", () => {
    mockApi.getConversations.mockResolvedValue([]);

    // Socket push is not active by default (socketPushActive = false)
    // so CONVERSATION_POLL_INTERVAL_DISCONNECTED (30s) should be used
    startConversationPolling();

    // After 30s (disconnected interval), should poll
    vi.advanceTimersByTime(30000);
    expect(mockApi.getConversations).toHaveBeenCalledTimes(2); // initial + 1 interval
  });
});

// ==========================================================================
// 13. Window focus
// ==========================================================================

describe("window focus", () => {
  it("setWindowFocused(true) triggers refetch", () => {
    mockApi.getConversations.mockResolvedValue([]);

    // Start unfocused
    conversationsStore.update((s) => ({ ...s, isWindowFocused: false }));

    setWindowFocused(true);

    // Should trigger fetchConversations
    expect(mockApi.getConversations).toHaveBeenCalled();
    expect(get(conversationsStore).isWindowFocused).toBe(true);
  });

  it("setWindowFocused(false) updates state", () => {
    setWindowFocused(false);
    expect(get(conversationsStore).isWindowFocused).toBe(false);
  });

  it("idempotent when already in same state (no state change emitted)", () => {
    // Initial state: isWindowFocused = true
    // setWindowFocused(true) early-returns the SAME state object from the update,
    // but fetchConversations is still called since the focus branch runs.
    // We verify idempotency of the store update itself: the isWindowFocused field
    // doesn't change.
    mockApi.getConversations.mockResolvedValue([]);

    const isWindowFocusedBefore = get(conversationsStore).isWindowFocused;
    setWindowFocused(true);
    const isWindowFocusedAfter = get(conversationsStore).isWindowFocused;

    expect(isWindowFocusedBefore).toBe(true);
    expect(isWindowFocusedAfter).toBe(true);
  });
});

// ==========================================================================
// 14. navigateToMessage
// ==========================================================================

describe("navigateToMessage", () => {
  it("selects conversation and sets scroll/highlight targets", async () => {
    mockApi.getMessages.mockResolvedValueOnce([makeMessage({ id: 42 })]);

    await navigateToMessage("chat-1", 42);

    expect(get(conversationsStore).selectedChatId).toBe("chat-1");
    expect(get(scrollToMessageId)).toBe(42);
    expect(get(highlightedMessageId)).toBe(42);
  });

  it("clears highlight after 3 seconds", async () => {
    mockApi.getMessages.mockResolvedValueOnce([makeMessage({ id: 42 })]);

    await navigateToMessage("chat-1", 42);

    expect(get(highlightedMessageId)).toBe(42);

    // Advance 3 seconds
    vi.advanceTimersByTime(3000);
    expect(get(highlightedMessageId)).toBeNull();
  });
});

// ==========================================================================
// 15. handleNewMessagePush (tested indirectly via initializeSocketPush)
// ==========================================================================

describe("handleNewMessagePush", () => {
  // Helper: get the 'new_message' handler registered via jarvis.on()
  function getNewMessageHandler(): (data: {
    message_id: number;
    chat_id: string;
    sender: string;
    text: string;
    date: string;
    is_from_me: boolean;
  }) => void {
    const onCalls = mockJarvis.on.mock.calls;
    const newMsgCall = onCalls.find(
      (call: unknown[]) => call[0] === "new_message"
    );
    if (!newMsgCall) {
      throw new Error("new_message handler not registered");
    }
    return newMsgCall[1] as (data: {
      message_id: number;
      chat_id: string;
      sender: string;
      text: string;
      date: string;
      is_from_me: boolean;
    }) => void;
  }

  /**
   * Initialize socket push WITHOUT using initializePolling (which starts intervals
   * that cause infinite timer loops). Instead, we directly trigger initializeSocketPush
   * by calling initializePolling and immediately stopping the polling afterwards.
   */
  async function setupSocketPush(): Promise<void> {
    // Make jarvis.connect resolve true so socket push initializes
    mockJarvis.connect.mockResolvedValueOnce(true);
    mockApi.getConversations.mockResolvedValue([]);

    // Stub window for initializePolling's addEventListener
    if (typeof globalThis.window === "undefined") {
      vi.stubGlobal("window", {
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
      });
    }

    const cleanup = initializePolling();

    // Flush only the microtask queue (Promises), not timers, to avoid
    // triggering the setInterval callbacks infinitely.
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();

    // Now stop all polling intervals so fake timers won't loop
    stopConversationPolling();
    stopMessagePolling();
  }

  it("marks non-selected conversation as new", async () => {
    await setupSocketPush();

    // No conversation is selected
    conversationsStore.update((s) => ({ ...s, selectedChatId: null }));

    // handleNewMessagePush calls fetchConversations(true) at the end
    mockApi.getConversations.mockResolvedValue([]);

    const handler = getNewMessageHandler();
    handler({
      message_id: 100,
      chat_id: "chat-other",
      sender: "alice",
      text: "Hi",
      date: "2024-01-02T00:00:00Z",
      is_from_me: false,
    });

    const checker = get(hasNewMessages);
    expect(checker("chat-other")).toBe(true);
  });

  it("does not mark selected conversation as new", async () => {
    await setupSocketPush();

    conversationsStore.update((s) => ({ ...s, selectedChatId: "chat-1" }));
    mockApi.getConversations.mockResolvedValue([]);

    const handler = getNewMessageHandler();
    handler({
      message_id: 100,
      chat_id: "chat-1",
      sender: "alice",
      text: "Hi",
      date: "2024-01-02T00:00:00Z",
      is_from_me: false,
    });

    const checker = get(hasNewMessages);
    expect(checker("chat-1")).toBe(false);
  });

  it("appends message to current conversation via fetchMessages fallback", async () => {
    await setupSocketPush();

    mockDb.isDirectAccessAvailable.mockReturnValue(false);

    const existingMsg = makeMessage({ id: 1 });
    const newMsg = makeMessage({ id: 2, text: "New push" });

    conversationsStore.update((s) => ({
      ...s,
      selectedChatId: "chat-1",
      messages: [existingMsg],
    }));

    // fetchMessages will be called for fallback path
    mockApi.getMessages.mockResolvedValueOnce([newMsg, existingMsg]);
    // handleNewMessagePush also calls fetchConversations(true)
    mockApi.getConversations.mockResolvedValue([]);

    const handler = getNewMessageHandler();
    handler({
      message_id: 2,
      chat_id: "chat-1",
      sender: "alice",
      text: "New push",
      date: "2024-01-02T00:00:00Z",
      is_from_me: false,
    });

    // Flush promises for the async fetchMessages call inside handleNewMessagePush
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();

    const state = get(conversationsStore);
    // The fallback fetchMessages replaces the messages array
    expect(state.messages).toHaveLength(2);
  });

  it("avoids duplicate message insertion (direct access path)", async () => {
    await setupSocketPush();

    mockDb.isDirectAccessAvailable.mockReturnValue(true);

    const existingMsg = makeMessage({ id: 100 });
    conversationsStore.update((s) => ({
      ...s,
      selectedChatId: "chat-1",
      messages: [existingMsg],
    }));

    // getMessage returns the same message that already exists
    mockDb.getMessage.mockResolvedValueOnce(existingMsg);
    mockApi.getConversations.mockResolvedValue([]);

    const handler = getNewMessageHandler();
    handler({
      message_id: 100,
      chat_id: "chat-1",
      sender: "alice",
      text: "Hi",
      date: "2024-01-02T00:00:00Z",
      is_from_me: false,
    });

    // Flush promises
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();

    // Should not have duplicated the message
    const state = get(conversationsStore);
    expect(state.messages).toHaveLength(1);
  });
});

// ==========================================================================
// 16. clearPrefetchedDraft
// ==========================================================================

describe("clearPrefetchedDraft", () => {
  it("clears draft when one exists", () => {
    conversationsStore.update((s) => ({
      ...s,
      prefetchedDraft: { chatId: "chat-1", suggestions: [{ text: "Hi", confidence: 0.9 }] },
    }));

    clearPrefetchedDraft();

    expect(get(conversationsStore).prefetchedDraft).toBeNull();
  });

  it("no-op when no draft exists", () => {
    const stateBefore = get(conversationsStore);
    clearPrefetchedDraft();
    const stateAfter = get(conversationsStore);
    // Early return should yield same reference
    expect(stateBefore).toBe(stateAfter);
  });
});
