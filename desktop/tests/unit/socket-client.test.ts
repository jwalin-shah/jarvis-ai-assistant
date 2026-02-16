/**
 * Unit tests for JarvisSocket client
 *
 * These tests run in Node environment (isTauri = false), so they exercise
 * the WebSocket (browser) code path. Tauri-specific paths are not covered.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { JarvisSocket } from "@/lib/socket/client";
import type { ConnectionState } from "@/lib/socket/client";

// ---------------------------------------------------------------------------
// MockWebSocket
// ---------------------------------------------------------------------------
class MockWebSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;

  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onerror: ((e: unknown) => void) | null = null;

  readyState = MockWebSocket.CONNECTING;

  send = vi.fn();
  close = vi.fn(() => {
    this.readyState = MockWebSocket.CLOSED;
    if (this.onclose) this.onclose();
  });

  /** Simulate the server opening the connection */
  simulateOpen(): void {
    this.readyState = MockWebSocket.OPEN;
    if (this.onopen) this.onopen();
  }

  /** Simulate receiving a message from the server */
  simulateMessage(data: object): void {
    if (this.onmessage) {
      this.onmessage({ data: JSON.stringify(data) });
    }
  }

  /** Simulate an error event */
  simulateError(error: unknown): void {
    if (this.onerror) this.onerror(error);
  }
}

// Install MockWebSocket as the global WebSocket before each test
let mockWsInstance: MockWebSocket | null = null;

function installMockWebSocket(): void {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (globalThis as any).WebSocket = class extends MockWebSocket {
    constructor(_url: string) {
      super();
      mockWsInstance = this;
    }
  };
  // WebSocket readyState constants expected by the source
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (globalThis as any).WebSocket.OPEN = MockWebSocket.OPEN;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (globalThis as any).WebSocket.CONNECTING = MockWebSocket.CONNECTING;
}

function removeMockWebSocket(): void {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  delete (globalThis as any).WebSocket;
  mockWsInstance = null;
}

// ---------------------------------------------------------------------------
// Helper: create a connected JarvisSocket in WebSocket (browser) mode
// ---------------------------------------------------------------------------
async function createConnectedClient(): Promise<{
  client: JarvisSocket;
  ws: MockWebSocket;
}> {
  const client = new JarvisSocket();
  const connectPromise = client.connect();

  // The constructor of MockWebSocket stashes the instance in `mockWsInstance`
  const ws = mockWsInstance!;
  expect(ws).not.toBeNull();

  // Simulate successful open
  ws.simulateOpen();

  const connected = await connectPromise;
  expect(connected).toBe(true);
  expect(client.getState()).toBe("connected");

  return { client, ws };
}

// ===========================================================================
// Tests
// ===========================================================================

describe("JarvisSocket", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    installMockWebSocket();
  });

  afterEach(async () => {
    vi.useRealTimers();
    removeMockWebSocket();
  });

  // -----------------------------------------------------------------------
  // Constructor & initial state
  // -----------------------------------------------------------------------
  describe("initial state", () => {
    it("starts in disconnected state", () => {
      const client = new JarvisSocket();
      expect(client.getState()).toBe("disconnected");
    });

    it("getState returns a valid ConnectionState", () => {
      const client = new JarvisSocket();
      const validStates: ConnectionState[] = [
        "disconnected",
        "connecting",
        "connected",
      ];
      expect(validStates).toContain(client.getState());
    });
  });

  // -----------------------------------------------------------------------
  // Event handler registration & dispatch
  // -----------------------------------------------------------------------
  describe("event handlers (on / off)", () => {
    it("registers a handler and calls it on emit", async () => {
      const client = new JarvisSocket();
      const handler = vi.fn();

      client.on("test_event", handler);

      // Trigger an internal emit by connecting (emits "connecting" and "connected")
      const connectPromise = client.connect();
      mockWsInstance!.simulateOpen();
      await connectPromise;

      expect(handler).not.toHaveBeenCalled(); // "test_event" was not emitted

      // Listen for "connected" instead
      const connectedHandler = vi.fn();
      client.on("connected", connectedHandler);

      // Reconnect to trigger "connected" again
      await client.disconnect();
      const reconnectPromise = client.connect();
      mockWsInstance!.simulateOpen();
      await reconnectPromise;

      expect(connectedHandler).toHaveBeenCalledTimes(1);
    });

    it("on() returns an unsubscribe function", async () => {
      const client = new JarvisSocket();
      const handler = vi.fn();
      const unsub = client.on("disconnected", handler);

      // Trigger disconnect
      const connectPromise = client.connect();
      mockWsInstance!.simulateOpen();
      await connectPromise;

      // Unsubscribe before disconnect
      unsub();
      await client.disconnect();

      // The handler for "disconnected" from the ws.close() callback fires,
      // but our handler was removed before we called disconnect()
      expect(handler).not.toHaveBeenCalled();
    });

    it("off() removes a specific handler", async () => {
      const { client } = await createConnectedClient();
      const handler = vi.fn();
      client.on("disconnected", handler);
      client.off("disconnected", handler);

      await client.disconnect();
      expect(handler).not.toHaveBeenCalled();
    });

    it("multiple handlers on the same event all fire", async () => {
      const client = new JarvisSocket();
      const handler1 = vi.fn();
      const handler2 = vi.fn();

      client.on("connected", handler1);
      client.on("connected", handler2);

      const connectPromise = client.connect();
      mockWsInstance!.simulateOpen();
      await connectPromise;

      expect(handler1).toHaveBeenCalledTimes(1);
      expect(handler2).toHaveBeenCalledTimes(1);
    });

    it("removing one handler does not affect others on the same event", async () => {
      const client = new JarvisSocket();
      const handler1 = vi.fn();
      const handler2 = vi.fn();

      client.on("connected", handler1);
      client.on("connected", handler2);
      client.off("connected", handler1);

      const connectPromise = client.connect();
      mockWsInstance!.simulateOpen();
      await connectPromise;

      expect(handler1).not.toHaveBeenCalled();
      expect(handler2).toHaveBeenCalledTimes(1);
    });
  });

  // -----------------------------------------------------------------------
  // Connection state transitions
  // -----------------------------------------------------------------------
  describe("connection state transitions", () => {
    it("transitions to connecting then connected on success", async () => {
      const client = new JarvisSocket();
      const states: ConnectionState[] = [];

      client.on("connecting", () => states.push("connecting"));
      client.on("connected", () => states.push("connected"));

      const connectPromise = client.connect();

      // Should be in "connecting" state now
      expect(client.getState()).toBe("connecting");

      mockWsInstance!.simulateOpen();
      await connectPromise;

      expect(client.getState()).toBe("connected");
      expect(states).toEqual(["connecting", "connected"]);
    });

    it("returns true immediately if already connected", async () => {
      const { client } = await createConnectedClient();
      const result = await client.connect();
      expect(result).toBe(true);
    });

    it("transitions to disconnected on disconnect()", async () => {
      const { client } = await createConnectedClient();
      const handler = vi.fn();
      client.on("disconnected", handler);

      await client.disconnect();

      expect(client.getState()).toBe("disconnected");
      expect(handler).toHaveBeenCalled();
    });

    it("connection timeout resolves false after 5 seconds", async () => {
      const client = new JarvisSocket();
      const connectPromise = client.connect();

      // Don't call simulateOpen - let the 5s timeout fire
      vi.advanceTimersByTime(5000);

      const result = await connectPromise;
      expect(result).toBe(false);
      expect(client.getState()).toBe("disconnected");
    });
  });

  // -----------------------------------------------------------------------
  // WebSocket request/response matching
  // -----------------------------------------------------------------------
  describe("WebSocket request/response", () => {
    it("sends JSON-RPC request and resolves on matching response", async () => {
      const { client, ws } = await createConnectedClient();

      const callPromise = client.call("ping");

      // The client should have sent a JSON-RPC request
      expect(ws.send).toHaveBeenCalledTimes(1);
      const sentMsg = JSON.parse(ws.send.mock.calls[0][0]);
      expect(sentMsg).toMatchObject({
        jsonrpc: "2.0",
        method: "ping",
        params: {},
      });
      expect(typeof sentMsg.id).toBe("number");

      // Simulate server response
      ws.simulateMessage({
        id: sentMsg.id,
        result: { status: "pong" },
      });

      const result = await callPromise;
      expect(result).toEqual({ status: "pong" });
    });

    it("rejects on server error response", async () => {
      const { client, ws } = await createConnectedClient();

      const callPromise = client.call("bad_method");

      const sentMsg = JSON.parse(ws.send.mock.calls[0][0]);
      ws.simulateMessage({
        id: sentMsg.id,
        error: { message: "Method not found" },
      });

      await expect(callPromise).rejects.toThrow("Method not found");
    });

    it("request IDs are unique and sequential", async () => {
      const { client, ws } = await createConnectedClient();

      // Fire off two calls
      const call1 = client.call("ping");
      const call2 = client.call("ping");

      expect(ws.send).toHaveBeenCalledTimes(2);
      const id1 = JSON.parse(ws.send.mock.calls[0][0]).id;
      const id2 = JSON.parse(ws.send.mock.calls[1][0]).id;

      expect(id1).not.toBe(id2);
      expect(id2).toBe(id1 + 1);

      // Resolve both so the test can clean up
      ws.simulateMessage({ id: id1, result: { status: "pong" } });
      ws.simulateMessage({ id: id2, result: { status: "pong" } });
      await Promise.all([call1, call2]);
    });

    it("request timeout rejects after configured duration", async () => {
      const { client, ws } = await createConnectedClient();

      const callPromise = client.call("slow_method");
      expect(ws.send).toHaveBeenCalledTimes(1);

      // Advance past the 30s request timeout
      vi.advanceTimersByTime(30001);

      await expect(callPromise).rejects.toThrow("Request timeout");
    });

    it("passes params in the JSON-RPC request", async () => {
      const { client, ws } = await createConnectedClient();

      const params = { text: "hello world", limit: 5 };
      const callPromise = client.call("classify_intent", params);

      const sentMsg = JSON.parse(ws.send.mock.calls[0][0]);
      expect(sentMsg.params).toEqual(params);

      ws.simulateMessage({
        id: sentMsg.id,
        result: { intent: "greeting", confidence: 0.95 },
      });
      await callPromise;
    });
  });

  // -----------------------------------------------------------------------
  // WebSocket message parsing (handleWebSocketMessage)
  // -----------------------------------------------------------------------
  describe("WebSocket message parsing", () => {
    it("dispatches new_message notification to event handlers", async () => {
      const { client, ws } = await createConnectedClient();
      const handler = vi.fn();
      client.on("new_message", handler);

      ws.simulateMessage({
        method: "new_message",
        params: {
          message_id: 42,
          chat_id: "chat123",
          sender: "Alice",
          text_preview: "Hey there",
          is_from_me: false,
        },
      });

      expect(handler).toHaveBeenCalledTimes(1);
      expect(handler).toHaveBeenCalledWith(
        expect.objectContaining({
          message_id: 42,
          chat_id: "chat123",
          sender: "Alice",
        })
      );
    });

    it("ignores responses with no matching pending request", async () => {
      const { ws } = await createConnectedClient();

      // This should not throw - just silently ignored
      ws.simulateMessage({
        id: 99999,
        result: { status: "pong" },
      });
    });

    it("handles malformed JSON gracefully", async () => {
      const { ws } = await createConnectedClient();
      const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

      // Send raw bad JSON
      if (ws.onmessage) {
        ws.onmessage({ data: "not valid json{{{" });
      }

      expect(warnSpy).toHaveBeenCalledWith(
        expect.stringContaining("Failed to parse WebSocket message"),
        expect.anything()
      );

      warnSpy.mockRestore();
    });
  });

  // -----------------------------------------------------------------------
  // Disconnect behavior
  // -----------------------------------------------------------------------
  describe("disconnect", () => {
    it("rejects all pending requests on WebSocket close", async () => {
      const { client, ws } = await createConnectedClient();

      const call1 = client.call("slow1");
      const call2 = client.call("slow2");

      // Simulate the WebSocket closing (server disconnect)
      ws.readyState = MockWebSocket.CLOSED;
      if (ws.onclose) ws.onclose();

      await expect(call1).rejects.toThrow("WebSocket disconnected");
      await expect(call2).rejects.toThrow("WebSocket disconnected");
    });

    it("clears pending requests on explicit disconnect()", async () => {
      const { client, ws } = await createConnectedClient();

      // Start a request but don't respond
      const callPromise = client.call("hanging");

      await client.disconnect();

      // The pending request should be rejected because the ws.onclose handler fires
      // when we call ws.close() from disconnect()
      await expect(callPromise).rejects.toThrow("WebSocket disconnected");
    });

    it("rejects batched requests on disconnect", async () => {
      const { client } = await createConnectedClient();

      // Queue a batched call - won't flush yet due to BATCH_WINDOW_MS
      const batchPromise = client.callBatched("test_method");

      // Disconnect before the batch flushes
      await client.disconnect();

      await expect(batchPromise).rejects.toThrow("Disconnected");
    });
  });

  // -----------------------------------------------------------------------
  // High-level API methods
  // -----------------------------------------------------------------------
  describe("high-level API", () => {
    it("ping() calls 'ping' method and returns result", async () => {
      const { client, ws } = await createConnectedClient();

      const pingPromise = client.ping();

      const sentMsg = JSON.parse(ws.send.mock.calls[0][0]);
      expect(sentMsg.method).toBe("ping");

      ws.simulateMessage({
        id: sentMsg.id,
        result: { status: "pong" },
      });

      const result = await pingPromise;
      expect(result).toEqual({ status: "pong" });
    });

    it("generateDraft() calls 'generate_draft' with params", async () => {
      const { client, ws } = await createConnectedClient();

      const draftPromise = client.generateDraft({
        chat_id: "chat_abc",
        instruction: "Be brief",
      });

      const sentMsg = JSON.parse(ws.send.mock.calls[0][0]);
      expect(sentMsg.method).toBe("generate_draft");
      expect(sentMsg.params).toEqual({
        chat_id: "chat_abc",
        instruction: "Be brief",
      });

      ws.simulateMessage({
        id: sentMsg.id,
        result: {
          suggestions: [{ text: "Sure!", confidence: 0.9 }],
          context_used: { num_messages: 5, participants: ["Alice"], last_message: "Hi" },
        },
      });

      const result = await draftPromise;
      expect(result.suggestions).toHaveLength(1);
      expect(result.suggestions[0].text).toBe("Sure!");
    });

    it("classifyIntent() calls 'classify_intent' with text", async () => {
      const { client, ws } = await createConnectedClient();

      const intentPromise = client.classifyIntent("Where are you?");

      const sentMsg = JSON.parse(ws.send.mock.calls[0][0]);
      expect(sentMsg.method).toBe("classify_intent");
      expect(sentMsg.params).toEqual({ text: "Where are you?" });

      ws.simulateMessage({
        id: sentMsg.id,
        result: {
          intent: "question",
          confidence: 0.88,
          requires_response: true,
        },
      });

      const result = await intentPromise;
      expect(result.intent).toBe("question");
      expect(result.requires_response).toBe(true);
    });

    it("summarize() calls 'summarize' with chatId and numMessages", async () => {
      const { client, ws } = await createConnectedClient();

      const summaryPromise = client.summarize("chat_xyz", 25);

      const sentMsg = JSON.parse(ws.send.mock.calls[0][0]);
      expect(sentMsg.method).toBe("summarize");
      expect(sentMsg.params).toEqual({
        chat_id: "chat_xyz",
        num_messages: 25,
      });

      ws.simulateMessage({
        id: sentMsg.id,
        result: {
          summary: "A friendly chat",
          key_points: ["Said hello"],
          message_count: 25,
        },
      });

      const result = await summaryPromise;
      expect(result.summary).toBe("A friendly chat");
    });

    it("semanticSearch() calls 'semantic_search' with params", async () => {
      const { client, ws } = await createConnectedClient();

      const searchPromise = client.semanticSearch({
        query: "weekend plans",
        limit: 10,
      });

      const sentMsg = JSON.parse(ws.send.mock.calls[0][0]);
      expect(sentMsg.method).toBe("semantic_search");
      expect(sentMsg.params).toEqual({ query: "weekend plans", limit: 10 });

      ws.simulateMessage({
        id: sentMsg.id,
        result: { results: [], total_results: 0 },
      });

      const result = await searchPromise;
      expect(result.total_results).toBe(0);
    });

    it("getSmartReplies() calls 'get_smart_replies' with params", async () => {
      const { client, ws } = await createConnectedClient();

      const repliesPromise = client.getSmartReplies({
        last_message: "Want to grab lunch?",
        num_suggestions: 3,
      });

      const sentMsg = JSON.parse(ws.send.mock.calls[0][0]);
      expect(sentMsg.method).toBe("get_smart_replies");

      ws.simulateMessage({
        id: sentMsg.id,
        result: {
          suggestions: [
            { text: "Sure!", score: 0.9 },
            { text: "Maybe later", score: 0.7 },
          ],
        },
      });

      const result = await repliesPromise;
      expect(result.suggestions).toHaveLength(2);
    });
  });

  // -----------------------------------------------------------------------
  // Auto-connect on call()
  // -----------------------------------------------------------------------
  describe("auto-connect on call()", () => {
    it("attempts to connect if not already connected", async () => {
      const client = new JarvisSocket();
      expect(client.getState()).toBe("disconnected");

      // call() triggers connect() internally, which is async.
      // We need to let connect() resolve before callWebSocket sends.
      const callPromise = client.call("ping");

      // The constructor of MockWebSocket stashes the instance
      const ws = mockWsInstance!;
      expect(ws).not.toBeNull();
      ws.simulateOpen();

      // Flush microtask queue so callWebSocket runs after connect resolves
      await vi.advanceTimersByTimeAsync(0);

      expect(ws.send).toHaveBeenCalled();
      const sentMsg = JSON.parse(ws.send.mock.calls[0][0]);
      ws.simulateMessage({
        id: sentMsg.id,
        result: { status: "pong" },
      });

      const result = await callPromise;
      expect(result).toEqual({ status: "pong" });
    });

    it("rejects call if connect fails", async () => {
      const client = new JarvisSocket();

      const callPromise = client.call("ping");

      // Let the 5s connection timeout fire without opening
      vi.advanceTimersByTime(5000);

      await expect(callPromise).rejects.toThrow("Not connected to socket server");
    });
  });

  // -----------------------------------------------------------------------
  // Batch request handling
  // -----------------------------------------------------------------------
  describe("batch requests", () => {
    it("single batched call is sent directly (no batch wrapper)", async () => {
      const { client, ws } = await createConnectedClient();

      const batchPromise = client.callBatched("ping");

      // Flush the batch timer
      vi.advanceTimersByTime(20);

      // Should have sent a direct request, not a batch
      expect(ws.send).toHaveBeenCalledTimes(1);
      const sentMsg = JSON.parse(ws.send.mock.calls[0][0]);
      expect(sentMsg.method).toBe("ping");

      ws.simulateMessage({
        id: sentMsg.id,
        result: { status: "pong" },
      });

      const result = await batchPromise;
      expect(result).toEqual({ status: "pong" });
    });

    it("multiple rapid calls are batched into a single request", async () => {
      const { client, ws } = await createConnectedClient();

      const p1 = client.callBatched("ping");
      const p2 = client.callBatched("classify_intent", { text: "hi" });

      // Flush the batch timer (BATCH_WINDOW_MS = 15ms)
      vi.advanceTimersByTime(20);

      // Should have sent one batch request
      expect(ws.send).toHaveBeenCalledTimes(1);
      const sentMsg = JSON.parse(ws.send.mock.calls[0][0]);
      expect(sentMsg.method).toBe("batch");
      expect(sentMsg.params.requests).toHaveLength(2);
      expect(sentMsg.params.requests[0].method).toBe("ping");
      expect(sentMsg.params.requests[1].method).toBe("classify_intent");

      // Simulate batch response
      ws.simulateMessage({
        id: sentMsg.id,
        result: {
          results: [
            { result: { status: "pong" } },
            { result: { intent: "greeting", confidence: 0.9 } },
          ],
        },
      });

      const [r1, r2] = await Promise.all([p1, p2]);
      expect(r1).toEqual({ status: "pong" });
      expect(r2).toEqual({ intent: "greeting", confidence: 0.9 });
    });

    it("batch with error results rejects individual promises", async () => {
      const { client, ws } = await createConnectedClient();

      const p1 = client.callBatched("good_method");
      const p2 = client.callBatched("bad_method");

      vi.advanceTimersByTime(20);

      const sentMsg = JSON.parse(ws.send.mock.calls[0][0]);

      ws.simulateMessage({
        id: sentMsg.id,
        result: {
          results: [
            { result: { ok: true } },
            { error: { message: "Not allowed" } },
          ],
        },
      });

      const r1 = await p1;
      expect(r1).toEqual({ ok: true });
      await expect(p2).rejects.toThrow("Not allowed");
    });

    it("batch failure rejects all individual promises", async () => {
      const { client, ws } = await createConnectedClient();

      const p1 = client.callBatched("method1");
      const p2 = client.callBatched("method2");

      vi.advanceTimersByTime(20);

      const sentMsg = JSON.parse(ws.send.mock.calls[0][0]);

      // Simulate server error on the batch itself
      ws.simulateMessage({
        id: sentMsg.id,
        error: { message: "Internal server error" },
      });

      await expect(p1).rejects.toThrow("Internal server error");
      await expect(p2).rejects.toThrow("Internal server error");
    });

    it("flushes immediately when MAX_BATCH_SIZE is reached", async () => {
      const { client, ws } = await createConnectedClient();

      // MAX_BATCH_SIZE = 10, queue exactly 10 requests
      const promises = [];
      for (let i = 0; i < 10; i++) {
        promises.push(client.callBatched("ping"));
      }

      // Should flush immediately without needing timer advance
      expect(ws.send).toHaveBeenCalledTimes(1);
      const sentMsg = JSON.parse(ws.send.mock.calls[0][0]);
      expect(sentMsg.method).toBe("batch");
      expect(sentMsg.params.requests).toHaveLength(10);

      // Respond to clean up
      ws.simulateMessage({
        id: sentMsg.id,
        result: {
          results: Array.from({ length: 10 }, () => ({ result: { status: "pong" } })),
        },
      });

      await Promise.all(promises);
    });
  });

  // -----------------------------------------------------------------------
  // Streaming (WebSocket path)
  // -----------------------------------------------------------------------
  describe("streaming via callStream", () => {
    it("sends request with stream: true and yields tokens", async () => {
      const { client, ws } = await createConnectedClient();

      const tokens: string[] = [];
      const tokenCallback = vi.fn();

      const streamPromise = (async () => {
        for await (const token of client.callStream(
          "generate_draft",
          { chat_id: "abc" },
          tokenCallback
        )) {
          tokens.push(token);
        }
      })();

      // Let the async generator start and register its pending request
      await vi.advanceTimersByTimeAsync(0);

      // Verify request was sent with stream: true
      const sentMsg = JSON.parse(ws.send.mock.calls[0][0]);
      expect(sentMsg.method).toBe("generate_draft");
      expect(sentMsg.params.stream).toBe(true);
      const requestId = sentMsg.id;

      // Simulate streaming tokens with microtask flushes between them
      // so the async generator can consume each one
      ws.simulateMessage({
        method: "stream.token",
        params: { token: "Hello", index: 0, final: false, request_id: requestId },
      });
      await vi.advanceTimersByTimeAsync(0);

      ws.simulateMessage({
        method: "stream.token",
        params: { token: " world", index: 1, final: false, request_id: requestId },
      });
      await vi.advanceTimersByTimeAsync(0);

      // Final response resolves the stream
      ws.simulateMessage({
        id: requestId,
        result: { done: true },
      });
      await vi.advanceTimersByTimeAsync(0);

      await streamPromise;

      expect(tokens).toEqual(["Hello", " world"]);
      expect(tokenCallback).toHaveBeenCalledTimes(2);
      expect(tokenCallback).toHaveBeenCalledWith("Hello", 0);
      expect(tokenCallback).toHaveBeenCalledWith(" world", 1);
    });

    it("streaming rejects on server error", async () => {
      const { client, ws } = await createConnectedClient();

      const streamPromise = (async () => {
        const collected: string[] = [];
        for await (const token of client.callStream("failing_method")) {
          collected.push(token);
        }
        return collected;
      })();

      const sentMsg = JSON.parse(ws.send.mock.calls[0][0]);

      // Simulate error response
      ws.simulateMessage({
        id: sentMsg.id,
        error: { message: "Generation failed" },
      });

      await expect(streamPromise).rejects.toThrow("Generation failed");
    });
  });

  // -----------------------------------------------------------------------
  // Reconnection scheduling
  // -----------------------------------------------------------------------
  describe("reconnection", () => {
    it("schedules reconnect on WebSocket disconnect", async () => {
      const { client, ws } = await createConnectedClient();
      const handler = vi.fn();
      client.on("disconnected", handler);

      // Simulate server-side close (not client-initiated)
      ws.readyState = MockWebSocket.CLOSED;
      if (ws.onclose) ws.onclose();

      expect(client.getState()).toBe("disconnected");
      expect(handler).toHaveBeenCalled();

      // After first reconnect delay (1000 * 2^0 = 1000ms)
      vi.advanceTimersByTime(1000);

      // A new MockWebSocket should have been created for the reconnect attempt
      expect(mockWsInstance).not.toBeNull();
    });

    it("emits max_reconnect_attempts after exhausting retries", async () => {
      const client = new JarvisSocket();
      const maxReconnectHandler = vi.fn();
      client.on("max_reconnect_attempts", maxReconnectHandler);

      // First connect attempt - trigger connection timeout
      const connectPromise = client.connect();
      vi.advanceTimersByTime(5000);
      await connectPromise;

      // Now exhaust reconnect attempts (5 max, exponential backoff)
      // Each attempt: timer fires -> connect() called -> 5s timeout -> scheduleReconnect
      for (let i = 0; i < 5; i++) {
        const delay = 1000 * Math.pow(2, i);
        vi.advanceTimersByTime(delay);
        // Wait for the connect() call to start and the WebSocket to be created
        await vi.advanceTimersByTimeAsync(5000);
      }

      expect(maxReconnectHandler).toHaveBeenCalled();
    });

    it("disconnect() cancels pending reconnect timer", async () => {
      const { client, ws } = await createConnectedClient();

      // Force a disconnect that would schedule reconnect
      ws.readyState = MockWebSocket.CLOSED;
      if (ws.onclose) ws.onclose();

      // Immediately call disconnect to cancel the reconnect
      await client.disconnect();

      // Advance past when reconnect would have fired
      vi.advanceTimersByTime(10000);

      expect(client.getState()).toBe("disconnected");
    });
  });

  // -----------------------------------------------------------------------
  // isConnected
  // -----------------------------------------------------------------------
  describe("isConnected()", () => {
    it("returns true when WebSocket is open", async () => {
      const { client } = await createConnectedClient();
      const connected = await client.isConnected();
      expect(connected).toBe(true);
    });

    it("returns false when disconnected", async () => {
      const client = new JarvisSocket();
      const connected = await client.isConnected();
      expect(connected).toBe(false);
    });

    it("returns false after disconnect()", async () => {
      const { client } = await createConnectedClient();
      await client.disconnect();
      const connected = await client.isConnected();
      expect(connected).toBe(false);
    });
  });

  // -----------------------------------------------------------------------
  // Error event emission
  // -----------------------------------------------------------------------
  describe("error handling", () => {
    it("emits error event on WebSocket error", async () => {
      const { client, ws } = await createConnectedClient();
      const errorHandler = vi.fn();
      client.on("error", errorHandler);

      ws.simulateError(new Error("connection lost"));

      expect(errorHandler).toHaveBeenCalledTimes(1);
      expect(errorHandler).toHaveBeenCalledWith(
        expect.objectContaining({ error: expect.any(Error) })
      );
    });
  });
});
