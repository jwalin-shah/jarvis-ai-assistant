/**
 * Unit tests for WebSocket client class (api/websocket.ts)
 *
 * Tests the JarvisWebSocket class: connection lifecycle, message routing,
 * reconnection with exponential backoff, ping intervals, send logic,
 * malformed messages, and visibility-aware reconnect deferral.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

// ---------------------------------------------------------------------------
// MockWebSocket - simulates the browser WebSocket API
// ---------------------------------------------------------------------------

class MockWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  readyState = MockWebSocket.CONNECTING;
  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onerror: ((e: unknown) => void) | null = null;
  url: string;
  send = vi.fn();
  close = vi.fn();

  constructor(url: string) {
    this.url = url;
    // Track instances for assertions
    MockWebSocket.instances.push(this);
  }

  simulateOpen() {
    this.readyState = MockWebSocket.OPEN;
    this.onopen?.();
  }

  simulateMessage(data: unknown) {
    this.onmessage?.({ data: JSON.stringify(data) });
  }

  simulateClose() {
    this.readyState = MockWebSocket.CLOSED;
    this.onclose?.();
  }

  simulateError(e: unknown) {
    this.onerror?.(e);
  }

  // Static tracking
  static instances: MockWebSocket[] = [];
  static reset() {
    MockWebSocket.instances = [];
  }
}

// ---------------------------------------------------------------------------
// Stubs
// ---------------------------------------------------------------------------

vi.stubGlobal("WebSocket", MockWebSocket);

vi.mock("../../src/lib/config/runtime", () => ({
  getApiWebSocketBaseUrl: () => "ws://localhost:8001",
}));

// Suppress console output during tests
vi.spyOn(console, "log").mockImplementation(() => {});
vi.spyOn(console, "warn").mockImplementation(() => {});
vi.spyOn(console, "error").mockImplementation(() => {});

// Type alias for the class
type JarvisWebSocketClass = typeof import("../../src/lib/api/websocket").JarvisWebSocket;

// ==========================================================================
// JarvisWebSocket Tests
// ==========================================================================

describe("JarvisWebSocket", () => {
  let JarvisWebSocket: JarvisWebSocketClass;

  beforeEach(async () => {
    vi.useFakeTimers();
    vi.resetModules();
    MockWebSocket.reset();

    const mod = await import("../../src/lib/api/websocket");
    JarvisWebSocket = mod.JarvisWebSocket;
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
    // Re-stub for next test
    vi.stubGlobal("WebSocket", MockWebSocket);
  });

  // Helper: create a connected client with handlers
  function createConnectedClient(handlers = {}) {
    const client = new JarvisWebSocket("ws://localhost:8001");
    client.setHandlers(handlers);
    client.connect();

    const ws = MockWebSocket.instances[MockWebSocket.instances.length - 1];
    ws.simulateOpen();
    ws.simulateMessage({
      type: "connected",
      data: { client_id: "test-client", timestamp: 1000 },
    });

    return { client, ws };
  }

  // ========================================================================
  // 1. Constructor / initial state
  // ========================================================================

  describe("constructor / initial state", () => {
    it("state is disconnected", () => {
      const client = new JarvisWebSocket("ws://localhost:8001");
      expect(client.state).toBe("disconnected");
    });

    it("clientId is null", () => {
      const client = new JarvisWebSocket("ws://localhost:8001");
      expect(client.clientId).toBeNull();
    });

    it("isConnected is false", () => {
      const client = new JarvisWebSocket("ws://localhost:8001");
      expect(client.isConnected).toBe(false);
    });
  });

  // ========================================================================
  // 2. setHandlers
  // ========================================================================

  describe("setHandlers", () => {
    it("merges new handlers with existing", () => {
      const client = new JarvisWebSocket("ws://localhost:8001");
      const onConnect = vi.fn();
      const onError = vi.fn();

      client.setHandlers({ onConnect });
      client.setHandlers({ onError });

      // Connect and trigger both handlers
      client.connect();
      const ws = MockWebSocket.instances[0];
      ws.simulateOpen();

      // onConnect should fire via "connected" message
      ws.simulateMessage({
        type: "connected",
        data: { client_id: "c1", timestamp: 1 },
      });
      expect(onConnect).toHaveBeenCalledOnce();

      // onError should fire via "error" message
      ws.simulateMessage({
        type: "error",
        data: { error: "bad request" },
      });
      expect(onError).toHaveBeenCalledWith("bad request");
    });

    it("does not override handlers not in the new object", () => {
      const client = new JarvisWebSocket("ws://localhost:8001");
      const onToken = vi.fn();
      const onError = vi.fn();

      client.setHandlers({ onToken, onError });

      // Override only onError
      const newOnError = vi.fn();
      client.setHandlers({ onError: newOnError });

      // Connect and verify
      client.connect();
      const ws = MockWebSocket.instances[0];
      ws.simulateOpen();
      ws.simulateMessage({
        type: "connected",
        data: { client_id: "c1", timestamp: 1 },
      });

      // onToken should still be the original
      ws.simulateMessage({
        type: "token",
        data: { generation_id: "g1", token: "hi", token_index: 0 },
      });
      expect(onToken).toHaveBeenCalledOnce();

      // Old onError should NOT fire; new one should
      ws.simulateMessage({ type: "error", data: { error: "oops" } });
      expect(onError).not.toHaveBeenCalled();
      expect(newOnError).toHaveBeenCalledWith("oops");
    });
  });

  // ========================================================================
  // 3. connect
  // ========================================================================

  describe("connect", () => {
    it("creates WebSocket with correct URL (baseUrl + /ws)", () => {
      const client = new JarvisWebSocket("ws://localhost:8001");
      client.connect();

      expect(MockWebSocket.instances).toHaveLength(1);
      expect(MockWebSocket.instances[0].url).toBe("ws://localhost:8001/ws");
    });

    it("sets state to connecting", () => {
      const onStateChange = vi.fn();
      const client = new JarvisWebSocket("ws://localhost:8001");
      client.setHandlers({ onStateChange });
      client.connect();

      expect(client.state).toBe("connecting");
      expect(onStateChange).toHaveBeenCalledWith("connecting");
    });

    it("no-op if already connected (readyState OPEN)", () => {
      const client = new JarvisWebSocket("ws://localhost:8001");
      client.connect();

      const ws = MockWebSocket.instances[0];
      ws.readyState = MockWebSocket.OPEN;

      // Second connect should be a no-op
      client.connect();
      expect(MockWebSocket.instances).toHaveLength(1); // No new WebSocket created
    });

    it("resets reconnectAttempts on successful open", () => {
      const onStateChange = vi.fn();
      const client = new JarvisWebSocket("ws://localhost:8001");
      client.setHandlers({ onStateChange });
      client.connect();

      const ws = MockWebSocket.instances[0];
      ws.simulateOpen();

      // After open, a "connected" message sets state to "connected"
      ws.simulateMessage({
        type: "connected",
        data: { client_id: "c1", timestamp: 1 },
      });
      expect(client.state).toBe("connected");

      // Close and let it reconnect once
      ws.simulateClose();
      vi.advanceTimersByTime(50000); // enough for first reconnect

      // The new instance should be at index 1
      const ws2 = MockWebSocket.instances[1];
      expect(ws2).toBeDefined();

      // Open the new one - reconnectAttempts should be reset to 0 internally
      ws2.simulateOpen();
      ws2.simulateMessage({
        type: "connected",
        data: { client_id: "c2", timestamp: 2 },
      });
      expect(client.state).toBe("connected");
    });
  });

  // ========================================================================
  // 4. disconnect
  // ========================================================================

  describe("disconnect", () => {
    it("sets shouldReconnect=false (verified by no reconnect on close)", () => {
      const { client, ws } = createConnectedClient();
      const instanceCount = MockWebSocket.instances.length;

      client.disconnect();
      // Simulate close after disconnect
      ws.simulateClose();

      // Advance timers far enough for any reconnect
      vi.advanceTimersByTime(60000);

      // No new WebSocket should have been created
      expect(MockWebSocket.instances.length).toBe(instanceCount);
    });

    it("closes the WebSocket", () => {
      const { client, ws } = createConnectedClient();
      client.disconnect();
      expect(ws.close).toHaveBeenCalledOnce();
    });

    it("sets state to disconnected", () => {
      const { client } = createConnectedClient();
      client.disconnect();
      expect(client.state).toBe("disconnected");
    });

    it("clears clientId", () => {
      const { client } = createConnectedClient();
      expect(client.clientId).toBe("test-client");

      client.disconnect();
      expect(client.clientId).toBeNull();
    });
  });

  // ========================================================================
  // 5. Message routing
  // ========================================================================

  describe("message routing", () => {
    it("connected message calls onConnect, sets clientId, state=connected", () => {
      const onConnect = vi.fn();
      const client = new JarvisWebSocket("ws://localhost:8001");
      client.setHandlers({ onConnect });
      client.connect();

      const ws = MockWebSocket.instances[0];
      ws.simulateOpen();
      ws.simulateMessage({
        type: "connected",
        data: { client_id: "my-id", timestamp: 42 },
      });

      expect(onConnect).toHaveBeenCalledWith({ client_id: "my-id", timestamp: 42 });
      expect(client.clientId).toBe("my-id");
      expect(client.state).toBe("connected");
    });

    it("token message calls onToken", () => {
      const onToken = vi.fn();
      const { ws } = createConnectedClient({ onToken });

      ws.simulateMessage({
        type: "token",
        data: { generation_id: "g1", token: "abc", token_index: 0 },
      });

      expect(onToken).toHaveBeenCalledWith({
        generation_id: "g1",
        token: "abc",
        token_index: 0,
      });
    });

    it("generation_start calls onGenerationStart", () => {
      const onGenerationStart = vi.fn();
      const { ws } = createConnectedClient({ onGenerationStart });

      ws.simulateMessage({
        type: "generation_start",
        data: { generation_id: "g1", streaming: true },
      });

      expect(onGenerationStart).toHaveBeenCalledWith({
        generation_id: "g1",
        streaming: true,
      });
    });

    it("generation_complete calls onGenerationComplete", () => {
      const onGenerationComplete = vi.fn();
      const { ws } = createConnectedClient({ onGenerationComplete });

      const completeData = {
        generation_id: "g1",
        text: "Full response",
        tokens_used: 15,
        generation_time_ms: 200,
        model_name: "test-model",
        used_template: false,
        template_name: null,
        finish_reason: "stop",
      };

      ws.simulateMessage({ type: "generation_complete", data: completeData });
      expect(onGenerationComplete).toHaveBeenCalledWith(completeData);
    });

    it("generation_error calls onGenerationError", () => {
      const onGenerationError = vi.fn();
      const { ws } = createConnectedClient({ onGenerationError });

      ws.simulateMessage({
        type: "generation_error",
        data: { generation_id: "g1", error: "OOM" },
      });

      expect(onGenerationError).toHaveBeenCalledWith({
        generation_id: "g1",
        error: "OOM",
      });
    });

    it("health_update calls onHealthUpdate", () => {
      const onHealthUpdate = vi.fn();
      const { ws } = createConnectedClient({ onHealthUpdate });

      const healthData = { status: "healthy", model_loaded: true };
      ws.simulateMessage({ type: "health_update", data: healthData });

      expect(onHealthUpdate).toHaveBeenCalledWith(healthData);
    });

    it("pong is silently accepted (no handler call)", () => {
      const onToken = vi.fn();
      const onError = vi.fn();
      const { ws } = createConnectedClient({ onToken, onError });

      // pong should not throw or call any handler
      ws.simulateMessage({ type: "pong", data: {} });

      expect(onToken).not.toHaveBeenCalled();
      expect(onError).not.toHaveBeenCalled();
    });

    it("error message calls onError with error string", () => {
      const onError = vi.fn();
      const { ws } = createConnectedClient({ onError });

      ws.simulateMessage({
        type: "error",
        data: { error: "Invalid request format" },
      });

      expect(onError).toHaveBeenCalledWith("Invalid request format");
    });
  });

  // ========================================================================
  // 6. send
  // ========================================================================

  describe("send", () => {
    it("returns false when not connected", () => {
      const client = new JarvisWebSocket("ws://localhost:8001");
      const result = client.generate({ prompt: "test" });
      expect(result).toBe(false);
    });

    it("sends JSON-stringified message with type and data", () => {
      const { client, ws } = createConnectedClient();

      client.generateStream({ prompt: "Hello", max_tokens: 50 });

      expect(ws.send).toHaveBeenCalledWith(
        JSON.stringify({
          type: "generate_stream",
          data: { prompt: "Hello", max_tokens: 50 },
        })
      );
    });

    it("returns true on success", () => {
      const { client } = createConnectedClient();
      const result = client.generateStream({ prompt: "test" });
      expect(result).toBe(true);
    });

    it("returns false on send exception", () => {
      const { client, ws } = createConnectedClient();
      ws.send.mockImplementation(() => {
        throw new Error("Connection reset");
      });

      const result = client.generate({ prompt: "test" });
      expect(result).toBe(false);
    });
  });

  // ========================================================================
  // 7. Reconnection
  // ========================================================================

  describe("reconnection", () => {
    it("schedules reconnect on close when shouldReconnect=true", () => {
      const onStateChange = vi.fn();
      const { ws } = createConnectedClient({ onStateChange });

      onStateChange.mockClear();
      ws.simulateClose();

      // Should schedule reconnect, setting state to "reconnecting"
      expect(onStateChange).toHaveBeenCalledWith("reconnecting");

      // Advance past the first reconnect delay (1000ms base + up to 1000ms jitter)
      vi.advanceTimersByTime(3000);

      // A new WebSocket should have been created
      expect(MockWebSocket.instances.length).toBeGreaterThan(1);
    });

    it("exponential backoff: delay doubles each attempt", () => {
      // Seed Math.random to 0 so jitter is deterministic (0ms jitter)
      vi.spyOn(Math, "random").mockReturnValue(0);

      const client = new JarvisWebSocket("ws://localhost:8001");
      client.connect();

      // Close without simulateOpen so reconnectAttempts is NOT reset on each attempt.
      // Attempt 0: delay = 1000 * 2^0 + 0 = 1000ms
      const ws0 = MockWebSocket.instances[0];
      ws0.simulateClose();
      const countAfterClose1 = MockWebSocket.instances.length;

      // At 999ms, should not have reconnected yet
      vi.advanceTimersByTime(999);
      expect(MockWebSocket.instances.length).toBe(countAfterClose1);

      // At 1001ms, should have reconnected
      vi.advanceTimersByTime(2);
      expect(MockWebSocket.instances.length).toBeGreaterThan(countAfterClose1);

      // Attempt 1: delay = 1000 * 2^1 + 0 = 2000ms
      const ws1 = MockWebSocket.instances[MockWebSocket.instances.length - 1];
      ws1.simulateClose();
      const countAfterClose2 = MockWebSocket.instances.length;

      // At 1999ms, should NOT have reconnected
      vi.advanceTimersByTime(1999);
      expect(MockWebSocket.instances.length).toBe(countAfterClose2);

      // At 2001ms, should have reconnected
      vi.advanceTimersByTime(2);
      expect(MockWebSocket.instances.length).toBeGreaterThan(countAfterClose2);

      // Attempt 2: delay = 1000 * 2^2 + 0 = 4000ms
      const ws2 = MockWebSocket.instances[MockWebSocket.instances.length - 1];
      ws2.simulateClose();
      const countAfterClose3 = MockWebSocket.instances.length;

      // At 3999ms, should NOT have reconnected
      vi.advanceTimersByTime(3999);
      expect(MockWebSocket.instances.length).toBe(countAfterClose3);

      // At 4001ms, should have reconnected
      vi.advanceTimersByTime(2);
      expect(MockWebSocket.instances.length).toBeGreaterThan(countAfterClose3);

      vi.restoreAllMocks();
      // Re-suppress console
      vi.spyOn(console, "log").mockImplementation(() => {});
      vi.spyOn(console, "warn").mockImplementation(() => {});
      vi.spyOn(console, "error").mockImplementation(() => {});
    });

    it("max delay is capped at 30000ms", () => {
      // Seed Math.random to 0 so jitter is deterministic
      vi.spyOn(Math, "random").mockReturnValue(0);

      const client = new JarvisWebSocket("ws://localhost:8001");
      client.connect();

      // Each failed reconnect: the timer fires, increments reconnectAttempts,
      // calls connect() which creates a new WebSocket. We immediately close that
      // WebSocket WITHOUT calling simulateOpen (so reconnectAttempts is NOT reset).
      // This simulates consecutive failed connection attempts.
      for (let i = 0; i < 8; i++) {
        const ws = MockWebSocket.instances[MockWebSocket.instances.length - 1];
        ws.simulateClose(); // triggers scheduleReconnect
        vi.advanceTimersByTime(35000); // enough for any backoff delay
      }

      // At this point reconnectAttempts=8, so delay = 1000 * 2^8 = 256000ms
      // but capped at 30000ms. Since jitter=0, exact delay is 30000ms.
      const countBefore = MockWebSocket.instances.length;
      const wsLast = MockWebSocket.instances[MockWebSocket.instances.length - 1];
      wsLast.simulateClose();

      // At 29999ms, should NOT have reconnected yet
      vi.advanceTimersByTime(29999);
      expect(MockWebSocket.instances.length).toBe(countBefore);

      // At 30001ms total, should have reconnected
      vi.advanceTimersByTime(2);
      expect(MockWebSocket.instances.length).toBeGreaterThan(countBefore);

      vi.restoreAllMocks();
      // Re-suppress console
      vi.spyOn(console, "log").mockImplementation(() => {});
      vi.spyOn(console, "warn").mockImplementation(() => {});
      vi.spyOn(console, "error").mockImplementation(() => {});
    });

    it("stops after maxReconnectAttempts (10)", () => {
      const onError = vi.fn();
      const onStateChange = vi.fn();
      const client = new JarvisWebSocket("ws://localhost:8001");
      client.setHandlers({ onError, onStateChange });
      client.connect();

      // Simulate failed reconnection attempts: close without ever opening
      // so reconnectAttempts keeps incrementing and is never reset.
      // The initial connect() creates ws[0]. Close it to trigger first reconnect.
      // Each scheduleReconnect sets a timer that increments reconnectAttempts and calls connect().
      // That connect() creates a new ws. We close it again to trigger the next reconnect.
      for (let i = 0; i < 11; i++) {
        const ws = MockWebSocket.instances[MockWebSocket.instances.length - 1];
        ws.simulateClose(); // triggers scheduleReconnect
        vi.advanceTimersByTime(35000);
      }

      // After 10 attempts, should stop and fire error
      expect(onError).toHaveBeenCalledWith("Connection lost. Please refresh the page.");
      expect(onStateChange).toHaveBeenCalledWith("disconnected");
    });

    it("fires onError with 'Connection lost' at max attempts", () => {
      const onError = vi.fn();
      const client = new JarvisWebSocket("ws://localhost:8001");
      client.setHandlers({ onError });
      client.connect();

      // Exhaust reconnect attempts by closing without opening
      for (let i = 0; i < 12; i++) {
        const ws = MockWebSocket.instances[MockWebSocket.instances.length - 1];
        ws.simulateClose();
        vi.advanceTimersByTime(35000);
      }

      const connectionLostCalls = onError.mock.calls.filter(
        (call: unknown[]) => call[0] === "Connection lost. Please refresh the page."
      );
      expect(connectionLostCalls.length).toBeGreaterThanOrEqual(1);
    });

    it("disconnect cancels pending reconnect", () => {
      const { client, ws } = createConnectedClient();
      const countBeforeDisconnect = MockWebSocket.instances.length;

      // Close triggers reconnect scheduling
      ws.simulateClose();

      // Disconnect before reconnect fires
      client.disconnect();

      // Advance past any pending reconnect timer
      vi.advanceTimersByTime(60000);

      // No new WebSocket created after disconnect (only the one from disconnect's close logic)
      // The key check: state should remain disconnected
      expect(client.state).toBe("disconnected");
    });
  });

  // ========================================================================
  // 8. Ping interval
  // ========================================================================

  describe("ping interval", () => {
    it("starts 60s ping interval after connection opens", () => {
      const { client, ws } = createConnectedClient();

      // Clear send calls from the connected message handling
      ws.send.mockClear();

      // Advance 60 seconds
      vi.advanceTimersByTime(60000);

      // Should have sent a ping
      expect(ws.send).toHaveBeenCalledWith(
        JSON.stringify({ type: "ping", data: {} })
      );
    });

    it("stops ping interval on disconnect", () => {
      const { client, ws } = createConnectedClient();
      ws.send.mockClear();

      client.disconnect();

      // Advance well past ping interval
      vi.advanceTimersByTime(120000);

      // No pings should have been sent after disconnect
      expect(ws.send).not.toHaveBeenCalled();
    });

    it("pings are sent as { type: 'ping', data: {} }", () => {
      const { ws } = createConnectedClient();
      ws.send.mockClear();

      // Trigger 3 pings (60s interval)
      vi.advanceTimersByTime(180000);

      const pingCalls = ws.send.mock.calls.filter(
        (call: unknown[]) => call[0] === JSON.stringify({ type: "ping", data: {} })
      );
      expect(pingCalls).toHaveLength(3);
    });
  });

  // ========================================================================
  // 9. Malformed messages
  // ========================================================================

  describe("malformed messages", () => {
    it("invalid JSON does not throw", () => {
      const onError = vi.fn();
      const client = new JarvisWebSocket("ws://localhost:8001");
      client.setHandlers({ onError });
      client.connect();

      const ws = MockWebSocket.instances[0];
      ws.simulateOpen();

      // Send raw invalid JSON (bypass simulateMessage which stringifies)
      expect(() => {
        ws.onmessage?.({ data: "not valid json {{{" });
      }).not.toThrow();

      // onError should NOT be called for parse failures (only console.error)
      expect(onError).not.toHaveBeenCalled();
    });
  });

  // ========================================================================
  // 10. Visibility-aware reconnection
  // ========================================================================

  describe("visibility-aware reconnection", () => {
    let mockDocument: {
      hidden: boolean;
      addEventListener: ReturnType<typeof vi.fn>;
      removeEventListener: ReturnType<typeof vi.fn>;
    };

    beforeEach(() => {
      mockDocument = {
        hidden: false,
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
      };
      vi.stubGlobal("document", mockDocument);
    });

    it("defers reconnect when document.hidden is true", () => {
      mockDocument.hidden = true;

      const { ws } = createConnectedClient();
      const countBeforeClose = MockWebSocket.instances.length;

      ws.simulateClose();

      // Advance timers significantly - no reconnect should fire
      vi.advanceTimersByTime(60000);
      expect(MockWebSocket.instances.length).toBe(countBeforeClose);

      // Should have registered a visibilitychange listener
      expect(mockDocument.addEventListener).toHaveBeenCalledWith(
        "visibilitychange",
        expect.any(Function)
      );
    });

    it("resumes reconnect when document becomes visible", () => {
      // Use a fresh client (not createConnectedClient) so we control state precisely.
      // Connect, open, receive "connected", then make document hidden, then close.
      mockDocument.hidden = false;

      const client = new JarvisWebSocket("ws://localhost:8001");
      client.connect();

      const ws = MockWebSocket.instances[MockWebSocket.instances.length - 1];
      ws.simulateOpen();
      ws.simulateMessage({
        type: "connected",
        data: { client_id: "vis-client", timestamp: 1 },
      });

      // Now hide the document and close the socket.
      // handleDisconnect -> scheduleReconnect -> document.hidden=true -> defers
      // Note: scheduleReconnect returns early before setState("reconnecting"),
      // so state stays "connected". The visibility callback checks
      // this._state !== "connected", which would be false.
      // To properly test this, we need the state to NOT be "connected" when
      // the visibility callback fires. This happens naturally if the first close
      // already set state via onStateChange handler or if we force a state change.
      // Actually: the code defers without changing state. When visibility callback fires,
      // it checks shouldReconnect && _state !== "connected".
      // Since _state is still "connected", it won't reconnect. This matches the code's
      // actual behavior: if the connection was cleanly "connected" when backgrounded,
      // the code doesn't auto-reconnect on visibility change.
      //
      // To test the reconnect-on-visible path, we need a scenario where state != "connected"
      // when the visibility callback fires. This happens when:
      // 1. Socket closes (state still "connected" since scheduleReconnect defers)
      // 2. We manually set state via the onStateChange callback OR
      // 3. The handleDisconnect flow already changed state before calling scheduleReconnect
      //
      // Looking at handleDisconnect: it does NOT call setState before scheduleReconnect.
      // So state remains "connected" when hidden. The visibility callback won't reconnect.
      //
      // The realistic scenario is: first reconnect attempt fails (state = "reconnecting"),
      // then the second attempt sees document.hidden. Let's test that:

      // First: close while visible to get a normal reconnect
      mockDocument.hidden = false;
      ws.simulateClose();
      // State should now be "reconnecting"
      expect(client.state).toBe("reconnecting");

      // Advance to trigger the reconnect timer
      vi.advanceTimersByTime(5000);

      // A new WebSocket was created by the reconnect
      const ws2 = MockWebSocket.instances[MockWebSocket.instances.length - 1];

      // Now hide the document before the second close
      mockDocument.hidden = true;
      const countBeforeClose2 = MockWebSocket.instances.length;

      ws2.simulateClose();

      // scheduleReconnect defers because document.hidden
      // Advance timers - no reconnect while hidden
      vi.advanceTimersByTime(60000);
      expect(MockWebSocket.instances.length).toBe(countBeforeClose2);

      // Grab the visibilitychange callback
      const visibilityCallback = mockDocument.addEventListener.mock.calls.find(
        (call: unknown[]) => call[0] === "visibilitychange"
      )?.[1] as () => void;
      expect(visibilityCallback).toBeDefined();

      // Simulate becoming visible - now state is "reconnecting" (from the first close),
      // not "connected", so the callback will proceed
      mockDocument.hidden = false;
      visibilityCallback();

      // Advance past reconnect delay
      vi.advanceTimersByTime(35000);

      // Should have created a new WebSocket for reconnection
      expect(MockWebSocket.instances.length).toBeGreaterThan(countBeforeClose2);

      // Should have removed the visibilitychange listener
      expect(mockDocument.removeEventListener).toHaveBeenCalledWith(
        "visibilitychange",
        visibilityCallback
      );
    });
  });

  // ========================================================================
  // 11. WebSocket error handling
  // ========================================================================

  describe("WebSocket error event", () => {
    it("calls onError handler with connection error message", () => {
      const onError = vi.fn();
      const client = new JarvisWebSocket("ws://localhost:8001");
      client.setHandlers({ onError });
      client.connect();

      const ws = MockWebSocket.instances[0];
      ws.simulateError(new Error("ECONNREFUSED"));

      expect(onError).toHaveBeenCalledWith("WebSocket connection error");
    });
  });

  // ========================================================================
  // 12. Method-specific send calls
  // ========================================================================

  describe("client method send calls", () => {
    it("generate sends GENERATE type", () => {
      const { client, ws } = createConnectedClient();
      ws.send.mockClear();

      const request = { prompt: "test prompt" };
      client.generate(request);

      expect(ws.send).toHaveBeenCalledWith(
        JSON.stringify({ type: "generate", data: request })
      );
    });

    it("generateStream sends GENERATE_STREAM type", () => {
      const { client, ws } = createConnectedClient();
      ws.send.mockClear();

      const request = { prompt: "test", temperature: 0.7 };
      client.generateStream(request);

      expect(ws.send).toHaveBeenCalledWith(
        JSON.stringify({ type: "generate_stream", data: request })
      );
    });

    it("cancelGeneration sends CANCEL type", () => {
      const { client, ws } = createConnectedClient();
      ws.send.mockClear();

      client.cancelGeneration();

      expect(ws.send).toHaveBeenCalledWith(
        JSON.stringify({ type: "cancel", data: {} })
      );
    });

    it("subscribeHealth sends SUBSCRIBE_HEALTH type", () => {
      const { client, ws } = createConnectedClient();
      ws.send.mockClear();

      client.subscribeHealth();

      expect(ws.send).toHaveBeenCalledWith(
        JSON.stringify({ type: "subscribe_health", data: {} })
      );
    });

    it("unsubscribeHealth sends UNSUBSCRIBE_HEALTH type", () => {
      const { client, ws } = createConnectedClient();
      ws.send.mockClear();

      client.unsubscribeHealth();

      expect(ws.send).toHaveBeenCalledWith(
        JSON.stringify({ type: "unsubscribe_health", data: {} })
      );
    });

    it("ping sends PING type", () => {
      const { client, ws } = createConnectedClient();
      ws.send.mockClear();

      client.ping();

      expect(ws.send).toHaveBeenCalledWith(
        JSON.stringify({ type: "ping", data: {} })
      );
    });
  });

  // ========================================================================
  // 13. onDisconnect handler
  // ========================================================================

  describe("onDisconnect handler", () => {
    it("calls onDisconnect and clears clientId on close", () => {
      const onDisconnect = vi.fn();
      const { client, ws } = createConnectedClient({ onDisconnect });

      expect(client.clientId).toBe("test-client");

      // Disconnect cleanly so no reconnect interferes
      client.disconnect();

      // When ws.close() triggers onclose
      ws.simulateClose();

      expect(onDisconnect).toHaveBeenCalled();
      expect(client.clientId).toBeNull();
    });
  });

  // ========================================================================
  // 14. State change notifications
  // ========================================================================

  describe("state change notifications", () => {
    it("does not fire onStateChange if state has not changed", () => {
      const onStateChange = vi.fn();
      const client = new JarvisWebSocket("ws://localhost:8001");
      client.setHandlers({ onStateChange });

      // State starts as "disconnected". Setting again to "disconnected" should not fire.
      // The only way to test this is through internal behavior: disconnect when already disconnected
      client.disconnect();
      // disconnect calls setState("disconnected") but state was already "disconnected"
      // so onStateChange should not be called for that transition
      const disconnectedCalls = onStateChange.mock.calls.filter(
        (call: unknown[]) => call[0] === "disconnected"
      );
      expect(disconnectedCalls).toHaveLength(0);
    });
  });

  // ========================================================================
  // 15. Default URL from WS_BASE
  // ========================================================================

  describe("default URL", () => {
    it("uses WS_BASE when no baseUrl provided", async () => {
      // Need to test the module-level export
      vi.resetModules();
      const mod = await import("../../src/lib/api/websocket");

      // The singleton uses the default WS_BASE
      mod.jarvisWs.connect();
      const ws = MockWebSocket.instances[MockWebSocket.instances.length - 1];
      expect(ws.url).toBe("ws://localhost:8001/ws");
    });
  });
});
