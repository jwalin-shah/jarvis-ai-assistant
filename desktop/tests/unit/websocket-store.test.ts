/**
 * Unit tests for WebSocket Svelte store (stores/websocket.ts)
 *
 * Tests connection state management, streaming generation lifecycle,
 * derived stores, and store action functions.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { get } from "svelte/store";

// ---------------------------------------------------------------------------
// Mocks - must be declared before imports that use them
// ---------------------------------------------------------------------------

vi.mock("../../src/lib/api/websocket", () => {
  return {
    jarvisWs: {
      setHandlers: vi.fn(),
      connect: vi.fn(),
      disconnect: vi.fn(),
      generateStream: vi.fn(() => true),
      generate: vi.fn(() => true),
      cancelGeneration: vi.fn(() => true),
      subscribeHealth: vi.fn(() => true),
      unsubscribeHealth: vi.fn(() => true),
    },
    WS_BASE: "ws://localhost:8001",
  };
});

// Type aliases for the store module exports
type StoreModule = typeof import("../../src/lib/stores/websocket");
type ApiModule = typeof import("../../src/lib/api/websocket");

// ==========================================================================
// WebSocket Store
// ==========================================================================

describe("websocket store", () => {
  // Store references re-imported each test
  let websocketStore: StoreModule["websocketStore"];
  let connectionState: StoreModule["connectionState"];
  let isConnected: StoreModule["isConnected"];
  let isStreaming: StoreModule["isStreaming"];
  let streamingText: StoreModule["streamingText"];
  let streamingError: StoreModule["streamingError"];
  let initializeWebSocket: StoreModule["initializeWebSocket"];
  let disconnectWebSocket: StoreModule["disconnectWebSocket"];
  let generateStream: StoreModule["generateStream"];
  let generate: StoreModule["generate"];
  let cancelGeneration: StoreModule["cancelGeneration"];
  let subscribeToHealth: StoreModule["subscribeToHealth"];
  let unsubscribeFromHealth: StoreModule["unsubscribeFromHealth"];
  let clearError: StoreModule["clearError"];
  let resetStreamingState: StoreModule["resetStreamingState"];
  let getWebSocketState: StoreModule["getWebSocketState"];

  // Mock reference
  let mockJarvisWs: ApiModule["jarvisWs"] & Record<string, ReturnType<typeof vi.fn>>;

  // Captured handlers from setHandlers call
  let capturedHandlers: Record<string, (...args: unknown[]) => void>;

  beforeEach(async () => {
    vi.useFakeTimers();
    vi.resetModules();

    const apiMod = await import("../../src/lib/api/websocket");
    mockJarvisWs = apiMod.jarvisWs as typeof mockJarvisWs;

    // Reset all mock call history
    Object.values(mockJarvisWs).forEach((fn) => {
      if (typeof fn === "function" && "mockClear" in fn) {
        (fn as ReturnType<typeof vi.fn>).mockClear();
      }
    });

    // Re-set default return values after clear
    (mockJarvisWs.generateStream as ReturnType<typeof vi.fn>).mockReturnValue(true);
    (mockJarvisWs.generate as ReturnType<typeof vi.fn>).mockReturnValue(true);
    (mockJarvisWs.cancelGeneration as ReturnType<typeof vi.fn>).mockReturnValue(true);
    (mockJarvisWs.subscribeHealth as ReturnType<typeof vi.fn>).mockReturnValue(true);
    (mockJarvisWs.unsubscribeHealth as ReturnType<typeof vi.fn>).mockReturnValue(true);

    const mod = await import("../../src/lib/stores/websocket");
    websocketStore = mod.websocketStore;
    connectionState = mod.connectionState;
    isConnected = mod.isConnected;
    isStreaming = mod.isStreaming;
    streamingText = mod.streamingText;
    streamingError = mod.streamingError;
    initializeWebSocket = mod.initializeWebSocket;
    disconnectWebSocket = mod.disconnectWebSocket;
    generateStream = mod.generateStream;
    generate = mod.generate;
    cancelGeneration = mod.cancelGeneration;
    subscribeToHealth = mod.subscribeToHealth;
    unsubscribeFromHealth = mod.unsubscribeFromHealth;
    clearError = mod.clearError;
    resetStreamingState = mod.resetStreamingState;
    getWebSocketState = mod.getWebSocketState;

    // Capture handlers on setHandlers mock
    capturedHandlers = {};
    (mockJarvisWs.setHandlers as ReturnType<typeof vi.fn>).mockImplementation(
      (handlers: Record<string, (...args: unknown[]) => void>) => {
        capturedHandlers = handlers;
      }
    );
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  // Helper: initialize and return the captured handlers
  function initialize(): Record<string, (...args: unknown[]) => void> {
    initializeWebSocket();
    return capturedHandlers;
  }

  // ========================================================================
  // 1. Initial state
  // ========================================================================

  describe("initial state", () => {
    it("has correct initial values", () => {
      const state = get(websocketStore);
      expect(state.connectionState).toBe("disconnected");
      expect(state.clientId).toBeNull();
      expect(state.connectedAt).toBeNull();
      expect(state.error).toBeNull();
      expect(state.streaming.isStreaming).toBe(false);
      expect(state.streaming.generationId).toBeNull();
      expect(state.streaming.tokens).toEqual([]);
      expect(state.streaming.fullText).toBe("");
      expect(state.streaming.error).toBeNull();
      expect(state.streaming.startTime).toBeNull();
      expect(state.streaming.endTime).toBeNull();
    });

    it("derived isConnected is false", () => {
      expect(get(isConnected)).toBe(false);
    });

    it("derived stores reflect initial streaming state", () => {
      expect(get(isStreaming)).toBe(false);
      expect(get(streamingText)).toBe("");
      expect(get(streamingError)).toBeNull();
    });
  });

  // ========================================================================
  // 2. initializeWebSocket
  // ========================================================================

  describe("initializeWebSocket", () => {
    it("calls jarvisWs.setHandlers with correct handler keys", () => {
      initializeWebSocket();

      expect(mockJarvisWs.setHandlers).toHaveBeenCalledOnce();

      const handlersArg = (mockJarvisWs.setHandlers as ReturnType<typeof vi.fn>).mock.calls[0][0];
      expect(handlersArg).toHaveProperty("onConnect");
      expect(handlersArg).toHaveProperty("onDisconnect");
      expect(handlersArg).toHaveProperty("onStateChange");
      expect(handlersArg).toHaveProperty("onToken");
      expect(handlersArg).toHaveProperty("onGenerationStart");
      expect(handlersArg).toHaveProperty("onGenerationComplete");
      expect(handlersArg).toHaveProperty("onGenerationError");
      expect(handlersArg).toHaveProperty("onHealthUpdate");
      expect(handlersArg).toHaveProperty("onError");
    });

    it("calls jarvisWs.connect", () => {
      initializeWebSocket();
      expect(mockJarvisWs.connect).toHaveBeenCalledOnce();
    });
  });

  // ========================================================================
  // 3. Connection events via handlers
  // ========================================================================

  describe("connection events via handlers", () => {
    it("onConnect sets connectionState, clientId, and connectedAt", () => {
      const handlers = initialize();

      handlers.onConnect({ client_id: "abc-123", timestamp: 1700000000 });

      const state = get(websocketStore);
      expect(state.connectionState).toBe("connected");
      expect(state.clientId).toBe("abc-123");
      expect(state.connectedAt).toBe(1700000000);
      expect(state.error).toBeNull();
    });

    it("onDisconnect clears clientId and connectedAt", () => {
      const handlers = initialize();

      // First connect
      handlers.onConnect({ client_id: "abc-123", timestamp: 1700000000 });
      expect(get(websocketStore).clientId).toBe("abc-123");

      // Then disconnect
      handlers.onDisconnect();

      const state = get(websocketStore);
      expect(state.clientId).toBeNull();
      expect(state.connectedAt).toBeNull();
    });

    it("onStateChange updates connectionState", () => {
      const handlers = initialize();

      handlers.onStateChange("reconnecting");
      expect(get(websocketStore).connectionState).toBe("reconnecting");
      expect(get(connectionState)).toBe("reconnecting");

      handlers.onStateChange("connecting");
      expect(get(websocketStore).connectionState).toBe("connecting");
    });

    it("onError sets error string", () => {
      const handlers = initialize();

      handlers.onError("Connection refused");

      expect(get(websocketStore).error).toBe("Connection refused");
    });
  });

  // ========================================================================
  // 4. Streaming generation lifecycle
  // ========================================================================

  describe("streaming generation lifecycle", () => {
    it("onGenerationStart sets isStreaming=true, generationId, clears tokens/fullText/error, sets startTime", () => {
      const handlers = initialize();

      const now = Date.now();
      handlers.onGenerationStart({ generation_id: "gen-1", streaming: true });

      const state = get(websocketStore);
      expect(state.streaming.isStreaming).toBe(true);
      expect(state.streaming.generationId).toBe("gen-1");
      expect(state.streaming.tokens).toEqual([]);
      expect(state.streaming.fullText).toBe("");
      expect(state.streaming.error).toBeNull();
      expect(state.streaming.startTime).toBeGreaterThanOrEqual(now);
      expect(state.streaming.endTime).toBeNull();
    });

    it("onToken accumulates a single token", () => {
      const handlers = initialize();

      handlers.onGenerationStart({ generation_id: "gen-1", streaming: true });
      handlers.onToken({ generation_id: "gen-1", token: "Hello", token_index: 0 });

      const state = get(websocketStore);
      expect(state.streaming.tokens).toEqual(["Hello"]);
      expect(state.streaming.fullText).toBe("Hello");
    });

    it("multiple tokens accumulate correctly", () => {
      const handlers = initialize();

      handlers.onGenerationStart({ generation_id: "gen-1", streaming: true });
      handlers.onToken({ generation_id: "gen-1", token: "Hello", token_index: 0 });
      handlers.onToken({ generation_id: "gen-1", token: " ", token_index: 1 });
      handlers.onToken({ generation_id: "gen-1", token: "world", token_index: 2 });

      const state = get(websocketStore);
      expect(state.streaming.tokens).toEqual(["Hello", " ", "world"]);
      expect(state.streaming.fullText).toBe("Hello world");
      expect(get(streamingText)).toBe("Hello world");
    });

    it("onGenerationComplete sets isStreaming=false, final text, and endTime", () => {
      const handlers = initialize();

      handlers.onGenerationStart({ generation_id: "gen-1", streaming: true });
      handlers.onToken({ generation_id: "gen-1", token: "Hi", token_index: 0 });

      const beforeComplete = Date.now();
      handlers.onGenerationComplete({
        generation_id: "gen-1",
        text: "Hi there, complete response",
        tokens_used: 10,
        generation_time_ms: 500,
        model_name: "test-model",
        used_template: false,
        template_name: null,
        finish_reason: "stop",
      });

      const state = get(websocketStore);
      expect(state.streaming.isStreaming).toBe(false);
      expect(state.streaming.fullText).toBe("Hi there, complete response");
      expect(state.streaming.endTime).toBeGreaterThanOrEqual(beforeComplete);
      expect(get(isStreaming)).toBe(false);
    });

    it("onGenerationComplete ignores mismatched generationId", () => {
      const handlers = initialize();

      handlers.onGenerationStart({ generation_id: "gen-1", streaming: true });
      handlers.onToken({ generation_id: "gen-1", token: "Hello", token_index: 0 });

      // Complete event with different generation_id
      handlers.onGenerationComplete({
        generation_id: "gen-WRONG",
        text: "Wrong response",
        tokens_used: 5,
        generation_time_ms: 100,
        model_name: "test",
        used_template: false,
        template_name: null,
        finish_reason: "stop",
      });

      const state = get(websocketStore);
      // Should still be streaming with original text
      expect(state.streaming.isStreaming).toBe(true);
      expect(state.streaming.fullText).toBe("Hello");
      expect(state.streaming.endTime).toBeNull();
    });

    it("onGenerationError sets isStreaming=false, error message, and endTime", () => {
      const handlers = initialize();

      handlers.onGenerationStart({ generation_id: "gen-1", streaming: true });

      const beforeError = Date.now();
      handlers.onGenerationError({
        generation_id: "gen-1",
        error: "Model crashed",
      });

      const state = get(websocketStore);
      expect(state.streaming.isStreaming).toBe(false);
      expect(state.streaming.error).toBe("Model crashed");
      expect(state.streaming.endTime).toBeGreaterThanOrEqual(beforeError);
      expect(get(streamingError)).toBe("Model crashed");
    });

    it("onGenerationError ignores mismatched generationId when store has an active generation", () => {
      const handlers = initialize();

      handlers.onGenerationStart({ generation_id: "gen-1", streaming: true });

      // Error for a different generation
      handlers.onGenerationError({
        generation_id: "gen-OTHER",
        error: "Wrong generation error",
      });

      const state = get(websocketStore);
      // Should still be streaming, no error
      expect(state.streaming.isStreaming).toBe(true);
      expect(state.streaming.error).toBeNull();
      expect(state.streaming.generationId).toBe("gen-1");
    });

    it("onToken ignores mismatched generationId", () => {
      const handlers = initialize();

      handlers.onGenerationStart({ generation_id: "gen-1", streaming: true });
      handlers.onToken({ generation_id: "gen-1", token: "Good", token_index: 0 });

      // Token for a different generation
      handlers.onToken({ generation_id: "gen-WRONG", token: " bad", token_index: 1 });

      const state = get(websocketStore);
      expect(state.streaming.tokens).toEqual(["Good"]);
      expect(state.streaming.fullText).toBe("Good");
    });
  });

  // ========================================================================
  // 5. generateStream
  // ========================================================================

  describe("generateStream", () => {
    it("resets streaming state before starting", () => {
      const handlers = initialize();

      // Set up some streaming state first
      handlers.onGenerationStart({ generation_id: "gen-old", streaming: true });
      handlers.onToken({ generation_id: "gen-old", token: "old text", token_index: 0 });
      expect(get(websocketStore).streaming.fullText).toBe("old text");

      const request = { prompt: "Hello" };
      generateStream(request);

      // After generateStream, streaming state should be reset
      const state = get(websocketStore);
      expect(state.streaming.isStreaming).toBe(false);
      expect(state.streaming.generationId).toBeNull();
      expect(state.streaming.tokens).toEqual([]);
      expect(state.streaming.fullText).toBe("");
      expect(state.streaming.error).toBeNull();
    });

    it("calls jarvisWs.generateStream with the request", () => {
      initialize();

      const request = { prompt: "Tell me a joke", max_tokens: 100 };
      generateStream(request);

      expect(mockJarvisWs.generateStream).toHaveBeenCalledWith(request);
    });

    it("returns the boolean from jarvisWs.generateStream", () => {
      initialize();

      expect(generateStream({ prompt: "test" })).toBe(true);

      (mockJarvisWs.generateStream as ReturnType<typeof vi.fn>).mockReturnValueOnce(false);
      expect(generateStream({ prompt: "test" })).toBe(false);
    });
  });

  // ========================================================================
  // 6. cancelGeneration
  // ========================================================================

  describe("cancelGeneration", () => {
    it("calls jarvisWs.cancelGeneration", () => {
      initialize();
      cancelGeneration();
      expect(mockJarvisWs.cancelGeneration).toHaveBeenCalledOnce();
    });

    it("sets isStreaming=false, error='Generation cancelled', and endTime", () => {
      const handlers = initialize();

      handlers.onGenerationStart({ generation_id: "gen-1", streaming: true });
      expect(get(websocketStore).streaming.isStreaming).toBe(true);

      const beforeCancel = Date.now();
      cancelGeneration();

      const state = get(websocketStore);
      expect(state.streaming.isStreaming).toBe(false);
      expect(state.streaming.error).toBe("Generation cancelled");
      expect(state.streaming.endTime).toBeGreaterThanOrEqual(beforeCancel);
    });
  });

  // ========================================================================
  // 7. clearError / resetStreamingState / getWebSocketState
  // ========================================================================

  describe("utility functions", () => {
    it("clearError sets error to null", () => {
      const handlers = initialize();

      handlers.onError("Something broke");
      expect(get(websocketStore).error).toBe("Something broke");

      clearError();
      expect(get(websocketStore).error).toBeNull();
    });

    it("resetStreamingState resets all streaming fields", () => {
      const handlers = initialize();

      // Build up streaming state
      handlers.onGenerationStart({ generation_id: "gen-1", streaming: true });
      handlers.onToken({ generation_id: "gen-1", token: "data", token_index: 0 });
      expect(get(websocketStore).streaming.isStreaming).toBe(true);

      resetStreamingState();

      const state = get(websocketStore);
      expect(state.streaming.isStreaming).toBe(false);
      expect(state.streaming.generationId).toBeNull();
      expect(state.streaming.tokens).toEqual([]);
      expect(state.streaming.fullText).toBe("");
      expect(state.streaming.error).toBeNull();
      expect(state.streaming.startTime).toBeNull();
      expect(state.streaming.endTime).toBeNull();
    });

    it("getWebSocketState returns current snapshot", () => {
      const handlers = initialize();

      handlers.onConnect({ client_id: "snap-1", timestamp: 9999 });
      handlers.onGenerationStart({ generation_id: "gen-snap", streaming: true });

      const snapshot = getWebSocketState();
      expect(snapshot.connectionState).toBe("connected");
      expect(snapshot.clientId).toBe("snap-1");
      expect(snapshot.streaming.isStreaming).toBe(true);
      expect(snapshot.streaming.generationId).toBe("gen-snap");
    });
  });

  // ========================================================================
  // 8. disconnectWebSocket
  // ========================================================================

  describe("disconnectWebSocket", () => {
    it("calls jarvisWs.disconnect", () => {
      initialize();
      disconnectWebSocket();
      expect(mockJarvisWs.disconnect).toHaveBeenCalledOnce();
    });
  });

  // ========================================================================
  // 9. generate (non-streaming)
  // ========================================================================

  describe("generate", () => {
    it("calls jarvisWs.generate with the request", () => {
      initialize();
      const request = { prompt: "Answer this" };
      generate(request);
      expect(mockJarvisWs.generate).toHaveBeenCalledWith(request);
    });

    it("returns the boolean from jarvisWs.generate", () => {
      initialize();
      expect(generate({ prompt: "test" })).toBe(true);

      (mockJarvisWs.generate as ReturnType<typeof vi.fn>).mockReturnValueOnce(false);
      expect(generate({ prompt: "test" })).toBe(false);
    });
  });

  // ========================================================================
  // 10. Health subscription
  // ========================================================================

  describe("health subscription", () => {
    it("subscribeToHealth calls jarvisWs.subscribeHealth and returns result", () => {
      initialize();
      expect(subscribeToHealth()).toBe(true);
      expect(mockJarvisWs.subscribeHealth).toHaveBeenCalledOnce();
    });

    it("unsubscribeFromHealth calls jarvisWs.unsubscribeHealth and returns result", () => {
      initialize();
      expect(unsubscribeFromHealth()).toBe(true);
      expect(mockJarvisWs.unsubscribeHealth).toHaveBeenCalledOnce();
    });
  });

  // ========================================================================
  // 11. Derived store reactivity
  // ========================================================================

  describe("derived stores", () => {
    it("isConnected becomes true when connectionState is connected", () => {
      const handlers = initialize();

      expect(get(isConnected)).toBe(false);
      handlers.onConnect({ client_id: "c1", timestamp: 1 });
      expect(get(isConnected)).toBe(true);
    });

    it("isStreaming tracks streaming.isStreaming", () => {
      const handlers = initialize();

      expect(get(isStreaming)).toBe(false);
      handlers.onGenerationStart({ generation_id: "g1", streaming: true });
      expect(get(isStreaming)).toBe(true);
      handlers.onGenerationComplete({
        generation_id: "g1",
        text: "done",
        tokens_used: 1,
        generation_time_ms: 10,
        model_name: "m",
        used_template: false,
        template_name: null,
        finish_reason: "stop",
      });
      expect(get(isStreaming)).toBe(false);
    });

    it("connectionState derived store tracks state changes", () => {
      const handlers = initialize();

      expect(get(connectionState)).toBe("disconnected");
      handlers.onStateChange("connecting");
      expect(get(connectionState)).toBe("connecting");
      handlers.onConnect({ client_id: "c1", timestamp: 1 });
      expect(get(connectionState)).toBe("connected");
    });
  });

  // ========================================================================
  // 12. onConnect clears previous error
  // ========================================================================

  describe("onConnect clears error", () => {
    it("clears any previous error on successful connect", () => {
      const handlers = initialize();

      handlers.onError("Previous error");
      expect(get(websocketStore).error).toBe("Previous error");

      handlers.onConnect({ client_id: "c1", timestamp: 1 });
      expect(get(websocketStore).error).toBeNull();
    });
  });
});
