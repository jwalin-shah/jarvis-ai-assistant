/**
 * JARVIS Socket Client
 *
 * TypeScript client for communication with the Python socket server.
 * Uses Tauri commands to bridge to Unix socket (in app) or WebSocket (in browser).
 */

// Check if running in Tauri context
const isTauri = typeof window !== "undefined" && "__TAURI__" in window;

// WebSocket configuration for browser mode
const WEBSOCKET_URL = "ws://127.0.0.1:8743";

// Request timeout configuration (ms)
const REQUEST_TIMEOUT = 30000; // 30s for normal requests
const STREAMING_TIMEOUT = 120000; // 120s for streaming requests
const STALE_REQUEST_CLEANUP_INTERVAL = 60000; // Clean up stale requests every 60s

// Request batching configuration
const BATCH_WINDOW_MS = 15; // Collect requests for 15ms before sending batch
const MAX_BATCH_SIZE = 10; // Maximum requests per batch

/**
 * Wrap a promise with a timeout.
 * Rejects with a timeout error if the promise doesn't resolve within the given ms.
 */
function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new Error(`${label} timed out after ${ms}ms`));
    }, ms);
    promise.then(
      (value) => { clearTimeout(timer); resolve(value); },
      (err) => { clearTimeout(timer); reject(err); },
    );
  });
}

// Dynamic imports for Tauri APIs - only available in Tauri context
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let invoke: ((cmd: string, args?: object) => Promise<any>) | null = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let listen: ((event: string, handler: (event: any) => void) => Promise<() => void>) | null = null;

/** Connection state */
export type ConnectionState = "disconnected" | "connecting" | "connected";

/** Event handler type */
type EventHandler<T = unknown> = (data: T) => void;

/** Streaming token event from Tauri */
export interface StreamTokenEvent {
  token: string;
  index: number;
  final_token: boolean;
  request_id: number;
}

/** New message notification from server */
export interface NewMessageEvent {
  message_id: number;
  chat_id: string;
  sender: string | null;
  text_preview: string | null;
  is_from_me: boolean;
}

/** Generate draft parameters */
export interface GenerateDraftParams {
  chat_id: string;
  instruction?: string;
  context_messages?: number;
}

/** Smart reply parameters */
export interface SmartReplyParams {
  last_message: string;
  num_suggestions?: number;
}

/** Semantic search parameters */
export interface SemanticSearchParams {
  query: string;
  limit?: number;
  threshold?: number;
  filters?: {
    chat_id?: string;
    sender?: string;
    after?: string;
    before?: string;
  };
}

/** Intent classification result */
export interface IntentResult {
  intent: string;
  confidence: number;
  requires_response: boolean;
}

/** Draft suggestion */
export interface DraftSuggestion {
  text: string;
  confidence: number;
}

/** Generate draft result */
export interface GenerateDraftResult {
  suggestions: DraftSuggestion[];
  context_used: {
    num_messages: number;
    participants: string[];
    last_message: string | null;
  };
  streamed?: boolean;
  tokens_generated?: number;
}

/** Summarize result */
export interface SummarizeResult {
  summary: string;
  key_points: string[];
  message_count: number;
  streamed?: boolean;
  tokens_generated?: number;
}

/**
 * JARVIS Socket Client
 *
 * Provides methods for communicating with the Python daemon.
 * Uses Unix socket (via Tauri) in app, WebSocket in browser.
 * Supports both request/response and real streaming patterns.
 */
class JarvisSocket {
  private state: ConnectionState = "disconnected";
  private eventHandlers: Map<string, Set<EventHandler>> = new Map();
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000; // ms

  // Tauri event listener cleanup functions
  private unlistenFns: (() => void)[] = [];

  // WebSocket connection for browser mode
  private ws: WebSocket | null = null;
  private wsRequestId = 0;
  private wsPendingRequests: Map<
    number,
    {
      resolve: (value: unknown) => void;
      reject: (error: Error) => void;
      onToken?: (token: string, index: number) => void;
      createdAt: number; // Timestamp for cleanup
      isStreaming: boolean; // Track streaming vs normal requests
    }
  > = new Map();
  private cleanupInterval: ReturnType<typeof setInterval> | null = null;

  // Active streaming requests
  private streamingRequests: Map<
    number,
    {
      tokens: string[];
      onToken?: (token: string, index: number) => void;
      onComplete: () => void;
    }
  > = new Map();

  // Request batching state (WebSocket mode only)
  private batchQueue: Array<{
    method: string;
    params: object;
    resolve: (value: unknown) => void;
    reject: (error: Error) => void;
  }> = [];
  private batchTimer: ReturnType<typeof setTimeout> | null = null;

  /**
   * Get current connection state
   */
  getState(): ConnectionState {
    return this.state;
  }

  /**
   * Initialize Tauri APIs
   */
  private async initTauriApis(): Promise<boolean> {
    if (!isTauri) return false;

    try {
      if (!invoke) {
        const coreModule = await import("@tauri-apps/api/core");
        invoke = coreModule.invoke;
      }
      if (!listen) {
        const eventModule = await import("@tauri-apps/api/event");
        listen = eventModule.listen;
      }
      return true;
    } catch (error) {
      console.warn("[JarvisSocket] Failed to import Tauri APIs:", error);
      return false;
    }
  }

  /**
   * Set up Tauri event listeners for streaming and push notifications
   */
  private async setupEventListeners(): Promise<void> {
    if (!listen) return;

    // Clean up existing listeners
    await this.cleanupEventListeners();

    // Listen for stream tokens
    const unlistenToken = await listen("jarvis:stream_token", (event: { payload: StreamTokenEvent }) => {
      this.handleStreamToken(event.payload);
    });
    this.unlistenFns.push(unlistenToken);

    // Listen for new message notifications
    const unlistenNewMsg = await listen("jarvis:new_message", (event: { payload: NewMessageEvent }) => {
      this.emit("new_message", event.payload);
    });
    this.unlistenFns.push(unlistenNewMsg);

    // Listen for connection events
    const unlistenConnected = await listen("jarvis:connected", () => {
      this.state = "connected";
      this.emit("connected", {});
    });
    this.unlistenFns.push(unlistenConnected);

    const unlistenDisconnected = await listen("jarvis:disconnected", () => {
      this.state = "disconnected";
      this.emit("disconnected", {});
      this.scheduleReconnect();
    });
    this.unlistenFns.push(unlistenDisconnected);
  }

  /**
   * Clean up Tauri event listeners
   */
  private async cleanupEventListeners(): Promise<void> {
    for (const unlisten of this.unlistenFns) {
      unlisten();
    }
    this.unlistenFns = [];
  }

  /**
   * Start periodic cleanup of stale pending requests (WebSocket mode)
   */
  private startStaleRequestCleanup(): void {
    if (this.cleanupInterval) return;

    this.cleanupInterval = setInterval(() => {
      const now = Date.now();
      for (const [requestId, pending] of this.wsPendingRequests) {
        const timeout = pending.isStreaming ? STREAMING_TIMEOUT : REQUEST_TIMEOUT;
        if (now - pending.createdAt > timeout) {
          console.warn(`[JarvisSocket] Cleaning up stale request ${requestId}`);
          pending.reject(new Error("Request timed out (cleanup)"));
          this.wsPendingRequests.delete(requestId);
        }
      }
    }, STALE_REQUEST_CLEANUP_INTERVAL);
  }

  /**
   * Stop stale request cleanup
   */
  private stopStaleRequestCleanup(): void {
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
      this.cleanupInterval = null;
    }
  }

  /**
   * Handle incoming stream token
   */
  private handleStreamToken(event: StreamTokenEvent): void {
    const request = this.streamingRequests.get(event.request_id);
    if (!request) return;

    // Accumulate token
    request.tokens.push(event.token);

    // Call token callback if provided
    if (request.onToken) {
      request.onToken(event.token, event.index);
    }

    // Emit token event for subscribers
    this.emit("stream_token", event);

    // Check if this is the final token
    if (event.final_token) {
      request.onComplete();
    }
  }

  /**
   * Connect to the socket server
   */
  async connect(): Promise<boolean> {
    if (this.state === "connected") return true;

    this.state = "connecting";
    this.emit("connecting", {});

    // Use WebSocket in browser, Tauri commands in app
    if (!isTauri) {
      return this.connectWebSocket();
    }

    try {
      // Initialize Tauri APIs
      const apisReady = await this.initTauriApis();
      if (!apisReady || !invoke) {
        throw new Error("Failed to initialize Tauri APIs");
      }

      // Set up event listeners before connecting
      await this.setupEventListeners();

      const connected = await withTimeout(
        invoke<boolean>("connect_socket"),
        REQUEST_TIMEOUT,
        "connect_socket"
      );

      if (connected) {
        this.state = "connected";
        this.reconnectAttempts = 0;
        this.emit("connected", {});
        return true;
      } else {
        this.state = "disconnected";
        this.scheduleReconnect();
        return false;
      }
    } catch (error) {
      console.warn("[JarvisSocket] Connection failed:", error);
      this.state = "disconnected";
      this.emit("error", { error });
      this.scheduleReconnect();
      return false;
    }
  }

  /**
   * Connect via WebSocket (browser mode)
   */
  private connectWebSocket(): Promise<boolean> {
    return new Promise((resolve) => {
      try {
        this.ws = new WebSocket(WEBSOCKET_URL);

        this.ws.onopen = () => {
          console.log("[JarvisSocket] WebSocket connected");
          this.state = "connected";
          this.reconnectAttempts = 0;
          this.startStaleRequestCleanup();
          this.emit("connected", {});
          resolve(true);
        };

        this.ws.onclose = () => {
          console.log("[JarvisSocket] WebSocket disconnected");
          this.state = "disconnected";
          this.ws = null;
          this.stopStaleRequestCleanup();
          // Reject all pending requests on disconnect
          for (const [requestId, pending] of this.wsPendingRequests) {
            pending.reject(new Error("WebSocket disconnected"));
          }
          this.wsPendingRequests.clear();
          this.emit("disconnected", {});
          this.scheduleReconnect();
        };

        this.ws.onerror = (error) => {
          console.warn("[JarvisSocket] WebSocket error:", error);
          this.emit("error", { error });
        };

        this.ws.onmessage = (event) => {
          this.handleWebSocketMessage(event.data);
        };

        // Timeout after 5 seconds
        setTimeout(() => {
          if (this.state === "connecting") {
            console.warn("[JarvisSocket] WebSocket connection timeout");
            this.ws?.close();
            this.state = "disconnected";
            resolve(false);
          }
        }, 5000);
      } catch (error) {
        console.warn("[JarvisSocket] WebSocket connection failed:", error);
        this.state = "disconnected";
        this.emit("error", { error });
        resolve(false);
      }
    });
  }

  /**
   * Handle incoming WebSocket message
   */
  private handleWebSocketMessage(data: string): void {
    try {
      const message = JSON.parse(data);

      // Check if it's a notification (no id)
      if (!("id" in message) && message.method) {
        // Handle streaming tokens
        if (message.method === "stream.token") {
          const { token, index, final } = message.params;
          const streamRequestId = (message.params as { request_id?: number }).request_id;
          if (streamRequestId !== undefined) {
            // Dispatch token to the specific request that owns this stream
            const pending = this.wsPendingRequests.get(streamRequestId);
            if (pending && pending.onToken) {
              pending.onToken(token, index);
            }
          }
          return;
        }

        // Handle other notifications (e.g., new_message)
        if (message.method === "new_message") {
          this.emit("new_message", message.params);
          return;
        }
      }

      // Handle response to a request
      if ("id" in message && message.id !== null) {
        const pending = this.wsPendingRequests.get(message.id);
        if (pending) {
          this.wsPendingRequests.delete(message.id);
          if (message.error) {
            pending.reject(new Error(message.error.message));
          } else {
            pending.resolve(message.result);
          }
        }
      }
    } catch (error) {
      console.warn("[JarvisSocket] Failed to parse WebSocket message:", error);
    }
  }

  /**
   * Disconnect from the socket server
   */
  async disconnect(): Promise<void> {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    await this.cleanupEventListeners();
    this.stopStaleRequestCleanup();

    // Clear batch timer and reject pending batched requests
    if (this.batchTimer) {
      clearTimeout(this.batchTimer);
      this.batchTimer = null;
    }
    this.batchQueue.forEach((req) => req.reject(new Error("Disconnected")));
    this.batchQueue = [];

    // Close WebSocket if open
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    // Clear any pending requests
    this.wsPendingRequests.clear();

    if (isTauri && invoke) {
      try {
        await invoke("disconnect_socket");
      } catch {
        // Ignore disconnect errors
      }
    }

    this.state = "disconnected";
    this.emit("disconnected", {});
  }

  /**
   * Check if connected
   */
  async isConnected(): Promise<boolean> {
    // Check WebSocket in browser mode
    if (!isTauri) {
      return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
    }

    if (!invoke) return false;
    try {
      return await invoke<boolean>("is_socket_connected");
    } catch {
      return false;
    }
  }

  /**
   * Call a method on the socket server (non-streaming)
   */
  async call<T>(method: string, params: object = {}): Promise<T> {
    // Ensure connected
    if (this.state !== "connected") {
      const connected = await this.connect();
      if (!connected) {
        throw new Error("Not connected to socket server");
      }
    }

    // Use WebSocket in browser mode
    if (!isTauri) {
      return this.callWebSocket<T>(method, params);
    }

    if (!invoke) {
      throw new Error("Tauri invoke not available");
    }

    try {
      const result = await withTimeout(
        invoke<T>("send_message", { method, params }),
        REQUEST_TIMEOUT,
        `send_message(${method})`
      );
      return result;
    } catch (error) {
      // On error, try to reconnect
      this.state = "disconnected";
      this.scheduleReconnect();
      throw error;
    }
  }

  /**
   * Call a method via WebSocket (browser mode)
   */
  private callWebSocket<T>(method: string, params: object = {}): Promise<T> {
    return new Promise((resolve, reject) => {
      if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
        reject(new Error("WebSocket not connected"));
        return;
      }

      const requestId = ++this.wsRequestId;
      const request = {
        jsonrpc: "2.0",
        method,
        params,
        id: requestId,
      };

      // Store pending request with metadata for cleanup
      this.wsPendingRequests.set(requestId, {
        resolve: resolve as (value: unknown) => void,
        reject,
        createdAt: Date.now(),
        isStreaming: false,
      });

      // Send request
      this.ws.send(JSON.stringify(request));

      // Timeout after configured duration
      setTimeout(() => {
        if (this.wsPendingRequests.has(requestId)) {
          this.wsPendingRequests.delete(requestId);
          reject(new Error("Request timeout"));
        }
      }, REQUEST_TIMEOUT);
    });
  }

  /**
   * Call a method with automatic batching (WebSocket mode only)
   * Collects rapid sequential calls and sends them as a single batch request.
   * This reduces round-trip overhead when making multiple calls in quick succession.
   *
   * @example
   * // These 3 calls made within 15ms will be batched into 1 request
   * const [result1, result2, result3] = await Promise.all([
   *   client.callBatched("ping", {}),
   *   client.callBatched("classify_intent", { text: "hello" }),
   *   client.callBatched("ping", {}),
   * ]);
   */
  async callBatched<T>(method: string, params: object = {}): Promise<T> {
    // In Tauri mode, just use regular call (batching handled differently)
    if (isTauri) {
      return this.call<T>(method, params);
    }

    // Ensure connected
    if (this.state !== "connected") {
      const connected = await this.connect();
      if (!connected) {
        throw new Error("Not connected to socket server");
      }
    }

    return new Promise<T>((resolve, reject) => {
      // Add to batch queue
      this.batchQueue.push({
        method,
        params,
        resolve: resolve as (value: unknown) => void,
        reject,
      });

      // If batch is full, send immediately
      if (this.batchQueue.length >= MAX_BATCH_SIZE) {
        this.flushBatch();
        return;
      }

      // Start batch timer if not already running
      if (!this.batchTimer) {
        this.batchTimer = setTimeout(() => {
          this.flushBatch();
        }, BATCH_WINDOW_MS);
      }
    });
  }

  /**
   * Flush the current batch queue, sending all queued requests
   */
  private flushBatch(): void {
    // Clear timer
    if (this.batchTimer) {
      clearTimeout(this.batchTimer);
      this.batchTimer = null;
    }

    // Nothing to flush
    if (this.batchQueue.length === 0) {
      return;
    }

    // Take all queued requests
    const batch = this.batchQueue.splice(0);

    // If only one request, send it directly (no batching overhead)
    if (batch.length === 1) {
      const { method, params, resolve, reject } = batch[0];
      this.callWebSocket(method, params)
        .then(resolve)
        .catch(reject);
      return;
    }

    // Build batch request
    const requests = batch.map((req, index) => ({
      method: req.method,
      params: req.params,
      _batch_index: index, // Track position for response mapping
    }));

    console.log(`[JarvisSocket] Sending batch of ${batch.length} requests`);

    // Send batch request
    this.callWebSocket<{ results: Array<{ result?: unknown; error?: { message: string } }> }>(
      "batch",
      { requests }
    )
      .then((response) => {
        // Map results back to individual promises
        const results = response.results || [];
        batch.forEach((req, index) => {
          const result = results[index];
          if (result?.error) {
            req.reject(new Error(result.error.message));
          } else {
            req.resolve(result?.result);
          }
        });
      })
      .catch((error) => {
        // Reject all requests on batch failure
        batch.forEach((req) => req.reject(error));
      });
  }

  /**
   * Call a streaming method with real token-by-token streaming
   */
  async *callStream(
    method: string,
    params: object = {},
    onToken?: (token: string, index: number) => void
  ): AsyncGenerator<string, void, unknown> {
    // Ensure connected
    if (this.state !== "connected") {
      const connected = await this.connect();
      if (!connected) {
        throw new Error("Not connected to socket server");
      }
    }

    // Use WebSocket streaming in browser mode
    if (!isTauri) {
      yield* this.callStreamWebSocket(method, params, onToken);
      return;
    }

    if (!invoke) {
      throw new Error("Tauri invoke not available");
    }

    // Start streaming request
    const requestId = await withTimeout(
      invoke<number>("send_streaming_message", { method, params }),
      STREAMING_TIMEOUT,
      `send_streaming_message(${method})`
    );

    // Token buffer for async iteration
    const tokenBuffer: string[] = [];
    let resolveNextToken: ((value: string | null) => void) | null = null;
    let completed = false;

    // Register streaming request handler
    this.streamingRequests.set(requestId, {
      tokens: [],
      onToken: (token, index) => {
        if (onToken) onToken(token, index);

        if (resolveNextToken) {
          resolveNextToken(token);
          resolveNextToken = null;
        } else {
          tokenBuffer.push(token);
        }
      },
      onComplete: () => {
        completed = true;
        if (resolveNextToken) {
          resolveNextToken(null);
          resolveNextToken = null;
        }
      },
    });

    try {
      // Yield tokens as they arrive
      while (!completed) {
        // Check if we have buffered tokens
        if (tokenBuffer.length > 0) {
          yield tokenBuffer.shift()!;
          continue;
        }

        // Wait for next token or completion
        const nextToken = await new Promise<string | null>((resolve) => {
          resolveNextToken = resolve;

          // Check again in case token arrived between check and promise
          if (tokenBuffer.length > 0) {
            resolve(tokenBuffer.shift()!);
            resolveNextToken = null;
          } else if (completed) {
            resolve(null);
            resolveNextToken = null;
          }
        });

        if (nextToken === null) {
          break;
        }

        yield nextToken;
      }
    } finally {
      this.streamingRequests.delete(requestId);
    }
  }

  /**
   * WebSocket streaming implementation
   */
  private async *callStreamWebSocket(
    method: string,
    params: object = {},
    onToken?: (token: string, index: number) => void
  ): AsyncGenerator<string, void, unknown> {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error("WebSocket not connected");
    }

    const requestId = ++this.wsRequestId;
    const request = {
      jsonrpc: "2.0",
      method,
      params: { ...params, stream: true },
      id: requestId,
    };

    // Token buffer for async iteration
    const tokenBuffer: string[] = [];
    let resolveNextToken: ((value: string | null) => void) | null = null;
    let completed = false;
    let error: Error | null = null;

    // Store pending request with streaming handler and metadata
    this.wsPendingRequests.set(requestId, {
      resolve: () => {
        completed = true;
        if (resolveNextToken) {
          resolveNextToken(null);
          resolveNextToken = null;
        }
      },
      reject: (err: Error) => {
        error = err;
        completed = true;
        if (resolveNextToken) {
          resolveNextToken(null);
          resolveNextToken = null;
        }
      },
      onToken: (token: string, index: number) => {
        if (onToken) onToken(token, index);

        if (resolveNextToken) {
          resolveNextToken(token);
          resolveNextToken = null;
        } else {
          tokenBuffer.push(token);
        }
      },
      createdAt: Date.now(),
      isStreaming: true, // Streaming requests get longer timeout
    });

    // Send request
    this.ws.send(JSON.stringify(request));

    try {
      // Yield tokens as they arrive
      while (!completed) {
        if (error) throw error;

        // Check if we have buffered tokens
        if (tokenBuffer.length > 0) {
          yield tokenBuffer.shift()!;
          continue;
        }

        // Wait for next token or completion
        const nextToken = await new Promise<string | null>((resolve) => {
          resolveNextToken = resolve;

          // Check again in case token arrived between check and promise
          if (tokenBuffer.length > 0) {
            resolve(tokenBuffer.shift()!);
            resolveNextToken = null;
          } else if (completed) {
            resolve(null);
            resolveNextToken = null;
          }
        });

        if (nextToken === null) {
          break;
        }

        yield nextToken;
      }

      if (error) throw error;
    } finally {
      this.wsPendingRequests.delete(requestId);
    }
  }

  /**
   * Register an event handler
   */
  on<T>(event: string, handler: EventHandler<T>): () => void {
    if (!this.eventHandlers.has(event)) {
      this.eventHandlers.set(event, new Set());
    }
    this.eventHandlers.get(event)!.add(handler as EventHandler);

    // Return unsubscribe function
    return () => {
      this.eventHandlers.get(event)?.delete(handler as EventHandler);
    };
  }

  /**
   * Remove an event handler
   */
  off<T>(event: string, handler: EventHandler<T>): void {
    this.eventHandlers.get(event)?.delete(handler as EventHandler);
  }

  /**
   * Emit an event to handlers
   */
  private emit<T>(event: string, data: T): void {
    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      handlers.forEach((handler) => handler(data));
    }
  }

  /**
   * Schedule a reconnection attempt
   */
  private scheduleReconnect(): void {
    if (this.reconnectTimer) return;
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.warn("[JarvisSocket] Max reconnect attempts reached");
      this.emit("max_reconnect_attempts", {});
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

    console.log(
      `[JarvisSocket] Scheduling reconnect in ${delay}ms (attempt ${this.reconnectAttempts})`
    );

    this.reconnectTimer = setTimeout(async () => {
      this.reconnectTimer = null;
      await this.connect();
    }, delay);
  }

  // ========== High-level API methods ==========

  /**
   * Generate draft replies for a conversation (non-streaming)
   */
  async generateDraft(params: GenerateDraftParams): Promise<GenerateDraftResult> {
    return this.call("generate_draft", params);
  }

  /**
   * Generate draft with real streaming
   */
  async generateDraftStream(
    params: GenerateDraftParams,
    onToken?: (token: string) => void
  ): Promise<GenerateDraftResult> {
    let fullText = "";
    let tokenCount = 0;

    for await (const token of this.callStream("generate_draft", params, (t) => {
      if (onToken) onToken(t);
    })) {
      fullText += token;
      tokenCount++;
    }

    return {
      suggestions: [{ text: fullText.trim(), confidence: 0.8 }],
      context_used: {
        num_messages: 0,
        participants: [],
        last_message: null,
      },
      streamed: true,
      tokens_generated: tokenCount,
    };
  }

  /**
   * Get smart reply suggestions
   */
  async getSmartReplies(params: SmartReplyParams): Promise<{
    suggestions: Array<{ text: string; score: number }>;
  }> {
    return this.call("get_smart_replies", params);
  }

  /**
   * Semantic search across messages
   */
  async semanticSearch(params: SemanticSearchParams): Promise<{
    results: Array<{
      message: {
        id: number;
        chat_id: string;
        text: string;
        sender: string;
        date: string;
      };
      similarity: number;
    }>;
    total_results: number;
  }> {
    return this.call("semantic_search", params);
  }

  /**
   * Classify message intent
   */
  async classifyIntent(text: string): Promise<IntentResult> {
    return this.call("classify_intent", { text });
  }

  /**
   * Get conversation summary (non-streaming)
   */
  async summarize(chatId: string, numMessages = 50): Promise<SummarizeResult> {
    return this.call("summarize", {
      chat_id: chatId,
      num_messages: numMessages,
    });
  }

  /**
   * Summarize with real streaming
   */
  async summarizeStream(
    chatId: string,
    numMessages = 50,
    onToken?: (token: string) => void
  ): Promise<SummarizeResult> {
    let fullText = "";
    let tokenCount = 0;

    for await (const token of this.callStream(
      "summarize",
      { chat_id: chatId, num_messages: numMessages },
      (t) => { if (onToken) onToken(t); }
    )) {
      fullText += token;
      tokenCount++;
    }

    // Parse the streamed response
    const lines = fullText.trim().split("\n");
    const summary = lines[0] || "Conversation summary unavailable";
    const keyPoints = lines
      .slice(1)
      .filter((line) => line.trim().length > 5)
      .map((line) => line.replace(/^[\-\*\•\d\.\)]+\s*/, "").trim())
      .slice(0, 5);

    return {
      summary,
      key_points: keyPoints.length > 0 ? keyPoints : ["See full conversation for details"],
      message_count: numMessages,
      streamed: true,
      tokens_generated: tokenCount,
    };
  }

  /**
   * Ping the server
   */
  async ping(): Promise<{ status: string }> {
    return this.call("ping");
  }
}

// Export singleton instance
export const jarvis = new JarvisSocket();

// Also export the class for testing
export { JarvisSocket };
