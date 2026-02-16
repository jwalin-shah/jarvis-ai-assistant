/**
 * JARVIS Socket Client
 *
 * TypeScript client for communication with the Python socket server.
 * Uses Tauri commands to bridge to Unix socket (in app) or WebSocket (in browser).
 *
 * Delegates streaming logic to StreamManager and batching to BatchManager (REF-14).
 */

import { getSocketRpcWebSocketUrl } from '../config/runtime';
import type { InvokeArgs, InvokeOptions } from '@tauri-apps/api/core';
import { StreamManager } from './stream-manager';
import { BatchQueue, WS_BATCH_CONFIG, TAURI_BATCH_CONFIG } from './batch-manager';
import type { BatchedRequest } from './batch-manager';

// Check if running in Tauri context
const isTauri = typeof window !== 'undefined' && '__TAURI__' in window;

// WebSocket configuration for browser mode
const WEBSOCKET_URL = getSocketRpcWebSocketUrl();

// Request timeout configuration (ms)
const REQUEST_TIMEOUT = 30000; // 30s for normal requests
const STREAMING_TIMEOUT = 120000; // 120s for streaming requests
const STALE_REQUEST_CLEANUP_INTERVAL = 60000; // Clean up stale requests every 60s

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
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (err) => {
        clearTimeout(timer);
        reject(err);
      }
    );
  });
}

// Dynamic imports for Tauri APIs - only available in Tauri context
let invoke: (<T>(cmd: string, args?: InvokeArgs, options?: InvokeOptions) => Promise<T>) | null =
  null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let listen: ((event: string, handler: (event: any) => void) => Promise<() => void>) | null = null;

/** Connection state */
export type ConnectionState = 'disconnected' | 'connecting' | 'connected';

/**
 * Connection transport — tracks which transport is active (UX-01).
 * "unix_socket" = Tauri IPC, "websocket" = browser or fallback, null = not connected.
 */
export type ConnectionTransport = 'unix_socket' | 'websocket' | null;

/** Connection info exposed to UI components (UX-01) */
export interface ConnectionInfo {
  state: ConnectionState;
  transport: ConnectionTransport;
  /** True when Tauri mode fell back from Unix socket to WebSocket */
  isFallback: boolean;
}

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
  gated?: boolean;
  gated_confidence?: number;
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

/** Budget tiers for RPC operations (ms) */
const RPC_BUDGETS: Record<string, number> = {
  ping: 100,
  get_conversations: 100,
  get_messages: 100,
  get_health: 100,
  classify_intent: 100,
  get_smart_replies: 500,
  semantic_search: 500,
  generate_draft: 5000,
  summarize: 5000,
};

/** RPC timing record */
interface RPCTimingRecord {
  method: string;
  elapsed_ms: number;
  budget_ms: number | undefined;
  exceeded: boolean;
  timestamp: number;
}

/** Bounded buffer of recent RPC timings */
const MAX_RPC_TIMINGS = 500;
const rpcTimings: RPCTimingRecord[] = [];

function recordRPCTiming(method: string, elapsed_ms: number): void {
  const budget_ms = RPC_BUDGETS[method];
  const exceeded = budget_ms !== undefined && elapsed_ms > budget_ms;
  if (exceeded) {
    console.warn(`[RPC Budget] ${method} took ${elapsed_ms.toFixed(1)}ms (budget: ${budget_ms}ms)`);
  }
  rpcTimings.push({ method, elapsed_ms, budget_ms, exceeded, timestamp: Date.now() });
  if (rpcTimings.length > MAX_RPC_TIMINGS) {
    rpcTimings.splice(0, rpcTimings.length - MAX_RPC_TIMINGS);
  }
}

/** Get RPC compliance stats */
export function getRPCCompliance(method?: string): {
  total: number;
  compliant: number;
  compliance_pct: number;
  p95_ms: number;
} {
  const filtered = method
    ? rpcTimings.filter((r) => r.method === method)
    : rpcTimings.filter((r) => r.budget_ms !== undefined);
  if (filtered.length === 0) {
    return { total: 0, compliant: 0, compliance_pct: 100, p95_ms: 0 };
  }
  const compliant = filtered.filter((r) => !r.exceeded).length;
  const sorted = filtered.map((r) => r.elapsed_ms).sort((a, b) => a - b);
  const p95Index = Math.min(Math.floor(sorted.length * 0.95), sorted.length - 1);
  return {
    total: filtered.length,
    compliant,
    compliance_pct: (compliant / filtered.length) * 100,
    p95_ms: sorted[p95Index] ?? 0,
  };
}

/**
 * JARVIS Socket Client
 *
 * Provides methods for communicating with the Python daemon.
 * Uses Unix socket (via Tauri) in app, WebSocket in browser.
 * Supports both request/response and real streaming patterns.
 *
 * Delegates streaming to StreamManager and batching to BatchQueue (REF-14).
 */
class JarvisSocket {
  private state: ConnectionState = 'disconnected';
  private transport: ConnectionTransport = null;
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
      createdAt: number;
      isStreaming: boolean;
      timeoutId?: ReturnType<typeof setTimeout>;
    }
  > = new Map();
  private cleanupInterval: ReturnType<typeof setInterval> | null = null;

  // Streaming manager (REF-14)
  private streamManager = new StreamManager();

  // Batch queues (REF-14)
  private wsBatchQueue: BatchQueue;
  private tauriBatchQueue: BatchQueue;

  constructor() {
    // Wire up stream manager's emit to our emit
    this.streamManager.onEmit = (event, data) => this.emit(event, data);

    // WebSocket batch queue - flushes via callWebSocket
    this.wsBatchQueue = new BatchQueue(WS_BATCH_CONFIG, (batch) => {
      this.flushWsBatch(batch);
    });

    // Tauri batch queue - flushes via invoke
    this.tauriBatchQueue = new BatchQueue(TAURI_BATCH_CONFIG, (batch) => {
      this.flushTauriBatch(batch);
    });
  }

  /**
   * Get current connection state
   */
  getState(): ConnectionState {
    return this.state;
  }

  /**
   * Get full connection info including transport and fallback status (UX-01).
   * Components can subscribe to connection_info_changed events.
   */
  getConnectionInfo(): ConnectionInfo {
    return {
      state: this.state,
      transport: this.transport,
      isFallback: isTauri && this.transport === 'websocket',
    };
  }

  /**
   * Initialize Tauri APIs (FE-01: explicit null guards after dynamic imports)
   */
  private async initTauriApis(): Promise<boolean> {
    if (!isTauri) return false;

    try {
      if (!invoke) {
        const coreModule = await import('@tauri-apps/api/core');
        if (!coreModule || typeof coreModule.invoke !== 'function') {
          console.warn('[JarvisSocket] Tauri core module missing invoke function');
          return false;
        }
        invoke = coreModule.invoke;
      }
      if (!listen) {
        const eventModule = await import('@tauri-apps/api/event');
        if (!eventModule || typeof eventModule.listen !== 'function') {
          console.warn('[JarvisSocket] Tauri event module missing listen function');
          return false;
        }
        listen = eventModule.listen;
      }
      return true;
    } catch (error) {
      console.warn('[JarvisSocket] Failed to import Tauri APIs:', error);
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
    const unlistenToken = await listen(
      'jarvis:stream_token',
      (event: { payload: StreamTokenEvent }) => {
        this.streamManager.handleStreamToken(event.payload);
      }
    );
    this.unlistenFns.push(unlistenToken);

    // Listen for new message notifications
    const unlistenNewMsg = await listen(
      'jarvis:new_message',
      (event: { payload: NewMessageEvent }) => {
        this.emit('new_message', event.payload);
      }
    );
    this.unlistenFns.push(unlistenNewMsg);

    // Listen for connection events
    const unlistenConnected = await listen('jarvis:connected', () => {
      this.state = 'connected';
      this.emit('connected', {});
    });
    this.unlistenFns.push(unlistenConnected);

    const unlistenDisconnected = await listen('jarvis:disconnected', () => {
      this.state = 'disconnected';
      this.emit('disconnected', {});
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
      // Don't clean up if socket is disconnected (already handled by onclose)
      if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
        return;
      }

      const now = Date.now();
      for (const [requestId, pending] of this.wsPendingRequests) {
        const timeout = pending.isStreaming ? STREAMING_TIMEOUT : REQUEST_TIMEOUT;
        if (now - pending.createdAt > timeout) {
          console.warn(`[JarvisSocket] Cleaning up stale request ${requestId}`);
          pending.reject(new Error('Request timed out (cleanup)'));
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
   * Connect to the socket server
   */
  async connect(): Promise<boolean> {
    if (this.state === 'connected') return true;

    this.state = 'connecting';
    this.emit('connecting', {});

    // Use WebSocket in browser, Tauri commands in app
    if (!isTauri) {
      return this.connectWebSocket();
    }

    try {
      // Initialize Tauri APIs
      const apisReady = await this.initTauriApis();
      if (!apisReady || !invoke) {
        throw new Error('Failed to initialize Tauri APIs');
      }

      // Set up event listeners before connecting
      await this.setupEventListeners();

      const connected = await withTimeout(
        invoke<boolean>('connect_socket'),
        REQUEST_TIMEOUT,
        'connect_socket'
      );

      if (connected) {
        this.state = 'connected';
        this.transport = 'unix_socket';
        this.reconnectAttempts = 0;
        this.emit('connected', {});
        this.emitConnectionInfoChanged();
        return true;
      } else {
        this.state = 'disconnected';
        this.scheduleReconnect();
        return false;
      }
    } catch (error) {
      console.warn('[JarvisSocket] Connection failed:', error);
      this.state = 'disconnected';
      this.emit('error', { error });
      this.scheduleReconnect();
      return false;
    }
  }

  /**
   * Connect via WebSocket (browser mode or Tauri fallback)
   */
  private connectWebSocket(): Promise<boolean> {
    return new Promise((resolve) => {
      try {
        this.ws = new WebSocket(WEBSOCKET_URL);

        this.ws.onopen = () => {
          console.log('[JarvisSocket] WebSocket connected');
          this.state = 'connected';
          this.transport = 'websocket';
          this.reconnectAttempts = 0;
          this.startStaleRequestCleanup();
          this.emit('connected', {});
          this.emitConnectionInfoChanged();
          resolve(true);
        };

        this.ws.onclose = () => {
          console.log('[JarvisSocket] WebSocket disconnected');
          this.state = 'disconnected';
          this.transport = null;
          this.ws = null;
          this.stopStaleRequestCleanup();
          // Reject pending batch requests
          this.wsBatchQueue.rejectAll(new Error('WebSocket disconnected'));
          // Reject all pending requests on disconnect
          for (const [_requestId, pending] of this.wsPendingRequests) {
            if (pending.timeoutId) {
              clearTimeout(pending.timeoutId);
            }
            pending.reject(new Error('WebSocket disconnected'));
          }
          this.wsPendingRequests.clear();
          this.emit('disconnected', {});
          this.emitConnectionInfoChanged();
          this.scheduleReconnect();
        };

        this.ws.onerror = (error) => {
          console.warn('[JarvisSocket] WebSocket error:', error);
          this.emit('error', { error });
        };

        this.ws.onmessage = (event) => {
          this.handleWebSocketMessage(event.data);
        };

        // Timeout after 5 seconds
        setTimeout(() => {
          if (this.state === 'connecting') {
            console.warn('[JarvisSocket] WebSocket connection timeout');
            this.ws?.close();
            this.state = 'disconnected';
            resolve(false);
          }
        }, 5000);
      } catch (error) {
        console.warn('[JarvisSocket] WebSocket connection failed:', error);
        this.state = 'disconnected';
        this.emit('error', { error });
        resolve(false);
      }
    });
  }

  /**
   * Handle incoming WebSocket message (FE-03: validate params before use)
   */
  private handleWebSocketMessage(data: string): void {
    try {
      const message = JSON.parse(data);

      // Check if it's a notification (no id)
      if (!('id' in message) && message.method) {
        // Handle streaming tokens (FE-03: validate params structure)
        if (message.method === 'stream.token') {
          const params = message.params;
          if (
            params &&
            typeof params === 'object' &&
            typeof params.token === 'string' &&
            typeof params.index === 'number'
          ) {
            const { token, index } = params;
            const streamRequestId =
              typeof params.request_id === 'number' ? params.request_id : undefined;
            if (streamRequestId !== undefined) {
              const pending = this.wsPendingRequests.get(streamRequestId);
              if (pending && pending.onToken) {
                pending.onToken(token, index);
              }
            }
          } else {
            console.warn('[JarvisSocket] Malformed stream.token params:', params);
          }
          return;
        }

        // Handle other notifications (e.g., new_message)
        if (message.method === 'new_message') {
          this.emit('new_message', message.params);
          return;
        }
      }

      // Handle response to a request
      if ('id' in message && message.id !== null) {
        const pending = this.wsPendingRequests.get(message.id);
        if (pending) {
          if (pending.timeoutId) {
            clearTimeout(pending.timeoutId);
          }
          this.wsPendingRequests.delete(message.id);
          if (message.error) {
            pending.reject(new Error(message.error.message));
          } else {
            pending.resolve(message.result);
          }
        }
      }
    } catch (error) {
      console.warn('[JarvisSocket] Failed to parse WebSocket message:', error);
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

    // Reject pending batch requests
    this.wsBatchQueue.rejectAll(new Error('Disconnected'));
    this.tauriBatchQueue.rejectAll(new Error('Disconnected'));

    // Close WebSocket if open
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    // Clear any pending requests
    this.wsPendingRequests.clear();

    // Clear streaming requests
    this.streamManager.clear();

    if (isTauri && invoke) {
      try {
        await invoke('disconnect_socket');
      } catch {
        // Ignore disconnect errors
      }
    }

    this.state = 'disconnected';
    this.transport = null;
    this.emit('disconnected', {});
    this.emitConnectionInfoChanged();
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
      return await invoke<boolean>('is_socket_connected');
    } catch {
      return false;
    }
  }

  /**
   * Call a method on the socket server (non-streaming)
   */
  async call<T>(method: string, params: object = {}): Promise<T> {
    // Ensure connected
    if (this.state !== 'connected') {
      const connected = await this.connect();
      if (!connected) {
        throw new Error('Not connected to socket server');
      }
    }

    // Use WebSocket in browser mode
    if (!isTauri) {
      const wsStart = performance.now();
      try {
        const result = await this.callWebSocket<T>(method, params);
        recordRPCTiming(method, performance.now() - wsStart);
        return result;
      } catch (error) {
        recordRPCTiming(method, performance.now() - wsStart);
        throw error;
      }
    }

    if (!invoke) {
      throw new Error('Tauri invoke not available');
    }

    const rpcStart = performance.now();
    try {
      const result = await withTimeout(
        invoke<T>('send_message', { method, params }),
        REQUEST_TIMEOUT,
        `send_message(${method})`
      );
      recordRPCTiming(method, performance.now() - rpcStart);
      return result;
    } catch (error) {
      recordRPCTiming(method, performance.now() - rpcStart);
      // On error, try to reconnect
      this.state = 'disconnected';
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
        reject(new Error('WebSocket not connected'));
        return;
      }

      const requestId = ++this.wsRequestId;
      const request = {
        jsonrpc: '2.0',
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
      const timeoutId = setTimeout(() => {
        if (this.wsPendingRequests.has(requestId)) {
          this.wsPendingRequests.delete(requestId);
          reject(new Error('Request timeout'));
        }
      }, REQUEST_TIMEOUT);

      // Store timeout ID so it can be cleared when request resolves
      const pending = this.wsPendingRequests.get(requestId);
      if (pending) {
        pending.timeoutId = timeoutId;
      }
    });
  }

  /**
   * Call a method with automatic batching.
   * Collects rapid sequential calls and sends them as a single batch request.
   */
  async callBatched<T>(method: string, params: object = {}): Promise<T> {
    // In Tauri mode, use Tauri-specific batching
    if (isTauri) {
      return new Promise<T>((resolve, reject) => {
        this.tauriBatchQueue.enqueue({
          method,
          params,
          resolve: resolve as (value: unknown) => void,
          reject,
        });
      });
    }

    // Ensure connected
    if (this.state !== 'connected') {
      const connected = await this.connect();
      if (!connected) {
        throw new Error('Not connected to socket server');
      }
    }

    return new Promise<T>((resolve, reject) => {
      this.wsBatchQueue.enqueue({
        method,
        params,
        resolve: resolve as (value: unknown) => void,
        reject,
      });
    });
  }

  /**
   * Flush Tauri batch queue
   */
  private async flushTauriBatch(batch: BatchedRequest[]): Promise<void> {
    // If only one request, send it directly
    if (batch.length === 1) {
      const req = batch[0];
      if (req) {
        this.call(req.method, req.params).then(req.resolve).catch(req.reject);
      }
      return;
    }

    // Send as batch
    console.log(`[JarvisSocket] Sending Tauri batch of ${batch.length} requests`);

    try {
      const results = (await withTimeout(
        invoke!('send_batch', {
          requests: batch.map((r) => ({ method: r.method, params: r.params })),
        }),
        REQUEST_TIMEOUT,
        'send_batch'
      )) as Array<{ result?: unknown; error?: string }>;

      // Map results back to individual promises
      batch.forEach((req, index) => {
        const result = results[index];
        if (result?.error) {
          req.reject(new Error(result.error));
        } else {
          req.resolve(result?.result);
        }
      });
    } catch (error) {
      // Reject all requests on batch failure
      batch.forEach((req) => req.reject(error as Error));
    }
  }

  /**
   * Flush WebSocket batch queue
   */
  private flushWsBatch(batch: BatchedRequest[]): void {
    // Check if WebSocket is still connected before sending
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.warn('[JarvisSocket] Cannot flush batch: WebSocket not connected');
      batch.forEach((req) => req.reject(new Error('WebSocket not connected')));
      return;
    }

    // If only one request, send it directly (no batching overhead)
    if (batch.length === 1) {
      const first = batch[0];
      if (first) {
        const { method, params, resolve, reject } = first;
        this.callWebSocket(method, params).then(resolve).catch(reject);
      }
      return;
    }

    // Build batch request
    const requests = batch.map((req, index) => ({
      method: req.method,
      params: req.params,
      _batch_index: index,
    }));

    console.log(`[JarvisSocket] Sending batch of ${batch.length} requests`);

    // Send batch request
    this.callWebSocket<{ results: Array<{ result?: unknown; error?: { message: string } }> }>(
      'batch',
      { requests }
    )
      .then((response) => {
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
    if (this.state !== 'connected') {
      const connected = await this.connect();
      if (!connected) {
        throw new Error('Not connected to socket server');
      }
    }

    // Use WebSocket streaming in browser mode
    if (!isTauri) {
      yield* this.callStreamWebSocket(method, params, onToken);
      return;
    }

    if (!invoke) {
      throw new Error('Tauri invoke not available');
    }

    // Start streaming request
    const requestId = await withTimeout(
      invoke<number>('send_streaming_message', { method, params }),
      STREAMING_TIMEOUT,
      `send_streaming_message(${method})`
    );

    // Token buffer for async iteration
    const tokenBuffer: string[] = [];
    let resolveNextToken: ((value: string | null) => void) | null = null;
    let completed = false;
    let timedOut = false;

    // Create idle timeout that auto-cleans if server drops mid-stream
    const createIdleTimeout = (): ReturnType<typeof setTimeout> => {
      return this.streamManager.createIdleTimeout(requestId, () => {
        timedOut = true;
        completed = true;
        if (resolveNextToken) {
          resolveNextToken(null);
          resolveNextToken = null;
        }
      });
    };

    // Register streaming request handler with idle timeout
    const initialTimeoutId = createIdleTimeout();
    this.streamManager.register(requestId, {
      tokens: [],
      timeoutId: initialTimeoutId,
      onToken: (token, index) => {
        // Reset idle timeout on each token received
        const entry = this.streamManager.get(requestId);
        if (entry) {
          clearTimeout(entry.timeoutId);
          entry.timeoutId = createIdleTimeout();
        }

        if (onToken) onToken(token, index);

        if (resolveNextToken) {
          resolveNextToken(token);
          resolveNextToken = null;
        } else {
          tokenBuffer.push(token);
        }
      },
      onComplete: () => {
        // Clear idle timeout on normal completion
        const entry = this.streamManager.get(requestId);
        if (entry) {
          clearTimeout(entry.timeoutId);
        }
        completed = true;
        if (resolveNextToken) {
          resolveNextToken(null);
          resolveNextToken = null;
        }
      },
    });

    // Overall timeout for the entire stream (2 minutes)
    const streamStartTime = Date.now();
    const MAX_STREAM_DURATION = 120000;

    try {
      // Yield tokens as they arrive
      while (!completed) {
        // Check overall timeout
        if (Date.now() - streamStartTime > MAX_STREAM_DURATION) {
          console.warn(`[JarvisSocket] Stream exceeded max duration of ${MAX_STREAM_DURATION}ms`);
          timedOut = true;
          break;
        }

        // Check if we have buffered tokens
        if (tokenBuffer.length > 0) {
          yield tokenBuffer.shift()!;
          continue;
        }

        // Wait for next token or completion (with shorter timeout to check overall duration)
        const nextToken = await Promise.race([
          new Promise<string | null>((resolve) => {
            resolveNextToken = resolve;

            // Check again in case token arrived between check and promise
            if (tokenBuffer.length > 0) {
              resolve(tokenBuffer.shift()!);
              resolveNextToken = null;
            } else if (completed) {
              resolve(null);
              resolveNextToken = null;
            }
          }),
          // 5 second timeout to periodically check overall duration
          new Promise<never>((_, reject) =>
            setTimeout(() => reject(new Error('TOKEN_TIMEOUT')), 5000)
          ),
        ]).catch((err) => {
          if (err.message === 'TOKEN_TIMEOUT') {
            return null; // Return null to continue loop and check overall timeout
          }
          throw err;
        });

        if (nextToken === null) {
          continue; // Continue to check overall timeout
        }

        yield nextToken;
      }

      if (timedOut) {
        throw new Error(`Streaming request timed out after ${MAX_STREAM_DURATION}ms`);
      }
    } finally {
      const entry = this.streamManager.get(requestId);
      if (entry) {
        clearTimeout(entry.timeoutId);
      }
      this.streamManager.delete(requestId);
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
      throw new Error('WebSocket not connected');
    }

    const requestId = ++this.wsRequestId;
    const request = {
      jsonrpc: '2.0',
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
      isStreaming: true,
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
   * Emit connection info changed event (UX-01).
   * UI components can subscribe to this to show connection indicators.
   */
  private emitConnectionInfoChanged(): void {
    this.emit('connection_info_changed', this.getConnectionInfo());
  }

  /**
   * Schedule a reconnection attempt.
   * In Tauri mode, falls back to WebSocket after max Unix socket retries (UX-01).
   */
  private scheduleReconnect(): void {
    if (this.reconnectTimer) return;
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      if (isTauri) {
        // Unix socket exhausted - fall back to WebSocket (UX-01: emit fallback event)
        console.warn(
          '[JarvisSocket] Unix socket reconnect failed after %d attempts, falling back to WebSocket',
          this.maxReconnectAttempts
        );
        this.emit('transport_fallback', {
          from: 'unix_socket',
          to: 'websocket',
          reason: `Unix socket failed after ${this.maxReconnectAttempts} attempts`,
        });
        this.reconnectAttempts = 0;
        this.reconnectTimer = setTimeout(async () => {
          this.reconnectTimer = null;
          const ok = await this.connectWebSocket();
          if (!ok) {
            this.emit('max_reconnect_attempts', {});
          }
        }, this.reconnectDelay);
        return;
      }
      console.warn('[JarvisSocket] Max reconnect attempts reached');
      this.emit('max_reconnect_attempts', {});
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
    return this.call('generate_draft', params);
  }

  /**
   * Generate draft with real streaming
   */
  async generateDraftStream(
    params: GenerateDraftParams,
    onToken?: (token: string) => void
  ): Promise<GenerateDraftResult> {
    let fullText = '';
    let tokenCount = 0;

    for await (const token of this.callStream('generate_draft', params, (t) => {
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
    return this.call('get_smart_replies', params);
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
    return this.call('semantic_search', params);
  }

  /**
   * Classify message intent
   */
  async classifyIntent(text: string): Promise<IntentResult> {
    return this.call('classify_intent', { text });
  }

  /**
   * Get conversation summary (non-streaming)
   */
  async summarize(chatId: string, numMessages = 50): Promise<SummarizeResult> {
    return this.call('summarize', {
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
    let fullText = '';
    let tokenCount = 0;

    for await (const token of this.callStream(
      'summarize',
      { chat_id: chatId, num_messages: numMessages },
      (t) => {
        if (onToken) onToken(t);
      }
    )) {
      fullText += token;
      tokenCount++;
    }

    // Parse the streamed response
    const lines = fullText.trim().split('\n');
    const summary = lines[0] || 'Conversation summary unavailable';
    const keyPoints = lines
      .slice(1)
      .filter((line) => line.trim().length > 5)
      .map((line) => line.replace(/^[\-\*\•\d\.\)]+\s*/, '').trim())
      .slice(0, 5);

    return {
      summary,
      key_points: keyPoints.length > 0 ? keyPoints : ['See full conversation for details'],
      message_count: numMessages,
      streamed: true,
      tokens_generated: tokenCount,
    };
  }

  /**
   * Chat directly with the SLM (streaming).
   * Returns the full response after streaming completes.
   */
  async chatStream(
    message: string,
    history: Array<{ role: 'user' | 'assistant'; content: string }>,
    onToken?: (token: string) => void
  ): Promise<{ response: string; tokens_generated: number }> {
    let fullText = '';
    let tokenCount = 0;
    const startTime = Date.now();

    console.log('[JarvisSocket] Starting chatStream...');

    try {
      for await (const token of this.callStream('chat', { message, history }, (t) => {
        if (onToken) onToken(t);
      })) {
        fullText += token;
        tokenCount++;

        // Safety: break if streaming takes too long (2 minutes)
        if (Date.now() - startTime > 120000) {
          console.warn('[JarvisSocket] chatStream timeout - breaking loop');
          break;
        }
      }
    } catch (error) {
      console.error('[JarvisSocket] chatStream error:', error);
      throw error;
    }

    console.log(`[JarvisSocket] chatStream completed: ${tokenCount} tokens`);

    return {
      response: fullText.trim(),
      tokens_generated: tokenCount,
    };
  }

  /**
   * Ping the server
   */
  async ping(): Promise<Record<string, unknown>> {
    return this.call('ping');
  }
}

// Export singleton instance
export const jarvis = new JarvisSocket();

// Also export the class for testing
export { JarvisSocket };
