/**
 * Stream Manager - handles streaming token logic for JarvisSocket
 *
 * Extracted from client.ts to reduce class size (REF-14).
 */

import type { StreamTokenEvent } from "./client";

/** Idle timeout before auto-cleaning a stale streaming request (15s) */
const STREAM_IDLE_TIMEOUT = 15000;

/** Internal state for an active streaming request */
export interface StreamingRequest {
  tokens: string[];
  onToken?: (token: string, index: number) => void;
  onComplete: () => void;
  timeoutId: ReturnType<typeof setTimeout>;
}

/**
 * Manages active streaming requests and their token delivery.
 */
export class StreamManager {
  private streamingRequests: Map<number, StreamingRequest> = new Map();

  /** Event emitter callback - set by JarvisSocket */
  onEmit: ((event: string, data: unknown) => void) | null = null;

  /**
   * Register a new streaming request.
   */
  register(requestId: number, request: StreamingRequest): void {
    this.streamingRequests.set(requestId, request);
  }

  /**
   * Check if a streaming request exists.
   */
  has(requestId: number): boolean {
    return this.streamingRequests.has(requestId);
  }

  /**
   * Get a streaming request entry.
   */
  get(requestId: number): StreamingRequest | undefined {
    return this.streamingRequests.get(requestId);
  }

  /**
   * Remove a streaming request.
   */
  delete(requestId: number): void {
    this.streamingRequests.delete(requestId);
  }

  /**
   * Handle an incoming stream token event.
   * Dispatches to the registered request, manages idle timeouts,
   * and emits events for subscribers.
   */
  handleStreamToken(event: StreamTokenEvent): void {
    const request = this.streamingRequests.get(event.request_id);
    if (!request) return;

    // Reset idle timeout on each token received
    clearTimeout(request.timeoutId);
    if (!event.final_token) {
      request.timeoutId = setTimeout(() => {
        console.warn(
          `[StreamManager] Streaming request ${event.request_id} timed out after ${STREAM_IDLE_TIMEOUT}ms of inactivity`
        );
        request.onComplete();
        this.streamingRequests.delete(event.request_id);
      }, STREAM_IDLE_TIMEOUT);
    }

    // Accumulate token
    request.tokens.push(event.token);

    // Call token callback if provided
    if (request.onToken) {
      request.onToken(event.token, event.index);
    }

    // Emit token event for subscribers
    if (this.onEmit) {
      this.onEmit("stream_token", event);
    }

    // Check if this is the final token
    if (event.final_token) {
      clearTimeout(request.timeoutId);
      request.tokens.length = 0;
      this.streamingRequests.delete(event.request_id);
      request.onComplete();
    }
  }

  /**
   * Clear all streaming requests and their idle timeouts.
   */
  clear(): void {
    for (const [, request] of this.streamingRequests) {
      clearTimeout(request.timeoutId);
    }
    this.streamingRequests.clear();
  }

  /**
   * Create an idle timeout that auto-cleans if server drops mid-stream.
   */
  createIdleTimeout(
    requestId: number,
    onTimeout: () => void
  ): ReturnType<typeof setTimeout> {
    return setTimeout(() => {
      console.warn(
        `[StreamManager] Streaming request ${requestId} timed out after ${STREAM_IDLE_TIMEOUT}ms of inactivity`
      );
      this.streamingRequests.delete(requestId);
      onTimeout();
    }, STREAM_IDLE_TIMEOUT);
  }
}
