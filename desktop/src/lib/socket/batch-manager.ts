/**
 * Batch Manager - handles request batching for JarvisSocket
 *
 * Extracted from client.ts to reduce class size (REF-14).
 * Supports both WebSocket and Tauri IPC batching.
 */

/** A queued request waiting to be batched */
export interface BatchedRequest {
  method: string;
  params: object;
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
}

/** Configuration for a batch queue */
interface BatchConfig {
  windowMs: number;
  maxSize: number;
}

/**
 * Manages request batching for a single transport (WebSocket or Tauri).
 * Collects rapid sequential calls and flushes them as a batch.
 */
export class BatchQueue {
  private queue: BatchedRequest[] = [];
  private timer: ReturnType<typeof setTimeout> | null = null;
  private config: BatchConfig;
  private flushCallback: (batch: BatchedRequest[]) => void;

  constructor(config: BatchConfig, flushCallback: (batch: BatchedRequest[]) => void) {
    this.config = config;
    this.flushCallback = flushCallback;
  }

  /**
   * Add a request to the batch queue.
   * Flushes immediately if batch is full, otherwise starts a timer.
   */
  enqueue(request: BatchedRequest): void {
    this.queue.push(request);

    if (this.queue.length >= this.config.maxSize) {
      this.flush();
      return;
    }

    if (!this.timer) {
      this.timer = setTimeout(() => {
        this.flush();
      }, this.config.windowMs);
    }
  }

  /**
   * Flush all queued requests by invoking the flush callback.
   */
  flush(): void {
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }

    if (this.queue.length === 0) return;

    const batch = this.queue.splice(0);
    this.flushCallback(batch);
  }

  /**
   * Reject all pending requests and clear the queue.
   */
  rejectAll(error: Error): void {
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    this.queue.forEach((req) => req.reject(error));
    this.queue = [];
  }

  /** Number of pending requests in the queue. */
  get size(): number {
    return this.queue.length;
  }
}

/** WebSocket batching configuration */
export const WS_BATCH_CONFIG = {
  windowMs: 15,
  maxSize: 10,
};

/** Tauri IPC batching configuration */
export const TAURI_BATCH_CONFIG = {
  windowMs: 10,
  maxSize: 5,
};
