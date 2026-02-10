/**
 * Polling Utilities
 * 
 * Standardized polling patterns with:
 * - Automatic cleanup
 * - Pause/resume
 * - Backoff on errors
 * - Visibility-aware pausing
 */

export type PollCallback = () => void | Promise<void>;

export interface PollingOptions {
  /** Polling interval in milliseconds */
  interval: number;
  /** Whether to run immediately on start */
  immediate?: boolean;
  /** Whether polling is initially enabled */
  enabled?: boolean;
  /** Callback for errors */
  onError?: (error: Error) => void;
  /** Whether to pause when tab is hidden */
  pauseWhenHidden?: boolean;
  /** Exponential backoff multiplier on error (1 = no backoff) */
  backoffMultiplier?: number;
  /** Maximum interval for backoff */
  maxBackoffInterval?: number;
}

export interface Poller {
  /** Start polling */
  start: () => void;
  /** Stop polling */
  stop: () => void;
  /** Pause polling (can be resumed) */
  pause: () => void;
  /** Resume polling */
  resume: () => void;
  /** Run callback immediately */
  tick: () => Promise<void>;
  /** Whether polling is currently active */
  readonly isRunning: boolean;
  /** Current polling interval */
  readonly currentInterval: number;
}

/**
 * Create a poller with the given callback and options
 */
export function createPoller(callback: PollCallback, options: PollingOptions): Poller {
  const {
    interval,
    immediate = true,
    enabled = true,
    onError,
    pauseWhenHidden = true,
    backoffMultiplier = 1,
    maxBackoffInterval = interval * 4,
  } = options;

  let intervalId: ReturnType<typeof setInterval> | null = null;
  let isRunning = false;
  let isPaused = false;
  let currentInterval = interval;
  let visibilityHandler: (() => void) | null = null;

  async function execute(): Promise<void> {
    try {
      await callback();
      // Reset interval on success
      if (currentInterval !== interval) {
        currentInterval = interval;
        if (intervalId) {
          clearInterval(intervalId);
          intervalId = setInterval(execute, currentInterval);
        }
      }
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      onError?.(error);

      // Apply backoff
      if (backoffMultiplier > 1 && intervalId) {
        currentInterval = Math.min(
          currentInterval * backoffMultiplier,
          maxBackoffInterval
        );
        clearInterval(intervalId);
        intervalId = setInterval(execute, currentInterval);
      }
    }
  }

  function start(): void {
    if (isRunning) return;
    isRunning = true;
    isPaused = false;

    if (immediate) {
      execute();
    }

    intervalId = setInterval(execute, currentInterval);

    // Setup visibility handling
    if (pauseWhenHidden && typeof document !== 'undefined') {
      visibilityHandler = () => {
        if (document.hidden) {
          pause();
        } else {
          resume();
        }
      };
      document.addEventListener('visibilitychange', visibilityHandler);
    }
  }

  function stop(): void {
    if (intervalId) {
      clearInterval(intervalId);
      intervalId = null;
    }
    isRunning = false;
    isPaused = false;
    currentInterval = interval;

    if (visibilityHandler) {
      document.removeEventListener('visibilitychange', visibilityHandler);
      visibilityHandler = null;
    }
  }

  function pause(): void {
    if (!isRunning || isPaused) return;
    isPaused = true;
    if (intervalId) {
      clearInterval(intervalId);
      intervalId = null;
    }
  }

  function resume(): void {
    if (!isRunning || !isPaused) return;
    isPaused = false;
    execute();
    intervalId = setInterval(execute, currentInterval);
  }

  // Auto-start if enabled
  if (enabled) {
    start();
  }

  return {
    start,
    stop,
    pause,
    resume,
    tick: execute,
    get isRunning() {
      return isRunning && !isPaused;
    },
    get currentInterval() {
      return currentInterval;
    },
  };
}

/**
 * Create a poller that runs multiple callbacks with different intervals
 */
export function createMultiPoller(
  callbacks: Array<{ callback: PollCallback; interval: number }>,
  options: Omit<PollingOptions, 'interval'> = {}
): { start: () => void; stop: () => void } {
  const pollers = callbacks.map(({ callback, interval }) =>
    createPoller(callback, { ...options, interval })
  );

  return {
    start: () => pollers.forEach((p) => p.start()),
    stop: () => pollers.forEach((p) => p.stop()),
  };
}
