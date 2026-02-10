/**
 * Store Factory Utilities
 * 
 * Provides standardized patterns for creating stores with:
 * - Async data fetching
 * - Loading states
 * - Error handling
 * - Batched updates
 */

import { writable, derived, get, type Readable, type Writable } from 'svelte/store';

export interface AsyncState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

export interface AsyncStore<T> extends Readable<AsyncState<T>> {
  setLoading: () => void;
  setData: (data: T) => void;
  setError: (error: string) => void;
  refresh: () => Promise<void>;
  getData: () => T | null;
}

export interface BatchedStoreOptions {
  batchMs?: number;
  maxBatchSize?: number;
}

/**
 * Create a store for async data with loading/error states
 */
export function createAsyncStore<T>(
  fetcher: () => Promise<T>,
  initialData: T | null = null
): AsyncStore<T> {
  const store = writable<AsyncState<T>>({
    data: initialData,
    loading: false,
    error: null,
  });

  async function refresh(): Promise<void> {
    store.update((s) => ({ ...s, loading: true, error: null }));
    try {
      const data = await fetcher();
      store.set({ data, loading: false, error: null });
    } catch (err) {
      store.set({
        data: null,
        loading: false,
        error: err instanceof Error ? err.message : 'Unknown error',
      });
    }
  }

  return {
    subscribe: store.subscribe,
    setLoading: () => store.update((s) => ({ ...s, loading: true })),
    setData: (data) => store.set({ data, loading: false, error: null }),
    setError: (error) => store.set({ data: null, loading: false, error }),
    refresh,
    getData: () => get(store).data,
  };
}

/**
 * Create a store with batched updates to reduce re-renders
 */
export function createBatchedStore<T>(
  initial: T,
  options: BatchedStoreOptions = {}
): Writable<T> {
  const { batchMs = 16 } = options;
  const store = writable<T>(initial);
  let pendingValue: T | undefined = undefined;
  let timeoutId: ReturnType<typeof setTimeout> | null = null;

  function flush(): void {
    if (pendingValue !== undefined) {
      store.set(pendingValue);
      pendingValue = undefined;
    }
    timeoutId = null;
  }

  return {
    subscribe: store.subscribe,
    set: (value: T) => {
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
      pendingValue = value;
      timeoutId = setTimeout(flush, batchMs);
    },
    update: (updater) => {
      store.update((current) => {
        const newValue = updater(current);
        if (timeoutId) {
          clearTimeout(timeoutId);
        }
        pendingValue = newValue;
        timeoutId = setTimeout(flush, batchMs);
        return current; // Return current until batch flush
      });
    },
  };
}

/**
 * Create a derived store that only updates when the value actually changes
 */
export function createMemoizedDerived<T, U>(
  store: Readable<T>,
  selector: (value: T) => U,
  isEqual: (a: U, b: U) => boolean = (a, b) => a === b
): Readable<U> {
  let previousValue: U | undefined = undefined;

  return derived(store, ($value, set) => {
    const newValue = selector($value);
    if (previousValue === undefined || !isEqual(newValue, previousValue)) {
      previousValue = newValue;
      set(newValue);
    }
  });
}

/**
 * Create a store with local storage persistence
 */
export function createPersistentStore<T>(
  key: string,
  initialValue: T,
  serializer: {
    stringify: (value: T) => string;
    parse: (value: string) => T;
  } = {
    stringify: JSON.stringify,
    parse: JSON.parse,
  }
): Writable<T> {
  const stored = typeof localStorage !== 'undefined' ? localStorage.getItem(key) : null;
  const initial = stored ? serializer.parse(stored) : initialValue;
  const store = writable<T>(initial);

  return {
    subscribe: store.subscribe,
    set: (value: T) => {
      store.set(value);
      if (typeof localStorage !== 'undefined') {
        localStorage.setItem(key, serializer.stringify(value));
      }
    },
    update: (updater) => {
      store.update((current) => {
        const newValue = updater(current);
        if (typeof localStorage !== 'undefined') {
          localStorage.setItem(key, serializer.stringify(newValue));
        }
        return newValue;
      });
    },
  };
}
