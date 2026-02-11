/**
 * Simple LRU (Least Recently Used) Cache implementation
 * Used to limit memory usage for resources like avatar blob URLs
 */

export class LRUCache<K, V> {
  private cache: Map<K, V>;
  private maxSize: number;
  private onEvict: ((key: K, value: V) => void) | undefined;

  constructor(maxSize: number, onEvict?: (key: K, value: V) => void) {
    this.cache = new Map();
    this.maxSize = maxSize;
    this.onEvict = onEvict;
  }

  get(key: K): V | undefined {
    const value = this.cache.get(key);
    if (value !== undefined) {
      // Move to end (most recently used)
      this.cache.delete(key);
      this.cache.set(key, value);
    }
    return value;
  }

  set(key: K, value: V): void {
    if (this.cache.has(key)) {
      // If key exists with a different value, call eviction callback before replacing
      const oldValue = this.cache.get(key)!;
      this.cache.delete(key);
      // Only call eviction callback if the value is actually changing
      if (oldValue !== value) {
        this.onEvict?.(key, oldValue);
      }
    } else if (this.cache.size >= this.maxSize) {
      // Remove oldest entry (first in map)
      const firstKey = this.cache.keys().next().value;
      if (firstKey !== undefined) {
        const evictedValue = this.cache.get(firstKey)!;
        this.cache.delete(firstKey);
        this.onEvict?.(firstKey, evictedValue);
      }
    }
    this.cache.set(key, value);
  }

  delete(key: K): boolean {
    const value = this.cache.get(key);
    const deleted = this.cache.delete(key);
    if (deleted && value !== undefined) {
      this.onEvict?.(key, value);
    }
    return deleted;
  }

  has(key: K): boolean {
    return this.cache.has(key);
  }

  get size(): number {
    return this.cache.size;
  }

  keys(): IterableIterator<K> {
    return this.cache.keys();
  }

  forEach(callback: (value: V, key: K) => void): void {
    this.cache.forEach(callback);
  }

  clear(): void {
    // Call eviction callback for each entry before clearing
    this.cache.forEach((value, key) => {
      this.onEvict?.(key, value);
    });
    this.cache.clear();
  }
}
