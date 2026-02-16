/**
 * Metrics store for routing metrics dashboard.
 *
 * Fetches routing metrics from the backend via socket RPC and provides
 * reactive state for the Dashboard component.
 */

import { writable, get } from 'svelte/store';
import { jarvis } from '../socket/client';

/** Latency breakdown for a single routing request */
export interface LatencyBreakdown {
  [phase: string]: number;
}

/** A single routing metrics event */
export interface MetricsRequest {
  timestamp: number;
  query_hash: string;
  routing_decision: string;
  similarity_score: number;
  cache_hit: boolean;
  model_loaded: boolean;
  embedding_computations: number;
  faiss_candidates: number;
  latency: LatencyBreakdown;
  total_latency_ms: number;
}

/** Aggregated summary stats */
export interface MetricsSummary {
  total_requests: number;
  avg_latency_ms: number;
  p50_latency_ms: number;
  p95_latency_ms: number;
  cache_hit_rate: number;
  decisions: Record<string, number>;
}

/** Full metrics state */
export interface MetricsState {
  recentRequests: MetricsRequest[];
  summary: MetricsSummary | null;
  loading: boolean;
  error: string | null;
  lastFetched: number | null;
}

const initialState: MetricsState = {
  recentRequests: [],
  summary: null,
  loading: false,
  error: null,
  lastFetched: null,
};

export const metricsStore = writable<MetricsState>(initialState);

let refreshInterval: ReturnType<typeof setInterval> | null = null;

/**
 * Fetch routing metrics from the backend
 */
export async function fetchMetrics(limit = 100): Promise<void> {
  const state = get(metricsStore);
  if (state.loading) return;

  metricsStore.update((s) => ({ ...s, loading: true, error: null }));

  try {
    const result = await jarvis.call<{
      recent_requests: MetricsRequest[];
      summary: MetricsSummary;
    }>('get_routing_metrics', { limit });

    metricsStore.update((s) => ({
      ...s,
      recentRequests: result.recent_requests || [],
      summary: result.summary || null,
      loading: false,
      lastFetched: Date.now(),
    }));
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Failed to fetch metrics';
    metricsStore.update((s) => ({
      ...s,
      loading: false,
      error: message,
    }));
  }
}

/**
 * Start auto-refreshing metrics every interval (default 10s)
 */
export function startMetricsPolling(intervalMs = 10000): void {
  stopMetricsPolling();
  fetchMetrics();
  refreshInterval = setInterval(() => fetchMetrics(), intervalMs);
}

/**
 * Stop auto-refreshing metrics
 */
export function stopMetricsPolling(): void {
  if (refreshInterval) {
    clearInterval(refreshInterval);
    refreshInterval = null;
  }
}
