/**
 * Health store for tracking API connection and system status
 *
 * Phase 4 Architecture V2: Prefers socket ping over HTTP for faster response
 */

import { writable } from "svelte/store";
import type { HealthResponse } from "../api/types";
import { api } from "../api/client";
import { jarvis } from "../socket";

// Check if running in Tauri context
const isTauri = typeof window !== "undefined" && "__TAURI__" in window;

export interface HealthState {
  connected: boolean;
  loading: boolean;
  error: string | null;
  data: HealthResponse | null;
  source: "socket" | "http" | null;
}

const initialState: HealthState = {
  connected: false,
  loading: false,
  error: null,
  data: null,
  source: null,
};

export const healthStore = writable<HealthState>(initialState);

/**
 * Check API connection - uses socket when available
 */
export async function checkApiConnection(): Promise<boolean> {
  healthStore.update((state) => ({ ...state, loading: true, error: null }));

  // Try socket first in Tauri context
  if (isTauri) {
    try {
      const connected = await jarvis.connect();
      if (connected) {
        const result = await jarvis.ping();
        healthStore.update((state) => ({
          ...state,
          connected: true,
          loading: false,
          source: "socket",
        }));
        return true;
      }
    } catch {
      // Fall through to HTTP
    }
  }

  // Fall back to HTTP
  try {
    await api.ping();
    healthStore.update((state) => ({
      ...state,
      connected: true,
      loading: false,
      source: "http",
    }));
    return true;
  } catch (error) {
    const message = error instanceof Error ? error.message : "Connection failed";
    healthStore.update((state) => ({
      ...state,
      connected: false,
      loading: false,
      error: message,
      source: null,
    }));
    return false;
  }
}

/**
 * Fetch full health status - uses socket when available
 */
export async function fetchHealth(): Promise<void> {
  healthStore.update((state) => ({ ...state, loading: true, error: null }));

  // Try socket first in Tauri context
  if (isTauri) {
    try {
      const connected = await jarvis.connect();
      if (connected) {
        const result = await jarvis.ping();
        const data: HealthResponse = {
          status: result.status === "ok" ? "healthy" : "degraded",
          imessage_access: true,
          memory_available_gb: 0,
          memory_used_gb: 0,
          memory_mode: "FULL",
          model_loaded: (result as { models_ready?: boolean }).models_ready ?? false,
          permissions_ok: true,
          details: null,
          jarvis_rss_mb: 0,
          jarvis_vms_mb: 0,
        };
        healthStore.update((state) => ({
          ...state,
          data,
          connected: true,
          loading: false,
          source: "socket",
        }));
        return;
      }
    } catch {
      // Fall through to HTTP
    }
  }

  // Fall back to HTTP
  try {
    const data = await api.getHealth();
    healthStore.update((state) => ({
      ...state,
      data,
      connected: true,
      loading: false,
      source: "http",
    }));
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to fetch health";
    healthStore.update((state) => ({
      ...state,
      loading: false,
      error: message,
      source: null,
    }));
  }
}
