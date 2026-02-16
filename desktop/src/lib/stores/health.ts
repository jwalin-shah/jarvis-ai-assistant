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

// Cache connection state to avoid redundant updates
let cachedConnectionState: Pick<HealthState, "connected" | "source"> | null = null;

// In-flight promise to deduplicate concurrent connection checks
let inflightCheck: Promise<boolean> | null = null;

/**
 * Check API connection - uses socket when available
 */
export async function checkApiConnection(): Promise<boolean> {
  // If a check is already in progress, return its promise instead of starting a new one
  if (inflightCheck) {
    return inflightCheck;
  }

  inflightCheck = checkApiConnectionInternal();
  try {
    return await inflightCheck;
  } finally {
    inflightCheck = null;
  }
}

async function checkApiConnectionInternal(): Promise<boolean> {
  healthStore.update((state) => ({ ...state, loading: true, error: null }));

  // Try socket first in Tauri context
  if (isTauri) {
    try {
      const connected = await jarvis.connect();
      if (connected) {
        await jarvis.ping();
        const newState = { connected: true, source: "socket" as const };
        // Only update if connection state actually changed
        if (!cachedConnectionState || cachedConnectionState.connected !== newState.connected || cachedConnectionState.source !== newState.source) {
          cachedConnectionState = newState;
          healthStore.update((state) => ({
            ...state,
            connected: true,
            loading: false,
            source: "socket",
          }));
        } else {
          healthStore.update((state) => ({ ...state, loading: false }));
        }
        return true;
      }
    } catch {
      // Fall through to HTTP
    }
  }

  // Fall back to HTTP
  try {
    await api.ping();
    const newState = { connected: true, source: "http" as const };
    if (!cachedConnectionState || cachedConnectionState.connected !== newState.connected || cachedConnectionState.source !== newState.source) {
      cachedConnectionState = newState;
      healthStore.update((state) => ({
        ...state,
        connected: true,
        loading: false,
        source: "http",
      }));
    } else {
      healthStore.update((state) => ({ ...state, loading: false }));
    }
    return true;
  } catch (error) {
    const message = error instanceof Error ? error.message : "Connection failed";
    const newState = { connected: false, source: null };
    if (!cachedConnectionState || cachedConnectionState.connected !== newState.connected) {
      cachedConnectionState = newState;
      healthStore.update((state) => ({
        ...state,
        connected: false,
        loading: false,
        error: message,
        source: null,
      }));
    } else {
      healthStore.update((state) => ({ ...state, loading: false, error: message }));
    }
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
        const result = await jarvis.request<any>("get_health", {});
        const data: HealthResponse = {
          status: (result.status as HealthResponse["status"]) ?? "degraded",
          imessage_access: (result.imessage_access as boolean) ?? null,
          memory_available_gb: (result.memory_available_gb as number) ?? null,
          memory_used_gb: (result.memory_used_gb as number) ?? null,
          memory_mode: (result.memory_mode as HealthResponse["memory_mode"]) ?? null,
          model_loaded: (result.model_loaded as boolean) ?? null,
          permissions_ok: (result.permissions_ok as boolean) ?? null,
          details: (result.details as HealthResponse["details"]) ?? null,
          jarvis_rss_mb: (result.jarvis_rss_mb as number) ?? null,
          jarvis_vms_mb: (result.jarvis_vms_mb as number) ?? null,
          model: result.model ?? null,
          recommended_model: result.recommended_model ?? null,
          system_ram_gb: (result.system_ram_gb as number) ?? null,
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
    } catch (e) {
      console.warn("Socket fetchHealth failed:", e);
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
