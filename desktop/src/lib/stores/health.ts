import { writable, derived } from "svelte/store";

const API_URL = "http://localhost:8742";

export interface HealthStatus {
  status: "healthy" | "degraded" | "unhealthy";
  imessage_access: boolean;
  memory_available_gb: number;
  memory_used_gb: number;
  memory_mode: "FULL" | "LITE" | "MINIMAL";
  model_loaded: boolean;
  permissions_ok: boolean;
  details: Record<string, string> | null;
  jarvis_rss_mb: number;
  jarvis_vms_mb: number;
}

export interface ModelStatus {
  state: "unloaded" | "loading" | "loaded" | "error";
  progress: number | null;
  message: string | null;
  memory_usage_mb: number | null;
  load_time_seconds: number | null;
  error: string | null;
}

// Connection state
export const apiConnected = writable<boolean>(false);
export const apiError = writable<string | null>(null);

// Health status
export const healthStatus = writable<HealthStatus | null>(null);

// Model status
export const modelStatus = writable<ModelStatus>({
  state: "unloaded",
  progress: null,
  message: null,
  memory_usage_mb: null,
  load_time_seconds: null,
  error: null,
});

// Derived store for model ready state
export const modelReady = derived(modelStatus, ($status) => $status.state === "loaded");

// Check API connection
export async function checkApiConnection(): Promise<boolean> {
  try {
    const response = await fetch(`${API_URL}/`);
    if (response.ok) {
      apiConnected.set(true);
      apiError.set(null);
      // Also fetch health status
      await fetchHealthStatus();
      return true;
    }
    throw new Error("API returned error");
  } catch (e) {
    apiConnected.set(false);
    apiError.set(e instanceof Error ? e.message : "Connection failed");
    return false;
  }
}

// Fetch health status
export async function fetchHealthStatus(): Promise<HealthStatus | null> {
  try {
    const response = await fetch(`${API_URL}/health`);
    if (response.ok) {
      const data = await response.json();
      healthStatus.set(data);
      return data;
    }
    return null;
  } catch {
    return null;
  }
}

// Fetch model status
export async function fetchModelStatus(): Promise<ModelStatus | null> {
  try {
    const response = await fetch(`${API_URL}/model-status`);
    if (response.ok) {
      const data = await response.json();
      modelStatus.set(data);
      return data;
    }
    return null;
  } catch {
    return null;
  }
}

// Preload the model
export async function preloadModel(): Promise<boolean> {
  try {
    const response = await fetch(`${API_URL}/model-preload`, {
      method: "POST",
    });
    if (response.ok) {
      // Start polling for status updates
      pollModelStatus();
      return true;
    }
    return false;
  } catch {
    return false;
  }
}

// Unload the model
export async function unloadModel(): Promise<boolean> {
  try {
    const response = await fetch(`${API_URL}/model-unload`, {
      method: "POST",
    });
    if (response.ok) {
      const data = await response.json();
      modelStatus.set({
        state: data.state,
        progress: null,
        message: data.message,
        memory_usage_mb: null,
        load_time_seconds: null,
        error: null,
      });
      return true;
    }
    return false;
  } catch {
    return false;
  }
}

// Poll model status during loading
let pollInterval: ReturnType<typeof setInterval> | null = null;

function pollModelStatus() {
  // Clear any existing poll
  if (pollInterval) {
    clearInterval(pollInterval);
  }

  // Poll every 500ms
  pollInterval = setInterval(async () => {
    const status = await fetchModelStatus();
    if (status && (status.state === "loaded" || status.state === "error" || status.state === "unloaded")) {
      if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
      }
    }
  }, 500);

  // Stop after 60 seconds max
  setTimeout(() => {
    if (pollInterval) {
      clearInterval(pollInterval);
      pollInterval = null;
    }
  }, 60000);
}

// Subscribe to model status via SSE
export function subscribeToModelStatus(
  onUpdate: (status: ModelStatus) => void,
  onComplete: () => void,
  onError: (error: string) => void
): () => void {
  const eventSource = new EventSource(`${API_URL}/model-status/stream`);

  eventSource.onmessage = (event) => {
    try {
      const status: ModelStatus = JSON.parse(event.data);
      modelStatus.set(status);
      onUpdate(status);

      if (status.state === "loaded") {
        eventSource.close();
        onComplete();
      } else if (status.state === "error") {
        eventSource.close();
        onError(status.error || "Unknown error");
      }
    } catch (e) {
      console.error("Failed to parse model status:", e);
    }
  };

  eventSource.onerror = () => {
    eventSource.close();
    onError("Connection lost");
  };

  // Return cleanup function
  return () => eventSource.close();
}
