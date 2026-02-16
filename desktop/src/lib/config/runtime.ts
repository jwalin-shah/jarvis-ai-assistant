/**
 * Shared runtime endpoint resolution for browser and Tauri builds.
 *
 * Keeping these values in one place prevents browser/desktop drift.
 */

const DEFAULT_API_HTTP = "http://127.0.0.1:8742";
const DEFAULT_API_WS = "ws://127.0.0.1:8742";
const DEFAULT_SOCKET_RPC_WS = "ws://127.0.0.1:8743";

function normalizedOrigin(url: string, fallback: string): string {
  try {
    return new URL(url).origin;
  } catch {
    return fallback;
  }
}

export function getApiBaseUrl(): string {
  return normalizedOrigin(import.meta.env.VITE_API_URL || DEFAULT_API_HTTP, DEFAULT_API_HTTP);
}

export function getApiWebSocketBaseUrl(): string {
  return normalizedOrigin(import.meta.env.VITE_WS_URL || DEFAULT_API_WS, DEFAULT_API_WS);
}

export function getSocketRpcWebSocketUrl(): string {
  return normalizedOrigin(import.meta.env.VITE_SOCKET_WS_URL || DEFAULT_SOCKET_RPC_WS, DEFAULT_SOCKET_RPC_WS);
}

