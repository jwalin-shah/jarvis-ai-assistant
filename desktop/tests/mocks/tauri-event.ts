/**
 * Mock for @tauri-apps/api/event when running E2E tests in browser
 */

export type EventCallback<T> = (event: { payload: T }) => void;
export type UnlistenFn = () => void;

/**
 * Mock listen function - returns a no-op unlisten function
 */
export async function listen<T>(
  _event: string,
  _handler: EventCallback<T>
): Promise<UnlistenFn> {
  // In browser tests, we don't have Tauri events - just return no-op
  return () => {};
}

/**
 * Mock emit function - does nothing in browser
 */
export async function emit(_event: string, _payload?: unknown): Promise<void> {
  // No-op in browser tests
}

/**
 * Mock once function - returns a no-op unlisten function
 */
export async function once<T>(
  _event: string,
  _handler: EventCallback<T>
): Promise<UnlistenFn> {
  return () => {};
}
