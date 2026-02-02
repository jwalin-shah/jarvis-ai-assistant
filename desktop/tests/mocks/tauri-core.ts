/**
 * Mock for @tauri-apps/api/core when running E2E tests in browser
 */

/**
 * Mock invoke function - throws to indicate Tauri is not available
 */
export async function invoke(_cmd: string, _args?: unknown): Promise<unknown> {
  throw new Error("Tauri invoke not available in browser context");
}

/**
 * Mock convertFileSrc function
 */
export function convertFileSrc(filePath: string, _protocol?: string): string {
  return filePath;
}

/**
 * Mock transformCallback function
 */
export function transformCallback(
  callback: (response: unknown) => void,
  _once?: boolean
): number {
  // Return a fake callback id
  return Math.random();
}
