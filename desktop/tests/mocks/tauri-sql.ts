/**
 * Mock for @tauri-apps/plugin-sql when running E2E tests in browser
 *
 * In browser tests, direct SQLite access is not available.
 * The app should fall back to HTTP API when this fails.
 */

export interface QueryResult {
  rowsAffected: number;
  lastInsertId: number;
}

class MockDatabase {
  async select<T>(_query: string, _params?: unknown[]): Promise<T[]> {
    // Direct DB access not available in browser - throw to trigger API fallback
    throw new Error('Direct database access not available in browser context');
  }

  async execute(_query: string, _params?: unknown[]): Promise<QueryResult> {
    throw new Error('Direct database access not available in browser context');
  }

  async close(): Promise<void> {
    // No-op
  }
}

/**
 * Mock Database class - load always fails in browser context
 */
const Database = {
  async load(_path: string): Promise<MockDatabase> {
    throw new Error('SQLite not available in browser context');
  },
};

export default Database;
