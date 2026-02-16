/**
 * Unit tests for the API client (src/lib/api/client.ts)
 *
 * The API client implements a triple-fallback strategy:
 *   1. Socket (LLM ops) or Direct DB (data reads)
 *   2. HTTP API
 *
 * Tests verify fallback logic, error handling, URL construction,
 * and response mapping for each code path.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// ---------------------------------------------------------------------------
// Mocks - must be declared before any import that triggers client.ts
// ---------------------------------------------------------------------------

// Mock socket client
vi.mock('../../src/lib/socket', () => ({
  jarvis: {
    connect: vi.fn(),
    on: vi.fn(),
    ping: vi.fn(),
    generateDraft: vi.fn(),
    classifyIntent: vi.fn(),
    summarize: vi.fn(),
    semanticSearch: vi.fn(),
    getSmartReplies: vi.fn(),
    call: vi.fn(),
  },
}));

// Mock direct DB access
vi.mock('../../src/lib/db', () => ({
  isDirectAccessAvailable: vi.fn(() => false),
  getConversations: vi.fn(),
  getMessages: vi.fn(),
}));

// Mock config
vi.mock('../../src/lib/config/runtime', () => ({
  getApiBaseUrl: () => 'http://localhost:8000',
}));

// Mock fetch globally
const mockFetch = vi.fn();
vi.stubGlobal('fetch', mockFetch);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Build a successful fetch Response */
function okResponse(body: unknown): Response {
  return {
    ok: true,
    status: 200,
    statusText: 'OK',
    json: () => Promise.resolve(body),
    headers: new Headers(),
  } as unknown as Response;
}

/** Build a failed fetch Response with parseable JSON error body */
function errorResponse(
  status: number,
  body: { error?: string; detail?: string } = {},
  statusText = 'Error'
): Response {
  return {
    ok: false,
    status,
    statusText,
    json: () => Promise.resolve(body),
    headers: new Headers(),
  } as unknown as Response;
}

/** Build a failed fetch Response whose body is NOT valid JSON */
function errorResponseUnparseable(status: number, statusText = 'Error'): Response {
  return {
    ok: false,
    status,
    statusText,
    json: () => Promise.reject(new SyntaxError('Unexpected token')),
    headers: new Headers(),
  } as unknown as Response;
}

/** Set window.__TAURI__ to simulate Tauri context */
function enableTauri(): void {
  (globalThis as Record<string, unknown>).window = {
    __TAURI__: true,
  };
}

/** Remove Tauri context */
function disableTauri(): void {
  if ('window' in globalThis) {
    delete (globalThis as Record<string, unknown>).window;
  }
}

// ---------------------------------------------------------------------------
// Test suite
// ---------------------------------------------------------------------------

describe('API Client', () => {
  // We need fresh module state per test because client.ts has module-level
  // variables (socketConnected, socketConnectionAttempted, isTauri).
  // vi.resetModules() + dynamic import ensures a clean slate.

  let APIError: typeof import('../../src/lib/api/client').APIError;
  let api: typeof import('../../src/lib/api/client').api;
  let jarvis: typeof import('../../src/lib/socket').jarvis;
  let db: typeof import('../../src/lib/db');

  beforeEach(async () => {
    vi.resetModules();
    mockFetch.mockReset();
    disableTauri();

    // Re-import to pick up clean module state
    const clientMod = await import('../../src/lib/api/client');
    APIError = clientMod.APIError;
    api = clientMod.api;

    const socketMod = await import('../../src/lib/socket');
    jarvis = socketMod.jarvis;

    db = await import('../../src/lib/db');
  });

  afterEach(() => {
    disableTauri();
    vi.restoreAllMocks();
  });

  // =========================================================================
  // 1. APIError class
  // =========================================================================
  describe('APIError', () => {
    it('sets message, status, and detail from constructor args', () => {
      const err = new APIError('Not found', 404, 'Resource missing');
      expect(err.message).toBe('Not found');
      expect(err.status).toBe(404);
      expect(err.detail).toBe('Resource missing');
    });

    it("has name set to 'APIError'", () => {
      const err = new APIError('fail', 500);
      expect(err.name).toBe('APIError');
    });

    it('defaults detail to null when omitted', () => {
      const err = new APIError('bad', 400);
      expect(err.detail).toBeNull();
    });

    it('is an instance of Error', () => {
      const err = new APIError('oops', 500);
      expect(err).toBeInstanceOf(Error);
    });
  });

  // =========================================================================
  // 2. ensureSocketConnected (tested indirectly through API methods)
  // =========================================================================
  describe('ensureSocketConnected', () => {
    it('returns false (falls to HTTP) when not in Tauri context', async () => {
      // isTauri is false because window.__TAURI__ is not set
      mockFetch.mockResolvedValueOnce(okResponse({ status: 'healthy' }));

      const result = await api.ping();
      // Should have gone through HTTP, not socket
      expect(jarvis.ping).not.toHaveBeenCalled();
      expect(mockFetch).toHaveBeenCalledTimes(1);
      expect(result).toEqual({ status: 'healthy' });
    });

    it('returns true after successful connection and uses socket', async () => {
      // Need a fresh import with Tauri enabled
      enableTauri();
      vi.resetModules();

      const clientMod = await import('../../src/lib/api/client');
      api = clientMod.api;
      const socketMod = await import('../../src/lib/socket');
      jarvis = socketMod.jarvis;

      vi.mocked(jarvis.connect).mockResolvedValueOnce(true);
      vi.mocked(jarvis.on).mockReturnValue(vi.fn());
      vi.mocked(jarvis.ping).mockResolvedValueOnce({ status: 'ok' });

      const result = await api.ping();
      expect(jarvis.connect).toHaveBeenCalledTimes(1);
      expect(jarvis.ping).toHaveBeenCalledTimes(1);
      expect(result).toEqual({ status: 'ok' });
    });

    it('caches connection - does not re-connect on subsequent calls', async () => {
      enableTauri();
      vi.resetModules();

      const clientMod = await import('../../src/lib/api/client');
      api = clientMod.api;
      const socketMod = await import('../../src/lib/socket');
      jarvis = socketMod.jarvis;

      // Clear call counts from previous tests (mock factory is shared)
      vi.mocked(jarvis.connect).mockClear();
      vi.mocked(jarvis.ping).mockClear();
      vi.mocked(jarvis.on).mockClear();

      vi.mocked(jarvis.connect).mockResolvedValue(true);
      vi.mocked(jarvis.on).mockReturnValue(vi.fn());
      vi.mocked(jarvis.ping).mockResolvedValue({ status: 'ok' });

      await api.ping();
      await api.ping();
      await api.ping();

      // connect should only be called once (cached after first success)
      expect(jarvis.connect).toHaveBeenCalledTimes(1);
    });

    it('returns false on connection failure and falls to HTTP', async () => {
      enableTauri();
      vi.resetModules();

      const clientMod = await import('../../src/lib/api/client');
      api = clientMod.api;
      const socketMod = await import('../../src/lib/socket');
      jarvis = socketMod.jarvis;

      vi.mocked(jarvis.connect).mockRejectedValueOnce(new Error('Connection refused'));
      mockFetch.mockResolvedValueOnce(okResponse({ status: 'healthy' }));

      const result = await api.ping();
      expect(mockFetch).toHaveBeenCalledTimes(1);
      expect(result).toEqual({ status: 'healthy' });
    });

    it('resets socketConnectionAttempted on failure allowing retry', async () => {
      enableTauri();
      vi.resetModules();

      const clientMod = await import('../../src/lib/api/client');
      api = clientMod.api;
      const socketMod = await import('../../src/lib/socket');
      jarvis = socketMod.jarvis;

      // Clear call counts from previous tests (mock factory is shared)
      vi.mocked(jarvis.connect).mockClear();
      vi.mocked(jarvis.ping).mockClear();
      vi.mocked(jarvis.on).mockClear();

      // First attempt fails
      vi.mocked(jarvis.connect).mockRejectedValueOnce(new Error('fail'));
      mockFetch.mockResolvedValueOnce(okResponse({ status: 'healthy' }));
      await api.ping();
      expect(jarvis.connect).toHaveBeenCalledTimes(1);

      // Second attempt succeeds - connect should be called again because
      // socketConnectionAttempted was reset on failure
      vi.mocked(jarvis.connect).mockResolvedValueOnce(true);
      vi.mocked(jarvis.on).mockReturnValue(vi.fn());
      vi.mocked(jarvis.ping).mockResolvedValueOnce({ status: 'ok' });

      const result2 = await api.ping();
      expect(jarvis.connect).toHaveBeenCalledTimes(2);
      expect(jarvis.ping).toHaveBeenCalledTimes(1);
      expect(result2).toEqual({ status: 'ok' });
    });

    it('registers disconnect/connected event handlers on successful connect', async () => {
      enableTauri();
      vi.resetModules();

      const clientMod = await import('../../src/lib/api/client');
      api = clientMod.api;
      const socketMod = await import('../../src/lib/socket');
      jarvis = socketMod.jarvis;

      vi.mocked(jarvis.connect).mockResolvedValueOnce(true);
      vi.mocked(jarvis.on).mockReturnValue(vi.fn());
      vi.mocked(jarvis.ping).mockResolvedValueOnce({ status: 'ok' });

      await api.ping();

      // Should have registered both "disconnected" and "connected" handlers
      const onCalls = vi.mocked(jarvis.on).mock.calls;
      const eventNames = onCalls.map((c) => c[0]);
      expect(eventNames).toContain('disconnected');
      expect(eventNames).toContain('connected');
    });
  });

  // =========================================================================
  // 3. request() base method (tested through getSettings)
  // =========================================================================
  describe('request() base method', () => {
    it('successful GET returns parsed JSON', async () => {
      const settingsData = { theme: 'dark', notifications: true };
      mockFetch.mockResolvedValueOnce(okResponse(settingsData));

      const result = await api.getSettings();
      expect(result).toEqual(settingsData);
    });

    it('sets Content-Type header to application/json', async () => {
      mockFetch.mockResolvedValueOnce(okResponse({}));

      await api.getSettings();

      const [, options] = mockFetch.mock.calls[0];
      expect(options.headers['Content-Type']).toBe('application/json');
    });

    it('non-OK response throws APIError with status and detail', async () => {
      mockFetch.mockResolvedValueOnce(
        errorResponse(403, { error: 'Forbidden', detail: 'No access' })
      );

      await expect(api.getSettings()).rejects.toThrow(
        expect.objectContaining({
          name: 'APIError',
          status: 403,
          detail: 'No access',
          message: 'Forbidden',
        })
      );
    });

    it('non-OK response with unparseable body still throws APIError', async () => {
      mockFetch.mockResolvedValueOnce(errorResponseUnparseable(500, 'Internal Server Error'));

      await expect(api.getSettings()).rejects.toThrow(
        expect.objectContaining({
          name: 'APIError',
          status: 500,
          message: 'Request failed',
          detail: 'Internal Server Error',
        })
      );
    });

    it('constructs correct URL from baseUrl + endpoint', async () => {
      mockFetch.mockResolvedValueOnce(okResponse({}));

      await api.getSettings();

      const [url] = mockFetch.mock.calls[0];
      expect(url).toBe('http://localhost:8000/settings');
    });
  });

  // =========================================================================
  // 4. getConversations - triple fallback
  // =========================================================================
  describe('getConversations - triple fallback', () => {
    it('direct DB available -> returns direct DB result', async () => {
      enableTauri();
      vi.resetModules();

      const clientMod = await import('../../src/lib/api/client');
      api = clientMod.api;
      db = await import('../../src/lib/db');

      const conversations = [{ id: 'c1', display_name: 'Alice' }];
      vi.mocked(db.isDirectAccessAvailable).mockReturnValue(true);
      vi.mocked(db.getConversations).mockResolvedValueOnce(conversations as any);

      const result = await api.getConversations();
      expect(result).toEqual(conversations);
      expect(db.getConversations).toHaveBeenCalledWith(50);
      expect(mockFetch).not.toHaveBeenCalled();
    });

    it('direct DB throws -> falls back to HTTP', async () => {
      enableTauri();
      vi.resetModules();

      const clientMod = await import('../../src/lib/api/client');
      api = clientMod.api;
      db = await import('../../src/lib/db');

      vi.mocked(db.isDirectAccessAvailable).mockReturnValue(true);
      vi.mocked(db.getConversations).mockRejectedValueOnce(new Error('DB locked'));

      const httpConversations = [{ id: 'c2', display_name: 'Bob' }];
      mockFetch.mockResolvedValueOnce(okResponse({ conversations: httpConversations, total: 1 }));

      const result = await api.getConversations();
      expect(result).toEqual(httpConversations);
      expect(mockFetch).toHaveBeenCalledTimes(1);
    });

    it('direct DB not available -> uses HTTP directly', async () => {
      // Not in Tauri context, so direct DB is never checked
      const httpConversations = [{ id: 'c3', display_name: 'Charlie' }];
      mockFetch.mockResolvedValueOnce(okResponse({ conversations: httpConversations, total: 1 }));

      const result = await api.getConversations();
      expect(result).toEqual(httpConversations);
      expect(mockFetch).toHaveBeenCalledTimes(1);
    });

    it('HTTP success -> extracts conversations from response wrapper', async () => {
      const conversations = [
        { id: 'c4', display_name: 'Dave' },
        { id: 'c5', display_name: 'Eve' },
      ];
      mockFetch.mockResolvedValueOnce(okResponse({ conversations, total: 2 }));

      const result = await api.getConversations();
      // Should unwrap from { conversations, total }
      expect(result).toEqual(conversations);
      expect(result).toHaveLength(2);
    });

    it('HTTP failure -> throws APIError', async () => {
      mockFetch.mockResolvedValueOnce(
        errorResponse(500, { error: 'Server down', detail: 'DB connection lost' })
      );

      await expect(api.getConversations()).rejects.toThrow(
        expect.objectContaining({
          name: 'APIError',
          status: 500,
        })
      );
    });
  });

  // =========================================================================
  // 5. getMessages - triple fallback
  // =========================================================================
  describe('getMessages - triple fallback', () => {
    it('direct DB available -> returns direct DB result', async () => {
      enableTauri();
      vi.resetModules();

      const clientMod = await import('../../src/lib/api/client');
      api = clientMod.api;
      db = await import('../../src/lib/db');

      const messages = [{ id: 1, text: 'Hello' }];
      vi.mocked(db.isDirectAccessAvailable).mockReturnValue(true);
      vi.mocked(db.getMessages).mockResolvedValueOnce(messages as any);

      const result = await api.getMessages('chat123', 25);
      expect(result).toEqual(messages);
      expect(db.getMessages).toHaveBeenCalledWith('chat123', 25, undefined);
      expect(mockFetch).not.toHaveBeenCalled();
    });

    it('direct DB throws -> falls back to HTTP', async () => {
      enableTauri();
      vi.resetModules();

      const clientMod = await import('../../src/lib/api/client');
      api = clientMod.api;
      db = await import('../../src/lib/db');

      vi.mocked(db.isDirectAccessAvailable).mockReturnValue(true);
      vi.mocked(db.getMessages).mockRejectedValueOnce(new Error('DB error'));

      const httpMessages = [{ id: 2, text: 'World' }];
      mockFetch.mockResolvedValueOnce(
        okResponse({ messages: httpMessages, chat_id: 'chat123', total: 1 })
      );

      const result = await api.getMessages('chat123');
      expect(result).toEqual(httpMessages);
      expect(mockFetch).toHaveBeenCalledTimes(1);
    });

    it('HTTP: constructs correct URL with limit param', async () => {
      mockFetch.mockResolvedValueOnce(okResponse({ messages: [], chat_id: 'chat-abc', total: 0 }));

      await api.getMessages('chat-abc', 30);

      const [url] = mockFetch.mock.calls[0];
      expect(url).toBe('http://localhost:8000/conversations/chat-abc/messages?limit=30');
    });

    it('HTTP: includes before param when provided (properly encoded)', async () => {
      mockFetch.mockResolvedValueOnce(okResponse({ messages: [], chat_id: 'c1', total: 0 }));

      await api.getMessages('c1', 50, '2024-01-15T10:30:00Z');

      const [url] = mockFetch.mock.calls[0];
      expect(url).toContain('before=2024-01-15T10%3A30%3A00Z');
    });

    it('HTTP: extracts messages from response wrapper', async () => {
      const messages = [
        { id: 10, text: 'Hi' },
        { id: 11, text: 'Hey' },
      ];
      mockFetch.mockResolvedValueOnce(okResponse({ messages, chat_id: 'c1', total: 2 }));

      const result = await api.getMessages('c1');
      expect(result).toEqual(messages);
    });
  });

  // =========================================================================
  // 6. ping - socket fallback
  // =========================================================================
  describe('ping - socket fallback', () => {
    it('socket connected -> uses socket ping', async () => {
      enableTauri();
      vi.resetModules();

      const clientMod = await import('../../src/lib/api/client');
      api = clientMod.api;
      const socketMod = await import('../../src/lib/socket');
      jarvis = socketMod.jarvis;

      vi.mocked(jarvis.connect).mockResolvedValueOnce(true);
      vi.mocked(jarvis.on).mockReturnValue(vi.fn());
      vi.mocked(jarvis.ping).mockResolvedValueOnce({
        status: 'ok',
        models_ready: true,
      } as any);

      const result = await api.ping();
      expect(result).toEqual({ status: 'ok' });
      expect(jarvis.ping).toHaveBeenCalled();
      expect(mockFetch).not.toHaveBeenCalled();
    });

    it('socket ping fails -> falls back to HTTP health endpoint', async () => {
      enableTauri();
      vi.resetModules();

      const clientMod = await import('../../src/lib/api/client');
      api = clientMod.api;
      const socketMod = await import('../../src/lib/socket');
      jarvis = socketMod.jarvis;

      vi.mocked(jarvis.connect).mockResolvedValueOnce(true);
      vi.mocked(jarvis.on).mockReturnValue(vi.fn());
      vi.mocked(jarvis.ping).mockRejectedValueOnce(new Error('Socket error'));

      mockFetch.mockResolvedValueOnce(okResponse({ status: 'healthy' }));

      const result = await api.ping();
      expect(result).toEqual({ status: 'healthy' });
      expect(mockFetch).toHaveBeenCalledTimes(1);
    });

    it('socket not available -> uses HTTP directly', async () => {
      // Not in Tauri context
      mockFetch.mockResolvedValueOnce(okResponse({ status: 'healthy' }));

      const result = await api.ping();
      expect(result).toEqual({ status: 'healthy' });
      expect(mockFetch).toHaveBeenCalledTimes(1);
    });

    it('returns { status: string } in all cases', async () => {
      mockFetch.mockResolvedValueOnce(okResponse({ status: 'healthy', extra_field: 'ignored' }));

      const result = await api.ping();
      expect(result).toHaveProperty('status');
      expect(typeof result.status).toBe('string');
    });
  });

  // =========================================================================
  // 7. getHealth - socket fallback
  // =========================================================================
  describe('getHealth - socket fallback', () => {
    it('socket connected -> maps socket ping response to HealthResponse', async () => {
      enableTauri();
      vi.resetModules();

      const clientMod = await import('../../src/lib/api/client');
      api = clientMod.api;
      const socketMod = await import('../../src/lib/socket');
      jarvis = socketMod.jarvis;

      vi.mocked(jarvis.connect).mockResolvedValueOnce(true);
      vi.mocked(jarvis.on).mockReturnValue(vi.fn());
      vi.mocked(jarvis.ping).mockResolvedValueOnce({
        status: 'ok',
        models_ready: true,
      } as any);

      const result = await api.getHealth();
      expect(result.status).toBe('healthy');
      expect(result.model_loaded).toBe(true);
      // These are null when coming from socket
      expect(result.imessage_access).toBeNull();
      expect(result.memory_available_gb).toBeNull();
    });

    it('socket fails -> falls through to HTTP', async () => {
      enableTauri();
      vi.resetModules();

      const clientMod = await import('../../src/lib/api/client');
      api = clientMod.api;
      const socketMod = await import('../../src/lib/socket');
      jarvis = socketMod.jarvis;

      vi.mocked(jarvis.connect).mockResolvedValueOnce(true);
      vi.mocked(jarvis.on).mockReturnValue(vi.fn());
      vi.mocked(jarvis.ping).mockRejectedValueOnce(new Error('timeout'));

      const healthData = {
        status: 'healthy',
        imessage_access: true,
        memory_available_gb: 4.5,
        memory_used_gb: 3.5,
        memory_mode: 'normal',
        model_loaded: true,
        permissions_ok: true,
        details: 'All good',
        jarvis_rss_mb: 200,
        jarvis_vms_mb: 500,
      };
      mockFetch.mockResolvedValueOnce(okResponse(healthData));

      const result = await api.getHealth();
      expect(result).toEqual(healthData);
    });

    it('HTTP returns full HealthResponse', async () => {
      const healthData = {
        status: 'degraded',
        imessage_access: false,
        memory_available_gb: 2.0,
        memory_used_gb: 6.0,
        memory_mode: 'constrained',
        model_loaded: false,
        permissions_ok: false,
        details: 'Low memory',
        jarvis_rss_mb: 400,
        jarvis_vms_mb: 800,
      };
      mockFetch.mockResolvedValueOnce(okResponse(healthData));

      const result = await api.getHealth();
      expect(result).toEqual(healthData);
      const [url] = mockFetch.mock.calls[0];
      expect(url).toBe('http://localhost:8000/health');
    });
  });

  // =========================================================================
  // 8. getDraftReplies - socket fallback
  // =========================================================================
  describe('getDraftReplies - socket fallback', () => {
    it('socket connected -> sends via socket with correct params', async () => {
      enableTauri();
      vi.resetModules();

      const clientMod = await import('../../src/lib/api/client');
      api = clientMod.api;
      const socketMod = await import('../../src/lib/socket');
      jarvis = socketMod.jarvis;

      vi.mocked(jarvis.connect).mockResolvedValueOnce(true);
      vi.mocked(jarvis.on).mockReturnValue(vi.fn());
      vi.mocked(jarvis.generateDraft).mockResolvedValueOnce({
        suggestions: [{ text: 'Sure!', confidence: 0.9 }],
        context_used: { num_messages: 5, participants: ['Alice'], last_message: 'Hi' },
      });

      await api.getDraftReplies('chat_abc', 'Be casual');

      expect(jarvis.generateDraft).toHaveBeenCalledWith({
        chat_id: 'chat_abc',
        instruction: 'Be casual',
        context_messages: 20,
      });
    });

    it('socket maps response to DraftReplyResponse format', async () => {
      enableTauri();
      vi.resetModules();

      const clientMod = await import('../../src/lib/api/client');
      api = clientMod.api;
      const socketMod = await import('../../src/lib/socket');
      jarvis = socketMod.jarvis;

      vi.mocked(jarvis.connect).mockResolvedValueOnce(true);
      vi.mocked(jarvis.on).mockReturnValue(vi.fn());
      vi.mocked(jarvis.generateDraft).mockResolvedValueOnce({
        suggestions: [
          { text: 'Hello!', confidence: 0.95 },
          { text: 'Hey there!', confidence: 0.8 },
        ],
        context_used: { num_messages: 10, participants: ['Bob'], last_message: "What's up" },
      });

      const result = await api.getDraftReplies('chat_xyz');
      expect(result).toEqual({
        suggestions: [
          { text: 'Hello!', confidence: 0.95 },
          { text: 'Hey there!', confidence: 0.8 },
        ],
        context_used: { num_messages: 10, participants: ['Bob'], last_message: "What's up" },
      });
    });

    it('socket fails -> falls back to HTTP POST', async () => {
      enableTauri();
      vi.resetModules();

      const clientMod = await import('../../src/lib/api/client');
      api = clientMod.api;
      const socketMod = await import('../../src/lib/socket');
      jarvis = socketMod.jarvis;

      vi.mocked(jarvis.connect).mockResolvedValueOnce(true);
      vi.mocked(jarvis.on).mockReturnValue(vi.fn());
      vi.mocked(jarvis.generateDraft).mockRejectedValueOnce(new Error('Generation failed'));

      const httpResponse = {
        suggestions: [{ text: 'HTTP reply', confidence: 0.7 }],
        context_used: {},
      };
      mockFetch.mockResolvedValueOnce(okResponse(httpResponse));

      const result = await api.getDraftReplies('chat_abc', 'Be brief', 3);

      expect(mockFetch).toHaveBeenCalledTimes(1);
      const [url, options] = mockFetch.mock.calls[0];
      expect(url).toBe('http://localhost:8000/drafts/reply');
      expect(options.method).toBe('POST');

      const body = JSON.parse(options.body);
      expect(body).toEqual({
        chat_id: 'chat_abc',
        instruction: 'Be brief',
        num_suggestions: 3,
        context_messages: 20,
      });

      expect(result).toEqual(httpResponse);
    });

    it('HTTP: constructs correct URL and body', async () => {
      mockFetch.mockResolvedValueOnce(okResponse({ suggestions: [], context_used: {} }));

      await api.getDraftReplies('my-chat', undefined, 5);

      const [url, options] = mockFetch.mock.calls[0];
      expect(url).toBe('http://localhost:8000/drafts/reply');

      const body = JSON.parse(options.body);
      expect(body.chat_id).toBe('my-chat');
      expect(body.instruction).toBeNull(); // undefined maps to null
      expect(body.num_suggestions).toBe(5);
      expect(body.context_messages).toBe(20);
    });
  });

  // =========================================================================
  // 9. semanticSearch - socket fallback
  // =========================================================================
  describe('semanticSearch - socket fallback', () => {
    it('socket connected -> sends via socket', async () => {
      enableTauri();
      vi.resetModules();

      const clientMod = await import('../../src/lib/api/client');
      api = clientMod.api;
      const socketMod = await import('../../src/lib/socket');
      jarvis = socketMod.jarvis;

      vi.mocked(jarvis.connect).mockResolvedValueOnce(true);
      vi.mocked(jarvis.on).mockReturnValue(vi.fn());
      vi.mocked(jarvis.semanticSearch).mockResolvedValueOnce({
        results: [],
        total_results: 0,
      });

      await api.semanticSearch('weekend plans', { limit: 10, threshold: 0.5 });

      expect(jarvis.semanticSearch).toHaveBeenCalledWith({
        query: 'weekend plans',
        limit: 10,
        threshold: 0.5,
        filters: undefined,
      });
    });

    it('socket maps response correctly (results + threshold)', async () => {
      enableTauri();
      vi.resetModules();

      const clientMod = await import('../../src/lib/api/client');
      api = clientMod.api;
      const socketMod = await import('../../src/lib/socket');
      jarvis = socketMod.jarvis;

      vi.mocked(jarvis.connect).mockResolvedValueOnce(true);
      vi.mocked(jarvis.on).mockReturnValue(vi.fn());
      vi.mocked(jarvis.semanticSearch).mockResolvedValueOnce({
        results: [{ message: { id: 1, text: "Let's go hiking" }, similarity: 0.85 }],
        total_results: 1,
      });

      const result = await api.semanticSearch('outdoor activity');

      expect(result.query).toBe('outdoor activity');
      expect(result.results).toHaveLength(1);
      expect(result.results[0].similarity).toBe(0.85);
      expect(result.threshold_used).toBe(0.3); // default threshold
      expect(result.total_results).toBe(1);
      expect(result.messages_searched).toBe(0);
    });

    it('socket fails -> falls back to HTTP POST', async () => {
      enableTauri();
      vi.resetModules();

      const clientMod = await import('../../src/lib/api/client');
      api = clientMod.api;
      const socketMod = await import('../../src/lib/socket');
      jarvis = socketMod.jarvis;

      vi.mocked(jarvis.connect).mockResolvedValueOnce(true);
      vi.mocked(jarvis.on).mockReturnValue(vi.fn());
      vi.mocked(jarvis.semanticSearch).mockRejectedValueOnce(new Error('timeout'));

      const httpResult = {
        query: 'test',
        results: [],
        total_results: 0,
        threshold_used: 0.3,
        messages_searched: 500,
      };
      mockFetch.mockResolvedValueOnce(okResponse(httpResult));

      const result = await api.semanticSearch('test');

      expect(mockFetch).toHaveBeenCalledTimes(1);
      const [url, options] = mockFetch.mock.calls[0];
      expect(url).toBe('http://localhost:8000/search/semantic');
      expect(options.method).toBe('POST');

      const body = JSON.parse(options.body);
      expect(body.query).toBe('test');
      expect(body.limit).toBe(20); // default
      expect(body.threshold).toBe(0.3); // default

      expect(result).toEqual(httpResult);
    });

    it('passes AbortSignal to HTTP request', async () => {
      mockFetch.mockResolvedValueOnce(
        okResponse({
          query: 'q',
          results: [],
          total_results: 0,
          threshold_used: 0.3,
          messages_searched: 0,
        })
      );

      const controller = new AbortController();
      await api.semanticSearch('q', {}, controller.signal);

      const [, options] = mockFetch.mock.calls[0];
      expect(options.signal).toBe(controller.signal);
    });
  });

  // =========================================================================
  // 10. searchMessages
  // =========================================================================
  describe('searchMessages', () => {
    it('constructs URLSearchParams correctly from filters', async () => {
      mockFetch.mockResolvedValueOnce(okResponse([]));

      await api.searchMessages(
        'hello',
        {
          sender: 'Alice',
          after: '2024-01-01',
          before: '2024-06-01',
          has_attachments: true,
        },
        25
      );

      const [url] = mockFetch.mock.calls[0];
      expect(url).toContain('q=hello');
      expect(url).toContain('limit=25');
      expect(url).toContain('sender=Alice');
      expect(url).toContain('after=2024-01-01');
      expect(url).toContain('before=2024-06-01');
      expect(url).toContain('has_attachments=true');
    });

    it('optional filters are only included when provided', async () => {
      mockFetch.mockResolvedValueOnce(okResponse([]));

      await api.searchMessages('test query', {});

      const [url] = mockFetch.mock.calls[0];
      expect(url).toContain('q=test+query');
      expect(url).toContain('limit=50'); // default
      expect(url).not.toContain('sender=');
      expect(url).not.toContain('after=');
      expect(url).not.toContain('before=');
      expect(url).not.toContain('has_attachments=');
    });

    it('passes AbortSignal to fetch', async () => {
      mockFetch.mockResolvedValueOnce(okResponse([]));

      const controller = new AbortController();
      await api.searchMessages('query', {}, 50, controller.signal);

      const [, options] = mockFetch.mock.calls[0];
      expect(options.signal).toBe(controller.signal);
    });

    it('throws APIError on failure', async () => {
      mockFetch.mockResolvedValueOnce(
        errorResponse(422, { error: 'Invalid query', detail: 'Query too short' })
      );

      await expect(api.searchMessages('', {})).rejects.toThrow(
        expect.objectContaining({
          name: 'APIError',
          status: 422,
          detail: 'Query too short',
        })
      );
    });
  });

  // =========================================================================
  // 11. URL encoding
  // =========================================================================
  describe('URL encoding', () => {
    it('chatId is properly URI-encoded in URLs', async () => {
      mockFetch.mockResolvedValueOnce(
        okResponse({ messages: [], chat_id: 'chat;/group+123', total: 0 })
      );

      await api.getMessages('chat;/group+123');

      const [url] = mockFetch.mock.calls[0];
      expect(url).toContain(encodeURIComponent('chat;/group+123'));
      // Should NOT contain the raw unencoded value with slashes
      expect(url).not.toMatch(/conversations\/chat;\/group/);
    });

    it('before date is properly URI-encoded', async () => {
      mockFetch.mockResolvedValueOnce(okResponse({ messages: [], chat_id: 'c1', total: 0 }));

      const dateStr = '2024-03-15T14:30:00+05:30';
      await api.getMessages('c1', 50, dateStr);

      const [url] = mockFetch.mock.calls[0];
      expect(url).toContain(`before=${encodeURIComponent(dateStr)}`);
    });

    it('special characters in query/filter params are encoded', async () => {
      mockFetch.mockResolvedValueOnce(okResponse([]));

      await api.searchMessages('hello & goodbye', {
        sender: "Alice O'Brien",
      });

      const [url] = mockFetch.mock.calls[0];
      // URLSearchParams handles encoding
      expect(url).toContain('q=hello+%26+goodbye');
      expect(url).toContain('sender=Alice+O%27Brien');
    });
  });

  // =========================================================================
  // 12. AbortSignal handling
  // =========================================================================
  describe('AbortSignal handling', () => {
    it('signal is passed through to fetch calls', async () => {
      mockFetch.mockResolvedValueOnce(
        okResponse({
          suggestions: [{ text: 'reply', confidence: 0.9 }],
          context_used: {},
        })
      );

      const controller = new AbortController();
      await api.getDraftReplies('chat1', undefined, 3, controller.signal);

      const [, options] = mockFetch.mock.calls[0];
      expect(options.signal).toBe(controller.signal);
    });

    it('aborted request throws appropriately', async () => {
      const controller = new AbortController();
      controller.abort();

      mockFetch.mockRejectedValueOnce(new DOMException('Aborted', 'AbortError'));

      await expect(api.getDraftReplies('chat1', undefined, 3, controller.signal)).rejects.toThrow(
        'Aborted'
      );
    });
  });

  // =========================================================================
  // Additional method coverage
  // =========================================================================
  describe('getSmartReplySuggestions - socket fallback', () => {
    it('socket connected -> sends via socket with correct params', async () => {
      enableTauri();
      vi.resetModules();

      const clientMod = await import('../../src/lib/api/client');
      api = clientMod.api;
      const socketMod = await import('../../src/lib/socket');
      jarvis = socketMod.jarvis;

      vi.mocked(jarvis.connect).mockResolvedValueOnce(true);
      vi.mocked(jarvis.on).mockReturnValue(vi.fn());
      vi.mocked(jarvis.getSmartReplies).mockResolvedValueOnce({
        suggestions: [
          { text: 'Sure!', score: 0.9 },
          { text: 'Sounds good', score: 0.8 },
        ],
      });

      const result = await api.getSmartReplySuggestions('Want to meet?', 2);

      expect(jarvis.getSmartReplies).toHaveBeenCalledWith({
        last_message: 'Want to meet?',
        num_suggestions: 2,
      });
      expect(result.suggestions).toHaveLength(2);
      expect(result.suggestions[0].text).toBe('Sure!');
    });

    it('socket fails -> falls back to HTTP', async () => {
      enableTauri();
      vi.resetModules();

      const clientMod = await import('../../src/lib/api/client');
      api = clientMod.api;
      const socketMod = await import('../../src/lib/socket');
      jarvis = socketMod.jarvis;

      vi.mocked(jarvis.connect).mockResolvedValueOnce(true);
      vi.mocked(jarvis.on).mockReturnValue(vi.fn());
      vi.mocked(jarvis.getSmartReplies).mockRejectedValueOnce(new Error('fail'));

      const httpResult = { suggestions: [{ text: 'OK', score: 0.5 }] };
      mockFetch.mockResolvedValueOnce(okResponse(httpResult));

      const result = await api.getSmartReplySuggestions('Hey');
      expect(mockFetch).toHaveBeenCalledTimes(1);
      expect(result).toEqual(httpResult);
    });
  });

  describe('getSummary - socket fallback', () => {
    it('socket connected -> uses socket summarize', async () => {
      enableTauri();
      vi.resetModules();

      const clientMod = await import('../../src/lib/api/client');
      api = clientMod.api;
      const socketMod = await import('../../src/lib/socket');
      jarvis = socketMod.jarvis;

      vi.mocked(jarvis.connect).mockResolvedValueOnce(true);
      vi.mocked(jarvis.on).mockReturnValue(vi.fn());
      vi.mocked(jarvis.summarize).mockResolvedValueOnce({
        summary: 'Discussed plans',
        key_points: ['Meeting at 3pm'],
        message_count: 30,
      });

      const result = await api.getSummary('chat1', 30);

      expect(jarvis.summarize).toHaveBeenCalledWith('chat1', 30);
      expect(result.summary).toBe('Discussed plans');
      expect(result.key_points).toEqual(['Meeting at 3pm']);
      expect(result.date_range).toEqual({ start: '', end: '' });
      expect(result.message_count).toBe(30);
    });

    it('socket fails -> falls back to HTTP POST', async () => {
      enableTauri();
      vi.resetModules();

      const clientMod = await import('../../src/lib/api/client');
      api = clientMod.api;
      const socketMod = await import('../../src/lib/socket');
      jarvis = socketMod.jarvis;

      vi.mocked(jarvis.connect).mockResolvedValueOnce(true);
      vi.mocked(jarvis.on).mockReturnValue(vi.fn());
      vi.mocked(jarvis.summarize).mockRejectedValueOnce(new Error('model not loaded'));

      const httpResult = {
        summary: 'HTTP summary',
        key_points: [],
        date_range: { start: '2024-01-01', end: '2024-01-31' },
      };
      mockFetch.mockResolvedValueOnce(okResponse(httpResult));

      const result = await api.getSummary('chat2', 50);

      expect(mockFetch).toHaveBeenCalledTimes(1);
      const [url, options] = mockFetch.mock.calls[0];
      expect(url).toBe('http://localhost:8000/drafts/summarize');
      expect(options.method).toBe('POST');

      const body = JSON.parse(options.body);
      expect(body).toEqual({ chat_id: 'chat2', num_messages: 50 });

      // message_count is appended by client
      expect(result.message_count).toBe(50);
    });
  });

  describe('getConversation (HTTP only)', () => {
    it('encodes chatId and returns Conversation', async () => {
      const conv = { id: 'chat;special/id', display_name: 'Test' };
      mockFetch.mockResolvedValueOnce(okResponse(conv));

      const result = await api.getConversation('chat;special/id');

      const [url] = mockFetch.mock.calls[0];
      expect(url).toBe(
        `http://localhost:8000/conversations/${encodeURIComponent('chat;special/id')}`
      );
      expect(result).toEqual(conv);
    });
  });

  describe('updateSettings', () => {
    it('sends PUT with JSON body', async () => {
      mockFetch.mockResolvedValueOnce(okResponse({ theme: 'light' }));

      await api.updateSettings({ theme: 'light' } as any);

      const [url, options] = mockFetch.mock.calls[0];
      expect(url).toBe('http://localhost:8000/settings');
      expect(options.method).toBe('PUT');
      expect(JSON.parse(options.body)).toEqual({ theme: 'light' });
    });
  });

  describe('getHealth socket status mapping', () => {
    it("maps 'ok' status to 'healthy'", async () => {
      enableTauri();
      vi.resetModules();

      const clientMod = await import('../../src/lib/api/client');
      api = clientMod.api;
      const socketMod = await import('../../src/lib/socket');
      jarvis = socketMod.jarvis;

      vi.mocked(jarvis.connect).mockResolvedValueOnce(true);
      vi.mocked(jarvis.on).mockReturnValue(vi.fn());
      vi.mocked(jarvis.ping).mockResolvedValueOnce({ status: 'ok' } as any);

      const result = await api.getHealth();
      expect(result.status).toBe('healthy');
    });

    it("maps non-'ok' status to 'degraded'", async () => {
      enableTauri();
      vi.resetModules();

      const clientMod = await import('../../src/lib/api/client');
      api = clientMod.api;
      const socketMod = await import('../../src/lib/socket');
      jarvis = socketMod.jarvis;

      vi.mocked(jarvis.connect).mockResolvedValueOnce(true);
      vi.mocked(jarvis.on).mockReturnValue(vi.fn());
      vi.mocked(jarvis.ping).mockResolvedValueOnce({ status: 'degraded' } as any);

      const result = await api.getHealth();
      expect(result.status).toBe('degraded');
    });
  });
});
