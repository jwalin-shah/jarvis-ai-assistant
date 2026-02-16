/**
 * Unit tests for desktop/src/lib/db/direct.ts
 *
 * Tests the data transformation pipeline from raw SQL rows to typed objects,
 * with focus on the attributedBody fallback when row.text is null.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/** Decode base64 to ArrayBuffer (works in Node via Buffer) */
function b64toArrayBuffer(b64: string): ArrayBuffer {
  const buf = Buffer.from(b64, 'base64');
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
}

// "Test" in typedstream format (NSArchiver legacy)
const TYPEDSTREAM_TEST =
  'BAtzdHJlYW10eXBlZIHoA4QBQISEhBJOU0F0dHJpYnV0ZWRTdHJpbmcAhIQITlNPYmplY3QAhZKEhIQITlNTdHJpbmcBlIQBKwRUZXN0hoQCaUkBBJKEhIQMTlNEaWN0aW9uYXJ5AJSEAWkBkoSWlh1fX2tJTU1lc3NhZ2VQYXJ0QXR0cmlidXRlTmFtZYaShISECE5TTnVtYmVyAISEB05TVmFsdWUAlIQBKoSZmQCGhoY=';

// ---------------------------------------------------------------------------
// Mock setup
// ---------------------------------------------------------------------------

// Mock the SQL plugin so we control what the database returns.
// The vitest config already aliases @tauri-apps/plugin-sql to the browser mock,
// but vi.mock overrides that alias with our controllable mock.
const mockSelect = vi.fn();
const mockClose = vi.fn();
const mockLoad = vi.fn();

vi.mock('@tauri-apps/plugin-sql', () => ({
  default: {
    load: (...args: unknown[]) => mockLoad(...args),
  },
}));

// Mock @tauri-apps/api/path for initDatabases
vi.mock('@tauri-apps/api/path', () => ({
  homeDir: vi.fn().mockResolvedValue('/Users/testuser'),
  join: vi.fn((...parts: string[]) => Promise.resolve(parts.join('/'))),
}));

// ---------------------------------------------------------------------------
// Module under test
//
// Because direct.ts has module-level state (chatDb, isInitialized, etc.) and a
// module-level `isTauri` check based on `window.__TAURI__`, we need to control
// the environment carefully. We re-import the module fresh for isolation where
// needed.
// ---------------------------------------------------------------------------

// Set up the window/__TAURI__ global before importing the module
// @ts-expect-error - setting up test environment
globalThis.window = { __TAURI__: true };

// We dynamically import the module after setting up mocks
let directModule: typeof import('../../src/lib/db/direct');

// ---------------------------------------------------------------------------
// Helpers to build mock rows
// ---------------------------------------------------------------------------

function makeConversationRow(overrides: Record<string, unknown> = {}) {
  return {
    chat_rowid: 1,
    chat_id: 'iMessage;-;+15551234567',
    display_name: null,
    chat_identifier: '+15551234567',
    participants: '+15551234567',
    message_count: 42,
    last_message_date: 700000000000000000, // Apple timestamp (nanoseconds)
    last_message_text: 'Hello there',
    last_message_attributed_body: null,
    ...overrides,
  };
}

function makeMessageRow(overrides: Record<string, unknown> = {}) {
  return {
    id: 100,
    guid: 'msg-guid-001',
    chat_id: 'iMessage;-;+15551234567',
    sender: '+15551234567',
    text: 'Hello world',
    attributedBody: null,
    date: 700000000000000000,
    is_from_me: 0,
    reply_to_guid: null,
    date_delivered: null,
    date_read: null,
    group_action_type: 0,
    affected_handle_id: null,
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('direct.ts', () => {
  beforeEach(async () => {
    vi.clearAllMocks();

    // Set up the mock DB instance that Database.load returns
    mockSelect.mockResolvedValue([]);
    mockClose.mockResolvedValue(undefined);
    mockLoad.mockResolvedValue({
      select: mockSelect,
      close: mockClose,
    });

    // Re-import the module to get fresh state.
    // vi.resetModules() clears the module cache so each test starts clean.
    vi.resetModules();

    // Re-apply mocks after module reset
    vi.doMock('@tauri-apps/plugin-sql', () => ({
      default: {
        load: (...args: unknown[]) => mockLoad(...args),
      },
    }));

    vi.doMock('@tauri-apps/api/path', () => ({
      homeDir: vi.fn().mockResolvedValue('/Users/testuser'),
      join: vi.fn((...parts: string[]) => Promise.resolve(parts.join('/'))),
    }));

    directModule = await import('../../src/lib/db/direct');
  });

  afterEach(async () => {
    // Clean up: close databases if initialized
    try {
      await directModule.closeDatabases();
    } catch {
      // Ignore errors during cleanup
    }
  });

  // -------------------------------------------------------------------------
  // isDirectAccessAvailable
  // -------------------------------------------------------------------------
  describe('isDirectAccessAvailable', () => {
    it('returns false before initialization', () => {
      expect(directModule.isDirectAccessAvailable()).toBe(false);
    });

    it('returns true after successful initialization', async () => {
      // detectSchemaVersion query returns v14 (no service_name column)
      mockSelect.mockResolvedValue([]);
      await directModule.initDatabases();
      expect(directModule.isDirectAccessAvailable()).toBe(true);
    });

    it('returns false after closeDatabases', async () => {
      mockSelect.mockResolvedValue([]);
      await directModule.initDatabases();
      expect(directModule.isDirectAccessAvailable()).toBe(true);

      await directModule.closeDatabases();
      expect(directModule.isDirectAccessAvailable()).toBe(false);
    });
  });

  // -------------------------------------------------------------------------
  // getInitError
  // -------------------------------------------------------------------------
  describe('getInitError', () => {
    it('returns null before any init attempt', () => {
      expect(directModule.getInitError()).toBe(null);
    });

    it('returns error when Database.load fails', async () => {
      mockLoad.mockRejectedValue(new Error('DB open failed'));
      await expect(directModule.initDatabases()).rejects.toThrow('DB open failed');
      expect(directModule.getInitError()).toBeInstanceOf(Error);
      expect(directModule.getInitError()!.message).toBe('DB open failed');
    });

    it('provides helpful message for Full Disk Access errors', async () => {
      mockLoad.mockRejectedValue(new Error('error code 14: unable to open database'));
      await expect(directModule.initDatabases()).rejects.toThrow('Full Disk Access');
      expect(directModule.getInitError()!.message).toContain('Full Disk Access');
    });
  });

  // -------------------------------------------------------------------------
  // initDatabases - idempotency
  // -------------------------------------------------------------------------
  describe('initDatabases', () => {
    it('only initializes once on repeated calls', async () => {
      mockSelect.mockResolvedValue([]);
      await directModule.initDatabases();
      await directModule.initDatabases();
      // load should be called only once
      expect(mockLoad).toHaveBeenCalledTimes(1);
    });

    it('allows retry after previous failure', async () => {
      mockLoad.mockRejectedValueOnce(new Error('transient fail'));
      await expect(directModule.initDatabases()).rejects.toThrow('transient fail');

      // Should allow retry - error is not permanently cached
      mockLoad.mockResolvedValue({
        select: mockSelect,
        close: mockClose,
      });
      mockSelect.mockResolvedValue([]);
      await directModule.initDatabases();
      expect(directModule.isDirectAccessAvailable()).toBe(true);
    });
  });

  // -------------------------------------------------------------------------
  // getConversations - data transformation
  // -------------------------------------------------------------------------
  describe('getConversations', () => {
    beforeEach(async () => {
      // Initialize the database first
      mockSelect.mockResolvedValueOnce([]); // detectSchemaVersion
      await directModule.initDatabases();
    });

    it('transforms a basic 1:1 conversation row correctly', async () => {
      const row = makeConversationRow();
      mockSelect.mockResolvedValueOnce([row]);

      const result = await directModule.getConversations(50);

      expect(result).toHaveLength(1);
      const conv = result[0];
      expect(conv.chat_id).toBe('iMessage;-;+15551234567');
      expect(conv.participants).toEqual(['+15551234567']);
      expect(conv.is_group).toBe(false);
      expect(conv.message_count).toBe(42);
      expect(conv.last_message_text).toBe('Hello there');
      // display_name is null (no contact resolution in direct mode)
      expect(conv.display_name).toBe(null);
      // last_message_date should be an ISO string
      expect(conv.last_message_date).toMatch(/^\d{4}-\d{2}-\d{2}T/);
    });

    it('identifies group conversations (multiple participants)', async () => {
      const row = makeConversationRow({
        chat_id: 'iMessage;+;chat123',
        participants: '+15551234567, +15559876543, john@example.com',
        display_name: 'The Gang',
      });
      mockSelect.mockResolvedValueOnce([row]);

      const result = await directModule.getConversations(50);

      expect(result).toHaveLength(1);
      expect(result[0].is_group).toBe(true);
      expect(result[0].participants).toHaveLength(3);
      expect(result[0].display_name).toBe('The Gang');
      // Email should be lowercased by normalizePhoneNumber
      expect(result[0].participants[2]).toBe('john@example.com');
    });

    it('falls back to attributedBody when last_message_text is null (typedstream)', async () => {
      const row = makeConversationRow({
        last_message_text: null,
        last_message_attributed_body: b64toArrayBuffer(TYPEDSTREAM_TEST),
      });
      mockSelect.mockResolvedValueOnce([row]);

      const result = await directModule.getConversations(50);

      expect(result).toHaveLength(1);
      expect(result[0].last_message_text).toBe('Test');
    });

    it('returns null last_message_text when both text and attributedBody are null', async () => {
      const row = makeConversationRow({
        last_message_text: null,
        last_message_attributed_body: null,
      });
      mockSelect.mockResolvedValueOnce([row]);

      const result = await directModule.getConversations(50);

      expect(result).toHaveLength(1);
      expect(result[0].last_message_text).toBe(null);
    });

    it('handles empty result set', async () => {
      mockSelect.mockResolvedValueOnce([]);

      const result = await directModule.getConversations(50);

      expect(result).toEqual([]);
    });

    it('normalizes phone numbers in participants', async () => {
      const row = makeConversationRow({
        participants: '+1 (555) 123-4567',
      });
      mockSelect.mockResolvedValueOnce([row]);

      const result = await directModule.getConversations(50);

      // normalizePhoneNumber strips formatting, keeps + and digits
      expect(result[0].participants[0]).toBe('+15551234567');
    });

    it('handles participants with empty string entries', async () => {
      const row = makeConversationRow({
        participants: '+15551234567, , +15559876543',
      });
      mockSelect.mockResolvedValueOnce([row]);

      const result = await directModule.getConversations(50);

      // Empty entries should be filtered out
      expect(result[0].participants).toHaveLength(2);
    });

    it('throws when database not initialized', async () => {
      // Close to reset state
      await directModule.closeDatabases();

      await expect(directModule.getConversations(50)).rejects.toThrow('Database not initialized');
    });
  });

  // -------------------------------------------------------------------------
  // getMessages - data transformation
  // -------------------------------------------------------------------------
  describe('getMessages', () => {
    beforeEach(async () => {
      // Initialize the database
      mockSelect.mockResolvedValueOnce([]); // detectSchemaVersion
      await directModule.initDatabases();
    });

    it('transforms a basic message row correctly', async () => {
      const row = makeMessageRow();

      // Main messages query
      mockSelect.mockResolvedValueOnce([row]);
      // Attachments query (called per message)
      mockSelect.mockResolvedValueOnce([]);
      // Reactions query (called per message)
      mockSelect.mockResolvedValueOnce([]);

      const result = await directModule.getMessages('iMessage;-;+15551234567');

      expect(result).toHaveLength(1);
      const msg = result[0];
      expect(msg.id).toBe(100);
      expect(msg.chat_id).toBe('iMessage;-;+15551234567');
      expect(msg.sender).toBe('+15551234567');
      expect(msg.text).toBe('Hello world');
      expect(msg.is_from_me).toBe(false);
      expect(msg.is_system_message).toBe(false);
      expect(msg.attachments).toEqual([]);
      expect(msg.reactions).toEqual([]);
      expect(msg.reply_to_id).toBe(null);
      expect(msg.date).toMatch(/^\d{4}-\d{2}-\d{2}T/);
    });

    it('falls back to attributedBody when text is null (typedstream)', async () => {
      const row = makeMessageRow({
        text: null,
        attributedBody: b64toArrayBuffer(TYPEDSTREAM_TEST),
      });

      mockSelect.mockResolvedValueOnce([row]);
      mockSelect.mockResolvedValueOnce([]); // attachments
      mockSelect.mockResolvedValueOnce([]); // reactions

      const result = await directModule.getMessages('iMessage;-;+15551234567');

      expect(result).toHaveLength(1);
      expect(result[0].text).toBe('Test');
    });

    it('falls back to attributedBody when text is empty string', async () => {
      const row = makeMessageRow({
        text: '',
        attributedBody: b64toArrayBuffer(TYPEDSTREAM_TEST),
      });

      mockSelect.mockResolvedValueOnce([row]);
      mockSelect.mockResolvedValueOnce([]); // attachments
      mockSelect.mockResolvedValueOnce([]); // reactions

      const result = await directModule.getMessages('iMessage;-;+15551234567');

      expect(result).toHaveLength(1);
      expect(result[0].text).toBe('Test');
    });

    it('skips messages with no content (no text, no attributedBody, no attachments)', async () => {
      const row = makeMessageRow({
        text: null,
        attributedBody: null,
      });

      mockSelect.mockResolvedValueOnce([row]);
      mockSelect.mockResolvedValueOnce([]); // batch attachments (empty)
      mockSelect.mockResolvedValueOnce([]); // batch reactions

      const result = await directModule.getMessages('iMessage;-;+15551234567');

      // Message should be filtered out
      expect(result).toHaveLength(0);
    });

    it('keeps messages with attachments even if text is empty', async () => {
      const row = makeMessageRow({
        text: null,
        attributedBody: null,
      });

      mockSelect.mockResolvedValueOnce([row]);
      // Batch attachments - includes message_id to link to the message
      mockSelect.mockResolvedValueOnce([
        {
          message_id: 100,
          attachment_id: 1,
          filename: '~/Library/Messages/Attachments/photo.jpg',
          mime_type: 'image/jpeg',
          file_size: 12345,
          transfer_name: 'photo.jpg',
        },
      ]);
      // Batch reactions
      mockSelect.mockResolvedValueOnce([]);

      const result = await directModule.getMessages('iMessage;-;+15551234567');

      expect(result).toHaveLength(1);
      expect(result[0].text).toBe('');
      expect(result[0].attachments).toHaveLength(1);
      expect(result[0].attachments[0].filename).toBe('photo.jpg');
      expect(result[0].attachments[0].mime_type).toBe('image/jpeg');
      expect(result[0].attachments[0].file_size).toBe(12345);
    });

    it('generates system message text for group events', async () => {
      // group_action_type = 1 means "left or removed"
      const row = makeMessageRow({
        group_action_type: 1,
        sender: '+15551234567',
        is_from_me: 0,
        affected_handle_id: null,
      });

      mockSelect.mockResolvedValueOnce([row]);
      mockSelect.mockResolvedValueOnce([]); // batch attachments
      mockSelect.mockResolvedValueOnce([]); // batch reactions

      const result = await directModule.getMessages('iMessage;-;+15551234567');

      expect(result).toHaveLength(1);
      expect(result[0].is_system_message).toBe(true);
      expect(result[0].text).toContain('left the group');
    });

    it("generates 'You' for system messages from me", async () => {
      const row = makeMessageRow({
        group_action_type: 2, // name changed
        is_from_me: 1,
        sender: 'me',
      });

      mockSelect.mockResolvedValueOnce([row]);
      mockSelect.mockResolvedValueOnce([]); // batch attachments
      mockSelect.mockResolvedValueOnce([]); // batch reactions

      const result = await directModule.getMessages('iMessage;-;+15551234567');

      expect(result).toHaveLength(1);
      expect(result[0].is_system_message).toBe(true);
      expect(result[0].text).toContain('You changed the group name');
    });

    it("generates 'added' text for group_action_type 3", async () => {
      const row = makeMessageRow({
        group_action_type: 3,
        sender: '+15551234567',
        is_from_me: 0,
        affected_handle_id: '+15559876543',
      });

      mockSelect.mockResolvedValueOnce([row]);
      mockSelect.mockResolvedValueOnce([]); // batch attachments
      mockSelect.mockResolvedValueOnce([]); // batch reactions

      const result = await directModule.getMessages('iMessage;-;+15551234567');

      expect(result).toHaveLength(1);
      expect(result[0].is_system_message).toBe(true);
      expect(result[0].text).toContain('added');
      expect(result[0].text).toContain('+15559876543');
    });

    it("generates 'removed' text for group_action_type 1 with different affected handle", async () => {
      const row = makeMessageRow({
        group_action_type: 1,
        sender: '+15551234567',
        is_from_me: 0,
        affected_handle_id: '+15559876543',
      });

      mockSelect.mockResolvedValueOnce([row]);
      mockSelect.mockResolvedValueOnce([]); // batch attachments
      mockSelect.mockResolvedValueOnce([]); // batch reactions

      const result = await directModule.getMessages('iMessage;-;+15551234567');

      expect(result).toHaveLength(1);
      expect(result[0].text).toContain('removed');
      expect(result[0].text).toContain('+15559876543');
    });

    it('generates fallback text for unknown group_action_type', async () => {
      const row = makeMessageRow({
        group_action_type: 99,
        is_from_me: 0,
      });

      mockSelect.mockResolvedValueOnce([row]);
      mockSelect.mockResolvedValueOnce([]); // batch attachments
      mockSelect.mockResolvedValueOnce([]); // batch reactions

      const result = await directModule.getMessages('iMessage;-;+15551234567');

      expect(result).toHaveLength(1);
      expect(result[0].text).toBe('Group event (type 99)');
    });

    it('sets is_from_me correctly', async () => {
      const row = makeMessageRow({
        is_from_me: 1,
        sender: 'me',
      });

      mockSelect.mockResolvedValueOnce([row]);
      mockSelect.mockResolvedValueOnce([]); // attachments
      mockSelect.mockResolvedValueOnce([]); // reactions

      const result = await directModule.getMessages('iMessage;-;+15551234567');

      expect(result).toHaveLength(1);
      expect(result[0].is_from_me).toBe(true);
      expect(result[0].sender_name).toBe(null);
    });

    it('includes delivery/read receipts for sent messages', async () => {
      const row = makeMessageRow({
        is_from_me: 1,
        sender: 'me',
        date_delivered: 700001000000000000,
        date_read: 700002000000000000,
      });

      mockSelect.mockResolvedValueOnce([row]);
      mockSelect.mockResolvedValueOnce([]); // attachments
      mockSelect.mockResolvedValueOnce([]); // reactions

      const result = await directModule.getMessages('iMessage;-;+15551234567');

      expect(result).toHaveLength(1);
      expect(result[0].date_delivered).toMatch(/^\d{4}-\d{2}-\d{2}T/);
      expect(result[0].date_read).toMatch(/^\d{4}-\d{2}-\d{2}T/);
    });

    it('does not set delivery/read receipts for received messages', async () => {
      const row = makeMessageRow({
        is_from_me: 0,
        date_delivered: 700001000000000000,
        date_read: 700002000000000000,
      });

      mockSelect.mockResolvedValueOnce([row]);
      mockSelect.mockResolvedValueOnce([]); // attachments
      mockSelect.mockResolvedValueOnce([]); // reactions

      const result = await directModule.getMessages('iMessage;-;+15551234567');

      expect(result).toHaveLength(1);
      // Receipts should be null for received messages (direct.ts only sets them for is_from_me)
      expect(result[0].date_delivered).toBe(null);
      expect(result[0].date_read).toBe(null);
    });

    it('resolves reply_to_guid to ROWID', async () => {
      const row = makeMessageRow({
        reply_to_guid: 'original-msg-guid',
      });

      mockSelect.mockResolvedValueOnce([row]); // main query
      mockSelect.mockResolvedValueOnce([]); // batch attachments
      mockSelect.mockResolvedValueOnce([]); // batch reactions
      // batch GUID→ROWID resolution
      mockSelect.mockResolvedValueOnce([{ guid: 'original-msg-guid', id: 42 }]);

      const result = await directModule.getMessages('iMessage;-;+15551234567');

      expect(result).toHaveLength(1);
      expect(result[0].reply_to_id).toBe(42);
    });

    it('handles multiple messages correctly', async () => {
      const rows = [
        makeMessageRow({ id: 100, text: 'First', guid: 'guid-100' }),
        makeMessageRow({ id: 101, text: 'Second', guid: 'guid-101' }),
        makeMessageRow({ id: 102, text: 'Third', guid: 'guid-102' }),
      ];

      mockSelect.mockResolvedValueOnce(rows);
      mockSelect.mockResolvedValueOnce([]); // batch attachments
      mockSelect.mockResolvedValueOnce([]); // batch reactions

      const result = await directModule.getMessages('iMessage;-;+15551234567');

      expect(result).toHaveLength(3);
      expect(result[0].text).toBe('First');
      expect(result[1].text).toBe('Second');
      expect(result[2].text).toBe('Third');
    });

    it('handles empty result set', async () => {
      mockSelect.mockResolvedValueOnce([]);

      const result = await directModule.getMessages('iMessage;-;+15551234567');

      expect(result).toEqual([]);
    });

    it('throws when database not initialized', async () => {
      await directModule.closeDatabases();

      await expect(directModule.getMessages('iMessage;-;+15551234567')).rejects.toThrow(
        'Database not initialized'
      );
    });
  });

  // -------------------------------------------------------------------------
  // getLastMessageRowid
  // -------------------------------------------------------------------------
  describe('getLastMessageRowid', () => {
    it('returns 0 when database not initialized', async () => {
      const result = await directModule.getLastMessageRowid();
      expect(result).toBe(0);
    });

    it('returns the last ROWID', async () => {
      mockSelect.mockResolvedValueOnce([]); // detectSchemaVersion
      await directModule.initDatabases();

      mockSelect.mockResolvedValueOnce([{ last_rowid: 99999 }]);
      const result = await directModule.getLastMessageRowid();
      expect(result).toBe(99999);
    });
  });

  // -------------------------------------------------------------------------
  // getNewMessagesSince
  // -------------------------------------------------------------------------
  describe('getNewMessagesSince', () => {
    it('returns empty array when database not initialized', async () => {
      const result = await directModule.getNewMessagesSince(100);
      expect(result).toEqual([]);
    });

    it('transforms new message rows to chatId/messageId pairs', async () => {
      mockSelect.mockResolvedValueOnce([]); // detectSchemaVersion
      await directModule.initDatabases();

      mockSelect.mockResolvedValueOnce([
        { id: 101, chat_id: 'iMessage;-;+15551234567' },
        { id: 102, chat_id: 'iMessage;-;+15559876543' },
      ]);

      const result = await directModule.getNewMessagesSince(100);

      expect(result).toHaveLength(2);
      expect(result[0]).toEqual({
        chatId: 'iMessage;-;+15551234567',
        messageId: 101,
      });
      expect(result[1]).toEqual({
        chatId: 'iMessage;-;+15559876543',
        messageId: 102,
      });
    });
  });

  // -------------------------------------------------------------------------
  // Attachment transformation
  // -------------------------------------------------------------------------
  describe('attachment transformation', () => {
    beforeEach(async () => {
      mockSelect.mockResolvedValueOnce([]); // detectSchemaVersion
      await directModule.initDatabases();
    });

    it('maps attachment rows to Attachment objects with transfer_name preferred', async () => {
      const row = makeMessageRow();
      mockSelect.mockResolvedValueOnce([row]);

      // Batch attachments
      mockSelect.mockResolvedValueOnce([
        {
          message_id: 100,
          attachment_id: 1,
          filename: '~/Library/Messages/Attachments/00/ABCDEF/photo.heic',
          mime_type: 'image/heic',
          file_size: 2048000,
          transfer_name: 'vacation.heic',
        },
      ]);
      // Batch reactions
      mockSelect.mockResolvedValueOnce([]);

      const result = await directModule.getMessages('iMessage;-;+15551234567');

      const att = result[0].attachments[0];
      // transfer_name is preferred for display filename
      expect(att.filename).toBe('vacation.heic');
      // file_path gets the full path
      expect(att.file_path).toBe('~/Library/Messages/Attachments/00/ABCDEF/photo.heic');
      expect(att.mime_type).toBe('image/heic');
      expect(att.file_size).toBe(2048000);
    });

    it('falls back to filename when transfer_name is null', async () => {
      const row = makeMessageRow();
      mockSelect.mockResolvedValueOnce([row]);

      mockSelect.mockResolvedValueOnce([
        {
          message_id: 100,
          attachment_id: 1,
          filename: 'document.pdf',
          mime_type: 'application/pdf',
          file_size: 1024,
          transfer_name: null,
        },
      ]);
      mockSelect.mockResolvedValueOnce([]);

      const result = await directModule.getMessages('iMessage;-;+15551234567');

      expect(result[0].attachments[0].filename).toBe('document.pdf');
    });

    it("uses 'attachment' as fallback when both names are null", async () => {
      const row = makeMessageRow();
      mockSelect.mockResolvedValueOnce([row]);

      mockSelect.mockResolvedValueOnce([
        {
          message_id: 100,
          attachment_id: 1,
          filename: null,
          mime_type: null,
          file_size: null,
          transfer_name: null,
        },
      ]);
      mockSelect.mockResolvedValueOnce([]);

      const result = await directModule.getMessages('iMessage;-;+15551234567');

      expect(result[0].attachments[0].filename).toBe('attachment');
    });
  });

  // -------------------------------------------------------------------------
  // Reaction filtering
  // -------------------------------------------------------------------------
  describe('reaction transformation', () => {
    beforeEach(async () => {
      mockSelect.mockResolvedValueOnce([]); // detectSchemaVersion
      await directModule.initDatabases();
    });

    it('maps reaction rows to Reaction objects', async () => {
      const row = makeMessageRow();
      mockSelect.mockResolvedValueOnce([row]); // main query
      mockSelect.mockResolvedValueOnce([]); // batch attachments

      // Batch reactions
      mockSelect.mockResolvedValueOnce([
        {
          message_guid: 'msg-guid-001',
          id: 200,
          associated_message_type: 2000, // love
          date: 700001000000000000,
          is_from_me: 0,
          sender: '+15559876543',
        },
      ]);

      const result = await directModule.getMessages('iMessage;-;+15551234567');

      expect(result[0].reactions).toHaveLength(1);
      expect(result[0].reactions[0].type).toBe('love');
      expect(result[0].reactions[0].sender).toBe('+15559876543');
    });

    it('filters out remove_* reaction types', async () => {
      const row = makeMessageRow();
      mockSelect.mockResolvedValueOnce([row]);
      mockSelect.mockResolvedValueOnce([]); // batch attachments

      mockSelect.mockResolvedValueOnce([
        {
          message_guid: 'msg-guid-001',
          id: 200,
          associated_message_type: 2001, // like
          date: 700001000000000000,
          is_from_me: 0,
          sender: '+15559876543',
        },
        {
          message_guid: 'msg-guid-001',
          id: 201,
          associated_message_type: 3001, // remove_like (should be filtered out)
          date: 700002000000000000,
          is_from_me: 0,
          sender: '+15559876543',
        },
      ]);

      const result = await directModule.getMessages('iMessage;-;+15551234567');

      expect(result[0].reactions).toHaveLength(1);
      expect(result[0].reactions[0].type).toBe('like');
    });

    it('filters out unknown reaction types', async () => {
      const row = makeMessageRow();
      mockSelect.mockResolvedValueOnce([row]);
      mockSelect.mockResolvedValueOnce([]); // batch attachments

      mockSelect.mockResolvedValueOnce([
        {
          message_guid: 'msg-guid-001',
          id: 200,
          associated_message_type: 9999, // unknown type
          date: 700001000000000000,
          is_from_me: 0,
          sender: '+15559876543',
        },
      ]);

      const result = await directModule.getMessages('iMessage;-;+15551234567');

      expect(result[0].reactions).toHaveLength(0);
    });
  });

  // -------------------------------------------------------------------------
  // closeDatabases
  // -------------------------------------------------------------------------
  describe('closeDatabases', () => {
    it('resets all state and calls db.close()', async () => {
      mockSelect.mockResolvedValueOnce([]); // detectSchemaVersion
      await directModule.initDatabases();
      expect(directModule.isDirectAccessAvailable()).toBe(true);

      await directModule.closeDatabases();

      expect(mockClose).toHaveBeenCalledTimes(1);
      expect(directModule.isDirectAccessAvailable()).toBe(false);
      expect(directModule.getInitError()).toBe(null);
    });
  });

  // -------------------------------------------------------------------------
  // getMessage (single)
  // -------------------------------------------------------------------------
  describe('getMessage', () => {
    beforeEach(async () => {
      mockSelect.mockResolvedValueOnce([]); // detectSchemaVersion
      await directModule.initDatabases();
    });

    it('returns null when no matching message found', async () => {
      mockSelect.mockResolvedValueOnce([]);

      const result = await directModule.getMessage('iMessage;-;+15551234567', 999);
      expect(result).toBe(null);
    });

    it('transforms a single message row', async () => {
      const row = makeMessageRow({ id: 42, text: 'Specific message' });
      mockSelect.mockResolvedValueOnce([row]);
      mockSelect.mockResolvedValueOnce([]); // attachments
      mockSelect.mockResolvedValueOnce([]); // reactions

      const result = await directModule.getMessage('iMessage;-;+15551234567', 42);

      expect(result).not.toBe(null);
      expect(result!.id).toBe(42);
      expect(result!.text).toBe('Specific message');
    });
  });
});
