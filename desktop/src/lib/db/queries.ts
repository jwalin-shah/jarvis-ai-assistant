/**
 * SQL queries for direct iMessage chat.db access
 * Ported from integrations/imessage/queries.py
 */

/**
 * Schema version for chat.db
 * v14 = macOS 14 and earlier
 * v15 = macOS 15+
 */
export type SchemaVersion = 'v14' | 'v15' | 'unknown';

/**
 * Detect chat.db schema version based on available columns
 */
export const DETECT_SCHEMA_SQL = `
  SELECT name FROM pragma_table_info('chat') WHERE name = 'service_name'
`;

/**
 * Query to get conversations with last message info
 */
export function getConversationsQuery(options: {
  withSinceFilter?: boolean;
  withBeforeFilter?: boolean;
}): string {
  const sinceFilter = options.withSinceFilter ? 'AND tc.last_date > ?' : '';
  const beforeFilter = options.withBeforeFilter ? 'AND tc.last_date < ?' : '';

  // Two-step query: index-only scan on chat_message_join to find top chats,
  // then MAX(message_id) per chat for last message (uses PK index, O(1) per chat).
  // Replaces ROW_NUMBER() which scanned all messages for 50 chats.
  // Performance: ~170ms → ~25ms with 400k messages.
  return `
    WITH chat_stats AS (
      SELECT chat_id,
             MAX(message_date) as last_date,
             COUNT(*) as message_count
      FROM chat_message_join
      GROUP BY chat_id
    ),
    top_chats AS (
      SELECT chat_id, last_date, message_count
      FROM chat_stats
      WHERE message_count > 0
      ${sinceFilter}
      ${beforeFilter}
      ORDER BY last_date DESC
      LIMIT ?
    ),
    chat_participants AS (
      SELECT chj.chat_id,
             GROUP_CONCAT(handle.id, ', ') as participants
      FROM chat_handle_join chj
      INNER JOIN top_chats tc ON chj.chat_id = tc.chat_id
      JOIN handle ON chj.handle_id = handle.ROWID
      GROUP BY chj.chat_id
    ),
    last_msg_ids AS (
      SELECT cmj.chat_id, MAX(cmj.message_id) as last_msg_id
      FROM chat_message_join cmj
      INNER JOIN top_chats tc ON cmj.chat_id = tc.chat_id
      GROUP BY cmj.chat_id
    )
    SELECT
      chat.ROWID as chat_rowid,
      chat.guid as chat_id,
      chat.display_name,
      chat.chat_identifier,
      COALESCE(cp.participants, '') as participants,
      tc.message_count,
      tc.last_date as last_message_date,
      COALESCE(last_m.text, '') as last_message_text,
      last_m.attributedBody as last_message_attributed_body
    FROM top_chats tc
    JOIN chat ON tc.chat_id = chat.ROWID
    LEFT JOIN chat_participants cp ON chat.ROWID = cp.chat_id
    LEFT JOIN last_msg_ids lm ON tc.chat_id = lm.chat_id
    LEFT JOIN message last_m ON lm.last_msg_id = last_m.ROWID
    ORDER BY tc.last_date DESC
  `;
}

/**
 * Query to get messages for a conversation
 */
export function getMessagesQuery(options: { withBeforeFilter?: boolean }): string {
  const beforeFilter = options.withBeforeFilter ? 'AND message.date < ?' : '';

  return `
    SELECT
      message.ROWID as id,
      message.guid,
      chat.guid as chat_id,
      COALESCE(handle.id, 'me') as sender,
      CASE
        WHEN message.text IS NOT NULL AND message.text != ''
        THEN message.text
        ELSE NULL
      END as text,
      message.attributedBody,
      message.date as date,
      message.is_from_me,
      message.thread_originator_guid as reply_to_guid,
      message.date_delivered,
      message.date_read,
      message.group_action_type,
      affected_handle.id as affected_handle_id
    FROM message
    JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
    JOIN chat ON chat_message_join.chat_id = chat.ROWID
    LEFT JOIN handle ON message.handle_id = handle.ROWID
    LEFT JOIN handle AS affected_handle ON message.other_handle = affected_handle.ROWID
    WHERE chat.guid = ?
    ${beforeFilter}
    ORDER BY message.date DESC
    LIMIT ?
  `;
}

/**
 * Query to get messages using chat ROWID directly (skips chat JOIN).
 * Used when chatGuidToRowid cache is populated from getConversations().
 */
export function getMessagesQueryDirect(options: { withBeforeFilter?: boolean }): string {
  const beforeFilter = options.withBeforeFilter ? 'AND message.date < ?' : '';

  return `
    SELECT
      message.ROWID as id,
      message.guid,
      COALESCE(handle.id, 'me') as sender,
      CASE
        WHEN message.text IS NOT NULL AND message.text != ''
        THEN message.text
        ELSE NULL
      END as text,
      message.attributedBody,
      message.date as date,
      message.is_from_me,
      message.thread_originator_guid as reply_to_guid,
      message.date_delivered,
      message.date_read,
      message.group_action_type,
      affected_handle.id as affected_handle_id
    FROM message
    JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
    LEFT JOIN handle ON message.handle_id = handle.ROWID
    LEFT JOIN handle AS affected_handle ON message.other_handle = affected_handle.ROWID
    WHERE chat_message_join.chat_id = ?
    ${beforeFilter}
    ORDER BY message.date DESC
    LIMIT ?
  `;
}

/**
 * Query to get attachments for a message
 */
export const ATTACHMENTS_QUERY = `
  SELECT
    attachment.ROWID as attachment_id,
    attachment.filename,
    attachment.mime_type,
    attachment.total_bytes as file_size,
    attachment.transfer_name
  FROM attachment
  JOIN message_attachment_join ON attachment.ROWID = message_attachment_join.attachment_id
  WHERE message_attachment_join.message_id = ?
`;

/**
 * Query to get reactions for a message
 */
export const REACTIONS_QUERY = `
  SELECT
    message.ROWID as id,
    message.associated_message_type,
    message.date,
    message.is_from_me,
    COALESCE(handle.id, 'me') as sender
  FROM message
  LEFT JOIN handle ON message.handle_id = handle.ROWID
  WHERE message.associated_message_guid = ?
    AND message.associated_message_type != 0
`;

/**
 * Query to get message ROWID by GUID
 */
export const MESSAGE_BY_GUID_QUERY = `
  SELECT message.ROWID as id
  FROM message
  WHERE message.guid = ?
  LIMIT 1
`;

/**
 * Query to get the last message ROWID for detecting new messages
 */
export const LAST_MESSAGE_ROWID_QUERY = `
  SELECT MAX(ROWID) as last_rowid FROM message
`;

/**
 * Query to get new messages since a ROWID
 */
export function getNewMessagesQuery(): string {
  return `
    SELECT
      message.ROWID as id,
      chat.guid as chat_id
    FROM message
    JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
    JOIN chat ON chat_message_join.chat_id = chat.ROWID
    WHERE message.ROWID > ?
    ORDER BY message.ROWID ASC
  `;
}

/**
 * Apple timestamp epoch: 2001-01-01 00:00:00 UTC
 * Difference from Unix epoch (1970-01-01) in seconds
 */
export const APPLE_EPOCH_OFFSET = 978307200;

/**
 * Convert Apple timestamp (nanoseconds since 2001-01-01) to JavaScript Date
 */
export function parseAppleTimestamp(timestamp: number | null): Date | null {
  if (timestamp === null || timestamp === undefined) {
    return null;
  }
  // Apple timestamps are in nanoseconds since 2001-01-01
  // Convert to milliseconds since Unix epoch
  const unixMs = timestamp / 1_000_000 + APPLE_EPOCH_OFFSET * 1000;
  return new Date(unixMs);
}

/**
 * Convert JavaScript Date to Apple timestamp
 */
export function toAppleTimestamp(date: Date): number {
  const unixMs = date.getTime();
  // Convert to nanoseconds since Apple epoch
  return (unixMs - APPLE_EPOCH_OFFSET * 1000) * 1_000_000;
}

/**
 * Format Date to ISO string for API compatibility
 */
export function formatDate(date: Date | null): string | null {
  if (!date) return null;
  return date.toISOString();
}

/**
 * Normalize phone number for consistent lookups
 * Strips formatting, keeps only digits and leading +
 */
export function normalizePhoneNumber(phone: string | null): string | null {
  if (!phone) return null;

  // For email addresses, lowercase
  if (phone.includes('@')) {
    return phone.toLowerCase();
  }

  // Strip everything except digits and leading +
  const hasPlus = phone.startsWith('+');
  const digits = phone.replace(/\D/g, '');

  if (!digits) return null;

  // Match backend normalization: add +1 for US numbers
  if (hasPlus) return `+${digits}`;
  if (digits.startsWith('1') && digits.length === 11) return `+${digits}`;
  if (digits.length === 10) return `+1${digits}`;

  return digits;
}

/**
 * Reaction type mapping from associated_message_type
 */
export const REACTION_TYPES: Record<number, string> = {
  2000: 'love',
  2001: 'like',
  2002: 'dislike',
  2003: 'laugh',
  2004: 'emphasis',
  2005: 'question',
  3000: 'remove_love',
  3001: 'remove_like',
  3002: 'remove_dislike',
  3003: 'remove_laugh',
  3004: 'remove_emphasis',
  3005: 'remove_question',
};

/**
 * Parse reaction type from associated_message_type
 */
export function parseReactionType(type: number): string | null {
  return REACTION_TYPES[type] || null;
}

/**
 * Parse attributedBody blob to extract text.
 *
 * attributedBody is a serialized NSAttributedString in one of two formats:
 * 1. NSKeyedArchive (binary plist / bplist00) - newer format
 * 2. Typedstream (legacy NSArchiver) - starts with "streamtyped"
 *
 * This implements a proper bplist parser + typedstream extractor,
 * mirroring the Python logic in integrations/imessage/parser.py.
 */

/** Metadata strings to skip when scanning NSKeyedArchive $objects */
const NS_METADATA_STRINGS = new Set([
  'NSMutableAttributedString',
  'NSAttributedString',
  'NSMutableString',
  'NSString',
  'NSDictionary',
  'NSMutableDictionary',
  'NSArray',
  'NSMutableArray',
  'NSNumber',
  'NSValue',
  'NSObject',
  'NSAttributes',
  'NSParagraphStyle',
  'NSFont',
  'NSColor',
  'NSKern',
  'NSOriginalFont',
]);

/** Strings to skip in typedstream fallback scanning */
const NS_SKIP_STRINGS = new Set([
  'streamtyped',
  'NSAttributedString',
  'NSObject',
  'NSString',
  'NSDictionary',
  'NSNumber',
  'NSValue',
  'NSArray',
  'NSMutableAttributedString',
  'NSMutableString',
  '__kIMMessagePartAttributeName',
  '__kIMFileTransferGUIDAttributeName',
  '__kIMDataDetectedAttributeName',
]);

// -- bplist00 parser helpers --

/** Read a big-endian unsigned integer */
function readBE(bytes: Uint8Array, offset: number, size: number): number {
  let value = 0;
  for (let i = 0; i < size; i++) {
    value = value * 256 + bytes[offset + i]!;
  }
  return value;
}

/** Read object length, handling extended-length encoding (info nibble = 0xF) */
function readObjectLength(bytes: Uint8Array, offset: number, info: number): [number, number] {
  if (info !== 0x0f) {
    return [info, offset + 1];
  }
  const intMarker = bytes[offset + 1]!;
  const intSize = 1 << (intMarker & 0x0f);
  const length = readBE(bytes, offset + 2, intSize);
  return [length, offset + 2 + intSize];
}

interface BplistArray extends Array<BplistValue> {}
interface BplistDict {
  [key: string]: BplistValue;
}
type BplistValue = null | boolean | number | string | Uint8Array | BplistArray | BplistDict;

interface BplistCtx {
  bytes: Uint8Array;
  objRefSize: number;
  numObjects: number;
  offsets: number[];
  cache: Map<number, BplistValue>;
}

/** Parse a single object from a bplist by its index in the offset table */
function parseBplistObject(ctx: BplistCtx, index: number): BplistValue {
  if (index >= ctx.numObjects) return null;
  if (ctx.cache.has(index)) return ctx.cache.get(index)!;

  ctx.cache.set(index, null); // prevent infinite recursion

  const off = ctx.offsets[index]!;
  if (off >= ctx.bytes.length) return null;

  const marker = ctx.bytes[off]!;
  const type = (marker & 0xf0) >> 4;
  const info = marker & 0x0f;
  let result: BplistValue = null;

  switch (type) {
    case 0x0: // null, bool, fill
      if (marker === 0x08) result = false;
      else if (marker === 0x09) result = true;
      break;

    case 0x1: {
      // integer
      const size = 1 << info;
      result = readBE(ctx.bytes, off + 1, size);
      break;
    }

    case 0x5: {
      // ASCII string
      const [len, start] = readObjectLength(ctx.bytes, off, info);
      try {
        result = new TextDecoder('ascii').decode(ctx.bytes.slice(start, start + len));
      } catch {
        /* skip malformed */
      }
      break;
    }

    case 0x6: {
      // UTF-16BE string (length = character count)
      const [len, start] = readObjectLength(ctx.bytes, off, info);
      try {
        result = new TextDecoder('utf-16be').decode(ctx.bytes.slice(start, start + len * 2));
      } catch {
        /* skip malformed */
      }
      break;
    }

    case 0x4: {
      // binary data
      const [len, start] = readObjectLength(ctx.bytes, off, info);
      result = ctx.bytes.slice(start, start + len);
      break;
    }

    case 0xa:
    case 0xc: {
      // array or set
      const [count, start] = readObjectLength(ctx.bytes, off, info);
      const arr: BplistValue[] = [];
      for (let i = 0; i < count; i++) {
        const ref = readBE(ctx.bytes, start + i * ctx.objRefSize, ctx.objRefSize);
        arr.push(parseBplistObject(ctx, ref));
      }
      result = arr;
      break;
    }

    case 0xd: {
      // dictionary
      const [count, start] = readObjectLength(ctx.bytes, off, info);
      const dict: Record<string, BplistValue> = {};
      for (let i = 0; i < count; i++) {
        const keyRef = readBE(ctx.bytes, start + i * ctx.objRefSize, ctx.objRefSize);
        const valRef = readBE(ctx.bytes, start + (count + i) * ctx.objRefSize, ctx.objRefSize);
        const key = parseBplistObject(ctx, keyRef);
        if (typeof key === 'string') {
          dict[key] = parseBplistObject(ctx, valRef);
        }
      }
      result = dict;
      break;
    }
  }

  ctx.cache.set(index, result);
  return result;
}

/** Extract message text from a bplist00 (NSKeyedArchive) blob */
function extractTextFromBplist(data: Uint8Array): string | null {
  if (data.length < 40) return null;

  // Verify "bplist00" header
  if (
    data[0] !== 0x62 ||
    data[1] !== 0x70 ||
    data[2] !== 0x6c ||
    data[3] !== 0x69 ||
    data[4] !== 0x73 ||
    data[5] !== 0x74 ||
    data[6] !== 0x30 ||
    data[7] !== 0x30
  ) {
    return null;
  }

  // Trailer: last 32 bytes
  const t = data.length - 32;
  const offsetSize = data[t + 6]!;
  const objRefSize = data[t + 7]!;
  const numObjects = readBE(data, t + 8, 8);
  const topObject = readBE(data, t + 16, 8);
  const offsetTableOffset = readBE(data, t + 24, 8);

  if (numObjects === 0 || numObjects > 100000) return null;

  const offsets: number[] = [];
  for (let i = 0; i < numObjects; i++) {
    offsets.push(readBE(data, offsetTableOffset + i * offsetSize, offsetSize));
  }

  const ctx: BplistCtx = {
    bytes: data,
    objRefSize,
    numObjects,
    offsets,
    cache: new Map(),
  };

  const root = parseBplistObject(ctx, topObject);
  if (!root || typeof root !== 'object' || Array.isArray(root)) return null;

  const objects = (root as Record<string, BplistValue>)['$objects'];
  if (!Array.isArray(objects)) return null;

  // Return first non-metadata string (mirrors Python parser logic)
  for (const obj of objects) {
    if (typeof obj === 'string' && obj.length > 0) {
      if (obj.startsWith('$')) continue;
      if (NS_METADATA_STRINGS.has(obj)) continue;
      return obj;
    }
    // Check dict objects for NS.string / NSString keys
    if (obj && typeof obj === 'object' && !Array.isArray(obj)) {
      const d = obj as Record<string, BplistValue>;
      for (const key of ['NS.string', 'NSString']) {
        if (typeof d[key] === 'string') return d[key] as string;
      }
    }
  }

  return null;
}

// -- Typedstream helpers --

/** Find a byte sequence (ASCII needle) in a Uint8Array */
function findBytes(haystack: Uint8Array, needle: string, start = 0, end?: number): number {
  const searchEnd = end ?? haystack.length;
  const needleBytes = new TextEncoder().encode(needle);
  outer: for (let i = start; i <= searchEnd - needleBytes.length; i++) {
    for (let j = 0; j < needleBytes.length; j++) {
      if (haystack[i + j] !== needleBytes[j]) continue outer;
    }
    return i;
  }
  return -1;
}

/** Extract text from typedstream format (legacy NSArchiver) */
function extractFromTypedstream(data: Uint8Array): string | null {
  // Strategy 1: Find NSString marker, then read length-prefixed string after "+"
  const nsIdx = findBytes(data, 'NSString');
  if (nsIdx !== -1) {
    const searchStart = nsIdx + 8; // skip past "NSString"
    const searchEnd = Math.min(searchStart + 20, data.length);
    for (let i = searchStart; i < searchEnd; i++) {
      if (data[i] === 0x2b) {
        // "+" byte precedes length
        const lengthPos = i + 1;
        if (lengthPos >= data.length) break;
        const length = data[lengthPos]!;
        const textStart = lengthPos + 1;
        const textEnd = textStart + length;
        if (textEnd <= data.length && length > 0) {
          try {
            const text = new TextDecoder('utf-8').decode(data.slice(textStart, textEnd));
            if (text.trim()) return text.trim();
          } catch {
            /* fall through */
          }
        }
      }
    }
  }

  // Strategy 2: Scan for longest printable text that isn't metadata
  const decoded = new TextDecoder('utf-8', { fatal: false }).decode(data);
  const printablePattern = /[\x20-\x7e\u00a0-\uffff]+/g;
  let match;
  while ((match = printablePattern.exec(decoded)) !== null) {
    const clean = match[0].trim();
    if (!clean || clean.length < 2) continue;
    if (clean.startsWith('$')) continue;
    if (NS_SKIP_STRINGS.has(clean)) continue;
    if (clean.includes('NS') || clean.includes('kIM') || clean.includes('Attribute')) {
      continue;
    }
    return clean;
  }

  return null;
}

export function parseAttributedBody(data: ArrayBuffer | null): string | null {
  if (!data) return null;

  try {
    const bytes = new Uint8Array(data);
    if (bytes.length < 8) return null;

    // Check for typedstream format ("streamtyped" near start)
    if (findBytes(bytes, 'streamtyped', 0, 20) !== -1) {
      const result = extractFromTypedstream(bytes);
      if (result) return result;
    }

    // Try bplist00 (NSKeyedArchive) format
    const result = extractTextFromBplist(bytes);
    if (result) return result;

    return null;
  } catch {
    return null;
  }
}
