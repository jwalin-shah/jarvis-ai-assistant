/**
 * Direct SQLite access layer for iMessage chat.db
 * Bypasses HTTP API for ~20-100x faster message loading
 */

import type { Conversation, Message, Attachment, Reaction } from "../api/types";
import { LRUCache } from "../utils/lru-cache";
import { Logger } from "../utils/logger";

// Dynamic import for Tauri plugin - only works in Tauri context.
// The plugin's default export is a class with static methods (e.g. Database.load()),
// but @tauri-apps/plugin-sql doesn't export the class type separately.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let Database: any = null;
interface SqlDatabase {
  path: string;
  select<T = unknown>(query: string, bindValues?: unknown[]): Promise<T>;
  close(db?: string): Promise<boolean>;
}

// Check if running in Tauri context
const isTauri = typeof window !== "undefined" && "__TAURI__" in window;
import {
  getConversationsQuery,
  getMessagesQuery,
  getMessagesQueryDirect,
  ATTACHMENTS_QUERY,
  REACTIONS_QUERY,
  MESSAGE_BY_GUID_QUERY,
  LAST_MESSAGE_ROWID_QUERY,
  getNewMessagesQuery,
  parseAppleTimestamp,
  toAppleTimestamp,
  formatDate,
  normalizePhoneNumber,
  parseReactionType,
  parseAttributedBody,
  type SchemaVersion,
} from "./queries";

// Cached home directory path (resolved via Tauri API)
let resolvedHomePath: string | null = null;

// Connection state
let chatDb: SqlDatabase | null = null;
let isInitialized = false;
let initError: Error | null = null;
let schemaVersion: SchemaVersion = "v14";

// Contact name cache (phone/email -> name)
const contactsCache = new Map<string, string>();
let contactsCacheLoaded = false;

// GUID to ROWID cache for reply_to_id resolution
// LRU cache with max 10k entries to prevent unbounded memory growth
const guidToRowidCache = new LRUCache<string, number>(10000);

// Chat GUID -> chat ROWID cache: eliminates JOIN chat in message queries
const chatGuidToRowid = new Map<string, number>();

const logger = new Logger("DirectDB");

/**
 * Initialize database connections
 * Opens chat.db in read-only mode
 */
export async function initDatabases(): Promise<void> {
  if (isInitialized) return;
  // Don't permanently cache errors - allow retry on transient failures
  if (initError) {
    initError = null;
  }

  // Only attempt direct DB access in Tauri context
  if (!isTauri) {
    initError = new Error("Direct database access only available in Tauri context");
    throw initError;
  }

  try {
    // Parallelize dynamic imports and home directory resolution
    const [sqlModule, pathModule] = await Promise.all([
      Database ? Promise.resolve(null) : import("@tauri-apps/plugin-sql"),
      resolvedHomePath ? Promise.resolve(null) : import("@tauri-apps/api/path"),
    ]);
    if (sqlModule) Database = sqlModule.default;
    if (pathModule && !resolvedHomePath) {
      resolvedHomePath = await pathModule.homeDir();
    }

    // Open chat.db in read-only mode via URI
    // tauri-plugin-sql uses sqlite:// protocol
    const { join } = await import("@tauri-apps/api/path");
    if (resolvedHomePath == null) {
      throw new Error("Home directory path not resolved");
    }
    const chatDbPath = await join(resolvedHomePath, "Library", "Messages", "chat.db");
    const uri = `sqlite:${chatDbPath}?mode=ro`;
    try {
      chatDb = await Database.load(uri);
    } catch (dbError: unknown) {
      const msg = dbError instanceof Error ? dbError.message : String(dbError);
      // SQLite error code 14 = SQLITE_CANTOPEN
      if (msg.includes("14") || msg.toLowerCase().includes("unable to open")) {
        throw new Error(
          `Cannot open chat.db at ${chatDbPath}. ` +
          "Grant Full Disk Access to this app in System Settings > Privacy & Security > Full Disk Access."
        );
      }
      throw dbError;
    }

    // Detect schema version
    schemaVersion = await detectSchemaVersion();

    isInitialized = true;
    logger.info(`Initialized with schema ${schemaVersion}`);
  } catch (error) {
    initError = error instanceof Error ? error : new Error(String(error));
    logger.error("Failed to initialize:", initError);
    throw initError;
  }
}

/**
 * Check if direct database access is available
 */
export function isDirectAccessAvailable(): boolean {
  return isInitialized && chatDb !== null && initError === null;
}

/**
 * Get initialization error if any
 */
export function getInitError(): Error | null {
  return initError;
}

/**
 * Close database connections
 */
export async function closeDatabases(): Promise<void> {
  if (chatDb) {
    await chatDb.close(chatDb.path);
    chatDb = null;
  }
  isInitialized = false;
  initError = null;
  contactsCache.clear();
  contactsCacheLoaded = false;
  guidToRowidCache.clear();
  chatGuidToRowid.clear();
}

/**
 * Detect chat.db schema version
 */
async function detectSchemaVersion(): Promise<SchemaVersion> {
  if (!chatDb) return "unknown";

  try {
    const result = await chatDb.select<{ name: string }[]>(
      "SELECT name FROM pragma_table_info('chat') WHERE name = 'service_name'"
    );
    return result.length > 0 ? "v15" : "v14";
  } catch {
    return "unknown";
  }
}

/**
 * Database row types
 */
interface ConversationRow {
  chat_rowid: number;
  chat_id: string;
  display_name: string | null;
  chat_identifier: string;
  participants: string | null;
  message_count: number;
  last_message_date: number | null;
  last_message_text: string | null;
  last_message_attributed_body: ArrayBuffer | null;
}

interface MessageRow {
  id: number;
  guid: string;
  chat_id: string;
  sender: string;
  text: string | null;
  attributedBody: ArrayBuffer | null;
  date: number;
  is_from_me: number;
  reply_to_guid: string | null;
  date_delivered: number | null;
  date_read: number | null;
  group_action_type: number | null;
  affected_handle_id: string | null;
}

interface AttachmentRow {
  attachment_id: number;
  filename: string | null;
  mime_type: string | null;
  file_size: number | null;
  transfer_name: string | null;
}

interface ReactionRow {
  id: number;
  associated_message_type: number;
  date: number;
  is_from_me: number;
  sender: string;
}

/**
 * Get recent conversations
 */
export async function getConversations(
  limit = 50,
  since?: Date,
  before?: Date
): Promise<Conversation[]> {
  if (!chatDb) {
    throw new Error("Database not initialized");
  }

  const query = getConversationsQuery({
    withSinceFilter: Boolean(since),
    withBeforeFilter: Boolean(before),
  });

  // Build parameters array
  const params: (number | string)[] = [];
  if (since) {
    params.push(toAppleTimestamp(since));
  }
  if (before) {
    params.push(toAppleTimestamp(before));
  }
  params.push(limit);

  try {
    const startTime = performance.now();
    logger.debug(`[LATENCY] Starting getConversations, limit=${limit}`);
    const rows = await chatDb.select<ConversationRow[]>(query, params);
    const elapsed = performance.now() - startTime;
    logger.debug(`[LATENCY] getConversations fetched ${rows.length} conversations in ${elapsed.toFixed(1)}ms`);
    if (elapsed > 100) {
      logger.warn(`[LATENCY WARNING] getConversations took ${elapsed.toFixed(1)}ms (threshold: 100ms)`);
    }

    // Populate chat GUID -> ROWID cache to skip JOIN chat in message queries
    for (const row of rows) {
      chatGuidToRowid.set(row.chat_id, row.chat_rowid);
    }

    return rows.map((row: ConversationRow) => {
      // Parse participants
      const participantsStr = row.participants || "";
      const participants = participantsStr
        .split(",")
        .map((p: string) => p.trim())
        .filter(Boolean)
        .map((p: string) => normalizePhoneNumber(p) || p);

      // Determine if group chat
      const isGroup = participants.length > 1;

      // Parse last message date
      const lastMessageDate = parseAppleTimestamp(row.last_message_date);

      // Get display name or resolve from contacts
      let displayName = row.display_name || null;
      if (!displayName && !isGroup && participants.length === 1) {
        const participant = participants[0];
        if (participant !== undefined) {
          displayName = resolveContactName(participant);
        }
      }

      // Get last message text
      let lastMessageText = row.last_message_text || null;
      if (!lastMessageText && row.last_message_attributed_body) {
        lastMessageText = parseAttributedBody(row.last_message_attributed_body);
      }

      return {
        chat_id: row.chat_id,
        participants,
        display_name: displayName,
        last_message_date: formatDate(lastMessageDate) || "",
        message_count: row.message_count,
        is_group: isGroup,
        last_message_text: lastMessageText,
      };
    });
  } catch (error) {
    logger.error("getConversations error:", error);
    throw error;
  }
}

/**
 * Get messages for a conversation
 */
export async function getMessages(
  chatId: string,
  limit = 100,
  before?: Date
): Promise<Message[]> {
  if (!chatDb) {
    throw new Error("Database not initialized");
  }

  // Use ROWID-direct query if cached (skips JOIN chat), else fall back
  const cachedRowid = chatGuidToRowid.get(chatId);
  const query = cachedRowid !== undefined
    ? getMessagesQueryDirect({ withBeforeFilter: Boolean(before) })
    : getMessagesQuery({ withBeforeFilter: Boolean(before) });

  // Build parameters: ROWID (number) or GUID (string)
  const params = [cachedRowid !== undefined ? cachedRowid : chatId];
  if (before) {
    params.push(toAppleTimestamp(before));
  }
  params.push(limit);

  try {
    const startTime = performance.now();
    logger.debug(`[LATENCY] Starting getMessages for chat_id=${chatId}, limit=${limit}, rowid=${cachedRowid ?? 'miss'}`);
    const rows = await chatDb.select<MessageRow[]>(query, params);

    // PERF FIX: Batch prefetch attachments, reactions, reply GUIDs in parallel
    // Before: 3 sequential queries (~60-220ms). After: 3 parallel queries (~30-50ms)
    const messageIds = rows.map((row) => row.id);
    const messageGuids = rows.map((row) => row.guid);
    const validGuids = messageGuids.filter((g) => Boolean(g));

    // Collect uncached reply GUIDs before launching parallel queries
    const replyGuids = rows
      .map((row) => row.reply_to_guid)
      .filter((g): g is string => Boolean(g) && !guidToRowidCache.has(g));
    const uncachedGuids = [...new Set(replyGuids)];

    // Run all 3 batch queries in parallel
    const [attachmentRows, reactionRows, guidRows] = await Promise.all([
      // Batch attachments
      messageIds.length > 0
        ? chatDb.select<(AttachmentRow & { message_id: number })[]>(
            `SELECT
              message_attachment_join.message_id,
              attachment.ROWID as attachment_id,
              attachment.filename,
              attachment.mime_type,
              attachment.total_bytes as file_size,
              attachment.transfer_name
            FROM message_attachment_join
            JOIN attachment ON message_attachment_join.attachment_id = attachment.ROWID
            WHERE message_attachment_join.message_id IN (${messageIds.map(() => "?").join(",")})`,
            messageIds
          )
        : Promise.resolve([]),
      // Batch reactions (no self-join: associated_message_guid IS the target GUID)
      validGuids.length > 0
        ? chatDb.select<(ReactionRow & { message_guid: string })[]>(
            `SELECT
              reaction.associated_message_guid as message_guid,
              reaction.ROWID as id,
              reaction.associated_message_type,
              reaction.date,
              reaction.is_from_me,
              COALESCE(handle.id, 'me') as sender
            FROM message AS reaction
            LEFT JOIN handle ON reaction.handle_id = handle.ROWID
            WHERE reaction.associated_message_guid IN (${validGuids.map(() => "?").join(",")})
              AND reaction.associated_message_type IS NOT NULL`,
            validGuids
          )
        : Promise.resolve([]),
      // Batch reply GUID→ROWID
      uncachedGuids.length > 0
        ? chatDb.select<{ guid: string; id: number }[]>(
            `SELECT guid, ROWID as id FROM message
            WHERE guid IN (${uncachedGuids.map(() => "?").join(",")})`,
            uncachedGuids
          )
        : Promise.resolve([]),
    ]);

    // Build attachments map
    const attachmentsByMessageId = new Map<number, Attachment[]>();
    for (const row of attachmentRows) {
      const msgId = row.message_id;
      if (!attachmentsByMessageId.has(msgId)) {
        attachmentsByMessageId.set(msgId, []);
      }
      attachmentsByMessageId.get(msgId)?.push({
        filename: row.transfer_name || row.filename || "attachment",
        file_path: row.filename,
        mime_type: row.mime_type,
        file_size: row.file_size,
      });
    }

    // Build reactions map
    const reactionsByMessageGuid = new Map<string, Reaction[]>();
    for (const row of reactionRows) {
      const reactionType = parseReactionType(row.associated_message_type);
      if (!reactionType || reactionType.startsWith("remove_")) continue;

      const guid = row.message_guid;
      if (!reactionsByMessageGuid.has(guid)) {
        reactionsByMessageGuid.set(guid, []);
      }
      const sender = normalizePhoneNumber(row.sender) || row.sender;
      reactionsByMessageGuid.get(guid)?.push({
        type: reactionType,
        sender,
        sender_name: row.is_from_me ? null : resolveContactName(sender),
        date: formatDate(parseAppleTimestamp(row.date)) || "",
      });
    }

    // Build reply GUID→ROWID map
    const replyGuidToRowid = new Map<string, number>();
    for (const gr of guidRows) {
      replyGuidToRowid.set(gr.guid, gr.id);
      guidToRowidCache.set(gr.guid, gr.id);
    }
    for (const row of rows) {
      if (row.reply_to_guid) {
        const cached = guidToRowidCache.get(row.reply_to_guid);
        if (cached !== undefined) {
          replyGuidToRowid.set(row.reply_to_guid, cached);
        }
      }
    }

    // Convert rows to messages in parallel (no I/O with prefetched data)
    const results = await Promise.all(
      rows.map((row) =>
        rowToMessage(
          row,
          chatId,
          attachmentsByMessageId.get(row.id) || [],
          reactionsByMessageGuid.get(row.guid) || [],
          row.reply_to_guid ? replyGuidToRowid.get(row.reply_to_guid) ?? null : null
        )
      )
    );
    const messages = results.filter((m): m is Message => m !== null);

    const elapsed = performance.now() - startTime;
    logger.debug(`getMessages loaded ${messages.length} messages in ${elapsed.toFixed(1)}ms`);
    if (elapsed > 100) {
      logger.warn(`[LATENCY WARNING] getMessages took ${elapsed.toFixed(1)}ms (threshold: 100ms) - possible N+1 pattern`);
    }

    return messages;
  } catch (error) {
    logger.error("getMessages error:", error);
    throw error;
  }
}

/**
 * Get a single message by ID
 */
export async function getMessage(
  chatId: string,
  messageId: number
): Promise<Message | null> {
  if (!chatDb) {
    throw new Error("Database not initialized");
  }

  // Use ROWID-direct query if cached (skips JOIN chat)
  const cachedRowid = chatGuidToRowid.get(chatId);
  const query = cachedRowid !== undefined
    ? `SELECT
        message.ROWID as id,
        message.guid,
        COALESCE(handle.id, 'me') as sender,
        message.text,
        message.attributedBody,
        message.date,
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
      WHERE chat_message_join.chat_id = ? AND message.ROWID = ?
      LIMIT 1`
    : `SELECT
        message.ROWID as id,
        message.guid,
        chat.guid as chat_id,
        COALESCE(handle.id, 'me') as sender,
        message.text,
        message.attributedBody,
        message.date,
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
      WHERE chat.guid = ? AND message.ROWID = ?
      LIMIT 1`;

  try {
    const rows = await chatDb.select<MessageRow[]>(
      query, [cachedRowid !== undefined ? cachedRowid : chatId, messageId]
    );
    if (rows.length === 0) return null;
    const row = rows[0];
    return await rowToMessage(row, chatId);
  } catch (error) {
    logger.error("getMessage error:", error);
    return null;
  }
}

/**
 * Get multiple messages by their ROWIDs in a single query (batch fetch)
 * Used by pollMessages() to avoid N+1 when fetching new messages
 */
export async function getMessagesBatch(
  chatId: string,
  messageIds: number[]
): Promise<Message[]> {
  if (!chatDb || messageIds.length === 0) return [];

  const placeholders = messageIds.map(() => "?").join(",");
  // Use ROWID-direct query if cached (skips JOIN chat)
  const cachedRowid = chatGuidToRowid.get(chatId);
  const query = cachedRowid !== undefined
    ? `SELECT
        message.ROWID as id,
        message.guid,
        COALESCE(handle.id, 'me') as sender,
        message.text,
        message.attributedBody,
        message.date,
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
      WHERE chat_message_join.chat_id = ? AND message.ROWID IN (${placeholders})
      ORDER BY message.date ASC`
    : `SELECT
        message.ROWID as id,
        message.guid,
        chat.guid as chat_id,
        COALESCE(handle.id, 'me') as sender,
        message.text,
        message.attributedBody,
        message.date,
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
      WHERE chat.guid = ? AND message.ROWID IN (${placeholders})
      ORDER BY message.date ASC`;

  try {
    const rows = await chatDb.select<MessageRow[]>(
      query, [cachedRowid !== undefined ? cachedRowid : chatId, ...messageIds]
    );

    // Run all 3 batch queries in parallel
    const rowIds = rows.map((r) => r.id);
    const guids = rows.map((r) => r.guid).filter((g) => g);
    const replyGuidsRaw = rows
      .map((r) => r.reply_to_guid)
      .filter((g): g is string => Boolean(g) && !guidToRowidCache.has(g));
    const uncachedGuids = [...new Set(replyGuidsRaw)];

    const [attachmentRows, reactionRows, guidRows] = await Promise.all([
      // Batch attachments
      rowIds.length > 0
        ? chatDb.select<(AttachmentRow & { message_id: number })[]>(
            `SELECT
              message_attachment_join.message_id,
              attachment.ROWID as attachment_id,
              attachment.filename,
              attachment.mime_type,
              attachment.total_bytes as file_size,
              attachment.transfer_name
            FROM message_attachment_join
            JOIN attachment ON message_attachment_join.attachment_id = attachment.ROWID
            WHERE message_attachment_join.message_id IN (${rowIds.map(() => "?").join(",")})`,
            rowIds
          )
        : Promise.resolve([]),
      // Batch reactions (no self-join: associated_message_guid IS the target GUID)
      guids.length > 0
        ? chatDb.select<(ReactionRow & { message_guid: string })[]>(
            `SELECT
              reaction.associated_message_guid as message_guid,
              reaction.ROWID as id,
              reaction.associated_message_type,
              reaction.date,
              reaction.is_from_me,
              COALESCE(handle.id, 'me') as sender
            FROM message AS reaction
            LEFT JOIN handle ON reaction.handle_id = handle.ROWID
            WHERE reaction.associated_message_guid IN (${guids.map(() => "?").join(",")})
              AND reaction.associated_message_type IS NOT NULL`,
            guids
          )
        : Promise.resolve([]),
      // Batch reply GUID→ROWID
      uncachedGuids.length > 0
        ? chatDb.select<{ guid: string; id: number }[]>(
            `SELECT guid, ROWID as id FROM message
            WHERE guid IN (${uncachedGuids.map(() => "?").join(",")})`,
            uncachedGuids
          )
        : Promise.resolve([]),
    ]);

    // Build attachments map
    const attachmentsByMessageId = new Map<number, Attachment[]>();
    for (const row of attachmentRows) {
      let attachments = attachmentsByMessageId.get(row.message_id);
      if (!attachments) {
        attachments = [];
        attachmentsByMessageId.set(row.message_id, attachments);
      }
      attachments.push({
        filename: row.transfer_name || row.filename || "attachment",
        file_path: row.filename,
        mime_type: row.mime_type,
        file_size: row.file_size,
      });
    }

    // Build reactions map
    const reactionsByMessageGuid = new Map<string, Reaction[]>();
    for (const rRow of reactionRows) {
      const reactionType = parseReactionType(rRow.associated_message_type);
      if (!reactionType || reactionType.startsWith("remove_")) continue;
      const guid = rRow.message_guid;
      let reactions = reactionsByMessageGuid.get(guid);
      if (!reactions) {
        reactions = [];
        reactionsByMessageGuid.set(guid, reactions);
      }
      const sender = normalizePhoneNumber(rRow.sender) || rRow.sender;
      reactions.push({
        type: reactionType,
        sender,
        sender_name: rRow.is_from_me ? null : resolveContactName(sender),
        date: formatDate(parseAppleTimestamp(rRow.date)) || "",
      });
    }

    // Build reply GUID→ROWID map
    const replyGuidToRowid = new Map<string, number>();
    for (const gr of guidRows) {
      replyGuidToRowid.set(gr.guid, gr.id);
      guidToRowidCache.set(gr.guid, gr.id);
    }
    for (const row of rows) {
      if (row.reply_to_guid) {
        const cached = guidToRowidCache.get(row.reply_to_guid);
        if (cached !== undefined) {
          replyGuidToRowid.set(row.reply_to_guid, cached);
        }
      }
    }

    // Convert rows to messages in parallel (no I/O with prefetched data)
    const results = await Promise.all(
      rows.map((row) =>
        rowToMessage(
          row,
          chatId,
          attachmentsByMessageId.get(row.id) || [],
          reactionsByMessageGuid.get(row.guid) || [],
          row.reply_to_guid ? replyGuidToRowid.get(row.reply_to_guid) ?? null : null
        )
      )
    );
    const messages = results.filter((m): m is Message => m !== null);

    return messages;
  } catch (error) {
    logger.error("getMessagesBatch error:", error);
    return [];
  }
}

/**
 * Get the last message ROWID (for detecting new messages)
 */
export async function getLastMessageRowid(): Promise<number> {
  if (!chatDb) return 0;

  try {
    const result = await chatDb.select<{ last_rowid: number }[]>(
      LAST_MESSAGE_ROWID_QUERY
    );
    return result[0]?.last_rowid || 0;
  } catch {
    return 0;
  }
}

/**
 * Get new messages since a specific ROWID
 */
export async function getNewMessagesSince(
  sinceRowid: number
): Promise<{ chatId: string; messageId: number }[]> {
  if (!chatDb) return [];

  try {
    const query = getNewMessagesQuery();
    const rows = await chatDb.select<{ id: number; chat_id: string }[]>(
      query,
      [sinceRowid]
    );

    return rows.map((row: { id: number; chat_id: string }) => ({
      chatId: row.chat_id,
      messageId: row.id,
    }));
  } catch (error) {
    logger.error("getNewMessagesSince error:", error);
    return [];
  }
}

/**
 * Convert a database row to a Message object
 *
 * @param row - Message row from database
 * @param chatId - Chat identifier
 * @param prefetchedAttachments - Optional prefetched attachments (to avoid N+1 query)
 * @param prefetchedReactions - Optional prefetched reactions (to avoid N+1 query)
 * @param prefetchedReplyRowid - Optional prefetched reply ROWID (to avoid N+1 query)
 */
async function rowToMessage(
  row: MessageRow,
  chatId: string,
  prefetchedAttachments?: Attachment[],
  prefetchedReactions?: Reaction[],
  prefetchedReplyRowid?: number | null
): Promise<Message | null> {
  // Get sender and resolve name
  const sender = normalizePhoneNumber(row.sender) || row.sender;
  const senderName = row.is_from_me ? null : resolveContactName(sender);

  // Check for system messages (group events)
  const groupActionType = row.group_action_type || 0;
  const isSystemMessage = groupActionType !== 0;

  if (isSystemMessage) {
    const text = generateGroupEventText(
      groupActionType,
      sender,
      senderName,
      row.affected_handle_id,
      Boolean(row.is_from_me)
    );

    return {
      id: row.id,
      chat_id: chatId,
      sender,
      sender_name: senderName,
      text,
      date: formatDate(parseAppleTimestamp(row.date)) || "",
      is_from_me: Boolean(row.is_from_me),
      attachments: [],
      reply_to_id: null,
      reactions: [],
      date_delivered: null,
      date_read: null,
      is_system_message: true,
    };
  }

  // Extract text
  let text = row.text || "";
  if (!text && row.attributedBody) {
    text = parseAttributedBody(row.attributedBody) || "";
  }

  // Get attachments (use prefetched if available to avoid N+1 query)
  const attachments = prefetchedAttachments !== undefined
    ? prefetchedAttachments
    : await getAttachmentsForMessage(row.id);

  // Skip messages with no content
  if (!text && attachments.length === 0) {
    return null;
  }

  // Resolve reply_to_id from GUID (use prefetched if available)
  let replyToId: number | null = null;
  if (row.reply_to_guid) {
    replyToId = prefetchedReplyRowid !== undefined
      ? prefetchedReplyRowid
      : await getMessageRowidByGuid(row.reply_to_guid);
  }

  // Get reactions (use prefetched if available to avoid N+1 query)
  const reactions = prefetchedReactions !== undefined
    ? prefetchedReactions
    : await getReactionsForMessage(row.guid);

  // Parse delivery/read receipts
  let dateDelivered: string | null = null;
  let dateRead: string | null = null;
  if (row.is_from_me) {
    dateDelivered = formatDate(parseAppleTimestamp(row.date_delivered));
    dateRead = formatDate(parseAppleTimestamp(row.date_read));
  }

  return {
    id: row.id,
    chat_id: chatId,
    sender,
    sender_name: senderName,
    text,
    date: formatDate(parseAppleTimestamp(row.date)) || "",
    is_from_me: Boolean(row.is_from_me),
    attachments,
    reply_to_id: replyToId,
    reactions,
    date_delivered: dateDelivered,
    date_read: dateRead,
    is_system_message: false,
  };
}

/**
 * Get attachments for a message
 */
async function getAttachmentsForMessage(messageId: number): Promise<Attachment[]> {
  if (!chatDb) return [];

  try {
    const rows = await chatDb.select<AttachmentRow[]>(ATTACHMENTS_QUERY, [
      messageId,
    ]);

    return rows.map((row: AttachmentRow) => ({
      filename: row.transfer_name || row.filename || "attachment",
      file_path: row.filename,
      mime_type: row.mime_type,
      file_size: row.file_size,
    }));
  } catch {
    return [];
  }
}

/**
 * Get reactions for a message
 */
async function getReactionsForMessage(messageGuid: string): Promise<Reaction[]> {
  if (!chatDb || !messageGuid) return [];

  try {
    const rows = await chatDb.select<ReactionRow[]>(REACTIONS_QUERY, [
      messageGuid,
    ]);

    return rows
      .map((row: ReactionRow) => {
        const reactionType = parseReactionType(row.associated_message_type);
        if (!reactionType || reactionType.startsWith("remove_")) {
          return null;
        }

        const sender = normalizePhoneNumber(row.sender) || row.sender;
        return {
          type: reactionType,
          sender,
          sender_name: row.is_from_me ? null : resolveContactName(sender),
          date: formatDate(parseAppleTimestamp(row.date)) || "",
        };
      })
      .filter((r: Reaction | null): r is Reaction => r !== null);
  } catch {
    return [];
  }
}

/**
 * Get message ROWID from GUID
 */
async function getMessageRowidByGuid(guid: string): Promise<number | null> {
  // Check cache first (LRU automatically updates recency)
  const cached = guidToRowidCache.get(guid);
  if (cached !== undefined) return cached;

  if (!chatDb) return null;

  try {
    const rows = await chatDb.select<{ id: number }[]>(MESSAGE_BY_GUID_QUERY, [
      guid,
    ]);
    if (rows.length > 0) {
      const row = rows[0];
      guidToRowidCache.set(guid, row.id);
      return row.id;
    }
    return null;
  } catch {
    return null;
  }
}

/**
 * Resolve a phone number or email to a contact name
 * Note: Contact resolution requires AddressBook access which isn't available
 * from the Tauri app. The HTTP API handles this via Python.
 * This is a placeholder that returns null - the API fallback will provide names.
 */
export function resolveContactName(identifier: string): string | null {
  if (!identifier || identifier === "me") return null;

  // Check cache with exact match
  const cached = contactsCache.get(identifier);
  if (cached) return cached;

  // Try normalized form as fallback
  const normalized = normalizePhoneNumber(identifier);
  if (normalized && normalized !== identifier) {
    const normalizedCached = contactsCache.get(normalized);
    if (normalizedCached) return normalizedCached;
  }

  return null;
}

/**
 * Populate the contacts cache from an external source (e.g., backend RPC).
 * After calling this, resolveContactName() will return cached names.
 */
export function populateContactsCache(contacts: Record<string, string | null>): void {
  for (const [identifier, name] of Object.entries(contacts)) {
    if (name) {
      contactsCache.set(identifier, name);
      // Also store under normalized form for consistent lookup
      const normalized = normalizePhoneNumber(identifier);
      if (normalized && normalized !== identifier) {
        contactsCache.set(normalized, name);
      }
    }
  }
  contactsCacheLoaded = true;
  logger.info(`Contacts cache populated: ${contactsCache.size} entries`);
}

/**
 * Load contacts directly from macOS AddressBook SQLite databases.
 * No socket server needed - reads AddressBook-v22.abcddb files directly.
 * Uses Rust command to discover source directories, then opens each via SQL plugin.
 * Falls back silently if AddressBook is inaccessible (no Full Disk Access).
 */
export async function loadContactsFromAddressBook(): Promise<void> {
  if (!isTauri || !Database) return;

  const { invoke } = await import("@tauri-apps/api/core");

  // Rust command lists AddressBook source DB paths
  let dbPaths: string[];
  try {
    dbPaths = await invoke<string[]>("list_addressbook_sources");
  } catch {
    logger.warn("Could not list AddressBook sources");
    return;
  }

  if (dbPaths.length === 0) {
    logger.info("No AddressBook sources found");
    return;
  }

  let loaded = 0;

  // Parallelize across all AddressBook sources
  await Promise.allSettled(dbPaths.map(async (dbPath) => {
    const db: SqlDatabase = await Database.load(`sqlite:${dbPath}?mode=ro`);
    try {
      // Run phone + email queries in parallel within each source
      const [phones, emails] = await Promise.all([
        db.select<
          { identifier: string; first_name: string | null; last_name: string | null }[]
        >(
          `SELECT p.ZFULLNUMBER as identifier, r.ZFIRSTNAME as first_name, r.ZLASTNAME as last_name
           FROM ZABCDPHONENUMBER p
           JOIN ZABCDRECORD r ON p.ZOWNER = r.Z_PK
           WHERE p.ZFULLNUMBER IS NOT NULL AND (r.ZFIRSTNAME IS NOT NULL OR r.ZLASTNAME IS NOT NULL)`
        ).catch(() => [] as { identifier: string; first_name: string | null; last_name: string | null }[]),
        db.select<
          { identifier: string; first_name: string | null; last_name: string | null }[]
        >(
          `SELECT e.ZADDRESS as identifier, r.ZFIRSTNAME as first_name, r.ZLASTNAME as last_name
           FROM ZABCDEMAILADDRESS e
           JOIN ZABCDRECORD r ON e.ZOWNER = r.Z_PK
           WHERE e.ZADDRESS IS NOT NULL AND (r.ZFIRSTNAME IS NOT NULL OR r.ZLASTNAME IS NOT NULL)`
        ).catch(() => [] as { identifier: string; first_name: string | null; last_name: string | null }[]),
      ]);

      for (const row of phones) {
        const name = formatContactName(row.first_name, row.last_name);
        const normalized = normalizePhoneNumber(row.identifier);
        if (name && normalized) {
          contactsCache.set(normalized, name);
          loaded++;
        }
      }
      for (const row of emails) {
        const name = formatContactName(row.first_name, row.last_name);
        if (name && row.identifier) {
          contactsCache.set(row.identifier.toLowerCase(), name);
          loaded++;
        }
      }
    } finally {
      // Pass the DB path to close() so it only closes THIS specific pool.
      // close() with no args closes ALL pools including chat.db.
      await db.close(db.path);
    }
  }));

  if (loaded > 0) {
    contactsCacheLoaded = true;
    logger.info(`Loaded ${loaded} contacts from AddressBook (${contactsCache.size} cache entries)`);
  }
}

function formatContactName(first: string | null, last: string | null): string | null {
  const parts: string[] = [];
  if (first) parts.push(first);
  if (last) parts.push(last);
  return parts.length > 0 ? parts.join(" ") : null;
}

/**
 * Check if contacts cache has been populated
 */
export function isContactsCacheLoaded(): boolean {
  return contactsCacheLoaded;
}

/**
 * Format a participant identifier for display
 * Resolves the contact name if available, otherwise formats the phone/email
 */
export function formatParticipant(identifier: string): string {
  if (!identifier || identifier === "me") return "You";
  
  // Try to resolve contact name
  const name = resolveContactName(identifier);
  if (name) return name;
  
  // Format phone number or email
  const normalized = normalizePhoneNumber(identifier);
  if (normalized) {
    // Show last 4 digits for privacy: "+1 (555) 123-4567" -> "...4567"
    const digits = normalized.replace(/\D/g, '');
    if (digits.length >= 10) {
      return `...${digits.slice(-4)}`;
    }
  }
  return identifier;
}

/**
 * Generate human-readable text for group events
 */
function generateGroupEventText(
  actionType: number,
  actor: string,
  actorName: string | null,
  affectedHandleId: string | null,
  isFromMe: boolean
): string {
  const actorDisplay = isFromMe ? "You" : actorName || actor;

  let affectedDisplay: string | null = null;
  if (affectedHandleId) {
    const normalized = normalizePhoneNumber(affectedHandleId) || affectedHandleId;
    affectedDisplay = resolveContactName(normalized) || normalized;
  }

  switch (actionType) {
    case 1: // Left or removed
      if (affectedDisplay && affectedDisplay !== actorDisplay) {
        return `${actorDisplay} removed ${affectedDisplay} from the group`;
      }
      return `${actorDisplay} left the group`;

    case 2: // Name changed
      return `${actorDisplay} changed the group name`;

    case 3: // Joined or added
      if (affectedDisplay && affectedDisplay !== actorDisplay) {
        return `${actorDisplay} added ${affectedDisplay} to the group`;
      }
      return `${actorDisplay} joined the group`;

    default:
      return `Group event (type ${actionType})`;
  }
}
