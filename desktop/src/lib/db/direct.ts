/**
 * Direct SQLite access layer for iMessage chat.db
 * Bypasses HTTP API for ~20-100x faster message loading
 */

import type { Conversation, Message, Attachment, Reaction } from "../api/types";

// Dynamic import for Tauri plugin - only works in Tauri context
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let Database: any = null;

// Check if running in Tauri context
const isTauri = typeof window !== "undefined" && "__TAURI__" in window;
import {
  getConversationsQuery,
  getMessagesQuery,
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

// Database paths (lazy evaluated to avoid process.env errors in browser)
const getChatDbPath = () => {
  if (typeof process !== "undefined" && process.env?.HOME) {
    return `${process.env.HOME}/Library/Messages/chat.db`;
  }
  // Fallback for browser context - will fail gracefully when trying to connect
  return "/Library/Messages/chat.db";
};

// Connection state
let chatDb: Database | null = null;
let isInitialized = false;
let initError: Error | null = null;
let schemaVersion: SchemaVersion = "v14";

// Contact name cache (phone/email -> name)
const contactsCache = new Map<string, string>();
let contactsCacheLoaded = false;

// GUID to ROWID cache for reply_to_id resolution
const guidToRowidCache = new Map<string, number>();

/**
 * Initialize database connections
 * Opens chat.db in read-only mode
 */
export async function initDatabases(): Promise<void> {
  if (isInitialized) return;
  if (initError) throw initError;

  // Only attempt direct DB access in Tauri context
  if (!isTauri) {
    initError = new Error("Direct database access only available in Tauri context");
    throw initError;
  }

  try {
    // Dynamically import the Tauri SQL plugin
    if (!Database) {
      const module = await import("@tauri-apps/plugin-sql");
      Database = module.default;
    }

    // Open chat.db in read-only mode via URI
    // tauri-plugin-sql uses sqlite:// protocol
    const uri = `sqlite:${getChatDbPath()}?mode=ro`;
    chatDb = await Database.load(uri);

    // Detect schema version
    schemaVersion = await detectSchemaVersion();

    isInitialized = true;
    console.log(`[DirectDB] Initialized with schema ${schemaVersion}`);
  } catch (error) {
    initError = error instanceof Error ? error : new Error(String(error));
    console.error("[DirectDB] Failed to initialize:", initError);
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
    await chatDb.close();
    chatDb = null;
  }
  isInitialized = false;
  initError = null;
  contactsCache.clear();
  contactsCacheLoaded = false;
  guidToRowidCache.clear();
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
  limit: number = 50,
  since?: Date,
  before?: Date
): Promise<Conversation[]> {
  if (!chatDb) {
    throw new Error("Database not initialized");
  }

  const query = getConversationsQuery({
    withSinceFilter: !!since,
    withBeforeFilter: !!before,
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
    const rows = await chatDb.select<ConversationRow[]>(query, params);

    return rows.map((row) => {
      // Parse participants
      const participantsStr = row.participants || "";
      const participants = participantsStr
        .split(",")
        .map((p) => p.trim())
        .filter(Boolean)
        .map((p) => normalizePhoneNumber(p) || p);

      // Determine if group chat
      const isGroup = participants.length > 1;

      // Parse last message date
      const lastMessageDate = parseAppleTimestamp(row.last_message_date);

      // Get display name or resolve from contacts
      let displayName = row.display_name || null;
      if (!displayName && !isGroup && participants.length === 1) {
        displayName = resolveContactName(participants[0]);
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
    console.error("[DirectDB] getConversations error:", error);
    throw error;
  }
}

/**
 * Get messages for a conversation
 */
export async function getMessages(
  chatId: string,
  limit: number = 100,
  before?: Date
): Promise<Message[]> {
  if (!chatDb) {
    throw new Error("Database not initialized");
  }

  const query = getMessagesQuery({
    withBeforeFilter: !!before,
  });

  // Build parameters
  const params: (string | number)[] = [chatId];
  if (before) {
    params.push(toAppleTimestamp(before));
  }
  params.push(limit);

  try {
    const rows = await chatDb.select<MessageRow[]>(query, params);
    const messages: Message[] = [];

    for (const row of rows) {
      const message = await rowToMessage(row, chatId);
      if (message) {
        messages.push(message);
      }
    }

    return messages;
  } catch (error) {
    console.error("[DirectDB] getMessages error:", error);
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

  const query = `
    SELECT
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
    LIMIT 1
  `;

  try {
    const rows = await chatDb.select<MessageRow[]>(query, [chatId, messageId]);
    if (rows.length === 0) return null;
    return await rowToMessage(rows[0], chatId);
  } catch (error) {
    console.error("[DirectDB] getMessage error:", error);
    return null;
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

    return rows.map((row) => ({
      chatId: row.chat_id,
      messageId: row.id,
    }));
  } catch (error) {
    console.error("[DirectDB] getNewMessagesSince error:", error);
    return [];
  }
}

/**
 * Convert a database row to a Message object
 */
async function rowToMessage(
  row: MessageRow,
  chatId: string
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
      !!row.is_from_me
    );

    return {
      id: row.id,
      chat_id: chatId,
      sender,
      sender_name: senderName,
      text,
      date: formatDate(parseAppleTimestamp(row.date)) || "",
      is_from_me: !!row.is_from_me,
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

  // Get attachments
  const attachments = await getAttachmentsForMessage(row.id);

  // Skip messages with no content
  if (!text && attachments.length === 0) {
    return null;
  }

  // Resolve reply_to_id from GUID
  let replyToId: number | null = null;
  if (row.reply_to_guid) {
    replyToId = await getMessageRowidByGuid(row.reply_to_guid);
  }

  // Get reactions
  const reactions = await getReactionsForMessage(row.guid);

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
    is_from_me: !!row.is_from_me,
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

    return rows.map((row) => ({
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
      .map((row) => {
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
      .filter((r): r is Reaction => r !== null);
  } catch {
    return [];
  }
}

/**
 * Get message ROWID from GUID
 */
async function getMessageRowidByGuid(guid: string): Promise<number | null> {
  // Check cache first
  const cached = guidToRowidCache.get(guid);
  if (cached !== undefined) return cached;

  if (!chatDb) return null;

  try {
    const rows = await chatDb.select<{ id: number }[]>(MESSAGE_BY_GUID_QUERY, [
      guid,
    ]);
    if (rows.length > 0) {
      guidToRowidCache.set(guid, rows[0].id);
      return rows[0].id;
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
function resolveContactName(identifier: string): string | null {
  if (!identifier || identifier === "me") return null;

  // Check cache
  const cached = contactsCache.get(identifier);
  if (cached) return cached;

  // Contact resolution not available in direct mode
  // Names will be resolved via HTTP API fallback
  return null;
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
