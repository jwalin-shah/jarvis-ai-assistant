/**
 * SQL queries for direct iMessage chat.db access
 * Ported from integrations/imessage/queries.py
 */

/**
 * Schema version for chat.db
 * v14 = macOS 14 and earlier
 * v15 = macOS 15+
 */
export type SchemaVersion = "v14" | "v15" | "unknown";

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
  const sinceFilter = options.withSinceFilter ? "AND last_message_date > ?" : "";
  const beforeFilter = options.withBeforeFilter ? "AND last_message_date < ?" : "";

  return `
    SELECT
      chat.ROWID as chat_rowid,
      chat.guid as chat_id,
      chat.display_name,
      chat.chat_identifier,
      (
        SELECT GROUP_CONCAT(handle.id, ', ')
        FROM chat_handle_join
        JOIN handle ON chat_handle_join.handle_id = handle.ROWID
        WHERE chat_handle_join.chat_id = chat.ROWID
      ) as participants,
      (
        SELECT COUNT(*)
        FROM chat_message_join
        WHERE chat_message_join.chat_id = chat.ROWID
      ) as message_count,
      (
        SELECT MAX(message.date)
        FROM chat_message_join
        JOIN message ON chat_message_join.message_id = message.ROWID
        WHERE chat_message_join.chat_id = chat.ROWID
      ) as last_message_date,
      (
        SELECT message.text
        FROM chat_message_join
        JOIN message ON chat_message_join.message_id = message.ROWID
        WHERE chat_message_join.chat_id = chat.ROWID
        ORDER BY message.date DESC
        LIMIT 1
      ) as last_message_text,
      (
        SELECT message.attributedBody
        FROM chat_message_join
        JOIN message ON chat_message_join.message_id = message.ROWID
        WHERE chat_message_join.chat_id = chat.ROWID
        ORDER BY message.date DESC
        LIMIT 1
      ) as last_message_attributed_body
    FROM chat
    WHERE message_count > 0
    ${sinceFilter}
    ${beforeFilter}
    ORDER BY last_message_date DESC
    LIMIT ?
  `;
}

/**
 * Query to get messages for a conversation
 */
export function getMessagesQuery(options: {
  withBeforeFilter?: boolean;
}): string {
  const beforeFilter = options.withBeforeFilter ? "AND message.date < ?" : "";

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
      message.is_from_me
    FROM message
    JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
    JOIN chat ON chat_message_join.chat_id = chat.ROWID
    LEFT JOIN handle ON message.handle_id = handle.ROWID
    WHERE message.ROWID > ?
    ORDER BY message.date ASC
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
  const unixMs = (timestamp / 1_000_000) + (APPLE_EPOCH_OFFSET * 1000);
  return new Date(unixMs);
}

/**
 * Convert JavaScript Date to Apple timestamp
 */
export function toAppleTimestamp(date: Date): number {
  const unixMs = date.getTime();
  // Convert to nanoseconds since Apple epoch
  return (unixMs - (APPLE_EPOCH_OFFSET * 1000)) * 1_000_000;
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
  if (phone.includes("@")) {
    return phone.toLowerCase();
  }

  // Strip everything except digits and leading +
  const hasPlus = phone.startsWith("+");
  const digits = phone.replace(/\D/g, "");

  if (!digits) return null;

  return hasPlus ? `+${digits}` : digits;
}

/**
 * Reaction type mapping from associated_message_type
 */
export const REACTION_TYPES: Record<number, string> = {
  2000: "love",
  2001: "like",
  2002: "dislike",
  2003: "laugh",
  2004: "emphasis",
  2005: "question",
  3000: "remove_love",
  3001: "remove_like",
  3002: "remove_dislike",
  3003: "remove_laugh",
  3004: "remove_emphasis",
  3005: "remove_question",
};

/**
 * Parse reaction type from associated_message_type
 */
export function parseReactionType(type: number): string | null {
  return REACTION_TYPES[type] || null;
}

/**
 * Parse attributedBody blob to extract text
 * attributedBody is a binary plist containing NSAttributedString data
 */
export function parseAttributedBody(data: ArrayBuffer | null): string | null {
  if (!data) return null;

  try {
    // attributedBody is a serialized NSAttributedString
    // The text is usually at a known offset after the "NSString" marker
    const bytes = new Uint8Array(data);
    const decoder = new TextDecoder("utf-8");

    // Simple extraction: find readable text between control characters
    // This is a simplified approach - the full implementation would parse the bplist
    let text = "";
    let inText = false;
    let consecutivePrintable = 0;

    for (let i = 0; i < bytes.length; i++) {
      const byte = bytes[i];
      // Check if printable ASCII or UTF-8 continuation
      if ((byte >= 32 && byte < 127) || byte >= 128) {
        consecutivePrintable++;
        if (consecutivePrintable >= 2) {
          inText = true;
        }
        if (inText) {
          // Accumulate bytes for UTF-8 decoding
        }
      } else {
        if (inText && consecutivePrintable >= 3) {
          // Try to decode the accumulated text
          const chunk = bytes.slice(i - consecutivePrintable, i);
          try {
            const decoded = decoder.decode(chunk);
            // Filter out internal markers
            if (!decoded.includes("NSAttributedString") &&
                !decoded.includes("NSDictionary") &&
                !decoded.includes("NSMutableAttributedString")) {
              text += decoded;
            }
          } catch {
            // Ignore decoding errors
          }
        }
        inText = false;
        consecutivePrintable = 0;
      }
    }

    // Handle remaining text
    if (inText && consecutivePrintable >= 3) {
      const chunk = bytes.slice(bytes.length - consecutivePrintable);
      try {
        const decoded = decoder.decode(chunk);
        if (!decoded.includes("NSAttributedString")) {
          text += decoded;
        }
      } catch {
        // Ignore
      }
    }

    return text.trim() || null;
  } catch {
    return null;
  }
}
