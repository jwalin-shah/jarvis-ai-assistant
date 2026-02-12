import { describe, it, expect } from "vitest";
import {
  parseAttributedBody,
  parseAppleTimestamp,
  toAppleTimestamp,
  formatDate,
  normalizePhoneNumber,
  parseReactionType,
  getConversationsQuery,
  getMessagesQuery,
  getNewMessagesQuery,
  APPLE_EPOCH_OFFSET,
  REACTION_TYPES,
  DETECT_SCHEMA_SQL,
  ATTACHMENTS_QUERY,
  REACTIONS_QUERY,
  MESSAGE_BY_GUID_QUERY,
  LAST_MESSAGE_ROWID_QUERY,
} from "@/lib/db/queries";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Convert a base64 string to an ArrayBuffer (for typedstream / bplist fixtures) */
function b64toArrayBuffer(b64: string): ArrayBuffer {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes.buffer;
}

/** Build an ArrayBuffer filled with random bytes */
function randomBytes(length: number): ArrayBuffer {
  const buf = new Uint8Array(length);
  for (let i = 0; i < length; i++) buf[i] = Math.floor(Math.random() * 256);
  return buf.buffer;
}

// ---------------------------------------------------------------------------
// parseAttributedBody
// ---------------------------------------------------------------------------

describe("parseAttributedBody", () => {
  // -- Typedstream fixtures (real iMessage chat.db data) --------------------

  const typedstreamFixtures: Array<{ b64: string; expected: string }> = [
    {
      b64: "BAtzdHJlYW10eXBlZIHoA4QBQISEhBJOU0F0dHJpYnV0ZWRTdHJpbmcAhIQITlNPYmplY3QAhZKEhIQITlNTdHJpbmcBlIQBKxBpIHdvdWxkIG5ldmVyIG9rhoQCaUkBEJKEhIQMTlNEaWN0aW9uYXJ5AJSEAWkBkoSWlh1fX2tJTU1lc3NhZ2VQYXJ0QXR0cmlidXRlTmFtZYaShISECE5TTnVtYmVyAISEB05TVmFsdWUAlIQBKoSZmQCGhoY=",
      expected: "i would never ok",
    },
    {
      b64: "BAtzdHJlYW10eXBlZIHoA4QBQISEhBJOU0F0dHJpYnV0ZWRTdHJpbmcAhIQITlNPYmplY3QAhZKEhIQITlNTdHJpbmcBlIQBKzNJIGNvdWxkbuKAmXQgc2VlIHlvdXIgbWVzc2FnZXMgdW50aWwgSSB0ZXh0ZWQgZmlyc3SGhAJpSQExkoSEhAxOU0RpY3Rpb25hcnkAlIQBaQGShJaWHV9fa0lNTWVzc2FnZVBhcnRBdHRyaWJ1dGVOYW1lhpKEhIQITlNOdW1iZXIAhIQHTlNWYWx1ZQCUhAEqhJmZAIaGhg==",
      expected: "I couldn\u2019t see your messages until I texted first",
    },
    {
      b64: "BAtzdHJlYW10eXBlZIHoA4QBQISEhBJOU0F0dHJpYnV0ZWRTdHJpbmcAhIQITlNPYmplY3QAhZKEhIQITlNTdHJpbmcBlIQBKwtPayBpdCB3b3Jrc4aEAmlJAQuShISEDE5TRGljdGlvbmFyeQCUhAFpAZKElpYdX19rSU1NZXNzYWdlUGFydEF0dHJpYnV0ZU5hbWWGkoSEhAhOU051bWJlcgCEhAdOU1ZhbHVlAJSEASqEmZkAhoaG",
      expected: "Ok it works",
    },
    {
      b64: "BAtzdHJlYW10eXBlZIHoA4QBQISEhBJOU0F0dHJpYnV0ZWRTdHJpbmcAhIQITlNPYmplY3QAhZKEhIQITlNTdHJpbmcBlIQBKwRUZXN0hoQCaUkBBJKEhIQMTlNEaWN0aW9uYXJ5AJSEAWkBkoSWlh1fX2tJTU1lc3NhZ2VQYXJ0QXR0cmlidXRlTmFtZYaShISECE5TTnVtYmVyAISEB05TVmFsdWUAlIQBKoSZmQCGhoY=",
      expected: "Test",
    },
  ];

  describe("typedstream format", () => {
    for (const { b64, expected } of typedstreamFixtures) {
      it(`parses "${expected}"`, () => {
        const buf = b64toArrayBuffer(b64);
        expect(parseAttributedBody(buf)).toBe(expected);
      });
    }
  });

  // -- bplist (NSKeyedArchive) fixture --------------------------------------

  describe("bplist (NSKeyedArchive) format", () => {
    it('parses "Hello from bplist!"', () => {
      const b64 =
        "YnBsaXN0MDDUAQIDBAUGHiFZJGFyY2hpdmVyWCRvYmplY3RzVCR0b3BYJHZlcnNpb25fEA9OU0tleWVkQXJjaGl2ZXKmBwgPEBcbVSRudWxs0wkKCwwNDlYkY2xhc3NcTlNBdHRyaWJ1dGVzWE5TU3RyaW5ngAOABIACXxATSGVsbG8gZnJvbSBicGxpc3QhIdIREhMUWCRjbGFzc2VzWiRjbGFzc25hbWWjFBUWXxAZTlNNdXRhYmxlQXR0cmlidXRlZFN0cmluZ18QEk5TQXR0cmlidXRlZFN0cmluZ1hOU09iamVjdNIJGBkaV05TRW1wdHmABQnSERIcHaIdFlxOU0RpY3Rpb25hcnnRHyBUcm9vdIABEgABhqAACAARABsAJAApADIARABLAFEAWABfAGwAdQB3AHkAewCRAJYAnwCqAK4AygDfAOgA7QD1APcA+AD9AQABDQEQARUBFwAAAAAAAAIBAAAAAAAAACIAAAAAAAAAAAAAAAAAAAEc";
      const buf = b64toArrayBuffer(b64);
      expect(parseAttributedBody(buf)).toBe("Hello from bplist!!");
    });
  });

  // -- Null / empty / malformed inputs --------------------------------------

  describe("edge cases", () => {
    it("returns null for null input", () => {
      expect(parseAttributedBody(null)).toBeNull();
    });

    it("returns null for empty ArrayBuffer", () => {
      expect(parseAttributedBody(new ArrayBuffer(0))).toBeNull();
    });

    it("returns null for tiny buffer (<8 bytes)", () => {
      expect(parseAttributedBody(new ArrayBuffer(1))).toBeNull();
      expect(parseAttributedBody(new ArrayBuffer(4))).toBeNull();
      expect(parseAttributedBody(new ArrayBuffer(7))).toBeNull();
    });

    it("returns null for random garbage bytes", () => {
      // Run several times to decrease fluke chance
      for (let i = 0; i < 5; i++) {
        const buf = randomBytes(64 + Math.floor(Math.random() * 200));
        expect(parseAttributedBody(buf)).toBeNull();
      }
    });

    it("returns null for a buffer that starts with bplist but is truncated", () => {
      // "bplist00" header followed by nothing meaningful
      const header = new TextEncoder().encode("bplist00");
      const buf = new Uint8Array(20);
      buf.set(header, 0);
      expect(parseAttributedBody(buf.buffer)).toBeNull();
    });
  });
});

// ---------------------------------------------------------------------------
// parseAppleTimestamp
// ---------------------------------------------------------------------------

describe("parseAppleTimestamp", () => {
  it("returns null for null input", () => {
    expect(parseAppleTimestamp(null)).toBeNull();
  });

  it("returns null for undefined input", () => {
    // The implementation checks for undefined too
    expect(parseAppleTimestamp(undefined as unknown as number | null)).toBeNull();
  });

  it("converts timestamp 0 to 2001-01-01T00:00:00.000Z (Apple epoch)", () => {
    const date = parseAppleTimestamp(0);
    expect(date).not.toBeNull();
    expect(date!.toISOString()).toBe("2001-01-01T00:00:00.000Z");
  });

  it("converts a known recent timestamp correctly", () => {
    // 2024-01-15T12:00:00Z
    // Unix ms = 1705320000000
    // Apple ns = (1705320000000 - 978307200000) * 1_000_000
    //          = 727012800000 * 1_000_000
    //          = 727012800000000000
    const appleNs = (1705320000000 - APPLE_EPOCH_OFFSET * 1000) * 1_000_000;
    const date = parseAppleTimestamp(appleNs);
    expect(date).not.toBeNull();
    expect(date!.toISOString()).toBe("2024-01-15T12:00:00.000Z");
  });

  it("handles negative timestamps (dates before 2001)", () => {
    // 2000-01-01T00:00:00Z => Unix ms = 946684800000
    // Apple ns = (946684800000 - 978307200000) * 1_000_000
    //          = -31622400000 * 1_000_000
    //          = -31622400000000000
    const appleNs = (946684800000 - APPLE_EPOCH_OFFSET * 1000) * 1_000_000;
    const date = parseAppleTimestamp(appleNs);
    expect(date).not.toBeNull();
    expect(date!.toISOString()).toBe("2000-01-01T00:00:00.000Z");
  });
});

// ---------------------------------------------------------------------------
// toAppleTimestamp
// ---------------------------------------------------------------------------

describe("toAppleTimestamp", () => {
  it("converts Apple epoch date to timestamp 0", () => {
    const appleEpoch = new Date("2001-01-01T00:00:00.000Z");
    expect(toAppleTimestamp(appleEpoch)).toBe(0);
  });

  it("converts a known date correctly", () => {
    const date = new Date("2024-01-15T12:00:00.000Z");
    const expected = (date.getTime() - APPLE_EPOCH_OFFSET * 1000) * 1_000_000;
    expect(toAppleTimestamp(date)).toBe(expected);
  });

  it("round-trips with parseAppleTimestamp", () => {
    const timestamps = [
      0,
      1_000_000_000_000_000, // ~31.7 years after Apple epoch
      727012800000000000, // 2024-01-15
      -31622400000000000, // before Apple epoch
    ];

    for (const ts of timestamps) {
      const date = parseAppleTimestamp(ts);
      expect(date).not.toBeNull();
      expect(toAppleTimestamp(date!)).toBe(ts);
    }
  });

  it("round-trips from Date through toAppleTimestamp and back", () => {
    const dates = [
      new Date("2001-01-01T00:00:00.000Z"),
      new Date("2024-06-15T08:30:00.000Z"),
      new Date("2000-06-15T00:00:00.000Z"),
    ];

    for (const d of dates) {
      const ts = toAppleTimestamp(d);
      const roundTripped = parseAppleTimestamp(ts);
      expect(roundTripped).not.toBeNull();
      expect(roundTripped!.getTime()).toBe(d.getTime());
    }
  });
});

// ---------------------------------------------------------------------------
// formatDate
// ---------------------------------------------------------------------------

describe("formatDate", () => {
  it("returns null for null input", () => {
    expect(formatDate(null)).toBeNull();
  });

  it("returns ISO string for a valid Date", () => {
    const date = new Date("2024-01-15T12:00:00.000Z");
    expect(formatDate(date)).toBe("2024-01-15T12:00:00.000Z");
  });

  it("returns ISO string preserving milliseconds", () => {
    const date = new Date("2024-06-15T08:30:45.123Z");
    expect(formatDate(date)).toBe("2024-06-15T08:30:45.123Z");
  });
});

// ---------------------------------------------------------------------------
// normalizePhoneNumber
// ---------------------------------------------------------------------------

describe("normalizePhoneNumber", () => {
  it("returns null for null input", () => {
    expect(normalizePhoneNumber(null)).toBeNull();
  });

  it("returns null for empty string", () => {
    expect(normalizePhoneNumber("")).toBeNull();
  });

  it('strips formatting from "+1 (408) 555-1234"', () => {
    expect(normalizePhoneNumber("+1 (408) 555-1234")).toBe("+14085551234");
  });

  it('strips formatting from "408-555-1234"', () => {
    expect(normalizePhoneNumber("408-555-1234")).toBe("4085551234");
  });

  it("preserves leading + when present", () => {
    expect(normalizePhoneNumber("+44 20 7946 0958")).toBe("+442079460958");
  });

  it("does not add + when absent", () => {
    expect(normalizePhoneNumber("(650) 555-0100")).toBe("6505550100");
  });

  it("lowercases email addresses", () => {
    expect(normalizePhoneNumber("user@example.com")).toBe("user@example.com");
  });

  it("lowercases mixed-case email addresses", () => {
    expect(normalizePhoneNumber("User@Example.COM")).toBe("user@example.com");
  });

  it("returns null for a string with no digits and no @", () => {
    expect(normalizePhoneNumber("---")).toBeNull();
  });

  it("handles a string with only whitespace", () => {
    expect(normalizePhoneNumber("   ")).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// parseReactionType
// ---------------------------------------------------------------------------

describe("parseReactionType", () => {
  it("maps 2000 to love", () => {
    expect(parseReactionType(2000)).toBe("love");
  });

  it("maps 2001 to like", () => {
    expect(parseReactionType(2001)).toBe("like");
  });

  it("maps 2002 to dislike", () => {
    expect(parseReactionType(2002)).toBe("dislike");
  });

  it("maps 2003 to laugh", () => {
    expect(parseReactionType(2003)).toBe("laugh");
  });

  it("maps 2004 to emphasis", () => {
    expect(parseReactionType(2004)).toBe("emphasis");
  });

  it("maps 2005 to question", () => {
    expect(parseReactionType(2005)).toBe("question");
  });

  it("maps 3000 to remove_love", () => {
    expect(parseReactionType(3000)).toBe("remove_love");
  });

  it("maps 3001 to remove_like", () => {
    expect(parseReactionType(3001)).toBe("remove_like");
  });

  it("maps 3002 to remove_dislike", () => {
    expect(parseReactionType(3002)).toBe("remove_dislike");
  });

  it("maps 3003 to remove_laugh", () => {
    expect(parseReactionType(3003)).toBe("remove_laugh");
  });

  it("maps 3004 to remove_emphasis", () => {
    expect(parseReactionType(3004)).toBe("remove_emphasis");
  });

  it("maps 3005 to remove_question", () => {
    expect(parseReactionType(3005)).toBe("remove_question");
  });

  it("returns null for unknown type 0", () => {
    expect(parseReactionType(0)).toBeNull();
  });

  it("returns null for unknown type 9999", () => {
    expect(parseReactionType(9999)).toBeNull();
  });

  it("returns null for unknown type -1", () => {
    expect(parseReactionType(-1)).toBeNull();
  });

  it("covers all entries in REACTION_TYPES map", () => {
    for (const [key, value] of Object.entries(REACTION_TYPES)) {
      expect(parseReactionType(Number(key))).toBe(value);
    }
  });
});

// ---------------------------------------------------------------------------
// getConversationsQuery
// ---------------------------------------------------------------------------

describe("getConversationsQuery", () => {
  it("returns a valid SQL string with no filters", () => {
    const sql = getConversationsQuery({});
    expect(sql).toContain("SELECT");
    expect(sql).toContain("FROM chat");
    expect(sql).toContain("ORDER BY");
    expect(sql).toContain("DESC");
    expect(sql).toContain("LIMIT ?");
    expect(sql).not.toContain("tc.last_date > ?");
    expect(sql).not.toContain("tc.last_date < ?");
  });

  it("includes since filter when withSinceFilter is true", () => {
    const sql = getConversationsQuery({ withSinceFilter: true });
    expect(sql).toContain("tc.last_date > ?");
    expect(sql).not.toContain("tc.last_date < ?");
  });

  it("includes before filter when withBeforeFilter is true", () => {
    const sql = getConversationsQuery({ withBeforeFilter: true });
    expect(sql).not.toContain("tc.last_date > ?");
    expect(sql).toContain("tc.last_date < ?");
  });

  it("includes both filters when both options are true", () => {
    const sql = getConversationsQuery({
      withSinceFilter: true,
      withBeforeFilter: true,
    });
    expect(sql).toContain("tc.last_date > ?");
    expect(sql).toContain("tc.last_date < ?");
  });

  it("omits filters when options are explicitly false", () => {
    const sql = getConversationsQuery({
      withSinceFilter: false,
      withBeforeFilter: false,
    });
    expect(sql).not.toContain("tc.last_date > ?");
    expect(sql).not.toContain("tc.last_date < ?");
  });

  it("selects expected columns", () => {
    const sql = getConversationsQuery({});
    expect(sql).toContain("chat.ROWID as chat_rowid");
    expect(sql).toContain("chat.guid as chat_id");
    expect(sql).toContain("chat.display_name");
    expect(sql).toContain("chat.chat_identifier");
    expect(sql).toContain("participants");
    expect(sql).toContain("message_count");
    expect(sql).toContain("last_message_date");
    expect(sql).toContain("last_message_text");
    expect(sql).toContain("last_message_attributed_body");
  });
});

// ---------------------------------------------------------------------------
// getMessagesQuery
// ---------------------------------------------------------------------------

describe("getMessagesQuery", () => {
  it("returns valid SQL with no before filter", () => {
    const sql = getMessagesQuery({});
    expect(sql).toContain("SELECT");
    expect(sql).toContain("FROM message");
    expect(sql).toContain("WHERE chat.guid = ?");
    expect(sql).toContain("ORDER BY message.date DESC");
    expect(sql).toContain("LIMIT ?");
    expect(sql).not.toContain("AND message.date < ?");
  });

  it("includes before filter when withBeforeFilter is true", () => {
    const sql = getMessagesQuery({ withBeforeFilter: true });
    expect(sql).toContain("AND message.date < ?");
  });

  it("omits before filter when withBeforeFilter is false", () => {
    const sql = getMessagesQuery({ withBeforeFilter: false });
    expect(sql).not.toContain("AND message.date < ?");
  });

  it("selects expected columns", () => {
    const sql = getMessagesQuery({});
    expect(sql).toContain("message.ROWID as id");
    expect(sql).toContain("message.guid");
    expect(sql).toContain("chat.guid as chat_id");
    expect(sql).toContain("COALESCE(handle.id, 'me') as sender");
    expect(sql).toContain("message.attributedBody");
    expect(sql).toContain("message.date as date");
    expect(sql).toContain("message.is_from_me");
    expect(sql).toContain("message.thread_originator_guid as reply_to_guid");
    expect(sql).toContain("message.date_delivered");
    expect(sql).toContain("message.date_read");
    expect(sql).toContain("message.group_action_type");
    expect(sql).toContain("affected_handle.id as affected_handle_id");
  });

  it("joins on chat_message_join, chat, handle, and affected_handle", () => {
    const sql = getMessagesQuery({});
    expect(sql).toContain("JOIN chat_message_join");
    expect(sql).toContain("JOIN chat ON");
    expect(sql).toContain("LEFT JOIN handle ON");
    expect(sql).toContain("LEFT JOIN handle AS affected_handle");
  });
});

// ---------------------------------------------------------------------------
// getNewMessagesQuery
// ---------------------------------------------------------------------------

describe("getNewMessagesQuery", () => {
  it("returns valid SQL", () => {
    const sql = getNewMessagesQuery();
    expect(sql).toContain("SELECT");
    expect(sql).toContain("FROM message");
    expect(sql).toContain("WHERE message.ROWID > ?");
    expect(sql).toContain("ORDER BY message.date ASC");
  });

  it("does not include LIMIT clause", () => {
    const sql = getNewMessagesQuery();
    expect(sql).not.toContain("LIMIT");
  });
});

// ---------------------------------------------------------------------------
// Static SQL constants
// ---------------------------------------------------------------------------

describe("static SQL constants", () => {
  it("DETECT_SCHEMA_SQL references pragma_table_info", () => {
    expect(DETECT_SCHEMA_SQL).toContain("pragma_table_info('chat')");
    expect(DETECT_SCHEMA_SQL).toContain("service_name");
  });

  it("ATTACHMENTS_QUERY selects expected columns", () => {
    expect(ATTACHMENTS_QUERY).toContain("attachment.filename");
    expect(ATTACHMENTS_QUERY).toContain("attachment.mime_type");
    expect(ATTACHMENTS_QUERY).toContain("attachment.total_bytes as file_size");
    expect(ATTACHMENTS_QUERY).toContain("attachment.transfer_name");
    expect(ATTACHMENTS_QUERY).toContain("message_attachment_join.message_id = ?");
  });

  it("REACTIONS_QUERY filters by associated_message_guid", () => {
    expect(REACTIONS_QUERY).toContain("message.associated_message_guid = ?");
    expect(REACTIONS_QUERY).toContain("associated_message_type != 0");
  });

  it("MESSAGE_BY_GUID_QUERY looks up by guid", () => {
    expect(MESSAGE_BY_GUID_QUERY).toContain("message.guid = ?");
    expect(MESSAGE_BY_GUID_QUERY).toContain("LIMIT 1");
  });

  it("LAST_MESSAGE_ROWID_QUERY selects MAX(ROWID)", () => {
    expect(LAST_MESSAGE_ROWID_QUERY).toContain("MAX(ROWID) as last_rowid");
  });
});

// ---------------------------------------------------------------------------
// APPLE_EPOCH_OFFSET constant
// ---------------------------------------------------------------------------

describe("APPLE_EPOCH_OFFSET", () => {
  it("equals the number of seconds between 1970-01-01 and 2001-01-01", () => {
    // 2001-01-01T00:00:00Z in Unix seconds
    const expected = new Date("2001-01-01T00:00:00Z").getTime() / 1000;
    expect(APPLE_EPOCH_OFFSET).toBe(expected);
  });

  it("is exactly 978307200", () => {
    expect(APPLE_EPOCH_OFFSET).toBe(978307200);
  });
});
