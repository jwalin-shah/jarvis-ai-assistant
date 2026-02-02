/**
 * Database access layer
 * Provides direct SQLite access for fast reads with HTTP API fallback
 */

export {
  initDatabases,
  closeDatabases,
  isDirectAccessAvailable,
  getInitError,
  getConversations,
  getMessages,
  getMessage,
  getLastMessageRowid,
  getNewMessagesSince,
} from "./direct";

export {
  parseAppleTimestamp,
  toAppleTimestamp,
  formatDate,
  normalizePhoneNumber,
  type SchemaVersion,
} from "./queries";
