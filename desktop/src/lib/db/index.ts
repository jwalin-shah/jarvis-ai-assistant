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
  getMessagesBatch,
  getLastMessageRowid,
  getNewMessagesSince,
  populateContactsCache,
  isContactsCacheLoaded,
  loadContactsFromAddressBook,
  resolveContactName,
  formatParticipant,
} from "./direct";

export {
  parseAppleTimestamp,
  toAppleTimestamp,
  formatDate,
  normalizePhoneNumber,
  type SchemaVersion,
} from "./queries";
