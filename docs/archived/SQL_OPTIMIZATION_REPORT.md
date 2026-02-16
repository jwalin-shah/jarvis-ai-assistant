# SQL Optimization Report

Generated: 2026-02-10 16:48:06Z

## Scope

- Source set: all tracked `.py` files (`git ls-files "*.py"`).
- Query sources included: SQL passed to `execute`/`executemany`/`executescript`, plus SQL template definitions in `integrations/imessage/queries.py` and `jarvis/db/schema.py`.
- SQL statements analyzed: **399**.
- Analysis dimensions per query: N+1 pattern risk, missing index risk, full-table scan risk, JOIN efficiency.

## Summary Findings

- Potential N+1 query patterns: **11**
- Queries with missing-index candidates: **34**
- Queries with full-table scan risk: **65**
- Queries with JOIN efficiency risk: **4**

Highest-impact hotspots (manual validation):

- `jarvis/contacts/fact_extractor.py:782`: full contacts scan invoked per entity resolution call (N+1 behavior).
- `jarvis/tags/manager.py:736` and `jarvis/tags/manager.py:796`: looped per-row inserts, should batch.
- `jarvis/eval/feedback.py:460`: bulk path currently loops `execute`, should use `executemany`.
- `jarvis/search/embeddings.py:743`: context fetch query executes per similar message.
- `jarvis/classifiers/relationship_classifier.py:452`: correlated COUNT subqueries instead of grouped JOIN aggregation.

## Schema Review (`jarvis/db/schema.py`)

Existing core indexes are solid. The following are the most valuable missing additions based on observed query predicates:

1. `CREATE INDEX IF NOT EXISTS idx_pairs_holdout_quality_ts ON pairs(is_holdout, quality_score, trigger_timestamp DESC);`
2. `CREATE INDEX IF NOT EXISTS idx_pairs_response_da_rank ON pairs(response_da_type, is_holdout, response_da_conf DESC, quality_score DESC);`
3. `CREATE INDEX IF NOT EXISTS idx_pairs_trigger_da_rank ON pairs(trigger_da_type, is_holdout, trigger_da_conf DESC, quality_score DESC);`
4. `CREATE INDEX IF NOT EXISTS idx_contacts_phone_or_email ON contacts(phone_or_email);`
5. `CREATE INDEX IF NOT EXISTS idx_contacts_display_name_ci ON contacts(LOWER(display_name));`

Optional lower-priority indexes (add only if query plans show need):

- `CREATE INDEX IF NOT EXISTS idx_pair_artifacts_gate_reason ON pair_artifacts(gate_a_reason);`
- `CREATE INDEX IF NOT EXISTS idx_index_versions_active ON index_versions(is_active);`

## Detailed Query Review

Each entry includes location, current query, issues found, optimized approach, and suggested indexes.

### `api/routers/graph.py:304`

**Current query**

```sql
SELECT category, subject, predicate, value, confidence FROM contact_facts WHERE contact_id = ? ORDER BY confidence DESC
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `archive/scripts/experiment_clustering.py:191`

**Current query**

```sql
SELECT message.text, message.attributedBody, message.is_from_me FROM message WHERE (message.text IS NOT NULL AND message.text != '') OR message.attributedBody IS NOT NULL ORDER BY message.date DESC
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `core/health/schema.py:180`

**Current query**

```sql
SELECT name FROM sqlite_master WHERE type='table' ORDER BY name
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `evals/benchmarks/templates/mine.py:123`

**Current query**

```sql
SELECT text FROM message WHERE is_from_me = 1 AND text IS NOT NULL AND text != '' AND length(text) > 0 ORDER BY RANDOM()
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `evals/rag_eval.py:271`

**Current query**

```sql
SELECT rowid, trigger_text, response_text FROM vec_chunks WHERE trigger_text IS NOT NULL AND response_text IS NOT NULL ORDER BY RANDOM() LIMIT ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `integrations/imessage/avatar.py:144`

**Current query**

```sql
SELECT ZABCDRECORD.ZTHUMBNAILIMAGEDATA as image_data, ZABCDRECORD.ZFIRSTNAME as first_name, ZABCDRECORD.ZLASTNAME as last_name, ZABCDRECORD.ZDISPLAYNAME as display_name, ZABCDPHONENUMBER.ZFULLNUMBER as phone_number FROM ZABCDPHONENUMBER JOIN ZABCDRECORD ON ZABCDPHONENUMBER.ZOWNER = ZABCDRECORD.Z_PK WHERE ZABCDPHONENUMBER.ZFULLNUMBER IS NOT NULL AND REPLACE(REPLACE(REPLACE(REPLACE( ZABCDPHONENUMBER.ZFULLNUMBER, ' ', ''), '-', ''), '(', ''), ')', '') LIKE ?
```

**Issues found**
- LIKE predicate may bypass indexes for leading-wildcard search values.
- JOIN index coverage not fully verifiable for external-schema tables.

**Optimized query / approach**
- Prefer prefix-search patterns or FTS for substring matching.

**Suggested indexes**
- None.

### `integrations/imessage/avatar.py:195`

**Current query**

```sql
SELECT ZABCDRECORD.ZTHUMBNAILIMAGEDATA as image_data, ZABCDRECORD.ZFIRSTNAME as first_name, ZABCDRECORD.ZLASTNAME as last_name, ZABCDRECORD.ZDISPLAYNAME as display_name FROM ZABCDEMAILADDRESS JOIN ZABCDRECORD ON ZABCDEMAILADDRESS.ZOWNER = ZABCDRECORD.Z_PK WHERE LOWER(ZABCDEMAILADDRESS.ZADDRESS) = ?
```

**Issues found**
- Potential non-sargable predicate: function-wrapped column in WHERE.
- JOIN index coverage not fully verifiable for external-schema tables.

**Optimized query / approach**
- Use an expression index or normalized shadow column for this predicate.

**Suggested indexes**
- None.

### `integrations/imessage/queries.py:50`

**Current query**

```sql
WITH message_ranked AS ( SELECT cmj.chat_id, m.text, m.attributedBody, m.date, ROW_NUMBER() OVER (PARTITION BY cmj.chat_id ORDER BY m.date DESC) as msg_rank, COUNT(*) OVER (PARTITION BY cmj.chat_id) as message_count FROM chat_message_join cmj JOIN message m ON cmj.message_id = m.ROWID ), chat_participants AS ( SELECT chat_handle_join.chat_id, GROUP_CONCAT(handle.id, ', ') as participants FROM chat_handle_join JOIN handle ON chat_handle_join.handle_id = handle.ROWID GROUP BY chat_handle_join.chat_id ) SELECT chat.ROWID as chat_rowid, chat.guid as chat_id, chat.display_name, chat.chat_identifier, cp.participants, COALESCE(mr.message_count, 0) as message_count, mr.date as last_message_date, mr.text as last_message_text, mr.attributedBody as last_message_attributed_body FROM chat LEFT JOIN chat_participants cp ON cp.chat_id = chat.ROWID LEFT JOIN message_ranked mr ON mr.chat_id = chat.ROWID AND mr.msg_rank = 1 WHERE COALESCE(mr.message_count, 0) > 0 {since_filter} {before_filter} ORDER BY mr.date DESC LIMIT ?
```

**Issues found**
- JOIN index coverage not fully verifiable for external-schema tables.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `integrations/imessage/queries.py:89`

**Current query**

```sql
SELECT message.ROWID as id, chat.guid as chat_id, COALESCE(handle.id, 'me') as sender, CASE WHEN message.text IS NOT NULL AND message.text != '' THEN message.text ELSE NULL END as text, message.attributedBody, message.date as date, message.is_from_me, message.thread_originator_guid as reply_to_guid, message.date_delivered, message.date_read, message.group_action_type, affected_handle.id as affected_handle_id FROM message JOIN chat_message_join ON message.ROWID = chat_message_join.message_id JOIN chat ON chat_message_join.chat_id = chat.ROWID LEFT JOIN handle ON message.handle_id = handle.ROWID LEFT JOIN handle AS affected_handle ON message.other_handle = affected_handle.ROWID WHERE chat.guid = ? {before_filter} ORDER BY message.date DESC LIMIT ?
```

**Issues found**
- JOIN index coverage not fully verifiable for external-schema tables.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `integrations/imessage/queries.py:117`

**Current query**

```sql
SELECT message.ROWID as id, chat.guid as chat_id, COALESCE(handle.id, 'me') as sender, message.text, message.attributedBody, message.date, message.is_from_me, message.thread_originator_guid as reply_to_guid FROM message JOIN chat_message_join ON message.ROWID = chat_message_join.message_id JOIN chat ON chat_message_join.chat_id = chat.ROWID LEFT JOIN handle ON message.handle_id = handle.ROWID WHERE message.text LIKE ? ESCAPE '\' {sender_filter} {after_filter} {before_filter} {chat_id_filter} {has_attachments_filter} ORDER BY message.date DESC LIMIT ?
```

**Issues found**
- LIKE predicate may bypass indexes for leading-wildcard search values.
- JOIN index coverage not fully verifiable for external-schema tables.

**Optimized query / approach**
- Prefer prefix-search patterns or FTS for substring matching.

**Suggested indexes**
- None.

### `integrations/imessage/queries.py:140`

**Current query**

```sql
SELECT message.ROWID as id, chat.guid as chat_id, COALESCE(handle.id, 'me') as sender, message.text, message.attributedBody, message.date, message.is_from_me, message.thread_originator_guid as reply_to_guid, message.group_action_type, affected_handle.id as affected_handle_id FROM message JOIN chat_message_join ON message.ROWID = chat_message_join.message_id JOIN chat ON chat_message_join.chat_id = chat.ROWID LEFT JOIN handle ON message.handle_id = handle.ROWID LEFT JOIN handle AS affected_handle ON message.other_handle = affected_handle.ROWID WHERE chat.guid = ? ORDER BY ABS(message.ROWID - ?) LIMIT ?
```

**Issues found**
- JOIN index coverage not fully verifiable for external-schema tables.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `integrations/imessage/queries.py:161`

**Current query**

```sql
SELECT attachment.ROWID as attachment_id, attachment.filename, attachment.mime_type, attachment.total_bytes as file_size, attachment.transfer_name FROM attachment JOIN message_attachment_join ON attachment.ROWID = message_attachment_join.attachment_id WHERE message_attachment_join.message_id = ?
```

**Issues found**
- JOIN index coverage not fully verifiable for external-schema tables.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `integrations/imessage/queries.py:172`

**Current query**

```sql
SELECT attachment.ROWID as attachment_id, attachment.filename, attachment.mime_type, attachment.total_bytes as file_size, attachment.transfer_name, message_attachment_join.message_id FROM attachment JOIN message_attachment_join ON attachment.ROWID = message_attachment_join.attachment_id WHERE message_attachment_join.message_id IN ({placeholders})
```

**Issues found**
- JOIN index coverage not fully verifiable for external-schema tables.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `integrations/imessage/queries.py:184`

**Current query**

```sql
SELECT attachment.ROWID as attachment_id, attachment.filename, attachment.mime_type, attachment.total_bytes as file_size, attachment.transfer_name, attachment.width, attachment.height, attachment.uti, attachment.is_sticker, attachment.created_date FROM attachment JOIN message_attachment_join ON attachment.ROWID = message_attachment_join.attachment_id WHERE message_attachment_join.message_id = ?
```

**Issues found**
- JOIN index coverage not fully verifiable for external-schema tables.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `integrations/imessage/queries.py:200`

**Current query**

```sql
SELECT attachment.ROWID as attachment_id, attachment.filename, attachment.mime_type, attachment.total_bytes as file_size, attachment.transfer_name, attachment.width, attachment.height, attachment.uti, attachment.is_sticker, attachment.created_date, message.ROWID as message_id, message.date as message_date, chat.guid as chat_id, COALESCE(handle.id, 'me') as sender, message.is_from_me FROM attachment JOIN message_attachment_join ON attachment.ROWID = message_attachment_join.attachment_id JOIN message ON message_attachment_join.message_id = message.ROWID JOIN chat_message_join ON message.ROWID = chat_message_join.message_id JOIN chat ON chat_message_join.chat_id = chat.ROWID LEFT JOIN handle ON message.handle_id = handle.ROWID {chat_filter} {type_filter} {date_after_filter} {date_before_filter} ORDER BY message.date DESC LIMIT ?
```

**Issues found**
- JOIN index coverage not fully verifiable for external-schema tables.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `integrations/imessage/queries.py:230`

**Current query**

```sql
SELECT COUNT(*) as total_count, COALESCE(SUM(attachment.total_bytes), 0) as total_size, attachment.mime_type FROM attachment JOIN message_attachment_join ON attachment.ROWID = message_attachment_join.attachment_id JOIN message ON message_attachment_join.message_id = message.ROWID JOIN chat_message_join ON message.ROWID = chat_message_join.message_id JOIN chat ON chat_message_join.chat_id = chat.ROWID WHERE chat.guid = ? GROUP BY CASE WHEN attachment.mime_type LIKE 'image/%' THEN 'images' WHEN attachment.mime_type LIKE 'video/%' THEN 'videos' WHEN attachment.mime_type LIKE 'audio/%' THEN 'audio' WHEN attachment.mime_type IN ('application/pdf', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain', 'application/rtf') THEN 'documents' ELSE 'other' END
```

**Issues found**
- JOIN index coverage not fully verifiable for external-schema tables.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `integrations/imessage/queries.py:252`

**Current query**

```sql
SELECT chat.guid as chat_id, chat.display_name, COUNT(DISTINCT attachment.ROWID) as attachment_count, COALESCE(SUM(attachment.total_bytes), 0) as total_size FROM chat JOIN chat_message_join ON chat.ROWID = chat_message_join.chat_id JOIN message ON chat_message_join.message_id = message.ROWID JOIN message_attachment_join ON message.ROWID = message_attachment_join.message_id JOIN attachment ON message_attachment_join.attachment_id = attachment.ROWID GROUP BY chat.guid ORDER BY total_size DESC LIMIT ?
```

**Issues found**
- JOIN index coverage not fully verifiable for external-schema tables.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `integrations/imessage/queries.py:267`

**Current query**

```sql
SELECT message.ROWID as id, message.associated_message_type, message.date, message.is_from_me, COALESCE(handle.id, 'me') as sender FROM message LEFT JOIN handle ON message.handle_id = handle.ROWID WHERE message.associated_message_guid = ? AND message.associated_message_type != 0
```

**Issues found**
- JOIN index coverage not fully verifiable for external-schema tables.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `integrations/imessage/queries.py:279`

**Current query**

```sql
SELECT message.ROWID as id, message.associated_message_type, message.associated_message_guid, message.date, message.is_from_me, COALESCE(handle.id, 'me') as sender FROM message LEFT JOIN handle ON message.handle_id = handle.ROWID WHERE message.associated_message_guid IN ({placeholders}) AND message.associated_message_type != 0
```

**Issues found**
- JOIN index coverage not fully verifiable for external-schema tables.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `integrations/imessage/queries.py:292`

**Current query**

```sql
SELECT message.ROWID as id, message.guid FROM message WHERE message.ROWID IN ({placeholders})
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `integrations/imessage/queries.py:297`

**Current query**

```sql
SELECT message.ROWID as id FROM message WHERE message.guid = ? LIMIT 1
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `integrations/imessage/queries.py:303`

**Current query**

```sql
SELECT message.ROWID as id, chat.guid as chat_id, COALESCE(handle.id, 'me') as sender, message.text, message.attributedBody, message.date, message.is_from_me, message.thread_originator_guid as reply_to_guid FROM message JOIN chat_message_join ON message.ROWID = chat_message_join.message_id JOIN chat ON chat_message_join.chat_id = chat.ROWID LEFT JOIN handle ON message.handle_id = handle.ROWID WHERE message.ROWID > ? AND chat.guid = ? ORDER BY message.date ASC LIMIT ?
```

**Issues found**
- JOIN index coverage not fully verifiable for external-schema tables.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `integrations/imessage/reader.py:863`

**Current query**

```sql
SELECT ZABCDPHONENUMBER.ZFULLNUMBER as identifier, ZABCDRECORD.ZFIRSTNAME as first_name, ZABCDRECORD.ZLASTNAME as last_name FROM ZABCDPHONENUMBER JOIN ZABCDRECORD ON ZABCDPHONENUMBER.ZOWNER = ZABCDRECORD.Z_PK WHERE ZABCDPHONENUMBER.ZFULLNUMBER IS NOT NULL
```

**Issues found**
- JOIN index coverage not fully verifiable for external-schema tables.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `integrations/imessage/reader.py:884`

**Current query**

```sql
SELECT ZABCDEMAILADDRESS.ZADDRESS as identifier, ZABCDRECORD.ZFIRSTNAME as first_name, ZABCDRECORD.ZLASTNAME as last_name FROM ZABCDEMAILADDRESS JOIN ZABCDRECORD ON ZABCDEMAILADDRESS.ZOWNER = ZABCDRECORD.Z_PK WHERE ZABCDEMAILADDRESS.ZADDRESS IS NOT NULL
```

**Issues found**
- JOIN index coverage not fully verifiable for external-schema tables.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `integrations/imessage/reader.py:962`

**Current query**

```sql
SELECT 1 FROM chat LIMIT 1
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `integrations/imessage/reader.py:1011`

**Current query**

```sql
SELECT 1 FROM chat LIMIT 1
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `integrations/imessage/reader.py:1762`

**Current query**

```sql
SELECT guid FROM message WHERE ROWID = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/_cli_main.py:677`

**Current query**

```sql
SELECT COUNT(*) as cnt FROM vec_chunks
```

**Issues found**
- Potential full-table scan: `COUNT(*)` without WHERE filter.

**Optimized query / approach**
- Keep for explicit global metrics only; otherwise add selective filters.

**Suggested indexes**
- None.

### `jarvis/classifiers/relationship_classifier.py:452`

**Current query**

```sql
SELECT c.chat_identifier, c.display_name, (SELECT COUNT(*) FROM chat_handle_join WHERE chat_id = c.ROWID) as handle_count, (SELECT COUNT(*) FROM chat_message_join WHERE chat_id = c.ROWID) as msg_count FROM chat c WHERE handle_count = 1 ORDER BY msg_count DESC
```

**Issues found**
- JOIN index coverage not fully verifiable for external-schema tables.
- JOIN efficiency: correlated COUNT subqueries per chat row can degrade on large datasets.

**Optimized query / approach**
- Rewrite as grouped JOIN aggregation (single pass) instead of correlated subqueries.

**Suggested indexes**
- None.

### `jarvis/contacts/fact_extractor.py:782`

**Current query**

```sql
SELECT id, display_name FROM contacts
```

**Issues found**
- Potential full-table scan: SELECT without WHERE/LIMIT.
- Potential N+1: `_resolve_person_to_contact()` runs this contacts query once per extracted person entity.

**Optimized query / approach**
- Load contacts once per message/batch and reuse in-memory matching for all entities.

**Suggested indexes**
- None.

### `jarvis/contacts/fact_storage.py:65`

**Current query**

```sql
SELECT COUNT(*) FROM contact_facts WHERE contact_id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/contacts/fact_storage.py:72`

**Current query**

```sql
INSERT OR IGNORE INTO contact_facts (contact_id, category, subject, predicate, value, confidence, source_message_id, source_text, extracted_at, linked_contact_id, valid_from, valid_until) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/contacts/fact_storage.py:84`

**Current query**

```sql
SELECT COUNT(*) FROM contact_facts WHERE contact_id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/contacts/fact_storage.py:109`

**Current query**

```sql
SELECT category, subject, predicate, value, confidence, source_text, source_message_id, extracted_at, valid_from, valid_until FROM contact_facts WHERE contact_id = ? ORDER BY confidence DESC
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/contacts/fact_storage.py:146`

**Current query**

```sql
SELECT contact_id, category, subject, predicate, value, confidence, source_text, source_message_id, extracted_at, valid_from, valid_until FROM contact_facts ORDER BY confidence DESC
```

**Issues found**
- Potential full-table scan: SELECT without WHERE/LIMIT.

**Optimized query / approach**
- Add WHERE and/or LIMIT when full scan is not required.

**Suggested indexes**
- None.

### `jarvis/contacts/fact_storage.py:181`

**Current query**

```sql
DELETE FROM contact_facts WHERE contact_id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/contacts/fact_storage.py:199`

**Current query**

```sql
SELECT COUNT(*) FROM contact_facts
```

**Issues found**
- Potential full-table scan: `COUNT(*)` without WHERE filter.

**Optimized query / approach**
- Keep for explicit global metrics only; otherwise add selective filters.

**Suggested indexes**
- None.

### `jarvis/db/artifacts.py:42`

**Current query**

```sql
INSERT OR REPLACE INTO pair_artifacts (pair_id, context_json, gate_a_reason, gate_c_scores_json, raw_trigger_text, raw_response_text) VALUES (?, ?, ?, ?, ?, ?)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/artifacts.py:70`

**Current query**

```sql
SELECT * FROM pair_artifacts WHERE pair_id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/artifacts.py:86`

**Current query**

```sql
DELETE FROM pair_artifacts
```

**Issues found**
- Full-table delete detected (verify intent).

**Optimized query / approach**
- If not maintenance/reset behavior, add WHERE predicates.

**Suggested indexes**
- None.

### `jarvis/db/artifacts.py:115`

**Current query**

```sql
INSERT OR REPLACE INTO contact_style_targets (contact_id, median_reply_length, punctuation_rate, emoji_rate, greeting_rate, updated_at) VALUES (?, ?, ?, ?, ?, ?)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/artifacts.py:136`

**Current query**

```sql
SELECT * FROM contact_style_targets WHERE contact_id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/artifacts.py:197`

**Current query**

```sql
INSERT INTO pairs (contact_id, trigger_text, response_text, trigger_timestamp, response_timestamp, chat_id, trigger_msg_id, response_msg_id, trigger_msg_ids_json, response_msg_ids_json, quality_score, flags_json, is_group, gate_a_passed, gate_b_score, gate_c_verdict, validity_status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/artifacts.py:231`

**Current query**

```sql
INSERT INTO pair_artifacts (pair_id, context_json, gate_a_reason, gate_c_scores_json, raw_trigger_text, raw_response_text) VALUES (?, ?, ?, ?, ?, ?)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/clusters.py:30`

**Current query**

```sql
INSERT INTO clusters (name, description, example_triggers, example_responses) VALUES (?, ?, ?, ?) ON CONFLICT(name) DO UPDATE SET description = excluded.description, example_triggers = excluded.example_triggers, example_responses = excluded.example_responses
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/clusters.py:42`

**Current query**

```sql
SELECT * FROM clusters WHERE name = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/clusters.py:61`

**Current query**

```sql
SELECT * FROM clusters WHERE id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/clusters.py:70`

**Current query**

```sql
SELECT * FROM clusters WHERE name = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/clusters.py:79`

**Current query**

```sql
SELECT * FROM clusters ORDER BY name
```

**Issues found**
- Potential full-table scan: SELECT without WHERE/LIMIT.

**Optimized query / approach**
- Add WHERE and/or LIMIT when full scan is not required.

**Suggested indexes**
- None.

### `jarvis/db/clusters.py:88`

**Current query**

```sql
UPDATE clusters SET name = ?, description = ? WHERE id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/clusters.py:93`

**Current query**

```sql
UPDATE clusters SET name = ? WHERE id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/clusters.py:102`

**Current query**

```sql
UPDATE pair_embeddings SET cluster_id = NULL
```

**Issues found**
- Full-table update detected (verify intent).

**Optimized query / approach**
- If not maintenance/reset behavior, add WHERE predicates.

**Suggested indexes**
- None.

### `jarvis/db/clusters.py:103`

**Current query**

```sql
DELETE FROM clusters
```

**Issues found**
- Full-table delete detected (verify intent).

**Optimized query / approach**
- If not maintenance/reset behavior, add WHERE predicates.

**Suggested indexes**
- None.

### `jarvis/db/clusters.py:131`

**Current query**

```sql
SELECT * FROM clusters WHERE id IN ({expr})
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/contacts.py:54`

**Current query**

```sql
SELECT id FROM contacts WHERE chat_id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/contacts.py:59`

**Current query**

```sql
UPDATE contacts SET display_name = ?, phone_or_email = ?, relationship = ?, style_notes = ?, handles_json = ?, updated_at = ? WHERE chat_id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/contacts.py:91`

**Current query**

```sql
INSERT INTO contacts (chat_id, display_name, phone_or_email, relationship, style_notes, handles_json) VALUES (?, ?, ?, ?, ?, ?)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/contacts.py:126`

**Current query**

```sql
SELECT id, chat_id, display_name, phone_or_email, relationship, style_notes, handles_json, created_at, updated_at FROM contacts WHERE id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/contacts.py:152`

**Current query**

```sql
SELECT id, chat_id, display_name, phone_or_email, relationship, style_notes, handles_json, created_at, updated_at FROM contacts WHERE chat_id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/contacts.py:164`

**Current query**

```sql
SELECT id, chat_id, display_name, phone_or_email, relationship, style_notes, handles_json, created_at, updated_at FROM contacts WHERE chat_id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/contacts.py:182`

**Current query**

```sql
SELECT id, chat_id, display_name, phone_or_email, relationship, style_notes, handles_json, created_at, updated_at FROM contacts WHERE chat_id = ? OR phone_or_email = ?
```

**Issues found**
- Missing index candidate(s): contacts.phone_or_email

**Optimized query / approach**
- Keep query shape and add supporting index(es) below.

**Suggested indexes**
- `CREATE INDEX IF NOT EXISTS idx_contacts_phone_or_email ON contacts(phone_or_email);`

### `jarvis/db/contacts.py:195`

**Current query**

```sql
SELECT id, chat_id, display_name, phone_or_email, relationship, style_notes, handles_json, created_at, updated_at FROM contacts WHERE LOWER(display_name) = LOWER(?)
```

**Issues found**
- Potential non-sargable predicate: function-wrapped column in WHERE.

**Optimized query / approach**
- Use an expression index or normalized shadow column for this predicate.

**Suggested indexes**
- None.

### `jarvis/db/contacts.py:206`

**Current query**

```sql
SELECT id, chat_id, display_name, phone_or_email, relationship, style_notes, handles_json, created_at, updated_at FROM contacts WHERE LOWER(display_name) LIKE LOWER(?) ESCAPE '\'
```

**Issues found**
- Potential non-sargable predicate: function-wrapped column in WHERE.

**Optimized query / approach**
- Use an expression index or normalized shadow column for this predicate.

**Suggested indexes**
- None.

### `jarvis/db/contacts.py:221`

**Current query**

```sql
SELECT id, chat_id, display_name, phone_or_email, relationship, style_notes, handles_json, created_at, updated_at FROM contacts ORDER BY display_name LIMIT ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/contacts.py:233`

**Current query**

```sql
SELECT chat_id FROM contacts WHERE id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/contacts.py:238`

**Current query**

```sql
DELETE FROM pair_embeddings WHERE pair_id IN (SELECT id FROM pairs WHERE contact_id = ?)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/contacts.py:244`

**Current query**

```sql
DELETE FROM pairs WHERE contact_id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/contacts.py:245`

**Current query**

```sql
DELETE FROM contacts WHERE id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/core.py:186`

**Current query**

```sql
SELECT name FROM sqlite_master WHERE type='table' AND name IN ('vec_chunks', 'vec_messages', 'vec_binary')
```

**Issues found**
- Potential N+1: query executes inside a loop.

**Optimized query / approach**
- Fetch all required rows in one query (`IN (...)`, JOIN, or prefetch) instead of per-iteration calls.

**Suggested indexes**
- None.

### `jarvis/db/core.py:193`

**Current query**

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0( embedding int8[384] distance_metric=L2, contact_id integer partition key, chat_id text, response_da_type text, quality_score float, source_timestamp float, +topic_label text, +trigger_text text, +response_text text, +formatted_text text, +keywords_json text, +message_count integer, +response_da_conf float, +source_type text, +source_id text )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/core.py:217`

**Current query**

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS vec_messages USING vec0( embedding int8[384] distance_metric=L2, chat_id text partition key, +text_preview text, +sender text, +timestamp integer, +is_from_me integer )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/core.py:232`

**Current query**

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS vec_binary USING vec0( embedding bit[384], +chunk_rowid integer, +embedding_int8 blob )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/core.py:257`

**Current query**

```sql
SELECT version FROM schema_version ORDER BY version DESC LIMIT 1
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/core.py:390`

**Current query**

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0( embedding int8[384] distance_metric=L2, contact_id integer partition key, chat_id text, response_da_type text, quality_score float, source_timestamp float, +topic_label text, +trigger_text text, +response_text text, +formatted_text text, +keywords_json text, +message_count integer, +response_da_conf float, +source_type text, +source_id text )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/core.py:411`

**Current query**

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS vec_messages USING vec0( embedding int8[384] distance_metric=L2, chat_id text partition key, +text_preview text, +sender text, +timestamp integer, +is_from_me integer )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/core.py:423`

**Current query**

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS vec_binary USING vec0( embedding bit[384], +chunk_rowid integer, +embedding_int8 blob )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/core.py:446`

**Current query**

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS vec_binary USING vec0( embedding bit[384], +chunk_rowid integer, +embedding_int8 blob )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/core.py:470`

**Current query**

```sql
INSERT OR REPLACE INTO schema_version (version) VALUES (?)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/core.py:500`

**Current query**

```sql
SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/core.py:513`

**Current query**

```sql
SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/embeddings.py:25`

**Current query**

```sql
INSERT OR REPLACE INTO pair_embeddings (pair_id, faiss_id, cluster_id, index_version) VALUES (?, ?, ?, ?)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/embeddings.py:43`

**Current query**

```sql
INSERT OR REPLACE INTO pair_embeddings (pair_id, faiss_id, cluster_id, index_version) VALUES (?, ?, ?, ?)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/embeddings.py:59`

**Current query**

```sql
SELECT * FROM pair_embeddings WHERE pair_id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/embeddings.py:76`

**Current query**

```sql
SELECT p.* FROM pairs p JOIN pair_embeddings e ON p.id = e.pair_id WHERE e.faiss_id = ? AND e.index_version = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/embeddings.py:85`

**Current query**

```sql
SELECT p.* FROM pairs p JOIN pair_embeddings e ON p.id = e.pair_id WHERE e.faiss_id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/embeddings.py:122`

**Current query**

```sql
SELECT p.*, e.faiss_id FROM pairs p JOIN pair_embeddings e ON p.id = e.pair_id WHERE e.faiss_id IN ({expr}) AND e.index_version = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/embeddings.py:131`

**Current query**

```sql
SELECT p.*, e.faiss_id FROM pairs p JOIN pair_embeddings e ON p.id = e.pair_id WHERE e.faiss_id IN ({expr})
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/embeddings.py:166`

**Current query**

```sql
SELECT * FROM pair_embeddings WHERE pair_id IN ({expr})
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/embeddings.py:213`

**Current query**

```sql
SELECT p.*, e.faiss_id, e.cluster_id as embedding_cluster_id, c.name as cluster_name, c.description as cluster_description FROM pairs p JOIN pair_embeddings e ON p.id = e.pair_id LEFT JOIN clusters c ON e.cluster_id = c.id WHERE e.faiss_id IN ({expr}) AND e.index_version = ?
```

**Issues found**
- Missing index candidate(s): pair_embeddings.cluster_id
- JOIN efficiency risk: join/filter columns include non-indexed fields.

**Optimized query / approach**
- Keep query shape and add supporting index(es) below.

**Suggested indexes**
- `CREATE INDEX IF NOT EXISTS idx_pair_embeddings_cluster_id ON pair_embeddings(cluster_id);`

### `jarvis/db/embeddings.py:225`

**Current query**

```sql
SELECT p.*, e.faiss_id, e.cluster_id as embedding_cluster_id, c.name as cluster_name, c.description as cluster_description FROM pairs p JOIN pair_embeddings e ON p.id = e.pair_id LEFT JOIN clusters c ON e.cluster_id = c.id WHERE e.faiss_id IN ({expr})
```

**Issues found**
- Missing index candidate(s): pair_embeddings.cluster_id
- JOIN efficiency risk: join/filter columns include non-indexed fields.

**Optimized query / approach**
- Keep query shape and add supporting index(es) below.

**Suggested indexes**
- `CREATE INDEX IF NOT EXISTS idx_pair_embeddings_cluster_id ON pair_embeddings(cluster_id);`

### `jarvis/db/embeddings.py:254`

**Current query**

```sql
DELETE FROM pair_embeddings WHERE index_version = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/embeddings.py:259`

**Current query**

```sql
DELETE FROM pair_embeddings
```

**Issues found**
- Full-table delete detected (verify intent).

**Optimized query / approach**
- If not maintenance/reset behavior, add WHERE predicates.

**Suggested indexes**
- None.

### `jarvis/db/embeddings.py:266`

**Current query**

```sql
SELECT COUNT(*) as cnt FROM pair_embeddings WHERE index_version = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/embeddings.py:271`

**Current query**

```sql
SELECT COUNT(*) as cnt FROM pair_embeddings
```

**Issues found**
- Potential full-table scan: `COUNT(*)` without WHERE filter.

**Optimized query / approach**
- Keep for explicit global metrics only; otherwise add selective filters.

**Suggested indexes**
- None.

### `jarvis/db/index_versions.py:29`

**Current query**

```sql
UPDATE index_versions SET is_active = FALSE
```

**Issues found**
- Full-table update detected (verify intent).

**Optimized query / approach**
- If not maintenance/reset behavior, add WHERE predicates.

**Suggested indexes**
- None.

### `jarvis/db/index_versions.py:31`

**Current query**

```sql
INSERT INTO index_versions (version_id, model_name, embedding_dim, num_vectors, index_path, is_active) VALUES (?, ?, ?, ?, ?, ?)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/index_versions.py:53`

**Current query**

```sql
SELECT * FROM index_versions WHERE is_active = TRUE LIMIT 1
```

**Issues found**
- Missing index candidate(s): index_versions.is_active

**Optimized query / approach**
- Keep query shape and add supporting index(es) below.

**Suggested indexes**
- `CREATE INDEX IF NOT EXISTS idx_index_versions_is_active ON index_versions(is_active);`

### `jarvis/db/index_versions.py:71`

**Current query**

```sql
UPDATE index_versions SET is_active = FALSE
```

**Issues found**
- Full-table update detected (verify intent).

**Optimized query / approach**
- If not maintenance/reset behavior, add WHERE predicates.

**Suggested indexes**
- None.

### `jarvis/db/index_versions.py:72`

**Current query**

```sql
UPDATE index_versions SET is_active = TRUE WHERE version_id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/index_versions.py:81`

**Current query**

```sql
SELECT * FROM index_versions ORDER BY created_at DESC
```

**Issues found**
- Potential full-table scan: SELECT without WHERE/LIMIT.

**Optimized query / approach**
- Add WHERE and/or LIMIT when full scan is not required.

**Suggested indexes**
- None.

### `jarvis/db/pairs.py:46`

**Current query**

```sql
SELECT content_hash FROM pairs WHERE content_hash IN ({expr})
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/pairs.py:95`

**Current query**

```sql
INSERT INTO pairs (contact_id, trigger_text, response_text, trigger_timestamp, response_timestamp, chat_id, trigger_msg_id, response_msg_id, trigger_msg_ids_json, response_msg_ids_json, context_text, quality_score, flags_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/pairs.py:213`

**Current query**

```sql
INSERT OR IGNORE INTO pairs (contact_id, trigger_text, response_text, trigger_timestamp, response_timestamp, chat_id, trigger_msg_id, response_msg_id, trigger_msg_ids_json, response_msg_ids_json, context_text, quality_score, flags_json, is_group, source_timestamp, content_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/pairs.py:250`

**Current query**

```sql
SELECT id, contact_id, trigger_text, response_text, trigger_timestamp, response_timestamp, chat_id, trigger_msg_id, response_msg_id, trigger_msg_ids_json, response_msg_ids_json, context_text, quality_score, flags_json, is_group, is_holdout, gate_a_passed, gate_b_score, gate_c_verdict, validity_status, trigger_da_type, trigger_da_conf, response_da_type, response_da_conf, cluster_id FROM pairs WHERE {expr} ORDER BY trigger_timestamp DESC LIMIT ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/pairs.py:273`

**Current query**

```sql
SELECT id, contact_id, trigger_text, response_text, trigger_timestamp, response_timestamp, chat_id, trigger_msg_id, response_msg_id, trigger_msg_ids_json, response_msg_ids_json, context_text, quality_score, flags_json, is_group, is_holdout, gate_a_passed, gate_b_score, gate_c_verdict, validity_status, trigger_da_type, trigger_da_conf, response_da_type, response_da_conf, cluster_id FROM pairs WHERE id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/pairs.py:310`

**Current query**

```sql
SELECT id, contact_id, trigger_text, response_text, trigger_timestamp, response_timestamp, chat_id, trigger_msg_id, response_msg_id, trigger_msg_ids_json, response_msg_ids_json, context_text, quality_score, flags_json, is_group, is_holdout, gate_a_passed, gate_b_score, gate_c_verdict, validity_status, trigger_da_type, trigger_da_conf, response_da_type, response_da_conf, cluster_id FROM pairs WHERE id IN ({expr})
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/pairs.py:332`

**Current query**

```sql
SELECT COUNT(*) as cnt FROM pairs WHERE quality_score >= ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/pairs.py:345`

**Current query**

```sql
UPDATE pairs SET quality_score = ?, flags_json = ? WHERE id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/pairs.py:350`

**Current query**

```sql
UPDATE pairs SET quality_score = ? WHERE id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/pairs.py:359`

**Current query**

```sql
DELETE FROM pair_embeddings
```

**Issues found**
- Full-table delete detected (verify intent).

**Optimized query / approach**
- If not maintenance/reset behavior, add WHERE predicates.

**Suggested indexes**
- None.

### `jarvis/db/pairs.py:360`

**Current query**

```sql
DELETE FROM pairs
```

**Issues found**
- Full-table delete detected (verify intent).

**Optimized query / approach**
- If not maintenance/reset behavior, add WHERE predicates.

**Suggested indexes**
- None.

### `jarvis/db/pairs.py:377`

**Current query**

```sql
UPDATE pairs SET trigger_da_type = ?, trigger_da_conf = ?, response_da_type = ?, response_da_conf = ? WHERE id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/pairs.py:405`

**Current query**

```sql
UPDATE pairs SET cluster_id = ? WHERE id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/schema.py:4`

**Current query**

```sql
CREATE TABLE IF NOT EXISTS schema_version ( version INTEGER PRIMARY KEY, applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/schema.py:9`

**Current query**

```sql
CREATE TABLE IF NOT EXISTS contacts ( id INTEGER PRIMARY KEY, chat_id TEXT UNIQUE, display_name TEXT NOT NULL, phone_or_email TEXT, handles_json TEXT, relationship TEXT, style_notes TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/schema.py:22`

**Current query**

```sql
CREATE TABLE IF NOT EXISTS contact_style_targets ( contact_id INTEGER PRIMARY KEY REFERENCES contacts(id), median_reply_length INTEGER DEFAULT 10, punctuation_rate REAL DEFAULT 0.5, emoji_rate REAL DEFAULT 0.1, greeting_rate REAL DEFAULT 0.2, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/schema.py:32`

**Current query**

```sql
CREATE TABLE IF NOT EXISTS pairs ( id INTEGER PRIMARY KEY, contact_id INTEGER REFERENCES contacts(id), trigger_text TEXT NOT NULL, response_text TEXT NOT NULL, trigger_timestamp TIMESTAMP, response_timestamp TIMESTAMP, chat_id TEXT, trigger_msg_id INTEGER, response_msg_id INTEGER, trigger_msg_ids_json TEXT, response_msg_ids_json TEXT, context_text TEXT, quality_score REAL DEFAULT 1.0, flags_json TEXT, is_group BOOLEAN DEFAULT FALSE, is_holdout BOOLEAN DEFAULT FALSE, gate_a_passed BOOLEAN, gate_b_score REAL, gate_c_verdict TEXT, validity_status TEXT, trigger_da_type TEXT, trigger_da_conf REAL, response_da_type TEXT, response_da_conf REAL, cluster_id INTEGER, usage_count INTEGER DEFAULT 0, last_used_at TIMESTAMP, last_verified_at TIMESTAMP, source_timestamp TIMESTAMP, content_hash TEXT, UNIQUE(trigger_msg_id, response_msg_id) )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/schema.py:75`

**Current query**

```sql
CREATE TABLE IF NOT EXISTS pair_artifacts ( pair_id INTEGER PRIMARY KEY REFERENCES pairs(id), context_json TEXT, gate_a_reason TEXT, gate_c_scores_json TEXT, raw_trigger_text TEXT, raw_response_text TEXT )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/schema.py:85`

**Current query**

```sql
CREATE TABLE IF NOT EXISTS clusters ( id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, description TEXT, example_triggers TEXT, example_responses TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/schema.py:95`

**Current query**

```sql
CREATE TABLE IF NOT EXISTS pair_embeddings ( pair_id INTEGER PRIMARY KEY REFERENCES pairs(id), faiss_id INTEGER UNIQUE, cluster_id INTEGER REFERENCES clusters(id), index_version TEXT )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/schema.py:104`

**Current query**

```sql
CREATE TABLE IF NOT EXISTS index_versions ( id INTEGER PRIMARY KEY, version_id TEXT UNIQUE NOT NULL, model_name TEXT NOT NULL, embedding_dim INTEGER NOT NULL, num_vectors INTEGER NOT NULL, index_path TEXT NOT NULL, is_active BOOLEAN DEFAULT FALSE, normalized BOOLEAN DEFAULT TRUE, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/schema.py:117`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_pairs_contact ON pairs(contact_id)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/schema.py:120`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_pairs_chat ON pairs(chat_id)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/schema.py:121`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_pairs_quality ON pairs(quality_score)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/schema.py:122`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_pairs_validity ON pairs(validity_status)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/schema.py:123`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_contacts_chat ON contacts(chat_id)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/schema.py:124`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_embeddings_index ON pair_embeddings(index_version)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/schema.py:125`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_embeddings_faiss ON pair_embeddings(faiss_id)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/schema.py:126`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_pairs_trigger_text ON pairs(contact_id, LOWER(TRIM(trigger_text)))
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/schema.py:129`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_pairs_timestamp ON pairs(trigger_timestamp DESC)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/schema.py:130`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_pairs_source_timestamp ON pairs(source_timestamp DESC)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/schema.py:131`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_contacts_id ON contacts(id)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/schema.py:132`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_pairs_chat_timestamp ON pairs(chat_id, trigger_timestamp DESC)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/schema.py:135`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_pairs_contact_quality ON pairs(contact_id, quality_score DESC)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/schema.py:136`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_pairs_content_hash ON pairs(content_hash)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/schema.py:138`

**Current query**

```sql
CREATE TABLE IF NOT EXISTS scheduled_drafts ( id TEXT PRIMARY KEY, draft_id TEXT NOT NULL, contact_id INTEGER REFERENCES contacts(id), chat_id TEXT NOT NULL, message_text TEXT NOT NULL, send_at TIMESTAMP NOT NULL, priority TEXT DEFAULT 'normal', status TEXT DEFAULT 'pending', timezone TEXT, depends_on TEXT, retry_count INTEGER DEFAULT 0, max_retries INTEGER DEFAULT 3, expires_at TIMESTAMP, result_json TEXT, metadata_json TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/schema.py:159`

**Current query**

```sql
CREATE TABLE IF NOT EXISTS contact_timing_prefs ( contact_id INTEGER PRIMARY KEY REFERENCES contacts(id), timezone TEXT, quiet_hours_json TEXT, preferred_hours_json TEXT, optimal_weekdays_json TEXT, avg_response_time_mins REAL, last_interaction TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/schema.py:171`

**Current query**

```sql
CREATE TABLE IF NOT EXISTS send_queue ( id TEXT PRIMARY KEY, scheduled_draft_id TEXT REFERENCES scheduled_drafts(id), status TEXT DEFAULT 'pending', queued_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, sent_at TIMESTAMP, error TEXT, attempts INTEGER DEFAULT 0, next_retry_at TIMESTAMP )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/schema.py:183`

**Current query**

```sql
CREATE TABLE IF NOT EXISTS contact_facts ( id INTEGER PRIMARY KEY AUTOINCREMENT, contact_id TEXT NOT NULL, category TEXT NOT NULL, subject TEXT NOT NULL, predicate TEXT NOT NULL, value TEXT DEFAULT '', confidence REAL DEFAULT 1.0, source_message_id INTEGER, source_text TEXT DEFAULT '', extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, linked_contact_id TEXT, valid_from TIMESTAMP, valid_until TIMESTAMP, UNIQUE(contact_id, category, subject, predicate) )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/schema.py:201`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_facts_contact ON contact_facts(contact_id)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/schema.py:203`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_facts_category ON contact_facts(category)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/schema.py:204`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_facts_linked_contact ON contact_facts(linked_contact_id)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/schema.py:205`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_scheduled_contact ON scheduled_drafts(contact_id)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/schema.py:208`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_scheduled_status ON scheduled_drafts(status)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/schema.py:209`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_scheduled_send_at ON scheduled_drafts(send_at)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/schema.py:210`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_scheduled_priority ON scheduled_drafts(priority, send_at)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/schema.py:211`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_send_queue_status ON send_queue(status)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/schema.py:212`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_send_queue_scheduled ON send_queue(scheduled_draft_id)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/search.py:43`

**Current query**

```sql
SELECT trigger_da_type, COUNT(*) as cnt FROM pairs WHERE trigger_da_type IS NOT NULL GROUP BY trigger_da_type ORDER BY cnt DESC
```

**Issues found**
- Missing index candidate(s): pairs.trigger_da_type

**Optimized query / approach**
- Keep query shape and add supporting index(es) below.

**Suggested indexes**
- `CREATE INDEX IF NOT EXISTS idx_pairs_trigger_da_type ON pairs(trigger_da_type);`

### `jarvis/db/search.py:51`

**Current query**

```sql
SELECT response_da_type, COUNT(*) as cnt FROM pairs WHERE response_da_type IS NOT NULL GROUP BY response_da_type ORDER BY cnt DESC
```

**Issues found**
- Missing index candidate(s): pairs.response_da_type

**Optimized query / approach**
- Keep query shape and add supporting index(es) below.

**Suggested indexes**
- `CREATE INDEX IF NOT EXISTS idx_pairs_response_da_type ON pairs(response_da_type);`

### `jarvis/db/search.py:59`

**Current query**

```sql
SELECT COUNT(*) as cnt FROM pairs WHERE trigger_da_type IS NOT NULL
```

**Issues found**
- Missing index candidate(s): pairs.trigger_da_type

**Optimized query / approach**
- Keep query shape and add supporting index(es) below.

**Suggested indexes**
- `CREATE INDEX IF NOT EXISTS idx_pairs_trigger_da_type ON pairs(trigger_da_type);`

### `jarvis/db/search.py:242`

**Current query**

```sql
SELECT contact_id, COUNT(*) as pair_count FROM pairs WHERE contact_id IS NOT NULL GROUP BY contact_id HAVING pair_count >= ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/search.py:269`

**Current query**

```sql
UPDATE pairs SET is_holdout = FALSE
```

**Issues found**
- Full-table update detected (verify intent).

**Optimized query / approach**
- If not maintenance/reset behavior, add WHERE predicates.

**Suggested indexes**
- None.

### `jarvis/db/search.py:277`

**Current query**

```sql
UPDATE pairs SET is_holdout = TRUE WHERE contact_id IN ({expr})
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/search.py:283`

**Current query**

```sql
SELECT COUNT(*) as cnt FROM pairs WHERE is_holdout = FALSE
```

**Issues found**
- Missing index candidate(s): pairs.is_holdout

**Optimized query / approach**
- Keep query shape and add supporting index(es) below.

**Suggested indexes**
- `CREATE INDEX IF NOT EXISTS idx_pairs_is_holdout ON pairs(is_holdout);`

### `jarvis/db/search.py:286`

**Current query**

```sql
SELECT COUNT(*) as cnt FROM pairs WHERE is_holdout = TRUE
```

**Issues found**
- Missing index candidate(s): pairs.is_holdout

**Optimized query / approach**
- Keep query shape and add supporting index(es) below.

**Suggested indexes**
- `CREATE INDEX IF NOT EXISTS idx_pairs_is_holdout ON pairs(is_holdout);`

### `jarvis/db/search.py:308`

**Current query**

```sql
SELECT * FROM pairs WHERE is_holdout = FALSE AND quality_score >= ? ORDER BY trigger_timestamp DESC
```

**Issues found**
- Missing index candidate(s): pairs.is_holdout

**Optimized query / approach**
- Keep query shape and add supporting index(es) below.

**Suggested indexes**
- `CREATE INDEX IF NOT EXISTS idx_pairs_is_holdout ON pairs(is_holdout);`

### `jarvis/db/search.py:317`

**Current query**

```sql
SELECT * FROM pairs WHERE is_holdout = FALSE AND quality_score >= ? ORDER BY trigger_timestamp DESC LIMIT ?
```

**Issues found**
- Missing index candidate(s): pairs.is_holdout

**Optimized query / approach**
- Keep query shape and add supporting index(es) below.

**Suggested indexes**
- `CREATE INDEX IF NOT EXISTS idx_pairs_is_holdout ON pairs(is_holdout);`

### `jarvis/db/search.py:334`

**Current query**

```sql
SELECT * FROM pairs WHERE is_holdout = TRUE AND quality_score >= ? ORDER BY trigger_timestamp DESC
```

**Issues found**
- Missing index candidate(s): pairs.is_holdout

**Optimized query / approach**
- Keep query shape and add supporting index(es) below.

**Suggested indexes**
- `CREATE INDEX IF NOT EXISTS idx_pairs_is_holdout ON pairs(is_holdout);`

### `jarvis/db/search.py:343`

**Current query**

```sql
SELECT * FROM pairs WHERE is_holdout = TRUE AND quality_score >= ? ORDER BY trigger_timestamp DESC LIMIT ?
```

**Issues found**
- Missing index candidate(s): pairs.is_holdout

**Optimized query / approach**
- Keep query shape and add supporting index(es) below.

**Suggested indexes**
- `CREATE INDEX IF NOT EXISTS idx_pairs_is_holdout ON pairs(is_holdout);`

### `jarvis/db/search.py:360`

**Current query**

```sql
SELECT COUNT(*) as cnt FROM pairs WHERE is_holdout = FALSE
```

**Issues found**
- Missing index candidate(s): pairs.is_holdout

**Optimized query / approach**
- Keep query shape and add supporting index(es) below.

**Suggested indexes**
- `CREATE INDEX IF NOT EXISTS idx_pairs_is_holdout ON pairs(is_holdout);`

### `jarvis/db/search.py:364`

**Current query**

```sql
SELECT COUNT(*) as cnt FROM pairs WHERE is_holdout = TRUE
```

**Issues found**
- Missing index candidate(s): pairs.is_holdout

**Optimized query / approach**
- Keep query shape and add supporting index(es) below.

**Suggested indexes**
- `CREATE INDEX IF NOT EXISTS idx_pairs_is_holdout ON pairs(is_holdout);`

### `jarvis/db/search.py:368`

**Current query**

```sql
SELECT COUNT(DISTINCT contact_id) as cnt FROM pairs WHERE is_holdout = FALSE AND contact_id IS NOT NULL
```

**Issues found**
- Missing index candidate(s): pairs.is_holdout

**Optimized query / approach**
- Keep query shape and add supporting index(es) below.

**Suggested indexes**
- `CREATE INDEX IF NOT EXISTS idx_pairs_is_holdout ON pairs(is_holdout);`

### `jarvis/db/search.py:377`

**Current query**

```sql
SELECT COUNT(DISTINCT contact_id) as cnt FROM pairs WHERE is_holdout = TRUE AND contact_id IS NOT NULL
```

**Issues found**
- Missing index candidate(s): pairs.is_holdout

**Optimized query / approach**
- Keep query shape and add supporting index(es) below.

**Suggested indexes**
- `CREATE INDEX IF NOT EXISTS idx_pairs_is_holdout ON pairs(is_holdout);`

### `jarvis/db/search.py:395`

**Current query**

```sql
SELECT * FROM pairs WHERE validity_status = 'valid' AND quality_score >= ? ORDER BY trigger_timestamp DESC LIMIT ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/search.py:434`

**Current query**

```sql
SELECT * FROM pairs WHERE response_da_type = ? AND response_da_conf >= ? AND quality_score >= ? AND is_holdout = FALSE ORDER BY response_da_conf DESC, quality_score DESC LIMIT ?
```

**Issues found**
- Missing index candidate(s): pairs.is_holdout, pairs.response_da_conf, pairs.response_da_type

**Optimized query / approach**
- Keep query shape and add supporting index(es) below.

**Suggested indexes**
- `CREATE INDEX IF NOT EXISTS idx_pairs_is_holdout ON pairs(is_holdout);`
- `CREATE INDEX IF NOT EXISTS idx_pairs_response_da_conf ON pairs(response_da_conf);`
- `CREATE INDEX IF NOT EXISTS idx_pairs_response_da_type ON pairs(response_da_type);`

### `jarvis/db/search.py:447`

**Current query**

```sql
SELECT * FROM pairs WHERE response_da_type = ? AND response_da_conf >= ? AND quality_score >= ? ORDER BY response_da_conf DESC, quality_score DESC LIMIT ?
```

**Issues found**
- Missing index candidate(s): pairs.response_da_conf, pairs.response_da_type

**Optimized query / approach**
- Keep query shape and add supporting index(es) below.

**Suggested indexes**
- `CREATE INDEX IF NOT EXISTS idx_pairs_response_da_conf ON pairs(response_da_conf);`
- `CREATE INDEX IF NOT EXISTS idx_pairs_response_da_type ON pairs(response_da_type);`

### `jarvis/db/search.py:485`

**Current query**

```sql
SELECT * FROM pairs WHERE trigger_da_type = ? AND trigger_da_conf >= ? AND quality_score >= ? AND is_holdout = FALSE ORDER BY trigger_da_conf DESC, quality_score DESC LIMIT ?
```

**Issues found**
- Missing index candidate(s): pairs.is_holdout, pairs.trigger_da_conf, pairs.trigger_da_type

**Optimized query / approach**
- Keep query shape and add supporting index(es) below.

**Suggested indexes**
- `CREATE INDEX IF NOT EXISTS idx_pairs_is_holdout ON pairs(is_holdout);`
- `CREATE INDEX IF NOT EXISTS idx_pairs_trigger_da_conf ON pairs(trigger_da_conf);`
- `CREATE INDEX IF NOT EXISTS idx_pairs_trigger_da_type ON pairs(trigger_da_type);`

### `jarvis/db/search.py:498`

**Current query**

```sql
SELECT * FROM pairs WHERE trigger_da_type = ? AND trigger_da_conf >= ? AND quality_score >= ? ORDER BY trigger_da_conf DESC, quality_score DESC LIMIT ?
```

**Issues found**
- Missing index candidate(s): pairs.trigger_da_conf, pairs.trigger_da_type

**Optimized query / approach**
- Keep query shape and add supporting index(es) below.

**Suggested indexes**
- `CREATE INDEX IF NOT EXISTS idx_pairs_trigger_da_conf ON pairs(trigger_da_conf);`
- `CREATE INDEX IF NOT EXISTS idx_pairs_trigger_da_type ON pairs(trigger_da_type);`

### `jarvis/db/search.py:538`

**Current query**

```sql
SELECT * FROM pairs WHERE trigger_da_type = ? AND response_da_type = ? AND trigger_da_conf >= ? AND response_da_conf >= ? AND quality_score >= ? AND is_holdout = FALSE ORDER BY quality_score DESC, response_da_conf DESC LIMIT ?
```

**Issues found**
- Missing index candidate(s): pairs.is_holdout, pairs.response_da_conf, pairs.response_da_type, pairs.trigger_da_conf, pairs.trigger_da_type

**Optimized query / approach**
- Keep query shape and add supporting index(es) below.

**Suggested indexes**
- `CREATE INDEX IF NOT EXISTS idx_pairs_is_holdout ON pairs(is_holdout);`
- `CREATE INDEX IF NOT EXISTS idx_pairs_response_da_conf ON pairs(response_da_conf);`
- `CREATE INDEX IF NOT EXISTS idx_pairs_response_da_type ON pairs(response_da_type);`
- `CREATE INDEX IF NOT EXISTS idx_pairs_trigger_da_conf ON pairs(trigger_da_conf);`
- `CREATE INDEX IF NOT EXISTS idx_pairs_trigger_da_type ON pairs(trigger_da_type);`

### `jarvis/db/search.py:553`

**Current query**

```sql
SELECT * FROM pairs WHERE trigger_da_type = ? AND response_da_type = ? AND trigger_da_conf >= ? AND response_da_conf >= ? AND quality_score >= ? ORDER BY quality_score DESC, response_da_conf DESC LIMIT ?
```

**Issues found**
- Missing index candidate(s): pairs.response_da_conf, pairs.response_da_type, pairs.trigger_da_conf, pairs.trigger_da_type

**Optimized query / approach**
- Keep query shape and add supporting index(es) below.

**Suggested indexes**
- `CREATE INDEX IF NOT EXISTS idx_pairs_response_da_conf ON pairs(response_da_conf);`
- `CREATE INDEX IF NOT EXISTS idx_pairs_response_da_type ON pairs(response_da_type);`
- `CREATE INDEX IF NOT EXISTS idx_pairs_trigger_da_conf ON pairs(trigger_da_conf);`
- `CREATE INDEX IF NOT EXISTS idx_pairs_trigger_da_type ON pairs(trigger_da_type);`

### `jarvis/db/search.py:590`

**Current query**

```sql
SELECT * FROM pairs WHERE response_da_type = ? AND response_da_conf >= ? AND quality_score >= ? AND is_holdout = FALSE ORDER BY quality_score DESC, response_da_conf DESC LIMIT ?
```

**Issues found**
- Missing index candidate(s): pairs.is_holdout, pairs.response_da_conf, pairs.response_da_type

**Optimized query / approach**
- Keep query shape and add supporting index(es) below.

**Suggested indexes**
- `CREATE INDEX IF NOT EXISTS idx_pairs_is_holdout ON pairs(is_holdout);`
- `CREATE INDEX IF NOT EXISTS idx_pairs_response_da_conf ON pairs(response_da_conf);`
- `CREATE INDEX IF NOT EXISTS idx_pairs_response_da_type ON pairs(response_da_type);`

### `jarvis/db/search.py:614`

**Current query**

```sql
SELECT trigger_da_type, response_da_type, COUNT(*) as cnt FROM pairs WHERE trigger_da_type IS NOT NULL AND response_da_type IS NOT NULL GROUP BY trigger_da_type, response_da_type ORDER BY trigger_da_type, cnt DESC
```

**Issues found**
- Missing index candidate(s): pairs.response_da_type, pairs.trigger_da_type

**Optimized query / approach**
- Keep query shape and add supporting index(es) below.

**Suggested indexes**
- `CREATE INDEX IF NOT EXISTS idx_pairs_response_da_type ON pairs(response_da_type);`
- `CREATE INDEX IF NOT EXISTS idx_pairs_trigger_da_type ON pairs(trigger_da_type);`

### `jarvis/db/stats.py:35`

**Current query**

```sql
SELECT COUNT(*) as cnt FROM contacts
```

**Issues found**
- Potential full-table scan: `COUNT(*)` without WHERE filter.

**Optimized query / approach**
- Keep for explicit global metrics only; otherwise add selective filters.

**Suggested indexes**
- None.

### `jarvis/db/stats.py:39`

**Current query**

```sql
SELECT COUNT(*) as cnt FROM pairs
```

**Issues found**
- Potential full-table scan: `COUNT(*)` without WHERE filter.

**Optimized query / approach**
- Keep for explicit global metrics only; otherwise add selective filters.

**Suggested indexes**
- None.

### `jarvis/db/stats.py:42`

**Current query**

```sql
SELECT COUNT(*) as cnt FROM pairs WHERE quality_score >= 0.5
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/stats.py:46`

**Current query**

```sql
SELECT COUNT(*) as cnt FROM clusters
```

**Issues found**
- Potential full-table scan: `COUNT(*)` without WHERE filter.

**Optimized query / approach**
- Keep for explicit global metrics only; otherwise add selective filters.

**Suggested indexes**
- None.

### `jarvis/db/stats.py:50`

**Current query**

```sql
SELECT COUNT(*) as cnt FROM pair_embeddings
```

**Issues found**
- Potential full-table scan: `COUNT(*)` without WHERE filter.

**Optimized query / approach**
- Keep for explicit global metrics only; otherwise add selective filters.

**Suggested indexes**
- None.

### `jarvis/db/stats.py:58`

**Current query**

```sql
SELECT c.display_name, COUNT(p.id) as pair_count FROM contacts c LEFT JOIN pairs p ON c.id = p.contact_id GROUP BY c.id ORDER BY pair_count DESC LIMIT 10
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/stats.py:81`

**Current query**

```sql
SELECT COUNT(*) as cnt FROM pairs WHERE validity_status IS NOT NULL
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/db/stats.py:88`

**Current query**

```sql
SELECT COUNT(*) as cnt FROM pairs WHERE validity_status = ?
```

**Issues found**
- Potential N+1: query executes inside a loop.

**Optimized query / approach**
- Fetch all required rows in one query (`IN (...)`, JOIN, or prefetch) instead of per-iteration calls.

**Suggested indexes**
- None.

### `jarvis/db/stats.py:95`

**Current query**

```sql
SELECT COUNT(*) as cnt FROM pairs WHERE gate_a_passed = FALSE
```

**Issues found**
- Missing index candidate(s): pairs.gate_a_passed

**Optimized query / approach**
- Keep query shape and add supporting index(es) below.

**Suggested indexes**
- `CREATE INDEX IF NOT EXISTS idx_pairs_gate_a_passed ON pairs(gate_a_passed);`

### `jarvis/db/stats.py:99`

**Current query**

```sql
SELECT gate_a_reason, COUNT(*) as cnt FROM pair_artifacts WHERE gate_a_reason IS NOT NULL GROUP BY gate_a_reason ORDER BY cnt DESC
```

**Issues found**
- Missing index candidate(s): pair_artifacts.gate_a_reason

**Optimized query / approach**
- Keep query shape and add supporting index(es) below.

**Suggested indexes**
- `CREATE INDEX IF NOT EXISTS idx_pair_artifacts_gate_a_reason ON pair_artifacts(gate_a_reason);`

### `jarvis/db/stats.py:111`

**Current query**

```sql
SELECT CASE WHEN gate_b_score >= 0.62 THEN 'accept' WHEN gate_b_score >= 0.48 THEN 'borderline' ELSE 'reject' END as band, COUNT(*) as cnt FROM pairs WHERE gate_b_score IS NOT NULL GROUP BY band
```

**Issues found**
- Missing index candidate(s): pairs.gate_b_score

**Optimized query / approach**
- Keep query shape and add supporting index(es) below.

**Suggested indexes**
- `CREATE INDEX IF NOT EXISTS idx_pairs_gate_b_score ON pairs(gate_b_score);`

### `jarvis/db/stats.py:128`

**Current query**

```sql
SELECT gate_c_verdict, COUNT(*) as cnt FROM pairs WHERE gate_c_verdict IS NOT NULL GROUP BY gate_c_verdict
```

**Issues found**
- Missing index candidate(s): pairs.gate_c_verdict

**Optimized query / approach**
- Keep query shape and add supporting index(es) below.

**Suggested indexes**
- `CREATE INDEX IF NOT EXISTS idx_pairs_gate_c_verdict ON pairs(gate_c_verdict);`

### `jarvis/eval/feedback.py:153`

**Current query**

```sql
CREATE TABLE IF NOT EXISTS feedback ( id INTEGER PRIMARY KEY, message_id TEXT NOT NULL, suggestion_id TEXT NOT NULL, action TEXT NOT NULL CHECK(action IN ('accepted', 'rejected', 'edited')), timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP, metadata_json TEXT, contact_id INTEGER, original_suggestion TEXT, edited_text TEXT, UNIQUE(message_id, suggestion_id) )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/eval/feedback.py:166`

**Current query**

```sql
CREATE TABLE IF NOT EXISTS feedback_schema_version ( version INTEGER PRIMARY KEY, applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/eval/feedback.py:172`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_feedback_message ON feedback(message_id)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/eval/feedback.py:175`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_feedback_suggestion ON feedback(suggestion_id)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/eval/feedback.py:176`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_feedback_action ON feedback(action)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/eval/feedback.py:177`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback(timestamp DESC)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/eval/feedback.py:178`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_feedback_contact ON feedback(contact_id)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/eval/feedback.py:179`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_feedback_timestamp_date ON feedback(DATE(timestamp))
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/eval/feedback.py:253`

**Current query**

```sql
SELECT name FROM sqlite_master WHERE type='table' AND name='feedback_schema_version'
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/eval/feedback.py:259`

**Current query**

```sql
SELECT name FROM sqlite_master WHERE type='table' AND name='feedback'
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/eval/feedback.py:265`

**Current query**

```sql
SELECT MAX(version) FROM feedback_schema_version
```

**Issues found**
- Potential full-table scan: SELECT without WHERE/LIMIT.

**Optimized query / approach**
- Add WHERE and/or LIMIT when full scan is not required.

**Suggested indexes**
- None.

### `jarvis/eval/feedback.py:290`

**Current query**

```sql
CREATE TABLE IF NOT EXISTS feedback_schema_version ( version INTEGER PRIMARY KEY, applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/eval/feedback.py:298`

**Current query**

```sql
INSERT OR REPLACE INTO feedback_schema_version (version) VALUES (?)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/eval/feedback.py:319`

**Current query**

```sql
INSERT INTO feedback_schema_version (version) VALUES (?)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/eval/feedback.py:383`

**Current query**

```sql
INSERT INTO feedback (message_id, suggestion_id, action, timestamp, metadata_json, contact_id, original_suggestion, edited_text) VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(message_id, suggestion_id) DO UPDATE SET action = excluded.action, timestamp = excluded.timestamp, metadata_json = excluded.metadata_json, contact_id = excluded.contact_id, original_suggestion = excluded.original_suggestion, edited_text = excluded.edited_text
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/eval/feedback.py:460`

**Current query**

```sql
INSERT INTO feedback (message_id, suggestion_id, action, timestamp, metadata_json, contact_id, original_suggestion, edited_text) VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(message_id, suggestion_id) DO UPDATE SET action = excluded.action, timestamp = excluded.timestamp, metadata_json = excluded.metadata_json, contact_id = excluded.contact_id, original_suggestion = excluded.original_suggestion, edited_text = excluded.edited_text
```

**Issues found**
- Potential N+1: query executes inside a loop.

**Optimized query / approach**
- Batch parameters and use `executemany()` where possible.

**Suggested indexes**
- None.

### `jarvis/eval/feedback.py:518`

**Current query**

```sql
SELECT id, message_id, suggestion_id, action, timestamp, metadata_json, contact_id, original_suggestion, edited_text FROM feedback WHERE id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/eval/feedback.py:543`

**Current query**

```sql
SELECT id, message_id, suggestion_id, action, timestamp, metadata_json, contact_id, original_suggestion, edited_text FROM feedback WHERE suggestion_id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/eval/feedback.py:568`

**Current query**

```sql
SELECT id, message_id, suggestion_id, action, timestamp, metadata_json, contact_id, original_suggestion, edited_text FROM feedback WHERE message_id = ? ORDER BY timestamp DESC
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/eval/feedback.py:597`

**Current query**

```sql
SELECT id, message_id, suggestion_id, action, timestamp, metadata_json, contact_id, original_suggestion, edited_text FROM feedback WHERE contact_id = ? ORDER BY timestamp DESC LIMIT ? OFFSET ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/eval/feedback.py:712`

**Current query**

```sql
SELECT COUNT(*) FROM feedback
```

**Issues found**
- Potential full-table scan: `COUNT(*)` without WHERE filter.

**Optimized query / approach**
- Keep for explicit global metrics only; otherwise add selective filters.

**Suggested indexes**
- None.

### `jarvis/eval/feedback.py:743`

**Current query**

```sql
DELETE FROM feedback WHERE id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/eval/feedback.py:753`

**Current query**

```sql
DELETE FROM feedback
```

**Issues found**
- Full-table delete detected (verify intent).

**Optimized query / approach**
- If not maintenance/reset behavior, add WHERE predicates.

**Suggested indexes**
- None.

### `jarvis/eval/feedback.py:764`

**Current query**

```sql
SELECT COUNT(*) FROM feedback
```

**Issues found**
- Potential full-table scan: `COUNT(*)` without WHERE filter.

**Optimized query / approach**
- Keep for explicit global metrics only; otherwise add selective filters.

**Suggested indexes**
- None.

### `jarvis/eval/feedback.py:767`

**Current query**

```sql
SELECT action, COUNT(*) as count FROM feedback GROUP BY action
```

**Issues found**
- Potential full-table scan: SELECT without WHERE/LIMIT.

**Optimized query / approach**
- Add WHERE and/or LIMIT when full scan is not required.

**Suggested indexes**
- None.

### `jarvis/eval/feedback.py:800`

**Current query**

```sql
SELECT DATE(timestamp) as date, COUNT(*) as total, SUM(CASE WHEN action = 'accepted' THEN 1 ELSE 0 END) as accepted, SUM(CASE WHEN action = 'rejected' THEN 1 ELSE 0 END) as rejected, SUM(CASE WHEN action = 'edited' THEN 1 ELSE 0 END) as edited FROM feedback WHERE timestamp >= ? GROUP BY DATE(timestamp) ORDER BY DATE(timestamp) DESC
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/eval/feedback.py:840`

**Current query**

```sql
SELECT contact_id, COUNT(*) as total, SUM(CASE WHEN action = 'accepted' THEN 1 ELSE 0 END) as accepted, SUM(CASE WHEN action = 'rejected' THEN 1 ELSE 0 END) as rejected, SUM(CASE WHEN action = 'edited' THEN 1 ELSE 0 END) as edited FROM feedback WHERE contact_id IS NOT NULL GROUP BY contact_id ORDER BY total DESC
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/graph/builder.py:233`

**Current query**

```sql
SELECT id, chat_id, display_name, phone_or_email, relationship, style_notes, handles_json, created_at, updated_at FROM contacts WHERE display_name IS NOT NULL ORDER BY display_name
```

**Issues found**
- Missing index candidate(s): contacts.display_name

**Optimized query / approach**
- Keep query shape and add supporting index(es) below.

**Suggested indexes**
- `CREATE INDEX IF NOT EXISTS idx_contacts_display_name ON contacts(display_name);`

### `jarvis/graph/knowledge_graph.py:104`

**Current query**

```sql
SELECT contact_id, contact_name, relationship, message_count FROM contact_profiles
```

**Issues found**
- Potential full-table scan: SELECT without WHERE/LIMIT.

**Optimized query / approach**
- Add WHERE and/or LIMIT when full scan is not required.

**Suggested indexes**
- None.

### `jarvis/graph/knowledge_graph.py:139`

**Current query**

```sql
SELECT contact_id, category, subject, predicate, value, confidence FROM contact_facts ORDER BY confidence DESC
```

**Issues found**
- Potential full-table scan: SELECT without WHERE/LIMIT.

**Optimized query / approach**
- Add WHERE and/or LIMIT when full scan is not required.

**Suggested indexes**
- None.

### `jarvis/observability/metrics_router.py:135`

**Current query**

```sql
CREATE TABLE IF NOT EXISTS routing_metrics ( id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL NOT NULL, query_hash TEXT NOT NULL, routing_decision TEXT NOT NULL, similarity_score REAL NOT NULL, cache_hit INTEGER NOT NULL, model_loaded INTEGER NOT NULL, embedding_computations INTEGER NOT NULL, faiss_candidates INTEGER NOT NULL, latency_json TEXT NOT NULL, generation_time_ms REAL NOT NULL DEFAULT 0.0, tokens_per_second REAL NOT NULL DEFAULT 0.0 )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/observability/metrics_router.py:172`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_routing_metrics_timestamp ON routing_metrics(timestamp)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/observability/metrics_router.py:178`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_routing_metrics_decision ON routing_metrics(routing_decision)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/observability/metrics_router.py:259`

**Current query**

```sql
INSERT INTO routing_metrics ( timestamp, query_hash, routing_decision, similarity_score, cache_hit, model_loaded, embedding_computations, faiss_candidates, latency_json, generation_time_ms, tokens_per_second, speculative_enabled, draft_acceptance_rate ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/observability/metrics_router.py:459`

**Current query**

```sql
SELECT COUNT(*) as cnt FROM routing_metrics {expr}
```

**Issues found**
- Potential full-table scan: `COUNT(*)` without WHERE filter.

**Optimized query / approach**
- Keep for explicit global metrics only; otherwise add selective filters.

**Suggested indexes**
- None.

### `jarvis/observability/metrics_router.py:571`

**Current query**

```sql
SELECT * FROM routing_metrics ORDER BY timestamp DESC
```

**Issues found**
- Potential full-table scan: SELECT without WHERE/LIMIT.

**Optimized query / approach**
- Add WHERE and/or LIMIT when full scan is not required.

**Suggested indexes**
- None.

### `jarvis/observability/metrics_validation.py:131`

**Current query**

```sql
SELECT timestamp, query_hash, routing_decision FROM routing_metrics WHERE timestamp >= ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/observability/metrics_validation.py:137`

**Current query**

```sql
SELECT timestamp, query_hash, routing_decision FROM routing_metrics
```

**Issues found**
- Potential full-table scan: SELECT without WHERE/LIMIT.

**Optimized query / approach**
- Add WHERE and/or LIMIT when full scan is not required.

**Suggested indexes**
- None.

### `jarvis/prefetch/cache.py:270`

**Current query**

```sql
CREATE TABLE IF NOT EXISTS cache_entries ( key TEXT PRIMARY KEY, value BLOB NOT NULL, value_type TEXT NOT NULL, created_at REAL NOT NULL, accessed_at REAL NOT NULL, ttl_seconds REAL NOT NULL, access_count INTEGER DEFAULT 0, size_bytes INTEGER DEFAULT 0, tags_json TEXT DEFAULT '[]' )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/prefetch/cache.py:281`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_cache_created_at ON cache_entries(created_at)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/prefetch/cache.py:283`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_cache_accessed_at ON cache_entries(accessed_at)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/prefetch/cache.py:284`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_cache_ttl ON cache_entries(created_at, ttl_seconds)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/prefetch/cache.py:301`

**Current query**

```sql
SELECT key, value, value_type, created_at, accessed_at, ttl_seconds, access_count, size_bytes, tags_json FROM cache_entries WHERE key = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/prefetch/cache.py:344`

**Current query**

```sql
UPDATE cache_entries SET accessed_at = ?, access_count = access_count + 1 WHERE key = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/prefetch/cache.py:368`

**Current query**

```sql
INSERT OR REPLACE INTO cache_entries (key, value, value_type, created_at, accessed_at, ttl_seconds, access_count, size_bytes, tags_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/prefetch/cache.py:399`

**Current query**

```sql
DELETE FROM cache_entries WHERE key = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/prefetch/cache.py:414`

**Current query**

```sql
DELETE FROM cache_entries WHERE tags_json LIKE ?
```

**Issues found**
- LIKE predicate may bypass indexes for leading-wildcard search values.
- Missing index candidate(s): cache_entries.tags_json

**Optimized query / approach**
- Prefer prefix-search patterns or FTS for substring matching.

**Suggested indexes**
- `CREATE INDEX IF NOT EXISTS idx_cache_entries_tags_json ON cache_entries(tags_json);`

### `jarvis/prefetch/cache.py:431`

**Current query**

```sql
DELETE FROM cache_entries WHERE (? - created_at) > ttl_seconds
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/prefetch/cache.py:444`

**Current query**

```sql
DELETE FROM cache_entries
```

**Issues found**
- Full-table delete detected (verify intent).

**Optimized query / approach**
- If not maintenance/reset behavior, add WHERE predicates.

**Suggested indexes**
- None.

### `jarvis/prefetch/cache.py:450`

**Current query**

```sql
SELECT key FROM cache_entries
```

**Issues found**
- Potential full-table scan: SELECT without WHERE/LIMIT.

**Optimized query / approach**
- Add WHERE and/or LIMIT when full scan is not required.

**Suggested indexes**
- None.

### `jarvis/prefetch/cache.py:456`

**Current query**

```sql
SELECT COUNT(*) as entries, SUM(size_bytes) as total_bytes, AVG(access_count) as avg_access_count FROM cache_entries
```

**Issues found**
- Potential full-table scan: `COUNT(*)` without WHERE filter.

**Optimized query / approach**
- Keep for explicit global metrics only; otherwise add selective filters.

**Suggested indexes**
- None.

### `jarvis/prefetch/predictor.py:233`

**Current query**

```sql
SELECT chat.guid as chat_id, COUNT(*) as msg_count, MAX(message.date) as last_msg FROM message JOIN chat_message_join ON message.ROWID = chat_message_join.message_id JOIN chat ON chat_message_join.chat_id = chat.ROWID WHERE message.date > ? AND message.is_from_me = 0 GROUP BY chat.guid ORDER BY msg_count DESC LIMIT 50
```

**Issues found**
- JOIN index coverage not fully verifiable for external-schema tables.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/prefetch/predictor.py:353`

**Current query**

```sql
SELECT chat.guid as chat_id, message.date as msg_date FROM message JOIN chat_message_join ON message.ROWID = chat_message_join.message_id JOIN chat ON chat_message_join.chat_id = chat.ROWID WHERE message.date > ? AND message.is_from_me = 0 ORDER BY message.date DESC LIMIT 5000
```

**Issues found**
- JOIN index coverage not fully verifiable for external-schema tables.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/router.py:438`

**Current query**

```sql
SELECT COUNT(*) as cnt FROM vec_chunks
```

**Issues found**
- Potential full-table scan: `COUNT(*)` without WHERE filter.

**Optimized query / approach**
- Keep for explicit global metrics only; otherwise add selective filters.

**Suggested indexes**
- None.

### `jarvis/search/embeddings.py:410`

**Current query**

```sql
CREATE TABLE IF NOT EXISTS message_embeddings ( message_id INTEGER PRIMARY KEY, chat_id TEXT NOT NULL, embedding BLOB NOT NULL, text_hash TEXT NOT NULL, sender TEXT, sender_name TEXT, timestamp INTEGER NOT NULL, is_from_me INTEGER NOT NULL, text_preview TEXT, created_at INTEGER DEFAULT (strftime('%s', 'now')) )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/search/embeddings.py:423`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_embeddings_chat_id ON message_embeddings(chat_id)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/search/embeddings.py:427`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_embeddings_timestamp ON message_embeddings(timestamp)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/search/embeddings.py:429`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_embeddings_sender ON message_embeddings(sender)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/search/embeddings.py:431`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_embeddings_is_from_me ON message_embeddings(is_from_me)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/search/embeddings.py:433`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_embeddings_text_hash ON message_embeddings(text_hash)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/search/embeddings.py:435`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_embeddings_chat_timestamp ON message_embeddings(chat_id, timestamp DESC)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/search/embeddings.py:439`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_embeddings_sender_time ON message_embeddings(is_from_me, timestamp DESC)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/search/embeddings.py:441`

**Current query**

```sql
CREATE TABLE IF NOT EXISTS relationship_profiles ( contact_id TEXT PRIMARY KEY, display_name TEXT, profile_data TEXT NOT NULL, updated_at INTEGER DEFAULT (strftime('%s', 'now')) )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/search/embeddings.py:449`

**Current query**

```sql
CREATE TABLE IF NOT EXISTS index_stats ( stat_key TEXT PRIMARY KEY, stat_value TEXT, updated_at INTEGER DEFAULT (strftime('%s', 'now')) )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/search/embeddings.py:490`

**Current query**

```sql
SELECT 1 FROM message_embeddings WHERE message_id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/search/embeddings.py:503`

**Current query**

```sql
INSERT INTO message_embeddings (message_id, chat_id, embedding, text_hash, sender, sender_name, timestamp, is_from_me, text_preview) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/search/embeddings.py:554`

**Current query**

```sql
SELECT message_id FROM message_embeddings WHERE message_id IN ({expr})
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/search/embeddings.py:593`

**Current query**

```sql
INSERT OR IGNORE INTO message_embeddings (message_id, chat_id, embedding, text_hash, sender, sender_name, timestamp, is_from_me, text_preview) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/search/embeddings.py:635`

**Current query**

```sql
SELECT message_id, chat_id, embedding, text_preview, sender, sender_name, timestamp, is_from_me FROM message_embeddings
```

**Issues found**
- Potential full-table scan: SELECT without WHERE/LIMIT.

**Optimized query / approach**
- Add WHERE and/or LIMIT when full scan is not required.

**Suggested indexes**
- None.

### `jarvis/search/embeddings.py:743`

**Current query**

```sql
SELECT message_id, chat_id, text_preview, sender, sender_name, timestamp, is_from_me FROM message_embeddings WHERE chat_id = ? AND timestamp BETWEEN ? - 3600 AND ? + 3600 ORDER BY timestamp LIMIT ?
```

**Issues found**
- Potential N+1: query executes inside a loop.

**Optimized query / approach**
- Fetch all required rows in one query (`IN (...)`, JOIN, or prefetch) instead of per-iteration calls.

**Suggested indexes**
- None.

### `jarvis/search/embeddings.py:848`

**Current query**

```sql
SELECT COUNT(*) as total, SUM(CASE WHEN is_from_me = 1 THEN 1 ELSE 0 END) as sent, SUM(CASE WHEN is_from_me = 0 THEN 1 ELSE 0 END) as received, AVG(LENGTH(text_preview)) as avg_length, MAX(timestamp) as last_interaction, MIN(sender_name) as display_name FROM message_embeddings WHERE chat_id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/search/embeddings.py:867`

**Current query**

```sql
SELECT text_preview, is_from_me FROM message_embeddings WHERE chat_id = ? ORDER BY timestamp DESC LIMIT 100
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/search/embeddings.py:965`

**Current query**

```sql
SELECT timestamp, is_from_me FROM message_embeddings WHERE chat_id = ? ORDER BY timestamp LIMIT 500
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/search/embeddings.py:1006`

**Current query**

```sql
SELECT COUNT(*) FROM message_embeddings
```

**Issues found**
- Potential full-table scan: `COUNT(*)` without WHERE filter.

**Optimized query / approach**
- Keep for explicit global metrics only; otherwise add selective filters.

**Suggested indexes**
- None.

### `jarvis/search/embeddings.py:1007`

**Current query**

```sql
SELECT COUNT(DISTINCT chat_id) FROM message_embeddings
```

**Issues found**
- Potential full-table scan: SELECT without WHERE/LIMIT.

**Optimized query / approach**
- Add WHERE and/or LIMIT when full scan is not required.

**Suggested indexes**
- None.

### `jarvis/search/embeddings.py:1010`

**Current query**

```sql
SELECT MIN(timestamp) FROM message_embeddings
```

**Issues found**
- Potential full-table scan: SELECT without WHERE/LIMIT.

**Optimized query / approach**
- Add WHERE and/or LIMIT when full scan is not required.

**Suggested indexes**
- None.

### `jarvis/search/embeddings.py:1011`

**Current query**

```sql
SELECT MAX(timestamp) FROM message_embeddings
```

**Issues found**
- Potential full-table scan: SELECT without WHERE/LIMIT.

**Optimized query / approach**
- Add WHERE and/or LIMIT when full scan is not required.

**Suggested indexes**
- None.

### `jarvis/search/embeddings.py:1025`

**Current query**

```sql
DELETE FROM message_embeddings
```

**Issues found**
- Full-table delete detected (verify intent).

**Optimized query / approach**
- If not maintenance/reset behavior, add WHERE predicates.

**Suggested indexes**
- None.

### `jarvis/search/embeddings.py:1026`

**Current query**

```sql
DELETE FROM relationship_profiles
```

**Issues found**
- Full-table delete detected (verify intent).

**Optimized query / approach**
- If not maintenance/reset behavior, add WHERE predicates.

**Suggested indexes**
- None.

### `jarvis/search/embeddings.py:1027`

**Current query**

```sql
DELETE FROM index_stats
```

**Issues found**
- Full-table delete detected (verify intent).

**Optimized query / approach**
- If not maintenance/reset behavior, add WHERE predicates.

**Suggested indexes**
- None.

### `jarvis/search/ingest.py:27`

**Current query**

```sql
SELECT p.ZFULLNUMBER as identifier, c.ZFIRSTNAME as first_name, c.ZLASTNAME as last_name, c.ZORGANIZATION as org_name FROM ZABCDPHONENUMBER p JOIN ZABCDRECORD c ON p.ZOWNER = c.Z_PK UNION ALL SELECT e.ZADDRESS as identifier, c.ZFIRSTNAME as first_name, c.ZLASTNAME as last_name, c.ZORGANIZATION as org_name FROM ZABCDEMAILADDRESS e JOIN ZABCDRECORD c ON e.ZOWNER = c.Z_PK
```

**Issues found**
- Potential N+1: query executes inside a loop.
- Potential full-table scan: SELECT without WHERE/LIMIT.
- JOIN index coverage not fully verifiable for external-schema tables.

**Optimized query / approach**
- Fetch all required rows in one query (`IN (...)`, JOIN, or prefetch) instead of per-iteration calls.

**Suggested indexes**
- None.

### `jarvis/search/semantic_search.py:123`

**Current query**

```sql
CREATE TABLE IF NOT EXISTS schema_version ( version INTEGER PRIMARY KEY )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/search/semantic_search.py:132`

**Current query**

```sql
SELECT version FROM schema_version LIMIT 1
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/search/semantic_search.py:138`

**Current query**

```sql
CREATE TABLE IF NOT EXISTS embeddings ( message_id INTEGER PRIMARY KEY, chat_id TEXT NOT NULL, text_hash TEXT NOT NULL, embedding BLOB NOT NULL, created_at REAL NOT NULL )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/search/semantic_search.py:149`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_embeddings_chat_id ON embeddings(chat_id)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/search/semantic_search.py:155`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_embeddings_text_hash ON embeddings(text_hash)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/search/semantic_search.py:163`

**Current query**

```sql
DELETE FROM schema_version
```

**Issues found**
- Full-table delete detected (verify intent).

**Optimized query / approach**
- If not maintenance/reset behavior, add WHERE predicates.

**Suggested indexes**
- None.

### `jarvis/search/semantic_search.py:164`

**Current query**

```sql
INSERT INTO schema_version (version) VALUES (?)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/search/semantic_search.py:181`

**Current query**

```sql
SELECT embedding FROM embeddings WHERE message_id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/search/semantic_search.py:205`

**Current query**

```sql
SELECT message_id, embedding FROM embeddings WHERE message_id IN ({expr})
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/search/semantic_search.py:231`

**Current query**

```sql
INSERT OR REPLACE INTO embeddings (message_id, chat_id, text_hash, embedding, created_at) VALUES (?, ?, ?, ?, ?)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/search/semantic_search.py:262`

**Current query**

```sql
INSERT OR REPLACE INTO embeddings (message_id, chat_id, text_hash, embedding, created_at) VALUES (?, ?, ?, ?, ?)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/search/semantic_search.py:289`

**Current query**

```sql
DELETE FROM embeddings WHERE message_id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/search/semantic_search.py:303`

**Current query**

```sql
DELETE FROM embeddings WHERE chat_id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/search/semantic_search.py:313`

**Current query**

```sql
DELETE FROM embeddings
```

**Issues found**
- Full-table delete detected (verify intent).

**Optimized query / approach**
- If not maintenance/reset behavior, add WHERE predicates.

**Suggested indexes**
- None.

### `jarvis/search/semantic_search.py:324`

**Current query**

```sql
SELECT COUNT(*) as count FROM embeddings
```

**Issues found**
- Potential full-table scan: `COUNT(*)` without WHERE filter.

**Optimized query / approach**
- Keep for explicit global metrics only; otherwise add selective filters.

**Suggested indexes**
- None.

### `jarvis/search/semantic_search.py:329`

**Current query**

```sql
SELECT SUM(LENGTH(embedding)) as size FROM embeddings
```

**Issues found**
- Potential full-table scan: SELECT without WHERE/LIMIT.

**Optimized query / approach**
- Add WHERE and/or LIMIT when full scan is not required.

**Suggested indexes**
- None.

### `jarvis/search/vec_search.py:113`

**Current query**

```sql
SELECT name FROM sqlite_master WHERE type='table' AND name='vec_messages'
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/search/vec_search.py:137`

**Current query**

```sql
INSERT INTO vec_messages( rowid, embedding, chat_id, text_preview, sender, timestamp, is_from_me ) VALUES (?, vec_int8(?), ?, ?, ?, ?, ?)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/search/vec_search.py:200`

**Current query**

```sql
INSERT INTO vec_messages( rowid, embedding, chat_id, text_preview, sender, timestamp, is_from_me ) VALUES (?, vec_int8(?), ?, ?, ?, ?, ?)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/search/vec_search.py:287`

**Current query**

```sql
INSERT INTO vec_chunks( embedding, contact_id, chat_id, response_da_type, source_timestamp, quality_score, topic_label, trigger_text, response_text, formatted_text, keywords_json, message_count, source_type, source_id ) VALUES (vec_int8(?), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/search/vec_search.py:327`

**Current query**

```sql
INSERT INTO vec_binary(embedding, chunk_rowid, embedding_int8) VALUES (vec_bit(?), ?, ?)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/search/vec_search.py:411`

**Current query**

```sql
INSERT INTO vec_chunks( embedding, contact_id, chat_id, response_da_type, source_timestamp, quality_score, topic_label, trigger_text, response_text, formatted_text, keywords_json, message_count, source_type, source_id ) VALUES (vec_int8(?), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/search/vec_search.py:463`

**Current query**

```sql
INSERT INTO vec_binary(embedding, chunk_rowid, embedding_int8) VALUES (vec_bit(?), ?, ?)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/search/vec_search.py:484`

**Current query**

```sql
SELECT rowid FROM vec_chunks WHERE chat_id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/search/vec_search.py:495`

**Current query**

```sql
DELETE FROM vec_binary WHERE chunk_rowid IN ({expr})
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/search/vec_search.py:502`

**Current query**

```sql
DELETE FROM vec_chunks WHERE chat_id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/search/vec_search.py:592`

**Current query**

```sql
SELECT rowid, distance, chat_id, trigger_text, response_text, response_da_type, response_da_conf, topic_label, quality_score FROM vec_chunks WHERE embedding MATCH vec_int8(?) AND k = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/search/vec_search.py:673`

**Current query**

```sql
SELECT rowid, chunk_rowid, embedding_int8 FROM vec_binary WHERE embedding MATCH vec_bit(?) AND k = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/search/vec_search.py:707`

**Current query**

```sql
SELECT rowid, chat_id, trigger_text, response_text, response_da_type, response_da_conf, topic_label, quality_score FROM vec_chunks WHERE rowid IN ({expr})
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/search/vec_search.py:757`

**Current query**

```sql
SELECT name FROM sqlite_master WHERE type='table' AND name='vec_binary'
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/search/vec_search.py:765`

**Current query**

```sql
SELECT COUNT(*) FROM vec_binary
```

**Issues found**
- Potential full-table scan: `COUNT(*)` without WHERE filter.

**Optimized query / approach**
- Keep for explicit global metrics only; otherwise add selective filters.

**Suggested indexes**
- None.

### `jarvis/search/vec_search.py:771`

**Current query**

```sql
SELECT rowid, embedding FROM vec_chunks
```

**Issues found**
- Potential full-table scan: SELECT without WHERE/LIMIT.

**Optimized query / approach**
- Add WHERE and/or LIMIT when full scan is not required.

**Suggested indexes**
- None.

### `jarvis/search/vec_search.py:786`

**Current query**

```sql
INSERT INTO vec_binary(embedding, chunk_rowid, embedding_int8) VALUES (vec_bit(?), ?, ?)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/search/vec_search.py:795`

**Current query**

```sql
INSERT INTO vec_binary(embedding, chunk_rowid, embedding_int8) VALUES (vec_bit(?), ?, ?)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/search/vec_search.py:813`

**Current query**

```sql
SELECT count(*) FROM vec_messages
```

**Issues found**
- Potential full-table scan: `COUNT(*)` without WHERE filter.

**Optimized query / approach**
- Keep for explicit global metrics only; otherwise add selective filters.

**Suggested indexes**
- None.

### `jarvis/search/vec_search.py:816`

**Current query**

```sql
SELECT count(DISTINCT chat_id) FROM vec_messages
```

**Issues found**
- Potential full-table scan: SELECT without WHERE/LIMIT.

**Optimized query / approach**
- Add WHERE and/or LIMIT when full scan is not required.

**Suggested indexes**
- None.

### `jarvis/setup.py:231`

**Current query**

```sql
SELECT name FROM sqlite_master WHERE type='table'
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/tags/auto_tagger.py:514`

**Current query**

```sql
SELECT t.id, t.name, COUNT(*) as count FROM tags t JOIN conversation_tags ct ON t.id = ct.tag_id JOIN tag_usage_history h ON t.id = h.tag_id WHERE h.action = 'add' AND h.context_json LIKE ? ESCAPE '\' AND t.id NOT IN ( SELECT tag_id FROM conversation_tags WHERE chat_id = ? ) GROUP BY t.id ORDER BY count DESC LIMIT 5
```

**Issues found**
- Potential N+1: query executes inside a loop.
- LIKE predicate may bypass indexes for leading-wildcard search values.
- Missing index candidate(s): tag_usage_history.action, tag_usage_history.context_json
- JOIN efficiency risk: join/filter columns include non-indexed fields.

**Optimized query / approach**
- Fetch all required rows in one query (`IN (...)`, JOIN, or prefetch) instead of per-iteration calls.

**Suggested indexes**
- `CREATE INDEX IF NOT EXISTS idx_tag_usage_history_action ON tag_usage_history(action);`
- `CREATE INDEX IF NOT EXISTS idx_tag_usage_history_context_json ON tag_usage_history(context_json);`

### `jarvis/tags/auto_tagger.py:668`

**Current query**

```sql
INSERT INTO tag_usage_history (tag_id, chat_id, action, context_json, created_at) VALUES (?, ?, ?, ?, ?)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:70`

**Current query**

```sql
CREATE TABLE IF NOT EXISTS tags ( id INTEGER PRIMARY KEY, name TEXT NOT NULL, color TEXT DEFAULT '#3b82f6', icon TEXT DEFAULT 'tag', parent_id INTEGER REFERENCES tags(id) ON DELETE SET NULL, description TEXT, aliases_json TEXT, sort_order INTEGER DEFAULT 0, is_system BOOLEAN DEFAULT FALSE, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, UNIQUE(name, parent_id) )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:85`

**Current query**

```sql
CREATE TABLE IF NOT EXISTS conversation_tags ( chat_id TEXT NOT NULL, tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE, added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, added_by TEXT DEFAULT 'user', confidence REAL DEFAULT 1.0, PRIMARY KEY (chat_id, tag_id) )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:95`

**Current query**

```sql
CREATE TABLE IF NOT EXISTS smart_folders ( id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, icon TEXT DEFAULT 'folder', color TEXT DEFAULT '#64748b', rules_json TEXT, sort_order INTEGER DEFAULT 0, is_default BOOLEAN DEFAULT FALSE, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:108`

**Current query**

```sql
CREATE TABLE IF NOT EXISTS tag_rules ( id INTEGER PRIMARY KEY, name TEXT NOT NULL, trigger TEXT DEFAULT 'on_new_message', conditions_json TEXT, tag_ids_json TEXT, priority INTEGER DEFAULT 0, is_enabled BOOLEAN DEFAULT TRUE, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, last_triggered_at TIMESTAMP, trigger_count INTEGER DEFAULT 0 )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:122`

**Current query**

```sql
CREATE TABLE IF NOT EXISTS tag_usage_history ( id INTEGER PRIMARY KEY, tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE, chat_id TEXT NOT NULL, action TEXT NOT NULL, context_json TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:132`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_tags_parent ON tags(parent_id)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:135`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:136`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_conv_tags_chat ON conversation_tags(chat_id)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:137`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_conv_tags_tag ON conversation_tags(tag_id)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:138`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_tag_rules_trigger ON tag_rules(trigger, is_enabled)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:139`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_tag_usage_tag ON tag_usage_history(tag_id)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:140`

**Current query**

```sql
CREATE INDEX IF NOT EXISTS idx_tag_usage_chat ON tag_usage_history(chat_id)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:267`

**Current query**

```sql
SELECT name FROM sqlite_master WHERE type='table' AND name='tags'
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:279`

**Current query**

```sql
INSERT OR IGNORE INTO smart_folders (name, icon, color, rules_json, is_default, sort_order) VALUES (?, ?, ?, ?, ?, ?)
```

**Issues found**
- Potential N+1: query executes inside a loop.

**Optimized query / approach**
- Batch parameters and use `executemany()` where possible.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:297`

**Current query**

```sql
INSERT OR IGNORE INTO tags (name, color, icon, is_system, sort_order) VALUES (?, ?, ?, ?, ?)
```

**Issues found**
- Potential N+1: query executes inside a loop.

**Optimized query / approach**
- Batch parameters and use `executemany()` where possible.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:351`

**Current query**

```sql
INSERT INTO tags (name, color, icon, parent_id, description, aliases_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:384`

**Current query**

```sql
SELECT * FROM tags WHERE id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:397`

**Current query**

```sql
SELECT * FROM tags WHERE name = ? AND parent_id IS NULL
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:401`

**Current query**

```sql
SELECT * FROM tags WHERE name = ? AND parent_id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:440`

**Current query**

```sql
SELECT * FROM tags WHERE {expr} ORDER BY sort_order, name
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:517`

**Current query**

```sql
UPDATE tags SET {expr} WHERE id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:532`

**Current query**

```sql
DELETE FROM tags WHERE id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:563`

**Current query**

```sql
SELECT * FROM tags WHERE LOWER(name) LIKE ? OR LOWER(aliases_json) LIKE ? ORDER BY CASE WHEN LOWER(name) = ? THEN 0 WHEN LOWER(name) LIKE ? THEN 1 ELSE 2 END, name LIMIT ?
```

**Issues found**
- Potential non-sargable predicate: function-wrapped column in WHERE.
- LIKE predicate may bypass indexes for leading-wildcard search values.

**Optimized query / approach**
- Use an expression index or normalized shadow column for this predicate.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:603`

**Current query**

```sql
INSERT INTO conversation_tags (chat_id, tag_id, added_by, confidence, added_at) VALUES (?, ?, ?, ?, ?)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:624`

**Current query**

```sql
DELETE FROM conversation_tags WHERE chat_id = ? AND tag_id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:641`

**Current query**

```sql
SELECT t.*, ct.added_at, ct.added_by, ct.confidence FROM tags t JOIN conversation_tags ct ON t.id = ct.tag_id WHERE ct.chat_id = ? ORDER BY t.sort_order, t.name
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:669`

**Current query**

```sql
SELECT chat_id FROM conversation_tags WHERE tag_id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:698`

**Current query**

```sql
SELECT chat_id FROM conversation_tags WHERE tag_id IN ({expr}) GROUP BY chat_id HAVING COUNT(DISTINCT tag_id) = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:708`

**Current query**

```sql
SELECT DISTINCT chat_id FROM conversation_tags WHERE tag_id IN ({expr})
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:736`

**Current query**

```sql
INSERT INTO conversation_tags (chat_id, tag_id, added_by, added_at) VALUES (?, ?, ?, ?)
```

**Issues found**
- Potential N+1: query executes inside a loop.

**Optimized query / approach**
- Batch parameters and use `executemany()` where possible.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:767`

**Current query**

```sql
DELETE FROM conversation_tags WHERE chat_id IN ({expr}) AND tag_id IN ({expr})
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:791`

**Current query**

```sql
DELETE FROM conversation_tags WHERE chat_id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:796`

**Current query**

```sql
INSERT INTO conversation_tags (chat_id, tag_id, added_by, added_at) VALUES (?, ?, ?, ?)
```

**Issues found**
- Potential N+1: query executes inside a loop.

**Optimized query / approach**
- Batch parameters and use `executemany()` where possible.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:819`

**Current query**

```sql
INSERT INTO smart_folders (name, icon, color, rules_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:846`

**Current query**

```sql
SELECT * FROM smart_folders WHERE id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:859`

**Current query**

```sql
SELECT * FROM smart_folders ORDER BY is_default DESC, sort_order, name
```

**Issues found**
- Potential full-table scan: SELECT without WHERE/LIMIT.

**Optimized query / approach**
- Add WHERE and/or LIMIT when full scan is not required.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:863`

**Current query**

```sql
SELECT * FROM smart_folders WHERE is_default = FALSE ORDER BY sort_order, name
```

**Issues found**
- Missing index candidate(s): smart_folders.is_default

**Optimized query / approach**
- Keep query shape and add supporting index(es) below.

**Suggested indexes**
- `CREATE INDEX IF NOT EXISTS idx_smart_folders_is_default ON smart_folders(is_default);`

### `jarvis/tags/manager.py:911`

**Current query**

```sql
UPDATE smart_folders SET {expr} WHERE id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:926`

**Current query**

```sql
DELETE FROM smart_folders WHERE id = ? AND is_default = FALSE
```

**Issues found**
- Missing index candidate(s): smart_folders.is_default

**Optimized query / approach**
- Keep query shape and add supporting index(es) below.

**Suggested indexes**
- `CREATE INDEX IF NOT EXISTS idx_smart_folders_is_default ON smart_folders(is_default);`

### `jarvis/tags/manager.py:942`

**Current query**

```sql
INSERT INTO tag_rules (name, trigger, conditions_json, tag_ids_json, priority, is_enabled, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:966`

**Current query**

```sql
SELECT * FROM tag_rules WHERE id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:987`

**Current query**

```sql
SELECT * FROM tag_rules WHERE {expr} ORDER BY priority DESC, name
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:999`

**Current query**

```sql
UPDATE tag_rules SET name = ?, trigger = ?, conditions_json = ?, tag_ids_json = ?, priority = ?, is_enabled = ? WHERE id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:1022`

**Current query**

```sql
DELETE FROM tag_rules WHERE id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:1028`

**Current query**

```sql
UPDATE tag_rules SET last_triggered_at = ?, trigger_count = trigger_count + 1 WHERE id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:1046`

**Current query**

```sql
SELECT COUNT(*) FROM tags
```

**Issues found**
- Potential full-table scan: `COUNT(*)` without WHERE filter.

**Optimized query / approach**
- Keep for explicit global metrics only; otherwise add selective filters.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:1049`

**Current query**

```sql
SELECT chat_id, COUNT(*) as count FROM conversation_tags GROUP BY chat_id
```

**Issues found**
- Potential full-table scan: SELECT without WHERE/LIMIT.

**Optimized query / approach**
- Add WHERE and/or LIMIT when full scan is not required.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:1060`

**Current query**

```sql
SELECT t.id, t.name, COUNT(ct.chat_id) as usage_count FROM tags t LEFT JOIN conversation_tags ct ON t.id = ct.tag_id GROUP BY t.id ORDER BY usage_count DESC LIMIT 10
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:1087`

**Current query**

```sql
SELECT t.*, COUNT(*) as co_count FROM tags t JOIN conversation_tags ct ON t.id = ct.tag_id WHERE ct.chat_id IN ( SELECT chat_id FROM conversation_tags WHERE tag_id = ? ) AND t.id != ? GROUP BY t.id ORDER BY co_count DESC LIMIT ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/tags/manager.py:1163`

**Current query**

```sql
INSERT INTO tag_usage_history (tag_id, chat_id, action, context_json, created_at) VALUES (?, ?, ?, ?, ?)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/watcher.py:533`

**Current query**

```sql
SELECT name FROM sqlite_master WHERE type='table'
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `jarvis/watcher.py:575`

**Current query**

```sql
SELECT MAX(ROWID) FROM message
```

**Issues found**
- Potential full-table scan: SELECT without WHERE/LIMIT.

**Optimized query / approach**
- Add WHERE and/or LIMIT when full scan is not required.

**Suggested indexes**
- None.

### `jarvis/watcher.py:632`

**Current query**

```sql
SELECT message.ROWID as id, chat.guid as chat_id, COALESCE(handle.id, 'me') as sender, message.text, message.date, message.is_from_me FROM message JOIN chat_message_join ON message.ROWID = chat_message_join.message_id JOIN chat ON chat_message_join.chat_id = chat.ROWID LEFT JOIN handle ON message.handle_id = handle.ROWID WHERE message.ROWID > ? ORDER BY message.date ASC LIMIT ?
```

**Issues found**
- JOIN index coverage not fully verifiable for external-schema tables.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `scripts/build_fact_goldset.py:354`

**Current query**

```sql
SELECT m.ROWID AS message_id, m.text AS message_text, m.date AS message_date_raw, m.is_from_me AS is_from_me, COALESCE(h.id, 'me') AS sender_handle FROM message m JOIN chat_message_join cmj ON m.ROWID = cmj.message_id LEFT JOIN handle h ON m.handle_id = h.ROWID WHERE cmj.chat_id = ? AND m.text IS NOT NULL AND TRIM(m.text) != '' AND ( m.date < ? OR (m.date = ? AND m.ROWID < ?) ) ORDER BY m.date DESC, m.ROWID DESC LIMIT ?
```

**Issues found**
- JOIN index coverage not fully verifiable for external-schema tables.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `scripts/build_fact_goldset.py:375`

**Current query**

```sql
SELECT m.ROWID AS message_id, m.text AS message_text, m.date AS message_date_raw, m.is_from_me AS is_from_me, COALESCE(h.id, 'me') AS sender_handle FROM message m JOIN chat_message_join cmj ON m.ROWID = cmj.message_id LEFT JOIN handle h ON m.handle_id = h.ROWID WHERE cmj.chat_id = ? AND m.text IS NOT NULL AND TRIM(m.text) != '' AND ( m.date > ? OR (m.date = ? AND m.ROWID > ?) ) ORDER BY m.date ASC, m.ROWID ASC LIMIT ?
```

**Issues found**
- JOIN index coverage not fully verifiable for external-schema tables.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `scripts/extract_and_validate_facts.py:102`

**Current query**

```sql
SELECT m.text, m.ROWID, c.chat_identifier FROM message m JOIN chat_message_join cmj ON m.ROWID = cmj.message_id JOIN chat c ON cmj.chat_id = c.ROWID WHERE m.text IS NOT NULL AND LENGTH(TRIM(m.text)) > 0 ORDER BY m.ROWID DESC LIMIT ?
```

**Issues found**
- JOIN index coverage not fully verifiable for external-schema tables.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `scripts/prepare_gliner_training.py:148`

**Current query**

```sql
SELECT m.text, m.date, c.display_name FROM message m JOIN chat_message_join cmj ON m.ROWID = cmj.message_id JOIN chat c ON cmj.chat_id = c.ROWID WHERE m.text IS NOT NULL AND LENGTH(m.text) > 10 AND ( LOWER(m.text) LIKE '%i love%' OR LOWER(m.text) LIKE '%i like%' OR LOWER(m.text) LIKE '%i hate%' OR LOWER(m.text) LIKE '%i work%' OR LOWER(m.text) LIKE '%my sister%' OR LOWER(m.text) LIKE '%my mom%' OR LOWER(m.text) LIKE '%live in%' OR LOWER(m.text) LIKE '%moving to%' OR LOWER(m.text) LIKE '%allergic%' OR LOWER(m.text) LIKE '%obsessed with%' ) ORDER BY RANDOM() LIMIT ?
```

**Issues found**
- Potential non-sargable predicate: function-wrapped column in WHERE.
- JOIN index coverage not fully verifiable for external-schema tables.

**Optimized query / approach**
- Use an expression index or normalized shadow column for this predicate.

**Suggested indexes**
- None.

### `tests/unit/test_db.py:64`

**Current query**

```sql
SELECT name FROM sqlite_master WHERE type='table' ORDER BY name
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `tests/unit/test_db.py:86`

**Current query**

```sql
SELECT version FROM schema_version
```

**Issues found**
- Potential full-table scan: SELECT without WHERE/LIMIT.

**Optimized query / approach**
- Add WHERE and/or LIMIT when full scan is not required.

**Suggested indexes**
- None.

### `tests/unit/test_db.py:1027`

**Current query**

```sql
UPDATE pairs SET is_holdout = FALSE
```

**Issues found**
- Full-table update detected (verify intent).

**Optimized query / approach**
- If not maintenance/reset behavior, add WHERE predicates.

**Suggested indexes**
- None.

### `tests/unit/test_db.py:1044`

**Current query**

```sql
UPDATE pairs SET is_holdout = TRUE WHERE contact_id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `tests/unit/test_db.py:1061`

**Current query**

```sql
UPDATE pairs SET is_holdout = TRUE WHERE contact_id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `tests/unit/test_db.py:1111`

**Current query**

```sql
UPDATE pairs SET is_holdout = TRUE WHERE contact_id = ?
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `tests/unit/test_db.py:1912`

**Current query**

```sql
INSERT INTO contacts (display_name, chat_id) VALUES (?, ?)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `tests/unit/test_db.py:1917`

**Current query**

```sql
INSERT INTO contacts (display_name, chat_id) VALUES (?, ?)
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `tests/unit/test_embeddings.py:238`

**Current query**

```sql
SELECT name FROM sqlite_master WHERE type='table'
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `tests/unit/test_feedback.py:57`

**Current query**

```sql
SELECT name FROM sqlite_master WHERE type='table' AND name='feedback'
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `tests/unit/test_feedback.py:84`

**Current query**

```sql
SELECT name FROM sqlite_master WHERE type='index'
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `tests/unit/test_feedback.py:104`

**Current query**

```sql
SELECT version FROM feedback_schema_version ORDER BY version DESC LIMIT 1
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `tests/unit/test_metrics_router.py:95`

**Current query**

```sql
SELECT COUNT(*) FROM routing_metrics
```

**Issues found**
- Potential full-table scan: `COUNT(*)` without WHERE filter.

**Optimized query / approach**
- Keep for explicit global metrics only; otherwise add selective filters.

**Suggested indexes**
- None.

### `tests/unit/test_metrics_router.py:116`

**Current query**

```sql
SELECT COUNT(*) FROM routing_metrics
```

**Issues found**
- Potential full-table scan: `COUNT(*)` without WHERE filter.

**Optimized query / approach**
- Keep for explicit global metrics only; otherwise add selective filters.

**Suggested indexes**
- None.

### `tests/unit/test_metrics_router.py:135`

**Current query**

```sql
SELECT COUNT(*) FROM routing_metrics
```

**Issues found**
- Potential full-table scan: `COUNT(*)` without WHERE filter.

**Optimized query / approach**
- Keep for explicit global metrics only; otherwise add selective filters.

**Suggested indexes**
- None.

### `tests/unit/test_metrics_router.py:151`

**Current query**

```sql
SELECT name FROM sqlite_master WHERE type='table'
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `tests/unit/test_metrics_router.py:156`

**Current query**

```sql
SELECT name FROM sqlite_master WHERE type='index'
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `tests/unit/test_metrics_router.py:179`

**Current query**

```sql
SELECT * FROM routing_metrics
```

**Issues found**
- Potential full-table scan: SELECT without WHERE/LIMIT.

**Optimized query / approach**
- Add WHERE and/or LIMIT when full scan is not required.

**Suggested indexes**
- None.

### `tests/unit/test_metrics_router.py:219`

**Current query**

```sql
SELECT COUNT(*) FROM routing_metrics
```

**Issues found**
- Potential full-table scan: `COUNT(*)` without WHERE filter.

**Optimized query / approach**
- Keep for explicit global metrics only; otherwise add selective filters.

**Suggested indexes**
- None.

### `tests/unit/test_metrics_router.py:243`

**Current query**

```sql
SELECT COUNT(*) FROM routing_metrics
```

**Issues found**
- Potential full-table scan: `COUNT(*)` without WHERE filter.

**Optimized query / approach**
- Keep for explicit global metrics only; otherwise add selective filters.

**Suggested indexes**
- None.

### `tests/unit/test_metrics_router.py:406`

**Current query**

```sql
SELECT COUNT(*) FROM routing_metrics
```

**Issues found**
- Potential full-table scan: `COUNT(*)` without WHERE filter.

**Optimized query / approach**
- Keep for explicit global metrics only; otherwise add selective filters.

**Suggested indexes**
- None.

### `tests/unit/test_metrics_router.py:427`

**Current query**

```sql
SELECT COUNT(*) FROM routing_metrics
```

**Issues found**
- Potential full-table scan: `COUNT(*)` without WHERE filter.

**Optimized query / approach**
- Keep for explicit global metrics only; otherwise add selective filters.

**Suggested indexes**
- None.

### `tests/unit/test_setup.py:206`

**Current query**

```sql
CREATE TABLE message ( ROWID INTEGER PRIMARY KEY, text TEXT, date INTEGER, is_from_me INTEGER, handle_id INTEGER, thread_originator_guid TEXT )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `tests/unit/test_setup.py:218`

**Current query**

```sql
CREATE TABLE chat ( ROWID INTEGER PRIMARY KEY, guid TEXT, display_name TEXT )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `tests/unit/test_setup.py:227`

**Current query**

```sql
CREATE TABLE handle ( ROWID INTEGER PRIMARY KEY, id TEXT )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `tests/unit/test_setup.py:235`

**Current query**

```sql
CREATE TABLE chat_message_join ( chat_id INTEGER, message_id INTEGER )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `tests/unit/test_setup.py:243`

**Current query**

```sql
CREATE TABLE chat_handle_join ( chat_id INTEGER, handle_id INTEGER )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `tests/unit/test_setup.py:272`

**Current query**

```sql
CREATE TABLE message ( ROWID INTEGER PRIMARY KEY, text TEXT, date INTEGER, is_from_me INTEGER, handle_id INTEGER, thread_originator_guid TEXT )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `tests/unit/test_setup.py:284`

**Current query**

```sql
CREATE TABLE chat ( ROWID INTEGER PRIMARY KEY, guid TEXT, display_name TEXT, service_name TEXT )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `tests/unit/test_setup.py:294`

**Current query**

```sql
CREATE TABLE handle ( ROWID INTEGER PRIMARY KEY, id TEXT )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `tests/unit/test_setup.py:302`

**Current query**

```sql
CREATE TABLE chat_message_join ( chat_id INTEGER, message_id INTEGER )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `tests/unit/test_setup.py:310`

**Current query**

```sql
CREATE TABLE chat_handle_join ( chat_id INTEGER, handle_id INTEGER )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `tests/unit/test_setup.py:869`

**Current query**

```sql
CREATE TABLE message ( ROWID INTEGER PRIMARY KEY, custom_column TEXT )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `tests/unit/test_setup.py:877`

**Current query**

```sql
CREATE TABLE chat ( ROWID INTEGER PRIMARY KEY )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `tests/unit/test_setup.py:884`

**Current query**

```sql
CREATE TABLE handle ( ROWID INTEGER PRIMARY KEY )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `tests/unit/test_setup.py:891`

**Current query**

```sql
CREATE TABLE chat_message_join ( chat_id INTEGER, message_id INTEGER )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

### `tests/unit/test_setup.py:899`

**Current query**

```sql
CREATE TABLE chat_handle_join ( chat_id INTEGER, handle_id INTEGER )
```

**Issues found**
- No material issue detected.

**Optimized query / approach**
- No query rewrite required.

**Suggested indexes**
- None.

