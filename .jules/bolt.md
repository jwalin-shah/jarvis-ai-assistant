## 2026-02-16 - [SQLite Pagination Performance]
**Learning:** Ordering by `ROWID` when paging with `WHERE ROWID > ?` enables an index scan, avoiding a costly sort operation that occurs when ordering by `date` (which may not be strictly monotonic with ID).
**Action:** Always align `ORDER BY` with the pagination cursor (here `ROWID`) to ensure $O(1)$ complexity for fetching the next page.
