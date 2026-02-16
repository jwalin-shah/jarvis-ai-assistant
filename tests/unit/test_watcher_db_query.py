import sqlite3
from unittest.mock import create_autospec

from jarvis.watcher_db import query_new_messages


class TestWatcherDBQuery:
    @staticmethod
    def test_query_new_messages_orders_by_rowid():
        """Verify that query_new_messages orders results by ROWID for performance and consistency."""
        # Mock connection and cursor
        mock_conn = create_autospec(sqlite3.Connection)
        mock_cursor = create_autospec(sqlite3.Cursor)
        mock_conn.cursor.return_value = mock_cursor

        # Setup fetchall to return empty list so the function completes
        mock_cursor.fetchall.return_value = []

        # Call the function
        query_new_messages(mock_conn, 100, limit=50)

        # Verify execute was called
        assert mock_cursor.execute.called

        # Get the SQL query executed
        args, _ = mock_cursor.execute.call_args
        query = args[0]

        # Check that it contains ORDER BY message.ROWID ASC
        # normalize whitespace for comparison
        query_normalized = " ".join(query.split())

        # It should contain ORDER BY message.ROWID ASC
        assert "ORDER BY message.ROWID ASC" in query_normalized, (
            f"Query should order by ROWID, but got: {query}"
        )

        # It should NOT order by date
        assert "ORDER BY message.date" not in query_normalized, "Query should not order by date"
