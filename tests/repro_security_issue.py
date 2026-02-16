
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile
import shutil
from jarvis.tasks.models import Task, TaskType, TaskResult
from jarvis.tasks.worker import TaskWorker
import integrations.imessage

class TestSecurityIssue(unittest.TestCase):
    def setUp(self):
        self.worker = TaskWorker()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch("integrations.imessage.ChatDBReader")
    @patch("jarvis.export.export_messages")
    def test_arbitrary_write(self, mock_export, mock_reader_cls):
        # Setup mock export
        mock_export.return_value = '{"test": "data"}'

        # Setup mock reader
        mock_reader = mock_reader_cls.return_value
        mock_reader.__enter__.return_value = mock_reader

        # Mock conversations
        mock_conv = MagicMock()
        mock_conv.chat_id = "chat1"
        mock_conv.display_name = "Test Chat"
        mock_reader.get_conversations.return_value = [mock_conv]

        # Mock messages
        mock_msg = MagicMock()
        mock_msg.text = "Hello"
        mock_reader.get_messages.return_value = [mock_msg]

        # Define a dangerous output directory (e.g., a hidden dir mimicking .ssh)
        dangerous_dir = Path(self.temp_dir) / ".ssh"
        dangerous_dir.mkdir()

        # Create task
        task = Task(
            task_type=TaskType.BATCH_EXPORT,
            params={
                "chat_ids": ["chat1"],
                "output_dir": str(dangerous_dir)
            }
        )

        # Execute handler directly
        # Currently, this should SUCCEED (vulnerability)

        # We mock update_progress
        update_progress = MagicMock()

        result = self.worker._handle_batch_export(task, update_progress)

        # Check if file was written
        expected_files = list(dangerous_dir.glob("conversation_chat1_*.json"))

        print(f"Result success: {result.success}")
        print(f"Files in dangerous dir: {expected_files}")

        self.assertTrue(result.success)
        self.assertTrue(len(expected_files) > 0)

if __name__ == "__main__":
    unittest.main()
