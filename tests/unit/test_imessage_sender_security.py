
import unittest
import shutil
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the module under test
from integrations.imessage.sender import _validate_file_path, IMessageSender

class TestIMessageSenderSecurity(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory structure for testing
        # IMPORTANT: We use os.getcwd() to avoid creating it in /tmp,
        # because /tmp is in the allowed_bases list, which would make
        # all files in the test allowed, masking the security check logic
        # for non-tmp files.
        self.test_dir = tempfile.mkdtemp(dir=os.getcwd())
        self.home = Path(self.test_dir) / "home" / "user"
        self.home.mkdir(parents=True)

        # Patch Path.home() to point to our temp home
        self.home_patcher = patch("pathlib.Path.home", return_value=self.home)
        self.mock_home = self.home_patcher.start()

        # Create standard directories
        (self.home / "Downloads").mkdir()
        (self.home / "Documents").mkdir()
        (self.home / "Desktop").mkdir()
        (self.home / "Pictures").mkdir()
        (self.home / "Music").mkdir()
        (self.home / "Movies").mkdir()
        (self.home / "Public").mkdir()
        (self.home / ".jarvis" / "attachments").mkdir(parents=True)

    def tearDown(self):
        self.home_patcher.stop()
        shutil.rmtree(self.test_dir)

    def create_file(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path

    def test_allowed_paths(self):
        """Test that files in allowed directories are permitted."""
        allowed_files = [
            self.home / "Downloads" / "file.txt",
            self.home / "Documents" / "doc.pdf",
            self.home / "Desktop" / "image.png",
            self.home / "Pictures" / "photo.jpg",
            self.home / "Music" / "song.mp3",
            self.home / "Movies" / "video.mp4",
            self.home / "Public" / "public_file.txt",
            self.home / ".jarvis" / "attachments" / "attachment.pdf",
            Path("/tmp/safe_tmp.txt"),
        ]

        for file_path in allowed_files:
            # We must create the file for resolve(strict=True) to work
            if str(file_path).startswith("/tmp"):
                # Special handling for /tmp as we can't easily mock /tmp existence
                # But we can create it if we have permissions
                try:
                    self.create_file(file_path)
                except OSError:
                    # Skip if we can't write to /tmp
                    continue
            else:
                self.create_file(file_path)

            self.assertTrue(_validate_file_path(file_path), f"Should allow {file_path}")

    def test_denied_paths(self):
        """Test that files outside allowed directories are denied."""
        denied_files = [
            self.home / "sensitive.txt",
            self.home / ".ssh" / "id_rsa",
            self.home / ".bashrc",
            self.home / "other_folder" / "file.txt",
            Path("/etc/passwd"),  # System file
            Path("/usr/bin/python3"), # System binary
        ]

        for file_path in denied_files:
            if str(file_path).startswith("/etc") or str(file_path).startswith("/usr"):
                # System files exist, check directly
                if file_path.exists():
                     self.assertFalse(_validate_file_path(file_path), f"Should deny {file_path}")
            else:
                self.create_file(file_path)
                self.assertFalse(_validate_file_path(file_path), f"Should deny {file_path}")

    def test_path_traversal(self):
        """Test path traversal attempts."""
        # Setup: allowed/file.txt
        allowed = self.home / "Documents" / "allowed.txt"
        self.create_file(allowed)

        # specific traversal: allowed/../sensitive.txt
        # This resolves to home/sensitive.txt which is denied
        sensitive = self.home / "sensitive.txt"
        self.create_file(sensitive)

        traversal = self.home / "Documents" / ".." / "sensitive.txt"
        # resolve(strict=True) will resolve to sensitive.txt

        self.assertFalse(_validate_file_path(traversal), "Should deny traversal to sensitive file")

    def test_symlink_attack(self):
        """Test symlink attack where a link in allowed dir points to denied file."""
        sensitive = self.home / "sensitive.txt"
        self.create_file(sensitive)

        link = self.home / "Documents" / "link_to_sensitive.txt"
        try:
            link.symlink_to(sensitive)
        except OSError:
            # Symlinks might not be supported on some platforms/filesystems
            print("Skipping symlink test due to OSError")
            return

        # resolved path is sensitive.txt, which is denied
        self.assertFalse(_validate_file_path(link), "Should deny symlink to sensitive file")

    @patch("integrations.imessage.sender.IMessageSender._run_applescript")
    def test_send_attachment_security(self, mock_run_script):
        """Test that send_attachment enforces security check."""
        sender = IMessageSender()

        # Case 1: Denied file
        denied = self.home / "sensitive.txt"
        self.create_file(denied)

        result = sender.send_attachment(denied, recipient="+1234567890")
        self.assertFalse(result.success)
        self.assertIn("outside allowed directories", result.error)
        mock_run_script.assert_not_called()

        # Case 2: Allowed file
        allowed = self.home / "Documents" / "allowed.txt"
        self.create_file(allowed)

        mock_run_script.return_value = MagicMock(success=True)
        result = sender.send_attachment(allowed, recipient="+1234567890")
        self.assertTrue(result.success)
        mock_run_script.assert_called_once()

if __name__ == "__main__":
    unittest.main()
