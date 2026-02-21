from unittest.mock import patch

from integrations.imessage.sender import IMessageSender, SendResult


def test_validate_recipient_format_accepts_short_code() -> None:
    assert IMessageSender._validate_recipient_format("88683") is True
    assert IMessageSender._validate_recipient_format("65161") is True


def test_validate_recipient_format_rejects_invalid() -> None:
    assert IMessageSender._validate_recipient_format("not-a-recipient") is False


@patch("integrations.imessage.sender.IMessageSender._run_applescript")
def test_send_message_includes_sms_fallback(mock_run_applescript) -> None:
    mock_run_applescript.return_value = SendResult(success=True)

    sender = IMessageSender()
    result = sender.send_message("hello", recipient="88683")

    assert result.success is True
    script = mock_run_applescript.call_args.args[0]
    assert "service type = iMessage" in script
    assert "service type = SMS" in script
