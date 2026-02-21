import subprocess

from integrations.imessage.sender import IMessageSender

sender = IMessageSender()

# Test case 1: Send to group chat (this is failing)
# NOTE: Using a fake ID here to test failure case logic, as we can't send real messages
chat_id = "iMessage;+;chat405620999465107993"
text = "Test message from reproduction script"
print(f"Attempting to send to group chat: {chat_id}")

try:
    result = sender.send_message(text, chat_id=chat_id, is_group=True)
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception during send: {e}")
    result = None

# Test case 2: Try extracting just the chat part
if result and not result.success:
    print("\nAttempting with extracted chat ID...")
    extracted_id = chat_id.split(";")[-1]
    print(f"Extracted ID: {extracted_id}")

    applescript = f'''
    tell application "Messages"
        try
            set targetChat to chat id "{extracted_id}"
            send "{text}" to targetChat
        on error errMsg number errNum
            return errMsg & " (" & errNum & ")"
        end try
    end tell
    '''
    try:
        proc = subprocess.run(
            ["osascript", "-e", applescript], check=True, capture_output=True, text=True
        )
        print(f"Result with extracted ID: {proc.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"Failed with extracted ID: {e.stderr}")
