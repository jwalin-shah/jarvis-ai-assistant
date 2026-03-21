import re

file_path = "jarvis/contacts/extractors/instruction_adapter.py"
with open(file_path, "r") as f:
    content = f.read()

# Make it cast to Message or use proper typing
content = content.replace(
    'messages=[MockMessage(text, is_from_me)],',
    'messages=[cast("Message", MockMessage(text, is_from_me))],'
)
# Add typing.cast if not present
if "from typing import cast" not in content and "from typing import " in content:
    content = content.replace("from typing import ", "from typing import cast, ")
elif "import typing" not in content:
    content = "from typing import cast\n" + content

with open(file_path, "w") as f:
    f.write(content)
