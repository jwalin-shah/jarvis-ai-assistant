file_path = "jarvis/contacts/extractors/instruction_adapter.py"
with open(file_path, "r") as f:
    content = f.read()

if "from jarvis.contracts.imessage import Message" not in content:
    content = content.replace("from typing import cast", "from typing import cast\nfrom jarvis.contracts.imessage import Message")

with open(file_path, "w") as f:
    f.write(content)
