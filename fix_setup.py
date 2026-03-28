with open("tests/unit/test_setup.py") as f:
    text = f.read()

# Fix import jarvis.core.exceptions
text = text.replace("from core.exceptions import", "from jarvis.core.exceptions import")
text = text.replace("from core.health import", "from jarvis.core.health import")
text = text.replace("from core.memory import", "from jarvis.core.memory import")

with open("tests/unit/test_setup.py", "w") as f:
    f.write(text)
