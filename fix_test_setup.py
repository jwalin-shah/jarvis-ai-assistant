with open('tests/unit/test_setup.py', 'r') as f:
    data = f.read()
data = data.replace('from core.memory.controller import get_memory_controller', 'from jarvis.core.memory.controller import get_memory_controller')
data = data.replace('patch("core.memory.controller', 'patch("jarvis.core.memory.controller')
with open('tests/unit/test_setup.py', 'w') as f:
    f.write(data)
