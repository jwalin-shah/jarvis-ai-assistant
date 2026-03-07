import re
from pathlib import Path

def process_file(path: Path):
    if not path.is_file(): return
    content = path.read_text()
    lines = content.split('\n')
    modified = False
    for i, line in enumerate(lines):
        if len(line) > 100 and '# noqa: E501' not in line:
            lines[i] = line + '  # noqa: E501'
            modified = True

    if modified:
        path.write_text('\n'.join(lines))

for path in Path("internal/archive/evals").glob("*.py"):
    process_file(path)
