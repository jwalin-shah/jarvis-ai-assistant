import os
import glob

def find_files(directory, pattern="*.py"):
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(".py"):
                files.append(os.path.join(root, filename))
    return files

evals_files = find_files("internal/archive/evals/")

for filepath in evals_files:
    try:
        with open(filepath, "r") as f:
            content = f.read()

        # Disable all formatting rules in these archived script files
        if not content.startswith("# type: ignore\n# ruff: noqa\n"):
            content = "# type: ignore\n# ruff: noqa\n" + content

        with open(filepath, "w") as f:
            f.write(content)
    except Exception as e:
        print(f"Error on {filepath}: {e}")
