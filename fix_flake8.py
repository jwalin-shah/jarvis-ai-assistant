import re

files_to_fix = [
    ("internal/archive/evals/sweep_pipeline.py", 'from evals.dspy_reply', 'from evals.dspy_reply'),
    ("internal/archive/evals/sweep_pipeline.py", 'from evals.judge_config', 'from evals.judge_config'),
    ("internal/archive/evals/sweep_pipeline.py", 'from models.loader', 'from models.loader')
]

for path, _, _ in files_to_fix:
    with open(path) as f:
        content = f.read()
    content = content.replace("  # noqa: E402", "")
    content = re.sub(r'^(from .*)', r'\1  # noqa: E402', content, flags=re.MULTILINE)
    with open(path, "w") as f:
        f.write(content)
