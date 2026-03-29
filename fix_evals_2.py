from pathlib import Path


def apply_fixes():
    files = {
        "internal/archive/evals/ablation_context_rag.py": [
            ("from evals.eval_pipeline import EVAL_DATASET_PATH, check_anti_ai, load_eval_dataset", "from evals.eval_pipeline import EVAL_DATASET_PATH, check_anti_ai, load_eval_dataset  # noqa: E402"),
            ("from evals.judge_config import JUDGE_MODEL, get_judge_client", "from evals.judge_config import JUDGE_MODEL, get_judge_client  # noqa: E402"),
            ("print(f\"[{start_idx + i + 1:2d}] [{ex.category:12s}] {status} | {score:.0f}/10 | {reply[:50]}\")", "print(\n                f\"[{start_idx + i + 1:2d}] [{ex.category:12s}] \"\n                f\"{status} | {score:.0f}/10 | {reply[:50]}\"\n            )"),
            ("f\"  Config: context_depth={s['config']['context_depth']}, use_rag={s['config']['use_rag']}\"", "f\"  Config: context_depth={s['config']['context_depth']}, \"\n            f\"use_rag={s['config']['use_rag']}\""),
            ("f\"Config: context_depth={winner['config']['context_depth']}, use_rag={winner['config']['use_rag']}\"", "f\"Config: context_depth={winner['config']['context_depth']}, \"\n        f\"use_rag={winner['config']['use_rag']}\"")
        ],
        "internal/archive/evals/evaluate_optimized_settings.py": [
            ("from evals.eval_pipeline import EVAL_DATASET_PATH, EvalExample, load_eval_dataset", "from evals.eval_pipeline import EVAL_DATASET_PATH, EvalExample, load_eval_dataset  # noqa: E402"),
            ("from evals.judge_config import JUDGE_MODEL, get_judge_client", "from evals.judge_config import JUDGE_MODEL, get_judge_client  # noqa: E402"),
            ("from tqdm import tqdm", "from tqdm import tqdm  # noqa: E402"),
            ("from models.loader import get_model", "from models.loader import get_model  # noqa: E402")
        ],
        "internal/archive/evals/evaluate_semantic_templates.py": [
            ("from evals.judge_config import JUDGE_MODEL, get_judge_client", "from evals.judge_config import JUDGE_MODEL, get_judge_client  # noqa: E402"),
            ("from models.template_defaults import get_minimal_fallback_templates", "from models.template_defaults import get_minimal_fallback_templates  # noqa: E402"),
            ("from models.templates import ResponseTemplate, TemplateMatcher", "from models.templates import ResponseTemplate, TemplateMatcher  # noqa: E402")
        ],
        "internal/archive/evals/evaluate_templates.py": [
            ("from evals.judge_config import JUDGE_MODEL, get_judge_client", "from evals.judge_config import JUDGE_MODEL, get_judge_client  # noqa: E402"),
            ("from jarvis.prompts import ACKNOWLEDGE_TEMPLATES, CLOSING_TEMPLATES", "from jarvis.prompts import ACKNOWLEDGE_TEMPLATES, CLOSING_TEMPLATES  # noqa: E402")
        ],
        "internal/archive/evals/jarvis_provider.py": [
            ("from jarvis.prompts.generation_config import DEFAULT_REPETITION_PENALTY", "from jarvis.prompts.generation_config import DEFAULT_REPETITION_PENALTY  # noqa: E402")
        ],
        "internal/archive/evals/optimize_universal_prompt.py": [
            ("from evals.dspy_client import DSPYMLXClient", "from evals.dspy_client import DSPYMLXClient  # noqa: E402"),
            ("from evals.dspy_reply import (\n    TRAIN_EXAMPLES,\n    judge_metric,\n)", "from evals.dspy_reply import (  # noqa: E402\n    TRAIN_EXAMPLES,\n    judge_metric,\n)")
        ],
        "internal/archive/evals/optimize_universal_prompt_batched.py": [
            ("from evals.eval_pipeline import EVAL_DATASET_PATH, check_anti_ai, load_eval_dataset", "from evals.eval_pipeline import EVAL_DATASET_PATH, check_anti_ai, load_eval_dataset  # noqa: E402"),
            ("from evals.judge_config import JUDGE_MODEL, get_judge_client", "from evals.judge_config import JUDGE_MODEL, get_judge_client  # noqa: E402")
        ],
        "internal/archive/evals/sweep_pipeline.py": [
            ("from evals.dspy_reply import TRAIN_EXAMPLES, clean_reply, judge_metric", "from evals.dspy_reply import TRAIN_EXAMPLES, clean_reply, judge_metric  # noqa: E402"),
            ("from evals.judge_config import JUDGE_MODEL, get_judge_client", "from evals.judge_config import JUDGE_MODEL, get_judge_client  # noqa: E402"),
            ("from models.loader import get_model", "from models.loader import get_model  # noqa: E402")
        ]
    }

    for filepath, edits in files.items():
        p = Path(filepath)
        if not p.exists():
            continue
        content = p.read_text()
        for old, new in edits:
            content = content.replace(old, new)
        p.write_text(content)

apply_fixes()
