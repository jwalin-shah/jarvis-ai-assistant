from pathlib import Path


def apply_fixes():
    files = {
        "internal/archive/evals/ablation_context_rag.py": [
            ("sys.path.insert(0, str(PROJECT_ROOT))", "sys.path.insert(0, str(PROJECT_ROOT))\n# noqa: E402"),
            ("print(\n                f\"[{start_idx + i + 1:2d}] [{ex.category:12s}] {status} | {score:.0f}/10 | {reply[:50]}\"\n            )", "print(f\"[{start_idx + i + 1:2d}] [{ex.category:12s}] {status} | {score:.0f}/10 | {reply[:50]}\")"),
        ],
        "internal/archive/evals/eval_pipeline.py": [
            ("def _judge_single_item(judge_client: object, judge_model: str, ex: EvalExample, generated: str) -> tuple[float | None, str]:", "def _judge_single_item(\n    judge_client: object, judge_model: str, ex: EvalExample, generated: str\n) -> tuple[float | None, str]:")
        ],
        "internal/archive/evals/evaluate_optimized_settings.py": [
            ("sys.path.insert(0, str(PROJECT_ROOT))", "sys.path.insert(0, str(PROJECT_ROOT))\n# noqa: E402"),
            ("f\"   Length: -{length_reduction:.0f} chars ({length_reduction / baseline_results['avg_length'] * 100:.0f}% shorter)\"", "f\"   Length: -{length_reduction:.0f} chars \"\n        f\"({length_reduction / baseline_results['avg_length'] * 100:.0f}% shorter)\"")
        ],
        "internal/archive/evals/evaluate_semantic_templates.py": [
            ("sys.path.insert(0, str(PROJECT_ROOT))", "sys.path.insert(0, str(PROJECT_ROOT))\n# noqa: E402"),
            ("{{\"score\": <number>, \"reasoning\": \"<brief explanation>\", \"better_alternative\": \"<suggested better response or null>\"}},", "{{\n    \"score\": <number>,\n    \"reasoning\": \"<brief explanation>\",\n    \"better_alternative\": \"<suggested better response or null>\"\n  }},"),
            ("f\"  ✓ Batch {i // batch_size + 1}/{(len(evaluations) + batch_size - 1) // batch_size} complete\"", "f\"  ✓ Batch {i // batch_size + 1}/\"\n                f\"{(len(evaluations) + batch_size - 1) // batch_size} complete\""),
            ("f\"   (Batch size: {args.batch_size}, Estimated time: {len(evaluations) // args.batch_size * 2.1:.0f}s)\"", "f\"   (Batch size: {args.batch_size}, \"\n                    f\"Estimated time: {len(evaluations) // args.batch_size * 2.1:.0f}s)\"")
        ],
        "internal/archive/evals/evaluate_templates.py": [
            ("sys.path.insert(0, str(PROJECT_ROOT))", "sys.path.insert(0, str(PROJECT_ROOT))\n# noqa: E402"),
            ("{{\"score\": <number>, \"reasoning\": \"<brief explanation>\", \"better_alternative\": \"<suggested better response or null>\"}}", "{{\n  \"score\": <number>,\n  \"reasoning\": \"<brief explanation>\",\n  \"better_alternative\": \"<suggested better response or null>\"\n}}")
        ],
        "internal/archive/evals/jarvis_provider.py": [
            ("os.chdir(PROJECT_ROOT)", "os.chdir(PROJECT_ROOT)\n# noqa: E402")
        ],
        "internal/archive/evals/optimize_universal_prompt.py": [
            ("sys.path.insert(0, str(PROJECT_ROOT))", "sys.path.insert(0, str(PROJECT_ROOT))\n# noqa: E402")
        ],
        "internal/archive/evals/optimize_universal_prompt_batched.py": [
            ("sys.path.insert(0, str(PROJECT_ROOT))", "sys.path.insert(0, str(PROJECT_ROOT))\n# noqa: E402")
        ],
        "internal/archive/evals/run_all_prompt_experiments.py": [
            ("f\"Total Duration: {summary['total_duration_s']:.1f}s ({summary['total_duration_s'] / 60:.1f} min)\"", "f\"Total Duration: {summary['total_duration_s']:.1f}s \"\n        f\"({summary['total_duration_s'] / 60:.1f} min)\"")
        ],
        "internal/archive/evals/sweep_pipeline.py": [
            ("sys.path.insert(0, str(PROJECT_ROOT))", "sys.path.insert(0, str(PROJECT_ROOT))\n# noqa: E402")
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
