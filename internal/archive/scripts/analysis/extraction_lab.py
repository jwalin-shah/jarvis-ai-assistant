#!/usr/bin/env python3
"""Extraction Lab: config-driven experiment runner for fact extraction.

Dispatches experiments as subprocesses for memory isolation (8GB constraint).
GLiNER experiments run in compat venv via run_gliner_compat.sh.

Usage:
    # Run a single named experiment
    uv run python scripts/extraction_lab.py --experiment spacy_sm

    # Run all Round 1 baselines
    uv run python scripts/extraction_lab.py --config scripts/extraction_lab_configs.json

    # Run with custom config from stdin
    echo '{"name": "test", "extractor": "spacy", "params": {}}' | \
        uv run python scripts/extraction_lab.py --stdin

    # Run on specific split
    uv run python scripts/extraction_lab.py --experiment spacy_sm --split dev
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
GOLDSET_DIR = ROOT / "training_data" / "goldset_v6"
RESULTS_BASE = ROOT / "results" / "extraction_lab"


def run_experiment(
    config: dict,
    split: str = "dev",
    output_dir: Path | None = None,
    cv_folds: int = 0,
) -> dict:
    """Run a single experiment as a subprocess.

    Args:
        config: Experiment config dict with name, extractor, params.
        split: Which goldset split to evaluate on (train, dev, test, train+dev).
        output_dir: Directory to write results. Default: results/extraction_lab/
        cv_folds: If >0, run cross-validation with this many folds over train+dev.

    Returns:
        Results dict with metrics, timing, and config.
    """
    name = config["name"]
    extractor = config["extractor"]
    params = config.get("params", {})

    if output_dir is None:
        output_dir = RESULTS_BASE
    output_dir.mkdir(parents=True, exist_ok=True)

    result_path = output_dir / f"{name}.json"

    print(f"\n{'=' * 60}", flush=True)
    print(f"Experiment: {name} (extractor={extractor}, split={split})", flush=True)
    print(f"{'=' * 60}", flush=True)

    # Build subprocess command
    runner_script = str(ROOT / "scripts" / "run_single_experiment.py")

    # Prepare experiment payload
    payload = {
        "name": name,
        "extractor": extractor,
        "params": params,
        "split": split,
        "cv_folds": cv_folds,
        "goldset_dir": str(GOLDSET_DIR),
        "output_path": str(result_path),
    }

    if extractor == "gliner":
        # GLiNER must run in compat venv
        cmd = [
            "bash",
            str(ROOT / "scripts" / "run_gliner_compat.sh"),
            str(ROOT / "scripts" / "gliner_extract_standalone.py"),
        ]
        # Pass config as argument
        payload_json = json.dumps(payload)
        cmd.append("--payload")
        cmd.append(payload_json)
    else:
        # Normal experiments run with uv
        cmd = ["uv", "run", "python", runner_script, "--payload", json.dumps(payload)]

    start = time.time()
    print(f"Running: {' '.join(cmd[:4])}...", flush=True)

    try:
        result = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=600,  # 10 min max per experiment
        )

        elapsed = time.time() - start

        if result.returncode != 0:
            print(f"FAILED ({elapsed:.1f}s): {result.stderr[-500:]}", flush=True)
            error_result = {
                "name": name,
                "extractor": extractor,
                "status": "error",
                "error": result.stderr[-1000:],
                "stdout": result.stdout[-1000:],
                "elapsed_s": round(elapsed, 1),
            }
            with open(result_path, "w") as f:
                json.dump(error_result, f, indent=2)
            return error_result

        # Print subprocess stdout (progress)
        if result.stdout:
            for line in result.stdout.strip().split("\n")[-20:]:
                print(f"  {line}", flush=True)

        # Read results from output file
        if result_path.exists():
            with open(result_path) as f:
                results = json.load(f)
            results["elapsed_s"] = round(elapsed, 1)
            print(
                f"Done ({elapsed:.1f}s): micro_f1={results.get('micro_f1', 'N/A')}, "
                f"tp={results.get('total_tp', '?')}, fp={results.get('total_fp', '?')}, "
                f"fn={results.get('total_fn', '?')}",
                flush=True,
            )
            return results
        else:
            print(f"WARNING: No result file at {result_path}", flush=True)
            return {
                "name": name,
                "extractor": extractor,
                "status": "error",
                "error": "No result file produced",
                "stdout": result.stdout[-1000:],
                "elapsed_s": round(elapsed, 1),
            }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"TIMEOUT after {elapsed:.1f}s", flush=True)
        return {
            "name": name,
            "extractor": extractor,
            "status": "timeout",
            "elapsed_s": round(elapsed, 1),
        }


def run_all(
    configs: list[dict],
    split: str = "dev",
    output_dir: Path | None = None,
    cv_folds: int = 0,
) -> list[dict]:
    """Run all experiments sequentially (memory safety: one at a time)."""
    results = []
    total = len(configs)

    for i, config in enumerate(configs):
        print(f"\n[{i + 1}/{total}] Starting experiment: {config['name']}", flush=True)
        result = run_experiment(config, split=split, output_dir=output_dir, cv_folds=cv_folds)
        results.append(result)

    return results


def print_summary(results: list[dict]) -> None:
    """Print summary table of all experiment results."""
    print(f"\n{'=' * 80}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'=' * 80}", flush=True)
    print(
        f"{'Name':<25} {'Extractor':<10} {'micro_F1':>8} {'P':>6} {'R':>6} "
        f"{'TP':>4} {'FP':>4} {'FN':>4} {'Time':>6}",
        flush=True,
    )
    print("-" * 80, flush=True)

    for r in sorted(results, key=lambda x: x.get("micro_f1", 0), reverse=True):
        if r.get("status") in ("error", "timeout"):
            print(
                f"{r['name']:<25} {r['extractor']:<10} {'ERR':>8} {'':>6} {'':>6} "
                f"{'':>4} {'':>4} {'':>4} {r.get('elapsed_s', 0):>5.1f}s",
                flush=True,
            )
        else:
            print(
                f"{r['name']:<25} {r['extractor']:<10} {r.get('micro_f1', 0):>8.3f} "
                f"{r.get('micro_precision', 0):>6.3f} {r.get('micro_recall', 0):>6.3f} "
                f"{r.get('total_tp', 0):>4} {r.get('total_fp', 0):>4} "
                f"{r.get('total_fn', 0):>4} {r.get('elapsed_s', 0):>5.1f}s",
                flush=True,
            )

    print(f"{'=' * 80}\n", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extraction Lab experiment runner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--experiment", help="Run a single named experiment from configs")
    group.add_argument("--config", help="Path to JSON config file with experiment list")
    group.add_argument("--stdin", action="store_true", help="Read single config from stdin")
    group.add_argument("--payload", help="JSON experiment config (for programmatic use)")

    parser.add_argument(
        "--split",
        default="dev",
        help="Goldset split (train, dev, test, train+dev)",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory for results")
    parser.add_argument("--cv-folds", type=int, default=0, help="Cross-validation folds (0=no CV)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_BASE

    if args.stdin:
        config = json.load(sys.stdin)
        results = [
            run_experiment(
                config,
                split=args.split,
                output_dir=output_dir,
                cv_folds=args.cv_folds,
            )
        ]
    elif args.payload:
        config = json.loads(args.payload)
        results = [
            run_experiment(
                config,
                split=args.split,
                output_dir=output_dir,
                cv_folds=args.cv_folds,
            )
        ]
    elif args.experiment:
        # Load from default configs
        config_path = ROOT / "scripts" / "extraction_lab_configs.json"
        if not config_path.exists():
            print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
            sys.exit(1)
        with open(config_path) as f:
            all_configs = json.load(f)
        config = next((c for c in all_configs if c["name"] == args.experiment), None)
        if config is None:
            names = [c["name"] for c in all_configs]
            print(
                f"ERROR: Unknown experiment '{args.experiment}'. Available: {names}",
                file=sys.stderr,
            )
            sys.exit(1)
        results = [
            run_experiment(
                config,
                split=args.split,
                output_dir=output_dir,
                cv_folds=args.cv_folds,
            )
        ]
    else:
        with open(args.config) as f:
            all_configs = json.load(f)
        results = run_all(
            all_configs,
            split=args.split,
            output_dir=output_dir,
            cv_folds=args.cv_folds,
        )

    print_summary(results)

    # Write combined results
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Summary written to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
