#!/usr/bin/env python3  # noqa: E501
"""Phase eval runner for reply pipeline.  # noqa: E501
  # noqa: E501
Modes:  # noqa: E501
- baseline: run eval_pipeline and capture/update baseline artifact  # noqa: E501
- candidate: run eval_pipeline and check against baseline via quality gate  # noqa: E501
"""  # noqa: E501
  # noqa: E501
from __future__ import annotations  # noqa: E402  # noqa: E501

# noqa: E501
import argparse  # noqa: E501
import subprocess  # noqa: E501
import sys  # noqa: E501
from pathlib import Path  # noqa: E402  # noqa: E501

  # noqa: E501
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # noqa: E501
  # noqa: E501
  # noqa: E501
def _run(cmd: list[str]) -> int:  # noqa: E501
    print(f"[phase-eval] $ {' '.join(cmd)}")  # noqa: E501
    return subprocess.run(cmd, cwd=PROJECT_ROOT, check=False).returncode  # noqa: E501
  # noqa: E501
  # noqa: E501
def main() -> int:  # noqa: E501
    parser = argparse.ArgumentParser(description="Run phase eval workflow")  # noqa: E501
    parser.add_argument(  # noqa: E501
        "mode",  # noqa: E501
        choices=["baseline", "candidate"],  # noqa: E501
        help="baseline: capture baseline; candidate: run regression gate",  # noqa: E501
    )  # noqa: E501
    parser.add_argument("--judge", action="store_true", help="Enable LLM judge in eval pipeline")  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--judge-batch-size",  # noqa: E501
        type=int,  # noqa: E501
        default=1,  # noqa: E501
        help="Judge batch size passed to eval_pipeline.py",  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--judge-delay-seconds",  # noqa: E501
        type=float,  # noqa: E501
        default=2.2,  # noqa: E501
        help="Delay between judge calls passed to eval_pipeline.py",  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--force-model-load",  # noqa: E501
        action="store_true",  # noqa: E501
        help="Pass through force-model-load to eval pipeline",  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--source",  # noqa: E501
        default="results/eval_pipeline_baseline.json",  # noqa: E501
        help="Eval output path",  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--baseline",  # noqa: E501
        default="evals/baselines/baseline_20260221.json",  # noqa: E501
        help="Baseline artifact path",  # noqa: E501
    )  # noqa: E501
    args = parser.parse_args()  # noqa: E501
  # noqa: E501
    eval_cmd = ["uv", "run", "python", "evals/eval_pipeline.py"]  # noqa: E501
    if args.judge:  # noqa: E501
        eval_cmd.append("--judge")  # noqa: E501
        eval_cmd.extend(  # noqa: E501
            [  # noqa: E501
                "--judge-batch-size",  # noqa: E501
                str(args.judge_batch_size),  # noqa: E501
                "--judge-delay-seconds",  # noqa: E501
                str(args.judge_delay_seconds),  # noqa: E501
            ]  # noqa: E501
        )  # noqa: E501
    if args.force_model_load:  # noqa: E501
        eval_cmd.append("--force-model-load")  # noqa: E501
  # noqa: E501
    rc = _run(eval_cmd)  # noqa: E501
    if rc != 0:  # noqa: E501
        return rc  # noqa: E501
  # noqa: E501
    if args.mode == "baseline":  # noqa: E501
        return _run(  # noqa: E501
            [  # noqa: E501
                "uv",  # noqa: E501
                "run",  # noqa: E501
                "python",  # noqa: E501
                "evals/capture_baseline.py",  # noqa: E501
                "--source",  # noqa: E501
                args.source,  # noqa: E501
                "--output",  # noqa: E501
                args.baseline,  # noqa: E501
            ]  # noqa: E501
        )  # noqa: E501
  # noqa: E501
    return _run(  # noqa: E501
        [  # noqa: E501
            "uv",  # noqa: E501
            "run",  # noqa: E501
            "python",  # noqa: E501
            "evals/quality_gate.py",  # noqa: E501
            "--baseline",  # noqa: E501
            args.baseline,  # noqa: E501
            "--candidate",  # noqa: E501
            args.source,  # noqa: E501
            *(  # noqa: E501
                [  # noqa: E501
                    "--require-judge",  # noqa: E501
                    "--judge-min-absolute",  # noqa: E501
                    "6.0",  # noqa: E501
                ]  # noqa: E501
                if args.judge  # noqa: E501
                else []  # noqa: E501
            ),  # noqa: E501
        ]  # noqa: E501
    )  # noqa: E501
  # noqa: E501
  # noqa: E501
if __name__ == "__main__":  # noqa: E501
    sys.exit(main())  # noqa: E501
