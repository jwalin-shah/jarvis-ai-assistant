#!/usr/bin/env python3
"""Benchmark two-step reply gate + intent mapping on Gemini labels.

Usage:
    uv run python scripts/benchmark_two_step_intent.py
    uv run python scripts/benchmark_two_step_intent.py --backends keyword alias:falconsai
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from sklearn.metrics import accuracy_score, f1_score

from jarvis.classifiers.cascade import classify_with_cascade
from jarvis.classifiers.intent_classifier import create_intent_classifier
from jarvis.classifiers.response_mobilization import ResponsePressure

ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = ROOT / "evals" / "data" / "pipeline_eval_labeled.jsonl"
OUT_PATH = ROOT / "evals" / "results" / "two_step_benchmark.json"
LOG_PATH = ROOT / "benchmark_two_step_intent.log"

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    """Configure logging with both file and console handlers."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
    )


@dataclass
class BackendMetrics:
    backend: str
    examples: int
    should_reply_accuracy: float
    should_reply_f1: float
    mobilization_accuracy: float
    mobilization_macro_f1: float
    fallback_rate: float
    latency_ms_mean: float
    latency_ms_p95: float


def _iter_gemini_rows(limit: int | None = None) -> Iterable[dict]:
    seen = 0
    with EVAL_PATH.open() as f:
        for line in f:
            ex = json.loads(line)
            if ex.get("label_confidence") == "gemini" and ex.get("mobilization"):
                yield ex
                seen += 1
                if limit is not None and seen >= limit:
                    return


def _parse_backend(spec: str):
    if spec == "keyword":
        return create_intent_classifier("keyword"), "keyword"
    if spec.startswith("alias:"):
        alias = spec.split(":", 1)[1]
        return create_intent_classifier("alias", model_alias=alias), f"alias:{alias}"
    if spec.startswith("mlx:"):
        alias = spec.split(":", 1)[1]
        return create_intent_classifier("mlx", model_alias=alias), f"mlx:{alias}"
    raise ValueError(f"Unsupported backend spec '{spec}'")


def _partial_snapshot(
    *,
    backend_name: str,
    seen: int,
    y_true_reply: list[bool],
    y_pred_reply: list[bool],
    y_true_mob: list[str],
    y_pred_mob: list[str],
    latencies: list[float],
    fallbacks: int,
) -> str:
    return (
        f"[{backend_name}] n={seen} "
        f"reply_acc={accuracy_score(y_true_reply, y_pred_reply):.4f} "
        f"reply_f1={f1_score(y_true_reply, y_pred_reply, zero_division=0):.4f} "
        f"mob_acc={accuracy_score(y_true_mob, y_pred_mob):.4f} "
        f"mob_f1={f1_score(y_true_mob, y_pred_mob, average='macro', zero_division=0):.4f} "
        f"fallback={fallbacks / seen:.4f} "
        f"lat_ms_mean={sum(latencies) / seen:.3f}"
    )


def _run_backend(rows: Iterable[dict], backend_spec: str, *, progress_every: int = 0) -> BackendMetrics:
    classifier, backend_name = _parse_backend(backend_spec)
    # Fail fast for unavailable heavy backends (e.g., missing MLX models).
    classifier.classify("health-check", ["reply_casual_chat", "no_reply_ack"])

    y_true_reply = []
    y_pred_reply = []
    y_true_mob = []
    y_pred_mob = []
    latencies = []
    fallbacks = 0

    for row in rows:
        true_pressure = row["mobilization"].upper()
        should_reply_true = true_pressure != "NONE"

        start = time.perf_counter()
        pred = classify_with_cascade(
            row["text"],
            intent_classifier=classifier,
            threshold=0.80,
            should_reply_threshold=0.80,
        )
        latencies.append((time.perf_counter() - start) * 1000)

        y_true_reply.append(should_reply_true)
        y_pred_reply.append(pred.pressure != ResponsePressure.NONE)
        y_true_mob.append(true_pressure)
        y_pred_mob.append(pred.pressure.value.upper())
        if pred.features.get("intent_fallback", False):
            fallbacks += 1

        seen = len(y_true_reply)
        if progress_every > 0 and seen % progress_every == 0:
            print(
                _partial_snapshot(
                    backend_name=backend_name,
                    seen=seen,
                    y_true_reply=y_true_reply,
                    y_pred_reply=y_pred_reply,
                    y_true_mob=y_true_mob,
                    y_pred_mob=y_pred_mob,
                    latencies=latencies,
                    fallbacks=fallbacks,
                ),
                flush=True,
            )

    if not y_true_reply:
        raise ValueError("No evaluation rows found.")

    lat_sorted = sorted(latencies)
    total = len(y_true_reply)
    return BackendMetrics(
        backend=backend_name,
        examples=total,
        should_reply_accuracy=round(float(accuracy_score(y_true_reply, y_pred_reply)), 4),
        should_reply_f1=round(float(f1_score(y_true_reply, y_pred_reply, zero_division=0)), 4),
        mobilization_accuracy=round(float(accuracy_score(y_true_mob, y_pred_mob)), 4),
        mobilization_macro_f1=round(
            float(f1_score(y_true_mob, y_pred_mob, average="macro", zero_division=0)),
            4,
        ),
        fallback_rate=round(fallbacks / total, 4),
        latency_ms_mean=round(sum(latencies) / total, 3),
        latency_ms_p95=round(lat_sorted[int(total * 0.95) - 1], 3),
    )


def main() -> None:
    _setup_logging()
    logger.info("Starting benchmark_two_step_intent.py")
    parser = argparse.ArgumentParser(description="Benchmark two-step intent cascade")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["keyword", "alias:falconsai", "alias:deberta", "alias:flant5", "alias:mindpadi"],
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250,
        help="Print partial metrics every N examples (0 disables partial output).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max Gemini examples to evaluate (0 means all).",
    )
    args = parser.parse_args()

    limit = args.limit if args.limit > 0 else None
    results = []
    for bi, backend in enumerate(args.backends):
        logger.info("Backend %d/%d: %s", bi + 1, len(args.backends), backend)
        try:
            rows = _iter_gemini_rows(limit=limit)
            metrics = _run_backend(rows, backend, progress_every=max(args.progress_every, 0))
            results.append(asdict(metrics))
            print(f"{backend}: {metrics}", flush=True)
        except Exception as exc:
            print(f"{backend}: skipped ({exc})", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        json.dump(
            {"examples": results[0]["examples"] if results else 0, "results": results},
            f,
            indent=2,
        )
    print(f"saved: {OUT_PATH}", flush=True)
    logger.info("Finished benchmark_two_step_intent.py")


if __name__ == "__main__":
    main()
