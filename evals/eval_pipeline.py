#!/usr/bin/env python3
"""Baseline evaluation pipeline for the 6-category reply generation system.

Runs the full pipeline (classifier + generation) on the gold eval dataset,
scores outputs across multiple dimensions, and produces a report.

Usage:
    uv run python evals/eval_pipeline.py                  # local checks only
    uv run python evals/eval_pipeline.py --judge           # + LLM judge scoring
    uv run python evals/eval_pipeline.py --similarity      # + BERT cosine similarity
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

ANTI_AI_PHRASES = [
    "i'd be happy to",
    "i hope this helps",
    "let me know if",
    "i understand",
    "as an ai",
    "i'm an ai",
    "certainly!",
    "of course!",
    "great question",
]

EVAL_DATASET_PATH = PROJECT_ROOT / "evals" / "eval_dataset.jsonl"

CATEGORIES = ["acknowledge", "closing", "question", "request", "emotion", "statement"]


@dataclass
class EvalExample:
    category: str
    context: list[str]
    last_message: str
    ideal_response: str
    contact_style: str
    notes: str


@dataclass
class EvalResult:
    example: EvalExample
    predicted_category: str
    generated_response: str
    latency_ms: float
    category_match: bool
    anti_ai_violations: list[str] = field(default_factory=list)
    response_length: int = 0
    similarity_score: float | None = None
    judge_score: float | None = None
    judge_reasoning: str = ""


def load_eval_dataset(path: Path) -> list[EvalExample]:
    """Load eval dataset from JSONL."""
    examples = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        examples.append(EvalExample(
            category=data["category"],
            context=data["context"],
            last_message=data["last_message"],
            ideal_response=data["ideal_response"],
            contact_style=data.get("contact_style", "casual"),
            notes=data.get("notes", ""),
        ))
    return examples


def check_anti_ai(text: str) -> list[str]:
    """Check for AI-sounding phrases."""
    lower = text.lower()
    return [phrase for phrase in ANTI_AI_PHRASES if phrase in lower]


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="JARVIS Eval Pipeline")
    parser.add_argument("--judge", action="store_true", help="Enable LLM judge scoring")
    parser.add_argument("--similarity", action="store_true", help="Enable BERT cosine similarity")
    args = parser.parse_args()

    # Load dataset
    if not EVAL_DATASET_PATH.exists():
        print(f"ERROR: Eval dataset not found at {EVAL_DATASET_PATH}")
        return 1

    examples = load_eval_dataset(EVAL_DATASET_PATH)
    print("=" * 70, flush=True)
    print("JARVIS EVAL PIPELINE - Baseline Measurement", flush=True)
    print("=" * 70, flush=True)
    print(f"Examples:    {len(examples)}", flush=True)
    print(f"Categories:  {', '.join(CATEGORIES)}", flush=True)
    print(f"Judge:       {'enabled' if args.judge else 'disabled (use --judge)'}", flush=True)
    print(f"Similarity:  {'enabled' if args.similarity else 'disabled (use --similarity)'}", flush=True)
    print(flush=True)

    # Initialize components
    print("Loading classifier...", flush=True)
    from jarvis.classifiers.category_classifier import classify_category
    from jarvis.classifiers.response_mobilization import classify_response_pressure

    # Initialize reply service for generation
    print("Loading reply service...", flush=True)
    from jarvis.reply_service import ReplyService
    reply_service = ReplyService()

    # Optional: BERT embedder for similarity
    embedder = None
    if args.similarity:
        print("Loading embedder for similarity...", flush=True)
        from jarvis.embedding_adapter import get_embedder
        embedder = get_embedder()

    # Optional: judge
    judge_client = None
    if args.judge:
        from evals.judge_config import JUDGE_MODEL, get_judge_client
        judge_client = get_judge_client()
        if judge_client is None:
            print("WARNING: Judge API key not set, skipping judge", flush=True)
        else:
            print(f"Judge ready: {JUDGE_MODEL}", flush=True)

    print(flush=True)
    print("-" * 70, flush=True)

    results: list[EvalResult] = []
    total_start = time.perf_counter()

    for i, ex in enumerate(examples, 1):
        print(f"\n[{i:2d}/{len(examples)}] [{ex.category}] {ex.last_message[:50]}...", flush=True)

        gen_start = time.perf_counter()

        # 1. Classify category
        mobilization = classify_response_pressure(ex.last_message)
        category_result = classify_category(
            ex.last_message, context=ex.context, mobilization=mobilization,
        )
        predicted_category = category_result.category

        # 2. Generate reply via full pipeline (empty search_results to skip RAG)
        try:
            reply_result = reply_service.generate_reply(
                incoming=ex.last_message,
                thread=ex.context,
                mobilization=mobilization,
                search_results=[],
            )
            generated = reply_result.get("response", "")
        except Exception as e:
            generated = f"[ERROR: {e}]"

        latency_ms = (time.perf_counter() - gen_start) * 1000

        # 3. Score
        category_match = predicted_category == ex.category
        anti_ai = check_anti_ai(generated)

        # Similarity scoring
        sim_score = None
        if embedder and generated and not generated.startswith("[ERROR"):
            try:
                import numpy as np
                emb_gen = embedder.encode([generated])[0]
                emb_ideal = embedder.encode([ex.ideal_response])[0]
                sim_score = float(np.dot(emb_gen, emb_ideal) / (
                    np.linalg.norm(emb_gen) * np.linalg.norm(emb_ideal)
                ))
            except Exception:
                pass

        # Judge scoring
        j_score = None
        j_reasoning = ""
        if judge_client and generated and not generated.startswith("[ERROR"):
            try:
                prompt = (
                    "You are an expert evaluator for a text message reply generator.\n\n"
                    f"CONVERSATION CONTEXT:\n{chr(10).join(ex.context)}\n\n"
                    f"LAST MESSAGE (to reply to):\n{ex.last_message}\n\n"
                    f"IDEAL RESPONSE:\n{ex.ideal_response}\n\n"
                    f"GENERATED REPLY:\n{generated}\n\n"
                    f"CATEGORY: {ex.category}\n"
                    f"NOTES: {ex.notes}\n\n"
                    "Score the generated reply from 0-10. Consider:\n"
                    "- Does it match the tone and intent of the ideal response?\n"
                    "- Does it sound like a real person texting (not an AI)?\n"
                    "- Is it appropriate for the category?\n"
                    "- Is the length appropriate?\n\n"
                    'Respond in JSON: {"score": <0-10>, "reasoning": "<1-2 sentences>"}'
                )
                resp = judge_client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=150,
                )
                text = resp.choices[0].message.content.strip()
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                data = json.loads(text)
                j_score = float(data["score"])
                j_reasoning = data.get("reasoning", "")
            except Exception as e:
                j_reasoning = f"judge error: {e}"

        result = EvalResult(
            example=ex,
            predicted_category=predicted_category,
            generated_response=generated,
            latency_ms=latency_ms,
            category_match=category_match,
            anti_ai_violations=anti_ai,
            response_length=len(generated),
            similarity_score=sim_score,
            judge_score=j_score,
            judge_reasoning=j_reasoning,
        )
        results.append(result)

        # Print per-example
        cat_status = "OK" if category_match else f"MISS (got {predicted_category})"
        ai_status = f"AI:{len(anti_ai)}" if anti_ai else "clean"
        sim_str = f"sim={sim_score:.2f}" if sim_score is not None else ""
        judge_str = f"judge={j_score:.0f}/10" if j_score is not None else ""
        print(f'  Response: "{generated[:60]}"', flush=True)
        print(f"  Cat: {cat_status} | {ai_status} | {latency_ms:.0f}ms {sim_str} {judge_str}",
              flush=True)

    total_ms = (time.perf_counter() - total_start) * 1000

    # =========================================================================
    # Summary Report
    # =========================================================================
    print(flush=True)
    print("=" * 70, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 70, flush=True)

    n = len(results)
    cat_matches = sum(1 for r in results if r.category_match)
    ai_clean = sum(1 for r in results if not r.anti_ai_violations)
    latencies = [r.latency_ms for r in results]
    avg_lat = sum(latencies) / n if n else 0
    sorted_lat = sorted(latencies)
    p50 = sorted_lat[n // 2] if n else 0
    p95 = sorted_lat[min(int(n * 0.95), n - 1)] if n else 0

    print(f"Category accuracy:  {cat_matches}/{n} ({cat_matches / n * 100:.0f}%)", flush=True)
    print(f"Anti-AI clean:      {ai_clean}/{n} ({ai_clean / n * 100:.0f}%)", flush=True)
    print(f"Total time:         {total_ms:.0f}ms", flush=True)
    print(f"Avg latency:        {avg_lat:.0f}ms", flush=True)
    print(f"P50/P95 latency:    {p50:.0f}ms / {p95:.0f}ms", flush=True)

    # Similarity summary
    scored_sim = [r for r in results if r.similarity_score is not None]
    if scored_sim:
        sims = [r.similarity_score for r in scored_sim]
        print(f"Avg similarity:     {sum(sims) / len(sims):.3f}", flush=True)

    # Judge summary
    scored_judge = [r for r in results if r.judge_score is not None and r.judge_score >= 0]
    if scored_judge:
        scores = [r.judge_score for r in scored_judge]
        avg_j = sum(scores) / len(scores)
        pass_7 = sum(1 for s in scores if s >= 7)
        print(f"Judge avg:          {avg_j:.1f}/10", flush=True)
        print(f"Judge pass (>=7):   {pass_7}/{len(scores)} ({pass_7 / len(scores) * 100:.0f}%)",
              flush=True)

    # Per-category breakdown
    print(flush=True)
    print("PER-CATEGORY BREAKDOWN", flush=True)
    print("-" * 70, flush=True)
    for cat in CATEGORIES:
        cat_results = [r for r in results if r.example.category == cat]
        if not cat_results:
            continue
        cat_correct = sum(1 for r in cat_results if r.category_match)
        cat_clean = sum(1 for r in cat_results if not r.anti_ai_violations)
        parts = [
            f"classify={cat_correct}/{len(cat_results)}",
            f"clean={cat_clean}/{len(cat_results)}",
        ]
        cat_sim = [r for r in cat_results if r.similarity_score is not None]
        if cat_sim:
            avg_s = sum(r.similarity_score for r in cat_sim) / len(cat_sim)
            parts.append(f"sim={avg_s:.2f}")
        cat_j = [r for r in cat_results if r.judge_score is not None and r.judge_score >= 0]
        if cat_j:
            avg_jj = sum(r.judge_score for r in cat_j) / len(cat_j)
            parts.append(f"judge={avg_jj:.1f}")
        print(f"  {cat:15s}  {' | '.join(parts)}", flush=True)

    # Category misclassifications
    misses = [r for r in results if not r.category_match]
    if misses:
        print(flush=True)
        print("CATEGORY MISCLASSIFICATIONS", flush=True)
        print("-" * 70, flush=True)
        for r in misses:
            print(f"  [{r.example.category} -> {r.predicted_category}] "
                  f"{r.example.last_message[:50]}", flush=True)

    # Anti-AI violations
    violations = [r for r in results if r.anti_ai_violations]
    if violations:
        print(flush=True)
        print("ANTI-AI VIOLATIONS", flush=True)
        print("-" * 70, flush=True)
        for r in violations:
            print(f'  "{r.generated_response[:60]}" -> {r.anti_ai_violations}', flush=True)

    # Worst judge scores
    if scored_judge:
        low = sorted(scored_judge, key=lambda r: r.judge_score)[:5]
        if low and low[0].judge_score < 7:
            print(flush=True)
            print("LOWEST JUDGE SCORES", flush=True)
            print("-" * 70, flush=True)
            for r in low:
                if r.judge_score < 7:
                    print(f"  [{r.example.category}] {r.judge_score:.0f}/10 - "
                          f'"{r.generated_response[:50]}" - {r.judge_reasoning}', flush=True)

    # Save results
    output_path = PROJECT_ROOT / "results" / "eval_pipeline_baseline.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_examples": n,
        "category_accuracy": round(cat_matches / n, 4) if n else 0,
        "anti_ai_clean_rate": round(ai_clean / n, 4) if n else 0,
        "avg_similarity": round(sum(sims) / len(sims), 4) if scored_sim else None,
        "judge_avg": round(avg_j, 2) if scored_judge else None,
        "latency": {
            "avg_ms": round(avg_lat, 1),
            "p50_ms": round(p50, 1),
            "p95_ms": round(p95, 1),
            "total_ms": round(total_ms, 1),
        },
        "per_category": {},
        "results": [
            {
                "category_expected": r.example.category,
                "category_predicted": r.predicted_category,
                "last_message": r.example.last_message,
                "ideal_response": r.example.ideal_response,
                "generated_response": r.generated_response,
                "category_match": r.category_match,
                "anti_ai_violations": r.anti_ai_violations,
                "similarity_score": r.similarity_score,
                "judge_score": r.judge_score,
                "judge_reasoning": r.judge_reasoning,
                "latency_ms": round(r.latency_ms, 1),
            }
            for r in results
        ],
    }

    # Per-category stats
    for cat in CATEGORIES:
        cat_results = [r for r in results if r.example.category == cat]
        if not cat_results:
            continue
        output_data["per_category"][cat] = {
            "count": len(cat_results),
            "classify_accuracy": round(
                sum(1 for r in cat_results if r.category_match) / len(cat_results), 4
            ),
            "anti_ai_clean_rate": round(
                sum(1 for r in cat_results if not r.anti_ai_violations) / len(cat_results), 4
            ),
        }

    output_path.write_text(json.dumps(output_data, indent=2))
    print(f"\nResults saved to: {output_path}", flush=True)
    print("=" * 70, flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
