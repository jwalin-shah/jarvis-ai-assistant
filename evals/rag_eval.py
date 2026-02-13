#!/usr/bin/env python3
"""RAG retrieval quality evaluation.

Measures whether retrieved conversation pairs actually help generation quality:
1. Retrieval relevance: Are retrieved trigger_texts semantically relevant?
2. Generation ablation: Does adding RAG examples improve judge scores?
3. Pair quality audit: Are stored pairs (trigger_text, response_text) good quality?

Usage:
    uv run python evals/rag_eval.py                    # Full eval (needs Cerebras key)
    uv run python evals/rag_eval.py --relevance-only   # Just retrieval relevance
    uv run python evals/rag_eval.py --ablation-only    # Just generation ablation
    uv run python evals/rag_eval.py --audit-only       # Just pair quality audit
    uv run python evals/rag_eval.py --audit-sample 50  # Audit N random pairs
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

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

from evals.judge_config import JUDGE_MODEL, get_judge_client  # noqa: E402

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class RelevanceResult:
    """Result of retrieval relevance scoring for one test case."""

    test_name: str
    query: str
    retrieved_triggers: list[str]
    retrieved_scores: list[float]
    relevance_scores: list[float]  # Judge scores (0-10) per retrieved item
    avg_relevance: float


@dataclass
class AblationResult:
    """Result of with/without RAG comparison for one test case."""

    test_name: str
    score_without_rag: float
    score_with_rag: float
    delta: float
    reply_without_rag: str
    reply_with_rag: str


@dataclass
class PairAuditResult:
    """Quality audit of a stored trigger-response pair."""

    rowid: int
    trigger_text: str
    response_text: str
    quality_score: float  # Judge score (0-10)
    reasoning: str


# ---------------------------------------------------------------------------
# Judge client
# ---------------------------------------------------------------------------


# get_judge_client imported from evals.judge_config


def judge_call(client, prompt: str) -> tuple[float, str]:
    """Make a judge call, return (score, reasoning)."""
    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        text = resp.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text)
        return float(data["score"]), data.get("reasoning", "")
    except Exception as e:
        return -1.0, f"judge error: {e}"


# ---------------------------------------------------------------------------
# 1. Retrieval Relevance
# ---------------------------------------------------------------------------


def eval_retrieval_relevance(client, test_cases: list[dict]) -> list[RelevanceResult]:
    """Score whether retrieved trigger_texts are relevant to test case queries."""
    from jarvis.search.vec_search import get_vec_searcher

    searcher = get_vec_searcher()
    results = []

    for tc in tqdm(test_cases, desc="Retrieval relevance"):
        query = tc["last_message"]
        name = tc["name"]

        # Retrieve similar chunks
        search_results = searcher.search_with_chunks_global(query=query, limit=3)

        triggers = []
        vec_scores = []
        relevance_scores = []

        for sr in search_results:
            trigger = sr.trigger_text or ""
            triggers.append(trigger)
            vec_scores.append(sr.score)

            if not trigger.strip():
                relevance_scores.append(0.0)
                continue

            # Judge: is this retrieved trigger relevant to the query?
            prompt = (
                "You are evaluating retrieval quality for a text message reply system.\n\n"
                f"QUERY (message to reply to):\n{query}\n\n"
                f"RETRIEVED CONTEXT:\n{trigger}\n\n"
                "Score 0-10: How relevant is the retrieved context to the query?\n"
                "10 = Perfect match (same topic/intent), 0 = Completely unrelated.\n"
                "Respond in JSON: "
                '{"score": <0-10>, "reasoning": "<1 sentence>"}'
            )
            score, _ = judge_call(client, prompt)
            relevance_scores.append(max(0.0, score))

        avg_rel = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        results.append(
            RelevanceResult(
                test_name=name,
                query=query,
                retrieved_triggers=triggers,
                retrieved_scores=vec_scores,
                relevance_scores=relevance_scores,
                avg_relevance=avg_rel,
            )
        )

        print(f"  {name}: avg_relevance={avg_rel:.1f}/10 ({len(triggers)} retrieved)", flush=True)

    return results


# ---------------------------------------------------------------------------
# 2. Generation Ablation (with vs without RAG)
# ---------------------------------------------------------------------------


def eval_generation_ablation(client, test_cases: list[dict]) -> list[AblationResult]:
    """Compare generation quality WITH vs WITHOUT RAG examples."""
    from evals.batch_eval import build_prompt, judge_response
    from jarvis.search.vec_search import get_vec_searcher
    from models.loader import get_model

    searcher = get_vec_searcher()
    results = []

    # Load model once

    loader = get_model()
    if not loader.is_loaded():
        loader.load()

    for tc in tqdm(test_cases, desc="Generation ablation"):
        name = tc["name"]
        query = tc["last_message"]

        # --- Without RAG: base prompt ---
        prompt_no_rag = build_prompt(tc)
        result_no_rag = loader.generate_sync(
            prompt=prompt_no_rag,
            temperature=0.1,
            max_tokens=50,
            top_p=0.1,
            top_k=50,
            repetition_penalty=1.05,
        )
        reply_no_rag = result_no_rag.text.strip()

        # --- With RAG: add retrieved examples to prompt ---
        search_results = searcher.search_with_chunks_global(query=query, limit=3)
        rag_examples = ""
        for sr in search_results:
            if sr.trigger_text and sr.response_text:
                rag_examples += (
                    f"\nExample:\n"
                    f"  Context: {sr.trigger_text[:200]}\n"
                    f"  Reply: {sr.response_text[:100]}\n"
                )

        # Build RAG-enhanced prompt by inserting examples before <reply>
        prompt_with_rag = prompt_no_rag
        if rag_examples:
            prompt_with_rag = prompt_no_rag.replace(
                "<reply>",
                f"<similar_exchanges>{rag_examples}\n</similar_exchanges>\n\n<reply>",
            )

        result_with_rag = loader.generate_sync(
            prompt=prompt_with_rag,
            temperature=0.1,
            max_tokens=50,
            top_p=0.1,
            top_k=50,
            repetition_penalty=1.05,
        )
        reply_with_rag = result_with_rag.text.strip()

        # Score both with judge
        score_no, _ = judge_response(client, tc, reply_no_rag)
        score_rag, _ = judge_response(client, tc, reply_with_rag)

        delta = score_rag - score_no
        results.append(
            AblationResult(
                test_name=name,
                score_without_rag=score_no,
                score_with_rag=score_rag,
                delta=delta,
                reply_without_rag=reply_no_rag,
                reply_with_rag=reply_with_rag,
            )
        )

        marker = "+" if delta > 0 else ("-" if delta < 0 else "=")
        print(
            f"  {name}: no_rag={score_no:.0f} rag={score_rag:.0f} delta={marker}{abs(delta):.0f}",
            flush=True,
        )

    return results


# ---------------------------------------------------------------------------
# 3. Pair Quality Audit
# ---------------------------------------------------------------------------


def eval_pair_quality(client, sample_size: int = 50) -> list[PairAuditResult]:
    """Audit quality of stored trigger-response pairs in vec_chunks."""
    from jarvis.db.core import JarvisDB

    db = JarvisDB()
    db.connect()

    # Sample random pairs from vec_chunks
    rows = db.execute(
        "SELECT rowid, trigger_text, response_text FROM vec_chunks "
        "WHERE trigger_text IS NOT NULL AND response_text IS NOT NULL "
        "ORDER BY RANDOM() LIMIT ?",
        (sample_size,),
    ).fetchall()

    if not rows:
        print("  No pairs found in vec_chunks. Is the database populated?", flush=True)
        return []

    print(f"  Auditing {len(rows)} random pairs from vec_chunks...", flush=True)
    results = []

    for rowid, trigger, response in tqdm(rows, desc="Auditing pairs"):
        if not trigger or not response:
            continue

        # Truncate for judge prompt
        trigger_preview = trigger[:300]
        response_preview = response[:200]

        prompt = (
            "You are auditing conversation pair quality for a text message system.\n\n"
            f"TRIGGER (incoming message):\n{trigger_preview}\n\n"
            f"RESPONSE (user's actual reply):\n{response_preview}\n\n"
            "Score 0-10: Is this response a good, natural reply to the trigger?\n"
            "10 = Perfect match, natural exchange. "
            "0 = Wrong pairing, makes no sense as a reply.\n"
            "Consider: Does the response relate to the trigger? "
            "Is it a plausible human reply?\n"
            "Respond in JSON: "
            '{"score": <0-10>, "reasoning": "<1 sentence>"}'
        )

        score, reasoning = judge_call(client, prompt)
        results.append(
            PairAuditResult(
                rowid=rowid,
                trigger_text=trigger_preview,
                response_text=response_preview,
                quality_score=max(0.0, score),
                reasoning=reasoning,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="JARVIS RAG Quality Evaluation")
    parser.add_argument(
        "--relevance-only", action="store_true", help="Only run retrieval relevance"
    )
    parser.add_argument("--ablation-only", action="store_true", help="Only run generation ablation")
    parser.add_argument("--audit-only", action="store_true", help="Only run pair quality audit")
    parser.add_argument(
        "--audit-sample", type=int, default=50, help="Number of pairs to audit (default: 50)"
    )
    args = parser.parse_args()

    # Setup logging
    log_path = PROJECT_ROOT / "results" / "rag_eval.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger(__name__)

    run_all = not (args.relevance_only or args.ablation_only or args.audit_only)

    print("=" * 70, flush=True)
    print("JARVIS RAG Quality Evaluation", flush=True)
    print("=" * 70, flush=True)

    client = get_judge_client()
    if client is None:
        print("ERROR: Judge API key not set in .env", flush=True)
        print("       Required for judge scoring.", flush=True)
        return 1

    # Load test cases
    from evals.batch_eval import TEST_CASES

    print(f"Test cases: {len(TEST_CASES)}", flush=True)
    print(f"Judge: {JUDGE_MODEL} via DeepInfra", flush=True)
    print(flush=True)

    start = time.perf_counter()
    output: dict = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

    # --- 1. Retrieval Relevance ---
    if run_all or args.relevance_only:
        print("-" * 70, flush=True)
        print("1. RETRIEVAL RELEVANCE", flush=True)
        print("-" * 70, flush=True)
        relevance_results = eval_retrieval_relevance(client, TEST_CASES)

        if relevance_results:
            scores = [r.avg_relevance for r in relevance_results if r.avg_relevance >= 0]
            avg = sum(scores) / len(scores) if scores else 0
            high = sum(1 for s in scores if s >= 7)
            print(f"\n  Avg relevance: {avg:.1f}/10", flush=True)
            print(
                f"  High relevance (>=7): {high}/{len(scores)} ({high / len(scores) * 100:.0f}%)",
                flush=True,
            )
            output["retrieval_relevance"] = {
                "avg_score": round(avg, 2),
                "high_relevance_rate": round(high / len(scores), 4) if scores else 0,
                "per_case": [
                    {
                        "name": r.test_name,
                        "avg_relevance": round(r.avg_relevance, 2),
                        "n_retrieved": len(r.retrieved_triggers),
                    }
                    for r in relevance_results
                ],
            }
        print(flush=True)

    # --- 2. Generation Ablation ---
    if run_all or args.ablation_only:
        print("-" * 70, flush=True)
        print("2. GENERATION ABLATION (with vs without RAG)", flush=True)
        print("-" * 70, flush=True)
        ablation_results = eval_generation_ablation(client, TEST_CASES)

        if ablation_results:
            valid = [
                r for r in ablation_results if r.score_without_rag >= 0 and r.score_with_rag >= 0
            ]
            if valid:
                avg_no_rag = sum(r.score_without_rag for r in valid) / len(valid)
                avg_with_rag = sum(r.score_with_rag for r in valid) / len(valid)
                avg_delta = sum(r.delta for r in valid) / len(valid)
                improved = sum(1 for r in valid if r.delta > 0)
                degraded = sum(1 for r in valid if r.delta < 0)
                neutral = sum(1 for r in valid if r.delta == 0)

                print(f"\n  Avg score WITHOUT RAG: {avg_no_rag:.1f}/10", flush=True)
                print(f"  Avg score WITH RAG:    {avg_with_rag:.1f}/10", flush=True)
                print(f"  Avg delta:             {avg_delta:+.1f}", flush=True)
                print(
                    f"  Improved: {improved}  Degraded: {degraded}  Neutral: {neutral}", flush=True
                )

                output["generation_ablation"] = {
                    "avg_without_rag": round(avg_no_rag, 2),
                    "avg_with_rag": round(avg_with_rag, 2),
                    "avg_delta": round(avg_delta, 2),
                    "improved": improved,
                    "degraded": degraded,
                    "neutral": neutral,
                }
        print(flush=True)

    # --- 3. Pair Quality Audit ---
    if run_all or args.audit_only:
        print("-" * 70, flush=True)
        print(f"3. PAIR QUALITY AUDIT (sample={args.audit_sample})", flush=True)
        print("-" * 70, flush=True)
        audit_results = eval_pair_quality(client, args.audit_sample)

        if audit_results:
            scores = [r.quality_score for r in audit_results if r.quality_score >= 0]
            avg = sum(scores) / len(scores) if scores else 0
            good = sum(1 for s in scores if s >= 7)
            bad = sum(1 for s in scores if s < 4)

            print(f"\n  Pairs audited: {len(audit_results)}", flush=True)
            print(f"  Avg quality: {avg:.1f}/10", flush=True)
            print(
                f"  Good (>=7): {good}/{len(scores)} ({good / len(scores) * 100:.0f}%)", flush=True
            )
            print(f"  Bad (<4): {bad}/{len(scores)} ({bad / len(scores) * 100:.0f}%)", flush=True)

            if bad > 0:
                print("\n  Worst pairs:", flush=True)
                worst = sorted(audit_results, key=lambda r: r.quality_score)[:5]
                for r in worst:
                    print(
                        f"    rowid={r.rowid} score={r.quality_score:.0f}: "
                        f"{r.trigger_text[:50]!r} -> {r.response_text[:50]!r}",
                        flush=True,
                    )
                    print(f"      {r.reasoning}", flush=True)

            output["pair_quality_audit"] = {
                "sample_size": len(audit_results),
                "avg_quality": round(avg, 2),
                "good_rate": round(good / len(scores), 4) if scores else 0,
                "bad_rate": round(bad / len(scores), 4) if scores else 0,
            }
        print(flush=True)

    # --- Summary ---
    elapsed = time.perf_counter() - start
    print("=" * 70, flush=True)
    print(f"RAG eval completed in {elapsed:.1f}s", flush=True)
    print("=" * 70, flush=True)

    # Save results
    output_path = PROJECT_ROOT / "results" / "rag_eval_latest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(f"Results saved to: {output_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
