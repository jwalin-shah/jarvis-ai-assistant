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
  # noqa: E402
# Load .env  # noqa: E402
_env_path = PROJECT_ROOT / ".env"  # noqa: E402
if _env_path.exists():  # noqa: E402
    for line in _env_path.read_text().splitlines():  # noqa: E402
        line = line.strip()  # noqa: E402
        if line and not line.startswith("#") and "=" in line:  # noqa: E402
            key, _, val = line.partition("=")  # noqa: E402
            os.environ.setdefault(key.strip(), val.strip())  # noqa: E402
  # noqa: E402
from evals.judge_config import JUDGE_MODEL, get_judge_client  # noqa: E402


  # noqa: E402
# ---------------------------------------------------------------------------  # noqa: E402
# Data types  # noqa: E402
# ---------------------------------------------------------------------------  # noqa: E402
  # noqa: E402
  # noqa: E402
@dataclass  # noqa: E402
class RelevanceResult:  # noqa: E402
    """Result of retrieval relevance scoring for one test case."""  # noqa: E402
  # noqa: E402
    test_name: str  # noqa: E402
    query: str  # noqa: E402
    retrieved_triggers: list[str]  # noqa: E402
    retrieved_scores: list[float]  # noqa: E402
    relevance_scores: list[float]  # Judge scores (0-10) per retrieved item  # noqa: E402
    avg_relevance: float  # noqa: E402
  # noqa: E402
  # noqa: E402
@dataclass  # noqa: E402
class AblationResult:  # noqa: E402
    """Result of with/without RAG comparison for one test case."""  # noqa: E402
  # noqa: E402
    test_name: str  # noqa: E402
    score_without_rag: float  # noqa: E402
    score_with_rag: float  # noqa: E402
    delta: float  # noqa: E402
    reply_without_rag: str  # noqa: E402
    reply_with_rag: str  # noqa: E402
  # noqa: E402
  # noqa: E402
@dataclass  # noqa: E402
class PairAuditResult:  # noqa: E402
    """Quality audit of a stored trigger-response pair."""  # noqa: E402
  # noqa: E402
    rowid: int  # noqa: E402
    trigger_text: str  # noqa: E402
    response_text: str  # noqa: E402
    quality_score: float  # Judge score (0-10)  # noqa: E402
    reasoning: str  # noqa: E402
  # noqa: E402
  # noqa: E402
# ---------------------------------------------------------------------------  # noqa: E402
# Judge client  # noqa: E402
# ---------------------------------------------------------------------------  # noqa: E402
  # noqa: E402
  # noqa: E402
# get_judge_client imported from evals.judge_config  # noqa: E402
  # noqa: E402
  # noqa: E402
def judge_call(client, prompt: str) -> tuple[float, str]:  # noqa: E402
    """Make a judge call, return (score, reasoning)."""  # noqa: E402
    try:  # noqa: E402
        resp = client.chat.completions.create(  # noqa: E402
            model=JUDGE_MODEL,  # noqa: E402
            messages=[{"role": "user", "content": prompt}],  # noqa: E402
            temperature=0.0,  # noqa: E402
            max_tokens=200,  # noqa: E402
        )  # noqa: E402
        text = resp.choices[0].message.content.strip()  # noqa: E402
        if text.startswith("```"):  # noqa: E402
            text = text.split("```")[1]  # noqa: E402
            if text.startswith("json"):  # noqa: E402
                text = text[4:]  # noqa: E402
        data = json.loads(text)  # noqa: E402
        return float(data["score"]), data.get("reasoning", "")  # noqa: E402
    except Exception as e:  # noqa: E402
        return -1.0, f"judge error: {e}"  # noqa: E402
  # noqa: E402
  # noqa: E402
# ---------------------------------------------------------------------------  # noqa: E402
# 1. Retrieval Relevance  # noqa: E402
# ---------------------------------------------------------------------------  # noqa: E402
  # noqa: E402
  # noqa: E402
def eval_retrieval_relevance(client, test_cases: list[dict]) -> list[RelevanceResult]:  # noqa: E402
    """Score whether retrieved trigger_texts are relevant to test case queries."""  # noqa: E402
    from jarvis.search.vec_search import get_vec_searcher  # noqa: E402
  # noqa: E402
    searcher = get_vec_searcher()  # noqa: E402
    results = []  # noqa: E402
  # noqa: E402
    for tc in tqdm(test_cases, desc="Retrieval relevance"):  # noqa: E402
        query = tc["last_message"]  # noqa: E402
        name = tc["name"]  # noqa: E402
  # noqa: E402
        # Retrieve similar chunks  # noqa: E402
        search_results = searcher.search_with_chunks_global(query=query, limit=3)  # noqa: E402
  # noqa: E402
        triggers = []  # noqa: E402
        vec_scores = []  # noqa: E402
        relevance_scores = []  # noqa: E402
  # noqa: E402
        for sr in search_results:  # noqa: E402
            trigger = sr.context_text or ""  # noqa: E402
            triggers.append(trigger)  # noqa: E402
            vec_scores.append(sr.score)  # noqa: E402
  # noqa: E402
            if not trigger.strip():  # noqa: E402
                relevance_scores.append(0.0)  # noqa: E402
                continue  # noqa: E402
  # noqa: E402
            # Judge: is this retrieved trigger relevant to the query?  # noqa: E402
            prompt = (  # noqa: E402
                "You are evaluating retrieval quality for a text message reply system.\n\n"  # noqa: E402
                f"QUERY (message to reply to):\n{query}\n\n"  # noqa: E402
                f"RETRIEVED CONTEXT:\n{trigger}\n\n"  # noqa: E402
                "Score 0-10: How relevant is the retrieved context to the query?\n"  # noqa: E402
                "10 = Perfect match (same topic/intent), 0 = Completely unrelated.\n"  # noqa: E402
                "Respond in JSON: "  # noqa: E402
                '{"score": <0-10>, "reasoning": "<1 sentence>"}'  # noqa: E402
            )  # noqa: E402
            score, _ = judge_call(client, prompt)  # noqa: E402
            relevance_scores.append(max(0.0, score))  # noqa: E402
  # noqa: E402
        avg_rel = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0  # noqa: E402
        results.append(  # noqa: E402
            RelevanceResult(  # noqa: E402
                test_name=name,  # noqa: E402
                query=query,  # noqa: E402
                retrieved_triggers=triggers,  # noqa: E402
                retrieved_scores=vec_scores,  # noqa: E402
                relevance_scores=relevance_scores,  # noqa: E402
                avg_relevance=avg_rel,  # noqa: E402
            )  # noqa: E402
        )  # noqa: E402
  # noqa: E402
        print(f"  {name}: avg_relevance={avg_rel:.1f}/10 ({len(triggers)} retrieved)", flush=True)  # noqa: E402
  # noqa: E402
    return results  # noqa: E402
  # noqa: E402
  # noqa: E402
# ---------------------------------------------------------------------------  # noqa: E402
# 2. Generation Ablation (with vs without RAG)  # noqa: E402
# ---------------------------------------------------------------------------  # noqa: E402
  # noqa: E402
  # noqa: E402
def eval_generation_ablation(client, test_cases: list[dict]) -> list[AblationResult]:  # noqa: E402
    """Compare generation quality WITH vs WITHOUT RAG examples."""  # noqa: E402
    from evals.batch_eval import build_prompt, judge_response  # noqa: E402

    # noqa: E402
    from jarvis.search.vec_search import get_vec_searcher  # noqa: E402
    from models.loader import get_model  # noqa: E402
  # noqa: E402
    searcher = get_vec_searcher()  # noqa: E402
    results = []  # noqa: E402
  # noqa: E402
    # Load model once  # noqa: E402
  # noqa: E402
    loader = get_model()  # noqa: E402
    if not loader.is_loaded():  # noqa: E402
        loader.load()  # noqa: E402
  # noqa: E402
    for tc in tqdm(test_cases, desc="Generation ablation"):  # noqa: E402
        name = tc["name"]  # noqa: E402
        query = tc["last_message"]  # noqa: E402
  # noqa: E402
        # --- Without RAG: base prompt ---  # noqa: E402
        prompt_no_rag = build_prompt(tc)  # noqa: E402
        result_no_rag = loader.generate_sync(  # noqa: E402
            prompt=prompt_no_rag,  # noqa: E402
            temperature=0.1,  # noqa: E402
            max_tokens=50,  # noqa: E402
            top_p=0.1,  # noqa: E402
            top_k=50,  # noqa: E402
            repetition_penalty=1.05,  # noqa: E402
        )  # noqa: E402
        reply_no_rag = result_no_rag.text.strip()  # noqa: E402
  # noqa: E402
        # --- With RAG: add retrieved examples to prompt ---  # noqa: E402
        search_results = searcher.search_with_chunks_global(query=query, limit=3)  # noqa: E402
        rag_examples = ""  # noqa: E402
        for sr in search_results:  # noqa: E402
            if sr.context_text and sr.reply_text:  # noqa: E402
                rag_examples += (  # noqa: E402
                    f"\nExample:\n"  # noqa: E402
                    f"  Context: {sr.context_text[:200]}\n"  # noqa: E402
                    f"  Reply: {sr.reply_text[:100]}\n"  # noqa: E402
                )  # noqa: E402
  # noqa: E402
        # Build RAG-enhanced prompt by inserting examples before <reply>  # noqa: E402
        prompt_with_rag = prompt_no_rag  # noqa: E402
        if rag_examples:  # noqa: E402
            prompt_with_rag = prompt_no_rag.replace(  # noqa: E402
                "<reply>",  # noqa: E402
                f"<similar_exchanges>{rag_examples}\n</similar_exchanges>\n\n<reply>",  # noqa: E402
            )  # noqa: E402
  # noqa: E402
        result_with_rag = loader.generate_sync(  # noqa: E402
            prompt=prompt_with_rag,  # noqa: E402
            temperature=0.1,  # noqa: E402
            max_tokens=50,  # noqa: E402
            top_p=0.1,  # noqa: E402
            top_k=50,  # noqa: E402
            repetition_penalty=1.05,  # noqa: E402
        )  # noqa: E402
        reply_with_rag = result_with_rag.text.strip()  # noqa: E402
  # noqa: E402
        # Score both with judge  # noqa: E402
        score_no, _ = judge_response(client, tc, reply_no_rag)  # noqa: E402
        score_rag, _ = judge_response(client, tc, reply_with_rag)  # noqa: E402
  # noqa: E402
        delta = score_rag - score_no  # noqa: E402
        results.append(  # noqa: E402
            AblationResult(  # noqa: E402
                test_name=name,  # noqa: E402
                score_without_rag=score_no,  # noqa: E402
                score_with_rag=score_rag,  # noqa: E402
                delta=delta,  # noqa: E402
                reply_without_rag=reply_no_rag,  # noqa: E402
                reply_with_rag=reply_with_rag,  # noqa: E402
            )  # noqa: E402
        )  # noqa: E402
  # noqa: E402
        marker = "+" if delta > 0 else ("-" if delta < 0 else "=")  # noqa: E402
        print(  # noqa: E402
            f"  {name}: no_rag={score_no:.0f} rag={score_rag:.0f} delta={marker}{abs(delta):.0f}",  # noqa: E402
            flush=True,  # noqa: E402
        )  # noqa: E402
  # noqa: E402
    return results  # noqa: E402
  # noqa: E402
  # noqa: E402
# ---------------------------------------------------------------------------  # noqa: E402
# 3. Pair Quality Audit  # noqa: E402
# ---------------------------------------------------------------------------  # noqa: E402
  # noqa: E402
  # noqa: E402
def eval_pair_quality(client, sample_size: int = 50) -> list[PairAuditResult]:  # noqa: E402
    """Audit quality of stored trigger-response pairs in vec_chunks."""  # noqa: E402
    from jarvis.db.core import JarvisDB  # noqa: E402
  # noqa: E402
    db = JarvisDB()  # noqa: E402
    db.connect()  # noqa: E402
  # noqa: E402
    # Sample random pairs from vec_chunks  # noqa: E402
    rows = db.execute(  # noqa: E402
        "SELECT rowid, trigger_text, response_text FROM vec_chunks "  # noqa: E402
        "WHERE trigger_text IS NOT NULL AND response_text IS NOT NULL "  # noqa: E402
        "ORDER BY RANDOM() LIMIT ?",  # noqa: E402
        (sample_size,),  # noqa: E402
    ).fetchall()  # noqa: E402
  # noqa: E402
    if not rows:  # noqa: E402
        print("  No pairs found in vec_chunks. Is the database populated?", flush=True)  # noqa: E402
        return []  # noqa: E402
  # noqa: E402
    print(f"  Auditing {len(rows)} random pairs from vec_chunks...", flush=True)  # noqa: E402
    results = []  # noqa: E402
  # noqa: E402
    for rowid, trigger, response in tqdm(rows, desc="Auditing pairs"):  # noqa: E402
        if not trigger or not response:  # noqa: E402
            continue  # noqa: E402
  # noqa: E402
        # Truncate for judge prompt  # noqa: E402
        trigger_preview = trigger[:300]  # noqa: E402
        response_preview = response[:200]  # noqa: E402
  # noqa: E402
        prompt = (  # noqa: E402
            "You are auditing conversation pair quality for a text message system.\n\n"  # noqa: E402
            f"TRIGGER (incoming message):\n{trigger_preview}\n\n"  # noqa: E402
            f"RESPONSE (user's actual reply):\n{response_preview}\n\n"  # noqa: E402
            "Score 0-10: Is this response a good, natural reply to the trigger?\n"  # noqa: E402
            "10 = Perfect match, natural exchange. "  # noqa: E402
            "0 = Wrong pairing, makes no sense as a reply.\n"  # noqa: E402
            "Consider: Does the response relate to the trigger? "  # noqa: E402
            "Is it a plausible human reply?\n"  # noqa: E402
            "Respond in JSON: "  # noqa: E402
            '{"score": <0-10>, "reasoning": "<1 sentence>"}'  # noqa: E402
        )  # noqa: E402
  # noqa: E402
        score, reasoning = judge_call(client, prompt)  # noqa: E402
        results.append(  # noqa: E402
            PairAuditResult(  # noqa: E402
                rowid=rowid,  # noqa: E402
                trigger_text=trigger_preview,  # noqa: E402
                response_text=response_preview,  # noqa: E402
                quality_score=max(0.0, score),  # noqa: E402
                reasoning=reasoning,  # noqa: E402
            )  # noqa: E402
        )  # noqa: E402
  # noqa: E402
    return results  # noqa: E402
  # noqa: E402
  # noqa: E402
# ---------------------------------------------------------------------------  # noqa: E402
# Main  # noqa: E402
# ---------------------------------------------------------------------------  # noqa: E402
  # noqa: E402
  # noqa: E402
def main() -> int:  # noqa: E402
    parser = argparse.ArgumentParser(description="JARVIS RAG Quality Evaluation")  # noqa: E402
    parser.add_argument(  # noqa: E402
        "--relevance-only", action="store_true", help="Only run retrieval relevance"  # noqa: E402
    )  # noqa: E402
    parser.add_argument("--ablation-only", action="store_true", help="Only run generation ablation")  # noqa: E402
    parser.add_argument("--audit-only", action="store_true", help="Only run pair quality audit")  # noqa: E402
    parser.add_argument(  # noqa: E402
        "--audit-sample", type=int, default=50, help="Number of pairs to audit (default: 50)"  # noqa: E402
    )  # noqa: E402
    args = parser.parse_args()  # noqa: E402
  # noqa: E402
    # Setup logging  # noqa: E402
    log_path = PROJECT_ROOT / "results" / "rag_eval.log"  # noqa: E402
    log_path.parent.mkdir(parents=True, exist_ok=True)  # noqa: E402
    logging.basicConfig(  # noqa: E402
        level=logging.INFO,  # noqa: E402
        format="%(asctime)s - %(levelname)s - %(message)s",  # noqa: E402
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],  # noqa: E402
    )  # noqa: E402
    logging.getLogger(__name__)  # noqa: E402
  # noqa: E402
    run_all = not (args.relevance_only or args.ablation_only or args.audit_only)  # noqa: E402
  # noqa: E402
    print("=" * 70, flush=True)  # noqa: E402
    print("JARVIS RAG Quality Evaluation", flush=True)  # noqa: E402
    print("=" * 70, flush=True)  # noqa: E402
  # noqa: E402
    client = get_judge_client()  # noqa: E402
    if client is None:  # noqa: E402
        print("ERROR: Judge API key not set in .env", flush=True)  # noqa: E402
        print("       Required for judge scoring.", flush=True)  # noqa: E402
        return 1  # noqa: E402
  # noqa: E402
    # Load test cases  # noqa: E402
    from evals.batch_eval import TEST_CASES  # noqa: E402
  # noqa: E402
    print(f"Test cases: {len(TEST_CASES)}", flush=True)  # noqa: E402
    print(f"Judge: {JUDGE_MODEL} via DeepInfra", flush=True)  # noqa: E402
    print(flush=True)  # noqa: E402
  # noqa: E402
    start = time.perf_counter()  # noqa: E402
    output: dict = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}  # noqa: E402
  # noqa: E402
    # --- 1. Retrieval Relevance ---  # noqa: E402
    if run_all or args.relevance_only:  # noqa: E402
        print("-" * 70, flush=True)  # noqa: E402
        print("1. RETRIEVAL RELEVANCE", flush=True)  # noqa: E402
        print("-" * 70, flush=True)  # noqa: E402
        relevance_results = eval_retrieval_relevance(client, TEST_CASES)  # noqa: E402
  # noqa: E402
        if relevance_results:  # noqa: E402
            scores = [r.avg_relevance for r in relevance_results if r.avg_relevance >= 0]  # noqa: E402
            avg = sum(scores) / len(scores) if scores else 0  # noqa: E402
            high = sum(1 for s in scores if s >= 7)  # noqa: E402
            print(f"\n  Avg relevance: {avg:.1f}/10", flush=True)  # noqa: E402
            print(  # noqa: E402
                f"  High relevance (>=7): {high}/{len(scores)} ({high / len(scores) * 100:.0f}%)",  # noqa: E402
                flush=True,  # noqa: E402
            )  # noqa: E402
            output["retrieval_relevance"] = {  # noqa: E402
                "avg_score": round(avg, 2),  # noqa: E402
                "high_relevance_rate": round(high / len(scores), 4) if scores else 0,  # noqa: E402
                "per_case": [  # noqa: E402
                    {  # noqa: E402
                        "name": r.test_name,  # noqa: E402
                        "avg_relevance": round(r.avg_relevance, 2),  # noqa: E402
                        "n_retrieved": len(r.retrieved_triggers),  # noqa: E402
                    }  # noqa: E402
                    for r in relevance_results  # noqa: E402
                ],  # noqa: E402
            }  # noqa: E402
        print(flush=True)  # noqa: E402
  # noqa: E402
    # --- 2. Generation Ablation ---  # noqa: E402
    if run_all or args.ablation_only:  # noqa: E402
        print("-" * 70, flush=True)  # noqa: E402
        print("2. GENERATION ABLATION (with vs without RAG)", flush=True)  # noqa: E402
        print("-" * 70, flush=True)  # noqa: E402
        ablation_results = eval_generation_ablation(client, TEST_CASES)  # noqa: E402
  # noqa: E402
        if ablation_results:  # noqa: E402
            valid = [  # noqa: E402
                r for r in ablation_results if r.score_without_rag >= 0 and r.score_with_rag >= 0  # noqa: E402
            ]  # noqa: E402
            if valid:  # noqa: E402
                avg_no_rag = sum(r.score_without_rag for r in valid) / len(valid)  # noqa: E402
                avg_with_rag = sum(r.score_with_rag for r in valid) / len(valid)  # noqa: E402
                avg_delta = sum(r.delta for r in valid) / len(valid)  # noqa: E402
                improved = sum(1 for r in valid if r.delta > 0)  # noqa: E402
                degraded = sum(1 for r in valid if r.delta < 0)  # noqa: E402
                neutral = sum(1 for r in valid if r.delta == 0)  # noqa: E402
  # noqa: E402
                print(f"\n  Avg score WITHOUT RAG: {avg_no_rag:.1f}/10", flush=True)  # noqa: E402
                print(f"  Avg score WITH RAG:    {avg_with_rag:.1f}/10", flush=True)  # noqa: E402
                print(f"  Avg delta:             {avg_delta:+.1f}", flush=True)  # noqa: E402
                print(  # noqa: E402
                    f"  Improved: {improved}  Degraded: {degraded}  Neutral: {neutral}", flush=True  # noqa: E402
                )  # noqa: E402
  # noqa: E402
                output["generation_ablation"] = {  # noqa: E402
                    "avg_without_rag": round(avg_no_rag, 2),  # noqa: E402
                    "avg_with_rag": round(avg_with_rag, 2),  # noqa: E402
                    "avg_delta": round(avg_delta, 2),  # noqa: E402
                    "improved": improved,  # noqa: E402
                    "degraded": degraded,  # noqa: E402
                    "neutral": neutral,  # noqa: E402
                }  # noqa: E402
        print(flush=True)  # noqa: E402
  # noqa: E402
    # --- 3. Pair Quality Audit ---  # noqa: E402
    if run_all or args.audit_only:  # noqa: E402
        print("-" * 70, flush=True)  # noqa: E402
        print(f"3. PAIR QUALITY AUDIT (sample={args.audit_sample})", flush=True)  # noqa: E402
        print("-" * 70, flush=True)  # noqa: E402
        audit_results = eval_pair_quality(client, args.audit_sample)  # noqa: E402
  # noqa: E402
        if audit_results:  # noqa: E402
            scores = [r.quality_score for r in audit_results if r.quality_score >= 0]  # noqa: E402
            avg = sum(scores) / len(scores) if scores else 0  # noqa: E402
            good = sum(1 for s in scores if s >= 7)  # noqa: E402
            bad = sum(1 for s in scores if s < 4)  # noqa: E402
  # noqa: E402
            print(f"\n  Pairs audited: {len(audit_results)}", flush=True)  # noqa: E402
            print(f"  Avg quality: {avg:.1f}/10", flush=True)  # noqa: E402
            print(  # noqa: E402
                f"  Good (>=7): {good}/{len(scores)} ({good / len(scores) * 100:.0f}%)", flush=True  # noqa: E402
            )  # noqa: E402
            print(f"  Bad (<4): {bad}/{len(scores)} ({bad / len(scores) * 100:.0f}%)", flush=True)  # noqa: E402
  # noqa: E402
            if bad > 0:  # noqa: E402
                print("\n  Worst pairs:", flush=True)  # noqa: E402
                worst = sorted(audit_results, key=lambda r: r.quality_score)[:5]  # noqa: E402
                for r in worst:  # noqa: E402
                    print(  # noqa: E402
                        f"    rowid={r.rowid} score={r.quality_score:.0f}: "  # noqa: E402
                        f"{r.trigger_text[:50]!r} -> {r.response_text[:50]!r}",  # noqa: E402
                        flush=True,  # noqa: E402
                    )  # noqa: E402
                    print(f"      {r.reasoning}", flush=True)  # noqa: E402
  # noqa: E402
            output["pair_quality_audit"] = {  # noqa: E402
                "sample_size": len(audit_results),  # noqa: E402
                "avg_quality": round(avg, 2),  # noqa: E402
                "good_rate": round(good / len(scores), 4) if scores else 0,  # noqa: E402
                "bad_rate": round(bad / len(scores), 4) if scores else 0,  # noqa: E402
            }  # noqa: E402
        print(flush=True)  # noqa: E402
  # noqa: E402
    # --- Summary ---  # noqa: E402
    elapsed = time.perf_counter() - start  # noqa: E402
    print("=" * 70, flush=True)  # noqa: E402
    print(f"RAG eval completed in {elapsed:.1f}s", flush=True)  # noqa: E402
    print("=" * 70, flush=True)  # noqa: E402
  # noqa: E402
    # Save results  # noqa: E402
    output_path = PROJECT_ROOT / "results" / "rag_eval_latest.json"  # noqa: E402
    output_path.parent.mkdir(parents=True, exist_ok=True)  # noqa: E402
    output_path.write_text(json.dumps(output, indent=2))  # noqa: E402
    print(f"Results saved to: {output_path}", flush=True)  # noqa: E402
  # noqa: E402
    return 0  # noqa: E402
  # noqa: E402
  # noqa: E402
if __name__ == "__main__":  # noqa: E402
    sys.exit(main())  # noqa: E402
