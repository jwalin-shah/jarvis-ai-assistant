#!/usr/bin/env python3  # noqa: E501
"""RAG retrieval quality evaluation.  # noqa: E501
  # noqa: E501
Measures whether retrieved conversation pairs actually help generation quality:  # noqa: E501
1. Retrieval relevance: Are retrieved trigger_texts semantically relevant?  # noqa: E501
2. Generation ablation: Does adding RAG examples improve judge scores?  # noqa: E501
3. Pair quality audit: Are stored pairs (trigger_text, response_text) good quality?  # noqa: E501
  # noqa: E501
Usage:  # noqa: E501
    uv run python evals/rag_eval.py                    # Full eval (needs Cerebras key)  # noqa: E501
    uv run python evals/rag_eval.py --relevance-only   # Just retrieval relevance  # noqa: E501
    uv run python evals/rag_eval.py --ablation-only    # Just generation ablation  # noqa: E501
    uv run python evals/rag_eval.py --audit-only       # Just pair quality audit  # noqa: E501
    uv run python evals/rag_eval.py --audit-sample 50  # Audit N random pairs  # noqa: E501
"""  # noqa: E501
  # noqa: E501
from __future__ import annotations  # noqa: E402  # noqa: E501

# noqa: E501
import argparse  # noqa: E501
import json  # noqa: E501
import logging  # noqa: E501
import os  # noqa: E501
import sys  # noqa: E501
import time  # noqa: E501
from dataclasses import dataclass  # noqa: E402  # noqa: E501
from pathlib import Path  # noqa: E402  # noqa: E501

# noqa: E501
from tqdm import tqdm  # noqa: E402  # noqa: E501

  # noqa: E501
PROJECT_ROOT = Path(__file__).parent.parent  # noqa: E501
sys.path.insert(0, str(PROJECT_ROOT))  # noqa: E501
  # noqa: E501
# Load .env  # noqa: E501
_env_path = PROJECT_ROOT / ".env"  # noqa: E501
if _env_path.exists():  # noqa: E501
    for line in _env_path.read_text().splitlines():  # noqa: E501
        line = line.strip()  # noqa: E501
        if line and not line.startswith("#") and "=" in line:  # noqa: E501
            key, _, val = line.partition("=")  # noqa: E501
            os.environ.setdefault(key.strip(), val.strip())  # noqa: E501
  # noqa: E501
from evals.judge_config import JUDGE_MODEL, get_judge_client  # noqa: E402  # noqa: E501


  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
# Data types  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
  # noqa: E501
  # noqa: E501
@dataclass  # noqa: E501
class RelevanceResult:  # noqa: E501
    """Result of retrieval relevance scoring for one test case."""  # noqa: E501
  # noqa: E501
    test_name: str  # noqa: E501
    query: str  # noqa: E501
    retrieved_triggers: list[str]  # noqa: E501
    retrieved_scores: list[float]  # noqa: E501
    relevance_scores: list[float]  # Judge scores (0-10) per retrieved item  # noqa: E501
    avg_relevance: float  # noqa: E501
  # noqa: E501
  # noqa: E501
@dataclass  # noqa: E501
class AblationResult:  # noqa: E501
    """Result of with/without RAG comparison for one test case."""  # noqa: E501
  # noqa: E501
    test_name: str  # noqa: E501
    score_without_rag: float  # noqa: E501
    score_with_rag: float  # noqa: E501
    delta: float  # noqa: E501
    reply_without_rag: str  # noqa: E501
    reply_with_rag: str  # noqa: E501
  # noqa: E501
  # noqa: E501
@dataclass  # noqa: E501
class PairAuditResult:  # noqa: E501
    """Quality audit of a stored trigger-response pair."""  # noqa: E501
  # noqa: E501
    rowid: int  # noqa: E501
    trigger_text: str  # noqa: E501
    response_text: str  # noqa: E501
    quality_score: float  # Judge score (0-10)  # noqa: E501
    reasoning: str  # noqa: E501
  # noqa: E501
  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
# Judge client  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
  # noqa: E501
  # noqa: E501
# get_judge_client imported from evals.judge_config  # noqa: E501
  # noqa: E501
  # noqa: E501
def judge_call(client, prompt: str) -> tuple[float, str]:  # noqa: E501
    """Make a judge call, return (score, reasoning)."""  # noqa: E501
    try:  # noqa: E501
        resp = client.chat.completions.create(  # noqa: E501
            model=JUDGE_MODEL,  # noqa: E501
            messages=[{"role": "user", "content": prompt}],  # noqa: E501
            temperature=0.0,  # noqa: E501
            max_tokens=200,  # noqa: E501
        )  # noqa: E501
        text = resp.choices[0].message.content.strip()  # noqa: E501
        if text.startswith("```"):  # noqa: E501
            text = text.split("```")[1]  # noqa: E501
            if text.startswith("json"):  # noqa: E501
                text = text[4:]  # noqa: E501
        data = json.loads(text)  # noqa: E501
        return float(data["score"]), data.get("reasoning", "")  # noqa: E501
    except Exception as e:  # noqa: E501
        return -1.0, f"judge error: {e}"  # noqa: E501
  # noqa: E501
  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
# 1. Retrieval Relevance  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
  # noqa: E501
  # noqa: E501
def eval_retrieval_relevance(client, test_cases: list[dict]) -> list[RelevanceResult]:  # noqa: E501
    """Score whether retrieved trigger_texts are relevant to test case queries."""  # noqa: E501
    from jarvis.search.vec_search import get_vec_searcher  # noqa: E501
  # noqa: E501
    searcher = get_vec_searcher()  # noqa: E501
    results = []  # noqa: E501
  # noqa: E501
    for tc in tqdm(test_cases, desc="Retrieval relevance"):  # noqa: E501
        query = tc["last_message"]  # noqa: E501
        name = tc["name"]  # noqa: E501
  # noqa: E501
        # Retrieve similar chunks  # noqa: E501
        search_results = searcher.search_with_chunks_global(query=query, limit=3)  # noqa: E501
  # noqa: E501
        triggers = []  # noqa: E501
        vec_scores = []  # noqa: E501
        relevance_scores = []  # noqa: E501
  # noqa: E501
        for sr in search_results:  # noqa: E501
            trigger = sr.context_text or ""  # noqa: E501
            triggers.append(trigger)  # noqa: E501
            vec_scores.append(sr.score)  # noqa: E501
  # noqa: E501
            if not trigger.strip():  # noqa: E501
                relevance_scores.append(0.0)  # noqa: E501
                continue  # noqa: E501
  # noqa: E501
            # Judge: is this retrieved trigger relevant to the query?  # noqa: E501
            prompt = (  # noqa: E501
                "You are evaluating retrieval quality for a text message reply system.\n\n"  # noqa: E501
                f"QUERY (message to reply to):\n{query}\n\n"  # noqa: E501
                f"RETRIEVED CONTEXT:\n{trigger}\n\n"  # noqa: E501
                "Score 0-10: How relevant is the retrieved context to the query?\n"  # noqa: E501
                "10 = Perfect match (same topic/intent), 0 = Completely unrelated.\n"  # noqa: E501
                "Respond in JSON: "  # noqa: E501
                '{"score": <0-10>, "reasoning": "<1 sentence>"}'  # noqa: E501
            )  # noqa: E501
            score, _ = judge_call(client, prompt)  # noqa: E501
            relevance_scores.append(max(0.0, score))  # noqa: E501
  # noqa: E501
        avg_rel = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0  # noqa: E501
        results.append(  # noqa: E501
            RelevanceResult(  # noqa: E501
                test_name=name,  # noqa: E501
                query=query,  # noqa: E501
                retrieved_triggers=triggers,  # noqa: E501
                retrieved_scores=vec_scores,  # noqa: E501
                relevance_scores=relevance_scores,  # noqa: E501
                avg_relevance=avg_rel,  # noqa: E501
            )  # noqa: E501
        )  # noqa: E501
  # noqa: E501
        print(f"  {name}: avg_relevance={avg_rel:.1f}/10 ({len(triggers)} retrieved)", flush=True)  # noqa: E501
  # noqa: E501
    return results  # noqa: E501
  # noqa: E501
  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
# 2. Generation Ablation (with vs without RAG)  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
  # noqa: E501
  # noqa: E501
def eval_generation_ablation(client, test_cases: list[dict]) -> list[AblationResult]:  # noqa: E501
    """Compare generation quality WITH vs WITHOUT RAG examples."""  # noqa: E501
    from evals.batch_eval import build_prompt, judge_response  # noqa: E501

    # noqa: E501
    from jarvis.search.vec_search import get_vec_searcher  # noqa: E501
    from models.loader import get_model  # noqa: E501
  # noqa: E501
    searcher = get_vec_searcher()  # noqa: E501
    results = []  # noqa: E501
  # noqa: E501
    # Load model once  # noqa: E501
  # noqa: E501
    loader = get_model()  # noqa: E501
    if not loader.is_loaded():  # noqa: E501
        loader.load()  # noqa: E501
  # noqa: E501
    for tc in tqdm(test_cases, desc="Generation ablation"):  # noqa: E501
        name = tc["name"]  # noqa: E501
        query = tc["last_message"]  # noqa: E501
  # noqa: E501
        # --- Without RAG: base prompt ---  # noqa: E501
        prompt_no_rag = build_prompt(tc)  # noqa: E501
        result_no_rag = loader.generate_sync(  # noqa: E501
            prompt=prompt_no_rag,  # noqa: E501
            temperature=0.1,  # noqa: E501
            max_tokens=50,  # noqa: E501
            top_p=0.1,  # noqa: E501
            top_k=50,  # noqa: E501
            repetition_penalty=1.05,  # noqa: E501
        )  # noqa: E501
        reply_no_rag = result_no_rag.text.strip()  # noqa: E501
  # noqa: E501
        # --- With RAG: add retrieved examples to prompt ---  # noqa: E501
        search_results = searcher.search_with_chunks_global(query=query, limit=3)  # noqa: E501
        rag_examples = ""  # noqa: E501
        for sr in search_results:  # noqa: E501
            if sr.context_text and sr.reply_text:  # noqa: E501
                rag_examples += (  # noqa: E501
                    f"\nExample:\n"  # noqa: E501
                    f"  Context: {sr.context_text[:200]}\n"  # noqa: E501
                    f"  Reply: {sr.reply_text[:100]}\n"  # noqa: E501
                )  # noqa: E501
  # noqa: E501
        # Build RAG-enhanced prompt by inserting examples before <reply>  # noqa: E501
        prompt_with_rag = prompt_no_rag  # noqa: E501
        if rag_examples:  # noqa: E501
            prompt_with_rag = prompt_no_rag.replace(  # noqa: E501
                "<reply>",  # noqa: E501
                f"<similar_exchanges>{rag_examples}\n</similar_exchanges>\n\n<reply>",  # noqa: E501
            )  # noqa: E501
  # noqa: E501
        result_with_rag = loader.generate_sync(  # noqa: E501
            prompt=prompt_with_rag,  # noqa: E501
            temperature=0.1,  # noqa: E501
            max_tokens=50,  # noqa: E501
            top_p=0.1,  # noqa: E501
            top_k=50,  # noqa: E501
            repetition_penalty=1.05,  # noqa: E501
        )  # noqa: E501
        reply_with_rag = result_with_rag.text.strip()  # noqa: E501
  # noqa: E501
        # Score both with judge  # noqa: E501
        score_no, _ = judge_response(client, tc, reply_no_rag)  # noqa: E501
        score_rag, _ = judge_response(client, tc, reply_with_rag)  # noqa: E501
  # noqa: E501
        delta = score_rag - score_no  # noqa: E501
        results.append(  # noqa: E501
            AblationResult(  # noqa: E501
                test_name=name,  # noqa: E501
                score_without_rag=score_no,  # noqa: E501
                score_with_rag=score_rag,  # noqa: E501
                delta=delta,  # noqa: E501
                reply_without_rag=reply_no_rag,  # noqa: E501
                reply_with_rag=reply_with_rag,  # noqa: E501
            )  # noqa: E501
        )  # noqa: E501
  # noqa: E501
        marker = "+" if delta > 0 else ("-" if delta < 0 else "=")  # noqa: E501
        print(  # noqa: E501
            f"  {name}: no_rag={score_no:.0f} rag={score_rag:.0f} delta={marker}{abs(delta):.0f}",  # noqa: E501
            flush=True,  # noqa: E501
        )  # noqa: E501
  # noqa: E501
    return results  # noqa: E501
  # noqa: E501
  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
# 3. Pair Quality Audit  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
  # noqa: E501
  # noqa: E501
def eval_pair_quality(client, sample_size: int = 50) -> list[PairAuditResult]:  # noqa: E501
    """Audit quality of stored trigger-response pairs in vec_chunks."""  # noqa: E501
    from jarvis.db.core import JarvisDB  # noqa: E501
  # noqa: E501
    db = JarvisDB()  # noqa: E501
    db.connect()  # noqa: E501
  # noqa: E501
    # Sample random pairs from vec_chunks  # noqa: E501
    rows = db.execute(  # noqa: E501
        "SELECT rowid, trigger_text, response_text FROM vec_chunks "  # noqa: E501
        "WHERE trigger_text IS NOT NULL AND response_text IS NOT NULL "  # noqa: E501
        "ORDER BY RANDOM() LIMIT ?",  # noqa: E501
        (sample_size,),  # noqa: E501
    ).fetchall()  # noqa: E501
  # noqa: E501
    if not rows:  # noqa: E501
        print("  No pairs found in vec_chunks. Is the database populated?", flush=True)  # noqa: E501
        return []  # noqa: E501
  # noqa: E501
    print(f"  Auditing {len(rows)} random pairs from vec_chunks...", flush=True)  # noqa: E501
    results = []  # noqa: E501
  # noqa: E501
    for rowid, trigger, response in tqdm(rows, desc="Auditing pairs"):  # noqa: E501
        if not trigger or not response:  # noqa: E501
            continue  # noqa: E501
  # noqa: E501
        # Truncate for judge prompt  # noqa: E501
        trigger_preview = trigger[:300]  # noqa: E501
        response_preview = response[:200]  # noqa: E501
  # noqa: E501
        prompt = (  # noqa: E501
            "You are auditing conversation pair quality for a text message system.\n\n"  # noqa: E501
            f"TRIGGER (incoming message):\n{trigger_preview}\n\n"  # noqa: E501
            f"RESPONSE (user's actual reply):\n{response_preview}\n\n"  # noqa: E501
            "Score 0-10: Is this response a good, natural reply to the trigger?\n"  # noqa: E501
            "10 = Perfect match, natural exchange. "  # noqa: E501
            "0 = Wrong pairing, makes no sense as a reply.\n"  # noqa: E501
            "Consider: Does the response relate to the trigger? "  # noqa: E501
            "Is it a plausible human reply?\n"  # noqa: E501
            "Respond in JSON: "  # noqa: E501
            '{"score": <0-10>, "reasoning": "<1 sentence>"}'  # noqa: E501
        )  # noqa: E501
  # noqa: E501
        score, reasoning = judge_call(client, prompt)  # noqa: E501
        results.append(  # noqa: E501
            PairAuditResult(  # noqa: E501
                rowid=rowid,  # noqa: E501
                trigger_text=trigger_preview,  # noqa: E501
                response_text=response_preview,  # noqa: E501
                quality_score=max(0.0, score),  # noqa: E501
                reasoning=reasoning,  # noqa: E501
            )  # noqa: E501
        )  # noqa: E501
  # noqa: E501
    return results  # noqa: E501
  # noqa: E501
  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
# Main  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
  # noqa: E501
  # noqa: E501
def main() -> int:  # noqa: E501
    parser = argparse.ArgumentParser(description="JARVIS RAG Quality Evaluation")  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--relevance-only", action="store_true", help="Only run retrieval relevance"  # noqa: E501
    )  # noqa: E501
    parser.add_argument("--ablation-only", action="store_true", help="Only run generation ablation")  # noqa: E501
    parser.add_argument("--audit-only", action="store_true", help="Only run pair quality audit")  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--audit-sample", type=int, default=50, help="Number of pairs to audit (default: 50)"  # noqa: E501
    )  # noqa: E501
    args = parser.parse_args()  # noqa: E501
  # noqa: E501
    # Setup logging  # noqa: E501
    log_path = PROJECT_ROOT / "results" / "rag_eval.log"  # noqa: E501
    log_path.parent.mkdir(parents=True, exist_ok=True)  # noqa: E501
    logging.basicConfig(  # noqa: E501
        level=logging.INFO,  # noqa: E501
        format="%(asctime)s - %(levelname)s - %(message)s",  # noqa: E501
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],  # noqa: E501
    )  # noqa: E501
    logging.getLogger(__name__)  # noqa: E501
  # noqa: E501
    run_all = not (args.relevance_only or args.ablation_only or args.audit_only)  # noqa: E501
  # noqa: E501
    print("=" * 70, flush=True)  # noqa: E501
    print("JARVIS RAG Quality Evaluation", flush=True)  # noqa: E501
    print("=" * 70, flush=True)  # noqa: E501
  # noqa: E501
    client = get_judge_client()  # noqa: E501
    if client is None:  # noqa: E501
        print("ERROR: Judge API key not set in .env", flush=True)  # noqa: E501
        print("       Required for judge scoring.", flush=True)  # noqa: E501
        return 1  # noqa: E501
  # noqa: E501
    # Load test cases  # noqa: E501
    from evals.batch_eval import TEST_CASES  # noqa: E501
  # noqa: E501
    print(f"Test cases: {len(TEST_CASES)}", flush=True)  # noqa: E501
    print(f"Judge: {JUDGE_MODEL} via DeepInfra", flush=True)  # noqa: E501
    print(flush=True)  # noqa: E501
  # noqa: E501
    start = time.perf_counter()  # noqa: E501
    output: dict = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}  # noqa: E501
  # noqa: E501
    # --- 1. Retrieval Relevance ---  # noqa: E501
    if run_all or args.relevance_only:  # noqa: E501
        print("-" * 70, flush=True)  # noqa: E501
        print("1. RETRIEVAL RELEVANCE", flush=True)  # noqa: E501
        print("-" * 70, flush=True)  # noqa: E501
        relevance_results = eval_retrieval_relevance(client, TEST_CASES)  # noqa: E501
  # noqa: E501
        if relevance_results:  # noqa: E501
            scores = [r.avg_relevance for r in relevance_results if r.avg_relevance >= 0]  # noqa: E501
            avg = sum(scores) / len(scores) if scores else 0  # noqa: E501
            high = sum(1 for s in scores if s >= 7)  # noqa: E501
            print(f"\n  Avg relevance: {avg:.1f}/10", flush=True)  # noqa: E501
            print(  # noqa: E501
                f"  High relevance (>=7): {high}/{len(scores)} ({high / len(scores) * 100:.0f}%)",  # noqa: E501
                flush=True,  # noqa: E501
            )  # noqa: E501
            output["retrieval_relevance"] = {  # noqa: E501
                "avg_score": round(avg, 2),  # noqa: E501
                "high_relevance_rate": round(high / len(scores), 4) if scores else 0,  # noqa: E501
                "per_case": [  # noqa: E501
                    {  # noqa: E501
                        "name": r.test_name,  # noqa: E501
                        "avg_relevance": round(r.avg_relevance, 2),  # noqa: E501
                        "n_retrieved": len(r.retrieved_triggers),  # noqa: E501
                    }  # noqa: E501
                    for r in relevance_results  # noqa: E501
                ],  # noqa: E501
            }  # noqa: E501
        print(flush=True)  # noqa: E501
  # noqa: E501
    # --- 2. Generation Ablation ---  # noqa: E501
    if run_all or args.ablation_only:  # noqa: E501
        print("-" * 70, flush=True)  # noqa: E501
        print("2. GENERATION ABLATION (with vs without RAG)", flush=True)  # noqa: E501
        print("-" * 70, flush=True)  # noqa: E501
        ablation_results = eval_generation_ablation(client, TEST_CASES)  # noqa: E501
  # noqa: E501
        if ablation_results:  # noqa: E501
            valid = [  # noqa: E501
                r for r in ablation_results if r.score_without_rag >= 0 and r.score_with_rag >= 0  # noqa: E501
            ]  # noqa: E501
            if valid:  # noqa: E501
                avg_no_rag = sum(r.score_without_rag for r in valid) / len(valid)  # noqa: E501
                avg_with_rag = sum(r.score_with_rag for r in valid) / len(valid)  # noqa: E501
                avg_delta = sum(r.delta for r in valid) / len(valid)  # noqa: E501
                improved = sum(1 for r in valid if r.delta > 0)  # noqa: E501
                degraded = sum(1 for r in valid if r.delta < 0)  # noqa: E501
                neutral = sum(1 for r in valid if r.delta == 0)  # noqa: E501
  # noqa: E501
                print(f"\n  Avg score WITHOUT RAG: {avg_no_rag:.1f}/10", flush=True)  # noqa: E501
                print(f"  Avg score WITH RAG:    {avg_with_rag:.1f}/10", flush=True)  # noqa: E501
                print(f"  Avg delta:             {avg_delta:+.1f}", flush=True)  # noqa: E501
                print(  # noqa: E501
                    f"  Improved: {improved}  Degraded: {degraded}  Neutral: {neutral}", flush=True  # noqa: E501
                )  # noqa: E501
  # noqa: E501
                output["generation_ablation"] = {  # noqa: E501
                    "avg_without_rag": round(avg_no_rag, 2),  # noqa: E501
                    "avg_with_rag": round(avg_with_rag, 2),  # noqa: E501
                    "avg_delta": round(avg_delta, 2),  # noqa: E501
                    "improved": improved,  # noqa: E501
                    "degraded": degraded,  # noqa: E501
                    "neutral": neutral,  # noqa: E501
                }  # noqa: E501
        print(flush=True)  # noqa: E501
  # noqa: E501
    # --- 3. Pair Quality Audit ---  # noqa: E501
    if run_all or args.audit_only:  # noqa: E501
        print("-" * 70, flush=True)  # noqa: E501
        print(f"3. PAIR QUALITY AUDIT (sample={args.audit_sample})", flush=True)  # noqa: E501
        print("-" * 70, flush=True)  # noqa: E501
        audit_results = eval_pair_quality(client, args.audit_sample)  # noqa: E501
  # noqa: E501
        if audit_results:  # noqa: E501
            scores = [r.quality_score for r in audit_results if r.quality_score >= 0]  # noqa: E501
            avg = sum(scores) / len(scores) if scores else 0  # noqa: E501
            good = sum(1 for s in scores if s >= 7)  # noqa: E501
            bad = sum(1 for s in scores if s < 4)  # noqa: E501
  # noqa: E501
            print(f"\n  Pairs audited: {len(audit_results)}", flush=True)  # noqa: E501
            print(f"  Avg quality: {avg:.1f}/10", flush=True)  # noqa: E501
            print(  # noqa: E501
                f"  Good (>=7): {good}/{len(scores)} ({good / len(scores) * 100:.0f}%)", flush=True  # noqa: E501
            )  # noqa: E501
            print(f"  Bad (<4): {bad}/{len(scores)} ({bad / len(scores) * 100:.0f}%)", flush=True)  # noqa: E501
  # noqa: E501
            if bad > 0:  # noqa: E501
                print("\n  Worst pairs:", flush=True)  # noqa: E501
                worst = sorted(audit_results, key=lambda r: r.quality_score)[:5]  # noqa: E501
                for r in worst:  # noqa: E501
                    print(  # noqa: E501
                        f"    rowid={r.rowid} score={r.quality_score:.0f}: "  # noqa: E501
                        f"{r.trigger_text[:50]!r} -> {r.response_text[:50]!r}",  # noqa: E501
                        flush=True,  # noqa: E501
                    )  # noqa: E501
                    print(f"      {r.reasoning}", flush=True)  # noqa: E501
  # noqa: E501
            output["pair_quality_audit"] = {  # noqa: E501
                "sample_size": len(audit_results),  # noqa: E501
                "avg_quality": round(avg, 2),  # noqa: E501
                "good_rate": round(good / len(scores), 4) if scores else 0,  # noqa: E501
                "bad_rate": round(bad / len(scores), 4) if scores else 0,  # noqa: E501
            }  # noqa: E501
        print(flush=True)  # noqa: E501
  # noqa: E501
    # --- Summary ---  # noqa: E501
    elapsed = time.perf_counter() - start  # noqa: E501
    print("=" * 70, flush=True)  # noqa: E501
    print(f"RAG eval completed in {elapsed:.1f}s", flush=True)  # noqa: E501
    print("=" * 70, flush=True)  # noqa: E501
  # noqa: E501
    # Save results  # noqa: E501
    output_path = PROJECT_ROOT / "results" / "rag_eval_latest.json"  # noqa: E501
    output_path.parent.mkdir(parents=True, exist_ok=True)  # noqa: E501
    output_path.write_text(json.dumps(output, indent=2))  # noqa: E501
    print(f"Results saved to: {output_path}", flush=True)  # noqa: E501
  # noqa: E501
    return 0  # noqa: E501
  # noqa: E501
  # noqa: E501
if __name__ == "__main__":  # noqa: E501
    sys.exit(main())  # noqa: E501
