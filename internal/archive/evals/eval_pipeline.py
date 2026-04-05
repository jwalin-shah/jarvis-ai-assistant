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
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
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
ANTI_AI_PHRASES = [  # noqa: E402
    "i'd be happy to",  # noqa: E402
    "i hope this helps",  # noqa: E402
    "let me know if",  # noqa: E402
    "i understand",  # noqa: E402
    "as an ai",  # noqa: E402
    "i'm an ai",  # noqa: E402
    "certainly!",  # noqa: E402
    "of course!",  # noqa: E402
    "great question",  # noqa: E402
]  # noqa: E402
  # noqa: E402
EVAL_DATASET_PATH = PROJECT_ROOT / "evals" / "eval_dataset.jsonl"  # noqa: E402
  # noqa: E402
CATEGORIES = ["acknowledge", "closing", "question", "request", "emotion", "statement"]  # noqa: E402
  # noqa: E402
  # noqa: E402
@dataclass  # noqa: E402
class EvalExample:  # noqa: E402
    category: str  # noqa: E402
    context: list[str]  # noqa: E402
    last_message: str  # noqa: E402
    ideal_response: str  # noqa: E402
    contact_style: str  # noqa: E402
    notes: str  # noqa: E402
  # noqa: E402
  # noqa: E402
@dataclass  # noqa: E402
class EvalResult:  # noqa: E402
    example: EvalExample  # noqa: E402
    predicted_category: str  # noqa: E402
    generated_response: str  # noqa: E402
    latency_ms: float  # noqa: E402
    category_match: bool  # noqa: E402
    anti_ai_violations: list[str] = field(default_factory=list)  # noqa: E402
    response_length: int = 0  # noqa: E402
    route_type: str = "unknown"  # noqa: E402
    route_reason: str = ""  # noqa: E402
    similarity_score: float | None = None  # noqa: E402
    judge_score: float | None = None  # noqa: E402
    judge_reasoning: str = ""  # noqa: E402
  # noqa: E402
  # noqa: E402
def load_eval_dataset(path: Path) -> list[EvalExample]:  # noqa: E402
    """Load eval dataset from JSONL."""  # noqa: E402
    examples = []  # noqa: E402
    for line in path.read_text().splitlines():  # noqa: E402
        line = line.strip()  # noqa: E402
        if not line:  # noqa: E402
            continue  # noqa: E402
        data = json.loads(line)  # noqa: E402
        examples.append(  # noqa: E402
            EvalExample(  # noqa: E402
                category=data["category"],  # noqa: E402
                context=data["context"],  # noqa: E402
                last_message=data["last_message"],  # noqa: E402
                ideal_response=data["ideal_response"],  # noqa: E402
                contact_style=data.get("contact_style", "casual"),  # noqa: E402
                notes=data.get("notes", ""),  # noqa: E402
            )  # noqa: E402
        )  # noqa: E402
    return examples  # noqa: E402
  # noqa: E402
  # noqa: E402
def check_anti_ai(text: str) -> list[str]:  # noqa: E402
    """Check for AI-sounding phrases."""  # noqa: E402
    lower = text.lower()  # noqa: E402
    return [phrase for phrase in ANTI_AI_PHRASES if phrase in lower]  # noqa: E402
  # noqa: E402
  # noqa: E402
def ensure_realistic_thread(context: list[str], last_message: str) -> list[str]:  # noqa: E402
    """Ensure eval thread has at least two turns to avoid thin-context artifacts."""  # noqa: E402
    clean = [c for c in context if isinstance(c, str) and c.strip()]  # noqa: E402
    if len(clean) >= 2:  # noqa: E402
        return clean  # noqa: E402
    if clean:  # noqa: E402
        return ["Me: quick follow-up", clean[0]]  # noqa: E402
    return ["Me: quick follow-up", f"Them: {last_message}"]  # noqa: E402
  # noqa: E402
  # noqa: E402
def _strip_fenced_json(text: str) -> str:  # noqa: E402
    text = (text or "").strip()  # noqa: E402
    if text.startswith("```"):  # noqa: E402
        parts = text.split("```")  # noqa: E402
        if len(parts) >= 2:  # noqa: E402
            text = parts[1]  # noqa: E402
        if text.startswith("json"):  # noqa: E402
            text = text[4:]  # noqa: E402
    return text.strip()  # noqa: E402
  # noqa: E402
  # noqa: E402
def _extract_json_blob(text: str) -> str:  # noqa: E402
    """Extract the most likely JSON object/array from model output."""  # noqa: E402
    s = _strip_fenced_json(text)  # noqa: E402
    if not s:  # noqa: E402
        return s  # noqa: E402
    # Prefer array payload for batched judge; fallback to object payload.  # noqa: E402
    arr_start = s.find("[")  # noqa: E402
    arr_end = s.rfind("]")  # noqa: E402
    if arr_start != -1 and arr_end != -1 and arr_end > arr_start:  # noqa: E402
        return s[arr_start : arr_end + 1]  # noqa: E402
    obj_start = s.find("{")  # noqa: E402
    obj_end = s.rfind("}")  # noqa: E402
    if obj_start != -1 and obj_end != -1 and obj_end > obj_start:  # noqa: E402
        return s[obj_start : obj_end + 1]  # noqa: E402
    return s  # noqa: E402
  # noqa: E402
  # noqa: E402
def _judge_single_item(judge_client: object, judge_model: str, ex: EvalExample, generated: str) -> tuple[float | None, str]:  # noqa: E402
    """Judge one item and return (score, reasoning)."""  # noqa: E402
    try:  # noqa: E402
        prompt = (  # noqa: E402
            "You are an expert evaluator for a text message reply generator.\n\n"  # noqa: E402
            f"CONVERSATION CONTEXT:\n{chr(10).join(ex.context)}\n\n"  # noqa: E402
            f"LAST MESSAGE (to reply to):\n{ex.last_message}\n\n"  # noqa: E402
            f"IDEAL RESPONSE:\n{ex.ideal_response}\n\n"  # noqa: E402
            f"GENERATED REPLY:\n{generated}\n\n"  # noqa: E402
            f"CATEGORY: {ex.category}\n"  # noqa: E402
            f"NOTES: {ex.notes}\n\n"  # noqa: E402
            "Score the generated reply from 0-10. Consider:\n"  # noqa: E402
            "- Does it match the tone and intent of the ideal response?\n"  # noqa: E402
            "- Does it sound like a real person texting (not an AI)?\n"  # noqa: E402
            "- Is it appropriate for the category?\n"  # noqa: E402
            "- Is the length appropriate?\n\n"  # noqa: E402
            'Respond in JSON: {"score": <0-10>, "reasoning": "<1-2 sentences>"}'  # noqa: E402
        )  # noqa: E402
        resp = judge_client.chat.completions.create(  # noqa: E402
            model=judge_model,  # noqa: E402
            messages=[{"role": "user", "content": prompt}],  # noqa: E402
            temperature=0.0,  # noqa: E402
            max_tokens=150,  # noqa: E402
        )  # noqa: E402
        payload = json.loads(_extract_json_blob(resp.choices[0].message.content or ""))  # noqa: E402
        score = float(payload["score"])  # noqa: E402
        if score < 0:  # noqa: E402
            score = 0.0  # noqa: E402
        if score > 10:  # noqa: E402
            score = 10.0  # noqa: E402
        return score, str(payload.get("reasoning", ""))  # noqa: E402
    except Exception as e:  # noqa: E402
        return None, f"judge error: {e}"  # noqa: E402
  # noqa: E402
  # noqa: E402
def main() -> int:  # noqa: E402
    import argparse  # noqa: E402
  # noqa: E402
    parser = argparse.ArgumentParser(description="JARVIS Eval Pipeline")  # noqa: E402
    parser.add_argument("--judge", action="store_true", help="Enable LLM judge scoring")  # noqa: E402
    parser.add_argument("--similarity", action="store_true", help="Enable BERT cosine similarity")  # noqa: E402
    parser.add_argument(  # noqa: E402
        "--judge-batch-size",  # noqa: E402
        type=int,  # noqa: E402
        default=1,  # noqa: E402
        help="Judge batch size (1 = per-example judging, >1 = batched judging)",  # noqa: E402
    )  # noqa: E402
    parser.add_argument(  # noqa: E402
        "--judge-delay-seconds",  # noqa: E402
        type=float,  # noqa: E402
        default=2.2,  # noqa: E402
        help="Delay between judge API calls/batches to reduce rate-limit errors",  # noqa: E402
    )  # noqa: E402
    parser.add_argument(  # noqa: E402
        "--force-model-load",  # noqa: E402
        action="store_true",  # noqa: E402
        help="Set JARVIS_FORCE_MODEL_LOAD=1 to bypass memory-pressure load guard (risky)",  # noqa: E402
    )  # noqa: E402
    args = parser.parse_args()  # noqa: E402
    if args.force_model_load:  # noqa: E402
        os.environ["JARVIS_FORCE_MODEL_LOAD"] = "1"  # noqa: E402
  # noqa: E402
    # Setup logging  # noqa: E402
    log_path = PROJECT_ROOT / "results" / "eval_pipeline.log"  # noqa: E402
    log_path.parent.mkdir(parents=True, exist_ok=True)  # noqa: E402
    logging.basicConfig(  # noqa: E402
        level=logging.INFO,  # noqa: E402
        format="%(asctime)s - %(levelname)s - %(message)s",  # noqa: E402
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],  # noqa: E402
    )  # noqa: E402
    logging.getLogger(__name__)  # noqa: E402
  # noqa: E402
    # Load dataset  # noqa: E402
    if not EVAL_DATASET_PATH.exists():  # noqa: E402
        print(f"ERROR: Eval dataset not found at {EVAL_DATASET_PATH}", flush=True)  # noqa: E402
        return 1  # noqa: E402
  # noqa: E402
    examples = load_eval_dataset(EVAL_DATASET_PATH)  # noqa: E402
    print("=" * 70, flush=True)  # noqa: E402
    print("JARVIS EVAL PIPELINE - Baseline Measurement", flush=True)  # noqa: E402
    print("=" * 70, flush=True)  # noqa: E402
    print(f"Examples:    {len(examples)}", flush=True)  # noqa: E402
    print(f"Categories:  {', '.join(CATEGORIES)}", flush=True)  # noqa: E402
    print(f"Judge:       {'enabled' if args.judge else 'disabled (use --judge)'}", flush=True)  # noqa: E402
    print(  # noqa: E402
        f"Similarity:  {'enabled' if args.similarity else 'disabled (use --similarity)'}",  # noqa: E402
        flush=True,  # noqa: E402
    )  # noqa: E402
    print(flush=True)  # noqa: E402
  # noqa: E402
    # Initialize components  # noqa: E402
    print("Loading classifier...", flush=True)  # noqa: E402
    from jarvis.classifiers.category_classifier import classify_category  # noqa: E402
    from jarvis.classifiers.response_mobilization import classify_response_pressure  # noqa: E402
  # noqa: E402
    # Initialize reply service for generation  # noqa: E402
    print("Loading reply service...", flush=True)  # noqa: E402
    from jarvis.contracts.pipeline import MessageContext  # noqa: E402
    from jarvis.reply_service import ReplyService  # noqa: E402
  # noqa: E402
    reply_service = ReplyService()  # noqa: E402
  # noqa: E402
    # Optional: BERT embedder for similarity  # noqa: E402
    embedder = None  # noqa: E402
    if args.similarity:  # noqa: E402
        print("Loading embedder for similarity...", flush=True)  # noqa: E402
        from jarvis.embedding_adapter import get_embedder  # noqa: E402
  # noqa: E402
        embedder = get_embedder()  # noqa: E402
  # noqa: E402
    # Optional: judge  # noqa: E402
    judge_client = None  # noqa: E402
    if args.judge:  # noqa: E402
        from evals.judge_config import JUDGE_MODEL, get_judge_client  # noqa: E402
  # noqa: E402
        judge_client = get_judge_client()  # noqa: E402
        if judge_client is None:  # noqa: E402
            print("WARNING: Judge API key not set, skipping judge", flush=True)  # noqa: E402
        else:  # noqa: E402
            print(f"Judge ready: {JUDGE_MODEL}", flush=True)  # noqa: E402
  # noqa: E402
    print(flush=True)  # noqa: E402
    print("-" * 70, flush=True)  # noqa: E402
  # noqa: E402
    results: list[EvalResult] = []  # noqa: E402
    total_start = time.perf_counter()  # noqa: E402
  # noqa: E402
    for i, ex in enumerate(tqdm(examples, desc="Evaluating"), 1):  # noqa: E402
        print(f"\n[{i:2d}/{len(examples)}] [{ex.category}] {ex.last_message[:50]}...", flush=True)  # noqa: E402
  # noqa: E402
        gen_start = time.perf_counter()  # noqa: E402
        eval_thread = ensure_realistic_thread(ex.context, ex.last_message)  # noqa: E402
  # noqa: E402
        # 1. Classify category  # noqa: E402
        mobilization = classify_response_pressure(ex.last_message)  # noqa: E402
        category_result = classify_category(  # noqa: E402
            ex.last_message,  # noqa: E402
            context=eval_thread,  # noqa: E402
            mobilization=mobilization,  # noqa: E402
        )  # noqa: E402
        predicted_category = category_result.category  # noqa: E402
  # noqa: E402
        # 2. Generate reply via full pipeline (empty search_results to skip RAG)  # noqa: E402
        route_type = "unknown"  # noqa: E402
        route_reason = ""  # noqa: E402
        try:  # noqa: E402
            context = MessageContext(  # noqa: E402
                chat_id="iMessage;-;+15555550123",  # noqa: E402
                message_text=ex.last_message,  # noqa: E402
                is_from_me=False,  # noqa: E402
                timestamp=datetime.now(UTC),  # noqa: E402
                metadata={"thread": eval_thread, "contact_name": "John"},  # noqa: E402
            )  # noqa: E402
            reply_result = reply_service.generate_reply(  # noqa: E402
                context=context,  # noqa: E402
                thread=eval_thread,  # noqa: E402
                search_results=[],  # noqa: E402
            )  # noqa: E402
            generated = reply_result.response  # noqa: E402
            route_type = str(reply_result.metadata.get("type", "unknown"))  # noqa: E402
            route_reason = str(reply_result.metadata.get("reason", ""))  # noqa: E402
        except Exception as e:  # noqa: E402
            generated = f"[ERROR: {e}]"  # noqa: E402
            route_type = "error"  # noqa: E402
            route_reason = str(e)  # noqa: E402
  # noqa: E402
        latency_ms = (time.perf_counter() - gen_start) * 1000  # noqa: E402
  # noqa: E402
        # 3. Score  # noqa: E402
        category_match = predicted_category == ex.category  # noqa: E402
        anti_ai = check_anti_ai(generated)  # noqa: E402
  # noqa: E402
        # Similarity scoring  # noqa: E402
        sim_score = None  # noqa: E402
        if embedder and generated and not generated.startswith("[ERROR"):  # noqa: E402
            try:  # noqa: E402
                import numpy as np  # noqa: E402
  # noqa: E402
                emb_gen = embedder.encode([generated])[0]  # noqa: E402
                emb_ideal = embedder.encode([ex.ideal_response])[0]  # noqa: E402
                sim_score = float(  # noqa: E402
                    np.dot(emb_gen, emb_ideal)  # noqa: E402
                    / (np.linalg.norm(emb_gen) * np.linalg.norm(emb_ideal))  # noqa: E402
                )  # noqa: E402
            except Exception:  # noqa: E402
                pass  # noqa: E402
  # noqa: E402
        # Judge scoring  # noqa: E402
        j_score = None  # noqa: E402
        j_reasoning = ""  # noqa: E402
        if (  # noqa: E402
            judge_client  # noqa: E402
            and args.judge_batch_size <= 1  # noqa: E402
            and generated  # noqa: E402
            and not generated.startswith("[ERROR")  # noqa: E402
        ):  # noqa: E402
            j_score, j_reasoning = _judge_single_item(judge_client, JUDGE_MODEL, ex, generated)  # noqa: E402
            if args.judge_delay_seconds > 0:  # noqa: E402
                time.sleep(args.judge_delay_seconds)  # noqa: E402
  # noqa: E402
        result = EvalResult(  # noqa: E402
            example=ex,  # noqa: E402
            predicted_category=predicted_category,  # noqa: E402
            generated_response=generated,  # noqa: E402
            latency_ms=latency_ms,  # noqa: E402
            category_match=category_match,  # noqa: E402
            anti_ai_violations=anti_ai,  # noqa: E402
            response_length=len(generated),  # noqa: E402
            route_type=route_type,  # noqa: E402
            route_reason=route_reason,  # noqa: E402
            similarity_score=sim_score,  # noqa: E402
            judge_score=j_score,  # noqa: E402
            judge_reasoning=j_reasoning,  # noqa: E402
        )  # noqa: E402
        results.append(result)  # noqa: E402
  # noqa: E402
        # Print per-example  # noqa: E402
        cat_status = "OK" if category_match else f"MISS (got {predicted_category})"  # noqa: E402
        ai_status = f"AI:{len(anti_ai)}" if anti_ai else "clean"  # noqa: E402
        sim_str = f"sim={sim_score:.2f}" if sim_score is not None else ""  # noqa: E402
        judge_str = f"judge={j_score:.0f}/10" if j_score is not None else ""  # noqa: E402
        print(f'  Response: "{generated[:60]}"', flush=True)  # noqa: E402
        print(  # noqa: E402
            f"  Cat: {cat_status} | {ai_status} | {latency_ms:.0f}ms {sim_str} {judge_str}",  # noqa: E402
            flush=True,  # noqa: E402
        )  # noqa: E402
  # noqa: E402
    # Optional batched judge pass (reduces API requests significantly)  # noqa: E402
    if judge_client and args.judge_batch_size > 1:  # noqa: E402
        judgeable_idx = [  # noqa: E402
            i  # noqa: E402
            for i, r in enumerate(results)  # noqa: E402
            if r.generated_response and not r.generated_response.startswith("[ERROR")  # noqa: E402
        ]  # noqa: E402
        if judgeable_idx:  # noqa: E402
            print(  # noqa: E402
                f"\nRunning batched judge: {len(judgeable_idx)} items, "  # noqa: E402
                f"batch_size={args.judge_batch_size}",  # noqa: E402
                flush=True,  # noqa: E402
            )  # noqa: E402
            for start in range(0, len(judgeable_idx), args.judge_batch_size):  # noqa: E402
                chunk = judgeable_idx[start : start + args.judge_batch_size]  # noqa: E402
                batch_prompt = (  # noqa: E402
                    "You are an expert evaluator for text message replies.\n"  # noqa: E402
                    "For each item, score generated reply 0-10 and provide brief reasoning.\n"  # noqa: E402
                    "Return ONLY JSON array with objects: "  # noqa: E402
                    '{"index": <1-based item index in this batch>, "score": <0-10>, '  # noqa: E402
                    '"reasoning": "<1-2 sentences>"}.\n\n'  # noqa: E402
                )  # noqa: E402
                for pos, idx in enumerate(chunk, 1):  # noqa: E402
                    r = results[idx]  # noqa: E402
                    ex = r.example  # noqa: E402
                    batch_prompt += (  # noqa: E402
                        f"ITEM {pos}\n"  # noqa: E402
                        f"CATEGORY: {ex.category}\n"  # noqa: E402
                        f"CONTEXT:\n{chr(10).join(ex.context)}\n"  # noqa: E402
                        f"LAST MESSAGE:\n{ex.last_message}\n"  # noqa: E402
                        f"IDEAL RESPONSE:\n{ex.ideal_response}\n"  # noqa: E402
                        f"GENERATED REPLY:\n{r.generated_response}\n\n"  # noqa: E402
                    )  # noqa: E402
                try:  # noqa: E402
                    resp = judge_client.chat.completions.create(  # noqa: E402
                        model=JUDGE_MODEL,  # noqa: E402
                        messages=[{"role": "user", "content": batch_prompt}],  # noqa: E402
                        temperature=0.0,  # noqa: E402
                        max_tokens=600,  # noqa: E402
                    )  # noqa: E402
                    text = _extract_json_blob(resp.choices[0].message.content or "")  # noqa: E402
                    payload = json.loads(text)  # noqa: E402
                    assigned: set[int] = set()  # noqa: E402
                    payload_items = payload  # noqa: E402
                    if isinstance(payload, dict):  # noqa: E402
                        payload_items = payload.get("items", [])  # noqa: E402
                    if isinstance(payload_items, list):  # noqa: E402
                        for i, item in enumerate(payload_items, 1):  # noqa: E402
                            try:  # noqa: E402
                                if isinstance(item, dict):  # noqa: E402
                                    local_idx = int(item.get("index", i))  # noqa: E402
                                    score = float(item.get("score"))  # noqa: E402
                                    reasoning = str(item.get("reasoning", ""))  # noqa: E402
                                elif isinstance(item, list | tuple) and len(item) >= 2:  # noqa: E402
                                    local_idx = i  # noqa: E402
                                    score = float(item[0])  # noqa: E402
                                    reasoning = str(item[1])  # noqa: E402
                                else:  # noqa: E402
                                    continue  # noqa: E402
                                if 1 <= local_idx <= len(chunk):  # noqa: E402
                                    if score < 0:  # noqa: E402
                                        score = 0.0  # noqa: E402
                                    if score > 10:  # noqa: E402
                                        score = 10.0  # noqa: E402
                                    target = results[chunk[local_idx - 1]]  # noqa: E402
                                    target.judge_score = score  # noqa: E402
                                    target.judge_reasoning = reasoning  # noqa: E402
                                    assigned.add(local_idx)  # noqa: E402
                            except Exception:  # noqa: E402
                                continue  # noqa: E402
                    for local_idx, idx in enumerate(chunk, 1):  # noqa: E402
                        if local_idx in assigned:  # noqa: E402
                            continue  # noqa: E402
                        score, reasoning = _judge_single_item(  # noqa: E402
                            judge_client,  # noqa: E402
                            JUDGE_MODEL,  # noqa: E402
                            results[idx].example,  # noqa: E402
                            results[idx].generated_response,  # noqa: E402
                        )  # noqa: E402
                        results[idx].judge_score = score  # noqa: E402
                        results[idx].judge_reasoning = reasoning  # noqa: E402
                        if args.judge_delay_seconds > 0:  # noqa: E402
                            time.sleep(min(args.judge_delay_seconds, 0.6))  # noqa: E402
                except Exception as e:  # noqa: E402
                    for idx in chunk:  # noqa: E402
                        score, reasoning = _judge_single_item(  # noqa: E402
                            judge_client,  # noqa: E402
                            JUDGE_MODEL,  # noqa: E402
                            results[idx].example,  # noqa: E402
                            results[idx].generated_response,  # noqa: E402
                        )  # noqa: E402
                        results[idx].judge_score = score  # noqa: E402
                        results[idx].judge_reasoning = (  # noqa: E402
                            f"batch_fail_then_single: {reasoning}; batch_error={e}"  # noqa: E402
                        )  # noqa: E402
                        if args.judge_delay_seconds > 0:  # noqa: E402
                            time.sleep(min(args.judge_delay_seconds, 0.6))  # noqa: E402
                if args.judge_delay_seconds > 0 and (start + args.judge_batch_size) < len(  # noqa: E402
                    judgeable_idx  # noqa: E402
                ):  # noqa: E402
                    time.sleep(args.judge_delay_seconds)  # noqa: E402
  # noqa: E402
    total_ms = (time.perf_counter() - total_start) * 1000  # noqa: E402
  # noqa: E402
    # =========================================================================  # noqa: E402
    # Summary Report  # noqa: E402
    # =========================================================================  # noqa: E402
    print(flush=True)  # noqa: E402
    print("=" * 70, flush=True)  # noqa: E402
    print("SUMMARY", flush=True)  # noqa: E402
    print("=" * 70, flush=True)  # noqa: E402
  # noqa: E402
    n = len(results)  # noqa: E402
    if n == 0:  # noqa: E402
        print("No results to summarize.", flush=True)  # noqa: E402
        return 0  # noqa: E402
  # noqa: E402
    cat_matches = sum(1 for r in results if r.category_match)  # noqa: E402
    ai_clean = sum(1 for r in results if not r.anti_ai_violations)  # noqa: E402
    latencies = [r.latency_ms for r in results]  # noqa: E402
    avg_lat = sum(latencies) / n  # noqa: E402
    sorted_lat = sorted(latencies)  # noqa: E402
    p50 = sorted_lat[n // 2]  # noqa: E402
    p95 = sorted_lat[min(int(n * 0.95), n - 1)]  # noqa: E402
  # noqa: E402
    print(f"Category accuracy:  {cat_matches}/{n} ({cat_matches / n * 100:.0f}%)", flush=True)  # noqa: E402
    print(f"Anti-AI clean:      {ai_clean}/{n} ({ai_clean / n * 100:.0f}%)", flush=True)  # noqa: E402
    print(f"Total time:         {total_ms:.0f}ms", flush=True)  # noqa: E402
    print(f"Avg latency:        {avg_lat:.0f}ms", flush=True)  # noqa: E402
    print(f"P50/P95 latency:    {p50:.0f}ms / {p95:.0f}ms", flush=True)  # noqa: E402
  # noqa: E402
    # Route-path summary  # noqa: E402
    print(flush=True)  # noqa: E402
    print("ROUTE PATHS", flush=True)  # noqa: E402
    print("-" * 70, flush=True)  # noqa: E402
    route_counts: dict[str, int] = {}  # noqa: E402
    route_empty_counts: dict[str, int] = {}  # noqa: E402
    for r in results:  # noqa: E402
        key = f"{r.route_type}:{r.route_reason}" if r.route_reason else r.route_type  # noqa: E402
        route_counts[key] = route_counts.get(key, 0) + 1  # noqa: E402
        if not (r.generated_response or "").strip():  # noqa: E402
            route_empty_counts[key] = route_empty_counts.get(key, 0) + 1  # noqa: E402
    for key, count in sorted(route_counts.items(), key=lambda kv: kv[1], reverse=True):  # noqa: E402
        empty = route_empty_counts.get(key, 0)  # noqa: E402
        print(f"  {key:<35} {count:>2d} (empty={empty})", flush=True)  # noqa: E402
  # noqa: E402
    # Similarity summary  # noqa: E402
    scored_sim = [r for r in results if r.similarity_score is not None]  # noqa: E402
    if scored_sim:  # noqa: E402
        sims = [r.similarity_score for r in scored_sim]  # noqa: E402
        print(f"Avg similarity:     {sum(sims) / len(sims):.3f}", flush=True)  # noqa: E402
  # noqa: E402
    # Judge summary  # noqa: E402
    scored_judge = [r for r in results if r.judge_score is not None and r.judge_score >= 0]  # noqa: E402
    if scored_judge:  # noqa: E402
        scores = [r.judge_score for r in scored_judge]  # noqa: E402
        avg_j = sum(scores) / len(scores)  # noqa: E402
        pass_7 = sum(1 for s in scores if s >= 7)  # noqa: E402
        print(f"Judge avg:          {avg_j:.1f}/10", flush=True)  # noqa: E402
        print(  # noqa: E402
            f"Judge pass (>=7):   {pass_7}/{len(scores)} ({pass_7 / len(scores) * 100:.0f}%)",  # noqa: E402
            flush=True,  # noqa: E402
        )  # noqa: E402
  # noqa: E402
    # Per-category breakdown  # noqa: E402
    print(flush=True)  # noqa: E402
    print("PER-CATEGORY BREAKDOWN", flush=True)  # noqa: E402
    print("-" * 70, flush=True)  # noqa: E402
    for cat in CATEGORIES:  # noqa: E402
        cat_results = [r for r in results if r.example.category == cat]  # noqa: E402
        if not cat_results:  # noqa: E402
            continue  # noqa: E402
        cat_correct = sum(1 for r in cat_results if r.category_match)  # noqa: E402
        cat_clean = sum(1 for r in cat_results if not r.anti_ai_violations)  # noqa: E402
        parts = [  # noqa: E402
            f"classify={cat_correct}/{len(cat_results)}",  # noqa: E402
            f"clean={cat_clean}/{len(cat_results)}",  # noqa: E402
        ]  # noqa: E402
        cat_sim = [r for r in cat_results if r.similarity_score is not None]  # noqa: E402
        if cat_sim:  # noqa: E402
            avg_s = sum(r.similarity_score for r in cat_sim) / len(cat_sim)  # noqa: E402
            parts.append(f"sim={avg_s:.2f}")  # noqa: E402
        cat_j = [r for r in cat_results if r.judge_score is not None and r.judge_score >= 0]  # noqa: E402
        if cat_j:  # noqa: E402
            avg_jj = sum(r.judge_score for r in cat_j) / len(cat_j)  # noqa: E402
            parts.append(f"judge={avg_jj:.1f}")  # noqa: E402
        print(f"  {cat:15s}  {' | '.join(parts)}", flush=True)  # noqa: E402
  # noqa: E402
    # Category misclassifications  # noqa: E402
    misses = [r for r in results if not r.category_match]  # noqa: E402
    if misses:  # noqa: E402
        print(flush=True)  # noqa: E402
        print("CATEGORY MISCLASSIFICATIONS", flush=True)  # noqa: E402
        print("-" * 70, flush=True)  # noqa: E402
        for r in misses:  # noqa: E402
            print(  # noqa: E402
                f"  [{r.example.category} -> {r.predicted_category}] {r.example.last_message[:50]}",  # noqa: E402
                flush=True,  # noqa: E402
            )  # noqa: E402
  # noqa: E402
    # Anti-AI violations  # noqa: E402
    violations = [r for r in results if r.anti_ai_violations]  # noqa: E402
    if violations:  # noqa: E402
        print(flush=True)  # noqa: E402
        print("ANTI-AI VIOLATIONS", flush=True)  # noqa: E402
        print("-" * 70, flush=True)  # noqa: E402
        for r in violations:  # noqa: E402
            print(f'  "{r.generated_response[:60]}" -> {r.anti_ai_violations}', flush=True)  # noqa: E402
  # noqa: E402
    # Worst judge scores  # noqa: E402
    if scored_judge:  # noqa: E402
        low = sorted(scored_judge, key=lambda r: r.judge_score)[:5]  # noqa: E402
        if low and low[0].judge_score < 7:  # noqa: E402
            print(flush=True)  # noqa: E402
            print("LOWEST JUDGE SCORES", flush=True)  # noqa: E402
            print("-" * 70, flush=True)  # noqa: E402
            for r in low:  # noqa: E402
                if r.judge_score < 7:  # noqa: E402
                    print(  # noqa: E402
                        f"  [{r.example.category}] {r.judge_score:.0f}/10 - "  # noqa: E402
                        f'"{r.generated_response[:50]}" - {r.judge_reasoning}',  # noqa: E402
                        flush=True,  # noqa: E402
                    )  # noqa: E402
  # noqa: E402
    # Save results  # noqa: E402
    output_path = PROJECT_ROOT / "results" / "eval_pipeline_baseline.json"  # noqa: E402
    output_path.parent.mkdir(parents=True, exist_ok=True)  # noqa: E402
    output_data = {  # noqa: E402
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),  # noqa: E402
        "total_examples": n,  # noqa: E402
        "category_accuracy": round(cat_matches / n, 4) if n else 0,  # noqa: E402
        "anti_ai_clean_rate": round(ai_clean / n, 4) if n else 0,  # noqa: E402
        "avg_similarity": round(sum(sims) / len(sims), 4) if scored_sim else None,  # noqa: E402
        "judge_avg": round(avg_j, 2) if scored_judge else None,  # noqa: E402
        "latency": {  # noqa: E402
            "avg_ms": round(avg_lat, 1),  # noqa: E402
            "p50_ms": round(p50, 1),  # noqa: E402
            "p95_ms": round(p95, 1),  # noqa: E402
            "total_ms": round(total_ms, 1),  # noqa: E402
        },  # noqa: E402
        "per_category": {},  # noqa: E402
        "results": [  # noqa: E402
            {  # noqa: E402
                "category_expected": r.example.category,  # noqa: E402
                "category_predicted": r.predicted_category,  # noqa: E402
                "last_message": r.example.last_message,  # noqa: E402
                "ideal_response": r.example.ideal_response,  # noqa: E402
                "generated_response": r.generated_response,  # noqa: E402
                "category_match": r.category_match,  # noqa: E402
                "anti_ai_violations": r.anti_ai_violations,  # noqa: E402
                "similarity_score": r.similarity_score,  # noqa: E402
                "judge_score": r.judge_score,  # noqa: E402
                "judge_reasoning": r.judge_reasoning,  # noqa: E402
                "latency_ms": round(r.latency_ms, 1),  # noqa: E402
                "route_type": r.route_type,  # noqa: E402
                "route_reason": r.route_reason,  # noqa: E402
            }  # noqa: E402
            for r in results  # noqa: E402
        ],  # noqa: E402
    }  # noqa: E402
  # noqa: E402
    # Per-category stats  # noqa: E402
    for cat in CATEGORIES:  # noqa: E402
        cat_results = [r for r in results if r.example.category == cat]  # noqa: E402
        if not cat_results:  # noqa: E402
            continue  # noqa: E402
        output_data["per_category"][cat] = {  # noqa: E402
            "count": len(cat_results),  # noqa: E402
            "classify_accuracy": round(  # noqa: E402
                sum(1 for r in cat_results if r.category_match) / len(cat_results), 4  # noqa: E402
            ),  # noqa: E402
            "anti_ai_clean_rate": round(  # noqa: E402
                sum(1 for r in cat_results if not r.anti_ai_violations) / len(cat_results), 4  # noqa: E402
            ),  # noqa: E402
        }  # noqa: E402
  # noqa: E402
    output_path.write_text(json.dumps(output_data, indent=2))  # noqa: E402
    print(f"\nResults saved to: {output_path}", flush=True)  # noqa: E402
    print("=" * 70, flush=True)  # noqa: E402
  # noqa: E402
    return 0  # noqa: E402
  # noqa: E402
  # noqa: E402
if __name__ == "__main__":  # noqa: E402
    sys.exit(main())  # noqa: E402
